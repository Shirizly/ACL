"""Compare original / preprocessed / commanded / measured motion for a telemetry session (see telemetry.py).

  python -m src.xarm_pipeline.analysis.compare_run outputs/telemetry/<timestamp>

Per run k (one commands_<k>.csv each) it writes into <session>/compare/:
  run<k>_tcp.png      TCP x,y,z,roll,pitch,yaw vs time + position error |measured - commanded|
  run<k>_joints.png   q1..q7 commanded vs encoders, one subplot per joint
  run<k>_aligned.csv  everything resampled on the command timeline (for further analysis)
  summary.json        per-run tracking error stats
Time axis: seconds since the run's first command. "original" is the recorded source placed in the robot frame by the
fit (same rigid motion as the preprocessed copy), at its real (unscaled) speed; available for recorded/ clean copies.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..safety.config import load_safety_config
from ..safety.kinematics import fk_pose, rpy_to_matrix
from ..telemetry import POSE_COLS, Q_COLS
from ..trajectory_io import read_trajectory_frame, split_trajectory_ref

ROOT = Path(__file__).resolve().parents[3]
STYLE = {"original": dict(color="0.6", lw=1.2, ls="--"), "preprocessed": dict(color="tab:blue", lw=1.2),
         "commanded": dict(color="tab:orange", lw=1.0, ls=":"), "measured": dict(color="tab:green", lw=1.2)}


def original_placed(spec: str, cfg) -> pd.DataFrame | None:
    """Recorded source rows of this part, rigidly placed like the fitted copy; t from 0 at the part start (s)."""
    path, part = split_trajectory_ref(spec)
    prep = path.with_suffix(".prep.json")
    if not prep.exists():
        return None
    rep = json.loads(prep.read_text())
    parts = rep.get("parts", [])
    entry = next((p for p in parts if p["name"] == part), parts[0] if len(parts) == 1 else None)
    src = ROOT / rep["source"]
    if entry is None or not src.exists():
        return None
    df = pd.read_csv(src)
    t = df["t"].to_numpy(float) - float(df["t"].iloc[0])
    a, b = entry["source_t_original"]
    sel = (t >= a - 1e-6) & (t <= b + 1e-6)
    p = df.loc[sel, ["x", "y", "z"]].to_numpy(float)
    th = entry["params"]
    R = rpy_to_matrix(math.radians(th.get("tilt_x", 0.0)), math.radians(th.get("tilt_y", 0.0)), math.radians(th["yaw"]))
    R_b = rpy_to_matrix(0.0, 0.0, math.radians(cfg.base_yaw_deg))
    c_base = R_b.T @ (np.array([th["cx"], th["cy"], th["cz"]]) - np.array([cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm]))
    center = 0.5 * (p.min(axis=0) + p.max(axis=0))
    placed = (p - center) @ R.T + c_base
    out = pd.DataFrame(placed, columns=["x", "y", "z"])
    out.insert(0, "t", t[sel] - a)
    return out


def tcp_from_q(Q: np.ndarray, cfg) -> np.ndarray:
    tcp = cfg.firmware_tcp_offset_rad
    return np.array([fk_pose(q, tcp_offset=tcp) for q in Q])


def angle_deg(a: np.ndarray, ref: float) -> np.ndarray:
    """Unwrapped angle [deg], shifted by whole turns so it starts near ``ref`` [rad] (common branch for all curves)."""
    a = np.unwrap(np.asarray(a, float))
    if len(a):
        a = a - 2 * np.pi * np.round((a[0] - ref) / (2 * np.pi))
    return np.degrees(a)


def min_span(ax, span: float) -> None:
    lo, hi = ax.get_ylim()
    if hi - lo < span:
        mid = 0.5 * (lo + hi)
        ax.set_ylim(mid - span / 2, mid + span / 2)


def compare_run(k: int, cmd_path: Path, measured: pd.DataFrame, out: Path, cfg_default: Path) -> dict:
    meta = json.loads(cmd_path.with_suffix(".json").read_text())
    spec = meta["trajectory"]
    title = str(Path(spec).resolve().relative_to(ROOT)) if Path(spec.split("#")[0]).resolve().is_relative_to(ROOT) else spec
    session = json.loads((cmd_path.parent / "session.json").read_text()) if (cmd_path.parent / "session.json").exists() else {}
    run_cfg = next((r.get("safety_config") for r in session.get("runs", []) if r["trajectory"] == spec), None)
    cfg = load_safety_config(run_cfg or cfg_default)
    cmd = pd.read_csv(cmd_path)
    if cmd.empty:
        return {"trajectory": spec, "error": "no commands logged"}
    t0 = float(cmd["t_wall"].iloc[0])
    tc = cmd["t_wall"].to_numpy() - t0
    m = measured[(measured["t_wall"] >= t0 - 0.5) & (measured["t_wall"] <= cmd["t_wall"].iloc[-1] + 1.0)]
    tm = m["t_wall"].to_numpy() - t0
    fw = None
    if (cmd_path.parent / "firmware_report.csv").exists():
        fw_all = pd.read_csv(cmd_path.parent / "firmware_report.csv")
        fw = fw_all[(fw_all["t_wall"] >= t0 - 0.5) & (fw_all["t_wall"] <= cmd["t_wall"].iloc[-1] + 1.0)]
        fw = fw if len(fw) else None
    tf = fw["t_wall"].to_numpy() - t0 if fw is not None else None

    pre = read_trajectory_frame(spec)
    joint_run = meta["kind"] == "joint"
    if joint_run:
        pre_pose = tcp_from_q(pre[list(Q_COLS)].to_numpy(float), cfg)
        cmd_pose = tcp_from_q(cmd[list(Q_COLS)].to_numpy(float), cfg)
    else:
        pre_pose = pre[list(POSE_COLS)].to_numpy(float)
        cmd_pose = cmd[list(POSE_COLS)].to_numpy(float)
    # Preprocessed sample idx is what command idx sent: place it on the scheduled (t_target) timeline.
    t_pre_raw = pre["t"].to_numpy(float)
    if len(pre) == len(cmd):
        sched = cmd["t_target"].to_numpy() + (tc - cmd["t_target"].to_numpy()).min()
        offset = float(np.median(sched[1:] - t_pre_raw[1:])) if len(pre) > 1 else 0.0
    else:
        offset = 0.0
    t_pre = t_pre_raw + offset
    orig = None if joint_run else original_placed(spec, cfg)

    # ---- TCP figure
    fig, axs = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    for i, col in enumerate(POSE_COLS):
        ax = axs[i]
        ang = i >= 3
        f = (lambda v, r=float(pre_pose[0, i]): angle_deg(v, r)) if ang else (lambda v: v)
        if orig is not None and not ang:
            ax.plot(orig["t"] + offset, orig[col], label="original (real speed, placed)", **STYLE["original"])
        ax.plot(t_pre, f(pre_pose[:, i]), label="preprocessed (scaled/cut/fitted)", **STYLE["preprocessed"])
        ax.plot(tc, f(cmd_pose[:, i]), label="commanded" + (" (FK)" if joint_run else ""), **STYLE["commanded"])
        ax.plot(tm, f(m[col].to_numpy()), label="measured", **STYLE["measured"])
        ax.set_ylabel(f"{col} [{'deg' if ang else 'mm'}]")
        ax.grid(alpha=0.3)
        min_span(ax, 2.0)
    meas_at_cmd = np.column_stack([np.interp(tc, tm, m[c]) for c in ("x", "y", "z")]) if len(m) else np.full((len(tc), 3), np.nan)
    err = np.linalg.norm(meas_at_cmd - cmd_pose[:, :3], axis=1)
    axs[6].plot(tc, err, color="tab:red", lw=1)
    axs[6].set_ylabel("|meas - cmd| [mm]")
    axs[6].set_xlabel("time since first command [s]")
    axs[6].grid(alpha=0.3)
    axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"TCP: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"run{k}_tcp.png", dpi=110)
    plt.close(fig)

    # ---- joints figure
    q_cmd = cmd[list(Q_COLS)].to_numpy(float)
    fig, axs = plt.subplots(7, 1, figsize=(12, 15), sharex=True)
    for j, col in enumerate(Q_COLS):
        ax = axs[j]
        if joint_run:
            ax.plot(t_pre, np.degrees(pre[col]), label="preprocessed", **STYLE["preprocessed"])
        if fw is not None:
            ax.plot(tf, np.degrees(fw[f"t{col}"].to_numpy()), label="firmware joint target (port 30000)",
                    color="tab:orange", lw=1.2)
        lbl = "commanded" if joint_run else "expected IK joints (safety model)"
        if not np.all(np.isnan(q_cmd[:, j])):
            ax.plot(tc, np.degrees(q_cmd[:, j]), label=lbl, **(STYLE["commanded"] | {"color": "tab:purple"}))
        ax.plot(tm, np.degrees(m[col].to_numpy()), label="measured (encoders)", **STYLE["measured"])
        ax.set_ylabel(f"{col} [deg]")
        ax.grid(alpha=0.3)
        min_span(ax, 2.0)
    axs[0].legend(loc="upper right", fontsize=8)
    axs[-1].set_xlabel("time since first command [s]")
    fig.suptitle(f"Joints: {title}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"run{k}_joints.png", dpi=110)
    plt.close(fig)

    # ---- aligned CSV on the command timeline
    al = pd.DataFrame({"t": tc})
    for i, c in enumerate(POSE_COLS):
        al[f"cmd_{c}"] = cmd_pose[:, i]
        if len(m):
            al[f"meas_{c}"] = np.interp(tc, tm, np.unwrap(m[c].to_numpy()) if i >= 3 else m[c].to_numpy())
    for j, c in enumerate(Q_COLS):
        al[f"cmd_{c}"] = q_cmd[:, j]
        if len(m):
            al[f"meas_{c}"] = np.interp(tc, tm, m[c].to_numpy())
    if fw is not None:
        for c in Q_COLS:
            al[f"fwtarget_{c}"] = np.interp(tc, tf, fw[f"t{c}"].to_numpy())
    al.to_csv(out / f"run{k}_aligned.csv", index=False, float_format="%.6f")

    qerr = np.degrees(np.abs(al[[f"meas_{c}" for c in Q_COLS]].to_numpy() - q_cmd)) if len(m) else np.full_like(q_cmd, np.nan)
    return {"trajectory": spec, "kind": meta["kind"], "commands": int(len(cmd)), "measured_samples": int(len(m)),
            "tcp_err_mm": {"rms": float(np.sqrt(np.nanmean(err ** 2))), "max": float(np.nanmax(err))},
            "joint_err_deg_max": [round(float(v), 3) for v in np.nanmax(qerr, axis=0)],
            "fw_target_vs_model_deg_max": ([round(float(v), 3) for v in np.degrees(np.nanmax(np.abs(
                al[[f"fwtarget_{c}" for c in Q_COLS]].to_numpy() - q_cmd), axis=0))] if fw is not None and not joint_run else None),
            "has_original": orig is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--safety-config", default=str(ROOT / "config" / "safety.yaml"),
                    help="fallback when session.json does not name one")
    a = ap.parse_args()
    d = Path(a.session_dir)
    measured = pd.read_csv(d / "measured.csv")
    out = d / "compare"
    out.mkdir(exist_ok=True)
    cmds = sorted(d.glob("commands_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    summary = {}
    for p in cmds:
        k = int(p.stem.split("_")[1])
        summary[f"run{k}"] = compare_run(k, p, measured, out, Path(a.safety_config))
        s = summary[f"run{k}"]
        print(f"run{k}: {s['trajectory']}  TCP err rms {s.get('tcp_err_mm', {}).get('rms', float('nan')):.2f} mm")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"compare: {len(cmds)} run(s) -> {out}")


if __name__ == "__main__":
    main()
