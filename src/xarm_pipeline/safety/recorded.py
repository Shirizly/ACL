"""Prepare a recorded (motion-capture) trajectory for the robot: slow it down, cut what is too fast, fit the rest.

    python -m src.xarm_pipeline.safety.recorded data/recorded_trajectories/5ec359d2 18 --time-scale 2.5
    scripts/control/prepare_recorded.sh data/recorded_trajectories/5ec359d2 2.5 18

Steps (positions only: the recorded orientation is ignored; the gripper points down, fit.default_orientation_rpy_deg):
  1. time scale: t' = time_scale * t  (speeds / time_scale, accelerations / time_scale^2). Positions are not
     filtered. For streaming, the output is densified to --command-rate (100 Hz) with a cubic curve that passes
     exactly through every recorded sample at its (scaled) time: in servo mode, sparse setpoints make the arm
     dash and wait.
  2. tool limits (placement-independent): spans where the tool speed or acceleration exceed the limits are
     CUT OUT (with a small guard band). The remaining pieces continue separately.
  3. workspace: each piece is re-placed (translation + yaw) to fit the safety cell; pieces that cannot fit
     are split near the middle, recursively (see safety.fit); what still does not fit is cut.
  4. joint limits (placement-dependent): joint speeds/accelerations along each fitted part (through IK)
     must stay within the limits; offending spans are cut and the pieces re-fitted once.
Everything that was cut is reported. Output goes into the preprocessed set of the current safety config:
  data/preprocessed/NNN_*/recorded/<slate>/<index>_t<scale>.csv  (+ .fit.json, + .prep.json report)
and is played with, e.g.
  scripts/control/play_cartesian_ik_record.sh "data/preprocessed/NNN_*/recorded/5ec359d2/0018_t2.50.csv#0018__p1" servo 60

Accelerations of 60 Hz motion capture are dominated by measurement noise after double differentiation, so
they are evaluated on a lightly smoothed copy (centered moving average, --accel-smoothing samples); the raw
values are reported too. The positions sent to the robot are not smoothed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_safety_config
from .fit import FIT_VERSION, Fitter, Segment, _frame_from_part, load_fit_config, read_log, upsample_frame
from .kinematics import IKFailure, rpy_to_matrix
from .preprocessed import find_or_create_set, setup_fingerprint
from .test_trajectories import JOINT_ACC_MAX, JOINT_SPEED_MAX, TCP_ACC_MAX, TCP_SPEED_MAX


@dataclass
class Limits:
    tcp_speed: float = TCP_SPEED_MAX       # mm/s
    tcp_acc: float = TCP_ACC_MAX           # mm/s^2
    joint_speed: float = JOINT_SPEED_MAX   # rad/s
    joint_acc: float = JOINT_ACC_MAX       # rad/s^2

    def scaled(self, frac: float) -> "Limits":
        return Limits(self.tcp_speed * frac, self.tcp_acc * frac, self.joint_speed * frac, self.joint_acc * frac)


def smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centered moving average along axis 0 (edges use the available samples)."""
    if n <= 1:
        return x
    k = np.ones(n) / n
    pad = n // 2
    xp = np.pad(x, [(pad, n - 1 - pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, xp)


def tcp_kinematics(t, p, n_smooth: int):
    v = np.linalg.norm(np.gradient(p, t, axis=0), axis=1)
    acc_raw = np.linalg.norm(np.gradient(np.gradient(p, t, axis=0), t, axis=0), axis=1)
    ps = smooth(p, n_smooth)
    acc = np.linalg.norm(np.gradient(np.gradient(ps, t, axis=0), t, axis=0), axis=1)
    return v, acc, acc_raw


def good_runs(bad: np.ndarray, t: np.ndarray, guard_s: float, min_s: float) -> list[tuple[int, int]]:
    """Index ranges [a, b] of samples kept after removing `bad` samples (+- guard_s), each >= min_s long."""
    bad = bad.copy()
    for i in np.flatnonzero(bad):
        bad[(t >= t[i] - guard_s) & (t <= t[i] + guard_s)] = True
    runs, a = [], None
    for i, b in enumerate(bad):
        if not b and a is None:
            a = i
        if (b or i == len(bad) - 1) and a is not None:
            e = i if not b else i - 1
            if t[e] - t[a] >= min_s:
                runs.append((a, e))
            a = None
    return runs


def spans(bad: np.ndarray, t: np.ndarray) -> list[list[float]]:
    out, a = [], None
    for i, b in enumerate(bad):
        if b and a is None:
            a = i
        if (not b or i == len(bad) - 1) and a is not None:
            e = i if b else i - 1
            out.append([round(float(t[a]), 3), round(float(t[e]), 3)])
            a = None
    return out


def joint_check(fitter: Fitter, part: dict, lim: Limits, n_smooth: int):
    """Joint speed/acc along a fitted part: (bad mask, peaks dict)."""
    seg, th = part["segment"], part["params"]
    poses = fitter.placed_poses(seg, th)
    q0 = fitter.start_joints(seg, th)
    _, _, _, Q = fitter.chk.solve_cartesian_path(poses, q0, fitter.tcp, poses[0])
    Q = np.asarray(Q)
    t = seg.t
    qd = np.gradient(Q, t, axis=0)
    qdd_raw = np.gradient(qd, t, axis=0)
    qdd = np.gradient(np.gradient(smooth(Q, n_smooth), t, axis=0), t, axis=0)
    bad = (np.abs(qd).max(axis=1) > lim.joint_speed) | (np.abs(qdd).max(axis=1) > lim.joint_acc)
    peaks = {"joint_speed_deg_s": float(np.degrees(np.abs(qd).max())),
             "joint_acc_deg_s2": float(np.degrees(np.abs(qdd).max())),
             "joint_acc_raw_deg_s2": float(np.degrees(np.abs(qdd_raw).max()))}
    return bad, peaks


def find_source(directory: Path, number) -> Path:
    for f in sorted(directory.glob("*.csv")):
        if f.stem.isdigit() and int(f.stem) == int(number):
            return f
    raise FileNotFoundError(f"no trajectory number {number} in {directory}")


def prepare(directory: str | Path, number, time_scale: float, cfg_path=None, limit_fraction: float = 0.9,
            n_smooth: int = 5, guard_s: float = 0.1, min_part_s: float | None = None,
            command_rate: float = 100.0) -> dict:
    t_start = time.time()
    cfg_path = Path(cfg_path or DEFAULT_CONFIG_PATH)
    cfg = load_safety_config(cfg_path)
    fcfg = load_fit_config(cfg_path)
    if min_part_s is not None:
        fcfg.min_part_s = min_part_s
    lim = Limits().scaled(limit_fraction)
    src = find_source(Path(directory), number)
    df = pd.read_csv(src)
    t = (df["t"].to_numpy(float) - float(df["t"].iloc[0])) * time_scale
    p = df[["x", "y", "z"]].to_numpy(float)
    name = src.stem
    rep = {"source": str(src), "time_scale": time_scale, "limit_fraction": limit_fraction,
           "limits": {"tcp_speed_mm_s": lim.tcp_speed, "tcp_acc_mm_s2": lim.tcp_acc,
                      "joint_speed_deg_s": math.degrees(lim.joint_speed), "joint_acc_deg_s2": math.degrees(lim.joint_acc)},
           "orientation": "ignored (recorded roll/pitch/yaw dropped); fixed tool orientation "
                          f"{list(fcfg.default_orientation_rpy_deg)} deg",
           "duration_s": float(t[-1]), "cuts": [], "parts": [], "dropped": []}

    # 1-2. tool speed / acceleration (placement-independent)
    v, acc, acc_raw = tcp_kinematics(t, p, n_smooth)
    rep["tcp_peaks"] = {"speed_mm_s": float(v.max()), "acc_mm_s2": float(acc.max()), "acc_raw_mm_s2": float(acc_raw.max())}
    bad = (v > lim.tcp_speed) | (acc > lim.tcp_acc)
    if bad.any():
        rep["cuts"].append({"stage": "tool speed/acceleration", "spans_s": spans(bad, t),
                            "removed_s": round(float(np.sum(np.diff(t)[bad[:-1]])), 2)})
    R = np.repeat(rpy_to_matrix(*np.radians(fcfg.default_orientation_rpy_deg))[None], len(t), axis=0)
    full = Segment(t, p, R, None, name, (0, len(t) - 1))
    runs = good_runs(bad, t, guard_s, fcfg.min_part_s) if bad.any() else [(0, len(t) - 1)]

    # 3. fit / split into the workspace
    set_dir = find_or_create_set(cfg_path)
    log_path = set_dir / "fit_log.jsonl"
    fitter = Fitter(cfg, fcfg, read_log(log_path, setup_fingerprint(cfg_path)))
    pieces = [full.part(a, b, f"{name}_{k}") for k, (a, b) in enumerate(runs)]
    fitted, dropped = [], []
    for piece in pieces:
        f, d = fitter.fit_or_split(piece)
        fitted += f
        dropped += [dict(x, reason="does not fit the workspace") for x in d]

    # 4. joint limits per fitted part; cut + refit once
    final = []
    joint_peaks = []
    for rnd in range(2):
        nxt = []
        for part in fitted:
            try:
                jb, peaks = joint_check(fitter, part, lim, n_smooth)
            except IKFailure as e:
                dropped.append({"segment": part["segment"], "reason": f"IK failure: {e}"})
                continue
            joint_peaks.append(peaks)
            if not jb.any():
                final.append(part)
                continue
            seg = part["segment"]
            rep["cuts"].append({"stage": f"joint speed/acceleration (round {rnd + 1})", "part_source_rows": list(seg.src_rows),
                                "spans_s": [[round(a + float(t[seg.src_rows[0]]), 3), round(b + float(t[seg.src_rows[0]]), 3)]
                                            for a, b in spans(jb, seg.t)], **{k: round(v_, 1) for k, v_ in peaks.items()}})
            if rnd == 1:
                dropped.append({"segment": seg, "reason": "joint limits still exceeded after one re-fit"})
                continue
            for a, b in good_runs(jb, seg.t, guard_s, fcfg.min_part_s):
                f, d = fitter.fit_or_split(seg.part(a, b, seg.name + f"j{a}"))
                nxt += f
                dropped += [dict(x, reason="does not fit the workspace") for x in d]
        fitted = nxt
        if not fitted:
            break

    # suggestion: the time scale that would have needed no speed cuts at all
    sugg = [time_scale * v.max() / lim.tcp_speed, time_scale * math.sqrt(acc.max() / lim.tcp_acc)]
    if joint_peaks:
        sugg += [time_scale * max(pk["joint_speed_deg_s"] for pk in joint_peaks) / math.degrees(lim.joint_speed),
                 time_scale * math.sqrt(max(pk["joint_acc_deg_s2"] for pk in joint_peaks) / math.degrees(lim.joint_acc))]
    rep["suggested_min_time_scale"] = round(max(sugg + [1.0]), 2)

    # output
    final.sort(key=lambda x: x["segment"].src_rows[0])
    # Parts cut from the middle of a movement start/end at speed: ease them in and out of rest along the same
    # path (positions unchanged, only the first/last fraction of a second re-timed).
    eased_info = []
    for part in final:
        seg0 = part["segment"]
        part["segment"] = seg0.eased()
        eased_info.append(round(float(part["segment"].t[-1] - seg0.t[-1]), 2))
    out = set_dir / "recorded" / Path(directory).name / f"{name}_t{time_scale:.2f}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames, parts_meta = {}, []
    for i, part in enumerate(final, 1):
        pname = f"{name}__p{i}"
        frames[pname] = upsample_frame(_frame_from_part(fitter, part), command_rate)  # dense setpoints for servo
        s = part["segment"]
        th = {k: round(float(x), 3) for k, x in part["params"].items()}
        meta = {"name": pname, "source_rows": list(s.src_rows),
                "source_t_scaled": [round(float(t[s.src_rows[0]]), 3), round(float(t[s.src_rows[1]]), 3)],
                "source_t_original": [round(float(t[s.src_rows[0]]) / time_scale, 3), round(float(t[s.src_rows[1]]) / time_scale, 3)],
                "duration_s": round(float(s.t[-1]), 2), "eased_extra_s": eased_info[i - 1],
                "params": th, "slack_mm": round(part["slack"], 1),
                "start_joints_deg": np.degrees(fitter.start_joints(s, part["params"])).round(4).tolist(),
                "approach_via_joints_deg": [round(float(x), 4) for x in fcfg.seed_joints_deg]}
        parts_meta.append(meta)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"fingerprint": setup_fingerprint(cfg_path), "source": f"{src}", "part": pname,
                                 "params": th, "shift_world_mm": [0, 0, 0], "slack_mm": meta["slack_mm"],
                                 "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    for d in dropped:
        s = d["segment"]
        rep["dropped"].append({"source_rows": list(s.src_rows), "reason": d.get("reason", ""),
                               "source_t_scaled": [round(float(t[s.src_rows[0]]), 3), round(float(t[s.src_rows[1]]), 3)]})
    kept_s = sum(m["source_t_scaled"][1] - m["source_t_scaled"][0] for m in parts_meta)  # source time, not eased
    rep["parts"] = parts_meta
    rep["kept_fraction"] = round(kept_s / max(float(t[-1]), 1e-9), 3)
    rep["seconds"] = round(time.time() - t_start, 1)
    if frames:
        df_out = pd.concat([f.assign(traj=n) for n, f in frames.items()], ignore_index=True)
        df_out = df_out[["traj"] + [c for c in df_out.columns if c != "traj"]]
        df_out.to_csv(out, index=False, float_format="%.4f")
    out.with_suffix(".fit.json").write_text(json.dumps(
        {"generated_by": "safety.recorded", "fit_version": FIT_VERSION, "grouped": True, "source": str(src),
         "trajectories": [{"name": m["name"], "result": "fitted", "parts": [m]} for m in parts_meta]}, indent=2))
    out.with_suffix(".prep.json").write_text(json.dumps(rep, indent=2))
    rep["output"] = str(out)
    return rep


def print_report(rep: dict) -> None:
    print(f"\n{Path(rep['source']).name}: time scale x{rep['time_scale']}, duration {rep['duration_s']:.1f} s "
          f"(limits at {rep['limit_fraction'] * 100:.0f}% of the robot's)")
    tp = rep["tcp_peaks"]
    print(f"  tool peaks after scaling: {tp['speed_mm_s']:.0f} mm/s, {tp['acc_mm_s2'] / 1000:.1f} m/s^2 "
          f"(raw, noise-dominated: {tp['acc_raw_mm_s2'] / 1000:.1f} m/s^2)")
    if not rep["cuts"] and not rep["dropped"]:
        print("  nothing cut")
    for c in rep["cuts"]:
        extra = (f"; peaks {c.get('joint_speed_deg_s', 0):.0f} deg/s, {c.get('joint_acc_deg_s2', 0):.0f} deg/s^2"
                 if "joint_speed_deg_s" in c else "")
        print(f"  NOTICE - cut for {c['stage']}: spans (s, scaled time) {c['spans_s']}{extra}")
    for d in rep["dropped"]:
        print(f"  NOTICE - dropped rows {d['source_rows']} (t {d['source_t_scaled']} s): {d['reason']}")
    for m in rep["parts"]:
        print(f"  part {m['name']}: {m['duration_s']:.1f} s (original t {m['source_t_original'][0]:.2f}-"
              f"{m['source_t_original'][1]:.2f} s), yaw {m['params']['yaw']:.0f} deg, slack {m['slack_mm']:.0f} mm"
              + (f", +{m['eased_extra_s']:.2f} s to start/stop at rest" if m.get("eased_extra_s") else ""))
    print(f"  kept {rep['kept_fraction'] * 100:.0f}% of the time; time scale needed for no speed cuts: "
          f">= {rep['suggested_min_time_scale']}")
    print(f"  wrote {rep.get('output')} (+ .fit.json, .prep.json)  [{rep['seconds']} s]")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", help="folder with the recorded CSVs (e.g. data/recorded_trajectories/5ec359d2)")
    p.add_argument("numbers", nargs="+", help="trajectory numbers (file index, e.g. 18 for 0018.csv)")
    p.add_argument("--time-scale", type=float, required=True, help="t' = scale * t; 2 = half speed")
    p.add_argument("--config", default=None)
    p.add_argument("--limit-fraction", type=float, default=0.9, help="use this fraction of the robot limits")
    p.add_argument("--accel-smoothing", type=int, default=5, help="samples in the moving average used to measure acceleration")
    p.add_argument("--guard", type=float, default=0.1, help="seconds removed around each too-fast sample")
    p.add_argument("--min-part", type=float, default=None, help="shortest part kept, s (default fit.min_part_s)")
    p.add_argument("--command-rate", type=float, default=100.0,
                   help="output setpoint rate, Hz (cubic interpolation through the recorded samples)")
    args = p.parse_args(argv)
    rc = 2
    for n in args.numbers:
        rep = prepare(args.directory, n, args.time_scale, args.config, args.limit_fraction, args.accel_smoothing,
                      args.guard, args.min_part, args.command_rate)
        print_report(rep)
        rc = 0 if rep["parts"] else rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
