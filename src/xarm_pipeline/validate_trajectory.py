"""Format pre-flight check for trajectory CSVs (see docs/trajectory_format.md).

Checks only what the CSV itself can tell: columns, numbers, timing, units sanity,
joint limits and step sizes. It needs no robot. Workcell/wall checks live in the
safety module.

Exit code: 0 = OK (warnings allowed), 1 = errors found, 2 = file unreadable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .trajectory_io import TRAJ_COLUMN, list_trajectories, select_trajectory, split_trajectory_ref

JOINT_COLS = [f"q{i}" for i in range(1, 8)]
JOINT_VEL_COLS = [f"dq{i}" for i in range(1, 8)]
POS_COLS = ["x", "y", "z"]
RPY_COLS = ["roll", "pitch", "yaw"]
GRIPPER_RPY_COLS = ["gripper_roll", "gripper_pitch", "gripper_yaw"]
TWIST_COLS = ["vx", "vy", "vz", "vroll", "vpitch", "vyaw"]
KNOWN_COLS = set(["t", "traj"] + JOINT_COLS + JOINT_VEL_COLS + POS_COLS + RPY_COLS + GRIPPER_RPY_COLS + TWIST_COLS)

# xArm 7 joint limits in rad, from xarm-python-sdk core/config/x_config.py.
# J4 depends on the hardware revision: X3/X13 use (-3.927, 0.192), X4 uses (-0.192, 3.927).
JOINT_LIMITS = {
    "x3": [
        (-2 * math.pi, 2 * math.pi),
        (-2.059488, 2.094395),
        (-2 * math.pi, 2 * math.pi),
        (-3.92699, 0.191986),
        (-2 * math.pi, 2 * math.pi),
        (-1.692969, math.pi),
        (-2 * math.pi, 2 * math.pi),
    ],
    "x4": [
        (-2 * math.pi, 2 * math.pi),
        (-2.059488, 2.094395),
        (-2 * math.pi, 2 * math.pi),
        (-0.191986, 3.92699),
        (-2 * math.pi, 2 * math.pi),
        (-1.692969, math.pi),
        (-2 * math.pi, 2 * math.pi),
    ],
}
# SDK TCP_LIMITS for xArm 7 (base frame, mm).
TCP_LIMITS_MM = {"x": (-750.0, 750.0), "y": (-750.0, 750.0), "z": (-400.0, 1000.0)}
# Approximate xArm 7 reach (mm) measured from the shoulder (J2 axis), which sits 267 mm above the base origin.
SHOULDER_Z_MM = 267.0
REACH_MM = 700.0


@dataclass
class Report:
    kind: str = "unknown"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _first_bad(mask: np.ndarray) -> int:
    return int(np.flatnonzero(mask)[0])


def _check_numeric(df: pd.DataFrame, cols: list[str], rep: Report, allow_nan: bool = False) -> None:
    for c in cols:
        values = pd.to_numeric(df[c], errors="coerce")
        non_numeric = values.isna() & df[c].notna()
        if non_numeric.any():
            i = _first_bad(non_numeric.to_numpy())
            rep.errors.append(f"column '{c}': non-numeric value {df[c].iloc[i]!r} at row {i}")
            continue
        arr = values.to_numpy(dtype=float)
        if not allow_nan and np.isnan(arr).any():
            rep.errors.append(f"column '{c}': empty/NaN at row {_first_bad(np.isnan(arr))}")
        if np.isinf(arr).any():
            rep.errors.append(f"column '{c}': inf at row {_first_bad(np.isinf(arr))}")


def _check_header(df: pd.DataFrame, rep: Report) -> None:
    for c in df.columns:
        s = str(c)
        if s in KNOWN_COLS:
            continue
        norm = s.strip().lower()
        if norm in KNOWN_COLS:
            rep.errors.append(
                f"header '{s}' is not an accepted name; did you mean '{norm}'? "
                "(names are case-sensitive and must have no surrounding spaces)"
            )
        elif ";" in s or "\t" in s:
            rep.errors.append(f"header '{s}' looks like a non-comma delimiter; the file must be comma-separated")
        elif s.startswith("Unnamed:"):
            rep.info.append(f"unnamed column '{s}' (e.g. a pandas index) is ignored")
        else:
            rep.info.append(f"extra column '{s}' is ignored by playback")


def _check_time(t: np.ndarray, rep: Report) -> None:
    if len(t) < 2:
        rep.errors.append("need at least 2 rows")
        return
    if abs(t[0]) > 1e-9:
        rep.warnings.append(f"t[0] = {t[0]} (expected 0.0)")
    dt = np.diff(t)
    if (dt <= 0).any():
        i = _first_bad(dt <= 0)
        rep.errors.append(f"t must be strictly increasing; t[{i}]={t[i]} -> t[{i + 1}]={t[i + 1]}")
        return
    rep.info.append(
        f"rows={len(t)} duration={t[-1] - t[0]:.3f}s dt min/median/max = "
        f"{dt.min():.4f}/{np.median(dt):.4f}/{dt.max():.4f}s"
    )
    if dt.min() < 0.004:
        rep.warnings.append(f"dt min {dt.min():.4f}s < 4 ms; the host loop cannot reliably stream faster than ~250 Hz")
    if dt.max() > 0.1:
        rep.warnings.append(f"dt max {dt.max():.3f}s > 100 ms; servo-mode streaming expects small steps (10-30 ms)")
    elif np.median(dt) > 0.025:
        rep.warnings.append(f"median dt {np.median(dt) * 1000:.0f} ms: for fast motion in servo mode use <= 10-20 ms "
                            "setpoints (sparse setpoints make the arm dash and wait); the fit/recorded pipelines densify to 100 Hz")


def _validate_joint(df: pd.DataFrame, t: np.ndarray, rep: Report, robot_type: str, max_joint_speed: float) -> None:
    _check_numeric(df, JOINT_COLS, rep)
    if rep.errors:
        return
    q = df[JOINT_COLS].to_numpy(dtype=float)

    types = ["x3", "x4"] if robot_type == "any" else [robot_type]
    for j in range(7):
        lo = min(JOINT_LIMITS[k][j][0] for k in types)
        hi = max(JOINT_LIMITS[k][j][1] for k in types)
        bad = (q[:, j] < lo) | (q[:, j] > hi)
        if bad.any():
            i = _first_bad(bad)
            rep.errors.append(
                f"q{j + 1}={q[i, j]:.4f} rad at row {i} outside limits [{lo:.4f}, {hi:.4f}] (robot type {robot_type})"
            )
    if robot_type == "any":
        in_x3 = ((q[:, 3] >= JOINT_LIMITS["x3"][3][0]) & (q[:, 3] <= JOINT_LIMITS["x3"][3][1])).all()
        in_x4 = ((q[:, 3] >= JOINT_LIMITS["x4"][3][0]) & (q[:, 3] <= JOINT_LIMITS["x4"][3][1])).all()
        if not (in_x3 or in_x4):
            rep.errors.append("q4 uses both signs beyond 0.19 rad; it cannot fit either J4 range (x3: negative, x4: positive)")
        elif not (in_x3 and in_x4):
            fits = "x4 (q4 >= -0.19)" if in_x4 else "x3 (q4 <= 0.19)"
            rep.warnings.append(f"q4 only fits hardware revision {fits}; confirm with --robot-type for the target arm")

    if len(t) >= 2 and (np.diff(t) > 0).all():
        speed = np.abs(np.diff(q, axis=0)) / np.diff(t)[:, None]
        peak = speed.max()
        j = int(np.unravel_index(speed.argmax(), speed.shape)[1])
        rep.info.append(f"peak finite-difference joint speed {peak:.3f} rad/s (q{j + 1})")
        if peak > max_joint_speed:
            rep.errors.append(f"joint speed {peak:.3f} rad/s on q{j + 1} exceeds --max-joint-speed {max_joint_speed} rad/s")

    present = [c for c in JOINT_VEL_COLS if c in df.columns]
    if present and len(present) < 7:
        rep.warnings.append(f"only {present} present; dq columns are ignored unless all of dq1..dq7 exist")
    elif present:
        _check_numeric(df, JOINT_VEL_COLS, rep)
        if not rep.errors:
            dq = df[JOINT_VEL_COLS].to_numpy(dtype=float)
            if np.abs(dq).max() > max_joint_speed:
                rep.errors.append(f"|dq| max {np.abs(dq).max():.3f} rad/s exceeds --max-joint-speed {max_joint_speed}")
            fd = np.gradient(q, t, axis=0)
            mismatch = np.abs(fd - dq).max()
            if mismatch > 0.2:
                rep.warnings.append(
                    f"dq disagrees with d(q)/dt by up to {mismatch:.3f} rad/s; velocity playback will drift from q"
                )


def _orientation(df: pd.DataFrame, rep: Report) -> np.ndarray | None:
    has_rpy = all(c in df.columns for c in RPY_COLS)
    has_grip = all(c in df.columns for c in GRIPPER_RPY_COLS)
    partial = [c for c in RPY_COLS + GRIPPER_RPY_COLS if c in df.columns]
    if has_rpy:
        cols = RPY_COLS
        if has_grip:
            rep.warnings.append("both roll/pitch/yaw and gripper_* present; gripper_* are ignored")
    elif has_grip:
        cols = GRIPPER_RPY_COLS
    else:
        if partial:
            rep.errors.append(f"incomplete orientation group {partial}; give all three or none")
        else:
            rep.info.append("no orientation columns: playback holds the robot's start orientation (or --default-orientation-rpy)")
        return None
    _check_numeric(df, cols, rep)
    if rep.errors:
        return None
    return df[cols].to_numpy(dtype=float)


def _validate_cartesian(
    df: pd.DataFrame,
    t: np.ndarray,
    rep: Report,
    max_tcp_speed: float,
    max_orientation_step_deg: float,
) -> None:
    _check_numeric(df, POS_COLS, rep)
    rpy = _orientation(df, rep)
    if rep.errors:
        return
    p = df[POS_COLS].to_numpy(dtype=float)

    for k, (lo, hi) in TCP_LIMITS_MM.items():
        v = df[k].to_numpy(dtype=float)
        bad = (v < lo) | (v > hi)
        if bad.any():
            i = _first_bad(bad)
            rep.errors.append(f"{k}={v[i]:.1f} mm at row {i} outside xArm 7 TCP limits [{lo}, {hi}] mm")
    span = np.abs(p).max()
    if span < 2.0:
        rep.errors.append(f"max |x,y,z| is {span:.4f}; positions look like meters, expected millimeters")
    reach = np.linalg.norm(p - np.array([0.0, 0.0, SHOULDER_Z_MM]), axis=1)
    if reach.max() > REACH_MM:
        i = int(reach.argmax())
        rep.warnings.append(
            f"row {i} is {reach.max():.0f} mm from the shoulder (reach ~{REACH_MM:.0f} mm); IK will likely fail"
        )

    dt = np.diff(t)
    if (dt > 0).all():
        lin = np.linalg.norm(np.diff(p, axis=0), axis=1) / dt
        rep.info.append(f"peak finite-difference TCP speed {lin.max():.1f} mm/s, max step {np.linalg.norm(np.diff(p, axis=0), axis=1).max():.2f} mm")
        if lin.max() > max_tcp_speed:
            i = int(lin.argmax())
            rep.errors.append(f"TCP speed {lin.max():.1f} mm/s between rows {i}-{i + 1} exceeds --max-tcp-speed {max_tcp_speed}")

    if rpy is not None:
        if np.abs(rpy).max() > 2 * math.pi + 1e-6 and np.abs(rpy).max() > 20:
            rep.errors.append(f"max |orientation| {np.abs(rpy).max():.1f}; looks like degrees, expected radians")
        d = np.diff(rpy, axis=0)
        d = (d + math.pi) % (2 * math.pi) - math.pi
        step_deg = np.degrees(np.abs(d)).max()
        rep.info.append(f"max per-row orientation step {step_deg:.2f} deg (after wrapping)")
        if step_deg > max_orientation_step_deg:
            rep.warnings.append(
                f"orientation step {step_deg:.2f} deg > {max_orientation_step_deg} deg; playback clamps it, so the commanded orientation will lag"
            )
        near_lock = np.abs(np.abs(rpy[:, 1]) - math.pi / 2) < math.radians(5)
        if near_lock.any():
            rep.warnings.append(
                f"pitch within 5 deg of +/-90 deg at row {_first_bad(near_lock)} (RPY gimbal lock; roll/yaw become ill-defined)"
            )

    present = [c for c in TWIST_COLS if c in df.columns]
    if present and len(present) < 6:
        rep.warnings.append(f"only {present} present; twist columns are ignored unless all six exist")
    elif present:
        _check_numeric(df, TWIST_COLS, rep)
        if not rep.errors:
            v = df[["vx", "vy", "vz"]].to_numpy(dtype=float)
            if np.linalg.norm(v, axis=1).max() > max_tcp_speed:
                rep.errors.append(f"|v| max {np.linalg.norm(v, axis=1).max():.1f} mm/s exceeds --max-tcp-speed")


def validate(
    path: str | Path,
    kind: str = "auto",
    robot_type: str = "any",
    max_joint_speed: float = math.pi,
    max_tcp_speed: float = 500.0,
    max_orientation_step_deg: float = 8.0,
) -> dict[str, Report]:
    """Validate 'file.csv' or 'file.csv#selector'. Returns one Report per trajectory, keyed
    'file.csv' for single-trajectory files and 'file.csv#name' for grouped ones."""
    p, selector = split_trajectory_ref(path)
    df = pd.read_csv(p, dtype={TRAJ_COLUMN: str})
    opts = dict(kind=kind, robot_type=robot_type, max_joint_speed=max_joint_speed,
                max_tcp_speed=max_tcp_speed, max_orientation_step_deg=max_orientation_step_deg)
    if TRAJ_COLUMN not in df.columns:
        if selector is not None:
            rep = Report()
            rep.errors.append(f"selector '#{selector}' given but the file has no '{TRAJ_COLUMN}' column")
            return {str(path): rep}
        return {str(p): validate_frame(df, **opts)}

    if df[TRAJ_COLUMN].isna().any():
        rep = Report()
        rep.errors.append(f"empty '{TRAJ_COLUMN}' cell at row {int(df[TRAJ_COLUMN].isna().to_numpy().argmax())}")
        return {str(p): rep}
    names = [n for n, _ in list_trajectories(df)]
    if selector is not None:
        try:
            part = select_trajectory(df, selector, source=str(p))
        except ValueError as e:
            rep = Report()
            rep.errors.append(str(e))
            return {str(path): rep}
        name = selector if selector in names else names[int(selector)]
        selected = {name: part}
    else:
        selected = {n: select_trajectory(df, n) for n in names}
    out = {}
    labels = df[TRAJ_COLUMN].to_numpy()
    for name, part in selected.items():
        rep = validate_frame(part, **opts)
        idx = np.flatnonzero(labels == name)
        if (np.diff(idx) != 1).any():
            rep.warnings.append(f"rows of '{name}' are not contiguous in the file (allowed, but easy to misread)")
        out[f"{p}#{name}"] = rep
    return out


def validate_frame(
    df: pd.DataFrame,
    kind: str = "auto",
    robot_type: str = "any",
    max_joint_speed: float = math.pi,
    max_tcp_speed: float = 500.0,
    max_orientation_step_deg: float = 8.0,
) -> Report:
    rep = Report()
    _check_header(df, rep)
    if kind == "auto":
        if all(c in df.columns for c in JOINT_COLS):
            kind = "joint"
        elif all(c in df.columns for c in POS_COLS):
            kind = "cartesian"
        else:
            rep.errors.append("cannot tell trajectory kind: need q1..q7 (joint) or x,y,z (Cartesian)")
            return rep
        if kind == "joint" and all(c in df.columns for c in POS_COLS):
            rep.warnings.append("file has both q1..q7 and x,y,z; joint playback uses q, Cartesian playback uses x,y,z")
    rep.kind = kind

    required = ["t"] + (JOINT_COLS if kind == "joint" else POS_COLS)
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.errors.append(f"missing required columns {missing}")
        return rep
    _check_numeric(df, ["t"], rep)
    if rep.errors:
        return rep
    t = df["t"].to_numpy(dtype=float)
    _check_time(t, rep)

    if kind == "joint":
        _validate_joint(df, t, rep, robot_type, max_joint_speed)
    else:
        _validate_cartesian(df, t, rep, max_tcp_speed, max_orientation_step_deg)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description="Validate a trajectory CSV against docs/trajectory_format.md")
    p.add_argument("csv", nargs="+", help="Trajectory CSV file(s); file.csv#name or file.csv#index selects one trajectory of a grouped file")
    p.add_argument("--kind", choices=["auto", "joint", "cartesian"], default="auto")
    p.add_argument("--robot-type", choices=["any", "x3", "x4"], default="any", help="xArm 7 hardware revision (affects J4 range)")
    p.add_argument("--max-joint-speed", type=float, default=math.pi, help="rad/s (xArm 7 max is pi)")
    p.add_argument("--max-tcp-speed", type=float, default=500.0, help="mm/s")
    p.add_argument("--max-orientation-step-deg", type=float, default=8.0, help="match cartesian_playback's clamp")
    p.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = p.parse_args()

    results = {}
    worst = 0
    for path in args.csv:
        try:
            reps = validate(
                path,
                kind=args.kind,
                robot_type=args.robot_type,
                max_joint_speed=args.max_joint_speed,
                max_tcp_speed=args.max_tcp_speed,
                max_orientation_step_deg=args.max_orientation_step_deg,
            )
        except Exception as e:  # unreadable file
            results[path] = {"ok": False, "kind": "unknown", "errors": [f"cannot read: {e}"], "warnings": [], "info": []}
            worst = max(worst, 2)
            continue
        for key, rep in reps.items():
            results[key] = {"ok": rep.ok, "kind": rep.kind, "errors": rep.errors, "warnings": rep.warnings, "info": rep.info}
            if not rep.ok:
                worst = max(worst, 1)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for path, r in results.items():
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {path} ({r['kind']})")
            for level, label in (("errors", "error"), ("warnings", "warn"), ("info", "info")):
                for msg in r[level]:
                    print(f"  [{label}] {msg}")
    sys.exit(worst)


if __name__ == "__main__":
    main()
