"""Can the xArm 7 recreate the recorded (mocap) trajectories, and with what extra hardware?

Compares end-effector designs on every recorded trajectory with the same machinery the playback uses
(safety.fit: rigid re-placement by translation + yaw, continuation IK, full-arm wall/self-collision checks,
recursive splitting when a trajectory cannot fit in one piece):

  tooldown      positions only, gripper pointing down (what safety.recorded does today); reference
  arm_down      sensor rigidly mounted at the gripper tip, FULL recorded orientation tracked; the mount is
                chosen so the trajectory's mean orientation has the gripper pointing down
  arm_fwd       same, mean orientation = gripper pointing horizontally away from the robot
  motor         one extra motor on the gripper, sensor on its shaft (on the axis). The motor absorbs all
                rotation about its axis; the arm only has to point the motor axis (2 DOF) + position
  tooldown_nowalls   as tooldown, without the workcell walls (only the robot's mounting plane and itself)
  stick_L<mm>_<rule> a stick held by the gripper along its approach axis, <mm> from the flange to the tip; only
                the tip's POSITION is recreated (orientation ignored), no walls. The stick direction follows a
                pivot rule computed after placement (StickFitter): v (vertical), up<h> (from a pivot h above the
                path), back<h> (from a pivot h toward the robot axis: a wand pointing away from the robot)

Recorded orientation must keep its relation to gravity, so placements only translate and yaw.

Outputs (outputs/feasibility/<run>/): one JSON per (design, trajectory), summary.csv, and the configs used.
Write-up: docs/experiments/2026-10-28_recreate-trajectories-feasibility.md (~10 min per job; use many --workers).

    .venv/bin/python -m src.xarm_pipeline.analysis.recreate_feasibility --designs tooldown arm_down --traj 18 19
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..safety.config import DEFAULT_CONFIG_PATH, load_safety_config
from ..safety.fit import Fitter, Segment, load_fit_config
from ..safety.kinematics import IKFailure, fk_pose, ik_step_to, matrix_to_rpy, rpy_to_matrix
from ..safety.preprocess import DEFAULT_CARTESIAN_SEED_DEG, interpolate_pose
from ..safety.recorded import Limits, smooth

REC_DIR = Path("data/recorded_trajectories/5ec359d2")
GRIPPER_TIP_MM = 172.0
TOOL_DOWN = rpy_to_matrix(math.pi, 0.0, 0.0)          # flange z pointing down, x along base +x
TOOL_FWD = rpy_to_matrix(0.0, math.pi / 2, 0.0)       # flange z along +x (away from the robot at yaw 0)
STICK_RADIUS_MM = 15.0


# ----------------------------------------------------------------------------- rotations
def rpy_mats(rpy: np.ndarray) -> np.ndarray:
    return np.stack([rpy_to_matrix(*r) for r in rpy])


def chordal_mean(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R.mean(axis=0))
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def axis_angle_rot(u: np.ndarray, phi: float) -> np.ndarray:
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(phi) * K + (1 - math.cos(phi)) * K @ K


def swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        return axis_angle_rot(perp / np.linalg.norm(perp), math.pi)
    return axis_angle_rot(v / s, math.atan2(s, c))


def frame_with_z(z: np.ndarray) -> np.ndarray:
    """A rotation whose third column is z (any completion)."""
    return swing(np.array([0.0, 0.0, 1.0]), z / np.linalg.norm(z))


# ----------------------------------------------------------------------------- data
def load_recording(num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(REC_DIR / f"{num:04d}.csv")
    return df["t"].to_numpy(float), df[["x", "y", "z"]].to_numpy(float), rpy_mats(df[["roll", "pitch", "yaw"]].to_numpy(float))


def best_motor_axis(Rs_all: np.ndarray, n: int = 4000) -> np.ndarray:
    """Sensor-frame axis whose world direction varies least over all recordings (max angle from its mean)."""
    i = np.arange(n) + 0.5
    ph, th = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    A = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], 1)
    U = np.einsum("nij,kj->nki", Rs_all, A)
    ub = U.mean(0)
    ub /= np.linalg.norm(ub, axis=1, keepdims=True)
    spread = np.arccos(np.clip(np.einsum("nki,ki->nk", U, ub), -1, 1)).max(0)
    return A[int(spread.argmin())]


# ----------------------------------------------------------------------------- designs
def design_targets(design: str, t, p, Rs, motor_axis_s=None) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """(TCP positions, TCP=flange orientations, config overrides, info) for one design, in the mocap frame."""
    info: dict = {}
    tip_cfg = {"tool_offset_mm": [0.0, 0.0, GRIPPER_TIP_MM], "firmware_tcp_offset": [0.0, 0.0, GRIPPER_TIP_MM, 0, 0, 0]}
    if design == "tooldown":
        return p, np.repeat(TOOL_DOWN[None], len(t), 0), tip_cfg, info
    if design in ("arm_down", "arm_fwd"):
        F = TOOL_DOWN if design == "arm_down" else TOOL_FWD
        Rbar = chordal_mean(Rs)
        Rf = Rs @ (Rbar.T @ F)[None]                    # R_f = R_s M^T with mount M = F^T Rbar
        dev = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", Rf, F) - 1) / 2, -1, 1)))
        info["flange_dev_from_nominal_deg_max"] = float(dev.max())
        return p, Rf, tip_cfg, info
    if design == "motor":
        a_s = motor_axis_s
        u = Rs @ a_s                                     # world direction of the motor axis (fixed by the data)
        ubar = u.mean(0)
        ubar /= np.linalg.norm(ubar)
        # Motor tilt from the gripper axis: chosen so that at the mean axis direction the gripper points down.
        beta = math.acos(float(np.clip(-ubar[2], -1, 1)))
        horiz = np.array([ubar[0], ubar[1], 0.0])
        psi = math.atan2(horiz[1], horiz[0]) if np.linalg.norm(horiz) > 1e-9 else 0.0
        F = rpy_to_matrix(0.0, 0.0, psi) @ TOOL_DOWN
        a_f = F.T @ ubar                                 # motor axis in the flange frame
        Rf = np.empty_like(Rs)
        for k in range(len(t)):
            R0 = swing(F @ a_f, u[k]) @ F
            B = R0 @ F.T
            uk = u[k]
            Kx = np.array([[0, -uk[2], uk[1]], [uk[2], 0, -uk[0]], [-uk[1], uk[0], 0]])
            c1 = np.trace(B) - uk @ B @ uk
            c2 = np.trace(Kx @ B)
            Rf[k] = axis_angle_rot(uk, math.atan2(c2, c1)) @ R0   # family member closest to F
        # motor angle: R_s = R_f M Rz(theta) S with M e_z = a_f, S^T e_z = a_s
        M = frame_with_z(a_f)
        S = frame_with_z(a_s).T
        X = np.einsum("ij,njk,nkl,lm->nim", M.T, np.transpose(Rf, (0, 2, 1)), Rs, S.T)
        theta = np.unwrap(np.arctan2(X[:, 1, 0], X[:, 0, 0]))
        resid = float(np.degrees(np.arccos(np.clip(X[:, 2, 2], -1, 1))).max())
        thd = np.gradient(theta, t)
        dev = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", Rf, F) - 1) / 2, -1, 1)))
        info.update(motor_tilt_from_gripper_axis_deg=math.degrees(beta), motor_axis_in_sensor=a_s.round(4).tolist(),
                    motor_axis_in_flange=a_f.round(4).tolist(), motor_range_deg=float(np.degrees(np.ptp(theta))),
                    motor_speed_deg_s_max=float(np.degrees(np.abs(thd).max())),
                    motor_speed_deg_s_p95=float(np.degrees(np.percentile(np.abs(thd), 95))),
                    motor_check_residual_deg=resid, flange_dev_from_nominal_deg_max=float(dev.max()))
        return p, Rf, tip_cfg, info
    if design == "tooldown_nowalls":
        return p, np.repeat(TOOL_DOWN[None], len(t), 0), dict(tip_cfg, walls=NO_WALLS), info
    if design.startswith("stick"):
        # stick_L<len>_<rule>: the stick direction is set after placement by StickFitter (see there).
        L, rule, h = parse_stick(design)
        if rule == "back":   # horizontal wand: the ready pose with J6 turned -90 deg (flange at (321, 0, 473), pointing +x)
            pref, seed = [330.0 + L, 0.0, 450.0], [0.0, -28.17, 0.0, 23.24, 0.0, -38.59, 0.0]
        else:                # pen held down (vertical or fanning out): flange high, tip low
            tip_z = max(200.0, 620.0 - L)
            pref, seed = [350.0, 0.0, tip_z], ready_pose_deg([300.0, 0.0, min(tip_z + L, 700.0)], TOOL_DOWN)
        cfg = {"tool_offset_mm": [0.0, 0.0, L], "firmware_tcp_offset": [0.0, 0.0, L, 0, 0, 0],
               "radii_tool": STICK_RADIUS_MM, "seed_joints_deg": seed, "walls": NO_WALLS,
               "preferred_center_base_mm": pref, "spin_range_deg": 90.0}
        info.update(stick_from_flange_mm=L, stick_rule=rule, pivot_mm=h)
        return p, np.repeat(TOOL_DOWN[None], len(t), 0), cfg, info
    raise ValueError(design)


# Stick tests: no walls, only the table the robot stands on (the base mounting plane) and the robot itself.
NO_WALLS = {"x_min": -5000.0, "x_max": 5000.0, "y_min": -5000.0, "y_max": 5000.0, "z_min": 0.0, "z_max": 5000.0}


def parse_stick(design: str) -> tuple[float, str, float]:
    parts = design.split("_")
    L = float(parts[1][1:])
    rule = parts[2].rstrip("0123456789")
    h = float(parts[2][len(rule):] or 0.0)
    if rule not in ("v", "up", "back"):
        raise ValueError(design)
    return L, rule, h


class StickFitter(Fitter):
    """Fitter for a stick whose tip position is all that matters. The tool orientation is not part of the
    trajectory: it is computed from the placed tip positions (robot base frame) by a pivot rule.

      v       stick always vertical, pointing down (a pen)
      up<h>   stick points from a pivot h mm above the placed path's center to the tip (fans out downward;
              the gripper moves on a copy of the path shrunk toward the pivot)
      back<h> stick points from a pivot h mm from the path's center horizontally toward the robot's z axis
              to the tip (a wand pointing away from the robot; reach = arm + stick)

    The search's 'spin' turns the gripper about the stick (free: it does not move the tip)."""

    def __init__(self, cfg, fcfg, rule: str, h: float) -> None:
        super().__init__(cfg, fcfg, None)
        self.rule, self.h = rule, h

    def stick_dirs(self, p: np.ndarray) -> np.ndarray:
        if self.rule == "v":
            return np.repeat(np.array([[0.0, 0.0, -1.0]]), len(p), 0)
        c = 0.5 * (p.min(0) + p.max(0))
        if self.rule == "up":
            P = c + np.array([0.0, 0.0, self.h])
        else:
            r = np.array([c[0], c[1], 0.0])
            n = np.linalg.norm(r)
            P = c - self.h * (r / n if n > 1.0 else np.array([1.0, 0.0, 0.0]))
        d = p - P
        return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)

    def apply(self, seg, th):
        p, _, tw = super().apply(seg, th)
        d = self.stick_dirs(p)
        ref = TOOL_FWD if self.rule == "back" else TOOL_DOWN
        spin = rpy_to_matrix(0.0, 0.0, math.radians(th.get("spin", 0.0)))
        Rs = np.stack([swing(ref[:, 2], dk) @ ref @ spin for dk in d])
        return p, Rs, tw


def ready_pose_deg(flange_xyz, R_target) -> list[float]:
    """Start/approach joint pose with the flange at flange_xyz (base frame) and orientation R_target, reached by
    continuation IK from the normal ready pose. With a long stick the normal ready pose (flange 300 mm above the
    base, gripper down) would put the stick through the table."""
    q = np.radians(DEFAULT_CARTESIAN_SEED_DEG)
    a = fk_pose(q)
    b = list(flange_xyz) + list(matrix_to_rpy(R_target))
    for s in np.linspace(0, 1, 60)[1:]:
        q = ik_step_to(interpolate_pose(a, b, s), q)
    return np.degrees(q).round(3).tolist()


def write_config(overrides: dict, out: Path) -> Path:
    raw = yaml.safe_load(Path(DEFAULT_CONFIG_PATH).read_text())
    for k in ("seed_joints_deg", "preferred_center_base_mm", "spin_range_deg"):
        if k in overrides:
            raw.setdefault("fit", {})[k] = overrides[k]
    if "walls" in overrides:
        raw["workcell"] = {"walls_in_base_mm": overrides["walls"]}
        raw.pop("robot_base_in_world", None)
    g = raw["geometry"]
    g["tool_offset_mm"] = overrides["tool_offset_mm"]
    g["firmware_tcp_offset"] = overrides["firmware_tcp_offset"]
    if "radii_tool" in overrides:
        g["radii_mm"]["tool"] = overrides["radii_tool"]
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


# ----------------------------------------------------------------------------- evaluation
def joint_peaks(fitter: Fitter, part: dict, time_scale: float, n_smooth: int = 5) -> dict:
    """Joint speed/acc along a fitted part at time_scale 1 (original speed) -> required slowdown."""
    seg, th = part["segment"], part["params"]
    poses = fitter.placed_poses(seg, th)
    q0 = fitter.start_joints(seg, th)
    _, _, _, Q = fitter.chk.solve_cartesian_path(poses, q0, fitter.tcp, poses[0])
    Q = np.asarray(Q)
    t1 = seg.t / time_scale
    qd = np.abs(np.gradient(Q, t1, axis=0)).max(0)
    qdd = np.abs(np.gradient(np.gradient(smooth(Q, n_smooth), t1, axis=0), t1, axis=0)).max(0)
    lim = Limits().scaled(0.9)
    p, _, _ = fitter.apply(seg, th)
    v = np.linalg.norm(np.gradient(p, t1, axis=0), axis=1)
    return {"joint_speed_deg_s": np.degrees(qd).round(1).tolist(), "joint_acc_deg_s2": np.degrees(qdd).round(0).tolist(),
            "scale_for_joint_limits": float(max(qd.max() / lim.joint_speed, math.sqrt(qdd.max() / lim.joint_acc))),
            "scale_for_tcp_speed": float(v.max() / lim.tcp_speed)}


def run_one(design: str, num: int, out_dir: str, time_scale: float, max_depth: int, space_scale: float = 1.0) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    res = {"design": design, "traj": num, "time_scale": time_scale, "space_scale": space_scale}
    try:
        t, p, Rs = load_recording(num)
        c = 0.5 * (p.min(0) + p.max(0))
        p = c + space_scale * (p - c)    # a bigger copy of the path (same timing): how much range is left
        motor_axis = np.load(out / "motor_axis_sensor.npy") if design == "motor" else None
        P, Rf, ov, info = design_targets(design, t, p, Rs, motor_axis)
        res.update(info)
        cfg_path = write_config(ov, out / "configs" / f"{design}__{num:04d}.yaml")  # one file per job: workers run in parallel
        cfg = load_safety_config(cfg_path)
        fcfg = load_fit_config(cfg_path)
        fcfg.max_depth = max_depth
        if design.startswith("stick"):
            fitter = StickFitter(cfg, fcfg, *parse_stick(design)[1:])
        else:
            fitter = Fitter(cfg, fcfg, None)
        seg = Segment((t - t[0]) * time_scale, P, Rf, None, f"{num:04d}", (0, len(t) - 1))
        # 1. the whole trajectory in one piece (the main question); joint speeds at the best placement found,
        #    even if it does not clear the walls, to separate speed limits from workspace limits
        th, slack, _ = fitter.fit(seg)
        res.update(whole_slack_mm=round(float(slack), 1), whole_params={k: round(float(v), 2) for k, v in th.items()})
        if isinstance(fitter, StickFitter):
            pw, Rw, _ = fitter.apply(seg, th)
            d = Rw[:, :, 2]
            flange = pw - cfg.tool_offset_mm[2] * d
            res.update(tip_path_extent_mm=np.ptp(pw, 0).round(1).tolist(), flange_path_extent_mm=np.ptp(flange, 0).round(1).tolist(),
                       tip_reach_from_base_axis_mm_max=float(np.linalg.norm(pw[:, :2], axis=1).max()),
                       stick_tilt_from_vertical_deg=[float(np.degrees(np.arccos(np.clip(-d[:, 2], -1, 1))).min()),
                                                     float(np.degrees(np.arccos(np.clip(-d[:, 2], -1, 1))).max())])
        try:
            res["whole_joint"] = joint_peaks(fitter, {"segment": seg, "params": th}, time_scale)
        except IKFailure as e:
            res["whole_joint"] = {"error": str(e)}
        # 2. if it does not fit, how it would split (as the playback pipeline would do)
        if slack >= fcfg.min_slack_mm:
            fitted, dropped = [{"segment": seg, "params": th, "slack": slack}], []
        elif max_depth > 0:
            fitted, dropped = fitter.fit_or_split(seg)
        else:
            fitted, dropped = [], [{"segment": seg, "best_slack": slack}]
        T = float(seg.t[-1])
        parts = []
        for f in sorted(fitted, key=lambda f: f["segment"].src_rows[0]):
            s = f["segment"]
            d = {"rows": list(s.src_rows), "duration_s_original": float(s.t[-1] / time_scale),
                 "params": {k: round(float(v), 2) for k, v in f["params"].items()}, "slack_mm": round(float(f["slack"]), 1)}
            try:
                d.update(joint_peaks(fitter, f, time_scale))
            except IKFailure as e:
                d["joint_error"] = str(e)
            parts.append(d)
        kept = sum(x["duration_s_original"] for x in parts) * time_scale
        res.update(n_parts=len(parts), n_dropped=len(dropped), kept_fraction=kept / T if T else 0.0,
                   longest_part_fraction=max((x["duration_s_original"] * time_scale for x in parts), default=0) / T,
                   whole=len(parts) == 1 and not dropped, parts=parts,
                   dropped=[{"rows": list(d["segment"].src_rows), "best_slack_mm": round(float(d["best_slack"]), 1)} for d in dropped],
                   required_time_scale=max([x.get("scale_for_joint_limits", 0) for x in parts] + [0]),
                   required_time_scale_tcp=max([x.get("scale_for_tcp_speed", 0) for x in parts] + [0]))
    except Exception as e:  # keep the sweep going; record the failure
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()
    res["seconds"] = round(time.time() - t0, 1)
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / f"{design}__{num:04d}.json").write_text(json.dumps(res, indent=1))
    return res


def summarize(out: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((out / "results").glob("*.json")):
        r = json.loads(f.read_text())
        wj = r.get("whole_joint") or {}
        r["whole_scale_joint"] = wj.get("scale_for_joint_limits")
        rows.append({k: r.get(k) for k in ("design", "traj", "whole", "whole_slack_mm", "whole_scale_joint", "n_parts", "n_dropped", "kept_fraction",
                                           "longest_part_fraction", "required_time_scale", "required_time_scale_tcp",
                                           "flange_dev_from_nominal_deg_max", "motor_speed_deg_s_max",
                                           "motor_tilt_from_gripper_axis_deg", "motor_range_deg", "error", "seconds")})
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", default=["tooldown", "arm_down", "arm_fwd", "motor"])
    ap.add_argument("--traj", nargs="*", type=int, default=None, help="trajectory numbers (default: all)")
    ap.add_argument("--time-scale", type=float, default=2.5, help="slowdown used while fitting (runtime-monitor margin)")
    ap.add_argument("--max-depth", type=int, default=0, help="split depth when the whole trajectory does not fit (0 = do not split, 3 = up to 8 parts)")
    ap.add_argument("--space-scale", type=float, default=1.0, help="enlarge the paths by this factor (range headroom test)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args(argv)
    nums = args.traj or sorted(int(f.stem) for f in REC_DIR.glob("*.csv"))
    out = Path(args.out or f"outputs/feasibility/{time.strftime('%Y%m%d_%H%M%S')}")
    (out / "configs").mkdir(parents=True, exist_ok=True)
    if "motor" in args.designs and not (out / "motor_axis_sensor.npy").exists():
        allR = np.concatenate([load_recording(n)[2] for n in sorted(int(f.stem) for f in REC_DIR.glob("*.csv"))])
        np.save(out / "motor_axis_sensor.npy", best_motor_axis(allR))
    for d in args.designs:  # write configs up front (workers would race)
        t, p, Rs = load_recording(nums[0])
        ma = np.load(out / "motor_axis_sensor.npy") if d == "motor" else None
        write_config(design_targets(d, t, p, Rs, ma)[2], out / "configs" / f"{d}.yaml")
    jobs = [(d, n) for d in args.designs for n in nums
            if not (out / "results" / f"{d}__{n:04d}.json").exists()]
    print(f"{len(jobs)} jobs -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run_one, d, n, str(out), args.time_scale, args.max_depth, args.space_scale) for d, n in jobs]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"[{k}/{len(jobs)}] {r['design']:>14} {r['traj']:04d}: "
                  + (f"ERROR {r['error']}" if "error" in r else
                     f"parts {r['n_parts']} dropped {r['n_dropped']} kept {r['kept_fraction']:.2f} "
                     f"scale {r['required_time_scale']:.2f}") + f"  ({r['seconds']} s)", flush=True)
    df = summarize(out)
    print(df.groupby("design")[["whole", "whole_slack_mm", "whole_scale_joint"]].agg(["mean", "median"]).round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
