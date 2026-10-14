"""Generate real-robot test trajectories for the CURRENT safety config.

    python -m src.xarm_pipeline.safety.test_trajectories corners   # slow moves into each cell corner
    python -m src.xarm_pipeline.safety.test_trajectories dynamic   # short strokes at 20/50/100 % of the limits

Both are written into the preprocessed set of the current config
(data/preprocessed/NNN_*/tests/), with the approach data playback uses, so they are run as-is:

    scripts/control/play_cartesian_ik_record.sh "data/preprocessed/NNN_*/tests/workspace_corners.csv#corner_xmax_ymin_zmin" servo 60

Every trajectory is generated in the robot base frame for the firmware TCP point given by
geometry.firmware_tcp_offset. If the robot's TCP offset differs, regenerate with a matching config.

corners: for each of the 8 corners of the cell, the reachable tool position closest to that corner, keeping
         the same clearances playback requires (+ a small buffer). Straight line out from the ready pose
         (via a side waypoint for corners behind the robot), hold, and back. Slow (default 50 mm/s).
dynamic: back-and-forth strokes along x, y and z around a point well inside the cell. Each move is
         rest-to-rest with a smooth (sine) acceleration profile, cruise speed v and peak acceleration a.
         "100 %" is the largest (v, a) along that stroke that keeps tool speed <= 1000 mm/s,
         tool acceleration <= 50 m/s^2, every joint speed <= 180 deg/s and joint acceleration
         <= 1146 deg/s^2 (the firmware limits), computed through IK. 20 % and 50 % scale v and a.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, load_safety_config
from .fit import FIT_VERSION, load_fit_config
from .geometry import MOVING_CAPSULES
from .kinematics import IKFailure, fk_pose, jacobian
from .measure import walls_in_base
from .preprocess import SafetyChecker, validate_cartesian_trajectory
from .preprocessed import find_or_create_set

# Firmware limits (sim firmware v2.4.0 reports these; the xArm 7 data sheet agrees).
TCP_SPEED_MAX = 1000.0                   # mm/s
TCP_ACC_MAX = 50000.0                    # mm/s^2
JOINT_SPEED_MAX = math.radians(180.0)    # rad/s
JOINT_ACC_MAX = math.radians(1145.9)     # rad/s^2


class Planner:
    def __init__(self, cfg_path=None, buffer_mm: float = 5.0, min_sigma: float = 25.0) -> None:
        self.min_sigma = min_sigma  # keep this far from singularities (smallest singular value of W*J, mm/rad)
        self.cfg_path = Path(cfg_path or DEFAULT_CONFIG_PATH)
        self.cfg = load_safety_config(self.cfg_path)
        self.fcfg = load_fit_config(self.cfg_path)
        self.chk = SafetyChecker(self.cfg)
        self.required = self.chk.required_cartesian_mm() + buffer_mm
        self.tcp = list(self.cfg.firmware_tcp_offset_rad)
        self.q_ready = np.radians(self.fcfg.seed_joints_deg)
        self.ready = fk_pose(self.q_ready, tcp_offset=self.tcp)
        self.walls = walls_in_base(self.cfg)

    def sigma(self, q) -> float:
        return float(np.linalg.svd(np.diag([1, 1, 1, 100, 100, 100.0]) @ jacobian(q)[0], compute_uv=False).min())

    def leg(self, poses, q_start, start_pose, check_sigma: bool = True):
        """Continuation IK through poses; (ok, q_end, slack_mm). Also fails near singularities."""
        try:
            Qd, _, _, q_at = self.chk.solve_cartesian_path(poses, q_start, self.tcp, start_pose)
        except IKFailure:
            return False, None, -1e9
        C, Pw = self.chk.wc.clearance_for_q(Qd, MOVING_CAPSULES)
        slack = min(float(np.min(C - self.required[None, :, None])),
                    float(self.chk.wc.base_column_slack(Pw).min()) - 5.0)  # gripper vs robot base, + buffer
        if check_sigma and slack >= 0:
            if min(self.sigma(q) for q in Qd[:: max(1, len(Qd) // 25)]) < self.min_sigma or self.sigma(Qd[-1]) < self.min_sigma:
                return False, q_at[-1], slack
        return slack >= 0, q_at[-1], slack

    def pose_at(self, xyz):
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])] + list(self.ready[3:6])

    # ------------------------------------------------------------------ corners
    def route_ok(self, via: list, target) -> bool:
        poses = [self.pose_at(v) for v in via] + [self.pose_at(target)]
        return self.leg(poses, self.q_ready, self.ready)[0]

    def corner_target(self, corner: np.ndarray) -> tuple[np.ndarray, list]:
        """Closest reachable+safe tool position to `corner` (base frame), and the via points used."""
        start = np.asarray(self.ready[:3], float)
        via = []
        if corner[0] < 100.0:  # behind/next to the base: go around the side, never past the base axis
            via = []
            for r in (350.0, 320.0, 290.0, 260.0, 230.0):
                for z in (start[2], start[2] + 100.0, start[2] + 200.0):
                    cand = np.array([60.0, math.copysign(r, corner[1]), z])
                    if self.route_ok([], cand):
                        via = [cand]
                        break
                if via:
                    break
        origin = via[-1] if via else start
        lo, hi = 0.0, 1.0
        if self.route_ok(via, origin + (corner - origin)):
            lo = 1.0
        else:
            for _ in range(14):  # bisection along the ray toward the corner
                mid = 0.5 * (lo + hi)
                if self.route_ok(via, origin + mid * (corner - origin)):
                    lo = mid
                else:
                    hi = mid
        p = origin + lo * (corner - origin)
        # slide along each axis toward the corner (gets into the corner along the walls)
        step = 40.0
        while step >= 2.0:
            moved = False
            for ax in range(3):
                d = corner[ax] - p[ax]
                if abs(d) < 1e-6:
                    continue
                q = p.copy()
                q[ax] += math.copysign(min(step, abs(d)), d)
                if self.route_ok(via, q):
                    p, moved = q, True
            if not moved:
                step /= 2
        return p, via

    def corner_trajectory(self, target, via, speed: float, hold_s: float, dt: float):
        pts = [np.asarray(self.ready[:3], float)] + [np.asarray(v, float) for v in via] + [np.asarray(target, float)]
        out = pts + pts[-2::-1]  # out and back the same way
        t_all, p_all = [0.0], [out[0]]
        for i in range(1, len(out)):
            ts, ps = rest_to_rest(out[i - 1], out[i], speed, speed * 2.0, dt)  # gentle accel (~speed/0.5 s)
            t_all += list(t_all[-1] + ts[1:])
            p_all += list(ps[1:])
            if i == len(pts) - 1:  # hold at the corner
                n_hold = int(round(hold_s / dt))
                t_all += [t_all[-1] + dt * (k + 1) for k in range(n_hold)]
                p_all += [p_all[-1]] * n_hold
        return np.asarray(t_all), np.asarray(p_all)

    # ------------------------------------------------------------------ dynamic
    def stroke_limits(self, center, axis: int, length: float):
        """(v100, a100) for a stroke of `length` along `axis` through `center`, plus diagnostics."""
        s = np.arange(-length / 2, length / 2 + 1e-9, 1.0)
        pts = np.tile(np.asarray(center, float), (len(s), 1))
        pts[:, axis] += s
        poses = [self.pose_at(p) for p in pts]
        ok, q_c, _ = self.leg([self.pose_at(center)], self.q_ready, self.ready)
        if not ok:
            raise RuntimeError("center of the dynamic tests is not reachable/safe")
        _, q0, _ = self.leg([poses[0]], q_c, self.pose_at(center))
        _, _, _, Q = self.chk.solve_cartesian_path(poses[1:], q0, self.tcp, poses[0])
        Q = np.vstack([q0, np.asarray(Q)])
        dq = np.gradient(Q, s, axis=0)       # rad per mm
        ddq = np.gradient(dq, s, axis=0)     # rad per mm^2
        g1 = np.abs(dq).max()
        v100 = min(TCP_SPEED_MAX, JOINT_SPEED_MAX / g1)
        # |q''| v^2 + |q'| a <= qdd_max for every joint and point
        with np.errstate(divide="ignore", invalid="ignore"):
            room = (JOINT_ACC_MAX - np.abs(ddq) * v100 ** 2) / np.abs(dq)
        room = np.where(np.abs(dq) < 1e-9, np.inf, room)
        a100 = float(min(TCP_ACC_MAX, room.min()))
        if a100 <= 0:  # curvature alone uses the acceleration budget: slow down
            v100 *= 0.7
            room = np.where(np.abs(dq) < 1e-9, np.inf, (JOINT_ACC_MAX - np.abs(ddq) * v100 ** 2) / np.abs(dq))
            a100 = float(min(TCP_ACC_MAX, room.min()))
        sig = min(np.linalg.svd(np.diag([1, 1, 1, 100, 100, 100.0]) @ jacobian(q)[0], compute_uv=False).min()
                  for q in Q[:: max(1, len(Q) // 20)])
        return float(v100), a100, {"min_singular_value": round(float(sig), 1),
                                   "binding": "tcp speed" if v100 >= TCP_SPEED_MAX - 1e-6 else "joint speed"}

    def joint_profile(self, t, P, q_start):
        poses = [self.pose_at(p) for p in P]
        _, _, _, Q = self.chk.solve_cartesian_path(poses[1:], q_start, self.tcp, poses[0])
        Q = np.vstack([q_start, np.asarray(Q)])
        qd = np.gradient(Q, t, axis=0)
        qdd = np.gradient(qd, t, axis=0)
        return Q, np.abs(qd).max(), np.abs(qdd).max()


def rest_to_rest(p0, p1, v: float, a: float, dt: float):
    """Straight move p0 -> p1 from rest to rest: sine-shaped acceleration (peak a), cruise v."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    D = float(np.linalg.norm(p1 - p0))
    if D < 1e-9:
        return np.array([0.0]), p0[None]
    v = min(v, math.sqrt(2 * a * D / math.pi))       # short move: never reaches the cruise speed
    Ta = math.pi * v / (2 * a)                       # accel phase duration; covers v*Ta/2
    Tc = (D - v * Ta) / v                            # cruise
    T = 2 * Ta + Tc
    t = np.arange(0.0, T + 1e-12, dt)
    if t[-1] < T - 1e-9:
        t = np.r_[t, T]
    w = math.pi / Ta

    def dist(tt):
        if tt <= Ta:
            return v / 2 * (tt - math.sin(w * tt) / w)
        if tt <= Ta + Tc:
            return v * Ta / 2 + v * (tt - Ta)
        r = T - tt
        return D - v / 2 * (r - math.sin(w * r) / w)

    s = np.array([dist(x) for x in t])
    return t, p0 + (s / D)[:, None] * (p1 - p0)


def write_bank(path: Path, frames: dict[str, pd.DataFrame], approach: dict[str, list], ready_deg: list,
               extra: dict) -> None:
    """Grouped CSV + a .fit.json with the approach data prepare_cartesian_start.sh uses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.concat([f.assign(traj=n) for n, f in frames.items()], ignore_index=True)
    df = df[["traj"] + [c for c in df.columns if c != "traj"]]
    df.to_csv(path, index=False, float_format="%.4f")
    meta = {"generated_by": "safety.test_trajectories", "fit_version": FIT_VERSION, "grouped": True,
            "trajectories": [{"name": n, "result": "generated",
                              "parts": [{"name": n, "start_joints_deg": approach[n], "approach_via_joints_deg": ready_deg}]}
                             for n in frames], **extra}
    path.with_suffix(".fit.json").write_text(json.dumps(meta, indent=2))


def frame(t, P, rpy) -> pd.DataFrame:
    return pd.DataFrame({"t": t, "x": P[:, 0], "y": P[:, 1], "z": P[:, 2],
                         "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]})


def make_corners(args) -> int:
    pl = Planner(args.config, min_sigma=args.min_sigma)
    w = pl.walls
    set_dir = find_or_create_set(pl.cfg_path)
    frames, approach, report = {}, {}, []
    ready_deg = np.degrees(pl.q_ready).round(4).tolist()
    for xk in ("x_min", "x_max"):
        for yk in ("y_min", "y_max"):
            for zk in ("z_min", "z_max"):
                corner = np.array([w[xk], w[yk], w[zk]])
                name = f"corner_{xk.replace('_', '')}_{yk.replace('_', '')}_{zk.replace('_', '')}"
                t0 = time.time()
                target, via = pl.corner_target(corner)
                origin = via[-1] if via else np.asarray(pl.ready[:3], float)
                for _ in range(20):  # the full out-and-back must pass the playback check; else pull back
                    t, P = pl.corner_trajectory(target, via, args.speed, args.hold, args.dt)
                    poses = [pl.pose_at(p) for p in P]
                    res = validate_cartesian_trajectory(pl.cfg, list(t), poses, pl.q_ready, pl.tcp,
                                                        include_approach=False, policy="reject")
                    if res.ok:
                        break
                    target = origin + 0.95 * (target - origin)
                d = target - corner
                info = {"name": name, "corner_mm": corner.round(1).tolist(), "reached_mm": target.round(1).tolist(),
                        "distance_to_corner_mm": round(float(np.linalg.norm(d)), 1),
                        "per_axis_gap_mm": np.abs(d).round(1).tolist(), "via": [v.round(1).tolist() for v in via],
                        "duration_s": round(float(t[-1]), 1), "safety_check": res.message,
                        "seconds": round(time.time() - t0, 1)}
                report.append(info)
                print(f"{name}: reached {info['reached_mm']} | gap to corner per axis {info['per_axis_gap_mm']} mm"
                      f" | {t[-1]:.0f} s | {res.message[:60]}")
                if not res.ok:
                    print(f"   NOT WRITTEN: {res.message}")
                    continue
                frames[name] = frame(t, P, pl.ready[3:6])
                approach[name] = ready_deg
    out = set_dir / "tests" / "workspace_corners.csv"
    write_bank(out, frames, approach, ready_deg, {"report": report, "speed_mm_s": args.speed, "config": str(pl.cfg_path)})
    print(f"\nwrote {out} ({len(frames)} trajectories) and {out.with_suffix('.fit.json')}")
    return 0


def make_dynamic(args) -> int:
    pl = Planner(args.config)
    set_dir = find_or_create_set(pl.cfg_path)
    w = pl.walls
    center = np.array(args.center if args.center else [250.0, (w["y_min"] + w["y_max"]) / 2, 450.0])
    ok, q_c, slack = pl.leg([pl.pose_at(center)], pl.q_ready, pl.ready)
    if not ok:
        raise SystemExit(f"dynamic-test center {center.tolist()} is not reachable/safe (slack {slack:.1f}); pass --center")
    frames, approach, report = {}, {}, []
    for axis, an in enumerate("xyz"):
        length = args.lengths[axis]
        v100, a100, diag = pl.stroke_limits(center, axis, length)
        for pct in args.percent:
            v, a = v100 * pct / 100, a100 * pct / 100
            e = np.zeros(3)
            e[axis] = length / 2
            seq = [center, center + e] + [center - e, center + e] * args.cycles + [center]
            t_all, P_all = [0.0], [center]
            for i in range(1, len(seq)):
                ts, ps = rest_to_rest(seq[i - 1], seq[i], v, a, args.dt)
                t_all += list(t_all[-1] + ts[1:])
                P_all += list(ps[1:])
                t_all.append(t_all[-1] + args.dwell)  # short dwell at each end
                P_all.append(P_all[-1])
            t, P = np.asarray(t_all), np.asarray(P_all)
            name = f"dyn_{an}_{pct:g}pct"
            poses = [pl.pose_at(p) for p in P]
            res = validate_cartesian_trajectory(pl.cfg, list(t), poses, q_c, pl.tcp, include_approach=False,
                                                policy="reject")
            _, qd, qdd = pl.joint_profile(t, P, q_c)
            vt = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.diff(t)
            vel = np.gradient(P, t, axis=0)
            at = np.linalg.norm(np.gradient(vel, t, axis=0), axis=1)
            info = {"name": name, "axis": an, "percent": pct, "v_cruise_mm_s": round(v, 1), "a_peak_mm_s2": round(a, 1),
                    "v100": round(v100, 1), "a100": round(a100, 1), **diag,
                    "measured_tcp_speed_max": round(float(vt.max()), 1), "measured_tcp_acc_max": round(float(at.max()), 1),
                    "joint_speed_max_deg_s": round(math.degrees(qd), 1), "joint_acc_max_deg_s2": round(math.degrees(qdd), 1),
                    "duration_s": round(float(t[-1]), 2), "safety_check": res.message}
            report.append(info)
            print(f"{name:14s} v {v:6.0f} mm/s  a {a / 1000:5.1f} m/s^2 | measured: tcp {vt.max():6.0f} mm/s "
                  f"{at.max() / 1000:5.1f} m/s^2, joints {math.degrees(qd):5.0f} deg/s {math.degrees(qdd):6.0f} deg/s^2"
                  f" | {t[-1]:5.1f} s | {'OK' if res.ok else 'REJECTED'}")
            if res.ok:
                frames[name] = frame(t, P, pl.ready[3:6])
                approach[name] = np.degrees(q_c).round(4).tolist()
            else:
                print(f"   NOT WRITTEN: {res.message}")
        print(f"   {an}-stroke 100 % = v {v100:.0f} mm/s, a {a100 / 1000:.1f} m/s^2 ({diag['binding']} binds; "
              f"min singular value {diag['min_singular_value']})")
    out = set_dir / "tests" / "dynamic_tests.csv"
    write_bank(out, frames, approach, np.degrees(pl.q_ready).round(4).tolist(),
               {"report": report, "center_mm": center.tolist(), "stroke_mm": dict(zip("xyz", args.lengths)), "config": str(pl.cfg_path),
                "limits": {"tcp_speed_mm_s": TCP_SPEED_MAX, "tcp_acc_mm_s2": TCP_ACC_MAX,
                           "joint_speed_deg_s": 180.0, "joint_acc_deg_s2": 1145.9}})
    print(f"\nwrote {out} ({len(frames)} trajectories) and {out.with_suffix('.fit.json')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corners")
    c.add_argument("--config", default=None)
    c.add_argument("--speed", type=float, default=50.0, help="mm/s")
    c.add_argument("--hold", type=float, default=1.0, help="seconds at the corner")
    c.add_argument("--dt", type=float, default=0.03)
    c.add_argument("--min-sigma", type=float, default=25.0, help="singularity margin (mm/rad)")
    d = sub.add_parser("dynamic")
    d.add_argument("--config", default=None)
    d.add_argument("--center", type=float, nargs=3, default=None, help="base-frame tool position, mm")
    d.add_argument("--lengths", type=float, nargs=3, default=[200.0, 250.0, 250.0], metavar=("X", "Y", "Z"),
                   help="stroke length per axis, mm (x is limited by the robot base behind the center)")
    d.add_argument("--cycles", type=int, default=2)
    d.add_argument("--percent", type=float, nargs="+", default=[20, 50, 100])
    d.add_argument("--dwell", type=float, default=0.2)
    d.add_argument("--dt", type=float, default=0.01, help="sample period; 10 ms = 10 mm steps at 1 m/s")
    args = p.parse_args(argv)
    return make_corners(args) if args.cmd == "corners" else make_dynamic(args)


if __name__ == "__main__":
    sys.exit(main())
