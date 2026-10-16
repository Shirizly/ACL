"""Generate real-robot test trajectories for the CURRENT safety config.

    python -m src.xarm_pipeline.safety.test_trajectories corners   # slow moves into each cell corner
    python -m src.xarm_pipeline.safety.test_trajectories dynamic   # short strokes at 20/50/100 % of the limits

Both are written into the preprocessed set of the current config
(data/preprocessed/NNN_*/tests/), with the approach data playback uses, so they are run as-is:

    scripts/control/play_cartesian_ik_record.sh "data/preprocessed/NNN_*/tests/workspace_corners.csv#corner_xmax_ymin_zmin" servo 60

Every trajectory is generated in the robot base frame for the firmware TCP point given by
geometry.firmware_tcp_offset. If the robot's TCP offset differs, regenerate with a matching config.

corners: for each of the 8 corners of the cell, the gripper TIP (center of the gripper box's far face) is brought
         as close to the corner as every check allows (walls + margin, robot body, singularity margin). The tool
         orientation is free: from straight down to aimed at the corner, two spins, best one kept. One tour
         visits the corners directly, corner to corner (around the cell's vertical edges, bottom/top
         alternating), with a waypoint beside the robot only where a direct leg fails. Slow (50 mm/s).
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
from .kinematics import IKFailure, fk_pose, jacobian, matrix_to_rpy, rpy_to_matrix
from .measure import walls_in_base
from .preprocess import SafetyChecker, interpolate_pose, validate_cartesian_trajectory, validate_joint_trajectory
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
    @property
    def tip_local(self) -> np.ndarray:
        """Gripper tip in the flange frame: center of the gripper box's far face (or tool_offset_mm)."""
        c = self.cfg.tool_box_corners_mm()
        if c is None:
            return np.asarray(self.cfg.tool_offset_mm, float)
        far = c[np.isclose(c[:, 2], c[:, 2].max())]
        return far.mean(axis=0)

    @staticmethod
    def frame_from_axis(axis, spin_deg: float = 0.0) -> np.ndarray:
        """Flange rotation whose z (tool) axis is `axis`; spin rotates about it. Down + spin 0 = roll pi."""
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = ref - (ref @ a) * a
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(a, x), a])
        c, s_ = math.cos(math.radians(spin_deg)), math.sin(math.radians(spin_deg))
        return R @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def pose_for_tip(self, tip, R_f) -> list[float]:
        """Firmware TCP pose that puts the gripper tip at `tip` with flange rotation R_f."""
        off = np.asarray(self.tcp[:3], float)
        R_off = rpy_to_matrix(*self.tcp[3:6])
        flange = np.asarray(tip, float) - R_f @ self.tip_local
        return list(flange + R_f @ off) + list(matrix_to_rpy(R_f @ R_off))

    def tip_of_pose(self, pose) -> np.ndarray:
        R_f = rpy_to_matrix(*pose[3:6]) @ rpy_to_matrix(*self.tcp[3:6]).T
        flange = np.asarray(pose[:3], float) - R_f @ np.asarray(self.tcp[:3], float)
        return flange + R_f @ self.tip_local

    def side_vias(self, y_sign: float, z: float) -> list:
        """Waypoints beside the robot (flange position, tool down): the way to corners behind it."""
        out = []
        for r in (350.0, 320.0, 290.0, 260.0, 230.0):
            for dz in (0.0, 100.0, 200.0):
                out.append(self.pose_at([60.0, math.copysign(r, y_sign), z + dz]))
        return out

    def joints_for(self, via_poses, pose):
        """Joints at `pose` reached from the ready pose through via_poses (continuation IK)."""
        return self.leg(via_poses + [pose], self.q_ready, self.ready)[1]

    def closest_tip(self, corner, R_f, via_poses):
        """Closest tip position to `corner` with flange rotation R_f, reachable from the ready pose (through
        via_poses) with every safety check passing. None if this orientation does not work at all."""
        def ok(tip):
            return self.leg(via_poses + [self.pose_for_tip(tip, R_f)], self.q_ready, self.ready)[0]

        start = self.tip_of_pose(via_poses[-1] if via_poses else self.ready)
        if not ok(start):
            start = start + 0.3 * (corner - start)
            if not ok(start):
                return None
        lo, hi = 0.0, 1.0
        if ok(corner):
            lo = 1.0
        else:
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if ok(start + mid * (corner - start)):
                    lo = mid
                else:
                    hi = mid
        tip = start + lo * (corner - start)
        for step in (40.0, 20.0, 10.0, 5.0):   # slide along each axis toward the corner (into it along the walls)
            moved = True
            while moved:
                moved = False
                for ax in range(3):
                    d = corner[ax] - tip[ax]
                    if abs(d) < 1e-6:
                        continue
                    cand = tip.copy()
                    cand[ax] += math.copysign(min(step, abs(d)), d)
                    if ok(cand):
                        tip, moved = cand, True
        return tip

    def joint_polyline_ok(self, nodes) -> bool:
        """Joint-linear moves through `nodes` (rad): wall clearance >= the playback requirement for joint
        trajectories, robot body >= the margin, singularity margin. (The corner targets were placed with 5 mm
        more than this, so paths have room to leave them.)"""
        req = self.chk.required_joint_mm()
        for qa, qb in zip(nodes[:-1], nodes[1:]):
            Q, _ = self.chk.densify_joint_path(np.asarray([qb]), np.asarray(qa))
            C, Pw = self.chk.wc.clearance_for_q(Q, MOVING_CAPSULES)
            if float(C.min()) < req or float(self.chk.wc.base_column_slack(Pw).min()) < 0.0:
                return False
            if min(self.sigma(q) for q in Q[:: max(1, len(Q) // 15)]) < self.min_sigma:
                return False
        return True

    def cart_leg_joints(self, waypoints, q_a, q_ref_a, q_ref_b, step_mm: float = 4.0, step_deg: float = 1.0):
        """Tip moves in straight lines through `waypoints` [(tip, R_f), ...] (orientation slerped), solved with
        posture-biased IK so the arm's posture morphs from q_ref_a to q_ref_b instead of drifting. Returns the
        dense joint path if it passes every check (walls, robot body, singularity margin, no joint jumps,
        arrival in B's posture), else None."""
        from .kinematics import ik_step_biased
        from .preprocess import _axis_angle, _exp_so3

        steps = []
        for (ta, Ra), (tb, Rb) in zip(waypoints[:-1], waypoints[1:]):
            w = _axis_angle(np.asarray(Ra).T @ np.asarray(Rb))
            n = max(1, int(math.ceil(max(np.linalg.norm(np.subtract(tb, ta)) / step_mm,
                                         math.degrees(np.linalg.norm(w)) / step_deg))))
            steps.append((np.asarray(ta), np.asarray(tb), np.asarray(Ra), w, n))
        total = sum(st[-1] for st in steps)
        q, Q, k = np.asarray(q_a, float), [np.asarray(q_a, float)], 0
        qa_ref, qb_ref = np.asarray(q_ref_a, float), np.asarray(q_ref_b, float)
        for ta, tb, Ra, w, n in steps:
            for i in range(1, n + 1):
                f = i / n
                k += 1
                pose = self.pose_for_tip(ta + f * (tb - ta), Ra @ _exp_so3(f * w))
                try:
                    q_new = ik_step_biased(pose, q, qa_ref + (k / total) * (qb_ref - qa_ref), tcp_offset=self.tcp)
                except IKFailure:
                    return None
                if np.degrees(np.abs(q_new - q)).max() > 3.0:   # a jump = branch switch, not a smooth path
                    return None
                q = q_new
                Q.append(q)
        if np.degrees(np.abs(q - qb_ref)).max() > 30.0:
            return None
        Qd, _ = self.chk.densify_joint_path(np.asarray(Q[1:]), Q[0])
        C, Pw = self.chk.wc.clearance_for_q(Qd, MOVING_CAPSULES)
        if float(C.min()) < self.chk.required_joint_mm() or float(self.chk.wc.base_column_slack(Pw).min()) < 0.0:
            return None
        if min(self.sigma(x) for x in Qd[:: max(1, len(Qd) // 30)]) < self.min_sigma:
            return None
        return [list(map(float, x)) for x in Q]

    def tip_points(self, Q) -> np.ndarray:
        return np.array([self.tip_of_pose(fk_pose(q, tcp_offset=self.tcp)) for q in Q])

    def plan_leg(self, pose_a, pose_b, q_a, route_a=(), route_b=(), q_ref_b=None, tol_deg: float = 45.0):
        """Waypoints from pose_a to pose_b that pass every check. In order of preference: direct; through a
        waypoint beside the robot; through the ready pose; and finally back along A's own route from the ready
        pose (route_a) and out along B's (route_b) - those routes were each found safe when the corner was
        searched. Returns (list of poses after pose_a, q at pose_b) or (None, None)."""
        def arrives_sanely(q_b):  # same arm posture as the corner search found (not a contorted equivalent)
            return q_ref_b is None or float(np.degrees(np.abs(np.asarray(q_b) - np.asarray(q_ref_b))).max()) <= tol_deg

        ok, q_b, _ = self.leg([pose_b], q_a, pose_a)
        if ok and arrives_sanely(q_b):
            self.last_leg_kind = "direct"
            return [pose_b], q_b
        ta, tb = self.tip_of_pose(pose_a), self.tip_of_pose(pose_b)
        center = self.tip_of_pose(self.ready)

        def retreat(pose, tip, f):  # same orientation, tip pulled a fraction f toward the middle of the cell
            R_f = rpy_to_matrix(*pose[3:6]) @ rpy_to_matrix(*self.tcp[3:6]).T
            return self.pose_for_tip(tip + f * (center - tip), R_f)

        # retreat from A, turn in open space, approach B: still corner to corner, but never rotating the
        # (long) gripper while it is tucked into a corner
        options = [[retreat(pose_a, ta, f), retreat(pose_b, tb, f), pose_b] for f in (0.25, 0.4, 0.6)]
        y_sign = (ta[1] + tb[1]) or 1.0
        z = max(float(self.ready[2]), 0.5 * (ta[2] + tb[2]))
        options += [[v, pose_b] for v in self.side_vias(y_sign, z)] + [[list(self.ready), pose_b]]
        options.append(list(reversed(route_a)) + [list(self.ready)] + list(route_b) + [pose_b])
        labels = ["retreat-turn-approach"] * 3 + ["beside the robot"] * (len(options) - 5) + ["via ready", "retrace routes via ready"]
        for wps, lab in zip(options, labels):
            ok, q_b, _ = self.leg(wps, q_a, pose_a)
            if ok and arrives_sanely(q_b):
                self.last_leg_kind = lab
                return wps, q_b
        return None, None

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


def _corner_job(job):
    """Best tip position + orientation for one corner (runs in a worker process)."""
    cfg_path, name, corner, min_sigma, max_tilt = job
    pl = Planner(cfg_path, min_sigma=min_sigma)
    corner = np.asarray(corner, float)
    via = []
    if corner[0] < 100.0:  # behind/next to the base: reach it around the side, never past the base axis
        for cand in pl.side_vias(corner[1], float(pl.ready[2])):
            if pl.leg([cand], pl.q_ready, pl.ready)[0]:
                via = [cand]
                break
    down = np.array([0.0, 0.0, -1.0])
    toward = corner - pl.tip_of_pose(via[-1] if via else pl.ready)
    toward /= np.linalg.norm(toward)
    best = None
    for s_ in (0.0, 0.25, 0.5, 0.75, 1.0):         # tool axis from straight down to aimed at the corner
        axis = (1 - s_) * down + s_ * toward
        if np.linalg.norm(axis) < 1e-6:
            continue
        tilt_c = math.degrees(math.acos(np.clip(-axis[2] / np.linalg.norm(axis), -1, 1)))
        if tilt_c > max_tilt + 1e-6:
            continue
        for spin in (0.0, 90.0):                  # the gripper box is longer along y: try both ways round
            R_f = pl.frame_from_axis(axis, spin)
            tip = pl.closest_tip(corner, R_f, via)
            if tip is None:
                continue
            d = float(np.linalg.norm(tip - corner))
            if best is None or d < best[0] - 1e-6:
                best = (d, tip, R_f, s_, spin)
    if best is None:
        return {"name": name, "corner": corner.tolist(), "ok": False}
    d, tip, R_f, s_, spin = best
    Qd, _, _, _ = pl.chk.solve_cartesian_path(via + [pl.pose_for_tip(tip, R_f)], pl.q_ready, pl.tcp, pl.ready)
    keep = [0]
    for k in range(1, len(Qd)):   # decimate the search path: joint steps <= 2 deg
        if np.degrees(np.abs(Qd[k] - Qd[keep[-1]])).max() > 2.0 or k == len(Qd) - 1:
            keep.append(k)
    q_ref = Qd[-1]
    search_path = [list(map(float, Qd[k])) for k in keep]
    tilt = math.degrees(math.acos(np.clip(-R_f[:, 2] @ np.array([0, 0, 1.0]), -1, 1)))
    return {"name": name, "corner": corner.tolist(), "ok": True, "tip": tip.tolist(), "R_f": R_f.tolist(),
            "route": [list(map(float, v)) for v in via], "q_ref": list(map(float, q_ref)), "search_path": search_path,
            "pose": pl.pose_for_tip(tip, R_f), "distance_mm": d, "tool_tilt_from_down_deg": tilt,
            "spin_deg": spin, "aim_fraction": s_}


def _sample_leg(poses, speed, ang_speed, dt):
    """Rest-to-rest timing along a pose polyline; returns (t, list of poses)."""
    t_all, P_all = [0.0], [list(poses[0])]
    for a, b in zip(poses[:-1], poses[1:]):
        dp = float(np.linalg.norm(np.subtract(b[:3], a[:3])))
        Ra, Rb = rpy_to_matrix(*a[3:6]), rpy_to_matrix(*b[3:6])
        ang = math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))
        L = max(dp, ang * speed / ang_speed, 1e-6)    # equivalent length so rotation is also slow
        ts, ss = rest_to_rest([0.0], [L], speed, speed * 2.0, dt)
        t_all += list(t_all[-1] + ts[1:])
        P_all += [interpolate_pose(a, b, float(x[0]) / L) for x in ss[1:]]
    return np.asarray(t_all), P_all


def _assemble_tour(pl, targets):
    """Legs through `targets` in the given order, starting and ending at the ready pose."""
    R_ready = rpy_to_matrix(*pl.ready[3:6]) @ rpy_to_matrix(*pl.tcp[3:6]).T
    ready_stop = {"name": "ready", "q_ref": list(pl.q_ready), "search_path": [list(pl.q_ready)],
                  "tip": list(pl.tip_of_pose(pl.ready)), "R_f": R_ready.tolist()}
    stops = targets + [ready_stop]
    center = np.asarray(ready_stop["tip"])
    legs, kinds, visited = [], [], []
    prev = ready_stop
    q_prev = list(pl.q_ready)
    for r in stops:
        q_b = np.asarray(r["q_ref"])
        ta, Ra, tb, Rb = np.asarray(prev["tip"]), np.asarray(prev["R_f"]), np.asarray(r["tip"]), np.asarray(r["R_f"])
        cart = [("direct, straight tip line", [(ta, Ra), (tb, Rb)])]
        for f in (0.25, 0.4, 0.6):  # retreat from A, turn in open space, approach B
            cart.append((f"retreat {f:.0%}-turn-approach",
                         [(ta, Ra), (ta + f * (center - ta), Ra), (tb + f * (center - tb), Rb), (tb, Rb)]))
        cart.append(("via the ready pose", [(ta, Ra), (center, R_ready), (tb, Rb)]))
        done = False
        for kind, wps in cart:
            Q = pl.cart_leg_joints(wps, q_prev, q_prev, q_b)
            if Q is not None:
                legs.append(Q)
                kinds.append(kind)
                done = True
                break
        if not done:  # last resort: back along A's search path, out along B's (safe by construction)
            nodes = list(reversed(prev["search_path"])) + r["search_path"][1:]
            if pl.joint_polyline_ok(nodes):
                legs.append(nodes)
                kinds.append("out along its search path from ready" if prev["name"] == "ready" else
                             "back along the search path to ready" if r["name"] == "ready" else
                             "retrace search paths via ready")
                done = True
        if not done:
            continue
        visited.append(r["name"])
        q_prev = legs[-1][-1]
        prev = r
    return legs, kinds, visited


def make_corners(args) -> int:
    from multiprocessing import get_context

    pl = Planner(args.config, min_sigma=args.min_sigma)
    w = pl.walls
    set_dir = find_or_create_set(pl.cfg_path)
    ready_deg = np.degrees(pl.q_ready).round(4).tolist()
    # Visit order: around the cell's vertical edges (by azimuth about the base), bottom/top alternating.
    edges = sorted([(xk, yk) for xk in ("x_min", "x_max") for yk in ("y_min", "y_max")],
                   key=lambda e: math.atan2(w[e[1]], w[e[0]]) % (2 * math.pi) if w[e[0]] < 0 else math.atan2(w[e[1]], w[e[0]]))
    def cname(xk, yk, zk):
        return f"corner_{xk.replace('_', '')}_{yk.replace('_', '')}_{zk.replace('_', '')}"

    order_edges = []
    for i, (xk, yk) in enumerate(edges):
        for zk in (("z_min", "z_max") if i % 2 == 0 else ("z_max", "z_min")):
            order_edges.append((cname(xk, yk, zk), [w[xk], w[yk], w[zk]]))
    if args.order == "rings":  # bottom ring around the cell, one climb, top ring back: one big wrist flip
        order = [(cname(xk, yk, "z_min"), [w[xk], w[yk], w["z_min"]]) for xk, yk in edges]
        order += [(cname(xk, yk, "z_max"), [w[xk], w[yk], w["z_max"]]) for xk, yk in reversed(edges)]
    else:                      # up and down each vertical edge in turn
        order = []
        for i, (xk, yk) in enumerate(edges):
            for zk in (("z_min", "z_max") if i % 2 == 0 else ("z_max", "z_min")):
                order.append((cname(xk, yk, zk), [w[xk], w[yk], w[zk]]))
    t0 = time.time()
    with get_context("fork").Pool(min(8, len(order))) as pool:
        results = pool.map(_corner_job, [(str(pl.cfg_path), n, c, args.min_sigma, args.max_tilt) for n, c in order])
    print(f"corner targets in {time.time() - t0:.0f} s (tip = gripper box far-face center, flange frame "
          f"{pl.tip_local.round(1).tolist()})")
    report = []
    for r in results:
        if not r["ok"]:
            print(f"{r['name']}: no reachable orientation found")
            report.append(r)
            continue
        gap = np.abs(np.subtract(r["tip"], r["corner"]))
        print(f"{r['name']}: tip {np.round(r['tip'], 1).tolist()} | gap per axis {gap.round(1).tolist()} mm "
              f"(distance {r['distance_mm']:.0f}) | tool tilted {r['tool_tilt_from_down_deg']:.0f} deg from down, "
              f"spin {r['spin_deg']:.0f}")
        report.append({k: (np.round(v, 2).tolist() if isinstance(v, list) else v) for k, v in r.items() if k != "R_f"}
                      | {"gap_per_axis_mm": gap.round(1).tolist()})
    targets = [r for r in results if r["ok"]]

    # Tour in JOINT space: the robot executes exactly the checked joint path (a Cartesian leg leaves the arm's
    # posture to the firmware IK). ready -> corner -> corner -> ... -> ready: a direct joint move where it
    # passes every check, else through the ready configuration, else back along A's search path and out
    # along B's (both safe by construction).
    # Try several visiting orders (the corner targets are computed once) and keep the one with the most
    # direct / retreat-turn-approach legs.
    by_name = {r["name"]: r for r in targets}
    ring_b = [r["name"] for r in targets if r["name"].endswith("zmin")]
    ring_t = [r["name"] for r in targets if r["name"].endswith("zmax")]
    candidates = {}
    if args.order == "rings":
        for k in range(len(ring_b)):
            for direction in (1, -1):
                rb = (ring_b[k:] + ring_b[:k])[::direction]
                tops = [n.replace("zmin", "zmax") for n in rb][::-1]
                candidates[f"rings start {k} dir {direction:+d}"] = [by_name[n] for n in rb + tops if n in by_name]
    candidates["edges"] = targets_edges = [by_name[n] for n, _ in order_edges if n in by_name]
    best = None
    for label, seq in candidates.items():
        legs, kinds, visited = _assemble_tour(pl, seq)
        score = (len([v for v in visited if v != "ready"]), "ready" in visited[-1:],
                 sum(k != "retrace search paths via ready" for k in kinds))  # first/last legs from/to ready are fine
        print(f"   order {label:22s}: {score[0]} corners, {score[2]} of {len(kinds)} legs direct/retreat")
        if best is None or score > best[0]:
            best = (score, label, legs, kinds, visited)
    _, label, legs, kinds, visited = best
    print(f"   chosen: {label}")
    for v, k in zip(visited, kinds):
        print(f"   leg to {v}: {k}")
    # timing: one smooth rest-to-rest profile per leg along its joint polyline; the tip moves at <= speed and
    # every joint at <= rot_speed
    t_all, Q_all = [0.0], [np.asarray(pl.q_ready)]
    omega = math.radians(args.rot_speed)
    for k, nodes in enumerate(legs):
        dense = [np.asarray(nodes[0])]
        for qa, qb in zip(nodes[:-1], nodes[1:]):
            Q, _ = pl.chk.densify_joint_path(np.asarray([qb]), np.asarray(qa))
            dense += list(Q[1:])
        dense = np.asarray(dense)
        tips = pl.tip_points(dense)
        seg = np.maximum(np.linalg.norm(np.diff(tips, axis=0), axis=1),
                         np.abs(np.diff(dense, axis=0)).max(axis=1) * args.speed / omega)
        cum = np.r_[0.0, np.cumsum(seg)]
        if cum[-1] < 1e-6:
            continue
        ts, ss = rest_to_rest([0.0], [cum[-1]], args.speed, args.speed * 2.0, args.dt)
        for x in ss[1:, 0]:
            Q_all.append(np.array([np.interp(x, cum, dense[:, j]) for j in range(7)]))
        t_all += list(t_all[-1] + ts[1:])
        if visited[k] != "ready":  # hold at the corner
            n_hold = int(round(args.hold / args.dt))
            t_all += [t_all[-1] + args.dt * (i + 1) for i in range(n_hold)]
            Q_all += [Q_all[-1]] * n_hold
    t = np.asarray(t_all)
    Qt = np.asarray(Q_all)
    res = validate_joint_trajectory(pl.cfg, list(t), Qt.tolist(), None, "servo", q_start=pl.q_ready)
    corners_done = [n for n in visited if n != "ready"]
    print(f"corner_tour: {len(corners_done)} of {len(order)} corners, "
          f"{'returns' if visited and visited[-1] == 'ready' else 'does NOT return'} to the ready pose, "
          f"{t[-1]:.0f} s, legs: {', '.join(kinds)} | {res.message[:80]}")
    if not res.ok:
        print("NOT WRITTEN")
        return 2
    frames = {"corner_tour": pd.DataFrame(np.column_stack([t, Qt]), columns=["t"] + [f"q{i}" for i in range(1, 8)])}
    out = set_dir / "tests" / "workspace_corners.csv"
    write_bank(out, frames, {"corner_tour": ready_deg}, ready_deg,
               {"report": report, "format": "joint (rad); play with playback --control servo",
                "speed_mm_s": args.speed, "joint_speed_deg_s": args.rot_speed, "config": str(pl.cfg_path),
                "tip_flange_frame_mm": pl.tip_local.round(2).tolist(), "legs": dict(zip(visited, kinds)),
                "safety_check": res.message})
    print(f"wrote {out} (joint trajectory) and {out.with_suffix('.fit.json')}")
    print(f"play: python -m src.xarm_pipeline.playback --trajectory '{out}#corner_tour' --control servo")
    return 0


def make_dynamic(args) -> int:
    pl = Planner(args.config)
    set_dir = find_or_create_set(pl.cfg_path)
    w = pl.walls
    center = np.array(args.center if args.center else [250.0, (w["y_min"] + w["y_max"]) / 2, 450.0])
    ok, q_c, slack = pl.leg([pl.pose_at(center)], pl.q_ready, pl.ready)
    if not ok:
        raise SystemExit(f"dynamic-test center {center.tolist()} is not reachable/safe (slack {slack:.1f}); pass --center")
    from .runtime import predict_monitor_margin

    frames, approach, report = {}, {}, []
    base_center = center.copy()
    for axis, an in enumerate("xyz"):
        length = args.lengths[axis]
        # Find a center where every version passes the safety check AND the runtime monitor's
        # speed-dependent prediction (otherwise the monitor stops the fast versions mid-stroke).
        chosen = None
        for dx, dz in [(0, 0), (25, 0), (50, 0), (0, 50), (25, 50), (50, 50), (-25, 0), (75, 0), (0, 100)]:
            cand = base_center + np.array([dx, 0.0, dz])
            ok_c, q_cand, _ = pl.leg([pl.pose_at(cand)], pl.q_ready, pl.ready)
            if not ok_c:
                continue
            try:
                v100, a100, diag = pl.stroke_limits(cand, axis, length)
            except Exception:
                continue
            versions = []
            for pct in args.percent:
                v, a = v100 * pct / 100, a100 * pct / 100
                e = np.zeros(3)
                e[axis] = length / 2
                seq = [cand, cand + e] + [cand - e, cand + e] * args.cycles + [cand]
                t_all, P_all = [0.0], [cand]
                for i in range(1, len(seq)):
                    ts, ps = rest_to_rest(seq[i - 1], seq[i], v, a, args.dt)
                    t_all += list(t_all[-1] + ts[1:])
                    P_all += list(ps[1:])
                    t_all.append(t_all[-1] + args.dwell)
                    P_all.append(P_all[-1])
                t, P = np.asarray(t_all), np.asarray(P_all)
                Q, _, _ = pl.joint_profile(t, P, q_cand)
                pm, _, what = predict_monitor_margin(pl.chk.wc, pl.cfg, t, Q)
                versions.append((pct, v, a, t, P, pm, what))
            if all(vv[5] >= 25.0 for vv in versions):  # 25 mm spare for tracking lag on the real arm
                chosen = (cand, q_cand, v100, a100, diag, versions)
                break
        if chosen is None:
            print(f"   {an}-stroke: no center found where the runtime monitor would not stop the fastest version")
            continue
        center, q_c, v100, a100, diag, versions = chosen
        print(f"   {an}-stroke center {center.round(1).tolist()} (monitor-safe at all speeds)")
        for pct, v, a, t, P, pm, what in versions:
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
                    "center_mm": center.round(1).tolist(), "predicted_monitor_margin_mm": round(pm, 1),
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
               {"report": report, "stroke_mm": dict(zip("xyz", args.lengths)), "config": str(pl.cfg_path),
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
    c.add_argument("--rot-speed", type=float, default=15.0, help="max joint speed during the tour, deg/s")
    c.add_argument("--order", choices=["rings", "edges"], default="rings",
                   help="rings: bottom ring, then top ring back (one wrist flip); edges: up/down each vertical edge")
    c.add_argument("--max-tilt", type=float, default=180.0,
                   help="max tool tilt from straight down, deg (smaller = similar wrist postures, more direct legs)")
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
