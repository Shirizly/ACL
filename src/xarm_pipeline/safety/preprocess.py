"""Layer 1: offline trajectory check against the workcell walls.

* Joint trajectories: FK of every sample plus densely interpolated joint-space
  segments (the xArm moves joint-linearly between joint targets).  The density
  is chosen so that no point of the arm moves more than
  ``interp_resolution_mm`` between two checked configurations (bound:
  sum|dq_i| * max_reach), and half of that resolution is added to the
  required clearance, so a crossing cannot hide between checked samples.
* Cartesian trajectories: the joint configuration is chosen by the firmware's
  IK, which (fw v2.4.0) is seeded from the robot's *current* joints, so
  ``get_inverse_kinematics`` cannot be seeded with the previous sample.  We
  therefore predict the joints with a local damped-least-squares IK run as a
  continuation along the densely interpolated Cartesian path, starting from
  the robot's actual joints (or a configurable seed offline).  The prediction
  matched the simulator's joint path within ~5 mm on every link (measured);
  ``ik_model_tolerance_mm`` is added to the required clearance to cover that.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .config import SafetyConfig, load_safety_config
from .geometry import (
    END_EFFECTOR_CAPSULES,
    MOVING_CAPSULES,
    POSE_DETERMINED_CAPSULES,
    WALL_NAMES,
    Violation,
    Workcell,
)
from .kinematics import (
    IKFailure,
    fk_pose,
    ik_step_to,
    matrix_to_rpy,
    rpy_to_matrix,
)

# Joint configuration the simulator reaches with set_position(300,0,300,180,0,0)
# from the all-zero home. Used as the offline IK seed for Cartesian CSVs when
# no robot is connected (pass --seed-joints-deg or --ip to override).
DEFAULT_CARTESIAN_SEED_DEG = (0.0, -28.17, 0.0, 23.24, 0.0, 51.41, 0.0)


class SafetyRejected(RuntimeError):
    """Raised by playback integrations when a trajectory is rejected."""

    def __init__(self, message: str, result: "ValidationResult") -> None:
        super().__init__(message)
        self.result = result


@dataclass
class ValidationResult:
    ok: bool
    action: str  # "pass" | "reject" | "clip"
    kind: str
    message: str
    violation: dict | None = None
    sanitized_pose: list[list[float]] | None = None
    report: dict = field(default_factory=dict)
    # Cartesian, when accepted: the joint configuration predicted at each sample of the poses to execute
    # (playback's per-command check reuses it instead of solving IK again).
    validated_q: list | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _axis_angle(R: np.ndarray) -> np.ndarray:
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = math.acos(cos_a)
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        return ang * axis
    return ang * v / (2.0 * s)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def interpolate_pose(pa, pb, s: float) -> list[float]:
    """Linear position + slerp orientation. Returns rpy continuous with pb's rpy."""
    pa = np.asarray(pa, dtype=float)
    pb = np.asarray(pb, dtype=float)
    Ra = rpy_to_matrix(*pa[3:6])
    Rb = rpy_to_matrix(*pb[3:6])
    w = _axis_angle(Ra.T @ Rb)
    R = Ra @ _exp_so3(s * w)
    pos = pa[:3] + s * (pb[:3] - pa[:3])
    rpy = matrix_to_rpy(R)
    return [*pos.tolist(), *rpy]


def _pose_step(pa, pb) -> tuple[float, float]:
    dp = float(np.linalg.norm(np.asarray(pb[:3]) - np.asarray(pa[:3])))
    Ra = rpy_to_matrix(*pa[3:6])
    Rb = rpy_to_matrix(*pb[3:6])
    da = float(np.linalg.norm(_axis_angle(Ra.T @ Rb)))
    return dp, math.degrees(da)


def _worst_text(info: dict) -> str:
    """'; worst: ...' summary of a _min_clearance_info dict (the first violation can be far milder)."""
    return (f"; worst: link '{info['link']}' vs {info['wall']} clearance {info['min_clearance_mm']:.1f} mm "
            f"at sample {info['sample_index']}")


def _violation_dict(v: Violation, seg_index: int, t: list[float] | None) -> dict:
    d = asdict(v)
    d["deficit_mm"] = v.deficit_mm
    d["dense_index"] = d.pop("sample_index")
    d["sample_index"] = int(seg_index)
    if seg_index < 0:
        d["segment"] = "approach (current robot state -> first sample)"
    elif seg_index == 0:
        d["segment"] = "at first sample"
    else:
        d["segment"] = f"between sample {seg_index - 1} and sample {seg_index}"
    if t is not None and 0 <= seg_index < len(t):
        d["t"] = float(t[seg_index])
    d["description"] = v.describe()
    return d


class SafetyChecker:
    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg
        self.wc = Workcell(cfg)
        self.reach_mm = self._reach_bound()

    def _reach_bound(self) -> float:
        # Upper bound on distance from ANY joint axis to ANY arm point.
        from .kinematics import XARM7_NOMINAL_DH

        links = sum(abs(d) + abs(a) for _, d, _, a in XARM7_NOMINAL_DH)
        tool = float(np.linalg.norm(self.cfg.tool_offset_mm))
        corners = self.cfg.tool_box_corners_mm()
        if corners is not None:
            tool = max(tool, float(np.linalg.norm(corners, axis=1).max()))
        return links + tool + max(self.cfg.radii_mm.values())

    @property
    def interp_tol_mm(self) -> float:
        return 0.5 * self.cfg.preprocess.interp_resolution_mm

    def required_joint_mm(self) -> float:
        return self.cfg.safety_margin_mm + self.interp_tol_mm

    def required_cartesian_mm(self) -> np.ndarray:
        """Per moving capsule: IK-model tolerance only where redundancy matters."""
        base = self.required_joint_mm()
        tol = self.cfg.preprocess.ik_model_tolerance_mm
        return np.array([
            base if name in POSE_DETERMINED_CAPSULES else base + tol
            for name, _, _ in MOVING_CAPSULES
        ])

    # ---------------- joint space ----------------
    def densify_joint_path(self, qs: np.ndarray, q_start: np.ndarray | None = None):
        """Return (dense Q, segment index per dense sample)."""
        qs = np.asarray(qs, dtype=float)
        res = self.cfg.preprocess.interp_resolution_mm
        out = []
        seg = []
        prev = None
        if q_start is not None:
            prev = np.asarray(q_start, dtype=float)
            out.append(prev)
            seg.append(-1)
        for k, q in enumerate(qs):
            if prev is not None:
                bound = float(np.sum(np.abs(q - prev))) * self.reach_mm
                n = max(1, int(math.ceil(bound / res)))
                s = np.arange(1, n + 1)[:, None] / n
                out.extend(prev + s * (q - prev))
                seg.extend([k] * n)
            else:
                out.append(q)
                seg.append(k)
            prev = q
        return np.asarray(out), np.asarray(seg)

    def check_configs(self, Q: np.ndarray, required: float):
        return self.wc.first_violation(Q, required, MOVING_CAPSULES)

    def _min_clearance_info(self, Q: np.ndarray, seg: np.ndarray) -> dict:
        C, _ = self.wc.clearance_for_q(Q, MOVING_CAPSULES)
        flat = C.reshape(C.shape[0], -1)
        n = int(np.argmin(flat.min(axis=1)))
        k, w = np.unravel_index(int(np.argmin(C[n])), C[n].shape)
        return {
            "min_clearance_mm": float(C[n, k, w]),
            "link": MOVING_CAPSULES[k][0],
            "wall": WALL_NAMES[w],
            "sample_index": int(seg[n]),
        }

    # ---------------- Cartesian ----------------
    def solve_cartesian_path(self, poses, q_start, tcp_offset=None, start_pose=None):
        """Continuation IK along the densely interpolated Cartesian path.

        Returns (dense Q, dense poses, seg index, final q per original sample).
        Raises IKFailure (with .sample_index) if a pose is unreachable.
        """
        res_mm = self.cfg.preprocess.interp_resolution_mm
        res_deg = self.cfg.preprocess.interp_resolution_deg
        q = np.asarray(q_start, dtype=float)
        if start_pose is None:
            start_pose = fk_pose(q, tcp_offset=tcp_offset)
        Q = [q]
        P = [list(start_pose)]
        seg = [-1]
        q_at_sample = []
        prev_pose = list(start_pose)
        for k, pose in enumerate(poses):
            dp, da = _pose_step(prev_pose, pose)
            n = max(1, int(math.ceil(max(dp / res_mm, da / res_deg))))
            for i in range(1, n + 1):
                target = interpolate_pose(prev_pose, pose, i / n) if i < n else list(pose)
                try:
                    q = self._solve_refined(P[-1], target, q, tcp_offset, Q, P, seg, k, depth=0)
                except IKFailure as e:
                    e.sample_index = k  # type: ignore[attr-defined]
                    raise
            q_at_sample.append(q)
            prev_pose = list(pose)
        return np.asarray(Q), P, np.asarray(seg), q_at_sample

    def _solve_refined(self, pose_a, pose_b, q_a, tcp_offset, Q, P, seg, k, depth):
        """Solve pose_b from q_a; if any arm point jumps > resolution, bisect."""
        q_b = ik_step_to(pose_b, q_a, tcp_offset=tcp_offset)
        jump = float(np.max(np.abs(
            self.wc.points_world(q_b[None])[0] - self.wc.points_world(np.asarray(q_a)[None])[0]
        )))
        if jump > self.cfg.preprocess.interp_resolution_mm and depth < 6:
            mid = interpolate_pose(pose_a, pose_b, 0.5)
            q_m = self._solve_refined(pose_a, mid, q_a, tcp_offset, Q, P, seg, k, depth + 1)
            return self._solve_refined(mid, pose_b, q_m, tcp_offset, Q, P, seg, k, depth + 1)
        Q.append(q_b)
        P.append(list(pose_b))
        seg.append(k)
        return q_b

    def ee_shift_world(self, q, required) -> np.ndarray | None:
        """World translation that pushes the end-effector capsules inside the
        box shrunk by ``required`` (float, or dict capsule -> mm); None if
        impossible (box too small)."""
        Pw = self.wc.points_world(np.asarray(q)[None])[0]
        from .geometry import CAPSULES

        need_lo = np.full(3, -np.inf)
        need_hi = np.full(3, -np.inf)
        for name, i, j in CAPSULES:
            if name not in END_EFFECTOR_CAPSULES:
                continue
            r = self.cfg.radii_mm[name] + (required[name] if isinstance(required, dict) else required)
            for p in (Pw[i], Pw[j]):
                need_lo = np.maximum(need_lo, (self.wc.lo + r) - p)
                need_hi = np.maximum(need_hi, p - (self.wc.hi - r))
        shift = np.zeros(3)
        for ax in range(3):
            lo_v, hi_v = need_lo[ax], need_hi[ax]
            if lo_v > 0 and hi_v > 0:
                return None
            if lo_v > 0:
                shift[ax] = lo_v
            elif hi_v > 0:
                shift[ax] = -hi_v
        return shift


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _config_summary(cfg: SafetyConfig) -> dict:
    return {
        "config_path": cfg.source_path,
        "box_D_W_H_mm": [cfg.depth_D_mm, cfg.width_W_mm, cfg.height_H_mm],
        "base_in_world_xyz_mm_yaw_deg": [cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm, cfg.base_yaw_deg],
        "safety_margin_mm": cfg.safety_margin_mm,
        "tool_offset_mm": list(cfg.tool_offset_mm),
        "radii_mm": cfg.radii_mm,
        "interp_resolution_mm": cfg.preprocess.interp_resolution_mm,
        "ik_model_tolerance_mm": cfg.preprocess.ik_model_tolerance_mm,
    }


def validate_joint_trajectory(
    cfg: SafetyConfig,
    t: list[float],
    q: list[list[float]],
    dq: list[list[float]] | None = None,
    control: str = "position",
    q_start=None,
    trajectory_name: str = "",
) -> ValidationResult:
    """Validate a joint trajectory (radians). Joint trajectories can only be
    rejected, never clipped (there is no TCP target to project)."""
    chk = SafetyChecker(cfg)
    required = chk.required_joint_mm()
    qs = np.asarray(q, dtype=float)
    report = {
        "trajectory": trajectory_name,
        "kind": "joint",
        "control": control,
        "policy": cfg.preprocess.policy,
        "required_clearance_mm": required,
        "config": _config_summary(cfg),
        "notes": [],
    }
    if cfg.preprocess.policy == "clip":
        report["notes"].append("clip policy is not applicable to joint trajectories; using reject")

    paths = [("q columns", qs, q_start)]
    if control == "velocity" and dq is not None:
        # Integrate the velocity commands (each held until the next sample).
        q0 = np.asarray(q_start if q_start is not None else qs[0], dtype=float)
        dqs = np.asarray(dq, dtype=float)
        tt = np.asarray(t, dtype=float)
        dts = np.diff(tt, append=tt[-1])
        integ = q0 + np.vstack([np.zeros(7), np.cumsum(dqs[:-1] * dts[:-1, None], axis=0)])
        paths.append(("integrated dq columns", integ, None))
        report["notes"].append("velocity mode: checked both q columns and the integrated dq path")

    n_dense_total = 0
    for label, path, qs0 in paths:
        Qd, seg = chk.densify_joint_path(path, qs0)
        n_dense_total += len(Qd)
        v, _ = chk.check_configs(Qd, required)
        info = chk._min_clearance_info(Qd, seg)
        report.setdefault("min_clearance", {})[label] = info
        bc = _base_column_violation(chk, Qd, seg, t) if v is None else None
        if bc is not None:
            bc["path"] = label
            report.update(result="rejected", violation=bc, n_samples=len(qs), n_dense=n_dense_total)
            return ValidationResult(False, "reject", "joint", f"SAFETY REJECT ({label}): sample "
                                    f"{bc['sample_index']}: {bc['description']}", bc, None, report)
        if v is not None:
            vd = _violation_dict(v, int(seg[v.sample_index]), t)
            vd["path"] = label
            report.update(result="rejected", violation=vd, n_samples=len(qs), n_dense=n_dense_total)
            msg = (
                f"SAFETY REJECT ({label}): first violation at sample {vd['sample_index']}"
                + (f" (t={vd['t']:.3f}s)" if "t" in vd else "")
                + f", {vd['segment']}: {v.describe()}"
                + _worst_text(info)
            )
            return ValidationResult(False, "reject", "joint", msg, vd, None, report)
    report.update(result="ok", n_samples=len(qs), n_dense=n_dense_total)
    mins = min(r["min_clearance_mm"] for r in report["min_clearance"].values())
    return ValidationResult(True, "pass", "joint", f"SAFETY OK: min clearance {mins:.1f} mm "
                            f"(required {required:.1f} mm), {n_dense_total} configs checked", None, None, report)


def _check_solved_path(chk, Qd, seg, required):
    v, _ = chk.check_configs(Qd, required)
    return v


def _base_column_violation(chk, Qd, seg, t=None) -> dict | None:
    """End effector closer to the robot's own base column than safety_margin_mm (first such config)."""
    sl = chk.wc.base_column_slack(chk.wc.points_world(Qd))
    bad = np.flatnonzero(sl < 0)
    if bad.size == 0:
        return None
    n = int(bad[0])
    k = int(seg[n])
    d = {"sample_index": k, "link": "end effector", "wall": "robot base column",
         "clearance_mm": float(sl[n] + chk.cfg.safety_margin_mm), "required_mm": chk.cfg.safety_margin_mm,
         "worst_clearance_mm": float(sl.min() + chk.cfg.safety_margin_mm),
         "description": (f"gripper/wrist {sl[n] + chk.cfg.safety_margin_mm:.1f} mm from the robot's own base column "
                         f"(required >= {chk.cfg.safety_margin_mm:.1f} mm; worst {sl.min() + chk.cfg.safety_margin_mm:.1f} mm)")}
    if t is not None and 0 <= k < len(t):
        d["t"] = float(t[k])
    return d


def validate_cartesian_trajectory(
    cfg: SafetyConfig,
    t: list[float],
    pose: list[list[float]],
    q_start,
    tcp_offset=None,
    twist: list[list[float]] | None = None,
    control: str = "servo",
    include_approach: bool = True,
    trajectory_name: str = "",
    policy: str | None = None,
) -> ValidationResult:
    """Validate (and optionally clip) a Cartesian trajectory.

    ``pose`` rows: [x,y,z mm, roll,pitch,yaw rad] exactly as they will be sent
    (after orientation preparation).  ``q_start``: joints [rad] the robot is at
    when the trajectory starts.
    """
    policy = policy or cfg.preprocess.policy
    chk = SafetyChecker(cfg)
    required = chk.required_cartesian_mm()
    q_start = np.asarray(q_start, dtype=float)
    report = {
        "trajectory": trajectory_name,
        "kind": "cartesian",
        "control": control,
        "policy": policy,
        "required_clearance_mm": dict(zip([c[0] for c in MOVING_CAPSULES], required.tolist())),
        "q_start_deg": np.degrees(q_start).round(3).tolist(),
        "config": _config_summary(cfg),
        "notes": [
            "joint configurations predicted by local continuation IK seeded from q_start",
        ],
    }
    if control == "velocity" and twist is not None:
        # Integrate linear velocity; orientation taken from the pose columns.
        tt = np.asarray(t, dtype=float)
        dts = np.diff(tt, append=tt[-1])
        v = np.asarray(twist, dtype=float)[:, :3]
        p0 = np.asarray(pose[0][:3], dtype=float)
        pos = p0 + np.vstack([np.zeros(3), np.cumsum(v[:-1] * dts[:-1, None], axis=0)])
        integ = [list(pos[k]) + list(pose[k][3:6]) for k in range(len(pose))]
        res = validate_cartesian_trajectory(
            cfg, t, integ, q_start, tcp_offset, None, "velocity-integrated",
            include_approach, trajectory_name, policy="reject",
        )
        res.report["notes"].append("velocity mode: checked integrated twist path (rejection only)")
        if not res.ok:
            return res
        policy = "reject"  # cannot clip velocity commands

    start_pose = fk_pose(q_start, tcp_offset=tcp_offset)
    poses = [list(map(float, p)) for p in pose]
    if not include_approach:
        start_pose = None

    def solve(ps):
        return chk.solve_cartesian_path(ps, q_start, tcp_offset, start_pose)

    try:
        Qd, Pd, seg, q_at = solve(poses)
    except IKFailure as e:
        k = getattr(e, "sample_index", -1)
        report.update(result="rejected", violation={"sample_index": k, "description": f"IK failure: {e}"})
        return ValidationResult(False, "reject", "cartesian",
                                f"SAFETY REJECT: sample {k} unreachable ({e})", report["violation"], None, report)

    v = _check_solved_path(chk, Qd, seg, required)
    report["min_clearance"] = {"original": chk._min_clearance_info(Qd, seg)}
    bc = _base_column_violation(chk, Qd, seg, t) if v is None else None
    if bc is not None:  # cannot be clipped: moving the target does not move it away from the robot itself
        report.update(result="rejected", violation=bc, n_samples=len(poses), n_dense=len(Qd))
        return ValidationResult(False, "reject", "cartesian",
                                f"SAFETY REJECT: sample {bc['sample_index']}: {bc['description']}", bc, None, report)
    if v is None:
        report.update(result="ok", n_samples=len(poses), n_dense=len(Qd))
        mc = report["min_clearance"]["original"]["min_clearance_mm"]
        return ValidationResult(True, "pass", "cartesian",
                                f"SAFETY OK: min clearance {mc:.1f} mm (required {required.min():.1f}-{required.max():.1f} mm), "
                                f"{len(Qd)} configs checked", None, poses, report, validated_q=list(q_at))

    first = _violation_dict(v, int(seg[v.sample_index]), t)
    if policy != "clip" or first["sample_index"] < 0:
        if policy == "clip":
            report["notes"].append("violation on the approach segment cannot be clipped")
        report.update(result="rejected", violation=first, n_samples=len(poses), n_dense=len(Qd))
        return ValidationResult(False, "reject", "cartesian",
                                f"SAFETY REJECT: first violation at sample {first['sample_index']}"
                                + (f" (t={first['t']:.3f}s)" if "t" in first else "")
                                + f", {first['segment']}: {v.describe()}"
                                + _worst_text(report["min_clearance"]["original"]), first, None, report)

    # ---------------- clip ----------------
    clip_req = {name: float(r) + 0.5 for (name, _, _), r in zip(MOVING_CAPSULES, required)}
    clipped = [list(p) for p in poses]
    shifts = np.zeros(len(clipped))
    q = q_start
    prev_pose = start_pose if start_pose is not None else clipped[0]
    for k in range(len(clipped)):
        for _ in range(6):
            try:
                Qs, _, _, qa = chk.solve_cartesian_path([clipped[k]], q, tcp_offset, prev_pose)
            except IKFailure as e:
                report.update(result="rejected", violation={"sample_index": k, "description": f"IK failure while clipping: {e}"})
                return ValidationResult(False, "reject", "cartesian",
                                        f"SAFETY REJECT: clipping failed at sample {k}: {e}", report["violation"], None, report)
            shift_w = chk.ee_shift_world(qa[-1], clip_req)
            if shift_w is None:
                report.update(result="rejected", violation={"sample_index": k, "description": "box too small for the tool"})
                return ValidationResult(False, "reject", "cartesian",
                                        f"SAFETY REJECT: sample {k}: tool cannot fit inside the box", report["violation"], None, report)
            if np.linalg.norm(shift_w) < 1e-3:
                break
            sb = chk.wc.world_vec_to_base(shift_w)
            clipped[k][0] += float(sb[0])
            clipped[k][1] += float(sb[1])
            clipped[k][2] += float(sb[2])
            shifts[k] += float(np.linalg.norm(sb))
        q = qa[-1]
        prev_pose = clipped[k]

    # Smooth the clip offsets so the clipped path decelerates before the
    # boundary instead of stopping instantly (infinite deceleration).
    if cfg.preprocess.clip_smoothing_s > 0 and len(clipped) > 2:
        dt = float(np.median(np.diff(np.asarray(t, dtype=float)))) if len(t) > 1 else 0.03
        r = max(1, int(round(cfg.preprocess.clip_smoothing_s / max(dt, 1e-3))))
        hard = np.asarray([c[:3] for c in clipped]) - np.asarray([p[:3] for p in poses])
        soft = smooth_shift_upper_envelope(hard, r)
        for k in range(len(clipped)):
            for ax in range(3):
                clipped[k][ax] = poses[k][ax] + float(soft[k, ax])
        shifts = np.linalg.norm(soft, axis=1)
    try:
        Qd2, _, seg2, q_at2 = solve(clipped)
    except IKFailure as e:
        report.update(result="rejected", violation={"description": f"IK failure on clipped path: {e}"})
        return ValidationResult(False, "reject", "cartesian", f"SAFETY REJECT: clipped path unreachable: {e}",
                                report["violation"], None, report)
    v2 = _check_solved_path(chk, Qd2, seg2, required)
    bc2 = _base_column_violation(chk, Qd2, seg2, t) if v2 is None else None
    if bc2 is not None:
        report.update(result="rejected", violation=bc2, n_samples=len(poses), n_dense=len(Qd2))
        return ValidationResult(False, "reject", "cartesian",
                                f"SAFETY REJECT (after clipping): sample {bc2['sample_index']}: {bc2['description']}",
                                bc2, None, report)
    report["original_first_violation"] = first
    report["clipped_samples"] = int(np.count_nonzero(shifts > 1e-3))
    report["max_clip_shift_mm"] = float(shifts.max())
    report["min_clearance"]["clipped"] = chk._min_clearance_info(Qd2, seg2)
    if v2 is not None:
        vd = _violation_dict(v2, int(seg2[v2.sample_index]), t)
        report.update(result="rejected", violation=vd, n_samples=len(poses), n_dense=len(Qd2))
        report["notes"].append("non-end-effector link still violates after clipping the TCP targets")
        return ValidationResult(False, "reject", "cartesian",
                                f"SAFETY REJECT (after clipping {report['clipped_samples']} samples): "
                                f"sample {vd['sample_index']}: {v2.describe()} -- a non-TCP link violates, "
                                "clipping cannot fix it", vd, None, report)
    report.update(result="clipped", violation=None, n_samples=len(poses), n_dense=len(Qd2))
    return ValidationResult(True, "clip", "cartesian",
                            f"SAFETY CLIP: {report['clipped_samples']} of {len(poses)} samples moved "
                            f"(max {shifts.max():.1f} mm); first original violation: {v.describe()} "
                            f"at sample {first['sample_index']}" + _worst_text(report["min_clearance"]["original"]),
                            None, clipped, report, validated_q=list(q_at2))


def smooth_shift_upper_envelope(hard: np.ndarray, r: int) -> np.ndarray:
    """Smooth per-sample clip offsets (N,3) without ever shrinking them.

    For the positive and negative part of each axis separately: dilate (running
    max, radius r) and then apply two box filters (half-width r//2 each, total
    support <= r). Every averaging window around sample k lies inside the
    dilation window of k, so |soft_k| >= |hard_k| component-wise, and the result
    has continuous velocity (piecewise-quadratic ramps of duration ~r samples).
    """
    hard = np.asarray(hard, dtype=float)
    n = hard.shape[0]
    out = np.zeros_like(hard)
    h = max(r // 2, 1)

    def dilate(x):
        return np.array([x[max(0, k - r): k + r + 1].max() for k in range(n)])

    def box(x):
        pad = np.pad(x, h, mode="edge")
        c = np.cumsum(np.insert(pad, 0, 0.0))
        return (c[2 * h + 1:] - c[: -2 * h - 1]) / (2 * h + 1)

    for ax in range(3):
        for sign in (1.0, -1.0):
            part = np.clip(sign * hard[:, ax], 0.0, None)
            if not part.any():
                continue
            sm = box(box(dilate(part)))
            out[:, ax] += sign * np.maximum(sm, part)
    return out


# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------
def write_report(result: ValidationResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(result.report)
    data["message"] = result.message
    data["action"] = result.action
    path.write_text(json.dumps(data, indent=2, default=float), encoding="utf-8")
    return path


def write_sanitized_cartesian(t, poses, path: str | Path, twist=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(poses, columns=["x", "y", "z", "roll", "pitch", "yaw"])
    df.insert(0, "t", t)
    if twist is not None:
        df[["vx", "vy", "vz", "vroll", "vpitch", "vyaw"]] = np.asarray(twist)
    df.to_csv(path, index=False)
    return path


def print_banner(msg: str) -> None:
    line = "!" * 78
    print(f"{line}\n{msg}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Validate a trajectory CSV against the safety walls (config/safety.yaml). "
        "Joint CSVs need no robot; Cartesian CSVs use a local IK seeded from --seed-joints-deg "
        "(offline) or from the robot's current joints (--ip)."
    )
    p.add_argument("--trajectory", required=True)
    p.add_argument("--config", default=None, help="Safety config (default config/safety.yaml)")
    p.add_argument("--kind", choices=["auto", "joint", "cartesian"], default="auto")
    p.add_argument("--control", default=None,
                   help="joint: position|velocity, cartesian: pose|servo|velocity (default position/servo)")
    p.add_argument("--policy", choices=["reject", "clip"], default=None, help="Override preprocess.policy")
    p.add_argument("--ip", default=None, help="Optional robot IP: use its current joints / TCP offset")
    p.add_argument("--seed-joints-deg", default=None, help="Comma separated q1..q7 start joints (deg)")
    p.add_argument("--default-orientation-rpy", default="",
                   help="Fallback roll,pitch,yaw (rad) for Cartesian CSVs without orientation")
    p.add_argument("--no-approach", action="store_true",
                   help="Do not check the move from the start configuration to the first sample")
    p.add_argument("--report", default=None, help="Write JSON report here")
    p.add_argument("--sanitized-out", default=None, help="Write the (clipped) Cartesian CSV here")
    p.add_argument("--check-robot-dh", action="store_true",
                   help="With --ip: compare the robot's DH parameters to the nominal model")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = load_safety_config(args.config)
    if args.policy:
        cfg.preprocess.policy = args.policy
    problems = Workcell(cfg).static_check()
    if problems:
        print("CONFIG ERROR: " + "; ".join(problems))
        return 3

    from ..trajectory_io import read_trajectory_frame

    df_cols = read_trajectory_frame(args.trajectory).columns
    kind = args.kind
    if kind == "auto":
        kind = "joint" if "q1" in df_cols else "cartesian"

    arm = None
    q_start = None
    tcp_offset = list(cfg.firmware_tcp_offset_rad)
    if args.ip:
        from xarm.wrapper import XArmAPI

        from .kinematics import check_dh_against_firmware

        arm = XArmAPI(args.ip, baud_checkset=False, check_joint_limit=False)
        code, qa = arm.get_servo_angle(is_radian=True)
        if code != 0:
            print(f"failed to read joints from robot: code={code}")
            return 4
        q_start = np.asarray(qa[:7], dtype=float)
        tcp_offset = robot_tcp_offset(arm, cfg)
        if args.check_robot_dh:
            diff = check_dh_against_firmware(arm)
            print(f"robot DH vs nominal: max abs diff = {diff:.6g} (mm / rad)")
    if args.seed_joints_deg:
        q_start = np.radians([float(v) for v in args.seed_joints_deg.split(",")])

    if kind == "joint":
        from ..trajectory_io import load_joint_trajectory_csv

        traj = load_joint_trajectory_csv(args.trajectory)
        res = validate_joint_trajectory(cfg, traj.t, traj.q, traj.dq, args.control or "position",
                                        q_start=None if args.no_approach else q_start,
                                        trajectory_name=str(args.trajectory))
    else:
        from ..trajectory_io import load_cartesian_trajectory_csv

        traj = load_cartesian_trajectory_csv(args.trajectory)
        if q_start is None:
            q_start = np.radians(DEFAULT_CARTESIAN_SEED_DEG)
            print(f"note: offline Cartesian check, IK seeded from q={list(DEFAULT_CARTESIAN_SEED_DEG)} deg; "
                  "pass --ip to use the robot's actual joints")
        poses = prepare_cartesian_poses(traj.pose, fk_pose(q_start, tcp_offset=tcp_offset),
                                        args.default_orientation_rpy)
        res = validate_cartesian_trajectory(cfg, traj.t, poses, q_start, tcp_offset, traj.twist,
                                            args.control or "servo", not args.no_approach,
                                            str(args.trajectory))
        if res.sanitized_pose is not None and args.sanitized_out:
            print(f"saved sanitized trajectory: {write_sanitized_cartesian(traj.t, res.sanitized_pose, args.sanitized_out, traj.twist)}")
    if arm is not None:
        arm.disconnect()
    print(res.message)
    if args.report:
        print(f"saved report: {write_report(res, args.report)}")
    return 0 if res.ok else 2


def robot_tcp_offset(arm, cfg: SafetyConfig) -> list[float]:
    """Read the firmware TCP offset [mm, rad]; warn if it differs from config."""
    off = list(arm.tcp_offset)  # degrees for angles unless is_radian
    off_rad = [float(v) for v in off[:3]] + [math.radians(float(v)) for v in off[3:6]]
    if np.max(np.abs(np.asarray(off_rad) - np.asarray(cfg.firmware_tcp_offset_rad))) > 1e-3:
        print(f"WARNING: robot TCP offset {off} differs from config geometry.firmware_tcp_offset "
              f"{list(cfg.firmware_tcp_offset)}; using the robot's value")
    return off_rad


def prepare_cartesian_poses(pose, start_pose, default_orientation_rpy: str = "",
                            max_orientation_step_deg: float = 8.0) -> list[list[float]]:
    """Same orientation preparation as cartesian_playback (fallback + unwrap + clamp)."""
    from ..cartesian_playback import (
        _parse_default_orientation_rpy,
        _prepare_continuous_orientation_trajectory,
        _resolve_missing_orientation,
    )

    fallback = _parse_default_orientation_rpy(default_orientation_rpy) or list(start_pose[3:6])
    p = _resolve_missing_orientation(pose, fallback)
    return _prepare_continuous_orientation_trajectory(p, list(start_pose), math.radians(max_orientation_step_deg))


if __name__ == "__main__":
    sys.exit(main())
