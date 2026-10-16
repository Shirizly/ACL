"""Fit Cartesian trajectories into the safety workcell and write "clean" copies.

Only relative motion is assumed to matter, so each trajectory may be re-placed by a rigid
motion before execution:

    p' = R (p - c) + c'          R_i' = R R_i Rz(spin)
    R  = Rz(yaw) Ry(tilt_y) Rx(tilt_x)        (base frame, about the trajectory's center c)

- translation (c', where the trajectory's center goes) is always searched;
- ``yaw`` rotates the whole trajectory about the vertical axis (relative motion kept exactly);
- ``spin`` turns the tool about its own approach (z) axis, orientations only (off by default);
- ``tilt`` tilts the whole trajectory relative to gravity (off by default).

Search: an analytic end-effector bound narrows the candidates, then a pattern search maximizes
the worst full-arm wall slack (clearance minus required clearance), computed with the same
continuation IK + capsule model as the safety preprocessing. Starting points come from the mean
of placements previously logged for the same workcell/tool configuration.

A trajectory that cannot fit is cut near its middle (preferring a slow sample) and each part is
fitted independently, recursively, down to ``min_part_s``; parts that still cannot fit are
dropped and reported.

Output goes to the preprocessed set matching the current safety config
(``data/preprocessed/NNN_<box>/trajectories/x.csv``, see ``preprocessed.py``) together with
``x.fit.json`` describing every transform applied.

The result is checked offline from a fixed seed configuration. Playback still runs the normal
safety layers from the robot's real start state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..trajectory_io import TRAJ_COLUMN, list_trajectories, select_trajectory, split_trajectory_ref
from .config import DEFAULT_CONFIG_PATH, SafetyConfig, load_safety_config
from .geometry import MOVING_CAPSULES
from .kinematics import IKFailure, matrix_to_rpy, rpy_to_matrix
from .preprocess import DEFAULT_CARTESIAN_SEED_DEG, SafetyChecker

# Bump when the fitting algorithm changes in a way that should redo existing clean copies.
FIT_VERSION = 8  # 3: base column keep-out; 4: runtime-monitor prediction in the score; 5: gripper vs arm links; 6: eased part ends; 7: min_slack_mm; 8: 100 Hz output

POS = ["x", "y", "z"]
RPY = ["roll", "pitch", "yaw"]
GRIPPER_RPY = ["gripper_roll", "gripper_pitch", "gripper_yaw"]
TWIST = ["vx", "vy", "vz", "vroll", "vpitch", "vyaw"]
PARAMS = ("cx", "cy", "cz", "yaw", "spin", "tilt_x", "tilt_y")  # cx..cz: world-frame center, mm; angles deg


@dataclass
class FitConfig:
    yaw_range_deg: float = 180.0
    spin_range_deg: float = 0.0
    tilt_range_deg: float = 0.0
    good_enough_slack_mm: float = 30.0
    # A placement is accepted only with at least this much slack: playback re-checks from the robot's actual
    # joints, which land a little off the prediction, so a 0 mm fit can be refused there.
    min_slack_mm: float = 5.0
    max_evals: int = 150
    split: bool = True
    min_part_s: float = 1.0
    max_depth: int = 3
    default_orientation_rpy_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)
    # Where trajectories should preferably be centered (robot BASE frame, mm): a comfortable spot in
    # front of the arm. Without it, maximizing wall clearance pulls everything over the base column.
    preferred_center_base_mm: tuple[float, float, float] = (300.0, 0.0, 250.0)
    # Once slack >= good_enough_slack_mm, trade: mm of slack per mm away from the preferred center,
    # and per degree of yaw/spin/tilt (prefer the least modification).
    center_weight: float = 0.05
    angle_weight: float = 0.2
    seed_joints_deg: tuple[float, ...] = DEFAULT_CARTESIAN_SEED_DEG


def load_fit_config(path: str | Path | None) -> FitConfig:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")) or {}
    fc = FitConfig()
    for k, v in (raw.get("fit") or {}).items():
        if not hasattr(fc, k):
            raise ValueError(f"fit config: unknown key '{k}'")
        setattr(fc, k, tuple(v) if isinstance(v, list) else v)
    return fc


def _rot(yaw, tilt_x=0.0, tilt_y=0.0) -> np.ndarray:
    return rpy_to_matrix(math.radians(tilt_x), math.radians(tilt_y), math.radians(yaw))


def _rz(deg: float) -> np.ndarray:
    return rpy_to_matrix(0.0, 0.0, math.radians(deg))


@dataclass
class Segment:
    """One Cartesian trajectory (or a part of one) in the robot base frame."""

    t: np.ndarray        # (N,) s, starts at 0
    p: np.ndarray        # (N,3) mm
    R: np.ndarray        # (N,3,3)
    twist: np.ndarray | None  # (N,6) or None
    name: str
    src_rows: tuple[int, int]  # inclusive row range in the source trajectory

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.p.min(axis=0) + self.p.max(axis=0))

    def eased(self, max_acc_mm_s2: float = 2000.0, min_ramp_s: float = 0.3, still_mm_s: float = 20.0) -> "Segment":
        """Same path, re-timed so the motion starts and ends at rest.

        A part cut out of a longer recording starts (and ends) mid-motion; the robot, at rest, cannot jump to
        that speed. The first/last stretch is slowed with a cosine speed ramp (peak acceleration about
        max_acc_mm_s2) while every position stays on the recorded path (linear interpolation between the
        recorded samples); the rest keeps the original timing. Samples keep the original period."""
        t, p = self.t, self.p
        if len(t) < 3:
            return self
        dt = float(np.median(np.diff(t)))
        v0 = float(np.linalg.norm(p[1] - p[0]) / (t[1] - t[0]))
        v1 = float(np.linalg.norm(p[-1] - p[-2]) / (t[-1] - t[-2]))
        r_in = max(min_ramp_s, math.pi * v0 / (2 * max_acc_mm_s2)) if v0 > still_mm_s else 0.0
        r_out = max(min_ramp_s, math.pi * v1 / (2 * max_acc_mm_s2)) if v1 > still_mm_s else 0.0
        if r_in == 0.0 and r_out == 0.0:
            return self
        T = float(t[-1])
        if r_in / 2 + r_out / 2 > T:  # very short part: scale the ramps down to fit
            f = T / (r_in / 2 + r_out / 2)
            r_in, r_out = r_in * f, r_out * f
        Tn = T + r_in / 2 + r_out / 2   # each ramp covers half its length of original time

        def tau(x):  # new time -> original time; d tau/dx ramps 0 -> 1 -> 0 with cosine shapes
            if r_in > 0 and x < r_in:
                return x / 2 - r_in / (2 * math.pi) * math.sin(math.pi * x / r_in)
            if r_out > 0 and x > Tn - r_out:
                y = Tn - x
                return T - (y / 2 - r_out / (2 * math.pi) * math.sin(math.pi * y / r_out))
            return x - r_in / 2

        tn = np.arange(0.0, Tn + 1e-9, dt)
        if tn[-1] < Tn - 1e-9:
            tn = np.r_[tn, Tn]
        ta = np.clip(np.array([tau(x) for x in tn]), 0.0, T)
        pn = np.column_stack([np.interp(ta, t, p[:, k]) for k in range(3)])
        Rn = np.empty((len(tn), 3, 3))
        for k in range(3):
            for m in range(3):
                Rn[:, k, m] = np.interp(ta, t, self.R[:, k, m])
        U, _, Vt = np.linalg.svd(Rn)
        Rn = U @ Vt  # back onto rotations
        tw = None
        if self.twist is not None:
            tw = np.column_stack([np.interp(ta, t, self.twist[:, k]) for k in range(6)])
        return Segment(tn, pn, Rn, tw, self.name, self.src_rows)

    def part(self, a: int, b: int, name: str) -> "Segment":
        tw = None if self.twist is None else self.twist[a : b + 1]
        return Segment(self.t[a : b + 1] - self.t[a], self.p[a : b + 1], self.R[a : b + 1], tw, name,
                       (self.src_rows[0] + a, self.src_rows[0] + b))


class Fitter:
    def __init__(self, cfg: SafetyConfig, fcfg: FitConfig, log_entries: list[dict] | None = None) -> None:
        self.cfg, self.fcfg = cfg, fcfg
        self.chk = SafetyChecker(cfg)
        self.required = self.chk.required_cartesian_mm()
        self.tcp = list(cfg.firmware_tcp_offset_rad)
        self.R_off = rpy_to_matrix(*self.tcp[3:6])
        self.p_off = np.asarray(self.tcp[:3])
        self.q_seed = np.radians(fcfg.seed_joints_deg)
        yaw_b = math.radians(cfg.base_yaw_deg)
        self.R_b = rpy_to_matrix(0.0, 0.0, yaw_b)  # base -> world
        self.b = np.array([cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm])
        self.lo, self.hi = self.chk.wc.lo, self.chk.wc.hi
        names = [c[0] for c in MOVING_CAPSULES]
        req = dict(zip(names, self.required))
        # End-effector points (flange, tool tip, gripper-box corners) and the clearance each needs,
        # for the analytic bound. All are rigid with the flange.
        r_ft = max(cfg.radii_mm["tool"] + req["tool"], cfg.radii_mm["wrist_offset"] + req["wrist_offset"])
        tool_off = np.asarray(cfg.tool_offset_mm, dtype=float)
        corners = cfg.tool_box_corners_mm()
        self.ee_local = np.vstack([np.zeros(3), tool_off] + ([] if corners is None else [corners]))
        self.ee_r = np.array([r_ft, r_ft] + ([] if corners is None else
                                             [cfg.radii_mm["tool_box"] + req["tool_box"]] * 8))
        self.log = log_entries or []
        self.evals = 0

    # ---------------- transforms ----------------
    def bounds(self) -> dict[str, tuple[float, float]]:
        f = self.fcfg
        return {"yaw": (-f.yaw_range_deg, f.yaw_range_deg), "spin": (-f.spin_range_deg, f.spin_range_deg),
                "tilt_x": (-f.tilt_range_deg, f.tilt_range_deg), "tilt_y": (-f.tilt_range_deg, f.tilt_range_deg)}

    def apply(self, seg: Segment, th: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        R = _rot(th["yaw"], th["tilt_x"], th["tilt_y"])
        c_new = self.R_b.T @ (np.array([th["cx"], th["cy"], th["cz"]]) - self.b)  # world center -> base frame
        p = (seg.p - seg.center) @ R.T + c_new
        Rs = R[None] @ seg.R @ _rz(th["spin"])[None]
        tw = None
        if seg.twist is not None:
            tw = np.hstack([seg.twist[:, :3] @ R.T, seg.twist[:, 3:] @ R.T])
        return p, Rs, tw

    def _ee_points_world(self, p: np.ndarray, Rs: np.ndarray) -> np.ndarray:
        """(N,K,3) end-effector points in the world frame (K = len(self.ee_local))."""
        R_f = Rs @ self.R_off.T
        flange = p - R_f @ self.p_off
        pts = flange[:, None, :] + np.einsum("nij,kj->nki", R_f, self.ee_local)
        return pts @ self.R_b.T + self.b

    def _ee_slack(self, ee: np.ndarray) -> float:
        r = self.ee_r[None, :, None]
        return float(np.min(np.minimum(ee - (self.lo + r), (self.hi - r) - ee)))

    def center_interval(self, seg: Segment, th: dict) -> tuple[np.ndarray, np.ndarray]:
        """World-center range per axis that keeps the end-effector points inside (necessary condition)."""
        th0 = dict(th, cx=0.0, cy=0.0, cz=0.0)
        p, Rs, _ = self.apply(seg, th0)
        rel = self._ee_points_world(p, Rs)  # trajectory center placed at the world origin
        r = self.ee_r[None, :, None]
        lo = (self.lo + r - rel).max(axis=(0, 1))
        hi = (self.hi - r - rel).min(axis=(0, 1))
        return lo, hi

    # ---------------- objective ----------------
    def placed_poses(self, seg: Segment, th: dict) -> list[list[float]]:
        """Placed poses [x,y,z,roll,pitch,yaw] with RPY kept continuous for interpolation."""
        p, Rs, _ = self.apply(seg, th)
        poses = [list(p[k]) + list(matrix_to_rpy(Rs[k])) for k in range(len(p))]
        for k in range(1, len(poses)):
            for j in range(3, 6):
                poses[k][j] = poses[k - 1][j] + (poses[k][j] - poses[k - 1][j] + math.pi) % (2 * math.pi) - math.pi
        return poses

    def start_joints(self, seg: Segment, th: dict) -> np.ndarray:
        """Joint configuration at the first pose that the evaluation assumed (reached from the seed).
        Playback must start from it for the fit's full-arm prediction to hold."""
        poses = self.placed_poses(seg, th)
        _, _, _, q_at = self.chk.solve_cartesian_path([poses[0]], self.q_seed, self.tcp, None)
        return q_at[-1]

    def evaluate(self, seg: Segment, th: dict) -> float:
        """Worst slack in mm (>= 0 means it fits); large negative values for IK failures."""
        self.evals += 1
        p, Rs, _ = self.apply(seg, th)
        ee_slack = self._ee_slack(self._ee_points_world(p, Rs))
        if ee_slack < -50.0:
            return ee_slack  # far off: skip the expensive IK
        poses = self.placed_poses(seg, th)
        try:
            _, _, _, q_at = self.chk.solve_cartesian_path([poses[0]], self.q_seed, self.tcp, None)
            Qd, _, _, q_samples = self.chk.solve_cartesian_path(poses, q_at[-1], self.tcp, poses[0])
        except IKFailure as e:
            k = getattr(e, "sample_index", 0)
            return -500.0 - 500.0 * (1.0 - k / max(len(poses), 1))
        C, Pw = self.chk.wc.clearance_for_q(Qd, MOVING_CAPSULES)
        path_slack = min(float(np.min(C - self.required[None, :, None])),
                         float(self.chk.wc.base_column_slack(Pw).min()))  # gripper vs the robot's own base
        if len(seg.t) == len(q_samples) and len(seg.t) > 1:
            # fast motion must also stay clear of the runtime monitor's speed-dependent stop
            from .runtime import predict_monitor_margin

            path_slack = min(path_slack, predict_monitor_margin(self.chk.wc, self.cfg, seg.t, q_samples)[0])
        # The approach playback will use: a joint-space move from the seed (ready) configuration
        # to the start configuration. It must be safe too.
        Qa, _ = self.chk.densify_joint_path(np.asarray([q_at[-1]]), self.q_seed)
        Ca, Pa = self.chk.wc.clearance_for_q(Qa, MOVING_CAPSULES)
        approach_slack = min(float(np.min(Ca)) - self.chk.required_joint_mm(),
                             float(self.chk.wc.base_column_slack(Pa).min()))
        # The approach must be safe, but its clearance is not maximized: it starts at the fixed ready pose,
        # whose own clearance would otherwise cap every placement's score and flatten the search.
        return path_slack if approach_slack >= 0 else min(path_slack, approach_slack)

    # ---------------- search ----------------
    def _seeds(self, seg: Segment, c_pref_w: np.ndarray) -> list[dict]:
        bnd = self.bounds()
        yaws = []
        if self.log:
            ang = np.radians([e["params"]["yaw"] for e in self.log])
            yaws.append(math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())))
        yaws += [0.0] + [s * a for a in range(30, 181, 30) for s in (1, -1)]
        yaws = [y for y in dict.fromkeys(round(y, 3) for y in yaws) if bnd["yaw"][0] <= y <= bnd["yaw"][1]]

        logged_centers, logged_shifts = [], []
        for e in self.log:
            logged_centers.append([e["params"][k] for k in ("cx", "cy", "cz")])
            logged_shifts.append(e["shift_world_mm"])
        seg_center_w = self.R_b @ seg.center + self.b

        seeds = []
        for y in yaws:
            base = {"yaw": y, "spin": 0.0, "tilt_x": 0.0, "tilt_y": 0.0}
            lo, hi = self.center_interval(seg, dict(base, cx=0, cy=0, cz=0))
            ok = bool(np.all(lo <= hi))
            mid = 0.5 * (lo + hi)
            # preferred spot first, then steps from it toward the middle of the allowed range
            cands = [c_pref_w + a * (mid - c_pref_w) for a in (0.0, 0.25, 0.5)]
            if logged_centers:
                cands.append(np.mean(logged_centers, axis=0))
                cands.append(seg_center_w + np.mean(logged_shifts, axis=0))
            cands.append(mid)
            for c in cands:
                c = np.clip(c, np.minimum(lo, hi), np.maximum(lo, hi))
                seeds.append(dict(base, cx=float(c[0]), cy=float(c[1]), cz=float(c[2]), _ee_ok=ok))
        # analytically feasible yaws first, in the order generated (logged mean, 0, +-30, ...)
        seeds.sort(key=lambda s: not s["_ee_ok"])
        return [{k: v for k, v in s.items() if k != "_ee_ok"} for s in seeds]

    def fit(self, seg: Segment) -> tuple[dict | None, float, int]:
        """Best placement found: (params, slack, evals). params None if nothing evaluated."""
        f = self.fcfg
        self.evals = 0
        cache: dict[tuple, float] = {}

        c_pref_w = self.R_b @ np.asarray(f.preferred_center_base_mm, float) + self.b

        def slack(th):
            key = tuple(round(th[k], 1) for k in PARAMS)
            if key not in cache:
                cache[key] = self.evaluate(seg, th)
            return cache[key]

        def score(th):
            # Feasibility first (slack capped at good_enough), then the least awkward placement.
            sl = slack(th)
            dist = float(np.linalg.norm(np.array([th["cx"], th["cy"], th["cz"]]) - c_pref_w))
            angles = abs(th["yaw"]) + abs(th["spin"]) + abs(th["tilt_x"]) + abs(th["tilt_y"])
            return min(sl, f.good_enough_slack_mm) - f.center_weight * dist - f.angle_weight * angles

        # Seeds in priority order (logged mean first): stop at the first one that already fits.
        seeds = self._seeds(seg, c_pref_w)
        scored = []
        for th in seeds[: max(6, f.max_evals // 3)]:
            scored.append((score(th), th))
            if slack(th) >= f.good_enough_slack_mm:
                break
        scored.sort(key=lambda x: -x[0])
        best_s, best = scored[0]

        # Pattern search from the best seed.
        bnd = self.bounds()
        steps = {"cx": 50.0, "cy": 50.0, "cz": 50.0, "yaw": 15.0, "spin": 15.0, "tilt_x": 5.0, "tilt_y": 5.0}

        active = ["cx", "cy", "cz"] + [k for k in ("yaw", "spin", "tilt_x", "tilt_y") if bnd[k][1] > 0]
        min_step = {"cx": 2.0, "cy": 2.0, "cz": 2.0, "yaw": 1.0, "spin": 1.0, "tilt_x": 0.5, "tilt_y": 0.5}
        while self.evals < f.max_evals:
            improved = False
            for k in active:
                for sgn in (1, -1):
                    th = dict(best)
                    th[k] = th[k] + sgn * steps[k]
                    if k in bnd:
                        th[k] = float(np.clip(th[k], *bnd[k]))
                    s = score(th)
                    if s > best_s + 1e-6:
                        best_s, best, improved = s, th, True
                        break
                if improved or self.evals >= f.max_evals:
                    break
            if not improved:
                for k in active:
                    steps[k] *= 0.5
                if all(steps[k] < min_step[k] for k in active):
                    break
        return best, slack(best), self.evals

    # ---------------- splitting ----------------
    def _ee_feasible_any_yaw(self, seg: Segment) -> bool:
        bnd = self.bounds()
        for y in np.arange(bnd["yaw"][0], bnd["yaw"][1] + 1e-9, 15.0) if bnd["yaw"][1] > 0 else [0.0]:
            lo, hi = self.center_interval(seg, {"cx": 0, "cy": 0, "cz": 0, "yaw": y, "spin": 0, "tilt_x": 0, "tilt_y": 0})
            if np.all(lo <= hi):
                return True
        return False

    def _cut_candidates(self, seg: Segment) -> list[int]:
        n = len(seg.t)
        speed = np.r_[0.0, np.linalg.norm(np.diff(seg.p, axis=0), axis=1) / np.maximum(np.diff(seg.t), 1e-9)]
        speed = speed / max(speed.max(), 1e-9)
        ks = [k for k in range(1, n - 1)
              if seg.t[k] >= self.fcfg.min_part_s and seg.t[-1] - seg.t[k] >= self.fcfg.min_part_s]
        # near the middle first, slow samples preferred
        return sorted(ks, key=lambda k: abs(k - (n - 1) / 2) / n + 0.25 * speed[k])

    def fit_or_split(self, seg: Segment, depth: int = 0) -> tuple[list[dict], list[dict]]:
        """Returns (fitted parts, dropped parts). Each fitted part: {segment, params, slack, evals}."""
        t0 = time.time()
        th, slack, evals = self.fit(seg)
        if slack >= self.fcfg.min_slack_mm:
            return [{"segment": seg, "params": th, "slack": slack, "evals": evals, "seconds": time.time() - t0}], []
        info = {"segment": seg, "best_slack": slack, "evals": evals}
        if not self.fcfg.split or depth >= self.fcfg.max_depth or seg.t[-1] < 2 * self.fcfg.min_part_s:
            return [], [info]
        cut = None
        for k in self._cut_candidates(seg)[:200]:
            if self._ee_feasible_any_yaw(seg.part(0, k, "")) and self._ee_feasible_any_yaw(seg.part(k, len(seg.t) - 1, "")):
                cut = k
                break
        if cut is None:
            cut = (len(seg.t) - 1) // 2  # nothing passes the bound; halve and let the halves split further
        fitted, dropped = [], []
        for part in (seg.part(0, cut, seg.name + "a"), seg.part(cut, len(seg.t) - 1, seg.name + "b")):
            f, d = self.fit_or_split(part, depth + 1)
            fitted += f
            dropped += d
        return fitted, dropped


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _segment_from_frame(df: pd.DataFrame, name: str, fcfg: FitConfig) -> tuple[Segment, str]:
    for c in ["t"] + POS:
        if c not in df.columns:
            raise ValueError(f"'{name}': missing column {c}")
    if all(c in df.columns for c in RPY):
        rpy, src = df[RPY].to_numpy(float), "rpy"
    elif all(c in df.columns for c in GRIPPER_RPY):
        rpy, src = df[GRIPPER_RPY].to_numpy(float), "gripper_rpy"
    else:
        rpy, src = np.tile(np.radians(fcfg.default_orientation_rpy_deg), (len(df), 1)), "fit.default_orientation_rpy_deg"
    R = np.stack([rpy_to_matrix(*r) for r in rpy])
    tw = df[TWIST].to_numpy(float) if all(c in df.columns for c in TWIST) else None
    t = df["t"].to_numpy(float)
    return Segment(t - t[0], df[POS].to_numpy(float), R, tw, name, (0, len(df) - 1)), src


def upsample_frame(df: pd.DataFrame, rate_hz: float = 100.0) -> pd.DataFrame:
    """Densify a trajectory frame for servo streaming.

    In servo mode the firmware heads for the latest setpoint, so setpoints far apart (e.g. 60 Hz recordings
    slowed down 2.5x = 42 ms) make the arm dash and wait, with peak speeds far above the planned ones.
    Positions use a cubic Hermite (Catmull-Rom) curve through every original sample at its original time, so the
    recorded points and their timing are kept exactly; orientation and twist columns are interpolated linearly."""
    t = df["t"].to_numpy(float)
    if len(t) < 3 or float(np.median(np.diff(t))) <= 1.0 / rate_hz * 1.05:
        return df
    T = float(t[-1])
    tn = np.arange(0.0, T, 1.0 / rate_hz)
    tn = np.unique(np.r_[tn, t])  # keep every original sample time
    i = np.clip(np.searchsorted(t, tn, side="right") - 1, 0, len(t) - 2)
    h = t[i + 1] - t[i]
    u = (tn - t[i]) / h
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    out = {"t": tn}
    for c in df.columns:
        if c in ("t", "traj"):
            continue
        y = df[c].to_numpy(float)
        if c in ("x", "y", "z"):
            m = np.gradient(y, t)  # central differences: Catmull-Rom tangents
            out[c] = h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]
        else:
            out[c] = np.interp(tn, t, y)
    return pd.DataFrame(out)[[c for c in df.columns if c != "traj"]]


def _frame_from_part(fitter: Fitter, part: dict) -> pd.DataFrame:
    seg = part["segment"]
    p, Rs, tw = fitter.apply(seg, part["params"])
    rpy = np.unwrap(np.array([matrix_to_rpy(R) for R in Rs]), axis=0)
    df = pd.DataFrame({"t": seg.t, "x": p[:, 0], "y": p[:, 1], "z": p[:, 2],
                       "roll": rpy[:, 0], "pitch": rpy[:, 1], "yaw": rpy[:, 2]})
    if tw is not None:
        df[TWIST] = tw
    return df


def read_log(path: Path, fingerprint: str) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            if e.get("fingerprint") == fingerprint:
                out.append(e)
    return out


def file_sha1(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _existing_output(out_path: Path, meta_path: Path, sha: str) -> tuple[dict[str, pd.DataFrame], dict | None]:
    """Clean frames + summary already written for this exact source file version (else empty)."""
    if not (out_path.exists() and meta_path.exists()):
        return {}, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("source_sha1") != sha or meta.get("fit_version") != FIT_VERSION:
        return {}, None
    df = pd.read_csv(out_path, dtype={TRAJ_COLUMN: str})
    if TRAJ_COLUMN in df.columns:
        frames = {n: select_trajectory(df, n) for n, _ in list_trajectories(df)}
    else:
        frames = {meta["trajectories"][0]["parts"][0]["name"]: df} if meta["trajectories"] else {}
    return frames, meta


def fit_file(ref: str, out_path: Path | None = None, cfg_path: str | None = None,
             log_path: Path | None = None) -> dict:
    """Fit the trajectories of ``ref`` (file or file#selector) and write/merge the clean copy.

    Default output: the preprocessed set matching the current config (data/preprocessed/NNN_*/...).
    Trajectories already fitted for the same source-file version are kept; the selected ones are redone.
    """
    from .preprocessed import clean_path_for, find_or_create_set, setup_fingerprint

    cfg = load_safety_config(cfg_path)
    fcfg = load_fit_config(cfg_path)
    src, selector = split_trajectory_ref(ref)
    df = pd.read_csv(src, dtype={TRAJ_COLUMN: str})
    grouped_in = TRAJ_COLUMN in df.columns
    if grouped_in:
        all_names = [n for n, _ in list_trajectories(df)]
        if selector is not None:
            names = [n for n in all_names if n == selector] or [all_names[int(selector)]]
        else:
            names = all_names
        frames = {n: select_trajectory(df, n, str(src)) for n in names}
    else:
        all_names = [src.stem]
        frames = {src.stem: select_trajectory(df, selector, str(src))}
    if out_path is None:
        set_dir = find_or_create_set(cfg_path)
        out_path = clean_path_for(set_dir, src)
        log_path = log_path or set_dir / "fit_log.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = log_path or out_path.parent / "fit_log.jsonl"
    meta_path = out_path.with_suffix(".fit.json")
    fp = setup_fingerprint(cfg_path)
    sha = file_sha1(src)

    out_frames, old = _existing_output(out_path, meta_path, sha)
    old_entries = {e["name"]: e for e in (old or {}).get("trajectories", [])}
    for name in frames:  # being redone: drop the old clean parts
        for part in old_entries.pop(name, {}).get("parts", []):
            out_frames.pop(part["name"], None)

    summary = {"source": str(src), "source_sha1": sha, "fit_version": FIT_VERSION, "config": cfg.source_path, "fingerprint": fp,
               "fit_config": fcfg.__dict__, "trajectories": []}
    new_entries: dict[str, dict] = {}
    for name, frame in frames.items():
        if all(f"q{i}" in frame.columns for i in range(1, 8)) and "x" not in frame.columns:
            new_entries[name] = {"name": name, "result": "skipped", "parts": [], "dropped": [],
                                 "reason": "joint trajectory: absolute joint angles cannot be re-placed"}
            print(f"[{name}] skipped (joint trajectory)")
            continue
        seg, orient_src = _segment_from_frame(frame, name, fcfg)
        fitter = Fitter(cfg, fcfg, read_log(log_path, fp))
        t0 = time.time()
        fitted, dropped = fitter.fit_or_split(seg)
        entry = {"name": name, "orientation_source": orient_src, "rows": len(seg.t), "duration_s": float(seg.t[-1]),
                 "seconds": round(time.time() - t0, 1), "parts": [], "dropped": []}
        multi = len(fitted) + len(dropped) > 1
        for i, part in enumerate(fitted, 1):
            pname = f"{name}__p{i}" if multi else name
            part = dict(part, segment=part["segment"].eased())   # start and end at rest (positions unchanged)
            out_frames[pname] = upsample_frame(_frame_from_part(fitter, part))  # >= 100 Hz setpoints for servo
            s = part["segment"]
            th = {k: round(float(v), 3) for k, v in part["params"].items()}
            shift = (np.array([th["cx"], th["cy"], th["cz"]]) - (fitter.R_b @ s.center + fitter.b)).round(2).tolist()
            entry["parts"].append({"name": pname, "source_rows": list(s.src_rows),
                                   "source_t": [float(seg.t[s.src_rows[0]]), float(seg.t[s.src_rows[1]])],
                                   "params": th, "shift_world_mm": shift, "slack_mm": round(part["slack"], 2),
                                   "evals": part["evals"],
                                   # validated start configuration: playback moves here in joint space first
                                   "start_joints_deg": np.degrees(fitter.start_joints(s, part["params"])).round(4).tolist(),
                                   "approach_via_joints_deg": [round(float(v), 4) for v in fcfg.seed_joints_deg]})
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"fingerprint": fp, "source": f"{src}#{name}", "part": pname, "params": th,
                                     "shift_world_mm": shift, "slack_mm": round(part["slack"], 2),
                                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        for d in dropped:
            s = d["segment"]
            entry["dropped"].append({"source_rows": list(s.src_rows),
                                     "source_t": [float(seg.t[s.src_rows[0]]), float(seg.t[s.src_rows[1]])],
                                     "best_slack_mm": round(d["best_slack"], 2)})
        entry["result"] = ("fitted" if not multi else "split") if fitted else "rejected"
        new_entries[name] = entry
        parts_txt = ", ".join(f"{p['name']} slack {p['slack_mm']:.1f} mm (yaw {p['params']['yaw']:.0f} deg, "
                              f"shift {p['shift_world_mm']})" for p in entry["parts"])
        print(f"[{name}] {entry['result']} in {entry['seconds']} s: {parts_txt or '-'}"
              + (f"; dropped {len(entry['dropped'])} part(s)" if entry["dropped"] else ""))

    # Source order; entries for trajectories not redone are kept as they were.
    entries = [new_entries.get(n) or old_entries.get(n) for n in all_names]
    summary["trajectories"] = [e for e in entries if e is not None]
    order = [p["name"] for e in summary["trajectories"] for p in e["parts"]]
    out_frames = {n: out_frames[n] for n in order if n in out_frames}
    grouped = grouped_in or len(out_frames) > 1 or any(len(e["parts"]) > 1 for e in summary["trajectories"])
    summary["grouped"] = grouped
    summary["output"] = str(out_path)
    if out_frames:
        if not grouped:
            out_df = next(iter(out_frames.values()))
        else:
            out_df = pd.concat([f.assign(**{TRAJ_COLUMN: n}) for n, f in out_frames.items()], ignore_index=True)
            out_df = out_df[[TRAJ_COLUMN] + [c for c in out_df.columns if c != TRAJ_COLUMN]]
        out_df.to_csv(out_path, index=False, float_format="%.6f")
    elif out_path.exists():
        out_path.unlink()
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_path if out_frames else '(no trajectories fitted)'} and {meta_path}", file=sys.stderr)
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fit Cartesian trajectories into the safety workcell (re-place / split) "
                                            "and write clean copies into the matching data/preprocessed/ set")
    p.add_argument("trajectories", nargs="+", help="CSV files, file.csv#name selections, or directories of CSVs")
    p.add_argument("--config", default=None, help="safety YAML (default config/safety.yaml); its 'fit:' section sets search options")
    p.add_argument("--out", default=None, help="explicit output file (single input only) instead of the preprocessed set")
    p.add_argument("--log", default=None, help="placement log (JSONL); default: fit_log.jsonl of the set / output directory")
    args = p.parse_args(argv)

    refs = []
    for r in args.trajectories:
        path = Path(r)
        refs += sorted(str(x) for x in path.glob("*.csv")) if path.is_dir() else [r]
    if args.out and len(refs) != 1:
        p.error("--out needs exactly one input")
    any_fit = False
    for r in refs:
        summary = fit_file(r, Path(args.out) if args.out else None, args.config, Path(args.log) if args.log else None)
        any_fit |= any(t["result"] in ("fitted", "split") for t in summary["trajectories"])
    return 0 if any_fit else 2


if __name__ == "__main__":
    sys.exit(main())
