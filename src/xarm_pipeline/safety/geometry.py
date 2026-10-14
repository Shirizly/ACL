"""Workcell box, frame transforms and capsule-vs-wall clearance.

Frames
------
world  : origin at the center of the box floor, z up. Box spans
         x in [-D/2, D/2], y in [-W/2, W/2], z in [0, H].
base   : the xArm base frame (the frame of get_position()).
         p_world = Rz(yaw) @ p_base + (bx, by, bz)

Clearance
---------
For a capsule (segment a-b, radius r) and an axis-aligned wall, the capsule
is inside the half-space iff both end spheres are (half-spaces are convex), so
the signed clearance to a wall is ``min(dist(a), dist(b)) - r``; positive =
inside.  This is exact for capsules.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import LINK_NAMES, SafetyConfig
from .kinematics import XARM7_NOMINAL_DH

WALL_NAMES = ("wall -x", "wall +x", "wall -y", "wall +y", "floor", "ceiling")

# Chain point indices (see kinematics.ArmModel) for each capsule.
CAPSULES: tuple[tuple[str, int, int], ...] = (
    ("base", 0, 1),
    ("upper_arm", 1, 2),
    ("elbow", 2, 3),
    ("elbow_offset", 3, 4),
    ("forearm", 4, 5),
    ("wrist", 5, 6),
    ("wrist_offset", 6, 7),
    ("tool", 7, 8),
    # Gripper box (geometry.tool_box): its 8 corners are chain points 9..16. Against a flat wall a box's
    # clearance is its nearest corner's, so 4 zero-radius segments covering all corners are exact.
    # Without a configured box all corners sit on the flange (dominated by the wrist_offset capsule).
    ("tool_box", 9, 16),
    ("tool_box", 10, 15),
    ("tool_box", 11, 14),
    ("tool_box", 12, 13),
)
STATIC_CAPSULES = ("base",)  # does not move with the joints
MOVING_CAPSULES = tuple(c for c in CAPSULES if c[0] not in STATIC_CAPSULES)
# Capsules rigidly attached to (or nearly so) the commanded Cartesian pose;
# these are the ones a Cartesian "clip" can move by translating the target.
END_EFFECTOR_CAPSULES = ("wrist", "wrist_offset", "tool", "tool_box")
# Capsules whose position is fully determined by the commanded flange pose
# (P6 = flange - d7*z7, P7 = flange, P8 = tool tip). They do not depend on how
# the IK resolves the arm's redundancy, so no IK-model tolerance is needed.
POSE_DETERMINED_CAPSULES = ("wrist_offset", "tool", "tool_box")
POINT_NAMES = (
    "base_floor", "shoulder", "elbow", "elbow_offset", "forearm_start",
    "wrist", "wrist_offset", "flange", "tool_tip",
) + tuple(f"tool_box_{i}" for i in range(8))


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------
# Batched forward kinematics -> chain points
# --------------------------------------------------------------------------
def chain_points_batch(Q, tool_offset_mm=(0.0, 0.0, 0.0), dh=XARM7_NOMINAL_DH, tool_box_corners=None) -> np.ndarray:
    """(N,7) joint angles [rad] -> (N,17,3) chain points in BASE frame [mm].

    Points 0..8 follow the arm (POINT_NAMES); 9..16 are the gripper-box corners given in the
    flange frame (``tool_box_corners``, (8,3)), or the flange point when there is no box."""
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    N = Q.shape[0]
    T = np.broadcast_to(np.eye(4), (N, 4, 4)).copy()
    O = []
    X = []
    R7 = None
    for i, (th0, d, alpha, a) in enumerate(dh):
        th = Q[:, i] + th0
        ct, st = np.cos(th), np.sin(th)
        ca, sa = math.cos(alpha), math.sin(alpha)
        A = np.zeros((N, 4, 4))
        A[:, 0, 0] = ct
        A[:, 0, 1] = -st
        A[:, 0, 3] = a
        A[:, 1, 0] = st * ca
        A[:, 1, 1] = ct * ca
        A[:, 1, 2] = -sa
        A[:, 1, 3] = -d * sa
        A[:, 2, 0] = st * sa
        A[:, 2, 1] = ct * sa
        A[:, 2, 2] = ca
        A[:, 2, 3] = d * ca
        A[:, 3, 3] = 1.0
        T = T @ A
        O.append(T[:, :3, 3].copy())
        X.append(T[:, :3, 0].copy())
        R7 = T[:, :3, :3]
    a4, a5, a7 = dh[3][3], dh[4][3], dh[6][3]
    P = np.empty((N, 17, 3))
    P[:, 0] = 0.0
    P[:, 1] = O[0]
    P[:, 2] = O[2]
    P[:, 3] = O[2] + a4 * X[2]
    P[:, 4] = O[3] + a5 * X[3]
    P[:, 5] = O[4]
    P[:, 6] = O[5] + a7 * X[5]
    P[:, 7] = O[6]
    P[:, 8] = O[6] + R7 @ np.asarray(tool_offset_mm, dtype=float)
    if tool_box_corners is None:
        P[:, 9:] = O[6][:, None, :]
    else:
        P[:, 9:] = O[6][:, None, :] + np.einsum("nij,kj->nki", R7, np.asarray(tool_box_corners, dtype=float))
    return P


# --------------------------------------------------------------------------
# Workcell
# --------------------------------------------------------------------------
@dataclass
class Violation:
    """First / worst violation found by a check."""

    sample_index: int        # index in the checked array (dense or original)
    link: str
    wall: str
    clearance_mm: float      # signed distance capsule surface -> wall (negative = through the wall)
    required_mm: float       # clearance required by the check
    point_world_mm: list[float]

    @property
    def deficit_mm(self) -> float:
        return self.required_mm - self.clearance_mm

    def describe(self) -> str:
        where = "THROUGH the wall" if self.clearance_mm < 0 else "inside the margin"
        return (
            f"link '{self.link}' vs {self.wall}: clearance {self.clearance_mm:.1f} mm "
            f"({where}), required >= {self.required_mm:.1f} mm, deficit {self.deficit_mm:.1f} mm"
        )


class Workcell:
    def __init__(self, cfg: SafetyConfig) -> None:
        self.cfg = cfg
        self.lo = np.array([-cfg.depth_D_mm / 2, -cfg.width_W_mm / 2, 0.0])
        self.hi = np.array([cfg.depth_D_mm / 2, cfg.width_W_mm / 2, cfg.height_H_mm])
        self.yaw = math.radians(cfg.base_yaw_deg)
        self.R = rot_z(self.yaw)  # world <- base
        self.t = np.array([cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm])
        self.radii = np.array([cfg.radii_mm[name] for name, _, _ in CAPSULES])
        self.tool_offset = np.asarray(cfg.tool_offset_mm, dtype=float)
        self.tool_box_corners = cfg.tool_box_corners_mm()

    # ---- transforms -------------------------------------------------------
    def base_to_world(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return p @ self.R.T + self.t

    def world_to_base(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return (p - self.t) @ self.R

    def world_vec_to_base(self, v) -> np.ndarray:
        return np.asarray(v, dtype=float) @ self.R

    # ---- distances --------------------------------------------------------
    def wall_distances(self, pw) -> np.ndarray:
        """(...,3) world points -> (...,6) signed distance to each wall (+ = inside)."""
        pw = np.asarray(pw, dtype=float)
        d = np.empty(pw.shape[:-1] + (6,))
        d[..., 0] = pw[..., 0] - self.lo[0]
        d[..., 1] = self.hi[0] - pw[..., 0]
        d[..., 2] = pw[..., 1] - self.lo[1]
        d[..., 3] = self.hi[1] - pw[..., 1]
        d[..., 4] = pw[..., 2] - self.lo[2]
        d[..., 5] = self.hi[2] - pw[..., 2]
        return d

    def points_world(self, Q) -> np.ndarray:
        """(N,7) joints -> (N,17,3) chain points in WORLD frame."""
        return self.base_to_world(chain_points_batch(Q, self.tool_offset, tool_box_corners=self.tool_box_corners))

    def capsule_clearance(self, Pw: np.ndarray, capsules=MOVING_CAPSULES) -> np.ndarray:
        """(N,9,3) world chain points -> (N, n_caps, 6) clearance per capsule/wall."""
        D = self.wall_distances(Pw)  # (N,9,6)
        out = np.empty((Pw.shape[0], len(capsules), 6))
        for k, (name, i, j) in enumerate(capsules):
            out[:, k, :] = np.minimum(D[:, i, :], D[:, j, :]) - self.cfg.radii_mm[name]
        return out

    def clearance_for_q(self, Q, capsules=MOVING_CAPSULES):
        Pw = self.points_world(Q)
        return self.capsule_clearance(Pw, capsules), Pw

    def first_violation(self, Q, required_mm, capsules=MOVING_CAPSULES) -> tuple[Violation | None, np.ndarray]:
        """Check (N,7) configs against ``required_mm`` (scalar or one value per
        capsule). Returns (first violation or None, min slack per sample) where
        slack = clearance - required."""
        C, Pw = self.clearance_for_q(Q, capsules)
        req = np.broadcast_to(np.asarray(required_mm, dtype=float), (len(capsules),))
        S = C - req[None, :, None]
        min_per_sample = S.reshape(S.shape[0], -1).min(axis=1)
        bad = np.nonzero(min_per_sample < 0.0)[0]
        if bad.size == 0:
            return None, min_per_sample
        n = int(bad[0])
        k, w = np.unravel_index(int(np.argmin(S[n])), S[n].shape)
        name, i, j = capsules[k]
        di = self.wall_distances(Pw[n, i])[w]
        dj = self.wall_distances(Pw[n, j])[w]
        pt = Pw[n, i] if di <= dj else Pw[n, j]
        return (
            Violation(n, name, WALL_NAMES[w], float(C[n, k, w]), float(req[k]), [float(v) for v in pt]),
            min_per_sample,
        )

    def _base_column_sampling(self):
        """Fixed weights W (M,17) so that W @ chain_points gives the end-effector sample points, and their radii.
        Built once: wrist/tool capsules at 7 points each, gripper-box edges <= 10 mm apart, box face centers."""
        if getattr(self, "_bc_W", None) is not None:
            return self._bc_W, self._bc_r
        rows, rad = [], []

        def lerp(i, j, f):
            w = np.zeros(17)
            w[i] += 1 - f
            w[j] += f
            return w

        for name, i, j in CAPSULES:
            if name in ("wrist", "wrist_offset", "tool"):
                for k in range(7):
                    rows.append(lerp(i, j, k / 6))
                    rad.append(self.cfg.radii_mm[name])
        corners = self.cfg.tool_box_corners_mm()
        r_box = self.cfg.radii_mm.get("tool_box", 0.0)
        for a in range(8):
            for b in range(a + 1, 8):
                if bin(a ^ b).count("1") != 1:
                    continue
                L = 0.0 if corners is None else float(np.linalg.norm(corners[a] - corners[b]))
                n = max(1, int(math.ceil(L / 10.0)))
                for k in range(n + 1):
                    rows.append(lerp(9 + a, 9 + b, k / n))
                    rad.append(r_box)
        for bit in range(3):  # face centers
            for side in (0, 1):
                w = np.zeros(17)
                for c in range(8):
                    if (c >> (2 - bit)) & 1 == side:
                        w[9 + c] = 0.25
                rows.append(w)
                rad.append(r_box)
        self._bc_W, self._bc_r = np.asarray(rows), np.asarray(rad)
        return self._bc_W, self._bc_r

    def base_column_slack(self, Pw: np.ndarray) -> np.ndarray:
        """(N,17,3) world chain points -> (N,) clearance of the end effector (wrist capsules, tool capsule,
        gripper box) from the robot's own base column, minus safety_margin_mm (< 0 = too close).

        The base column (floor -> shoulder, radii_mm.base) is a fixed obstacle like a wall: the gripper hanging
        down can swing into it when the wrist passes near the base axis."""
        W, r = self._base_column_sampling()
        X = np.einsum("mk,nkd->nmd", W, np.asarray(Pw, dtype=float)) - self.t   # relative to the base origin
        z = np.clip(X[..., 2], 0.0, XARM7_NOMINAL_DH[0][1])
        d = np.sqrt(X[..., 0] ** 2 + X[..., 1] ** 2 + (X[..., 2] - z) ** 2)
        return (d - r[None, :] - self.cfg.radii_mm["base"]).min(axis=1) - self.cfg.safety_margin_mm

    def static_check(self) -> list[str]:
        """Config sanity: the fixed base column against the side walls and ceiling, and the shoulder
        (the fixed end of the upper-arm link) against the floor.

        The base column is never checked against the floor, so the base may sit below it (base z < 0)
        and protrude through it. Every moving link, including the upper arm from the shoulder joint up,
        is still checked against the floor."""
        shoulder_z = XARM7_NOMINAL_DH[0][1]
        P = self.base_to_world(np.array([[0, 0, 0], [0, 0, shoulder_z]], dtype=float))
        d = np.minimum(self.wall_distances(P[0]), self.wall_distances(P[1])) - self.cfg.radii_mm["base"]
        problems = []
        for w in (0, 1, 2, 3, 5):
            if d[w] < 0:
                problems.append(f"robot base column is {-d[w]:.1f} mm through the {WALL_NAMES[w]}")
        shoulder_clear = self.wall_distances(P[1])[4] - self.cfg.radii_mm["upper_arm"] - self.cfg.safety_margin_mm
        if shoulder_clear < 0:
            problems.append(
                f"the shoulder joint housing (upper_arm radius {self.cfg.radii_mm['upper_arm']:.0f} mm + margin) is "
                f"{-shoulder_clear:.1f} mm into the floor: the base sits too far below it "
                f"(base z = {self.cfg.base_z_mm:.0f} mm; it must be >= "
                f"{-(shoulder_z - self.cfg.radii_mm['upper_arm'] - self.cfg.safety_margin_mm):.0f} mm)")
        return problems


__all__ = [
    "CAPSULES", "END_EFFECTOR_CAPSULES", "POSE_DETERMINED_CAPSULES", "LINK_NAMES", "MOVING_CAPSULES", "POINT_NAMES",
    "Violation", "WALL_NAMES", "Workcell", "chain_points_batch", "rot_z",
]
