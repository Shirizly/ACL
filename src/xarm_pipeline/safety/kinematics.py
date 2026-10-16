"""xArm 7 forward kinematics (pure numpy, no robot needed).

Kinematic parameters
--------------------
The nominal parameters below are the *modified* (Craig) DH parameters of the
xArm 7, read back from the firmware with ``arm.get_dh_params()`` (firmware
v2.4.0 in the UFACTORY simulator).  The firmware returns 28 floats, i.e. 7
joints x ``(theta_offset, d, alpha, a)``; distances in mm, angles in rad::

    joint  theta_off   d[mm]    alpha[rad]   a[mm]
      1      0         267.0      0.0         0.0
      2      0           0.0     -pi/2        0.0
      3      0         293.0     +pi/2        0.0
      4      0           0.0     +pi/2       52.5
      5      0         342.5     +pi/2       77.5
      6      0           0.0     +pi/2        0.0
      7      0          97.0     -pi/2       76.0

Per joint the modified-DH transform is::

    T_{i-1,i} = RotX(alpha_i) * TransX(a_i) * RotZ(q_i + theta_off_i) * TransZ(d_i)

At q = 0 this gives flange position (206, 0, 120.5) mm and orientation
roll=180 deg, which matches ``arm.get_position()`` of the simulator at home.

Real robots ship with *per-unit calibrated* DH parameters (small deviations
from nominal).  Use :func:`dh_from_firmware` to read the robot's own values;
``check_dh_against_firmware`` reports the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

# (theta_offset rad, d mm, alpha rad, a mm) per joint, modified DH.
XARM7_NOMINAL_DH: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 267.0, 0.0, 0.0),
    (0.0, 0.0, -math.pi / 2, 0.0),
    (0.0, 293.0, math.pi / 2, 0.0),
    (0.0, 0.0, math.pi / 2, 52.5),
    (0.0, 342.5, math.pi / 2, 77.5),
    (0.0, 0.0, math.pi / 2, 0.0),
    (0.0, 97.0, -math.pi / 2, 76.0),
)

# xArm 7 joint limits (deg), from xarm-python-sdk XCONF.Robot.JOINT_LIMITS[7][7].
# Used by the local IK to stay inside the range; the firmware enforces its own.
XARM7_JOINT_LIMITS_DEG = (
    (-360.0, 360.0), (-118.0, 120.0), (-360.0, 360.0), (-11.0, 225.0),
    (-360.0, 360.0), (-97.0, 180.0), (-360.0, 360.0),
)


def dh_from_firmware(arm) -> tuple[tuple[float, float, float, float], ...]:
    """Read DH parameters from a connected XArmAPI (``arm.get_dh_params``)."""
    code, vals = arm._arm.get_dh_params()  # noqa: SLF001 - not exposed on wrapper in 1.18.4
    if code != 0 or len(vals) < 28:
        raise RuntimeError(f"get_dh_params failed: code={code} vals={vals}")
    return tuple(tuple(float(v) for v in vals[4 * i: 4 * i + 4]) for i in range(7))  # type: ignore[return-value]


def check_dh_against_firmware(arm, dh=XARM7_NOMINAL_DH) -> float:
    """Return max abs difference between ``dh`` and the robot's DH params."""
    fw = dh_from_firmware(arm)
    return float(np.max(np.abs(np.asarray(fw) - np.asarray(dh))))


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """xArm convention: R = Rz(yaw) * Ry(pitch) * Rx(roll) (fixed-axis XYZ)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def _dh_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def joint_frames(q, dh=XARM7_NOMINAL_DH) -> list[np.ndarray]:
    """Return [T_0_1, ..., T_0_7] (4x4, mm) for joint vector q (rad)."""
    T = np.eye(4)
    out = []
    for qi, (th0, d, alpha, a) in zip(q, dh):
        T = T @ _dh_transform(alpha, a, float(qi) + th0, d)
        out.append(T)
    return out


def flange_pose(q, dh=XARM7_NOMINAL_DH) -> np.ndarray:
    """4x4 flange pose in robot base frame (mm)."""
    return joint_frames(q, dh)[-1]


def fk_pose(q, dh=XARM7_NOMINAL_DH, tcp_offset=None) -> list[float]:
    """[x,y,z (mm), roll,pitch,yaw (rad)] like ``arm.get_position``.

    ``tcp_offset`` = firmware TCP offset [x,y,z mm, r,p,y rad] (default 0).
    """
    T = flange_pose(q, dh)
    if tcp_offset is not None:
        To = np.eye(4)
        To[:3, :3] = rpy_to_matrix(*tcp_offset[3:6])
        To[:3, 3] = tcp_offset[:3]
        T = T @ To
    r, p, y = matrix_to_rpy(T[:3, :3])
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), r, p, y]


@dataclass
class ArmModel:
    """Collision geometry: chain of points connected by capsules.

    Chain points (robot base frame, mm):
      P0 base floor (0,0,0)       P1 shoulder (O1, z=267)
      P2 elbow (O3)               P3 elbow offset (O4 = O3 + a4*x3)
      P4 forearm start (O4 + a5*x4)  P5 wrist (O5)
      P6 wrist offset (O6 + a7*x6)   P7 flange (O7)
      P8 tool tip (flange * tool_offset)
    """

    radii: dict[str, float]
    tool_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dh: tuple = XARM7_NOMINAL_DH
    # Capsule names and chain point indices.
    capsules: tuple[tuple[str, int, int], ...] = field(
        default=(
            ("base", 0, 1),
            ("upper_arm", 1, 2),
            ("elbow", 2, 3),
            ("elbow_offset", 3, 4),
            ("forearm", 4, 5),
            ("wrist", 5, 6),
            ("wrist_offset", 6, 7),
            ("tool", 7, 8),
        )
    )

    def chain_points(self, q) -> np.ndarray:
        """(9,3) chain points in robot base frame (mm)."""
        F = joint_frames(q, self.dh)
        a4 = self.dh[3][3]
        a5 = self.dh[4][3]
        a7 = self.dh[6][3]
        O = [f[:3, 3] for f in F]
        X = [f[:3, 0] for f in F]
        p = np.empty((9, 3))
        p[0] = (0.0, 0.0, 0.0)
        p[1] = O[0]
        p[2] = O[2]
        p[3] = O[2] + a4 * X[2]  # == O[3] since d4 = 0
        p[4] = O[3] + a5 * X[3]
        p[5] = O[4]
        p[6] = O[5] + a7 * X[5]
        p[7] = O[6]
        p[8] = O[6] + F[6][:3, :3] @ np.asarray(self.tool_offset_mm, dtype=float)
        return p

    def capsule_list(self, q) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        pts = self.chain_points(q)
        return [(n, pts[i], pts[j], float(self.radii[n])) for n, i, j in self.capsules]


def max_reach_mm(model: ArmModel) -> float:
    """Upper bound of distance from any joint axis to any point of the arm."""
    links = sum(abs(d) + abs(a) for _, d, _, a in model.dh)
    return links + float(np.linalg.norm(model.tool_offset_mm)) + max(model.radii.values())


def jacobian(q, dh=XARM7_NOMINAL_DH, tcp_offset_mm=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Geometric Jacobian (6x7; rows vx,vy,vz [mm/rad], wx,wy,wz) of the point
    ``flange * tcp_offset_mm`` and its 4x4 pose."""
    F = joint_frames(q, dh)
    T = F[-1].copy()
    T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.asarray(tcp_offset_mm, dtype=float)
    p = T[:3, 3]
    J = np.zeros((6, 7))
    for i, Fi in enumerate(F):
        z = Fi[:3, 2]
        o = Fi[:3, 3]
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return J, T


def _rot_error(R_target: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Axis-angle vector (rad, base frame) rotating R onto R_target."""
    Re = R_target @ R.T
    cos_a = np.clip((np.trace(Re) - 1.0) / 2.0, -1.0, 1.0)
    angle = math.acos(cos_a)
    v = np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    if angle < 1e-9:
        return 0.5 * v
    s = math.sin(angle)
    if abs(s) < 1e-6:  # ~180 deg: pick axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(Re) + 1.0) / 2.0, 0.0, None))
        return angle * axis
    return angle * v / (2.0 * s)


class IKFailure(RuntimeError):
    pass


def ik_step_to(
    pose_target,
    q_seed,
    dh=XARM7_NOMINAL_DH,
    tcp_offset=None,
    tol_mm: float = 0.01,
    tol_rad: float = 1e-4,
    max_iter: int = 100,
    damping: float = 1e-3,
) -> np.ndarray:
    """Local IK: damped-least-squares Newton iterations from ``q_seed``.

    Minimum-norm joint motion from the seed (no null-space objective), which is
    how an incremental servo controller resolves the 7-DOF redundancy for small
    steps.  Use it with a dense path (continuation) so each call starts very
    close to its solution.  ``pose_target`` = [x,y,z mm, roll,pitch,yaw rad] of
    the *firmware TCP* (``tcp_offset`` = firmware TCP offset, default 0).
    """
    q = np.array(q_seed, dtype=float)
    R_t = rpy_to_matrix(*pose_target[3:6])
    p_t = np.asarray(pose_target[:3], dtype=float)
    off_p = (0.0, 0.0, 0.0)
    R_off = np.eye(3)
    if tcp_offset is not None:
        off_p = tuple(tcp_offset[:3])
        R_off = rpy_to_matrix(*tcp_offset[3:6])
    lo = np.radians([l for l, _ in XARM7_JOINT_LIMITS_DEG])
    hi = np.radians([h for _, h in XARM7_JOINT_LIMITS_DEG])
    for _ in range(max_iter):
        J, T = jacobian(q, dh, off_p)
        e_p = p_t - T[:3, 3]
        e_r = _rot_error(R_t, T[:3, :3] @ R_off)
        if np.linalg.norm(e_p) < tol_mm and np.linalg.norm(e_r) < tol_rad:
            return q
        # Scale rotation rows so mm and rad are comparable (1 rad ~ 100 mm).
        W = np.diag([1, 1, 1, 100, 100, 100.0])
        Jw = W @ J
        e = W @ np.concatenate([e_p, e_r])
        # Limit the Cartesian step for robustness.
        n = np.linalg.norm(e)
        if n > 20.0:
            e = e * (20.0 / n)
        dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + (damping ** 2) * np.eye(6), e)
        q = np.clip(q + dq, lo, hi)
    raise IKFailure(
        f"local IK did not converge to {list(np.round(pose_target, 3))}: "
        f"pos err {np.linalg.norm(e_p):.3f} mm, rot err {math.degrees(np.linalg.norm(e_r)):.3f} deg"
    )


def ik_step_biased(
    pose_target,
    q_seed,
    q_bias,
    bias_gain: float = 0.2,
    dh=XARM7_NOMINAL_DH,
    tcp_offset=None,
    tol_mm: float = 0.01,
    tol_rad: float = 1e-4,
    max_iter: int = 150,
    damping: float = 1e-3,
) -> np.ndarray:
    """Like ik_step_to, plus a null-space pull of the 7th degree of freedom toward ``q_bias``.

    The pose is reached exactly (same tolerances); among the joint solutions for it, the step prefers the
    ones closer to q_bias. Used to keep a preferred arm posture along a Cartesian path, which plain
    minimum-norm continuation does not control (it can drift into contorted but valid postures)."""
    q = np.array(q_seed, dtype=float)
    q_bias = np.asarray(q_bias, dtype=float)
    R_t = rpy_to_matrix(*pose_target[3:6])
    p_t = np.asarray(pose_target[:3], dtype=float)
    off_p = (0.0, 0.0, 0.0)
    R_off = np.eye(3)
    if tcp_offset is not None:
        off_p = tuple(tcp_offset[:3])
        R_off = rpy_to_matrix(*tcp_offset[3:6])
    lo = np.radians([l for l, _ in XARM7_JOINT_LIMITS_DEG])
    hi = np.radians([h for _, h in XARM7_JOINT_LIMITS_DEG])
    W = np.diag([1, 1, 1, 100, 100, 100.0])
    for it in range(max_iter):
        J, T = jacobian(q, dh, off_p)
        e_p = p_t - T[:3, 3]
        e_r = _rot_error(R_t, T[:3, :3] @ R_off)
        converged = np.linalg.norm(e_p) < tol_mm and np.linalg.norm(e_r) < tol_rad
        Jw = W @ J
        Jpinv = Jw.T @ np.linalg.inv(Jw @ Jw.T + (damping ** 2) * np.eye(6))
        null = (np.eye(7) - Jpinv @ Jw) @ (bias_gain * (q_bias - q))
        if converged and (it > 0 or np.linalg.norm(null) < 1e-4):
            return q
        e = W @ np.concatenate([e_p, e_r])
        n = np.linalg.norm(e)
        if n > 20.0:
            e = e * (20.0 / n)
        q = np.clip(q + Jpinv @ e + null, lo, hi)
    raise IKFailure(
        f"biased IK did not converge to {list(np.round(pose_target, 3))}: "
        f"pos err {np.linalg.norm(e_p):.3f} mm, rot err {math.degrees(np.linalg.norm(e_r)):.3f} deg"
    )
