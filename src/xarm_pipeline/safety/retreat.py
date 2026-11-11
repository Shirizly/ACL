"""Safe return to a joint target (usually the ready pose) from wherever the arm stopped.

A direct joint move is used when it passes the wall check. Otherwise the flange backs off along a
straight line towards the target's flange pose (orientation slerped along), tracked with local IK,
until the remaining direct joint move is safe. Every leg is validated as the linear joint path the
firmware executes (``set_servo_angle`` in mode 0), so the whole returned path is wall-checked.
"""
from __future__ import annotations

import os

import numpy as np

from .config import SafetyConfig
from .kinematics import IKFailure, flange_pose, ik_step_to, matrix_to_rpy
from .preprocess import ValidationResult, validate_joint_trajectory

STEP_MM = 5.0          # IK continuation step
WAYPOINT_MM = 25.0     # spacing of commanded waypoints / direct-move attempts


def _so3_log(R: np.ndarray) -> np.ndarray:
    ang = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-9:
        return np.zeros(3)
    if np.pi - ang < 1e-6:  # 180 deg: axis from the symmetric part
        i = int(np.argmax(np.diag(R)))
        axis = R[:, i] + np.eye(3)[:, i]
        return ang * axis / np.linalg.norm(axis)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return ang * w / (2.0 * np.sin(ang))


def _so3_exp(w: np.ndarray) -> np.ndarray:
    ang = float(np.linalg.norm(w))
    if ang < 1e-12:
        return np.eye(3)
    k = w / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * K @ K


def _check(cfg: SafetyConfig, q_from, path, name: str) -> ValidationResult:
    return validate_joint_trajectory(cfg, list(range(len(path))), [list(map(float, q)) for q in path],
                                     q_start=list(map(float, q_from)), trajectory_name=name)


SIM_HOSTS = ("127.0.0.1", "localhost")


def is_simulator(ip: str) -> bool:
    """The local UFACTORY simulator (Docker). XARM_IS_SIM=1/0 overrides the guess."""
    env = os.environ.get("XARM_IS_SIM")
    if env is not None:
        return env == "1"
    return ip.strip() in SIM_HOSTS


def plan_safe_return(cfg: SafetyConfig, q_now, q_target, simulator: bool = False) -> tuple[list[np.ndarray], str]:
    """Joint waypoints [rad] from q_now to q_target, all legs wall-checked. Raises RuntimeError if none found.

    simulator=True: if the CURRENT pose is already through a wall/the floor (the sim boots at the zero pose with
    the gripper below the floor), return an unchecked direct move instead of refusing. Never on the real robot."""
    q_now = np.asarray(q_now, float)[:7]
    q_target = np.asarray(q_target, float)[:7]
    here = _check(cfg, q_now, [q_now], "current pose")
    if not here.ok and "THROUGH" in here.message:
        if simulator:
            return [q_target], ("SIMULATOR RECOVERY: current pose is through a wall/the floor (sim restart?), "
                                "moving to the target WITHOUT the wall check. " + here.message)
        raise RuntimeError(
            "the arm's CURRENT pose is already through a wall/the floor in the safety model, so no checked move can "
            "start from it (typical right after a simulator restart: zero pose, gripper below the floor). Simulator: "
            "use 'Force ready pose' in the GUI or scripts/control/go_ready.sh --force --yes. Real robot: check "
            "config/safety.yaml matches the cell, then jog the arm clear by hand. Details: " + here.message)
    res = _check(cfg, q_now, [q_target], "direct return")
    if res.ok:
        return [q_target], "direct joint move: " + res.message
    first_msg = res.message

    T0, T1 = flange_pose(q_now), flange_pose(q_target)
    p0, p1 = T0[:3, 3], T1[:3, 3]
    dist = float(np.linalg.norm(p1 - p0))
    n = max(int(np.ceil(dist / STEP_MM)), 1)
    R0 = T0[:3, :3]
    w01 = _so3_log(R0.T @ T1[:3, :3])
    every = max(int(round(WAYPOINT_MM / STEP_MM)), 1)
    waypoints: list[np.ndarray] = []
    q = q_now.copy()
    for k in range(1, n + 1):
        s = k / n
        pose = list(p0 + s * (p1 - p0)) + list(matrix_to_rpy(R0 @ _so3_exp(s * w01)))
        try:
            q = ik_step_to(pose, q)
        except IKFailure as e:
            raise RuntimeError(f"retreat IK failed at {s:.0%} of the way: {e}") from e
        if k % every and k != n:
            continue
        leg = _check(cfg, waypoints[-1] if waypoints else q_now, [q], "retreat leg")
        if not leg.ok:
            raise RuntimeError(f"no safe retreat (direct: {first_msg}; retreat blocked at {s:.0%}: {leg.message})")
        waypoints.append(q.copy())
        if _check(cfg, q, [q_target], "return after retreat").ok:
            return waypoints + [q_target], f"retreated {s * dist:.0f} mm in Cartesian space, then joint move"
    raise RuntimeError(f"no safe retreat found (direct: {first_msg})")


def safe_move_to_joints(arm, cfg: SafetyConfig, target_deg, speed_deg_s: float = 20.0, check: bool = True) -> int:
    """Move the arm to target_deg via plan_safe_return (mode 0, blocking). Returns the SDK code."""
    _, q_now = arm.get_servo_angle(is_radian=True)
    if check:
        path, how = plan_safe_return(cfg, q_now[:7], np.radians(target_deg))
        print("safe return: " + how)
    else:
        path = [np.radians(target_deg)]
    code = 0
    for q in path:
        code = arm.set_servo_angle(angle=np.degrees(q).tolist(), is_radian=False, speed=speed_deg_s, wait=True)
        if code != 0:
            break
    return code
