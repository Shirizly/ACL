from __future__ import annotations

import argparse
import inspect
import time

from xarm.wrapper import XArmAPI

from .trajectory_io import load_joint_trajectory_csv


def connect_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
    arm.motion_enable(enable=True)
    arm.set_state(state=0)
    return arm


def set_mode_for_control(arm: XArmAPI, control: str) -> None:
    if control == "position":
        arm.set_mode(0)
    elif control == "velocity":
        arm.set_mode(4)
    else:
        raise ValueError(f"Unsupported control mode: {control}")
    arm.set_state(state=0)


def _call_joint_velocity(arm: XArmAPI, velocities: list[float]) -> None:
    fn = getattr(arm, "vc_set_joint_velocity", None)
    if fn is None:
        raise RuntimeError("SDK does not expose vc_set_joint_velocity")

    params = inspect.signature(fn).parameters

    if "is_sync" in params:
        fn(velocities, is_radian=True, is_sync=True)
        return
    if "is_radian" in params:
        fn(velocities, is_radian=True)
        return
    fn(velocities)


def playback_position(arm: XArmAPI, t: list[float], q: list[list[float]]) -> None:
    t0 = time.perf_counter()
    for idx, q_cmd in enumerate(q):
        target = t[idx]
        while (time.perf_counter() - t0) < target:
            time.sleep(0.001)

        arm.set_servo_angle(angle=q_cmd, is_radian=True, wait=False)


def playback_velocity(
    arm: XArmAPI,
    t: list[float],
    q: list[list[float]],
    dq: list[list[float]] | None,
) -> None:
    if dq is None:
        raise ValueError("Velocity control requires dq1..dq7 columns in trajectory CSV")

    t0 = time.perf_counter()
    for idx, dq_cmd in enumerate(dq):
        target = t[idx]
        while (time.perf_counter() - t0) < target:
            time.sleep(0.001)

        _call_joint_velocity(arm, dq_cmd)

    # Stop commanded joint velocities at the end of playback.
    _call_joint_velocity(arm, [0.0] * 7)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay joint trajectories on xArm")
    p.add_argument("--ip", default="127.0.0.1", help="Robot IP (sim: 127.0.0.1)")
    p.add_argument("--trajectory", required=True, help="CSV file with t,q1..q7[,dq1..dq7]")
    p.add_argument(
        "--control",
        choices=["position", "velocity"],
        default="position",
        help="Control mode to use",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traj = load_joint_trajectory_csv(args.trajectory)

    arm = connect_arm(args.ip)
    try:
        set_mode_for_control(arm, args.control)
        if args.control == "position":
            playback_position(arm, traj.t, traj.q)
        else:
            playback_velocity(arm, traj.t, traj.q, traj.dq)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
