from __future__ import annotations

import argparse
import inspect
import sys
import time

from xarm.wrapper import XArmAPI

from .safety.session import (
    EXIT_REJECTED,
    EXIT_SAFETY_STOP,
    SafetyRejected,
    SafetySession,
    SafetyStop,
    add_safety_args,
)
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


def playback_position(
    arm: XArmAPI,
    t: list[float],
    q: list[list[float]],
    safety: SafetySession | None = None,
) -> None:
    t0 = time.perf_counter()
    for idx, q_cmd in enumerate(q):
        target = t[idx]
        while (time.perf_counter() - t0) < target:
            time.sleep(0.001)

        if safety is not None:
            safety.check_abort(idx)
            safety.guard.check_joint_target(q_cmd, idx)
        arm.set_servo_angle(angle=q_cmd, is_radian=True, wait=False)


def playback_velocity(
    arm: XArmAPI,
    t: list[float],
    q: list[list[float]],
    dq: list[list[float]] | None,
    safety: SafetySession | None = None,
) -> None:
    if dq is None:
        raise ValueError("Velocity control requires dq1..dq7 columns in trajectory CSV")

    t0 = time.perf_counter()
    try:
        for idx, dq_cmd in enumerate(dq):
            target = t[idx]
            while (time.perf_counter() - t0) < target:
                time.sleep(0.001)

            if safety is not None:
                safety.check_abort(idx)
                dt_next = (t[idx + 1] - t[idx]) if idx + 1 < len(t) else 0.0
                safety.guard.check_joint_velocity(dq_cmd, dt_next, idx)
            _call_joint_velocity(arm, dq_cmd)
    finally:
        # Stop commanded joint velocities at the end of playback (or on abort).
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
    p.add_argument("--report-dir", default=None, help="Directory for safety reports and monitor logs")
    add_safety_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    traj = load_joint_trajectory_csv(args.trajectory)

    arm = connect_arm(args.ip)
    safety: SafetySession | None = None
    exit_code = 0
    try:
        safety = SafetySession(args, arm, args.ip, out_dir=args.report_dir)
        safety.preprocess_joint(traj.t, traj.q, traj.dq, args.control)
        set_mode_for_control(arm, args.control)
        safety.start_runtime()
        if args.control == "position":
            playback_position(arm, traj.t, traj.q, safety)
        else:
            playback_velocity(arm, traj.t, traj.q, traj.dq, safety)
        safety.wait_until_still()
        if safety.monitor is not None and safety.monitor.tripped.is_set():
            raise SafetyStop(f"runtime guard stopped the arm: {safety.monitor.trip_info.get('reason')}")
    except SafetyRejected as e:
        print(f"playback REJECTED by safety preprocessing, the robot was not moved: {e}", flush=True)
        exit_code = EXIT_REJECTED
    except SafetyStop as e:
        print(f"playback ABORTED by safety: {e}", flush=True)
        if safety is not None:
            safety.abort_reason = str(e)
        exit_code = EXIT_SAFETY_STOP
    finally:
        if safety is not None:
            safety.close()
        arm.disconnect()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
