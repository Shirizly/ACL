from __future__ import annotations

import argparse
import json
import math
import threading
import time
from pathlib import Path

import pandas as pd

from xarm.wrapper import XArmAPI

from .analyze_cartesian_error import analyze_cartesian_error
from .trajectory_io import load_cartesian_trajectory_csv


def _parse_default_orientation_rpy(value: str) -> list[float] | None:
    text = value.strip()
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--default-orientation-rpy must be 'roll,pitch,yaw' in radians")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _resolve_missing_orientation(
    pose: list[list[float]],
    fallback_rpy: list[float],
) -> list[list[float]]:
    out: list[list[float]] = []
    for row in pose:
        new_row = list(row)
        for k in range(3):
            idx = 3 + k
            if math.isnan(float(new_row[idx])):
                new_row[idx] = float(fallback_rpy[k])
        out.append(new_row)
    return out


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _unwrap_to_near(reference_rad: float, angle_rad: float) -> float:
    return reference_rad + _wrap_to_pi(angle_rad - reference_rad)


def _prepare_continuous_orientation_trajectory(
    pose: list[list[float]],
    start_pose: list[float] | None,
    max_orientation_step_rad: float,
) -> list[list[float]]:
    if not pose:
        return []

    prepared = [list(row) for row in pose]

    prev_rpy = (
        [float(start_pose[3]), float(start_pose[4]), float(start_pose[5])]
        if start_pose is not None and len(start_pose) >= 6
        else [float(prepared[0][3]), float(prepared[0][4]), float(prepared[0][5])]
    )

    for row in prepared:
        for k in range(3):
            idx = 3 + k
            unwrapped = _unwrap_to_near(prev_rpy[k], float(row[idx]))
            delta = unwrapped - prev_rpy[k]
            if abs(delta) > max_orientation_step_rad:
                unwrapped = prev_rpy[k] + math.copysign(max_orientation_step_rad, delta)
            row[idx] = float(unwrapped)
            prev_rpy[k] = float(unwrapped)

    return prepared


def _build_recording_timeline(t: list[float], start_hold_seconds: float) -> list[float]:
    if not t:
        return []

    if len(t) == 1:
        return [0.0]

    shifted = [0.0]
    base_dt = max(float(t[1]) - float(t[0]), 0.0)
    first_motion_time = max(start_hold_seconds, base_dt)
    for idx in range(1, len(t)):
        shifted.append(first_motion_time + (float(t[idx]) - float(t[1])))
    return shifted


def _enforce_joint_soft_limit(arm: XArmAPI, limit_rad: float, idx: int) -> None:
    code, joints = arm.get_servo_angle(is_radian=True)
    if code != 0 or joints is None or len(joints) < 7:
        raise RuntimeError(
            f"Failed to read joint angles for safety check at command {idx}, code={code}"
        )

    max_abs_joint = max(abs(float(v)) for v in joints[:7])
    if max_abs_joint > limit_rad:
        deg = max_abs_joint * 180.0 / math.pi
        limit_deg = limit_rad * 180.0 / math.pi
        raise RuntimeError(
            f"Joint soft limit exceeded before command {idx}: {deg:.3f} deg > {limit_deg:.3f} deg"
        )


def _enforce_joint_soft_limit_wrapped(arm: XArmAPI, limit_rad: float, idx: int) -> None:
    code, joints = arm.get_servo_angle(is_radian=True)
    if code != 0 or joints is None or len(joints) < 7:
        raise RuntimeError(
            f"Failed to read joint angles for safety check at command {idx}, code={code}"
        )

    max_abs_joint = max(abs(_wrap_to_pi(float(v))) for v in joints[:7])
    if max_abs_joint > limit_rad:
        deg = max_abs_joint * 180.0 / math.pi
        limit_deg = limit_rad * 180.0 / math.pi
        raise RuntimeError(
            f"Wrapped joint soft limit exceeded before command {idx}: {deg:.3f} deg > {limit_deg:.3f} deg"
        )


def connect_arm(ip: str) -> XArmAPI:
    arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_state(state=0)
    return arm


class FeedbackSampler:
    def __init__(self, arm: XArmAPI, sample_period_s: float) -> None:
        self.arm = arm
        self.sample_period_s = max(sample_period_s, 0.0)
        self.samples: list[dict[str, float]] = []
        self.error_count = 0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            t_now = time.perf_counter() - self._t0
            code, pose = self.arm.get_position(is_radian=True)
            if code == 0 and len(pose) >= 6:
                self.samples.append(
                    {
                        "t": t_now,
                        "x": float(pose[0]),
                        "y": float(pose[1]),
                        "z": float(pose[2]),
                        "roll": float(pose[3]),
                        "pitch": float(pose[4]),
                        "yaw": float(pose[5]),
                    }
                )
            else:
                self.error_count += 1

            if self.sample_period_s > 0.0:
                time.sleep(self.sample_period_s)

    @property
    def t0(self) -> float:
        return self._t0


def set_mode_for_control(arm: XArmAPI, control: str) -> None:
    if control == "pose":
        arm.set_mode(0)
    elif control == "servo":
        arm.set_mode(1)
    elif control == "velocity":
        arm.set_mode(5)
    else:
        raise ValueError(f"Unsupported control mode: {control}")
    arm.set_state(state=0)


def playback_pose(
    arm: XArmAPI,
    t: list[float],
    pose: list[list[float]],
    t0: float,
    command_log: list[dict[str, float | int]],
    joint_limit_rad: float,
    safety_every: int,
    joint_limit_mode: str,
) -> None:
    nonzero_codes = 0
    for idx, p_cmd in enumerate(pose):
        target = t[idx]
        while (time.perf_counter() - t0) < target:
            time.sleep(0.001)
        if idx % safety_every == 0:
            if joint_limit_mode == "wrapped":
                _enforce_joint_soft_limit_wrapped(arm, joint_limit_rad, idx)
            else:
                _enforce_joint_soft_limit(arm, joint_limit_rad, idx)
        sent_t = time.perf_counter() - t0
        code = arm.set_position(
            x=p_cmd[0],
            y=p_cmd[1],
            z=p_cmd[2],
            roll=p_cmd[3],
            pitch=p_cmd[4],
            yaw=p_cmd[5],
            is_radian=True,
            wait=False,
        )
        command_log.append(
            {
                "t_target": target,
                "t_sent": sent_t,
                "record_t": target,
                "phase": "initial_hold" if idx == 0 else "motion",
                "x": float(p_cmd[0]),
                "y": float(p_cmd[1]),
                "z": float(p_cmd[2]),
                "roll": float(p_cmd[3]),
                "pitch": float(p_cmd[4]),
                "yaw": float(p_cmd[5]),
                "return_code": int(code),
            }
        )
        if code != 0:
            nonzero_codes += 1

    print(f"playback_pose: commands={len(pose)} nonzero_return_codes={nonzero_codes}")


def playback_servo(
    arm: XArmAPI,
    t: list[float],
    pose: list[list[float]],
    t0: float,
    command_log: list[dict[str, float | int]],
    joint_limit_rad: float,
    safety_every: int,
    joint_limit_mode: str,
) -> None:
    nonzero_codes = 0
    for idx, p_cmd in enumerate(pose):
        target = t[idx]
        while (time.perf_counter() - t0) < target:
            time.sleep(0.001)
        if idx % safety_every == 0:
            if joint_limit_mode == "wrapped":
                _enforce_joint_soft_limit_wrapped(arm, joint_limit_rad, idx)
            else:
                _enforce_joint_soft_limit(arm, joint_limit_rad, idx)
        sent_t = time.perf_counter() - t0
        code = arm.set_servo_cartesian(p_cmd, is_radian=True)
        command_log.append(
            {
                "t_target": target,
                "t_sent": sent_t,
                "record_t": target,
                "phase": "initial_hold" if idx == 0 else "motion",
                "x": float(p_cmd[0]),
                "y": float(p_cmd[1]),
                "z": float(p_cmd[2]),
                "roll": float(p_cmd[3]),
                "pitch": float(p_cmd[4]),
                "yaw": float(p_cmd[5]),
                "return_code": int(code),
            }
        )
        if code != 0:
            nonzero_codes += 1

    print(f"playback_servo: commands={len(pose)} nonzero_return_codes={nonzero_codes}")


def playback_velocity(
    arm: XArmAPI,
    t: list[float],
    twist: list[list[float]] | None,
    t0: float,
    command_log: list[dict[str, float | int]],
    joint_limit_rad: float,
    safety_every: int,
    joint_limit_mode: str,
) -> None:
    if twist is None:
        raise ValueError(
            "Cartesian velocity mode requires vx,vy,vz,vroll,vpitch,vyaw columns"
        )

    nonzero_codes = 0
    for idx, v_cmd in enumerate(twist):
        target = t[idx]
        while (time.perf_counter() - t0) < target:
            time.sleep(0.001)
        if idx % safety_every == 0:
            if joint_limit_mode == "wrapped":
                _enforce_joint_soft_limit_wrapped(arm, joint_limit_rad, idx)
            else:
                _enforce_joint_soft_limit(arm, joint_limit_rad, idx)
        sent_t = time.perf_counter() - t0
        code = arm.vc_set_cartesian_velocity(v_cmd, is_radian=True, duration=-1)
        command_log.append(
            {
                "t_target": target,
                "t_sent": sent_t,
                "record_t": target,
                "phase": "initial_hold" if idx == 0 else "motion",
                "vx": float(v_cmd[0]),
                "vy": float(v_cmd[1]),
                "vz": float(v_cmd[2]),
                "vroll": float(v_cmd[3]),
                "vpitch": float(v_cmd[4]),
                "vyaw": float(v_cmd[5]),
                "return_code": int(code),
            }
        )
        if code != 0:
            nonzero_codes += 1

    arm.vc_set_cartesian_velocity([0.0] * 6, is_radian=True, duration=-1)
    print(f"playback_velocity: commands={len(twist)} nonzero_return_codes={nonzero_codes}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay Cartesian trajectories on xArm using firmware IK"
    )
    p.add_argument("--ip", default="127.0.0.1", help="Robot IP (sim: 127.0.0.1)")
    p.add_argument(
        "--trajectory",
        required=True,
        help="CSV file with t,x,y,z,roll,pitch,yaw[,vx,vy,vz,vroll,vpitch,vyaw]",
    )
    p.add_argument(
        "--control",
        choices=["pose", "servo", "velocity"],
        default="servo",
        help="Cartesian control mode",
    )
    p.add_argument(
        "--report-dir",
        default=None,
        help="Directory to save commanded/measured data and analysis outputs",
    )
    p.add_argument(
        "--feedback-sample-period",
        type=float,
        default=0.0,
        help="Seconds between feedback samples, 0 means max possible rate",
    )
    p.add_argument(
        "--analyze-report",
        action="store_true",
        help="Run error analysis and save figures after playback",
    )
    p.add_argument(
        "--joint-limit-deg",
        type=float,
        default=365.0,
        help="Abort playback if any |joint angle| exceeds this soft limit in degrees",
    )
    p.add_argument(
        "--joint-safety-every",
        type=int,
        default=1,
        help="Check joint soft limits every N commands",
    )
    p.add_argument(
        "--max-orientation-step-deg",
        type=float,
        default=8.0,
        help="Clamp per-command roll/pitch/yaw step to avoid wrap-induced IK jumps",
    )
    p.add_argument(
        "--start-hold-seconds",
        type=float,
        default=0.5,
        help="Seconds to maintain the initial trajectory point at the start of the recorded motion",
    )
    p.add_argument(
        "--joint-limit-mode",
        choices=["raw", "wrapped"],
        default="raw",
        help="Use raw reported joint angle or wrapped principal angle for soft-limit checks",
    )
    p.add_argument(
        "--default-orientation-rpy",
        default="",
        help="Optional fallback roll,pitch,yaw in radians when trajectory omits orientation",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traj = load_cartesian_trajectory_csv(args.trajectory)
    joint_limit_rad = math.radians(max(args.joint_limit_deg, 1.0))
    safety_every = max(int(args.joint_safety_every), 1)
    max_orientation_step_rad = math.radians(max(args.max_orientation_step_deg, 0.1))
    joint_limit_mode = args.joint_limit_mode
    start_hold_seconds = max(float(args.start_hold_seconds), 0.0)
    default_orientation_rpy = _parse_default_orientation_rpy(args.default_orientation_rpy)
    report_dir = Path(args.report_dir) if args.report_dir else None
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)

    arm = connect_arm(args.ip)
    command_log: list[dict[str, float | int]] = []
    sampler = FeedbackSampler(arm, sample_period_s=args.feedback_sample_period)
    start_code = -1
    end_code = -1
    start_pose: list[float] = []
    end_pose: list[float] = []
    try:
        start_code, start_pose = arm.get_position(is_radian=True)
        print(f"start_pose_code={start_code} start_pose={start_pose}")
        if default_orientation_rpy is not None:
            fallback_rpy = default_orientation_rpy
            orientation_fallback_source = "cli_default"
        else:
            fallback_rpy = [float(start_pose[3]), float(start_pose[4]), float(start_pose[5])]
            orientation_fallback_source = "start_pose"

        pose_with_orientation = _resolve_missing_orientation(traj.pose, fallback_rpy)
        prepared_pose = _prepare_continuous_orientation_trajectory(
            pose_with_orientation,
            start_pose,
            max_orientation_step_rad=max_orientation_step_rad,
        )
        recording_t = _build_recording_timeline(traj.t, start_hold_seconds)
        set_mode_for_control(arm, args.control)
        sampler.start()
        t0 = sampler.t0
        if args.control == "pose":
            playback_pose(
                arm,
                recording_t,
                prepared_pose,
                t0=t0,
                command_log=command_log,
                joint_limit_rad=joint_limit_rad,
                safety_every=safety_every,
                joint_limit_mode=joint_limit_mode,
            )
        elif args.control == "servo":
            playback_servo(
                arm,
                recording_t,
                prepared_pose,
                t0=t0,
                command_log=command_log,
                joint_limit_rad=joint_limit_rad,
                safety_every=safety_every,
                joint_limit_mode=joint_limit_mode,
            )
        else:
            playback_velocity(
                arm,
                recording_t,
                traj.twist,
                t0=t0,
                command_log=command_log,
                joint_limit_rad=joint_limit_rad,
                safety_every=safety_every,
                joint_limit_mode=joint_limit_mode,
            )
        sampler.stop()
        end_code, end_pose = arm.get_position(is_radian=True)
        print(f"end_pose_code={end_code} end_pose={end_pose}")
    finally:
        if sampler._thread is not None:
            sampler.stop()
        arm.disconnect()

    if report_dir is not None:
        commanded_path = report_dir / "commanded_cartesian.csv"
        measured_path = report_dir / "measured_cartesian.csv"
        summary_path = report_dir / "run_summary.json"

        pd.DataFrame(command_log).to_csv(commanded_path, index=False)
        pd.DataFrame(sampler.samples).to_csv(measured_path, index=False)

        summary = {
            "trajectory": args.trajectory,
            "control": args.control,
            "ip": args.ip,
            "feedback_sample_period": args.feedback_sample_period,
            "command_count": len(command_log),
            "feedback_count": len(sampler.samples),
            "feedback_error_count": sampler.error_count,
            "joint_limit_deg": args.joint_limit_deg,
            "joint_safety_every": safety_every,
            "joint_limit_mode": joint_limit_mode,
            "max_orientation_step_deg": args.max_orientation_step_deg,
            "start_hold_seconds": start_hold_seconds,
            "orientation_source": traj.orientation_source,
            "orientation_fallback_source": orientation_fallback_source,
            "default_orientation_rpy": default_orientation_rpy,
            "start_pose_code": start_code,
            "start_pose": start_pose,
            "end_pose_code": end_code,
            "end_pose": end_pose,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_report: {report_dir}")
        print(f"saved_commanded: {commanded_path}")
        print(f"saved_measured: {measured_path}")
        print(f"saved_summary: {summary_path}")

        if args.analyze_report:
            analysis = analyze_cartesian_error(
                commanded_path=commanded_path,
                measured_path=measured_path,
                out_dir=report_dir,
            )
            print(f"saved_error_figure: {analysis['figure_path']}")
            print(f"saved_error_csv: {analysis['error_csv_path']}")
            print(f"saved_error_metrics: {analysis['metrics_json_path']}")


if __name__ == "__main__":
    main()
