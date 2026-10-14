"""Glue used by playback.py and cartesian_playback.py to run all safety layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from . import firmware_fence
from .config import SafetyConfig, load_safety_config
from .geometry import Workcell
from .kinematics import fk_pose
from .preprocess import (
    SafetyRejected,
    ValidationResult,
    print_banner,
    robot_tcp_offset,
    validate_cartesian_trajectory,
    validate_joint_trajectory,
    write_report,
    write_sanitized_cartesian,
)
from .runtime import CommandGuard, SafetyMonitor, SafetyStop

EXIT_REJECTED = 2
EXIT_SAFETY_STOP = 3


def add_safety_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("safety walls (see docs/safety_walls.md)")
    g.add_argument("--safety-config", default=None, help="Safety config YAML (default config/safety.yaml)")
    g.add_argument("--safety-policy", choices=["reject", "clip"], default=None,
                   help="Override preprocess.policy from the config")
    g.add_argument("--disable-safety-preprocess", action="store_true",
                   help="DEMO ONLY: skip the pre-run wall check (still evaluated and logged)")
    g.add_argument("--disable-runtime-guard", action="store_true",
                   help="DEMO ONLY: monitor + command check only log, they never stop the arm")
    g.add_argument("--disable-firmware-fence", action="store_true",
                   help="Do not set the firmware TCP safety boundary for this run")
    g.add_argument("--safety-out-dir", default=None,
                   help="Where to write safety reports/logs (default: --report-dir if given)")


def controller_error_title(code: int) -> str:
    """C<code> meaning from the SDK's controller error table (e.g. C22 self-collision, C35 safety boundary)."""
    try:
        from xarm.core.config.x_code import ControllerError

        return f"C{code} " + ControllerError(code).title["en"]
    except Exception:
        return f"C{code}" if code else "no error code"


class SafetySession:
    def __init__(self, args: argparse.Namespace, arm, ip: str, out_dir: str | Path | None = None) -> None:
        self.cfg: SafetyConfig = load_safety_config(args.safety_config)
        if args.safety_policy:
            self.cfg.preprocess.policy = args.safety_policy
        self.arm = arm
        self.ip = ip
        self.preprocess_on = self.cfg.preprocess.enabled and not args.disable_safety_preprocess
        self.runtime_on = self.cfg.runtime.enabled and not args.disable_runtime_guard
        self.fence_on = self.cfg.firmware_fence.enabled and not args.disable_firmware_fence
        self.out_dir = Path(out_dir or args.safety_out_dir) if (out_dir or args.safety_out_dir) else None
        self.monitor: SafetyMonitor | None = None
        self.guard: CommandGuard | None = None
        self.fence_prev: dict | None = None
        self.validation: ValidationResult | None = None
        self.validated_path = None  # (poses, joints per pose) accepted by the preprocessing
        self.abort_reason: str | None = None
        self.t0 = time.perf_counter()

        problems = Workcell(self.cfg).static_check()
        if problems:
            raise SystemExit("SAFETY CONFIG ERROR: " + "; ".join(problems))
        print(f"safety: config={self.cfg.source_path} preprocess={'ON' if self.preprocess_on else 'OFF'} "
              f"(policy={self.cfg.preprocess.policy}) runtime_guard={'ON' if self.runtime_on else 'OFF'} "
              f"firmware_fence={'ON' if self.fence_on else 'OFF'}", flush=True)
        if not self.preprocess_on:
            print_banner("WARNING: SAFETY PREPROCESSING DISABLED (--disable-safety-preprocess / config). "
                         "Trajectory is NOT checked against the walls before moving.")
        if not self.runtime_on:
            print_banner("WARNING: RUNTIME SAFETY GUARD DISABLED (--disable-runtime-guard / config). "
                         "Nothing will stop the arm at the walls.")

        code, q = arm.get_servo_angle(is_radian=True)
        if code != 0:
            raise RuntimeError(f"safety: cannot read joints (code={code})")
        self.q_start = np.asarray(q[:7], dtype=float)
        self.tcp_offset = robot_tcp_offset(arm, self.cfg)

    # ---------------- layer 1 ----------------
    def _handle_result(self, res: ValidationResult) -> None:
        self.validation = res
        if self.out_dir is not None:
            write_report(res, self.out_dir / "safety_preprocess_report.json")
        if res.ok:
            if res.action != "clip" or not self.preprocess_on:
                print(res.message, flush=True)
            return
        if self.preprocess_on:
            print_banner(res.message)
            raise SafetyRejected(res.message, res)
        print_banner("PREPROCESSING DISABLED - would have rejected: " + res.message)

    def preprocess_joint(self, t, q, dq, control: str) -> None:
        res = validate_joint_trajectory(self.cfg, t, q, dq, control, q_start=self.q_start)
        self._handle_result(res)

    def preprocess_cartesian(self, t, poses, twist, control: str):
        """Returns the poses to execute (clipped if policy=clip and enabled)."""
        res = validate_cartesian_trajectory(self.cfg, t, poses, self.q_start, self.tcp_offset, twist, control,
                                            include_approach=True)
        self._handle_result(res)
        if res.ok and self.preprocess_on and res.validated_q is not None and control in ("servo", "pose"):
            self.validated_path = (res.sanitized_pose if res.action == "clip" else poses, res.validated_q)
        if res.action == "clip" and self.preprocess_on:
            print_banner(res.message)
            if self.out_dir is not None:
                write_sanitized_cartesian(t, res.sanitized_pose, self.out_dir / "safety_sanitized_trajectory.csv", twist)
            return res.sanitized_pose
        return poses

    # ---------------- layer 2 + 3 ----------------
    def start_runtime(self, t0: float | None = None) -> None:
        if self.fence_on:
            try:
                self.fence_prev = firmware_fence.apply_fence(self.arm, self.cfg)
            except Exception as e:
                print(f"WARNING: firmware fence could not be set: {e}", flush=True)
        self.t0 = t0 if t0 is not None else time.perf_counter()
        self.monitor = SafetyMonitor(self.cfg, self.ip, enforce=self.runtime_on)
        self.monitor.start(self.t0)
        self.guard = CommandGuard(self.cfg, self.monitor, enforce=self.runtime_on, q_start=self.q_start,
                                  tcp_offset=self.tcp_offset,
                                  start_pose=fk_pose(self.q_start, tcp_offset=self.tcp_offset),
                                  validated_path=self.validated_path)

    def check_abort(self, idx: int) -> None:
        """Call before every command: aborts on monitor trip or a firmware stop."""
        if self.monitor is not None and self.monitor.tripped.is_set():
            raise SafetyStop(f"runtime guard stopped the arm before command {idx}: "
                             f"{self.monitor.trip_info.get('reason') if self.monitor.trip_info else ''}")
        if self.arm.error_code != 0 or self.arm.state == 4:
            raise SafetyStop(f"robot stopped by firmware before command {idx}: state={self.arm.state} "
                             f"error_code={self.arm.error_code} ({controller_error_title(self.arm.error_code)})")

    def wait_until_still(self, timeout_s: float = 15.0, still_s: float = 0.3) -> None:
        """Keep monitoring until the arm has stopped moving (queued moves finish)."""
        if self.monitor is None:
            return
        t_end = time.perf_counter() + timeout_s
        still_since = None
        while time.perf_counter() < t_end and not self.monitor.tripped.is_set():
            s = self.monitor.samples[-1] if self.monitor.samples else None
            speed = s[12] if s is not None else 0.0
            if speed < 1.0:
                still_since = still_since or time.perf_counter()
                if time.perf_counter() - still_since > still_s:
                    return
            else:
                still_since = None
            time.sleep(0.02)

    def close(self) -> dict:
        summary: dict = {
            "preprocess_enabled": self.preprocess_on,
            "runtime_guard_enabled": self.runtime_on,
            "firmware_fence_enabled": self.fence_on,
            "policy": self.cfg.preprocess.policy,
            "validation_action": self.validation.action if self.validation else None,
            "validation_message": self.validation.message if self.validation else None,
            "abort_reason": self.abort_reason,
            "q_start_deg": np.degrees(self.q_start).round(3).tolist(),
        }
        if self.monitor is not None:
            if self.monitor.tripped.is_set():
                time.sleep(0.8)  # keep logging so the plots show the arm at rest after the stop
            self.monitor.stop()
            if self.out_dir is not None:
                summary["monitor"] = self.monitor.save(self.out_dir)
            self.monitor.close()
            ms = self.monitor
            print(f"safety monitor: min clearance {ms.min_clearance[0]:.1f} mm ({ms.min_clearance[1]} vs "
                  f"{ms.min_clearance[2]}), samples={len(ms.samples)}, tripped={ms.tripped.is_set()}", flush=True)
        if self.guard is not None:
            summary["command_guard"] = self.guard.summary()
        if self.fence_prev is not None and self.cfg.firmware_fence.restore_on_exit:
            try:
                firmware_fence.restore_fence(self.arm, self.fence_prev)
            except Exception as e:
                print(f"WARNING: could not restore firmware fence: {e}. Reset with "
                      f"'python -m src.xarm_pipeline.safety.firmware_fence --ip {self.ip} reset'", flush=True)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "safety_run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary


__all__ = ["EXIT_REJECTED", "EXIT_SAFETY_STOP", "SafetyRejected", "SafetySession", "SafetyStop", "add_safety_args"]
