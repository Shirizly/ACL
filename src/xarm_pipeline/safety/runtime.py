"""Layer 2: runtime guard.

* :class:`SafetyMonitor` - thread with its OWN TCP connection to the robot that
  polls the measured joint angles at ``monitor_rate_hz``, computes FK of all
  capsules, estimates each point's velocity and stops the arm when the
  *predicted* clearance (clearance - outward speed * (latency + stopping time))
  falls below ``stop_margin_mm``.  Stop = ``set_state(4)`` sent on the
  monitor's own socket (measured in the sim: halts within one 10 ms report
  sample; zeroing the velocity alone took ~370 ms / 33 mm), followed by a zero
  velocity command in velocity modes and ``emergency_stop()`` to confirm.
  With ``enforce=False`` it only logs (used for the "no safeguards" demo).

* :class:`CommandGuard` - checks every command before it is sent: FK of the
  joint target (dense segment from the previous target), local IK + full-arm
  check of Cartesian targets, and a lookahead prediction for velocity
  commands.  A failing command trips the monitor (stop) and raises
  :class:`SafetyStop` so playback aborts.
"""
from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd

from .config import SafetyConfig
from .geometry import CAPSULES, END_EFFECTOR_CAPSULES, MOVING_CAPSULES, WALL_NAMES, Workcell
from .kinematics import IKFailure, fk_pose, ik_step_to
from .preprocess import SafetyChecker, _pose_step, interpolate_pose, print_banner

# outward unit direction of each wall, in world frame
_WALL_OUT = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], dtype=float)


class SafetyStop(RuntimeError):
    """Playback aborted by the runtime guard."""


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig, ip: str, enforce: bool = True, arm=None) -> None:
        self.cfg = cfg
        self.wc = Workcell(cfg)
        self.enforce = enforce
        self.rate_hz = float(cfg.runtime.monitor_rate_hz)
        self.lookahead_s = cfg.runtime.lookahead_s
        self.stop_margin = cfg.runtime.stop_margin_mm
        if arm is None:
            from xarm.wrapper import XArmAPI

            # Dedicated connection: its command socket is never blocked by the
            # playback thread, so the stop command goes out immediately.
            arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False, report_type="real")
        self.arm = arm
        self.tripped = threading.Event()
        self.trip_info: dict | None = None
        self.would_trip_info: dict | None = None
        self.samples: list[tuple] = []
        self.latest_q: np.ndarray | None = None
        self._bc_hist: deque = deque()  # (t, end effector clearance to the base column)
        self.latest_t = 0.0
        self.read_errors = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = time.perf_counter()
        self.t0_epoch = time.time()
        self._hist: deque = deque()
        self._radii = np.array([cfg.radii_mm[n] for n, _, _ in MOVING_CAPSULES])
        self._ci = np.array([i for _, i, _ in MOVING_CAPSULES])
        self._cj = np.array([j for _, _, j in MOVING_CAPSULES])
        self.min_clearance = (math.inf, "", "")
        self.loop_dt_max = 0.0

    # ---------------- lifecycle ----------------
    def start(self, t0: float | None = None) -> None:
        self._t0 = t0 if t0 is not None else time.perf_counter()
        self.t0_epoch = time.time() - (time.perf_counter() - self._t0)
        # Prime with one reading so CommandGuard has the actual state.
        self._poll_once()
        self._thread = threading.Thread(target=self._run, name="safety-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop()
        try:
            self.arm.disconnect()
        except Exception:
            pass

    # ---------------- core ----------------
    def evaluate(self, q, v_world: np.ndarray | None = None):
        """Return (clearance (ncaps,6), predicted (ncaps,6), world points)."""
        Pw = self.wc.points_world(np.asarray(q, dtype=float)[None])[0]
        D = self.wc.wall_distances(Pw)  # (9,6)
        C = np.minimum(D[self._ci], D[self._cj]) - self._radii[:, None]
        if v_world is None:
            return C, C, Pw
        vout = v_world @ _WALL_OUT.T  # (9,6) outward speed per point/wall
        Dp = D - np.maximum(vout, 0.0) * self.lookahead_s
        Cp = np.minimum(Dp[self._ci], Dp[self._cj]) - self._radii[:, None]
        return C, Cp, Pw

    def _poll_once(self) -> bool:
        code, q = self.arm.get_servo_angle(is_radian=True)
        t = time.perf_counter() - self._t0
        if code != 0 or q is None or len(q) < 7:
            self.read_errors += 1
            return False
        q = np.asarray(q[:7], dtype=float)
        Pw = self.wc.points_world(q[None])[0]
        self._hist.append((t, Pw))
        win = self.cfg.runtime.velocity_window_s
        while len(self._hist) > 2 and t - self._hist[1][0] >= win:
            self._hist.popleft()
        v = None
        t_old, P_old = self._hist[0]
        if t - t_old > 1e-4:
            v = (Pw - P_old) / (t - t_old)
        C, Cp, _ = self.evaluate(q, v)
        k, w = np.unravel_index(int(np.argmin(Cp)), Cp.shape)
        cmin = float(C.min())
        if cmin < self.min_clearance[0]:
            kc, wc_ = np.unravel_index(int(np.argmin(C)), C.shape)
            self.min_clearance = (cmin, MOVING_CAPSULES[kc][0], WALL_NAMES[wc_])
        speed = float(np.max(np.linalg.norm(v, axis=1))) if v is not None else 0.0
        self.samples.append((t, *q.tolist(), cmin, float(Cp[k, w]), MOVING_CAPSULES[k][0], WALL_NAMES[w], speed,
                             int(self.tripped.is_set())))
        self.latest_q = q
        self.latest_t = t
        # The robot's own base column, a fixed obstacle for the end effector (no wall normal: use the
        # fastest point speed for the lookahead, conservatively).
        bc_raw = float(self.wc.base_column_slack(Pw[None])[0]) + self.cfg.safety_margin_mm
        # lookahead on the rate the clearance itself shrinks (moving past the column at speed is fine)
        self._bc_hist.append((t, bc_raw))
        while len(self._bc_hist) > 2 and t - self._bc_hist[1][0] >= win:
            self._bc_hist.popleft()
        tb, b0 = self._bc_hist[0]
        approach = max(0.0, (b0 - bc_raw) / (t - tb)) if t - tb > 1e-4 else 0.0
        ee_speed = approach
        if bc_raw - approach * self.lookahead_s < self.stop_margin and not self.tripped.is_set():
            self.trip(f"monitor: end effector {bc_raw:.1f} mm from the robot's own body (base column / arm links) "
                      f"(closing at {ee_speed:.0f} mm/s, stop margin {self.stop_margin:.1f} mm)",
                      {"t": t, "source": "monitor", "link": "end effector", "wall": "robot body",
                       "clearance_mm": bc_raw, "max_point_speed_mm_s": ee_speed, "q_deg": np.degrees(q).round(3).tolist()})
        if float(Cp[k, w]) < self.stop_margin and not self.tripped.is_set():
            info = {
                "t": t,
                "source": "monitor",
                "link": MOVING_CAPSULES[k][0],
                "wall": WALL_NAMES[w],
                "clearance_mm": float(C[k, w]),
                "predicted_clearance_mm": float(Cp[k, w]),
                "max_point_speed_mm_s": speed,
                "q_deg": np.degrees(q).round(3).tolist(),
            }
            msg = (f"monitor: link '{info['link']}' predicted clearance to {info['wall']} "
                   f"{info['predicted_clearance_mm']:.1f} mm < stop margin {self.stop_margin:.1f} mm "
                   f"(actual {info['clearance_mm']:.1f} mm, speed {speed:.0f} mm/s)")
            self.trip(msg, info)
        return True

    def _run(self) -> None:
        period = 1.0 / self.rate_hz
        next_t = time.perf_counter()
        last_ok = time.perf_counter()
        while not self._stop.is_set():
            t_start = time.perf_counter()
            ok = False
            try:
                ok = self._poll_once()
            except Exception as e:  # never let the monitor die silently
                print(f"safety monitor exception: {e}", flush=True)
            now = time.perf_counter()
            if ok:
                last_ok = now
            elif now - last_ok > 0.1 and not self.tripped.is_set():
                self.trip("monitor: lost joint feedback for > 100 ms", {"t": now - self._t0, "source": "monitor"})
            self.loop_dt_max = max(self.loop_dt_max, now - t_start)
            next_t += period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    def trip(self, reason: str, info: dict | None = None) -> None:
        info = dict(info or {})
        info["reason"] = reason
        info.setdefault("t", time.perf_counter() - self._t0)
        if not self.enforce:
            if self.would_trip_info is None:
                self.would_trip_info = info
                print_banner(f"RUNTIME GUARD DISABLED - would have stopped the arm here: {reason}")
            return
        if self.tripped.is_set():
            return
        t_call = time.perf_counter()
        self.arm.set_state(4)  # fastest halt measured in the sim
        try:
            mode = self.arm.mode
            if mode == 4:
                self.arm.vc_set_joint_velocity([0.0] * 7, is_radian=True)
            elif mode == 5:
                self.arm.vc_set_cartesian_velocity([0.0] * 6, is_radian=True, duration=-1)
        except Exception:
            pass
        self.arm.emergency_stop()  # loops set_state(4) until state 4 is confirmed
        info["stop_command_latency_ms"] = (time.perf_counter() - t_call) * 1e3
        self.trip_info = info
        self.tripped.set()
        print_banner(f"SAFETY STOP: {reason}")

    # ---------------- output ----------------
    def save(self, out_dir: str | Path, prefix: str = "safety") -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cols = ["t"] + [f"q{i}" for i in range(1, 8)] + [
            "min_clearance_mm", "min_predicted_clearance_mm", "pred_link", "pred_wall", "max_point_speed_mm_s", "tripped"]
        pd.DataFrame(self.samples, columns=cols).to_csv(out_dir / f"{prefix}_monitor_log.csv", index=False)
        ts = [s[0] for s in self.samples]
        summary = {
            "enforce": self.enforce,
            "t0_epoch": self.t0_epoch,
            "monitor_rate_hz_config": self.rate_hz,
            "monitor_rate_hz_measured": (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None,
            "monitor_loop_max_ms": self.loop_dt_max * 1e3,
            "read_errors": self.read_errors,
            "stop_margin_mm": self.stop_margin,
            "lookahead_s": self.lookahead_s,
            "min_clearance_mm": self.min_clearance[0] if self.samples else None,
            "min_clearance_link": self.min_clearance[1],
            "min_clearance_wall": self.min_clearance[2],
            "tripped": self.tripped.is_set(),
            "trip": self.trip_info,
            "would_have_tripped": self.would_trip_info,
        }
        (out_dir / f"{prefix}_monitor_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary


def predict_monitor_margin(wc: Workcell, cfg, t, Q) -> tuple[float, int, str]:
    """Simulate the runtime monitor along a planned path: joints Q (N,7) at times t (N,).

    For every capsule/wall clearance and for the end effector's clearance to the robot's base column, the
    monitor stops when  clearance - closing_speed * lookahead < stop_margin. Returns (worst value of
    clearance - closing*lookahead - stop_margin, sample index, what): < 0 means the monitor will stop it."""
    t = np.asarray(t, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if len(t) < 2:
        return float("inf"), 0, ""
    look = cfg.runtime.latency_s + cfg.runtime.stopping_time_s
    C, Pw = wc.clearance_for_q(Q, MOVING_CAPSULES)                     # (N, caps, 6)
    B = wc.base_column_slack(Pw) + cfg.safety_margin_mm                 # (N,)
    series = np.concatenate([C.reshape(len(t), -1), B[:, None]], axis=1)
    rate = np.gradient(series, t, axis=0)
    pred = series + np.minimum(rate, 0.0) * look - cfg.runtime.stop_margin_mm
    n, k = np.unravel_index(int(np.argmin(pred)), pred.shape)
    if k == series.shape[1] - 1:
        what = "end effector vs robot body"
    else:
        cap, w = divmod(k, 6)
        what = f"{MOVING_CAPSULES[cap][0]} vs {WALL_NAMES[w]}"
    return float(pred[n, k]), int(n), what


class CommandGuard:
    """Per-command check (before sending). Uses ``safety_margin_mm``."""

    def __init__(self, cfg: SafetyConfig, monitor: SafetyMonitor | None, enforce: bool = True,
                 q_start=None, tcp_offset=None, start_pose=None, validated_path=None) -> None:
        self.cfg = cfg
        # Poses already accepted by the preprocessing, with their predicted joints: a command equal to the
        # validated sample needs no IK here (the monitor still watches the actual arm state).
        self.validated_poses = None if validated_path is None else np.asarray(validated_path[0], dtype=float)
        self.validated_q = None if validated_path is None else validated_path[1]
        self.validated_hits = 0
        self.chk = SafetyChecker(cfg)
        self.monitor = monitor
        self.enforce = enforce and cfg.runtime.command_check
        self.required = cfg.safety_margin_mm
        self.tcp_offset = tcp_offset
        self.last_q = None if q_start is None else np.asarray(q_start, dtype=float)
        self.last_pose = start_pose
        self.rejections = 0
        self.max_check_ms = 0.0

    def _actual_q(self):
        if self.monitor is not None and self.monitor.latest_q is not None:
            return self.monitor.latest_q
        return self.last_q

    def _fail(self, idx: int, msg: str, info: dict) -> None:
        self.rejections += 1
        full = f"command {idx} refused: {msg}"
        info = dict(info, command_index=idx, source="command_check")
        if not self.enforce:
            if self.rejections == 1:
                print_banner(f"RUNTIME COMMAND CHECK DISABLED - would refuse: {full}")
            return
        if self.monitor is not None:
            self.monitor.trip(full, info)
        raise SafetyStop(full)

    def _check_Q(self, Q, idx: int, label: str, capsules=MOVING_CAPSULES) -> None:
        v, _ = self.chk.wc.first_violation(Q, self.required, capsules)
        if v is not None:
            self._fail(idx, f"{label}: {v.describe()}", {"link": v.link, "wall": v.wall,
                                                       "clearance_mm": v.clearance_mm})
            return
        sl = self.chk.wc.base_column_slack(self.chk.wc.points_world(np.asarray(Q)))
        if sl.min() < 0:
            c = float(sl.min() + self.cfg.safety_margin_mm)
            self._fail(idx, f"{label}: end effector {c:.1f} mm from the robot's own body (base column / arm links) "
                            f"(required >= {self.cfg.safety_margin_mm:.1f} mm)",
                       {"link": "end effector", "wall": "robot body", "clearance_mm": c})

    def _timed(fn):  # noqa: N805
        def wrapper(self, *a, **k):
            t = time.perf_counter()
            try:
                return fn(self, *a, **k)
            finally:
                self.max_check_ms = max(self.max_check_ms, (time.perf_counter() - t) * 1e3)
        return wrapper

    @_timed
    def check_joint_target(self, q_cmd, idx: int) -> None:
        q_cmd = np.asarray(q_cmd, dtype=float)
        prev = self.last_q if self.last_q is not None else self._actual_q()
        Q, _ = self.chk.densify_joint_path(q_cmd[None], prev)
        self._check_Q(Q, idx, "joint target")
        self.last_q = q_cmd

    @_timed
    def check_joint_velocity(self, dq, dt_next: float, idx: int) -> None:
        q_now = self._actual_q()
        if q_now is None:
            return
        horizon = max(dt_next, 0.0) + self.cfg.runtime.lookahead_s
        q_pred = q_now + np.asarray(dq, dtype=float) * horizon
        Q, _ = self.chk.densify_joint_path(q_pred[None], q_now)
        self._check_Q(Q, idx, f"joint velocity (predicted {horizon * 1e3:.0f} ms ahead)")

    @_timed
    def check_cartesian_pose(self, pose, idx: int) -> None:
        if (self.validated_poses is not None and idx < len(self.validated_poses)
                and np.allclose(np.asarray(pose, dtype=float), self.validated_poses[idx], atol=1e-6)):
            self.last_q = np.asarray(self.validated_q[idx], dtype=float)
            self.last_pose = [float(v) for v in pose]
            self.validated_hits += 1
            return
        seed = self._actual_q()
        if seed is None:
            return
        pose = [float(v) for v in pose]
        from_pose = fk_pose(seed, tcp_offset=self.tcp_offset)
        dp, da = _pose_step(from_pose, pose)
        n = max(1, int(math.ceil(max(dp / self.cfg.preprocess.interp_resolution_mm,
                                     da / self.cfg.preprocess.interp_resolution_deg))))
        q = seed
        Q = []
        try:
            for i in range(1, n + 1):
                q = ik_step_to(interpolate_pose(from_pose, pose, i / n) if i < n else pose, q,
                               tcp_offset=self.tcp_offset)
                Q.append(q)
        except IKFailure as e:
            self._fail(idx, f"Cartesian target unreachable: {e}", {})
            return
        self._check_Q(np.asarray(Q), idx, "Cartesian target (local IK)")
        self.last_q = q
        self.last_pose = pose

    @_timed
    def check_cartesian_velocity(self, twist, dt_next: float, idx: int) -> None:
        q_now = self._actual_q()
        if q_now is None:
            return
        horizon = max(dt_next, 0.0) + self.cfg.runtime.lookahead_s
        Pw = self.chk.wc.points_world(np.asarray(q_now)[None])
        shift_w = self.chk.wc.R @ (np.asarray(twist[:3], dtype=float) * horizon)
        # Translate the end-effector points (rigid with the flange) by v*horizon.
        ee_idx = sorted({i for n, i, j in CAPSULES if n in END_EFFECTOR_CAPSULES} |
                        {j for n, i, j in CAPSULES if n in END_EFFECTOR_CAPSULES})
        Pw[0, ee_idx] += shift_w
        D = self.chk.wc.wall_distances(Pw[0])
        for name, i, j in CAPSULES:
            if name not in END_EFFECTOR_CAPSULES:
                continue
            c = np.minimum(D[i], D[j]) - self.cfg.radii_mm[name]
            w = int(np.argmin(c))
            if c[w] < self.required:
                self._fail(idx, f"Cartesian velocity (predicted {horizon * 1e3:.0f} ms ahead): link '{name}' "
                                f"vs {WALL_NAMES[w]}: clearance {c[w]:.1f} mm < {self.required:.1f} mm",
                           {"link": name, "wall": WALL_NAMES[w], "clearance_mm": float(c[w])})
                return

    def summary(self) -> dict:
        return {"enforce": self.enforce, "rejections": self.rejections, "max_check_ms": self.max_check_ms,
                "validated_fast_path_commands": self.validated_hits}
