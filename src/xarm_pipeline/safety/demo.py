"""Safety-wall demos in the simulator (headless video + plots).

    python -m src.xarm_pipeline.safety.demo                # all runs
    python -m src.xarm_pipeline.safety.demo --only tcp_wall_runtime_guard

Requires the simulator running (scripts/sim/up.sh + firmware started) and
Playwright Chromium.  Writes to outputs/safety_demo/<run>/ and prints a
summary (also outputs/safety_demo/demo_summary.json).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .config import DEFAULT_CONFIG_PATH, load_safety_config
from .preprocess import DEFAULT_CARTESIAN_SEED_DEG

ROOT = Path(__file__).resolve().parents[3]
PY = sys.executable
TRAJ_DIR = ROOT / "data" / "trajectories"
OUT = ROOT / "outputs" / "safety_demo"
SAFE_HOME_DEG = list(DEFAULT_CARTESIAN_SEED_DEG)  # flange at (300, 0, 300) pointing down

TCP_TRAJ = TRAJ_DIR / "safety_demo_tcp_through_wall.csv"
ELBOW_TRAJ = TRAJ_DIR / "safety_demo_elbow_through_wall_joint.csv"


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------
def _smooth(s):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(s, 0, 1))


def make_tcp_trajectory(path: Path = TCP_TRAJ, dt: float = 0.03) -> Path:
    """Flange pointing down at (300,0,300) -> (300,560,300) -> back: the tool
    goes ~200 mm through the +y wall (y = +400)."""
    t = np.arange(0.0, 9.0 + 1e-9, dt)
    y = np.where(t < 1, 0.0, np.where(t < 5, 560 * _smooth((t - 1) / 4),
                 np.where(t < 6, 560.0, 560 * (1 - _smooth((t - 6) / 3)))))
    df = pd.DataFrame({"t": t, "x": 300.0, "y": y, "z": 300.0, "roll": math.pi, "pitch": 0.0, "yaw": 0.0})
    df["vx"] = 0.0
    df["vy"] = np.gradient(y, t)
    df["vz"] = 0.0
    df["vroll"] = 0.0
    df["vpitch"] = 0.0
    df["vyaw"] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def make_elbow_trajectory(path: Path = ELBOW_TRAJ, dt: float = 0.05) -> Path:
    """J1=90, J4=60, J6=120 fixed; J2 swings -40 -> -100 -> -40 deg. The elbow
    goes ~30 mm through the -y wall while the tool stays > 90 mm inside."""
    t = np.arange(0.0, 11.0 + 1e-9, dt)
    j2 = np.where(t < 1, -40.0, np.where(t < 6, -40 - 60 * _smooth((t - 1) / 5),
                  np.where(t < 7, -100.0, -100 + 60 * _smooth((t - 7) / 4))))
    q = np.zeros((len(t), 7))
    q[:, 0] = 90
    q[:, 1] = j2
    q[:, 3] = 60
    q[:, 5] = 120
    q = np.radians(q)
    df = pd.DataFrame(q, columns=[f"q{i}" for i in range(1, 8)])
    df.insert(0, "t", t)
    dq = np.gradient(q, t, axis=0)
    for i in range(7):
        df[f"dq{i + 1}"] = dq[:, i]
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# robot helpers
# ---------------------------------------------------------------------------
def _arm(ip):
    from xarm.wrapper import XArmAPI

    return XArmAPI(ip, baud_checkset=False, check_joint_limit=False)


def recover_and_home(ip: str, home_deg=SAFE_HOME_DEG) -> None:
    """clean_error, motion_enable, set_state(0), reset fence, joint move home."""
    from .firmware_fence import reset_fence

    arm = _arm(ip)
    try:
        arm.clean_error()
        arm.clean_warn()
        reset_fence(arm)
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.3)
        arm.set_servo_angle(angle=list(home_deg), speed=40, wait=True)
    finally:
        arm.disconnect()


def prep_cartesian(ip: str, traj: Path) -> None:
    from ..trajectory_io import load_cartesian_trajectory_csv

    p0 = load_cartesian_trajectory_csv(traj).pose[0]
    arm = _arm(ip)
    try:
        arm.set_mode(0)
        arm.set_state(0)
        arm.set_position(*p0, is_radian=True, speed=100, wait=True)
        time.sleep(0.3)
    finally:
        arm.disconnect()


def prep_joint(ip: str, traj: Path) -> None:
    from ..trajectory_io import load_joint_trajectory_csv

    q0 = load_joint_trajectory_csv(traj).q[0]
    arm = _arm(ip)
    try:
        arm.set_mode(0)
        arm.set_state(0)
        arm.set_servo_angle(angle=q0, is_radian=True, speed=0.7, wait=True)
        time.sleep(0.3)
    finally:
        arm.disconnect()


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
NO_SAFEGUARDS = ["--disable-safety-preprocess", "--disable-runtime-guard", "--disable-firmware-fence"]
RUNS = {
    # name: (kind, control, extra args, record?, config override, description)
    "tcp_wall_no_safeguards": ("cartesian", "servo", NO_SAFEGUARDS, True, None,
                               "TCP line through the +y wall, all safety layers disabled"),
    "tcp_wall_preprocess_reject": ("cartesian", "servo", ["--safety-policy", "reject"], False, None,
                                   "same trajectory, preprocessing policy=reject (robot must not move)"),
    "tcp_wall_preprocess_clip": ("cartesian", "servo", ["--safety-policy", "clip"], True, None,
                                 "same trajectory, preprocessing policy=clip + runtime guard + firmware fence"),
    "tcp_wall_runtime_guard": ("cartesian", "servo", ["--disable-safety-preprocess", "--disable-firmware-fence"],
                               True, None, "same trajectory, preprocessing bypassed, runtime guard ON"),
    "tcp_wall_velocity_monitor_only": ("cartesian", "velocity",
                                       ["--disable-safety-preprocess", "--disable-firmware-fence"], True,
                                       {"runtime": {"command_check": False}},
                                       "Cartesian velocity mode, preprocessing bypassed, per-command check "
                                       "off: only the monitor thread (lookahead) can stop it"),
    "tcp_wall_firmware_fence_only": ("cartesian", "servo",
                                     ["--disable-safety-preprocess", "--disable-runtime-guard"], True, None,
                                     "same trajectory, only the firmware TCP fence enabled"),
    "elbow_wall_no_safeguards": ("joint", "position", NO_SAFEGUARDS, True, None,
                                 "elbow swings through the -y wall (tool stays inside), all layers disabled"),
    "elbow_wall_preprocess_reject": ("joint", "position", [], False, None,
                                     "same joint trajectory, preprocessing ON (reject)"),
    "elbow_wall_runtime_guard": ("joint", "position", ["--disable-safety-preprocess", "--disable-firmware-fence"],
                                 True, None, "same joint trajectory, preprocessing bypassed, runtime guard ON"),
    "elbow_wall_runtime_guard_velocity": ("joint", "velocity",
                                          ["--disable-safety-preprocess", "--disable-firmware-fence"], True, None,
                                          "same joint path in joint-velocity mode (commands track real time), "
                                          "preprocessing bypassed, runtime guard ON"),
}
# Max recording length per run (s); recording stops 1.5 s after playback exits.
# (Joint position mode queues set_servo_angle moves at the SDK default joint
# speed, so it takes ~2x the CSV duration.)
RUN_SECONDS: dict[str, float] = {}


# Base safety config for the demo runs (--config); the demo trajectories were designed for the
# original test cell, tests/fixtures/safety_test.yaml.
BASE_CONFIG: Path = DEFAULT_CONFIG_PATH


def _write_override_config(run_dir: Path, override: dict) -> Path:
    raw = yaml.safe_load(BASE_CONFIG.read_text())
    for k, v in override.items():
        raw.setdefault(k, {}).update(v)
    p = run_dir / "safety_config_used.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return p


def run_one(name: str, ip: str, seconds: float) -> dict:
    kind, control, extra, record, override, desc = RUNS[name]
    run_dir = OUT / name
    if run_dir.exists():
        for f in run_dir.iterdir():
            if f.is_file():
                f.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = BASE_CONFIG
    if override:
        cfg_path = _write_override_config(run_dir, override)
    extra = extra + ["--safety-config", str(cfg_path)]
    traj = TCP_TRAJ if kind == "cartesian" else ELBOW_TRAJ

    print(f"\n=== {name}: {desc}", flush=True)
    recover_and_home(ip)
    (prep_cartesian if kind == "cartesian" else prep_joint)(ip, traj)

    module = "src.xarm_pipeline.cartesian_playback" if kind == "cartesian" else "src.xarm_pipeline.playback"
    cmd = [PY, "-m", module, "--ip", ip, "--trajectory", str(traj), "--control", control,
           "--report-dir", str(run_dir)] + extra
    if kind == "cartesian":
        # demos run the raw (wall-crossing) trajectories, not fitted clean copies
        cmd += ["--feedback-sample-period", "0.002", "--no-fit"]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    log_path = run_dir / "run_command.log"
    t_start = time.time()
    seconds = RUN_SECONDS.get(name, seconds)
    if record:
        wrapped = f"{cmd_str}; echo EXIT_CODE=$?"
        rec = subprocess.run([PY, "-m", "src.xarm_pipeline.record_web_sim", "--headless", "--seconds", str(seconds),
                              "--output-dir", str(run_dir), "--orbit-drag", "120,60", "--zoom-steps", "4",
                              "--stop-when-done", "1.5",
                              "--run-command", wrapped],
                             cwd=ROOT, capture_output=True, text=True)
        print(rec.stdout[-500:], rec.stderr[-1500:], flush=True)
        logs = sorted(run_dir.glob("*_run_command.log"))
        if logs:
            logs[-1].rename(log_path)
    else:
        with log_path.open("w") as fh:
            p = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        with log_path.open("a") as fh:
            fh.write(f"EXIT_CODE={p.returncode}\n")
    text = log_path.read_text() if log_path.exists() else ""
    exit_code = None
    for line in text.splitlines():
        if line.startswith("EXIT_CODE="):
            exit_code = int(line.split("=", 1)[1])

    # Robot state right after the run (before recovery).
    arm = _arm(ip)
    time.sleep(0.3)
    post_state = {"state": arm.get_state()[1], "error_code": arm.get_err_warn_code()[1][0]}
    arm.disconnect()
    recover_and_home(ip)

    result = {"name": name, "description": desc, "exit_code": exit_code, "robot_after_run": post_state,
              "wall_clock_s": time.time() - t_start}
    for fname, key in (("safety_run_summary.json", "safety"), ("safety_preprocess_report.json", "preprocess")):
        f = run_dir / fname
        if f.exists():
            result[key] = json.loads(f.read_text())
    mon = (result.get("safety") or {}).get("monitor") or {}
    result["measured_min_clearance_mm"] = mon.get("min_clearance_mm")
    result["measured_min_clearance_where"] = f"{mon.get('min_clearance_link')} vs {mon.get('min_clearance_wall')}"
    result["stopped_by"] = (mon.get("trip") or {}).get("source")
    result["stop_reason"] = (mon.get("trip") or {}).get("reason")
    result["would_have_stopped"] = (mon.get("would_have_tripped") or {}).get("reason")

    # plots / videos
    if (run_dir / "safety_monitor_log.csv").exists():
        from .plotting import animate_run, compose_side_by_side, plot_run

        cfg = load_safety_config(cfg_path)
        result["plot"] = str(plot_run(run_dir, cfg, title=f"{name}: {desc}"))
        anim = animate_run(run_dir, cfg)
        result["animation"] = str(anim)
        vids = sorted(run_dir.glob("*.webm"))
        if vids:
            sim_video = vids[-1]
            final = run_dir / f"{name}_sim.webm"
            sim_video.rename(final)
            meta = sim_video.with_suffix(".json")
            if meta.exists():
                meta.rename(final.with_suffix(".json"))
            result["sim_video"] = str(final)
            try:
                from .plotting import video_start_epoch

                v0 = video_start_epoch(final)
                offset = mon.get("t0_epoch", v0) - v0
                result["side_by_side"] = str(compose_side_by_side(final, anim, offset,
                                                                  run_dir / f"{name}_side_by_side.mp4"))
            except Exception as e:  # noqa: BLE001
                result["side_by_side_error"] = str(e)
    (run_dir / "demo_result.json").write_text(json.dumps(result, indent=2, default=float))
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--config", default=None,
                   help="base safety config (default config/safety.yaml; the demo trajectories were designed "
                        "for tests/fixtures/safety_test.yaml)")
    p.add_argument("--only", nargs="*", default=None, choices=list(RUNS))
    p.add_argument("--seconds", type=float, default=60.0, help="max recording length per run")
    args = p.parse_args(argv)
    global BASE_CONFIG
    if args.config:
        BASE_CONFIG = Path(args.config).resolve()
    make_tcp_trajectory()
    make_elbow_trajectory()
    OUT.mkdir(parents=True, exist_ok=True)
    names = args.only or list(RUNS)
    results = []
    for n in names:
        results.append(run_one(n, args.ip, args.seconds))
    summ_path = OUT / "demo_summary.json"
    allres = {}
    for name in RUNS:
        f = OUT / name / "demo_result.json"
        if f.exists():
            r = json.loads(f.read_text())
            allres[name] = {k: v for k, v in r.items() if k not in ("safety", "preprocess")}
    summ_path.write_text(json.dumps(allres, indent=2, default=float))
    print("\n================ SUMMARY ================")
    for r in allres.values():
        mc = r.get("measured_min_clearance_mm")
        print(f"{r['name']:34s} exit={r['exit_code']} min_clearance="
              f"{'n/a' if mc is None else f'{mc:.1f} mm'} ({r['measured_min_clearance_where']}) "
              f"stopped_by={r['stopped_by']} robot_after={r['robot_after_run']}")
    print(f"summary: {summ_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
