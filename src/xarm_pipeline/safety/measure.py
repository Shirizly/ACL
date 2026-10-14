"""Measure the workcell with the robot itself, and check the gripper model on the real arm.

    python -m src.xarm_pipeline.safety.measure where      [--ip IP]   # model of the gripper vs the walls, now
    python -m src.xarm_pipeline.safety.measure watch      [--ip IP]   # live clearance while you jog by hand
    python -m src.xarm_pipeline.safety.measure walls      [--ip IP]   # touch walls -> workcell.walls_in_base_mm
    python -m src.xarm_pipeline.safety.measure wall-test  [--ip IP] --direction y+   # slow stop test, virtual wall

All coordinates are in the ROBOT BASE frame (the frame of get_position(); z = 0 at the base mounting
surface), in mm. Positions come from the joint angles through the same kinematic and gripper model the
safety checks use, so what you measure here is exactly what the checks will see.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import DEFAULT_CONFIG_PATH, WALL_KEYS, SafetyConfig, load_safety_config
from .geometry import CAPSULES, END_EFFECTOR_CAPSULES, Workcell, chain_points_batch

REPO_ROOT = Path(__file__).resolve().parents[3]
AXES = {"x": 0, "y": 1, "z": 2}
WALL_HELP = {
    "x_min": "behind the robot (-x)", "x_max": "in front of the robot (+x)",
    "y_min": "robot's right (-y)", "y_max": "robot's left (+y)",
    "z_min": "floor", "z_max": "ceiling",
}


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------
def chain_points_base(cfg: SafetyConfig, q_rad) -> np.ndarray:
    """(17,3) chain points in the BASE frame for joints q (rad)."""
    return chain_points_batch(np.asarray(q_rad, float)[None], cfg.tool_offset_mm,
                              tool_box_corners=cfg.tool_box_corners_mm())[0]


def contact_parts(cfg: SafetyConfig, P: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    """Parts you can touch a wall with: name -> (points, radius). The contact surface along an axis is
    the extreme point coordinate +- radius."""
    parts = {
        "tip": (P[[8]], 0.0),                                   # tool_offset_mm point (finger tips)
        "flange": (P[[7]], 0.0),
        "tool": (P[[7, 8]], float(cfg.radii_mm["tool"])),      # round gripper model
    }
    if cfg.tool_box is not None:
        parts["box"] = (P[9:17], float(cfg.radii_mm["tool_box"]))
    return parts


def surface_coord(points: np.ndarray, radius: float, axis: int, sign: int) -> float:
    """Outermost coordinate of a part along +axis (sign=+1) or -axis (sign=-1)."""
    return float(sign * np.max(sign * points[:, axis]) + sign * radius)


def gripper_extreme(cfg: SafetyConfig, P: np.ndarray, axis: int, sign: int) -> tuple[float, str]:
    """Outermost end-effector capsule surface (what the checks use) along a direction, and its name."""
    best = None
    for name, i, j in CAPSULES:
        if name not in END_EFFECTOR_CAPSULES:
            continue
        c = surface_coord(P[[i, j]], float(cfg.radii_mm[name]), axis, sign)
        if best is None or sign * c > sign * best[0]:
            best = (c, name)
    return best


def walls_in_base(cfg: SafetyConfig) -> dict[str, float]:
    if abs(cfg.base_yaw_deg) > 1e-9:
        raise ValueError("this needs walls parallel to the base axes (base yaw 0)")
    return {"x_min": -cfg.depth_D_mm / 2 - cfg.base_x_mm, "x_max": cfg.depth_D_mm / 2 - cfg.base_x_mm,
            "y_min": -cfg.width_W_mm / 2 - cfg.base_y_mm, "y_max": cfg.width_W_mm / 2 - cfg.base_y_mm,
            "z_min": -cfg.base_z_mm, "z_max": cfg.height_H_mm - cfg.base_z_mm}


def connect(ip: str):
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
    if not arm.connected:
        raise SystemExit(f"cannot connect to the robot at {ip}")
    return arm


def read_joints(arm) -> np.ndarray:
    code, q = arm.get_servo_angle(is_radian=True)
    if code != 0:
        raise RuntimeError(f"get_servo_angle failed, code {code}")
    return np.asarray(q[:7], float)


# ---------------------------------------------------------------------------
# where / watch
# ---------------------------------------------------------------------------
def describe_pose(cfg: SafetyConfig, q: np.ndarray) -> str:
    P = chain_points_base(cfg, q)
    lines = [f"joints (deg): {np.degrees(q).round(2).tolist()}",
             f"flange      : {P[7].round(1).tolist()}",
             f"tool tip    : {P[8].round(1).tolist()}   (tool_offset_mm {list(cfg.tool_offset_mm)})"]
    if cfg.tool_box is not None:
        c = P[9:17]
        lines.append("gripper box (geometry.tool_box) occupies (base frame, mm):  "
                     + "  ".join(f"{a} [{c[:, k].min():.1f}, {c[:, k].max():.1f}]" for a, k in AXES.items()))
        lines.append("  corners: " + "; ".join(f"{p.round(1).tolist()}" for p in c))
    r = cfg.radii_mm["tool"]
    t = P[[7, 8]]
    lines.append(f"round tool model (radius {r:g}) spans:  "
                 + "  ".join(f"{a} [{t[:, k].min() - r:.1f}, {t[:, k].max() + r:.1f}]" for a, k in AXES.items()))
    wc = Workcell(cfg)
    from .geometry import MOVING_CAPSULES, WALL_NAMES

    C, _ = wc.clearance_for_q(q[None], MOVING_CAPSULES)
    names = [n for n, _, _ in MOVING_CAPSULES]
    lines.append(f"clearance to each wall (config {cfg.source_path}; margin {cfg.safety_margin_mm:g} mm):")
    for w, wname in enumerate(WALL_NAMES):
        k = int(np.argmin(C[0, :, w]))
        lines.append(f"  {wname:8s}: {C[0, k, w]:8.1f} mm  (closest part: {names[k]})")
    return "\n".join(lines)


def cmd_where(args) -> int:
    cfg = load_safety_config(args.config)
    arm = connect(args.ip)
    try:
        print(describe_pose(cfg, read_joints(arm)))
    finally:
        arm.disconnect()
    return 0


def cmd_watch(args) -> int:
    from .geometry import MOVING_CAPSULES, WALL_NAMES

    cfg = load_safety_config(args.config)
    wc = Workcell(cfg)
    names = [n for n, _, _ in MOVING_CAPSULES]
    arm = connect(args.ip)
    print(f"watching (Ctrl+C to quit); margin {cfg.safety_margin_mm:g} mm"
          + ("; will STOP the arm below the margin" if args.stop else "; display only"))
    stopped = False
    try:
        while True:
            q = read_joints(arm)
            C, _ = wc.clearance_for_q(q[None], MOVING_CAPSULES)
            k, w = np.unravel_index(int(np.argmin(C[0])), C[0].shape)
            c = float(C[0, k, w])
            flag = "  << INSIDE MARGIN" if c < cfg.safety_margin_mm else ""
            box = ""
            if cfg.tool_box is not None:
                kb = [i for i, n in enumerate(names) if n == "tool_box"]
                cb = C[0, kb].min(axis=0)
                box = f" | attachment: {cb.min():7.1f} mm to {WALL_NAMES[int(np.argmin(cb))]}"
            print(f"\rclosest: {c:7.1f} mm  {names[k]:12s} vs {WALL_NAMES[w]:8s}{box}{flag}      ", end="", flush=True)
            if args.stop and c < cfg.safety_margin_mm and not stopped:
                arm.set_state(4)
                stopped = True
                print(f"\nSTOP sent: {names[k]} {c:.1f} mm from {WALL_NAMES[w]}")
            time.sleep(1.0 / args.rate)
    except KeyboardInterrupt:
        print()
    finally:
        arm.disconnect()
    return 0


# ---------------------------------------------------------------------------
# walls: touch-off measurement
# ---------------------------------------------------------------------------
def touch_record(cfg: SafetyConfig, q: np.ndarray, wall: str, part: str) -> dict:
    """Wall coordinate implied by `part` touching `wall` at joints q."""
    if wall not in WALL_KEYS:
        raise ValueError(f"wall must be one of {WALL_KEYS}")
    axis, sign = AXES[wall[0]], (1 if wall.endswith("max") else -1)
    P = chain_points_base(cfg, q)
    parts = contact_parts(cfg, P)
    if part not in parts:
        raise ValueError(f"part must be one of {sorted(parts)}")
    pts, r = parts[part]
    coord = surface_coord(pts, r, axis, sign)
    # the touching point, for the parallelism estimate
    contact = pts[int(np.argmax(sign * pts[:, axis]))].copy()
    contact[axis] = coord
    return {"wall": wall, "part": part, "coord_mm": round(coord, 2), "contact_point_mm": contact.round(2).tolist(),
            "joints_deg": np.degrees(q).round(3).tolist(), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


def summarize_touches(touches: list[dict]) -> tuple[dict, list[str]]:
    """Mean wall coordinate per wall + warnings (spread, walls not parallel to the base axes)."""
    walls, notes = {}, []
    for wall in WALL_KEYS:
        ts = [t for t in touches if t["wall"] == wall]
        if not ts:
            continue
        v = np.array([t["coord_mm"] for t in ts])
        walls[wall] = round(float(v.mean()), 1)
        if len(ts) > 1:
            notes.append(f"{wall}: {len(ts)} touches, spread {v.max() - v.min():.1f} mm")
            axis = AXES[wall[0]]
            pts = np.array([t["contact_point_mm"] for t in ts])
            for other in (a for a in range(3) if a != axis):
                span = pts[:, other].max() - pts[:, other].min()
                if span > 50:
                    i, j = int(np.argmin(pts[:, other])), int(np.argmax(pts[:, other]))
                    ang = math.degrees(math.atan2(pts[j, axis] - pts[i, axis], pts[j, other] - pts[i, other]))
                    if abs(ang) > 1.0:
                        notes.append(f"WARNING {wall}: wall is tilted {ang:+.1f} deg relative to the base "
                                     f"{'xyz'[other]}-axis (touches {span:.0f} mm apart). The model assumes walls "
                                     "parallel to the base axes; the safe choice is the innermost touch "
                                     f"({(v.min() if wall.endswith('max') else v.max()):.1f}).")
    return walls, notes


def walls_yaml(walls: dict) -> str:
    lines = ["workcell:", "  walls_in_base_mm:          # measured by touch-off, robot base frame, mm"]
    for k in WALL_KEYS:
        if k in walls:
            lines.append(f"    {k}: {walls[k]:<10}  # {WALL_HELP[k]}")
        else:
            lines.append(f"    {k}: MISSING     # not measured yet: {WALL_HELP[k]}")
    return "\n".join(lines)


def cmd_walls(args) -> int:
    cfg = load_safety_config(args.config)
    session = Path(args.session) if args.session else (
        REPO_ROOT / "outputs" / "measurements" / f"walls_{time.strftime('%Y%m%d_%H%M%S')}.json")
    session.parent.mkdir(parents=True, exist_ok=True)
    touches = json.loads(session.read_text()) if session.exists() else []
    arm = connect(args.ip)
    parts_avail = sorted(contact_parts(cfg, chain_points_base(cfg, np.zeros(7))))
    print(__doc__.split("\n\n")[0])
    print(f"""
Move the arm (Studio jogging or hand-guiding) until the chosen gripper part just touches a wall, then type
   <wall> [part]        walls: {' '.join(WALL_KEYS)}   parts: {' '.join(parts_avail)} (default tip)
   list | undo | done
'tip' is the tool_offset_mm point (finger tips): the most precise contact. Touch each wall 2+ times at
points far apart to check it is parallel to the robot's axes. Session file: {session}
""")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line == "done":
                break
            if line == "list":
                for i, t in enumerate(touches):
                    print(f"  [{i}] {t['wall']:5s} {t['part']:6s} {t['coord_mm']:8.1f}  at {t['contact_point_mm']}")
                continue
            if line == "undo":
                if touches:
                    print(f"  removed {touches.pop()}")
                session.write_text(json.dumps(touches, indent=1))
                continue
            wall, *rest = line.split()
            try:
                t = touch_record(cfg, read_joints(arm), wall, rest[0] if rest else "tip")
            except ValueError as e:
                print(f"  {e}")
                continue
            touches.append(t)
            session.write_text(json.dumps(touches, indent=1))
            print(f"  {t['wall']} = {t['coord_mm']:.1f} mm  ({t['part']} at {t['contact_point_mm']})")
    finally:
        arm.disconnect()
    walls, notes = summarize_touches(touches)
    print("\n".join(notes))
    text = walls_yaml(walls)
    print("\nPaste into config/safety.yaml, replacing the whole 'workcell:' block AND the 'robot_base_in_world:' block:\n")
    print(text)
    session.with_suffix(".yaml").write_text(text + "\n")
    print(f"\n(saved {session} and {session.with_suffix('.yaml')})")
    return 0


# ---------------------------------------------------------------------------
# wall-test: slow stop test against a virtual wall
# ---------------------------------------------------------------------------
def make_test_config(cfg_path: Path, cfg: SafetyConfig, wall: str, coord: float, out: Path) -> dict:
    """Current config with one wall moved inward to `coord` (all real walls stay in force)."""
    walls = walls_in_base(cfg)
    sign = 1 if wall.endswith("max") else -1
    if sign * coord > sign * walls[wall]:
        raise ValueError(f"the virtual wall ({coord:.1f}) would be beyond the real {wall} ({walls[wall]:.1f}); "
                         "use a smaller --gap or start further from the wall")
    walls[wall] = round(coord, 2)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw["workcell"] = {"walls_in_base_mm": walls}
    raw.pop("robot_base_in_world", None)
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return walls


def cmd_wall_test(args) -> int:
    cfg_path = Path(args.config or DEFAULT_CONFIG_PATH)
    cfg = load_safety_config(cfg_path)
    d = args.direction if args.direction[0] in "+-" else args.direction[1] + args.direction[0]  # 'x-' -> '-x'
    axis, sign = AXES[d[1]], (1 if d[0] == "+" else -1)
    wall = f"{d[1]}_{'max' if sign > 0 else 'min'}"
    run_dir = REPO_ROOT / "outputs" / "wall_tests" / f"{time.strftime('%Y%m%d_%H%M%S')}_{d[1]}{'p' if sign > 0 else 'm'}"

    arm = connect(args.ip)
    q0 = read_joints(arm)
    code, pose0 = arm.get_position(is_radian=True)
    if code != 0:
        raise SystemExit(f"get_position failed: {code}")
    P0 = chain_points_base(cfg, q0)
    ext0, part0 = gripper_extreme(cfg, P0, axis, sign)
    virtual = ext0 + sign * args.gap
    test_cfg = run_dir / "safety_test_config.yaml"
    try:
        walls_in_base(cfg)  # needs base yaw 0
        run_dir.mkdir(parents=True, exist_ok=True)
        walls = make_test_config(cfg_path, cfg, wall, virtual, test_cfg)
    except ValueError as e:
        arm.disconnect()
        if run_dir.exists():
            for f in run_dir.iterdir():
                f.unlink()
            run_dir.rmdir()
        print(f"not running: {e}")
        return 2

    # trajectory: straight line along the direction, constant orientation, slow
    dist = args.gap + args.overshoot
    T = dist / args.speed
    t = np.arange(0.0, T + 1e-9, 0.03)
    s = np.clip(t / T, 0, 1)
    pts = np.tile(np.asarray(pose0[:6], float), (len(t), 1))
    pts[:, axis] += sign * dist * s
    traj = run_dir / "wall_test_trajectory.csv"
    pd.DataFrame({"t": t, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "roll": pts[:, 3], "pitch": pts[:, 4],
                  "yaw": pts[:, 5]}).to_csv(traj, index=False)

    # what the model expects: without the attachment, how much further could the arm go?
    no_box = [(n, i, j) for n, i, j in CAPSULES if n in END_EFFECTOR_CAPSULES and n != "tool_box"]
    ext0_nobox = max((surface_coord(P0[[i, j]], float(cfg.radii_mm[n]), axis, sign) * sign, n) for n, i, j in no_box)
    print(f"""
WALL STOP TEST (direction {d}, base frame)
  outermost gripper part now : {part0} at {d[1]} = {ext0:.1f} mm
  virtual wall ({wall})       : {virtual:.1f} mm  ({args.gap:g} mm beyond it; real walls stay active)
  planned move               : TCP {d} by {dist:g} mm at {args.speed:g} mm/s ({T:.1f} s), orientation fixed
  expected                   : runtime guard stops the arm with the {part0} about {cfg.safety_margin_mm:g}-{cfg.safety_margin_mm + 10:g} mm
                               short of the virtual wall, i.e. after ~{args.gap - cfg.safety_margin_mm - 10:.0f}-{args.gap - cfg.safety_margin_mm:.0f} mm of travel""")
    if part0 == "tool_box":
        print(f"  without the attachment     : the round tool model reaches only {ext0_nobox[0] * sign:.1f} mm, so the stop "
              f"would come ~{abs(ext0 - ext0_nobox[0] * sign):.0f} mm later. An early stop by that much shows the box is used.")
    print(f"""
PHYSICAL CHECK: before running, put a marker (ruler edge, card) {args.gap:g} mm beyond the outer edge of the
real {'attachment' if part0 == 'tool_box' else 'gripper'} along {d}. After the stop, measure the gap from that edge to the marker: it should
match the "model gap" printed at the end (within a few mm). A smaller real gap means the gripper model is too small.
Files: {run_dir}""")
    if not args.yes:
        if input("\nArm clear, speed OK, hand on the e-stop? Type 'yes' to move: ").strip().lower() != "yes":
            arm.disconnect()
            print("aborted, nothing moved")
            return 1
    arm.disconnect()

    cmd = [sys.executable, "-m", "src.xarm_pipeline.cartesian_playback", "--ip", args.ip, "--trajectory", str(traj),
           "--control", "servo", "--no-fit", "--safety-config", str(test_cfg), "--disable-safety-preprocess",
           "--report-dir", str(run_dir)]
    with (run_dir / "run.log").open("w") as fh:
        rc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    log = (run_dir / "run.log").read_text()
    stop_lines = [ln for ln in log.splitlines() if "SAFETY STOP" in ln or "ABORTED" in ln or "safety monitor:" in ln]

    arm = connect(args.ip)
    try:
        q1 = read_joints(arm)
        P1 = chain_points_base(cfg, q1)
        ext1, part1 = gripper_extreme(cfg, P1, axis, sign)
        travel = float(sign * (P1[7, axis] - P0[7, axis]))
        gap1 = float(sign * (virtual - ext1))
        print(f"\nplayback exit code {rc} ({'stopped by the safety layers' if rc == 3 else 'NOT stopped by safety (expected exit 3)'})")
        for ln in stop_lines:
            print("  " + ln.strip())
        print(f"  flange travelled  : {travel:.1f} mm")
        print(f"  model gap         : {gap1:.1f} mm between the {part1} and the virtual wall (margin {cfg.safety_margin_mm:g} mm)")
        ok = rc == 3 and gap1 >= cfg.safety_margin_mm - 1.0
        print(f"  RESULT            : {'PASS' if ok else 'CHECK'}: "
              + ("stopped before the margin" if ok else "see run.log"))
        # recover and go back to the start
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        if not args.no_return:
            time.sleep(0.5)
            code = arm.set_servo_angle(angle=q0.tolist(), is_radian=True, speed=0.2, wait=True)
            print(f"  returned to the start pose (rc {code})")
        (run_dir / "result.json").write_text(json.dumps({
            "direction": d, "virtual_wall": {wall: virtual}, "walls_used": walls, "exit_code": rc,
            "outermost_part_start": part0, "flange_travel_mm": travel, "model_gap_mm": gap1,
            "part_at_stop": part1, "pass": ok}, indent=2))
    finally:
        arm.disconnect()
    return 0 if ok else 3


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("where", "watch", "walls", "wall-test"):
        sp = sub.add_parser(name)
        sp.add_argument("--ip", default="127.0.0.1")
        sp.add_argument("--config", default=None, help="safety YAML (default config/safety.yaml)")
        if name == "watch":
            sp.add_argument("--rate", type=float, default=10.0)
            sp.add_argument("--stop", action="store_true", help="send a stop when inside the margin")
        if name == "walls":
            sp.add_argument("--session", default=None, help="JSON file to keep/continue touches in")
        if name == "wall-test":
            sp.add_argument("--direction", required=True,
                            choices=["x+", "x-", "y+", "y-", "z+", "z-", "+x", "-x", "+y", "-y", "+z", "-z"],
                            help="move direction in the base frame: x+ x- y+ y- z+ z- (write --direction=-z for the sign-first form)")
            sp.add_argument("--gap", type=float, default=80.0, help="virtual wall distance beyond the gripper, mm")
            sp.add_argument("--overshoot", type=float, default=40.0, help="planned travel past the virtual wall, mm")
            sp.add_argument("--speed", type=float, default=20.0, help="mm/s")
            sp.add_argument("--yes", action="store_true", help="do not ask for confirmation (simulator)")
            sp.add_argument("--no-return", action="store_true", help="stay where it stopped")
    args = p.parse_args(argv)
    return {"where": cmd_where, "watch": cmd_watch, "walls": cmd_walls, "wall-test": cmd_wall_test}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
