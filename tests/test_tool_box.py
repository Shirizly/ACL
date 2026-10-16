"""Gripper attachment box (geometry.tool_box) and a robot base sitting below the floor."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.xarm_pipeline.safety.config import load_safety_config
from src.xarm_pipeline.safety.geometry import MOVING_CAPSULES, WALL_NAMES, Workcell
from src.xarm_pipeline.safety.kinematics import fk_pose
from src.xarm_pipeline.safety.preprocess import DEFAULT_CARTESIAN_SEED_DEG

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "safety_test.yaml"
READY = np.radians(DEFAULT_CARTESIAN_SEED_DEG)  # flange at (300, 0, 300), pointing down (roll = pi)
BOX = {"frame": "flange", "x_mm": [-17.5, 17.5], "y_mm": [-90.0, 90.0], "z_mm": [0.0, 80.0]}


def _cfg(tmp_path, box=BOX, **base):
    raw = yaml.safe_load(FIXTURE.read_text())
    if box is not None:
        raw["geometry"]["tool_box"] = box
    raw["robot_base_in_world"].update(base)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_safety_config(p)


def _clearance(cfg, link, wall, q=READY):
    C, _ = Workcell(cfg).clearance_for_q(np.asarray(q)[None], MOVING_CAPSULES)
    ks = [k for k, c in enumerate(MOVING_CAPSULES) if c[0] == link]
    return float(C[0, ks, WALL_NAMES.index(wall)].min())


def test_ready_pose_reference():
    assert np.allclose(fk_pose(READY)[:3], [300, 0, 300], atol=0.1)


def test_box_corners_follow_the_flange(tmp_path):
    cfg = _cfg(tmp_path)
    assert _clearance(cfg, "tool_box", "floor") == pytest.approx(300 - 80, abs=0.1)  # box hangs 80 mm below the flange
    assert _clearance(cfg, "tool_box", "wall +y") == pytest.approx(400 - 90, abs=0.1)
    assert _clearance(cfg, "tool_box", "wall +x") == pytest.approx(600 - 300 - 17.5, abs=0.1)
    # the attachment, not the round gripper capsule (radius 50), is what limits sideways motion
    assert _clearance(cfg, "tool", "wall +y") == pytest.approx(400 - 50, abs=0.1)


def test_box_frame_tool_shifts_by_tool_offset(tmp_path):
    cfg = _cfg(tmp_path, box=dict(BOX, frame="tool"))
    assert _clearance(cfg, "tool_box", "floor") == pytest.approx(300 - 172 - 80, abs=0.1)


def test_without_box_nothing_changes(tmp_path):
    with_none = _cfg(tmp_path, box=None)
    assert with_none.tool_box is None
    # corners collapse onto the flange; the wrist_offset capsule (radius 50, ends at the flange) dominates
    assert _clearance(with_none, "tool_box", "floor") >= _clearance(with_none, "wrist_offset", "floor")


def test_violation_names_the_attachment(tmp_path):
    cfg = _cfg(tmp_path, box=dict(BOX, y_mm=[-380.0, 380.0]))  # absurdly wide: pokes through both y walls
    v, _ = Workcell(cfg).first_violation(READY[None], 22.5, MOVING_CAPSULES)
    assert v is not None and v.link == "tool_box" and v.wall in ("wall -y", "wall +y")


def test_base_may_sit_below_the_floor(tmp_path):
    low = _cfg(tmp_path, z_mm=-100.0)
    assert Workcell(low).static_check() == []
    # moving links are still checked against the floor, now 100 mm closer
    assert _clearance(low, "tool_box", "floor") == pytest.approx(300 - 80 - 100, abs=0.1)
    too_low = _cfg(tmp_path, z_mm=-200.0)
    problems = Workcell(too_low).static_check()
    assert len(problems) == 1 and "shoulder" in problems[0]


def test_gripper_vs_robot_base_column(tmp_path):
    """The hanging gripper must keep the margin from the robot's own base column (firmware C22 found this)."""
    from src.xarm_pipeline.safety.kinematics import ik_step_to
    from src.xarm_pipeline.safety.preprocess import validate_cartesian_trajectory

    cfg = _cfg(tmp_path)
    wc = Workcell(cfg)
    near = ik_step_to([52.3, 119.1, 276.5, np.pi, 0, 0], READY, max_iter=400)  # pose the sim firmware flagged
    assert wc.base_column_slack(wc.points_world(READY[None]))[0] > 20  # closest body part here: the forearm
    assert wc.base_column_slack(wc.points_world(near[None]))[0] < 0
    line = [[300 + (52.3 - 300) * s, 119.1 * s, 300 + (276.5 - 300) * s, np.pi, 0, 0] for s in np.linspace(0, 1, 30)]
    res = validate_cartesian_trajectory(cfg, list(np.linspace(0, 6, 30)), line, READY, None, include_approach=False)
    assert not res.ok and "base column" in res.message
