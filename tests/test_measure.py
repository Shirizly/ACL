"""Walls-in-base config format and the touch-off measurement maths (no robot needed)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.xarm_pipeline.safety import preprocessed as pp
from src.xarm_pipeline.safety.config import load_safety_config, walls_in_base_to_box
from src.xarm_pipeline.safety.measure import summarize_touches, touch_record, walls_in_base, walls_yaml
from src.xarm_pipeline.safety.preprocess import DEFAULT_CARTESIAN_SEED_DEG

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "safety_test.yaml"
READY = np.radians(DEFAULT_CARTESIAN_SEED_DEG)  # flange (300, 0, 300) pointing down
BOX = {"frame": "flange", "x_mm": [-17.5, 17.5], "y_mm": [-90.0, 90.0], "z_mm": [0.0, 80.0]}


def _write(tmp_path, name, mutate):
    raw = yaml.safe_load(FIXTURE.read_text())
    mutate(raw)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw))
    return p


def _walls_form(raw, walls):
    raw["workcell"] = {"walls_in_base_mm": walls}
    raw.pop("robot_base_in_world", None)


def test_walls_format_equals_box_form(tmp_path):
    box = _write(tmp_path, "box.yaml", lambda r: r["robot_base_in_world"].update(x_mm=-75.0, z_mm=-20.0))
    walls = {"x_min": -525.0, "x_max": 675.0, "y_min": -400.0, "y_max": 400.0, "z_min": 20.0, "z_max": 1120.0}
    wal = _write(tmp_path, "walls.yaml", lambda r: _walls_form(r, walls))
    a, b = load_safety_config(box), load_safety_config(wal)
    for f in ("depth_D_mm", "width_W_mm", "height_H_mm", "base_x_mm", "base_y_mm", "base_z_mm", "base_yaw_deg"):
        assert getattr(a, f) == pytest.approx(getattr(b, f)), f
    assert walls_in_base(b) == pytest.approx(walls)
    # the same cell in either spelling is the same preprocessed setup
    assert pp.setup_fingerprint(box) == pp.setup_fingerprint(wal)


def test_walls_format_rejects_mixing_and_bad_order(tmp_path):
    walls = {"x_min": -500, "x_max": 700, "y_min": -400, "y_max": 400, "z_min": 0, "z_max": 1100}
    def mixed(r):
        r["workcell"]["walls_in_base_mm"] = walls
    with pytest.raises(ValueError, match="not both"):
        load_safety_config(_write(tmp_path, "mixed.yaml", mixed))
    with pytest.raises(ValueError, match="x_max must be > x_min"):
        walls_in_base_to_box(dict(walls, x_max=-600))


def test_touch_off_coordinates(tmp_path):
    def add_box(r):
        r["geometry"]["tool_box"] = BOX
    cfg = load_safety_config(_write(tmp_path, "box.yaml", add_box))
    assert touch_record(cfg, READY, "x_max", "tip")["coord_mm"] == pytest.approx(300.0, abs=0.1)
    assert touch_record(cfg, READY, "z_min", "tip")["coord_mm"] == pytest.approx(300.0 - 172.0, abs=0.1)
    # flange y axis = base -y here; the box spans +-90 mm sideways
    assert touch_record(cfg, READY, "y_max", "box")["coord_mm"] == pytest.approx(90.0, abs=0.1)
    assert touch_record(cfg, READY, "y_min", "box")["coord_mm"] == pytest.approx(-90.0, abs=0.1)
    assert touch_record(cfg, READY, "y_max", "tool")["coord_mm"] == pytest.approx(50.0, abs=0.1)
    with pytest.raises(ValueError):
        touch_record(cfg, READY, "x_far", "tip")


def test_summary_averages_and_flags_tilted_walls():
    t = lambda c, p: {"wall": "x_max", "part": "tip", "coord_mm": c, "contact_point_mm": p}
    walls, notes = summarize_touches([t(600.0, [600, -150, 300]), t(602.0, [602, 150, 300])])
    assert walls["x_max"] == pytest.approx(601.0) and not any("WARNING" in n for n in notes)
    walls, notes = summarize_touches([t(600.0, [600, -150, 300]), t(620.0, [620, 150, 300])])
    assert any("WARNING x_max" in n and "tilted" in n for n in notes)
    assert "MISSING" in walls_yaml({"x_max": 601.0})
