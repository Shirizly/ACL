"""Trajectory fitting: transforms keep relative motion; search finds a fit; splitting covers the source."""
import math
from pathlib import Path

import numpy as np
import pytest

from src.xarm_pipeline.safety.config import load_safety_config
from src.xarm_pipeline.safety.fit import FitConfig, Fitter, Segment
from src.xarm_pipeline.safety.kinematics import rpy_to_matrix

CONFIG = Path(__file__).resolve().parent / "fixtures" / "safety_test.yaml"


def _line(length_mm, duration_s, dt=0.1, x=350.0, z=300.0):
    t = np.arange(0.0, duration_s + 1e-9, dt)
    s = 0.5 - 0.5 * np.cos(np.pi * t / duration_s)
    p = np.column_stack([np.full_like(t, x), -length_mm / 2 + length_mm * s, np.full_like(t, z)])
    R = np.repeat(rpy_to_matrix(math.pi, 0.0, 0.0)[None], len(t), axis=0)
    return Segment(t, p, R, None, "line", (0, len(t) - 1))


@pytest.fixture(scope="module")
def cfg():
    return load_safety_config(CONFIG)


def test_apply_keeps_relative_motion(cfg):
    f = Fitter(cfg, FitConfig())
    seg = _line(300, 3.0)
    th = {"cx": 100.0, "cy": -50.0, "cz": 400.0, "yaw": 40.0, "spin": 25.0, "tilt_x": 0.0, "tilt_y": 0.0}
    p, Rs, _ = f.apply(seg, th)
    Rz = rpy_to_matrix(0, 0, math.radians(40))
    assert np.allclose(p - p[0], (seg.p - seg.p[0]) @ Rz.T, atol=1e-9)
    assert np.allclose(0.5 * (p.min(0) + p.max(0)), [100.0, -50.0, 400.0], atol=1e-9)  # base at world origin here
    # relative orientation between samples is unchanged up to the constant spin conjugation
    assert np.allclose(Rs[0].T @ Rs[-1], rpy_to_matrix(0, 0, math.radians(-25)) @ seg.R[0].T @ seg.R[-1]
                       @ rpy_to_matrix(0, 0, math.radians(25)), atol=1e-9)


def test_fit_moves_infeasible_trajectory_inside(cfg):
    f = Fitter(cfg, FitConfig(max_evals=40))
    seg = _line(300, 2.0, z=60.0)  # tool far below the floor as given
    assert f.evaluate(seg, {"cx": 350.0, "cy": 0.0, "cz": 60.0, "yaw": 0, "spin": 0, "tilt_x": 0, "tilt_y": 0}) < 0
    th, slack, _ = f.fit(seg)
    assert slack >= 0


def test_split_covers_source_and_shares_cut(cfg):
    f = Fitter(cfg, FitConfig(max_evals=25, max_depth=1, min_part_s=1.0))
    seg = _line(2000, 6.0)  # longer than the box in every direction
    fitted, dropped = f.fit_or_split(seg)
    parts = [x["segment"] for x in fitted] + [x["segment"] for x in dropped]
    rows = sorted(s.src_rows for s in parts)
    assert len(parts) == 2 and rows[0][0] == 0 and rows[-1][1] == len(seg.t) - 1
    assert rows[0][1] == rows[1][0]  # parts share the cut sample: no motion lost
    assert all(s.t[0] == 0.0 for s in parts)
