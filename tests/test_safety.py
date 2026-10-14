"""Unit tests for the safety-wall package (no robot needed).

Run:  .venv/bin/python -m pytest tests -q
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.xarm_pipeline.safety.config import load_safety_config, validate_config
from src.xarm_pipeline.safety.firmware_fence import inscribed_base_box
from src.xarm_pipeline.safety.geometry import (
    MOVING_CAPSULES,
    WALL_NAMES,
    Workcell,
    chain_points_batch,
)
from src.xarm_pipeline.safety.kinematics import (
    ArmModel,
    fk_pose,
    ik_step_to,
    matrix_to_rpy,
    rpy_to_matrix,
)
from src.xarm_pipeline.safety.preprocess import (
    SafetyChecker,
    validate_cartesian_trajectory,
    validate_joint_trajectory,
)
from src.xarm_pipeline.safety.runtime import CommandGuard, SafetyMonitor, SafetyStop

# Fixed copy of the original sim test cell; config/safety.yaml holds the real (changing) measurements.
CONFIG = Path(__file__).resolve().parent / "fixtures" / "safety_test.yaml"


@pytest.fixture
def cfg():
    return load_safety_config(CONFIG)


def _rot_err_deg(rpy_a, rpy_b) -> float:
    Ra, Rb = rpy_to_matrix(*rpy_a), rpy_to_matrix(*rpy_b)
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


# ---------------------------------------------------------------- kinematics
# Reference values from the simulator firmware: arm.get_forward_kinematics(q deg)
FIRMWARE_FK = [
    ([0, 0, 0, 0, 0, 0, 0], [206.0, 0.0, 120.5, 3.1415927, 0.0, 0.0]),
    ([30, -20, 45, 60, -30, 40, 90], [74.4658737, 443.6212463, 503.3645019, -2.6304831, -0.7434121, -0.1095803]),
    ([-90, 40, -60, 120, 45, -30, -45], [-308.7011108, -466.3684082, 690.7332763, 0.6357522, -0.7678686, -0.2160700]),
    ([90, -100, 0, 60, 0, 120, 0], [0.0, -132.7797241, 590.7203369, -3.1415927, -0.6981316, 1.5707963]),
]


@pytest.mark.parametrize("q_deg,expected", FIRMWARE_FK)
def test_fk_matches_firmware(q_deg, expected):
    pose = fk_pose(np.radians(q_deg))
    assert np.allclose(pose[:3], expected[:3], atol=1e-3)
    assert _rot_err_deg(pose[3:], expected[3:]) < 1e-4


def test_rpy_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(100):
        rpy = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        assert np.allclose(rpy_to_matrix(*matrix_to_rpy(rpy_to_matrix(*rpy))), rpy_to_matrix(*rpy), atol=1e-9)


def test_batch_fk_equals_single():
    rng = np.random.default_rng(1)
    Q = rng.uniform(-2, 2, (50, 7))
    model = ArmModel(radii={}, tool_offset_mm=(0, 0, 172))
    P = chain_points_batch(Q, (0, 0, 172))
    for i in range(len(Q)):
        assert np.allclose(P[i][:9], model.chain_points(Q[i]))  # arm points; 9.. are gripper-box corners
    # flange point equals fk_pose position, tool tip is 172 mm along flange z
    for i in range(5):
        pose = fk_pose(Q[i])
        assert np.allclose(P[i, 7], pose[:3])
        assert np.isclose(np.linalg.norm(P[i, 8] - P[i, 7]), 172.0)


def test_local_ik_reaches_target():
    q0 = np.radians([0, -28.17, 0, 23.24, 0, 51.41, 0])
    target = [380.0, 120.0, 260.0, math.pi, 0.2, 0.3]
    q = q0
    # continuation along a straight line
    start = fk_pose(q0)
    for s in np.linspace(0.05, 1, 20):
        p = [start[i] + s * (target[i] - start[i]) for i in range(6)]
        q = ik_step_to(p, q)
    pose = fk_pose(q)
    assert np.allclose(pose[:3], target[:3], atol=0.05)
    assert _rot_err_deg(pose[3:], target[3:]) < 0.01


# ---------------------------------------------------------------- transforms
def test_base_world_transform_roundtrip(cfg):
    cfg.base_x_mm, cfg.base_y_mm, cfg.base_z_mm, cfg.base_yaw_deg = 100.0, -50.0, 20.0, 90.0
    wc = Workcell(cfg)
    # base +x axis points along world +y when yaw = 90 deg
    assert np.allclose(wc.base_to_world([100.0, 0.0, 0.0]), [100.0, 50.0, 20.0])
    pts = np.random.default_rng(2).uniform(-500, 500, (20, 3))
    assert np.allclose(wc.world_to_base(wc.base_to_world(pts)), pts)


def test_wall_distances(cfg):
    wc = Workcell(cfg)  # D=1200, W=800, H=1100
    d = wc.wall_distances([100.0, -300.0, 50.0])
    assert np.allclose(d, [700.0, 500.0, 100.0, 700.0, 50.0, 1050.0])


def test_wall_distances_with_yaw(cfg):
    cfg.base_yaw_deg = 90.0
    wc = Workcell(cfg)
    # robot-base point 350 mm along base x is at world y=+350 -> 50 mm from wall +y
    pw = wc.base_to_world([350.0, 0.0, 300.0])
    d = wc.wall_distances(pw)
    assert np.isclose(d[WALL_NAMES.index("wall +y")], 50.0)


# ---------------------------------------------------------------- box checks
def test_capsule_clearance_exact(cfg):
    wc = Workcell(cfg)
    q = np.radians([0, -28.17, 0, 23.24, 0, 51.41, 0])  # flange (300,0,300) pointing down
    C, Pw = wc.clearance_for_q(q[None])
    k = [c[0] for c in MOVING_CAPSULES].index("tool")
    # tool capsule: flange z=300 -> tip z=128; floor clearance = 128 - r(50)
    assert np.isclose(C[0, k, WALL_NAMES.index("floor")], 128.0 - cfg.radii_mm["tool"], atol=0.05)
    assert np.isclose(C[0, k, WALL_NAMES.index("wall +x")], 600 - 300 - cfg.radii_mm["tool"], atol=0.05)


def test_first_violation_reports_link_and_wall(cfg):
    wc = Workcell(cfg)
    q = np.radians([90, -100, 0, 60, 0, 120, 0])  # elbow through the -y wall, tool inside
    v, _ = wc.first_violation(q[None], cfg.safety_margin_mm)
    assert v is not None
    assert v.link in ("elbow_offset", "forearm")
    assert v.wall == "wall -y"
    assert v.clearance_mm < 0  # through the wall


def test_static_check_flags_base_outside(cfg):
    assert Workcell(cfg).static_check() == []
    cfg.base_x_mm = 590.0
    assert any("wall +x" in p for p in Workcell(cfg).static_check())


def test_config_validation(cfg):
    cfg.preprocess.policy = "maybe"
    with pytest.raises(ValueError):
        validate_config(cfg)


# ---------------------------------------------------------------- preprocessing
def _joint_traj(q_from, q_to, n=50):
    t = np.linspace(0, 5, n)
    s = np.linspace(0, 1, n)[:, None]
    q = np.radians(q_from) + s * (np.radians(q_to) - np.radians(q_from))
    return t.tolist(), q.tolist()


def test_joint_safe_trajectory_passes(cfg):
    t, q = _joint_traj([0, -28, 0, 23, 0, 51, 0], [30, -20, 0, 40, 0, 60, 0])
    res = validate_joint_trajectory(cfg, t, q)
    assert res.ok, res.message


def test_joint_elbow_violation_rejected(cfg):
    t, q = _joint_traj([90, -40, 0, 60, 0, 120, 0], [90, -100, 0, 60, 0, 120, 0])
    res = validate_joint_trajectory(cfg, t, q)
    assert not res.ok
    assert res.violation["link"] in ("elbow_offset", "forearm")
    assert res.violation["wall"] == "wall -y"
    assert 0 < res.violation["sample_index"] < len(t)


def test_joint_clip_policy_still_rejects(cfg):
    cfg.preprocess.policy = "clip"
    t, q = _joint_traj([90, -40, 0, 60, 0, 120, 0], [90, -100, 0, 60, 0, 120, 0])
    res = validate_joint_trajectory(cfg, t, q)
    assert not res.ok and res.action == "reject"


def test_crossing_between_samples_is_detected(cfg):
    """Two samples, both inside (TCP on the +-x axis); the joint-linear path
    between them (J1 0 -> 180 deg) swings the arm through the +y wall."""
    chk = SafetyChecker(cfg)
    qa = np.radians([0, 20, 0, 130, 0, 110, 0])
    qb = qa.copy()
    qb[0] = math.pi
    assert chk.check_configs(qa[None], chk.required_joint_mm())[0] is None
    assert chk.check_configs(qb[None], chk.required_joint_mm())[0] is None
    res = validate_joint_trajectory(cfg, [0.0, 2.0], [qa.tolist(), qb.tolist()])
    assert not res.ok
    assert res.violation["wall"] == "wall +y"
    assert res.violation["segment"] == "between sample 0 and sample 1"


def test_dense_interpolation_bound(cfg):
    """Checked clearance - resolution/2 is a lower bound for any config on the
    joint-linear segment (validated on a much finer grid)."""
    chk = SafetyChecker(cfg)
    qa = np.radians([10, -20, 5, 40, 0, 50, 0])
    qb = np.radians([60, 30, -40, 90, 30, 20, 45])
    Qd, _ = chk.densify_joint_path(qb[None], qa)
    C, _ = chk.wc.clearance_for_q(Qd)
    checked_min = C.min()
    fine = qa + np.linspace(0, 1, 5000)[:, None] * (qb - qa)
    Cf, _ = chk.wc.clearance_for_q(fine)
    assert Cf.min() >= checked_min - chk.interp_tol_mm - 1e-6


def _line_poses(y_end, n=60, z=300.0):
    t = np.linspace(0, 6, n)
    ys = np.concatenate([np.linspace(0, y_end, n // 2), np.linspace(y_end, 0, n - n // 2)])
    return t.tolist(), [[300.0, float(y), z, math.pi, 0.0, 0.0] for y in ys]


Q_HOME = np.radians([0, -28.17, 0, 23.24, 0, 51.41, 0])


def test_cartesian_safe_passes(cfg):
    t, poses = _line_poses(150.0)
    res = validate_cartesian_trajectory(cfg, t, poses, Q_HOME)
    assert res.ok and res.action == "pass", res.message


def test_cartesian_reject_tcp_crossing(cfg):
    t, poses = _line_poses(560.0)
    res = validate_cartesian_trajectory(cfg, t, poses, Q_HOME, policy="reject")
    assert not res.ok
    assert res.violation["wall"] == "wall +y"
    assert res.violation["link"] in ("tool", "wrist", "wrist_offset")


def test_cartesian_clip_tcp_crossing(cfg):
    t, poses = _line_poses(560.0)
    res = validate_cartesian_trajectory(cfg, t, poses, Q_HOME, policy="clip")
    assert res.ok and res.action == "clip", res.message
    ys = [p[1] for p in res.sanitized_pose]
    # wall at +400; tool radius 50 + margin 20 + resolution/2 -> flange y <= ~327.5
    assert max(ys) < 400 - cfg.radii_mm["tool"] - cfg.safety_margin_mm
    assert max(ys) > 250  # clipped, not discarded
    # untouched samples stay identical
    assert res.sanitized_pose[0] == poses[0]
    # re-validating the clipped trajectory passes in reject mode
    res2 = validate_cartesian_trajectory(cfg, t, res.sanitized_pose, Q_HOME, policy="reject")
    assert res2.ok, res2.message


def test_cartesian_clip_rejects_when_elbow_violates(cfg):
    """Poses generated from the elbow-through-wall joint path: the TCP is far
    from any wall, so clipping cannot help -> reject."""
    q_from = np.radians([90, -40, 0, 60, 0, 120, 0])
    q_to = np.radians([90, -100, 0, 60, 0, 120, 0])
    poses = [fk_pose(q_from + s * (q_to - q_from)) for s in np.linspace(0, 1, 40)]
    t = np.linspace(0, 4, 40).tolist()
    res = validate_cartesian_trajectory(cfg, t, poses, q_from, policy="clip")
    assert not res.ok
    assert res.violation["link"] not in ("tool", "wrist_offset")


# ---------------------------------------------------------------- firmware fence box
def test_fence_box_yaw0(cfg):
    s = cfg.fence_shrink_mm
    assert inscribed_base_box(cfg) == [int(600 - s), int(-600 + s), int(400 - s), int(-400 + s),
                                       int(1100 - s), int(s)]


@pytest.mark.parametrize("yaw", [30.0, 45.0, 90.0, 135.0])
def test_fence_box_inscribed_with_yaw(cfg, yaw):
    cfg.base_yaw_deg = yaw
    cfg.base_x_mm = 100.0
    cfg.base_y_mm = -50.0
    box = inscribed_base_box(cfg)
    wc = Workcell(cfg)
    s = cfg.fence_shrink_mm
    corners = np.array([[x, y, z] for x in box[0:2] for y in box[2:4] for z in box[4:6]], dtype=float)
    d = wc.wall_distances(wc.base_to_world(corners))
    assert (d >= s - 1.0).all()
    assert box[0] > box[1] and box[2] > box[3]


# ---------------------------------------------------------------- runtime guard
class FakeArm:
    """Minimal stand-in for XArmAPI used by SafetyMonitor."""

    def __init__(self, q_seq):
        self.q_seq = list(q_seq)
        self.i = 0
        self.state = 2
        self.mode = 1
        self.stop_calls = 0

    def get_servo_angle(self, is_radian=True):
        q = self.q_seq[min(self.i, len(self.q_seq) - 1)]
        self.i += 1
        return 0, list(q)

    def set_state(self, s):
        if s == 4:
            self.stop_calls += 1
            self.state = 4
        return 0

    def emergency_stop(self):
        self.set_state(4)

    def disconnect(self):
        pass


def test_monitor_trips_on_approach_with_lookahead(cfg):
    # Joint path that drives the elbow toward the -y wall.
    qs = [np.radians([90, -40 - k, 0, 60, 0, 120, 0]) for k in range(0, 61)]
    arm = FakeArm(qs)
    mon = SafetyMonitor(cfg, "fake", enforce=True, arm=arm)
    mon._t0 = 0.0
    import time as _time

    tripped_at = None
    t = _time.perf_counter()
    for k in range(len(qs)):
        mon._t0 = t - 0.01 * k  # pretend 10 ms between samples (fast motion)
        mon._poll_once()
        if mon.tripped.is_set():
            tripped_at = k
            break
    assert tripped_at is not None
    assert arm.stop_calls >= 1
    assert mon.trip_info["link"] in ("elbow_offset", "forearm", "elbow", "upper_arm")
    # it stopped before the arm geometry actually reached the wall
    assert mon.trip_info["clearance_mm"] > 0


def test_monitor_observe_only_does_not_stop(cfg):
    qs = [np.radians([90, -100, 0, 60, 0, 120, 0])] * 3
    arm = FakeArm(qs)
    mon = SafetyMonitor(cfg, "fake", enforce=False, arm=arm)
    mon._poll_once()
    assert arm.stop_calls == 0
    assert mon.would_trip_info is not None


def test_command_guard_refuses_joint_target(cfg):
    q0 = np.radians([90, -40, 0, 60, 0, 120, 0])
    arm = FakeArm([q0])
    mon = SafetyMonitor(cfg, "fake", enforce=True, arm=arm)
    guard = CommandGuard(cfg, mon, enforce=True, q_start=q0)
    guard.check_joint_target(np.radians([90, -45, 0, 60, 0, 120, 0]), 0)  # fine
    with pytest.raises(SafetyStop):
        guard.check_joint_target(np.radians([90, -100, 0, 60, 0, 120, 0]), 1)
    assert arm.stop_calls >= 1


def test_command_guard_refuses_cartesian_target(cfg):
    arm = FakeArm([Q_HOME])
    mon = SafetyMonitor(cfg, "fake", enforce=True, arm=arm)
    mon._poll_once()
    guard = CommandGuard(cfg, mon, enforce=True, q_start=Q_HOME)
    guard.check_cartesian_pose([300, 100, 300, math.pi, 0, 0], 0)
    with pytest.raises(SafetyStop):
        guard.check_cartesian_pose([300, 390, 300, math.pi, 0, 0], 1)


def test_clip_smoothing_never_shrinks_shift():
    from src.xarm_pipeline.safety.preprocess import smooth_shift_upper_envelope

    rng = np.random.default_rng(3)
    hard = np.zeros((200, 3))
    hard[60:120, 1] = -np.clip(np.sin(np.linspace(0, np.pi, 60)) * 200, 0, 150)
    hard[150:170, 2] = rng.uniform(0, 30, 20)
    soft = smooth_shift_upper_envelope(hard, 10)
    assert (np.abs(soft) >= np.abs(hard) - 1e-9).all()
    assert (np.sign(soft[np.abs(hard) > 0]) == np.sign(hard[np.abs(hard) > 0])).all()
    # smoother: the largest per-sample change is smaller than for the hard clip
    assert np.abs(np.diff(soft[:, 1], 2)).max() < np.abs(np.diff(hard[:, 1], 2)).max()
