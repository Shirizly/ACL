#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_cartesian_circle.csv}"
IP="${XARM_IP:-127.0.0.1}"
SETTLE_SECONDS="${2:-0.5}"
export TRAJ_FILE IP SETTLE_SECONDS

python - <<'PY'
from xarm.wrapper import XArmAPI
from src.xarm_pipeline.trajectory_io import load_cartesian_trajectory_csv
import time
import os

traj = load_cartesian_trajectory_csv(os.environ.get('TRAJ_FILE', 'data/trajectories/sample_cartesian_circle.csv'))
ip = os.environ.get('XARM_IP', '127.0.0.1')
settle_seconds = float(os.environ.get('SETTLE_SECONDS', '0.5'))

arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
arm.clean_error()
arm.clean_warn()
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)

pose0 = traj.pose[0]

# Clean copies from data/preprocessed/ carry the joint configuration their fit validated at the
# first pose. Go there in joint space so playback starts in the arm configuration that was checked
# (a linear Cartesian approach can leave the redundant elbow elsewhere).
# The fit validated this approach: joint move to its seed ('via', the ready configuration), then a
# joint move to the start configuration.
from src.xarm_pipeline.safety.preprocessed import approach_for
approach = approach_for(os.environ.get('TRAJ_FILE', ''))
via_deg, q0_deg = approach if approach else (None, None)

# Safety walls: check the approach move (current joints -> first pose) before
# moving. Disable only for demos with XARM_SAFETY_PREPROCESS=0.
if os.environ.get('XARM_SAFETY_PREPROCESS', '1') != '0':
    import math
    import numpy as np
    from src.xarm_pipeline.safety.config import load_safety_config
    from src.xarm_pipeline.safety.kinematics import fk_pose
    from src.xarm_pipeline.safety.preprocess import (
        prepare_cartesian_poses, robot_tcp_offset, validate_cartesian_trajectory)
    from src.xarm_pipeline.safety.preprocess import validate_joint_trajectory
    cfg = load_safety_config(os.environ.get('XARM_SAFETY_CONFIG') or None)
    _, q_now = arm.get_servo_angle(is_radian=True)
    tcp = robot_tcp_offset(arm, cfg)
    if q0_deg is not None:
        res = validate_joint_trajectory(cfg, [0.0, 1.0], [np.radians(via_deg).tolist(), np.radians(q0_deg).tolist()],
                                        q_start=q_now[:7], trajectory_name='joint-space approach via ready pose')
    else:
        start = fk_pose(q_now[:7], tcp_offset=tcp)
        target = prepare_cartesian_poses([pose0], start)
        res = validate_cartesian_trajectory(cfg, [0.0], target, q_now[:7], tcp, policy='reject')
    if not res.ok:
        print('PRESTART REFUSED by safety walls: ' + res.message)
        arm.disconnect()
        raise SystemExit(2)
    print('prestart safety check: ' + res.message)
else:
    print('WARNING: prestart safety check DISABLED (XARM_SAFETY_PREPROCESS=0)')

if q0_deg is not None:
    code = arm.set_servo_angle(angle=via_deg, is_radian=False, speed=30, wait=True)
    if code == 0:
        code = arm.set_servo_angle(angle=q0_deg, is_radian=False, speed=30, wait=True)
    _, reached = arm.get_position(is_radian=True)
    err = max(abs(a - b) for a, b in zip(reached[:3], pose0[:3]))
    print(f'prestart joint-space move rc={code}, TCP {err:.2f} mm from the trajectory start')
    if err > 2.0:
        print('PRESTART MISMATCH: the stored start joints do not reach the trajectory start; '
              'check that the robot TCP offset matches geometry.firmware_tcp_offset in the safety config')
        arm.disconnect()
        raise SystemExit(2)
else:
    code = arm.set_position(
        x=pose0[0],
        y=pose0[1],
        z=pose0[2],
        roll=pose0[3],
        pitch=pose0[4],
        yaw=pose0[5],
        is_radian=True,
        wait=True,
    )
print(f'prestart_rc={code}')
if settle_seconds > 0.0:
    time.sleep(settle_seconds)
print('prestart_ready')
arm.disconnect()
PY
