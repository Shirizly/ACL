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
import math
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
        from src.xarm_pipeline.safety.retreat import is_simulator, plan_safe_return
        try:
            via_path, how = plan_safe_return(cfg, q_now[:7], np.radians(via_deg), simulator=is_simulator(ip))
        except RuntimeError as e:
            print(f'PRESTART REFUSED by safety walls: return to the ready pose: {e}')
            arm.disconnect()
            raise SystemExit(2)
        print('prestart return to ready pose: ' + how)
        res = validate_joint_trajectory(cfg, [0.0], [np.radians(q0_deg).tolist()],
                                        q_start=np.radians(via_deg).tolist(), trajectory_name='joint-space approach from ready pose')
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
    import numpy as np
    if os.environ.get('XARM_SAFETY_PREPROCESS', '1') == '0':
        via_path = [np.radians(via_deg)]
    for q_via in via_path:
        code = arm.set_servo_angle(angle=np.degrees(q_via).tolist(), is_radian=False, speed=30, wait=True)
        if code != 0:
            break
    if code == 0:
        code = arm.set_servo_angle(angle=q0_deg, is_radian=False, speed=30, wait=True)
    _, reached = arm.get_position(is_radian=True)
    err = math.dist(reached[:3], pose0[:3])
    print(f'prestart joint-space move rc={code}, TCP {err:.2f} mm from the trajectory start')
    tol = float(os.environ.get('XARM_PRESTART_TOL_MM', '2.0'))
    if err > tol:
        # Diagnose: TCP offset mismatch (config error) vs per-robot kinematic calibration (real arms differ from the
        # nominal DH by a few mm). Only the latter is corrected, by a short slow linear move onto the exact start.
        from src.xarm_pipeline.safety.config import load_safety_config
        from src.xarm_pipeline.safety.kinematics import check_dh_against_firmware, fk_pose
        from src.xarm_pipeline.safety.preprocess import (
            prepare_cartesian_poses, robot_tcp_offset, validate_cartesian_trajectory)
        cfg = load_safety_config(os.environ.get('XARM_SAFETY_CONFIG') or None)
        tcp_robot = list(arm.tcp_offset)
        tcp_cfg = list(cfg.firmware_tcp_offset)
        tcp_ok = max(abs(float(a) - float(b)) for a, b in zip(tcp_robot[:3], tcp_cfg[:3])) < 1.0
        _, q_here = arm.get_servo_angle(is_radian=True)
        fw_code, fw_pose = arm.get_forward_kinematics(q_here[:7], input_is_radian=True, return_is_radian=True)
        model_pose = fk_pose(q_here[:7], tcp_offset=cfg.firmware_tcp_offset_rad)
        fk_gap = max(abs(a - b) for a, b in zip(fw_pose[:3], model_pose[:3])) if fw_code == 0 else float('nan')
        try:
            dh_gap = f'{check_dh_against_firmware(arm):.3f}'
        except Exception as e:  # noqa: BLE001
            dh_gap = f'unavailable ({e})'
        print(f'PRESTART MISMATCH diagnosis: robot TCP offset {tcp_robot} vs config {tcp_cfg} '
              f'({"match" if tcp_ok else "DIFFERENT"}); firmware FK vs local model at these joints: {fk_gap:.2f} mm; '
              f'robot DH vs nominal: max diff {dh_gap}')
        # The robot owns the TCP and the IK: differences up to max_fix are expected (calibration, TCP definition)
        # and closed by a short wall-checked linear move. The safety checks keep using the pipeline's model.
        max_fix = float(os.environ.get('XARM_PRESTART_MAX_CORRECTION_MM', '15'))
        if not tcp_ok:
            print('WARNING: robot TCP offset differs from geometry.firmware_tcp_offset in the safety config; '
                  'the robot\'s TCP is used for motion, the config\'s for the safety model')
        if err > max_fix:
            print(f'PRESTART MISMATCH: TCP {err:.2f} mm from the trajectory start after the joint move, more than the '
                  f'{max_fix:.0f} mm the robot may correct (XARM_PRESTART_MAX_CORRECTION_MM); check the robot TCP '
                  'setting against geometry.firmware_tcp_offset in the safety config')
            arm.disconnect()
            raise SystemExit(2)
        start = fk_pose(q_here[:7], tcp_offset=robot_tcp_offset(arm, cfg))
        res = validate_cartesian_trajectory(cfg, [0.0], prepare_cartesian_poses([pose0], start), q_here[:7],
                                            robot_tcp_offset(arm, cfg), policy='reject')
        if not res.ok:
            print('PRESTART REFUSED: correction move to the trajectory start: ' + res.message)
            arm.disconnect()
            raise SystemExit(2)
        code = arm.set_position(x=pose0[0], y=pose0[1], z=pose0[2], roll=pose0[3], pitch=pose0[4], yaw=pose0[5],
                                is_radian=True, speed=20, wait=True)
        _, reached = arm.get_position(is_radian=True)
        err = math.dist(reached[:3], pose0[:3])
        print(f'prestart correction (robot calibration vs nominal model): linear move rc={code}, '
              f'TCP now {err:.2f} mm from the trajectory start')
        if code != 0 or err > tol:
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
