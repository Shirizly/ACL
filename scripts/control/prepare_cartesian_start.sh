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
print(f'prestart_set_position_rc={code}')
if settle_seconds > 0.0:
    time.sleep(settle_seconds)
print('prestart_ready')
arm.disconnect()
PY
