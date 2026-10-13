#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

IP="${XARM_IP:-127.0.0.1}"

python - <<'PY'
import time
from xarm.wrapper import XArmAPI

ip = "127.0.0.1"
arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
arm.clean_error()
arm.clean_warn()
arm.motion_enable(True)
arm.set_mode(0)
arm.set_state(0)

waypoints = [
    [280.0, -120.0, 220.0, 3.141592653589793, 0.0, 0.0],
    [360.0, -120.0, 240.0, 3.141592653589793, 0.0, 0.2],
    [360.0, 40.0, 230.0, 3.141592653589793, 0.0, 0.4],
    [280.0, 40.0, 210.0, 3.141592653589793, 0.0, 0.2],
    [280.0, -120.0, 220.0, 3.141592653589793, 0.0, 0.0],
]

code0, p0 = arm.get_position(is_radian=True)
print(f"start_pose_code={code0} start_pose={p0}")

ok = 0
fail = 0
for wp in waypoints:
    rc = arm.set_position(
        x=wp[0], y=wp[1], z=wp[2],
        roll=wp[3], pitch=wp[4], yaw=wp[5],
        is_radian=True, wait=True, speed=80, mvacc=600,
    )
    if rc == 0:
        ok += 1
    else:
        fail += 1
    print(f"waypoint={wp} rc={rc}")
    time.sleep(0.2)

code1, p1 = arm.get_position(is_radian=True)
print(f"end_pose_code={code1} end_pose={p1}")
print(f"summary ok={ok} fail={fail}")
arm.disconnect()
PY
