#!/usr/bin/env bash
# Move the arm to the ready pose (joints 0,-28.17,0,23.24,0,51.41,0 deg: flange at (300,0,300) mm, tool down).
#   scripts/control/go_ready.sh            # checks the joint move against the safety config, asks before moving
#   scripts/control/go_ready.sh --yes      # no question (still checked)
#   scripts/control/go_ready.sh --force    # SIMULATOR ONLY: skip the check (after a sim restart the arm is in the
#                                          # folded zero pose, where the gripper model is below the floor)
# Clears errors/warnings and enables the motors first. Robot IP from XARM_IP (default 127.0.0.1 = simulator).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

YES=0
FORCE=0
for a in "$@"; do
  case "$a" in
    --yes) YES=1 ;;
    --force) FORCE=1 ;;
    *) echo "unknown option $a" >&2; exit 1 ;;
  esac
done
export YES FORCE

python - <<'PY'
import os, sys
import numpy as np
from xarm.wrapper import XArmAPI
from src.xarm_pipeline.safety.config import load_safety_config
from src.xarm_pipeline.safety.fit import load_fit_config
from src.xarm_pipeline.safety.preprocess import validate_joint_trajectory

ip = os.environ.get("XARM_IP", "127.0.0.1")
cfg_path = os.environ.get("XARM_SAFETY_CONFIG") or None
ready_deg = list(load_fit_config(cfg_path).seed_joints_deg)
arm = XArmAPI(ip, baud_checkset=False, check_joint_limit=False)
if not arm.connected:
    sys.exit(f"cannot connect to {ip}")
arm.clean_error(); arm.clean_warn(); arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
_, q_now = arm.get_servo_angle(is_radian=True)
print(f"robot {ip}: joints now {np.degrees(q_now[:7]).round(1).tolist()} deg -> ready {ready_deg}")
if os.environ["FORCE"] == "1":
    print("WARNING: safety check SKIPPED (--force). Use only on the simulator.")
else:
    res = validate_joint_trajectory(load_safety_config(cfg_path), [0.0], [np.radians(ready_deg).tolist()],
                                    q_start=q_now[:7], trajectory_name="move to ready")
    print("safety check of the move: " + res.message)
    if not res.ok:
        arm.disconnect()
        sys.exit("NOT MOVED. Jog the arm (Studio) to a clearer pose first.")
if os.environ["YES"] != "1" and input("Move to the ready pose now (20 deg/s)? Type 'yes': ").strip().lower() != "yes":
    arm.disconnect()
    sys.exit("not moved")
rc = arm.set_servo_angle(angle=ready_deg, is_radian=False, speed=20, wait=True)
_, pos = arm.get_position()
print(f"rc={rc}; flange at {[round(v) for v in pos[:3]]} mm; state {arm.state}, error {arm.error_code}")
arm.disconnect()
sys.exit(0 if rc == 0 else 3)
PY
