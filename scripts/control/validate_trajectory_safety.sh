#!/usr/bin/env bash
# Check a trajectory CSV against the safety walls (config/safety.yaml).
# Joint CSVs need no robot. For Cartesian CSVs set XARM_IP to use the robot's
# current joints as the IK seed (otherwise an offline default seed is used).
#   scripts/control/validate_trajectory_safety.sh data/trajectories/x.csv [reject|clip] [extra args...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:?usage: $0 <trajectory.csv> [reject|clip] [extra args]}"
POLICY="${2:-reject}"
shift $(( $# >= 2 ? 2 : 1 ))
TRAJ_BASE="$(basename "${TRAJ_FILE%%#*}" .csv)"            # file.csv#name -> file
[[ "$TRAJ_FILE" == *"#"* ]] && TRAJ_BASE="${TRAJ_BASE}_${TRAJ_FILE##*#}"
OUT_DIR="outputs/safety_checks/$(date +%Y%m%d_%H%M%S)_${TRAJ_BASE}"
mkdir -p "$OUT_DIR"

IP_ARGS=()
if [[ -n "${XARM_IP:-}" ]]; then
  IP_ARGS=(--ip "$XARM_IP")
fi

python -m src.xarm_pipeline.safety.validate \
  --trajectory "$TRAJ_FILE" \
  --policy "$POLICY" \
  --report "$OUT_DIR/safety_report.json" \
  --sanitized-out "$OUT_DIR/sanitized_trajectory.csv" \
  "${IP_ARGS[@]}" "$@"
