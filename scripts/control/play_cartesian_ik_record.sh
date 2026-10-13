#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_cartesian_circle.csv}"
CONTROL_MODE="${2:-servo}"
SECONDS_TO_RECORD="${3:-25}"
IP="${XARM_IP:-127.0.0.1}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="outputs/reports/${RUN_ID}_${CONTROL_MODE}"
mkdir -p "$REPORT_DIR"

TRAJ_FILE="$TRAJ_FILE" XARM_IP="$IP" SETTLE_SECONDS="0.5" scripts/control/prepare_cartesian_start.sh "$TRAJ_FILE" 0.5
CMD="scripts/control/play_cartesian_ik.sh '$TRAJ_FILE' '$CONTROL_MODE' --report-dir '$REPORT_DIR' --feedback-sample-period 0.0 --analyze-report"
XARM_IP="$IP" python -m src.xarm_pipeline.record_web_sim \
  --seconds "$SECONDS_TO_RECORD" \
  --url "${SIM_WEB_URL:-http://127.0.0.1:18333}" \
  --output-dir "$REPORT_DIR" \
  --run-command "$CMD"

echo "Saved report directory: $REPORT_DIR"
