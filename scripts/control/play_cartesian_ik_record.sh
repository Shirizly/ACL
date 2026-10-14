#!/usr/bin/env bash
# Play a Cartesian trajectory while recording the simulator, with a tracking-error report.
#   scripts/control/play_cartesian_ik_record.sh <traj.csv[#name]> [servo|pose|velocity] [seconds] [extra args]
# The trajectory is resolved to its clean copy in data/preprocessed/ first (pass --no-fit to skip).
# The approach move to the first part happens before recording starts.
# Extra recorder flags via XARM_RECORD_ARGS, e.g. XARM_RECORD_ARGS="--headless --arm-only --walls".
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_cartesian_circle.csv}"
CONTROL_MODE="${2:-servo}"
SECONDS_TO_RECORD="${3:-25}"
EXTRA_ARGS=("${@:4}")
IP="${XARM_IP:-127.0.0.1}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="outputs/reports/${RUN_ID}_${CONTROL_MODE}"
mkdir -p "$REPORT_DIR"

NO_FIT=0
for a in "${EXTRA_ARGS[@]}"; do [[ "$a" == "--no-fit" ]] && NO_FIT=1; done
FIRST="$TRAJ_FILE"
if [[ "$NO_FIT" == 0 ]]; then
  CFG_ARGS=()
  [[ -n "${XARM_SAFETY_CONFIG:-}" ]] && CFG_ARGS=(--config "$XARM_SAFETY_CONFIG")
  # Fit (if needed) before recording; play_cartesian_ik.sh then finds the clean copy up to date.
  FIRST="$(python -m src.xarm_pipeline.safety.preprocessed resolve "$TRAJ_FILE" "${CFG_ARGS[@]}" | head -n 1)"
  [[ -n "$FIRST" ]] || { echo "no executable trajectory for $TRAJ_FILE" >&2; exit 2; }
fi

XARM_IP="$IP" scripts/control/prepare_cartesian_start.sh "$FIRST" 0.5
EXTRA_Q=""
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && printf -v EXTRA_Q ' %q' "${EXTRA_ARGS[@]}"
CMD="XARM_PRESTART_FIRST=0 XARM_REPORT_DIR='$REPORT_DIR' scripts/control/play_cartesian_ik.sh '$TRAJ_FILE' '$CONTROL_MODE' --feedback-sample-period 0.0 --analyze-report${EXTRA_Q}"
RC=0
XARM_IP="$IP" python -m src.xarm_pipeline.record_web_sim \
  --seconds "$SECONDS_TO_RECORD" \
  --url "${SIM_WEB_URL:-http://127.0.0.1:18333}" \
  --output-dir "$REPORT_DIR" \
  --stop-when-done 1.5 \
  ${XARM_RECORD_ARGS:-} \
  --run-command "$CMD" || RC=$?

echo "Saved report directory: $REPORT_DIR"
exit "$RC"
