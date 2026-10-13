#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_cartesian_circle.csv}"
CONTROL_MODE="${2:-servo}"
DEFAULT_ORIENTATION_RPY="${3:-}"
IP="${XARM_IP:-127.0.0.1}"

EXTRA_ARGS=("${@:4}")

if [[ -n "$DEFAULT_ORIENTATION_RPY" ]]; then
  python -m src.xarm_pipeline.cartesian_playback \
    --trajectory "$TRAJ_FILE" \
    --control "$CONTROL_MODE" \
    --ip "$IP" \
    --default-orientation-rpy "$DEFAULT_ORIENTATION_RPY" \
    "${EXTRA_ARGS[@]}"
else
  python -m src.xarm_pipeline.cartesian_playback \
    --trajectory "$TRAJ_FILE" \
    --control "$CONTROL_MODE" \
    --ip "$IP" \
    "${EXTRA_ARGS[@]}"
fi
