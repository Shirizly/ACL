#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_cartesian_circle.csv}"
CONTROL_MODE="${2:-servo}"
EXTRA_ARGS=("${@:3}")
IP="${XARM_IP:-127.0.0.1}"

python -m src.xarm_pipeline.cartesian_playback \
  --trajectory "$TRAJ_FILE" \
  --control "$CONTROL_MODE" \
  --ip "$IP" \
  "${EXTRA_ARGS[@]}"
