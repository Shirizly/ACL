#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_sine_joint.csv}"
IP="${XARM_IP:-127.0.0.1}"

python -m src.xarm_pipeline.playback \
  --trajectory "$TRAJ_FILE" \
  --control position \
  --ip "$IP"
