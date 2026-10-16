#!/usr/bin/env bash
# Prepare recorded trajectories for the robot: time-scale, cut too-fast spans, fit/split into the cell.
#   scripts/control/prepare_recorded.sh <dir> <time_scale> <number> [number ...] [extra args]
#   e.g. scripts/control/prepare_recorded.sh data/recorded_trajectories/5ec359d2 2.5 18 9
# Output: data/preprocessed/NNN_*/recorded/<slate>/<number>_t<scale>.csv (+ .fit.json, .prep.json)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DIR="${1:?usage: $0 <dir> <time_scale> <number> [number ...]}"
SCALE="${2:?time scale}"
shift 2
python -m src.xarm_pipeline.safety.recorded "$DIR" "$@" --time-scale "$SCALE"
