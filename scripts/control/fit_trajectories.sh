#!/usr/bin/env bash
# Fit Cartesian trajectories into the safety workcell (re-place / rotate / split) and write clean copies
# into the preprocessed set of the current safety config:
#   data/trajectories/x.csv -> data/preprocessed/NNN_<box>/trajectories/x.csv
# (Playback does this automatically on first use; this prepares them ahead of time.)
#   scripts/control/fit_trajectories.sh data/trajectories/x.csv ["bank.csv#name" ...] [--config path.yaml]
#   scripts/control/fit_trajectories.sh data/trajectories/            # every CSV in a directory
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m src.xarm_pipeline.safety.fit "$@"
