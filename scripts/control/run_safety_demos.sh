#!/usr/bin/env bash
# Record the safety-wall demos in the simulator (see docs/safety_walls.md).
# Requires: scripts/sim/up.sh + firmware running, Playwright Chromium, imageio-ffmpeg.
#   scripts/control/run_safety_demos.sh                       # all runs
#   scripts/control/run_safety_demos.sh --only tcp_wall_runtime_guard
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m src.xarm_pipeline.safety.demo --ip "${XARM_IP:-127.0.0.1}" "$@"
