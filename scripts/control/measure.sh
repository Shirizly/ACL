#!/usr/bin/env bash
# Measure the workcell with the robot and check the gripper model (see docs/safety_walls.md,
# "Measuring the cell and checking the gripper model on the real robot").
#   scripts/control/measure.sh where                      # model vs walls at the current pose (no motion)
#   scripts/control/measure.sh watch [--stop]             # live clearance while jogging by hand
#   scripts/control/measure.sh walls                      # touch-off -> workcell.walls_in_base_mm
#   scripts/control/measure.sh wall-test --direction y+   # slow stop test against a virtual wall (moves!)
# Robot IP from XARM_IP (default 127.0.0.1 = simulator).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CMD="${1:?usage: $0 where|watch|walls|wall-test [args]}"
shift
python -m src.xarm_pipeline.safety.measure "$CMD" --ip "${XARM_IP:-127.0.0.1}" "$@"
