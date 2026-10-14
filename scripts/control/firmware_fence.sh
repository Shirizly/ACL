#!/usr/bin/env bash
# Inspect / apply / reset the xArm firmware TCP safety boundary derived from config/safety.yaml.
#   scripts/control/firmware_fence.sh status|apply|reset|compute
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m src.xarm_pipeline.safety.firmware_fence --ip "${XARM_IP:-127.0.0.1}" "${1:-status}"
