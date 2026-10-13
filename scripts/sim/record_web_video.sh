#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SECONDS_TO_RECORD="${1:-20}"

python -m src.xarm_pipeline.record_web_sim \
  --seconds "$SECONDS_TO_RECORD" \
  --url "${SIM_WEB_URL:-http://127.0.0.1:18333}" \
  --output-dir "outputs/videos"
