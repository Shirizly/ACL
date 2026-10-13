#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-uf_software}"

echo "Starting UFACTORY Studio + xArm 7 firmware in ${CONTAINER_NAME}..."
docker exec -it "$CONTAINER_NAME" /xarm_scripts/xarm_start.sh 7 7
