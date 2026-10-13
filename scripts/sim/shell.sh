#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-uf_software}"
docker exec -it "$CONTAINER_NAME" /bin/bash
