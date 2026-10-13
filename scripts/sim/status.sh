#!/usr/bin/env bash
set -euo pipefail

docker ps --filter name=uf_software

echo
echo "Web UI: http://127.0.0.1:18333"
