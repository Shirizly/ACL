#!/usr/bin/env bash
# Play a Cartesian trajectory with firmware IK.
#   scripts/control/play_cartesian_ik.sh <traj.csv[#name]> [servo|pose|velocity] [extra cartesian_playback args]
# By default the trajectory is first resolved to its clean copy in data/preprocessed/ (fitted into the
# workcell of the current safety config; created on first use). Split trajectories run part by part,
# each preceded by a safety-checked approach move (prepare_cartesian_start.sh).
# Pass --no-fit to play the file exactly as given (old behavior, no approach move).
# Env: XARM_IP, XARM_SAFETY_CONFIG (safety YAML), XARM_REPORT_DIR (per-part subdirs when split),
#      XARM_PRESTART_FIRST=0 (skip the approach move before the first part, e.g. done by the caller).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRAJ_FILE="${1:-data/trajectories/sample_cartesian_circle.csv}"
CONTROL_MODE="${2:-servo}"
EXTRA_ARGS=("${@:3}")
IP="${XARM_IP:-127.0.0.1}"

CFG_ARGS=()
PLAY_CFG_ARGS=()
if [[ -n "${XARM_SAFETY_CONFIG:-}" ]]; then
  CFG_ARGS=(--config "$XARM_SAFETY_CONFIG")
  PLAY_CFG_ARGS=(--safety-config "$XARM_SAFETY_CONFIG")
fi

NO_FIT=0
for a in "${EXTRA_ARGS[@]}"; do [[ "$a" == "--no-fit" ]] && NO_FIT=1; done
REFIT_ARGS=()
for a in "${EXTRA_ARGS[@]}"; do [[ "$a" == "--refit" ]] && REFIT_ARGS=(--refit); done

if [[ "$NO_FIT" == 1 ]]; then
  REPORT_ARGS=()
  [[ -n "${XARM_REPORT_DIR:-}" ]] && REPORT_ARGS=(--report-dir "$XARM_REPORT_DIR")
  exec python -m src.xarm_pipeline.cartesian_playback \
    --trajectory "$TRAJ_FILE" --control "$CONTROL_MODE" --ip "$IP" \
    "${PLAY_CFG_ARGS[@]}" "${REPORT_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

mapfile -t PARTS < <(python -m src.xarm_pipeline.safety.preprocessed resolve "$TRAJ_FILE" "${CFG_ARGS[@]}" "${REFIT_ARGS[@]}")
if [[ ${#PARTS[@]} -eq 0 ]]; then
  echo "no executable trajectory for $TRAJ_FILE (see message above)" >&2
  exit 2
fi
echo "executing ${#PARTS[@]} part(s): ${PARTS[*]}"

for i in "${!PARTS[@]}"; do
  PART="${PARTS[$i]}"
  if [[ $i -gt 0 || "${XARM_PRESTART_FIRST:-1}" != 0 ]]; then
    XARM_IP="$IP" scripts/control/prepare_cartesian_start.sh "$PART" 0.5
  fi
  REPORT_ARGS=()
  if [[ -n "${XARM_REPORT_DIR:-}" ]]; then
    if [[ ${#PARTS[@]} -gt 1 ]]; then
      REPORT_ARGS=(--report-dir "$XARM_REPORT_DIR/part$((i + 1))")
    else
      REPORT_ARGS=(--report-dir "$XARM_REPORT_DIR")
    fi
  fi
  python -m src.xarm_pipeline.cartesian_playback \
    --trajectory "$PART" --control "$CONTROL_MODE" --ip "$IP" \
    "${PLAY_CFG_ARGS[@]}" "${REPORT_ARGS[@]}" "${EXTRA_ARGS[@]}"
done
