#!/usr/bin/env bash
# usage: launch.sh <label> <entrypoint> [args...]   (detached driver for batch d; log and pid in logs/d_<label>.*)
set -euo pipefail
label="$1"; shift; entry="$1"; shift
name="probe-evasion_nabil-baugher_m49d-${label}_kill-2026-11-30"
[[ "$name" =~ ^probe-evasion_nabil-baugher_m49d-[a-z0-9-]+_kill-2026-11-30$ ]] || { echo "bad app name $name"; exit 1; }
cd /Users/nabilbaugher/dev/probe-evasion
log="missions/m49_new_task_toggle/logs/d_${label}.log"
[[ -e "$log" ]] && mv "$log" "${log%.log}_$(date +%H%M%S).log"
nohup caffeinate -i uv run modal run ${DETACH:+--detach} -n "$name" -m "probe_evasion.missions.m49_new_task_toggle::${entry}" "$@" > "$log" 2>&1 &
echo $! > "missions/m49_new_task_toggle/logs/d_${label}.pid"
echo "launched $name (pid $!)"
