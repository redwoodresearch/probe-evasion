#!/usr/bin/env bash
# usage: launch.sh <label> <entrypoint> [args...]   (detached local driver; log and pid in logs/<label>.*)
set -euo pipefail
label="$1"; shift; entry="$1"; shift
name="probe-evasion_nabil-baugher_m46-${label}_kill-2026-11-30"
[[ "$name" =~ ^probe-evasion_nabil-baugher_m46-[a-z0-9-]+_kill-2026-11-30$ ]] || { echo "bad app name $name"; exit 1; }
cd /Users/nabilbaugher/dev/probe-evasion
log="missions/m46_monitored_model_axis/logs/${label}.log"
nohup caffeinate -i uv run modal run -n "$name" -m "probe_evasion.missions.m46_monitored_model_axis::${entry}" "$@" > "$log" 2>&1 &
echo $! > "missions/m46_monitored_model_axis/logs/${label}.pid"
echo "launched $name (pid $!)"
