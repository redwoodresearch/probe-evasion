#!/bin/bash
# usage: retry_modal.sh <log> <modal args...>   retries while the workspace is at its ephemeral-app limit
log=$1; shift
for i in $(seq 1 60); do
  caffeinate -i uv run modal run "$@" > "$log" 2>&1
  grep -q "limit of 100 ephemeral apps" "$log" || exit 0
  sleep 30
done
