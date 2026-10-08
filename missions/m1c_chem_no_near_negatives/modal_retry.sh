#!/bin/bash
# usage: modal_retry.sh <log> <modal run args...>; retries while the workspace is at its ephemeral-app limit
log=$1; shift
for i in $(seq 1 120); do
  caffeinate -i uv run modal run "$@" > "$log" 2>&1
  if grep -q "limit of 100 ephemeral apps" "$log"; then sleep 30; else exit 0; fi
done
