#!/bin/bash
# Usage: run_retry.sh <log> <modal run args...>; retries while the workspace is at its ephemeral-app limit.
log=$1; shift
for i in $(seq 1 60); do
  caffeinate -i uv run modal run "$@" > "$log" 2>&1
  if ! grep -q "limit of 100 ephemeral apps\|modified during build process\|StreamTerminatedError\|Connection lost" "$log"; then exit 0; fi
  sleep 30
done
