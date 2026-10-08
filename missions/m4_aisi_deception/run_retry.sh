#!/bin/bash
# usage: run_retry.sh <logfile> <modal run args...>; retries while the workspace is at its ephemeral-app limit
log=$1; shift
for i in $(seq 1 120); do
  caffeinate -i uv run modal run "$@" > "$log" 2>&1
  if grep -qE "limit of 100 ephemeral apps|modified during build process" "$log"; then sleep 30; else break; fi
done
