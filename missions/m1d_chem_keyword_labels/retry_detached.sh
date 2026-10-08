#!/bin/bash
# usage: retry_detached.sh <log> <modal run args...>   detached launch, retried while the workspace is at its app limit
log=$1; shift
for i in $(seq 1 120); do
  caffeinate -i uv run modal run --detach "$@" > "$log" 2>&1
  grep -qE "limit of 100 ephemeral apps|modified during build process" "$log" || exit 0
  sleep 30
done
