#!/bin/sh
set -eu

python3 local_worker.py --port 8766 &
worker_pid=$!
npm run start -- --host 0.0.0.0 --port 8765 &
web_pid=$!

cleanup() {
  kill "$worker_pid" "$web_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait "$web_pid"
