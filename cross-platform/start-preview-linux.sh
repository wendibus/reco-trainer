#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then echo "Python 3.11 or 3.12 is required."; exit 1; fi
if ! command -v npm >/dev/null 2>&1; then echo "Node.js 22/npm is required."; exit 1; fi
if ! command -v ffmpeg >/dev/null 2>&1; then echo "FFmpeg is required to prepare videos on Linux."; exit 1; fi
if [ ! -d node_modules ]; then npm install; fi
python3 local_worker.py --port 8766 &
WORKER_PID=$!
npm run dev -- --host localhost --port 8765 &
SERVER_PID=$!
sleep 4
if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:8765/"; fi
trap 'kill "$SERVER_PID" "$WORKER_PID" 2>/dev/null || true' EXIT INT TERM
wait "$SERVER_PID"
