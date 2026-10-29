#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if ! command -v python3 >/dev/null 2>&1; then echo "Python 3.10 or newer (3.12 recommended) is required."; exit 1; fi
if ! command -v ffmpeg >/dev/null 2>&1; then echo "FFmpeg is required to prepare videos on Linux."; exit 1; fi
if [ ! -f ui/index.html ]; then echo "The interface (ui/ folder) is missing. Please download the release package again."; exit 1; fi
python3 local_worker.py --port 8766 &
WORKER_PID=$!
trap 'kill "$WORKER_PID" 2>/dev/null || true' EXIT INT TERM
sleep 2
if command -v xdg-open >/dev/null 2>&1; then xdg-open "http://localhost:8766/"; fi
echo "Reco Trainer runs locally at http://localhost:8766/ - press Ctrl+C to stop."
wait "$WORKER_PID"
