#!/bin/zsh

APP_DIR="${0:A:h}"
WORKER_PID_FILE="$APP_DIR/.local-worker.pid"

if [[ -f "$WORKER_PID_FILE" ]]; then
  WORKER_PID="$(<"$WORKER_PID_FILE")"
  if [[ "$WORKER_PID" == <-> ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID"
    rm -f "$WORKER_PID_FILE"
    echo "Reco Trainer wurde beendet."
    exit 0
  fi
  rm -f "$WORKER_PID_FILE"
fi
echo "Reco Trainer läuft nicht über den Doppelklick-Starter."
