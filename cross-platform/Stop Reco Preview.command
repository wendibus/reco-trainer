#!/bin/zsh

PREVIEW_DIR="${0:A:h}"
PID_FILE="$PREVIEW_DIR/.preview-server.pid"
WORKER_PID_FILE="$PREVIEW_DIR/.local-worker.pid"

STOPPED=0

if [[ -f "$PID_FILE" ]]; then
  SERVER_PID="$(<"$PID_FILE")"
  if [[ "$SERVER_PID" == <-> ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    STOPPED=1
  fi
  rm -f "$PID_FILE"
fi

if [[ -f "$WORKER_PID_FILE" ]]; then
  WORKER_PID="$(<"$WORKER_PID_FILE")"
  if [[ "$WORKER_PID" == <-> ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID"
    STOPPED=1
  fi
  rm -f "$WORKER_PID_FILE"
fi

if [[ "$STOPPED" == 1 ]]; then
  echo "Reco Trainer und der lokale ML-Worker wurden beendet."
else
  echo "Die Vorschau läuft nicht über den Doppelklick-Starter."
fi
