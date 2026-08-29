#!/bin/zsh
set -e

PREVIEW_DIR="${0:A:h}"
PID_FILE="$PREVIEW_DIR/.preview-server.pid"
WORKER_PID_FILE="$PREVIEW_DIR/.local-worker.pid"
cd "$PREVIEW_DIR"

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js/npm wurde nicht gefunden. Bitte Node.js installieren und erneut starten."
  echo "Node.js/npm was not found. Install Node.js and start again."
  read -r "?Zum Schließen Enter drücken / Press Enter to close: "
  exit 1
fi

if [[ ! -d node_modules ]]; then
  echo "Richte Reco Trainer einmalig ein …"
  npm install
fi

PYTHON_BIN=""
for CANDIDATE in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 /usr/bin/python3; do
  if [[ -x "$CANDIDATE" ]]; then
    PYTHON_BIN="$CANDIDATE"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3 wurde nicht gefunden."
  read -r "?Zum Schließen Enter drücken: "
  exit 1
fi

SERVER_STARTED=0
WORKER_STARTED=0

if [[ -f "$WORKER_PID_FILE" ]] && WORKER_PID="$(<"$WORKER_PID_FILE")" && [[ "$WORKER_PID" == <-> ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
  :
else
  "$PYTHON_BIN" "$PREVIEW_DIR/local_worker.py" --port 8766 &
  WORKER_PID=$!
  WORKER_STARTED=1
  print -r -- "$WORKER_PID" > "$WORKER_PID_FILE"
fi

if [[ -f "$PID_FILE" ]] && SERVER_PID="$(<"$PID_FILE")" && [[ "$SERVER_PID" == <-> ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
  :
else
  npm run dev -- --host localhost --port 8765 &
  SERVER_PID=$!
  SERVER_STARTED=1
  print -r -- "$SERVER_PID" > "$PID_FILE"
fi

cleanup() {
  if [[ "$SERVER_STARTED" == 1 ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$WORKER_STARTED" == 1 ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
    kill "$WORKER_PID" 2>/dev/null || true
  fi
  if [[ "$SERVER_STARTED" == 1 ]] && [[ -f "$PID_FILE" ]] && [[ "$(<"$PID_FILE")" == "$SERVER_PID" ]]; then
    rm -f "$PID_FILE"
  fi
  if [[ "$WORKER_STARTED" == 1 ]] && [[ -f "$WORKER_PID_FILE" ]] && [[ "$(<"$WORKER_PID_FILE")" == "$WORKER_PID" ]]; then
    rm -f "$WORKER_PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

sleep 4
open "http://localhost:8765/"
echo "Reco Trainer läuft vollständig lokal auf http://localhost:8765/"
echo "Weboberfläche: PID $SERVER_PID · ML-Worker: PID $WORKER_PID"
echo "Dieses Fenster geöffnet lassen. Mit Ctrl+C wird Reco Trainer beendet."
if [[ "$SERVER_STARTED" == 1 ]]; then wait "$SERVER_PID"; else wait "$WORKER_PID"; fi
