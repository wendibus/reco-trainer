#!/bin/zsh
set -e

APP_DIR="${0:A:h}"
WORKER_PID_FILE="$APP_DIR/.local-worker.pid"
PORT=8766
URL="http://localhost:$PORT/"
cd "$APP_DIR"

if [[ ! -f ui/index.html ]]; then
  echo "Die Oberfläche (Ordner ui/) fehlt. Bitte das Release-Paket neu laden."
  echo "The interface (ui/ folder) is missing. Please download the release package again."
  read -r "?Zum Schließen Enter drücken / Press Enter to close: "
  exit 1
fi

PYTHON_BIN=""
for CANDIDATE in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 /opt/homebrew/bin/python3.11 /usr/local/bin/python3.11 /opt/homebrew/bin/python3.13 /usr/local/bin/python3.13 /usr/bin/python3; do
  if [[ -x "$CANDIDATE" ]]; then
    PYTHON_BIN="$CANDIDATE"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  echo "Python 3 wurde nicht gefunden."
  echo "Python 3 was not found."
  read -r "?Zum Schließen Enter drücken: "
  exit 1
fi

# Already running (for example started earlier): just open the interface.
if [[ -f "$WORKER_PID_FILE" ]] && WORKER_PID="$(<"$WORKER_PID_FILE")" && [[ "$WORKER_PID" == <-> ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
  open "$URL"
  echo "Reco Trainer läuft bereits: $URL"
  exit 0
fi

"$PYTHON_BIN" "$APP_DIR/local_worker.py" --port "$PORT" &
WORKER_PID=$!
print -r -- "$WORKER_PID" > "$WORKER_PID_FILE"

cleanup() {
  kill "$WORKER_PID" 2>/dev/null || true
  if [[ -f "$WORKER_PID_FILE" ]] && [[ "$(<"$WORKER_PID_FILE")" == "$WORKER_PID" ]]; then
    rm -f "$WORKER_PID_FILE"
  fi
}
trap cleanup EXIT INT TERM

sleep 2
open "$URL"
echo "Reco Trainer läuft vollständig lokal auf $URL"
echo "Dieses Fenster geöffnet lassen. Mit Ctrl+C wird Reco Trainer beendet."
wait "$WORKER_PID"
