#!/bin/zsh
# Developer run: builds the payload (interface + workers) and starts the launcher against it.
set -euo pipefail
cd "${0:A:h}/.."
PAYLOAD_OUT="${TMPDIR:-/tmp}/reco-trainer-dev-payload"
../cross-platform/scripts/build-payload.sh "$PAYLOAD_OUT" >/dev/null
RECO_PAYLOAD_DIR="$PAYLOAD_OUT/payload" swift run RecoTrainerMac
