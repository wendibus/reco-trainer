#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
swift run RecoTrainerMac
