#!/bin/zsh
set -euo pipefail
cd "${0:A:h}/.."
python3 Sources/RecoTrainerMac/Resources/ml_worker.py doctor
python3 -m unittest discover -s Tests/Python -p 'test_*.py'
