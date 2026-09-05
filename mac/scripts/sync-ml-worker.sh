#!/bin/zsh
# cross-platform/ml_worker.py is the single source of truth for the local RF-DETR
# worker; the Mac app bundles its own copy as a SwiftPM resource because Package
# resources must be real files under Sources/, not a symlink out of the package.
# Run this after editing cross-platform/ml_worker.py so both copies stay in sync
# (see .github/workflows/tests.yml, which fails CI if they ever drift apart).
set -euo pipefail
cd "${0:A:h}/../.."
cp cross-platform/ml_worker.py mac/Sources/RecoTrainerMac/Resources/ml_worker.py
echo "Synced cross-platform/ml_worker.py -> mac/Sources/RecoTrainerMac/Resources/ml_worker.py"
