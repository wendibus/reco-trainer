# Reco Trainer 0.12.13 — Windows "Set Up ML" Fix, Model Classes in the Library

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Windows fix:** "Set Up ML" (and every local ML step after it) failed with "the system cannot find the file specified", because the app internally expected a macOS/Linux-only file path for the Python virtual environment (`venv/bin/python3`) instead of Windows' actual layout (`venv\Scripts\python.exe`). Fixed — this affects the Windows/Linux/Docker builds' local ML worker.
- Every model entry in the model library now shows directly which classes that model actually knows (e.g. "ball, player") — previously this could only be found by manually inspecting the model package's manifest.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.13.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.13.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.13.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.13.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
