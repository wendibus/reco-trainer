# Reco Trainer 0.15.3 — Ball-Tracking Simulation Fix (Windows/Linux/Docker)

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Fixed:** in the Windows/Linux/Docker web interface, the ball-tracking simulation could fail with `Expecting ',' delimiter: line 1 column 6 (char 5)`. The local worker parsed the entire output of the ML process as one JSON document, so any log line printed ahead of the result (for example a timestamped notice from the ML libraries while loading the model) broke it. Only the last output line - the actual result - is parsed now, matching the fix the Mac app already received in 0.14.x.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.15.3.dmg`
- **Windows:** `Reco.Trainer.Windows.0.15.3.zip`
- **Linux:** `Reco.Trainer.Linux.0.15.3.zip`
- **Docker:** `Reco.Trainer.Docker.0.15.3.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
