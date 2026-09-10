# Reco Trainer 0.12.6 — Fix a Second Training Crash After a New Class

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Fixed a training crash that could still occur after 0.12.5's fix: once an interrupted-run resume was correctly skipped for a class-count mismatch, the fallback fine-tuning cycle could still build its model from the checkpoint's stale, smaller class count (RF-DETR infers the class count from the referenced checkpoint rather than the current dataset), crashing later deep inside training once a batch had an annotation for a class index beyond that count. Reco Trainer now always computes the model's class count from the project's current dataset.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.6.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.6.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.6.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.6.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
