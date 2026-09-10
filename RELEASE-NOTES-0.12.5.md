# Reco Trainer 0.12.5 — Fix Training Crash on Resume After a New Class

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Fixed a training crash: resuming an interrupted training run hard-crashed with a `RuntimeError` ("size mismatch for model.class_embed.weight ...") if a new class had been annotated (e.g. via auto-labeling) since the run was interrupted, before a single epoch of the resumed run could complete. Reco Trainer now checks whether the interrupted checkpoint's class count still matches the project before resuming, and falls back to a normal fine-tuning cycle from the best available model instead of aborting.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.5.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.5.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.5.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.5.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
