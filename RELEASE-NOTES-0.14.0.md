# Reco Trainer 0.14.0 — Review Priority: Model Disagreement & Temporal Outliers

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Added:** two new signals automatically flag auto-labeled frames for priority review, so the frames most likely to need a second look surface first in the review queue.
  - **Ensemble disagreement** (opt-in, roughly doubles auto-label inference time): when a second compatible model is installed, a new toggle ("Use a second model to flag uncertain frames") runs it as a comparison-only pass and flags frames where the two models disagree about what's in them. It never writes the second model's boxes as annotations — only the first model's boxes are ever used for training.
  - **Temporal-consistency outliers** (always on, no extra inference cost): reuses the trajectory-reasoning idea behind the ball-tracking simulation (0.13.0) to flag a frame whose single-instance detection (e.g. the ball) sits far from where its temporal neighbors in the same video would put it.
  - Flagged frames get a warning badge with a tooltip explaining why, and sort first in both the regular review queue and the independent-model-test queue.
- Both signals are pure review-priority hints — they never change what gets trained, only which candidate frames a reviewer sees first.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.14.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.14.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.14.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.14.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
