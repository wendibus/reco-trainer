# Reco Trainer 0.14.1 — Train From Scratch

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Added:** a "Train from scratch" switch that ignores any existing checkpoint for one run and guarantees a start from the Apache-2.0 base model. When continuing from a checkpoint that has fewer classes than the project (for example a ball-only model after player and referee were added), RF-DETR keeps the checkpoint's smaller class count instead of expanding it, so the new classes could not be learned.
- **Changed:** a newly trained model whose classes differ from the active model's is no longer judged "worse" purely by its overall quality score, since those scores average over different class sets. It is archived, and the log points to the per-category model comparison and combined models.
- **Fixed:** the training record now reports the correct starting point ("base") for from-scratch runs.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.14.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.14.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.14.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.14.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
