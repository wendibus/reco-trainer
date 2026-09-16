# Reco Trainer 0.12.15 — Independent Model Test, Combined Models

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **New: Independent model test.** A new "Unabhängiger Modelltest" / "Independent model test" section lets you review a video folder that was never used for training. With app assistance across every category (not just one at a time), it becomes a genuinely independent test set. The model benchmark now requires this kind of independent, held-out images for its reference instead of any reviewed training images — so a model can no longer score well on the comparison simply because it already saw that exact footage during training. If you had an existing benchmark reference, it needs to be refrozen from an independent folder.
- **New: Combine models.** The model benchmark can now "bake" a new combined model out of several installed models: pick the strongest model per category — one model for referees, another for the ball, for example. This is not real weight merging (not technically possible with RF-DETR's architecture); each source model simply handles its assigned categories at inference time. The result installs into the model library like any other model.
- The Windows/Linux/Docker web interface now also shows the per-category comparison table (mAP@.50 broken down by category), matching what the Mac app has had since 0.12.9.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.15.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.15.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.15.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.15.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only — including a combined model's package, which bundles one weights file per source model plus which categories each handles.
