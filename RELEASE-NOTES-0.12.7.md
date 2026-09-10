# Reco Trainer 0.12.7 — Person-as-Player Default and Per-Category Benchmark Comparison

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- A freshly loaded project now preselects every category the current model actually supports for auto-labeling — with the base model, that's ball and player. Since the base model detects every person generically as "player", only the referee needs to be clicked and relabeled individually (via the existing click-to-relabel annotation editor) instead of being drawn by hand.
- The disabled referee/goalkeeper auto-label chip tooltip now explains this player-then-relabel workflow instead of just saying the class is unsupported.
- The model comparison now also shows a table of mAP@.50 broken down by category and model, alongside the overall ranking — so it's visible at a glance if, say, one model is better at balls and another is better at referees.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.7.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.7.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.7.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.7.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
