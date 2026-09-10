# Reco Trainer 0.12.4 — Multi-Class Labeling, Box Editing, Faster Extraction

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- Auto-labeling can now detect several selected classes in a single pass instead of requiring one full run per class. Classes the current model can't detect yet (no COCO alias and no custom-trained checkpoint) are greyed out in the selection instead of silently doing nothing.
- The OpenCV box refinement pass, previously limited to ball/puck, now covers every sport category (player, referee, goalkeeper, hoop, goal, goalpost) via a per-category plausibility table.
- The Mac app's annotation editor can now select, move, delete, and relabel existing boxes (e.g. player ↔ referee) by clicking them - previously the only way to fix a box was deleting and redrawing it from scratch.
- Preparing videos locally on the Mac app now reads each video in a single sequential pass instead of re-seeking separately for every extracted frame, which noticeably speeds up frame extraction.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.4.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.4.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.4.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.4.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
