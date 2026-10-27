# Reco Trainer 0.19.0 — Import YOLO and COCO Labels

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Added (all platforms):** labelled images from other tools can be imported into a project - **YOLO-txt** (including Ultralytics `images/` + `labels/` folders; polygon lines become boxes) and **COCO-JSON**. Run `ml_worker.py import-labels --project <project folder> --source <folder>` (see the README). There is no button in the app yet.
- Classes are mapped to the sport's Reco classes (for example `person` -> `player`); unknown classes are skipped and reported. Imported images are stored in `frames/imported/` and count as reviewed (`--candidates` imports them as suggestions instead). Importing the same image twice is skipped.
- **Windows/Linux/Docker:** imported frames are kept when the videos are prepared again.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.19.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.19.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.19.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.19.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
