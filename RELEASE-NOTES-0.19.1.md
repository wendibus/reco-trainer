# Reco Trainer 0.19.1 — "Import labels" Button

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Added (Mac app and web UI):** an **Import labels** button. Choose a folder with images and **YOLO-txt** or **COCO-JSON** labels; Reco Trainer adds them to the open project, reloads it and reports the number of images and boxes and any classes it skipped.
- **Import as suggestions:** on the Mac via the button's menu, in the web UI via "Import labels: As suggestions". The boxes then go unreviewed into the review queue.
- No ML environment is needed for the import. Class names come from `data.yaml`, `classes.txt`, or the COCO file; for a custom class mapping, `ml_worker.py import-labels --class-map` remains available (see the README).
- **Mac:** `mac/scripts/notarize.sh` now reads the version from `Info.plist` instead of a fixed 0.6.0 DMG name.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.19.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.19.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.19.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.19.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
