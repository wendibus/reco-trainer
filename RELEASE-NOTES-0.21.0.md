# Reco Trainer 0.21.0 — Progress & Cancel for Every Job, Anonymized Dataset Export

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Progress window and Cancel for every long operation

- Preparing videos, setting up ML, auto-labeling, refining boxes, training, the model comparison, model export, packages, imports, downloads and the new dataset export all run as **jobs** with one progress window: the step ("Step 2/3: Train"), a progress bar (indeterminate when the amount is unknown), percent, elapsed and remaining time.
- **Cancel** (with a confirmation) stops the whole process tree - the ML worker with its data loaders, pip, FFmpeg - within seconds, removes half-written files and leaves the project unchanged. A cancelled **training keeps its last checkpoint**; the next run continues from there. Quitting the app also stops running jobs.
- Further operations wait in a queue. The window can be minimized to a status chip and comes back after a page reload. Progress is streamed with server-sent events.
- Training now reports **"Epoch n of m"**.

## Dataset export, optionally anonymized

- **Export dataset** writes every reviewed image with its labels as **YOLO** (Ultralytics) and/or **COCO** (Roboflow / RF-DETR), split by video like training, with neutral file names, a `metadata.json` (version, method, models with their licenses) and an optional ZIP.
- **Anonymized:** heads and faces are strongly blurred (or pixelated) inside soft-edged ellipses - no opaque boxes, so the head shape stays and the data remains useful for training. Labelled people, spectators and the bench (RF-DETR COCO base model) and faces found by YuNet are covered; jersey numbers and the scoreboard optionally. The ball and all labels stay unchanged.
- Frames in which a face is still detectable after blurring are **not exported** and are listed for review.
- **Anonymized training images:** optionally 10, 25 or 50 % of the training images are trained with blurred heads.

## Simpler field boundaries

- **Set field boundaries** no longer asks for the field's width and length. Click at least four points in order along the edge; straight lines join them. If the field is not fully in the image, put points on the image edge where it leaves the frame - any shape is possible. The check always happens in the image; existing outlines with measurements keep working as outlines. An outline whose edges cross cannot be saved.

## Models and licenses

- New model registry: only **Apache-2.0** and **MIT** models, each downloaded with a SHA-256 check; licenses are shown in the export dialog and written into every export. RF-DETR XL/2XL (PML), SAM 3 and Ultralytics models are not used.

## Mac

- Existing Mac apps (0.20.0) install this version themselves after asking. New installs: download the DMG.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.21.0.dmg` (signed and notarized)
- **Mac update package:** `Reco.Trainer.Payload.0.21.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.21.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.21.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.21.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. Model downloads come only from the sources listed in `model_registry.json`.
