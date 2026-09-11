# Reco Trainer 0.12

> **Work in progress / alpha software.** Reco Trainer is ready for practical testing, but it is not a finished production release. Keep backups and validate every label, benchmark result and exported model before using it in a real workflow.

Reco Trainer is a privacy-first tool for improving and comparing sports-camera detection models. Sensitive videos stay on the user's own computer: frames are extracted locally, automatic suggestions are corrected locally, and RF-DETR models are trained and tested locally. Only an explicitly exported `.recomodel` package is intended to be shared. It contains model weights, checksums and aggregate metadata—not videos, frames, local paths or video file names.

The alpha supports basketball, football (soccer), futsal, handball, hockey, rugby, lacrosse and American football. The interface and walkthrough are available in German, English, Spanish and French.

## New in 0.12: safer editing and clearer exchange

Annotation editing now keeps exactly ten undo and redo steps per image, available from the buttons or with `Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z`. Before creating a `.recomodel` exchange package, Reco Trainer asks for a readable package name and uses it in both the package metadata and safe file name.

The local model benchmark now includes an expandable glossary explaining quality score, mAP@0.50, precision, recall, F1, false positives, false negatives, inference time and the confidence threshold in plain language.

## New in 0.11: versioned model library

Every completed local training run is copied into an immutable entry below `.reco-training/models/library/`. Reco Trainer shows a readable name, model size, creation time, independent test mAP when available (otherwise validation mAP), active state and best marker. Revisions can be renamed, activated and deleted; the active revision is protected from deletion.

Before the first 0.11 training run, Reco Trainer also preserves the mutable checkpoint left by an older version. After training, a new revision is activated only if its comparable independent test mAP improves on the active model; validation mAP is the fallback when no test result exists. A worse run remains available for inspection and benchmarking but cannot silently replace the better model.

## New in 0.9: reviewed active learning

Choose **Review new videos** and select a separate folder that was not used for the original training set. Reco Trainer extracts candidates locally, runs the current model at a recall-oriented threshold, and opens a review queue ordered toward uncertain detections.

Each candidate needs one deliberate decision: **Ball**, **No ball**, correct the box, or **Skip**. Only reviewed candidates enter training. Pending candidates are technically excluded from dataset generation, model packages, and benchmarks. Source videos stay unchanged, and videos already used by the project are skipped.

## Frame preparation introduced in 0.8

The frame count is now selected **per video** instead of being shared across the whole project. With the default of 240 images per video, ten videos can contribute up to 2,400 images. The value is configurable before preparation.

An unsuitable extracted image can now be removed directly from training. This deletes only the derived frame and derived dataset copies; the source video is never changed. If the image belonged to a frozen benchmark set, Reco Trainer invalidates that reference and asks the user to freeze the reviewed answers again.

## Local model benchmark

Reco Trainer can test all compatible imported models against the same frozen ground-truth image set and create an automatic ranking. It reports mAP@0.50, precision, recall, F1, mean IoU, false positives, false negatives and inference time per image.

The quality score uses 70% mAP@0.50 and 30% F1; latency is used only as a tie-breaker. Models run sequentially so that they do not compete for the same GPU, Apple Neural Engine or system memory.

For a meaningful comparison, use a separate, representative test set that was not used to train any of the compared models.

## Downloads and installation

Download the package for your system from [GitHub Releases](https://github.com/wendibus/reco-trainer/releases).

### macOS

1. Download `Reco-Trainer-Mac-0.12.8.dmg`.
2. Open it and copy **Reco Trainer** to **Applications**.
3. Start the app. This alpha is locally/ad-hoc signed and not Apple-notarized, so macOS may require Control-clicking the app, selecting **Open**, and confirming once.

Requirements: macOS 14 or newer and Apple silicon. Node.js 22 or newer and the Xcode Command Line Tools are needed for the local worker setup.

### Windows

1. Download and extract `Reco.Trainer.Windows.0.12.8.zip`.
2. Open the extracted folder.
3. Run `Start Reco Trainer Windows.bat`.
4. Keep the terminal windows open; the interface runs at `http://localhost:8765/`.

Requirements: Node.js 22, Python 3.11 or 3.12, and FFmpeg.

### Linux

1. Download and extract `Reco.Trainer.Linux.0.12.8.zip`.
2. Run `chmod +x "Start Reco Trainer Linux.sh"` once.
3. Run `./Start\ Reco\ Trainer\ Linux.sh`.
4. If necessary, open `http://localhost:8765/` manually.

Requirements: Node.js 22, Python 3.11 or 3.12, FFmpeg, and Zenity or KDialog for native folder selection.

### Docker

1. Download and extract `Reco.Trainer.Docker.0.12.8.zip`.
2. Set `RECO_VIDEO_FOLDER` to the absolute path of the local sports-video folder.
3. Run `docker compose up --build` from the extracted folder.
4. Open `http://localhost:8765/`.

## Basic workflow

1. Choose a language and complete the walkthrough.
2. Select a sport and a local video folder.
3. Choose the number of images per video and prepare frames locally below `.reco-training/`.
4. Remove unsuitable images, then generate automatic suggestions, correct them and remove false labels.
5. Train a local model or import compatible `.recomodel` packages.
6. Choose **Review new videos**, select a separate folder, and review the local candidate queue.
7. Retrain after a meaningful batch of confirmed images.
8. Open **Test models**, freeze the reviewed answers and run the benchmark.
9. Review the ranking and detailed error counts.
10. Export a `.recomodel` package only when you deliberately want to exchange a model.

## Privacy and security

- The local worker binds to `127.0.0.1` only.
- There is no media upload endpoint.
- Videos, frames, annotations, training runs, models and benchmark reports stay below `.reco-training/` in the selected folder.
- Benchmark result files contain numerical predictions and metrics, not source images.
- The initial ML setup can download dependencies and pretrained weights, but it does not upload sports footage.
- Import model packages only from trusted publishers. Checksums verify integrity, not the trustworthiness of model code or weights.

## Please test it

Feedback is especially useful for installation, frame extraction, box correction, false-positive removal, local training, model import/export and the new benchmark ranking. Include the operating system, hardware, sport, model size, exact failing action and full error message.

Please **do not** attach private match footage, extracted frames, datasets, `.reco-training` folders or logs containing personal paths. Prefer synthetic or redacted examples.

## Repository layout

- `cross-platform/` — browser interface, local worker, Windows/Linux launchers and Docker configuration.
- `mac/` — native Swift macOS application and its local Python ML worker.
- `FORUM-ANNOUNCEMENT.md` — copy-ready English forum announcement.
- `RELEASE-NOTES-0.12.8.md` — changes, installation details and SHA-256 checksums.

## Current limitations

- Windows and Linux still require broader real-system testing.
- Windows, Linux and Docker use CPU training unless a compatible local accelerator setup is added.
- The Mac build is not Apple-notarized.
- Automatic labels remain suggestions and must be reviewed.
- Benchmark results are meaningful only with accurate, representative and previously unseen test data.
- The current benchmark compares object detection; tracking stability across a complete video is not yet a ranking metric.

No software license has been granted in this repository yet. Third-party components remain subject to their own licenses.
