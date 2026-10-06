# Reco Trainer

> **Ready for real-world use, but not bug-free.** Reco Trainer is built for production workflows, yet it is under active development and still has bugs. Keep backups, review automatic labels, and validate benchmark results and exported models before relying on them in a critical workflow. If something breaks, please [report it](#feedback-and-bug-reports).

Reco Trainer is a privacy-first tool for training and comparing sports-camera detection models. Your videos stay on your own computer: frames are extracted locally, automatic suggestions are corrected locally, and RF-DETR models are trained and tested locally. Only an explicitly exported `.recomodel` package is meant to be shared, and it contains model weights, checksums and aggregate metadata — never videos, frames, local paths or video file names.

Supported sports: basketball, football (soccer), futsal, handball, hockey, rugby, lacrosse and American football. The interface, walkthrough video and guided tour are available in German, English, Spanish and French.

**New here?** Read the [User Guide](docs/USER-GUIDE.md). It explains in plain language how labeling and training work, why they matter, and how to use the advanced features such as model comparison — no machine-learning background required.

**Prefer to watch?** The [14-minute extended training video](https://wendibus.github.io/reco-trainer/) combines a practical walkthrough with the reasoning behind careful annotation, representative test data, local training, model comparison and ball-tracking simulation. It uses synthetic demonstration footage, offers American English and Spanish narration, and includes optional English, German, Spanish and French subtitles.

## Download

Always download from the **[latest release](https://github.com/wendibus/reco-trainer/releases/latest)**. Every release page lists one file per platform (the version number is part of the file name) plus `SHA256SUMS.txt` for verifying your download.

| Platform | File on the release page | Details |
|---|---|---|
| macOS (Apple silicon) | `Reco-Trainer-Mac-<version>.dmg` | native app, [install](#macos) |
| Windows | `Reco.Trainer.Windows.<version>.zip` | browser interface + local worker, [install](#windows) |
| Linux | `Reco.Trainer.Linux.<version>.zip` | browser interface + local worker, [install](#linux) |
| Docker | `Reco.Trainer.Docker.<version>.zip` | any system with Docker, [install](#docker) |

Reco Trainer checks GitHub for a newer release when it starts and shows a notice if one exists. That version check is the only network request it makes on its own; your footage is never uploaded.

Older releases stay available on the [releases page](https://github.com/wendibus/reco-trainer/releases). What changed in each version is described in that release's notes and in [`mac/CHANGELOG.md`](mac/CHANGELOG.md).

## Installation

Every platform needs internet access **once**, for the first "Set up ML" step: it installs RF-DETR and PyTorch into a private Python environment below `.reco-training/.runtime/` and downloads the pretrained weights. After that, everything runs offline.

### macOS

Requirements: macOS 14 or newer, Apple silicon, and **Python 3.11 or 3.12** (for example `brew install python@3.12`). The system Python that comes with macOS and the Xcode Command Line Tools is usually older and will not work. Node.js and Xcode are *not* needed for the app.

1. Download `Reco-Trainer-Mac-<version>.dmg` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Open it and drag **Reco Trainer** into **Applications**.
3. Start the app. It is signed with a Developer ID certificate but not yet Apple-notarized, so macOS may show a warning on first launch. Control-click the app, choose **Open**, and confirm once (or use **System Settings → Privacy & Security → Open Anyway**).
4. Choose your language, then watch the short walkthrough video or click through the guided tour.
5. Pick a sport and a video folder, then run **Set up ML** once.

### Windows

Requirements: Node.js 22, Python 3.11 or 3.12, and FFmpeg.

1. Download and extract `Reco.Trainer.Windows.<version>.zip` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Open the extracted folder and run `Start Reco Trainer Windows.bat`.
3. Keep the terminal windows open. The interface runs at <http://localhost:8765/>.
4. Pick a sport and a video folder, then run **Set up ML** once. If an NVIDIA GPU is detected, PyTorch is installed with CUDA support automatically.

### Linux

Requirements: Node.js 22, Python 3.11 or 3.12, FFmpeg, and Zenity or KDialog for the native folder dialog.

1. Download and extract `Reco.Trainer.Linux.<version>.zip` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Run `chmod +x "Start Reco Trainer Linux.sh"` once, then `./Start\ Reco\ Trainer\ Linux.sh`.
3. If your browser does not open by itself, go to <http://localhost:8765/>.
4. Pick a sport and a video folder, then run **Set up ML** once.

### Docker

Requirements: Docker with Compose. Training runs on the CPU inside the container.

1. Download and extract `Reco.Trainer.Docker.<version>.zip` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Set `RECO_VIDEO_FOLDER` to the absolute path of the folder with your sports videos.
3. Run `docker compose up --build` in the extracted folder.
4. Open <http://localhost:8765/>. Both ports are published on `127.0.0.1` only.

The extracted folders also contain `START-HERE.md` with a step-by-step tour of the interface.

## What you can do with it

1. **Prepare** — choose a sport and a video folder; frames are extracted locally (240 per video by default, configurable) and unsuitable ones can be removed.
2. **Label** — draw and correct boxes, or let a model suggest them (**auto-label**) and review each suggestion. Ten undo/redo steps per image.
3. **Train** — fine-tune RF-DETR Nano or Small locally. Every run is kept as its own model version; a worse run never silently replaces the active one.
4. **Improve** — **Review new videos** builds a review queue from footage the model has not seen, prioritizing uncertain frames (model disagreement, positional outliers). Only frames you reviewed enter training.
5. **Test** — **Independent model test** builds a held-out test set from videos never used for training; **Test models** ranks all compatible models on it (mAP@0.50, precision, recall, F1, mean IoU, false positives/negatives, inference time). You can also bake a **combined model** that uses the best model per category.
6. **Inspect tracking** — the **ball-tracking simulation** plays a short clip frame by frame and shows detected, interpolated, held and lost ball positions.
7. **Exchange** — export a `.recomodel` package, or import one from a source you trust. Imports are checked against their checksum and scanned for executable code before they are ever used.

The [User Guide](docs/USER-GUIDE.md) walks through all of this with explanations of *why* each step matters.

## Privacy and security

- The local worker binds to `127.0.0.1` only and has no media upload endpoint.
- Videos, frames, annotations, training runs, models and benchmark reports stay below `.reco-training/` in the selected folder.
- Benchmark result files contain numerical predictions and metrics, not source images.
- The initial ML setup downloads dependencies and pretrained weights, but never uploads sports footage.
- Import model packages only from trusted publishers. Checksums verify integrity, not trustworthiness; imports are additionally loaded through PyTorch's safe-loading mode, which rejects weight files containing executable code.
- The screenshots and demo video in this repository use AI-generated footage only.

## Feedback and bug reports

Feedback is especially useful for installation, frame extraction, box correction, local training, model import/export, the benchmark and the ball-tracking simulation. Please include the operating system, hardware, sport, model size, the exact action that failed and the full error message. Keep in mind that the app is developed on a Mac and only a small part of the translations could be checked by native speakers — reports about wording in Spanish or French and about Windows, Linux or Docker are especially welcome.

Please **do not** attach private match footage, extracted frames, datasets, `.reco-training` folders or logs containing personal paths. Prefer synthetic or redacted examples.

## Repository layout

- `cross-platform/` — browser interface, local worker, Windows/Linux launchers and Docker configuration.
- `mac/` — native Swift macOS application, its local Python ML worker and [`CHANGELOG.md`](mac/CHANGELOG.md).
- `docs/` — the [User Guide](docs/USER-GUIDE.md), the hosted [training video](docs/training-video/) page, model documentation and design proposals.
- `RELEASE-NOTES-<version>.md` — changes and download details for each release.

## Current limitations

- Windows, Linux and Docker still need broader real-system testing.
- Training uses Apple silicon or an NVIDIA GPU when available; AMD GPUs are not accelerated, and Docker runs on the CPU.
- The Mac app is signed but not Apple-notarized.
- Automatic labels are suggestions and must be reviewed.
- Benchmark results are only meaningful with accurate, representative and previously unseen test data.
- The benchmark compares object detection; tracking stability across a complete video is not yet a ranking metric (the ball-tracking simulation is a visual check only).

No software license has been granted in this repository yet. Third-party components remain subject to their own licenses.
