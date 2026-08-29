# Reco Trainer 0.6

> **Work in progress / alpha software.** Reco Trainer is ready for practical testing, but it is not a finished production release. Keep backups and validate every label and exported model before using it in a real workflow.

Reco Trainer explores a privacy-first way to improve sports-camera models: the sensitive footage stays on the user's own computer. Videos are converted into frames locally, automatic suggestions can be corrected locally, and RF-DETR can be fine-tuned from those corrections. Only an explicitly created `.recomodel` exchange package is intended to be shared. It contains model weights, checksums, and aggregate metadata—not videos, frames, local paths, or video file names.

The current alpha supports basketball, football (soccer), handball, hockey, rugby, lacrosse, and American football. The interface and walkthrough are available in German, English, Spanish, and French.

## Please test it

Real-world testing is the most useful next step. In particular, feedback is needed on installation, folder selection, frame extraction, ball labeling, false-positive removal, zoom and navigation, local training, Core ML export, and `.recomodel` import/export.

When reporting a problem, please include:

- operating system and version;
- CPU/GPU or Apple chip and available memory;
- sport and selected model size;
- the exact action that failed and the complete error message;
- whether the problem can be reproduced.

Please **do not** attach private match footage, extracted frames, datasets, `.reco-training` folders, or logs that expose personal file paths. A synthetic or redacted example is preferable.

## Downloads and installation

Download the package for your system from the GitHub **Releases** page.

### Reco Trainer Mac 0.6

Native Apple-silicon application for private, local sports-video annotation, RF-DETR training, Core ML export, and validated `.recomodel` exchange.

Requirements: macOS 14 or newer, Apple silicon, Node.js 22 or newer, and the Xcode Command Line Tools. FFmpeg is optional because the package can build and use its included AVFoundation frame extractor.

1. Download `Reco Trainer Mac 0.6.dmg`.
2. Open the disk image and copy **Reco Trainer** to **Applications**.
3. Start the application. Because this alpha is locally/ad-hoc signed and not Apple-notarized, macOS may require Control-clicking the app, choosing **Open**, and confirming once.
4. Keep the local worker window open while using the application.

### Reco Trainer Windows 0.6

Local Windows package with native folder selection, frame review, CPU-based RF-DETR training, model export, and validated model exchange.

Requirements: Node.js 22, Python 3.11 or 3.12, and FFmpeg.

1. Download and extract `Reco Trainer Windows 0.6.zip`.
2. Open the extracted `Reco Trainer Windows` folder.
3. Run `Start Reco Trainer Windows.bat`.
4. Leave the terminal windows open. The interface opens at `http://localhost:8765/`.

### Reco Trainer Linux 0.6

Local Linux package with native folder selection where supported, frame review, CPU-based RF-DETR training, model export, and validated model exchange.

Requirements: Node.js 22, Python 3.11 or 3.12, FFmpeg, and Zenity or KDialog for native folder selection.

1. Download and extract `Reco Trainer Linux 0.6.zip`.
2. In a terminal, change to the extracted `Reco Trainer Linux` folder.
3. Run `chmod +x "Start Reco Trainer Linux.sh"` once.
4. Run `./Start\ Reco\ Trainer\ Linux.sh` and leave the terminal open.
5. If no browser opens automatically, visit `http://localhost:8765/`.

### Reco Trainer Docker 0.6

Portable CPU-based container package for macOS, Windows, and Linux with an explicit local volume mount and no media upload endpoint.

Requirements: Docker Desktop or Docker Engine with Docker Compose.

1. Download and extract `Reco Trainer Docker 0.6.zip`.
2. Set `RECO_VIDEO_FOLDER` to the absolute path of the local sports-video folder.
3. From the extracted `Reco Trainer Docker` folder, run `docker compose up --build`.
4. Open `http://localhost:8765/`.

Example on macOS/Linux:

```bash
export RECO_VIDEO_FOLDER="/absolute/path/to/your/videos"
docker compose up --build
```

Example in PowerShell:

```powershell
$env:RECO_VIDEO_FOLDER = "C:\absolute\path\to\your\videos"
docker compose up --build
```

## Basic workflow

1. Choose a language and complete the walkthrough.
2. Select the sport and the local video folder.
3. Prepare videos locally. Frames and project data are created below `.reco-training/` in that folder.
4. Set up ML once. This downloads the Python packages and pretrained weights; sports footage is not uploaded.
5. Generate automatic suggestions, correct them, and remove false labels.
6. Train RF-DETR Nano or Small locally and review the validation result.
7. On Mac, export a Core ML model when required.
8. Export a `.recomodel` package if you deliberately want to exchange the improved model.

## Privacy and security

- The local worker binds to the loopback interface only (`127.0.0.1`).
- There is no media upload endpoint.
- Videos, frames, annotations, training runs, and local models stay below `.reco-training/` in the selected video folder.
- The first ML setup requires internet access for dependencies and pretrained weights, but not for sports footage.
- Import `.recomodel` files only from trusted publishers. Model packages are validated for structure, sport, model size, classes, and checksum, but PyTorch model files should still be treated as executable content from a trust perspective.

## Repository layout

- `cross-platform/` — browser interface, local Python worker, Windows/Linux launchers, and Docker configuration.
- `mac/` — native Swift macOS application and its Python ML worker.
- `FORUM-ANNOUNCEMENT.md` — copy-ready English announcement for a forum post.
- `RELEASE-NOTES-0.6.md` — package descriptions, highlights, and SHA-256 checksums.

## Current limitations

- This is an alpha and the workflow still needs broader testing on real Windows and Linux systems.
- Windows, Linux, and Docker currently use CPU training unless a compatible local configuration is added.
- The Mac build is not Apple-notarized.
- Automatic labels are proposals and must be reviewed; people outside the field of play may still be detected.
- Model quality depends heavily on varied, accurate labels and a representative validation split.

No software license has been granted in this repository yet. Third-party components remain subject to their own licenses.
