# Reco Trainer

> **Ready for real-world use, but not bug-free.** Reco Trainer is built for production workflows, yet it is under active development and still has bugs. Keep backups, review automatic labels, and validate benchmark results and exported models before relying on them in a critical workflow. If something breaks, please [report it](#feedback-and-bug-reports).

Reco Trainer is a privacy-first tool for training and comparing sports-camera detection models. Your videos stay on your own computer: frames are extracted locally, automatic suggestions are corrected locally, and RF-DETR models are trained and tested locally. Only an explicitly exported `.recomodel` package is meant to be shared, and it contains model weights, checksums and aggregate metadata — never videos, frames, local paths or video file names.

Supported sports: basketball, football (soccer), futsal, handball, hockey, rugby, lacrosse and American football. The interface, walkthrough video and guided tour are available in German, English, Spanish and French.

**New here?** Read the [User Guide](docs/USER-GUIDE.md). It explains in plain language how labeling and training work, why they matter, and how to use the advanced features such as model comparison — no machine-learning background required.

**Prefer to watch?** The [14-minute extended training video](https://wendibus.github.io/reco-trainer/) combines a practical walkthrough with the reasoning behind careful annotation, representative test data, local training, model comparison and ball-tracking simulation. It uses synthetic demonstration footage, offers American English, German and Spanish narration, and includes optional English, German, Spanish and French subtitles.

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

Requirements: macOS 14 or newer, Apple silicon, and **Python 3** (3.12 recommended, for example `brew install python@3.12`). "Set up ML" finds a suitable Python 3.10+ on its own; if none is installed, it tells you what to install. Node.js and Xcode are *not* needed.

1. Download `Reco-Trainer-Mac-<version>.dmg` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Open it and drag **Reco Trainer** into **Applications**.
3. Start the app. It is signed with a Developer ID certificate and notarized by Apple, so macOS opens it without a security warning.
4. Choose your language, then watch the short walkthrough video or click through the guided tour.
5. Pick a sport and a video folder, then run **Set up ML** once.

The Mac app is a thin launcher around the same interface that Windows, Linux and Docker use. It updates itself: when a new release is published it offers to install the new interface and workers (signature-checked), without a new app download.

### Windows

Requirements: Python 3.10 or newer (3.12 recommended) and FFmpeg. Node.js is *not* needed.

1. Download and extract `Reco.Trainer.Windows.<version>.zip` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Open the extracted folder and run `Start Reco Trainer Windows.bat`.
3. Keep the terminal window open. The interface runs at <http://localhost:8766/>.
4. Pick a sport and a video folder, then run **Set up ML** once. If an NVIDIA GPU is detected, PyTorch is installed with CUDA support automatically.

### Linux

Requirements: Python 3.10 or newer (3.12 recommended), FFmpeg, and Zenity or KDialog for the native folder dialog.

1. Download and extract `Reco.Trainer.Linux.<version>.zip` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Run `chmod +x "Start Reco Trainer Linux.sh"` once, then `./Start\ Reco\ Trainer\ Linux.sh`.
3. If your browser does not open by itself, go to <http://localhost:8766/>.
4. Pick a sport and a video folder, then run **Set up ML** once.

### Docker

Requirements: Docker with Compose. Training runs on the CPU inside the container.

1. Download and extract `Reco.Trainer.Docker.<version>.zip` from the [latest release](https://github.com/wendibus/reco-trainer/releases/latest).
2. Set `RECO_VIDEO_FOLDER` to the absolute path of the folder with your sports videos.
3. Run `docker compose up --build` in the extracted folder.
4. Open <http://localhost:8766/>. The port is published on `127.0.0.1` only.

The extracted folders also contain `START-HERE.md` with a step-by-step tour of the interface.

## What you can do with it

1. **Prepare** — choose a sport and a video folder; frames are extracted locally (60 per video by default; you can lower or raise the number) and unsuitable ones can be removed.
2. **Label** — draw and correct boxes, or let a model suggest them (**auto-label**) and review each suggestion. Mark the **field boundaries** (at least four points along the edge, joined by straight lines; points on the image edge when the field leaves the frame) so auto-label only considers people standing on the field, and tighten boxes with **Refine boxes with OpenCV**. Ten undo/redo steps per image.
3. **Train** — fine-tune RF-DETR Nano or Small locally. Every run is kept as its own model version; a worse run never silently replaces the active one.
4. **Improve** — **Review new videos** builds a review queue from footage the model has not seen, prioritizing uncertain frames (model disagreement, positional outliers). Only frames you reviewed enter training. **Check consistency** looks for missing, duplicate or jumping boxes (see [Consistency check](#consistency-check)).
5. **Test** — **Independent model test** builds a held-out test set from videos never used for training; **Test models** ranks all compatible models on it (mAP@0.50, precision, recall, F1, mean IoU, false positives/negatives, inference time). You can also bake a **combined model** that uses the best model per category.
6. **Inspect tracking** — the **ball-tracking simulation** plays a short clip frame by frame and shows detected, interpolated, held and lost ball positions.
7. **Exchange** — export a CPU model (ONNX) or, on a Mac, an Apple model (Core ML); export a `.recomodel` package, or import one from a source you trust. Imports are checked against their checksum and scanned for executable code before they are ever used. ONNX detector packages (automatic labeling and benchmark only) are checked against their checksum; they contain no executable checkpoint. A YOLO model in ONNX format (NMS built in) can also be imported directly: **Import model** on the Mac accepts the `.onnx` file, the web UI has **Import ONNX model**; Reco Trainer asks for the classes and builds the package itself (see [MODEL-EXCHANGE.md](cross-platform/MODEL-EXCHANGE.md)).

8. **Export a dataset** — all reviewed images with their labels as **YOLO** and/or **COCO**, split by video, optionally **anonymized** (see [Dataset export](#dataset-export)).

Every longer operation runs as a job with a **progress window**: step, progress bar, elapsed and remaining time, and **Cancel**. Cancelling removes half-written files and keeps the project unchanged; a cancelled training keeps its last checkpoint and the next run continues from there. Further operations wait in a queue; the window can be minimized to a status chip and survives a page reload.

The [User Guide](docs/USER-GUIDE.md) walks through all of this with explanations of *why* each step matters.

## Import labels from other tools

Reco Trainer keeps its labels in the project (`.reco-training/project.json`). **YOLO-txt** and **COCO-JSON** datasets can be imported with **Import labels** or **Import labels: As suggestions**. Choose the folder with the images and labels; the project is reloaded afterwards and the status line reports images, boxes and skipped classes. No ML environment is needed.

For your own class mapping (`--class-map`), `--overwrite` or a fixed `--format`, use the worker command:

```bash
python3 ml_worker.py import-labels --project "/path/to/videos/.reco-training" --source "/path/to/labelled-folder" \
  [--class-map "0=player,1=ball,2=referee"] [--format yolo|coco] [--candidates] [--overwrite]
```

- **YOLO:** images next to their `.txt` files, or an Ultralytics layout (`images/...`, `labels/...`). Class names come from `data.yaml` or `classes.txt`, or from `--class-map`. Polygon lines become their bounding box.
- **COCO:** `*.json` with `images`, `annotations` and `categories`; boxes are `[x, y, width, height]` in pixels.
- Classes are mapped to the sport's Reco classes (for example `person` -> `player`, `sports ball` -> `ball`). Other classes are skipped and listed in the output.
- Images are copied to `frames/imported/` and count as reviewed; `--candidates` imports them as suggestions to review first. Importing the same image again is skipped. A frame that already exists with the same file name only receives the labels (`--overwrite` replaces labels it already has).
- The train/validation split is made per video, and one import counts as one source. Import from several sources, or together with several videos, to get a separate validation set.
- When using the command line, close the project in the Mac app before importing and open it again afterwards. If you prepare the videos again in the Mac app, run the import again; the files stay in `frames/imported/`.

## Dataset export

**Export dataset** (next to "Create package") writes every reviewed image with its labels:

- **Formats:** YOLO (Ultralytics: `data.yaml`, `images/<train|val|test>`, `labels/...`) and/or COCO (Roboflow / RF-DETR: `<train|valid|test>/_annotations.coco.json`). The split is made per video, as for training, so frames of one video never end up in both training and test.
- **Neutral file names** (`frame-<id>.jpg`): no video names, source paths or project data. `metadata.json` records the Reco Trainer version, classes, split sizes, the anonymization method and every model used with its license. Optionally a ZIP file is created.
- **Anonymized** (needs "Set up ML"): heads and faces are strongly **blurred** (or **pixelated**) inside soft-edged ellipses – never covered with boxes, so a head keeps its shape and the data stays useful for training. Covered are the head of every labelled person, people without a label such as spectators and the bench (found with the RF-DETR COCO base model), and faces found by YuNet. **Jersey numbers and the scoreboard** can be hidden too (off by default). The ball and all labels stay unchanged.
- After blurring, the face detector runs again. A frame where it still finds an uncovered face is **not exported**; it is listed after the export (and in `<export>.review.json` next to, not inside, the export) so you can check it, label the person and export again.
- **Anonymized training images:** in the training options, a share of the training images (10, 25 or 50 %) can be trained with blurred heads, so a model also handles anonymized images. The original frames are never changed.

Automatic anonymization is very good but not perfect - check samples before you share a dataset.

## Consistency check

**Check consistency** (in the review area) compares your labels with a second detector and with the timeline of each video:

- **Second detector:** boxes that do not match (from an adjustable IoU on, default 0.5) mark a frame for review.
- **Timeline:** players who suddenly vanish between two close frames, and positional outliers, are flagged too.
- **Local VLM (optional):** a vision-language model (Qwen3.5) looks only at the flagged frames plus an optional sample of unremarkable ones. Confirmed problems become **hard examples**, which are used twice in training; unfounded warnings are cleared. With an anonymized export, the VLM can also check a sample of heads for faces that are still recognizable.

The VLM runs **inside Reco Trainer on every platform** - no Ollama or other separate installation. On first use the app downloads a pinned `llama-server` build of llama.cpp for the system (macOS Apple silicon with Metal, Windows and Linux with CUDA or CPU, Linux arm64) and the model, verifies their checksums, and keeps them once per computer (macOS: `~/Library/Application Support/Reco Trainer/models`, Windows: `%LOCALAPPDATA%\Reco Trainer\models`, Linux: `~/.local/share/reco-trainer/models`; in Docker inside the project folder). The server listens on `127.0.0.1` only, runs just for the job and is stopped afterwards, which frees the GPU memory.

- Qwen3.5-4B needs about 3.4 GB of downloads, Qwen3.5-9B (strong tier only) about 6.6 GB.
- Linux: the CUDA and arm64 builds need glibc 2.38 or newer (Ubuntu 24.04, Debian 13); otherwise the CPU build is used (glibc 2.34, Ubuntu 22.04, Debian 12). The Docker image is based on Debian 13.
- Intel Macs are not supported for the VLM; the check without the VLM still works everywhere.
- Docker and other CPU-only systems count as the weak tier, so the VLM is not offered by default. Choose **medium** under **Advanced options** to use it anyway; on the CPU expect roughly 20 seconds per image.

## Models and licenses

Reco Trainer only downloads models under **Apache-2.0** or **MIT**, each with a checksum ([`cross-platform/model_registry.json`](cross-platform/model_registry.json)); the licenses are shown in the export dialog and written into every export's `metadata.json`.

| Model | Used for | License |
|---|---|---|
| RF-DETR Nano / Small / Medium / Large (COCO) | training base, auto-label, finding people for anonymization | Apache-2.0 |
| YuNet face detector (OpenCV Zoo) | anonymization | MIT |
| PP-OCRv3 text detector (OpenCV Zoo) | hiding text (optional) | Apache-2.0 |
| SAM 2.1 Hiera Small / Base+ | refining boxes (optional) | Apache-2.0 |
| Florence-2 Base / Large | second opinion during auto-label | MIT |
| Qwen3.5-4B / 9B (GGUF by unsloth) | consistency check, anonymization check (optional) | Apache-2.0 |
| llama.cpp `llama-server` (b11985) | runs the VLM locally | MIT |

RF-DETR XL/2XL (PML 1.0), SAM 3 and Ultralytics models are not used. Hugging Face models are pinned to one exact revision.

## Hardware

Reco Trainer picks a **hardware tier** and uses the models that fit (override under **Advanced options**):

- **Strong** - NVIDIA GPU with 16 GB VRAM or Apple silicon with 32 GB or more: RF-DETR Large as suggestion model, Florence-2 Large as second opinion, SAM 2 Base+ when chosen, VLM on by default (Qwen3.5-4B, 9B selectable).
- **Medium** - NVIDIA with 8 GB or Apple with 16 GB: RF-DETR Medium, Florence-2 Base, SAM 2 Small, VLM optional (Qwen3.5-4B).
- **Weak** - CPU only: RF-DETR Small and Florence-2 Base, no SAM 2 and no VLM. Training and the anonymized export are much slower; a plain export needs no ML environment at all.

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
- `mac/` — the macOS launcher (a native window around the web interface, with signed self-updates) and [`CHANGELOG.md`](mac/CHANGELOG.md).
- `docs/` — the [User Guide](docs/USER-GUIDE.md), the hosted [training video](docs/training-video/) page, model documentation and design proposals.
- `RELEASE-NOTES-<version>.md` — changes and download details for each release.

## Current limitations

- Windows, Linux and Docker still need broader real-system testing.
- Training uses Apple silicon or an NVIDIA GPU when available; AMD GPUs are not accelerated, and Docker runs on the CPU.
- The Mac app is signed and Apple-notarized from 0.19.1 on (0.17.2 as well); older releases are signed only and need **Open** from the context menu on first launch.
- Automatic labels are suggestions and must be reviewed.
- Benchmark results are only meaningful with accurate, representative and previously unseen test data.
- The benchmark compares object detection; tracking stability across a complete video is not yet a ranking metric (the ball-tracking simulation is a visual check only).

No software license has been granted in this repository yet. Third-party components remain subject to their own licenses.
