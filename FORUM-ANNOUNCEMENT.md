# Reco Trainer 0.6 Alpha — Help test private, local sports-model training

I am looking for testers for **Reco Trainer 0.6**, a work-in-progress tool for improving sports-camera detection models without uploading sensitive match footage to an unknown server.

**GitHub links:**

- [Project repository and full installation guide](https://github.com/wendibus/reco-trainer)
- [Download Reco Trainer 0.6 — Work in Progress Alpha](https://github.com/wendibus/reco-trainer/releases/tag/v0.6.0)
- [Source code](https://github.com/wendibus/reco-trainer/tree/main)

The idea is simple: videos stay on your own computer. Reco Trainer extracts frames locally, proposes labels, lets you correct mistakes, and fine-tunes an RF-DETR model from those corrections. If you decide to share the result, you export a `.recomodel` package containing model weights, checksums, and aggregate metadata. The exchange package does **not** contain videos, frames, local paths, or video file names.

This is explicitly an **alpha / work in progress**, not a finished production tool. Please keep backups and review all labels and results. What I need now is practical feedback from different machines, sports, and recording conditions.

The current version supports:

- basketball;
- football (soccer);
- handball;
- hockey;
- rugby;
- lacrosse;
- American football.

The interface and six-step walkthrough are available in German, English, Spanish, and French. Label review includes zooming, panning, frame navigation, editing and deleting boxes, undo/redo, automatic suggestions, local RF-DETR Nano/Small training, validated model exchange, and Core ML export on Mac.

## Downloads

Choose one file from the [Reco Trainer 0.6 GitHub release](https://github.com/wendibus/reco-trainer/releases/tag/v0.6.0):

- **Reco Trainer Mac 0.6.dmg** — native Apple-silicon application for local annotation, Apple-accelerated RF-DETR training, Core ML export, and `.recomodel` exchange. Requires macOS 14+, Node.js 22+, and Xcode Command Line Tools. The alpha is ad-hoc signed and not Apple-notarized, so the first launch may require Control-click → **Open**.
- **Reco Trainer Windows 0.6.zip** — local Windows version with native folder selection and CPU training. Requires Node.js 22, Python 3.11/3.12, and FFmpeg. Extract it and run `Start Reco Trainer Windows.bat`.
- **Reco Trainer Linux 0.6.zip** — local Linux version with CPU training. Requires Node.js 22, Python 3.11/3.12, FFmpeg, and Zenity or KDialog. Extract it, make `Start Reco Trainer Linux.sh` executable, and run it.
- **Reco Trainer Docker 0.6.zip** — portable CPU-based container version for macOS, Windows, and Linux. Mount the sports-video folder through `RECO_VIDEO_FOLDER`, run `docker compose up --build`, and open `http://localhost:8765/`.

The first ML setup downloads Python dependencies and pretrained model weights. Your sports footage is not uploaded. The worker has no media-upload endpoint and is bound locally; project files remain in `.reco-training/` inside the chosen video folder.

## What to test

Please try the complete flow: installation, language selection, sport selection, local folder selection, frame extraction, automatic labeling, correcting and deleting false labels, zoom/navigation, training, export, and `.recomodel` import/export.

For a useful bug report, include your operating system, hardware/chip, memory, sport, selected model size, exact reproduction steps, and the full error message. Please never post private footage, extracted frames, datasets, `.reco-training` folders, or logs containing personal paths. Synthetic or redacted examples are ideal.

I would particularly value answers to these questions:

- Was installation understandable without additional help?
- Did folder selection and frame preparation work on your system?
- Could you find, correct, and delete automatic labels easily?
- Was the training progress understandable, and did training finish?
- Did the exported/imported model package behave as expected?
- Which parts of the walkthrough or wording were unclear?

Thank you for helping test a privacy-first approach to collaborative sports-model improvement.
