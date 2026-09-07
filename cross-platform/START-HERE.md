# Reco Trainer Local 0.12.2

This is no longer a UI-only preview. The browser provides the interface while a local worker, bound exclusively to `127.0.0.1`, processes the selected videos on the same computer.

## Start

1. Double-click `Start Reco Trainer.command` (macOS), `Start Reco Trainer Windows.bat` (Windows), or `Start Reco Trainer Linux.sh` (Linux).
2. Leave the terminal window open.
3. In Reco Trainer, choose a sport.
4. Click "Select video folder" and confirm the folder in the native file dialog.
5. Click "Prepare videos locally". Real JPEG frames are now generated under `.reco-training/frames/` inside the selected folder. A later re-run keeps existing annotations and creates a local backup first.
6. Run "Set up ML" once. This sets up an isolated Python environment under `.reco-training/.runtime/` and downloads RF-DETR.
7. Auto-label, set the minimum confidence, and review the suggestions. Automatic annotations can be accepted or discarded per frame.
8. Boxes can be selected, moved, resized from any of the four corners, and deleted. The last ten changes can be undone and redone; arrow keys switch frames. Pinch or the mouse wheel zooms; "Pan" moves the view.
9. Choose RF-DETR Nano or Small and train locally. After training, Reco Trainer shows the dataset split and, if the installed RF-DETR version reports it, the validation quality.
10. Choose "Review new videos" and open a separate folder. Reco Trainer builds a local review queue. For each image, decide Ball, No ball, or Skip, and correct wrong boxes. Unreviewed candidates are excluded from training.
11. Export the trained model as Core ML (Apple) or ONNX.
12. "Create exchange package" asks for a readable name and creates a `.recomodel` file containing model weights, checksums, and aggregate metadata. It contains no videos, frames, paths, or video file names.
13. "Import model" validates a received `.recomodel` package and only activates it if sport, model size, classes, package structure, and checksum match the project. The local library lives under `.reco-training/models/`.
14. "Test models" opens the local benchmark. First review every automatic suggestion and freeze the correct answers as ground truth. Every compatible imported model is then evaluated on the same images. An expandable glossary explains every metric. The ground truth, predictions, and reports stay under `.reco-training/benchmarks/`.
15. "Model management" keeps every training run as a separate, immutable revision. Give models readable names, activate one, or delete an inactive revision. The active model is protected from deletion; a worse run never replaces it automatically.

To stop, double-click `Stop Reco Preview.command` (macOS) or press `Ctrl+C` in the terminal.

## Requirements

### macOS

- macOS 14 or newer, Apple silicon
- Node.js 22 or newer
- Xcode Command Line Tools
- FFmpeg is optional. If present, it is used; otherwise Reco Trainer builds a small local AVFoundation extractor from the included Swift source.

### Windows

- Node.js 22, **Python 3.11 or 3.12** (a different version may work for browsing/uploading videos, but "Set up ML" needs 3.11/3.12 specifically), and FFmpeg.
- Start with `Start Reco Trainer Windows.bat`. The script warns if it finds a Python other than 3.11/3.12.
- No CUDA (NVIDIA) GPU is required. Without one, training falls back to CPU, which works but is significantly slower. AMD GPUs are not currently accelerated.

### Linux

- Node.js 22, **Python 3.11 or 3.12**, FFmpeg, and Zenity or KDialog for native folder selection.
- Start with `Start Reco Trainer Linux.sh`. The script warns if it finds a Python other than 3.11/3.12.
- Same CPU-fallback behavior as Windows without CUDA.

### Docker

- An explicitly mounted video folder, provided via `RECO_VIDEO_FOLDER`.
- Both ports are published only on `127.0.0.1` of the host machine.
- To import a model package, place the single `.recomodel` file in the mounted video folder or in `.reco-training/inbox/`.

## Privacy

- The worker binds exclusively to `127.0.0.1:8766`.
- It has no upload endpoint.
- Folders are selected through the native OS file dialog.
- Videos, frames, annotations, training runs, and models live exclusively under `.reco-training/` inside the selected video folder.
- Up to 30 local project backups are kept under `.reco-training/backups/` before changes.
- The first-time RF-DETR setup needs internet access for the ML packages and pretrained weights. Sports videos are never uploaded.

See `MODEL-EXCHANGE.md` for the model package and GitHub Releases workflow.
