# Reco Trainer 0.13.0 — Ball-Tracking Simulation, Automatic GPU Setup

The biggest update yet. Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## What's new

### Ball-tracking simulation

Pick a short clip and watch the currently active model's ball detection played back frame by frame — a diagnostic view of how the model would actually perform in motion, not just on isolated still frames. Each frame shows one of four states:

- **Detected** (green) — a real detection landed on this frame.
- **Interpolated** (orange, dashed) — no detection here, but the tool found a real detection both before and after within the lookahead window, and blends between them.
- **Held** (yellow, dashed) — a real detection exists before this frame within the window, but nothing after it — the last known position is held.
- **Lost** — nothing within reach in either direction; no marker is shown.

The "how far to look into the future" control is a live slider: dragging it recomputes the whole track instantly from detections already fetched, with no new inference call per adjustment. The methodology mirrors an existing sports-camera tracking system's approach (nearest-detection selection with a bounded hold), extended with real forward interpolation since — unlike a live camera — the whole clip is already available up front.

The clip is extracted to a throwaway location and processed once; it is never added to the project's training data, never saved, never trained on.

### Independent model test & combined models

(Previously shipped in 0.12.15, included here as part of this larger release.)

- **Unabhängiger Modelltest / Independent model test**: review a video folder that was never used for training and turn it — with app assistance across every category — into a genuinely independent test set. The model benchmark now requires this kind of held-out data for its reference instead of any reviewed training image, so a model can't score well on the comparison simply because it already saw that footage during training.
- **Combine models**: bake a new combined model out of several installed models, each contributing only the categories it scored best on. Not real weight merging (not technically possible with RF-DETR's architecture) — an ensemble at inference time, installed into the model library like any other model.
- The cross-platform web interface now also shows the per-category comparison table, matching what the Mac app has had since 0.12.9.

### Windows: automatic GPU setup

"Set up ML" now detects an NVIDIA GPU on Windows and installs PyTorch with CUDA support automatically. Previously, a plain `pip install` there silently produced a CPU-only build — PyPI only hosts CUDA-enabled PyTorch wheels for Linux, not Windows — so every Windows installation defaulted to CPU-only training and inference regardless of the graphics card actually present, with no way for the app itself to notice or say so.

- A new switch ("Automatically set up GPU acceleration") lets this be turned off if the CUDA install itself causes trouble (blocked network, unsupported driver) — falling back to the plain, always-working CPU install.
- The hardware indicator now correctly reports CUDA availability and the GPU name on Windows/Linux, instead of unconditionally showing "CPU" regardless of what's actually being used.
- Preparing many videos at once (e.g. a GoPro recording split into several chapter files) now shows progress while reading each file's metadata, instead of looking like it's hanging before extraction even starts.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.13.0.dmg`
- **Windows:** `Reco.Trainer.Windows.0.13.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.13.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.13.0.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. The ball-tracking simulation processes its clip entirely locally and discards the extracted frames afterward — nothing from it is added to the project or saved permanently.
