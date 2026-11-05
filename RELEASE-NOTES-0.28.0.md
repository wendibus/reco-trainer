# Reco Trainer 0.28.0 — AMD and Intel graphics on Windows

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Detection on AMD and Intel graphics (Windows):** on a computer without an NVIDIA GPU, RF-DETR detection (auto-labeling, the consistency check's second detector, benchmarks, the ball-tracking simulation, the people detector) runs on the graphics card through ONNX Runtime with DirectML. **Set up ML** installs `onnxruntime-directml` automatically. Each model is converted to ONNX once (a few seconds) and kept with the shared models. Results match PyTorch (same boxes, classes and scores; boxes differ by less than 0.05 px). If the graphics card cannot run the model, detection continues with PyTorch on the CPU. Set `RECO_ONNX_INFERENCE=off` to switch it off.
- **VLM on AMD and Intel graphics (Windows and Linux x64):** without an NVIDIA GPU, the consistency check's VLM uses the Vulkan build of llama.cpp (b11985, MIT). Without a Vulkan driver it runs on the CPU as before; if the graphics card cannot load the model, the server restarts on the CPU by itself. `RECO_VLM_RUNTIME=cpu|vulkan|cuda` chooses the build.
- **Hardware tier for AMD and Intel cards:** Reco Trainer reads the graphics cards from the Windows registry. A card with at least 8 GB of its own memory puts the computer in the **medium** tier (RF-DETR Medium as suggestion model, VLM available); integrated graphics stay **weak**. The hardware line names the card, e.g. "Training on the CPU · detection on AMD Radeon RX 7800 XT (DirectML)", and the job log names provider, card and memory.

## Good to know

- **Training still runs on the CPU with AMD and Intel graphics.** PyTorch publishes ROCm builds for Linux only; AMD's own Windows builds cover only some Radeon cards and an older PyTorch.
- **Not yet tested on real AMD or Intel hardware.** If detection falls back ("failed, continuing with PyTorch on the CPU" in the job log), please report the message.
- DirectML (`DirectML.dll` inside `onnxruntime-directml`) is under Microsoft's DirectML license terms (Windows only); it is installed from PyPI by **Set up ML**, not shipped with Reco Trainer.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.28.0 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.28.0 update. This release updates the interface and workers; the launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.28.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.28.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.28.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.28.0.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
