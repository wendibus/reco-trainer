# Reco Trainer 0.31.0 — Training on AMD graphics cards (Windows)

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Training on AMD graphics cards on Windows.** With an AMD Radeon RX 7700 to 7900, RX 9060/9070, Radeon PRO W7700 to W7900 or AI PRO R9700, **Set up AMD training** appears next to **Set up ML**. It installs AMD's ROCm 7.2.1 build of PyTorch with RF-DETR into a separate folder (`%LOCALAPPDATA%\Reco Trainer\rocm-env`, about 6 GB), then checks the card. Afterwards **Train locally** with RF-DETR runs on the graphics card. On an RX 7800 XT one test epoch took 16 s instead of 711 s on the CPU.
- Requirements: Python 3.12 from python.org (AMD ships PyTorch for Windows for 3.12 only; other Python versions stay installed) and Adrenalin 26.2.2 or newer.
- **Train on the AMD card** switches between the card and the CPU. Auto-labeling, benchmarks, export and YOLO keep running in the normal ML environment, with detection on the card through DirectML as before.

This is the first release with AMD training; please report how it works on your card.
