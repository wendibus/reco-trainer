# Reco Trainer 0.30.3 — Consistency review fixes, AMD training test

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- **Compute suggestions** in the consistency review said "No flagged images" although the review list showed flagged images: the app counted flagged frames of the independent test set, the worker left them out. Both now use the same frames. The second detector is an untrained COCO model, so test frames do not leak into training.
- Messages from the ML worker such as "No flagged images" ended with exit code 1. The app showed "exited with code 1" and, with an AMD/Intel graphics card switched on, started a diagnosis report. Such messages now appear as they are and no longer start a diagnosis.

## New

- **Test AMD Training.bat** (Windows, in `tools`): installs AMD's ROCm 7.2.1 build of PyTorch 2.9.1 (needs Python 3.12) and RF-DETR into a separate folder, `%LOCALAPPDATA%\Reco Trainer\amd-training-probe`, trains RF-DETR Nano for one epoch on 64 generated images on the graphics card and on the CPU, and writes `amd-training-report.txt` to the desktop. Reco Trainer's own ML environment is not touched. This is the groundwork for training on AMD graphics in a later version.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.30.3 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.30.3 update. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.30.3.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.30.3.zip`
- **Linux:** `Reco.Trainer.Linux.0.30.3.zip`
- **Docker:** `Reco.Trainer.Docker.0.30.3.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
