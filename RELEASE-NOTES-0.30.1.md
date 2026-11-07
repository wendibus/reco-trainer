# Reco Trainer 0.30.1 — AMD detection confirmed, diagnosis report fixes

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Confirmed

- **Detection on AMD graphics (Windows):** a tester ran the GPU diagnosis on an AMD Radeon RX 7800 XT (Windows 11, 16 GB). RF-DETR Nano through DirectML takes 6.7 ms per image instead of 110.7 ms on the CPU (about 16 times faster), with a largest box difference of 0.00018. Vulkan sees the card as well. Training on AMD still runs on the CPU.

## Fixed

- The diagnosis report no longer warns when dxdiag prints "Driver Problem Code: Unknown"; the driver just reports no code.
- The report on Windows and Linux showed "recoTrainer: unknown". The packages now carry a `version.json`.
- YOLO training left empty `runs/detect/val*` folders in the working folder (also inside the 0.30.0 Windows, Linux and Docker packages). The evaluation now stays inside the training run, and packaging leaves `runs/` out.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.30.1 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.30.1 update. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.30.1.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.30.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.30.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.30.1.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
