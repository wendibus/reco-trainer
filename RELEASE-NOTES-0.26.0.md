# Reco Trainer 0.26.0 — Import datasets directly and set up ML only once

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Import a dataset right after choosing a working folder:** a new **Import dataset** button sits directly below **Select working folder**. Pick a folder with images and COCO or YOLO labels; it is added as soon as a working folder is selected, even if nothing has been prepared there yet. In that case the project is started with the dataset. The images are copied into the project, the original folder stays unchanged, and the labels count as reviewed.
- **Set up ML only once per computer:** the ML environment now lives in Reco Trainer's own folder and is used by every working folder.
  - macOS: `~/Library/Application Support/Reco Trainer/ml-env`
  - Windows: `%LOCALAPPDATA%\Reco Trainer\ml-env`
  - Linux: `~/.local/share/reco-trainer/ml-env`
- **Existing setups keep working:** a project that already has its own ML environment from an earlier version keeps using it, and Reco Trainer remembers that environment for new working folders. You do not have to click **Set up ML** again. Should the remembered environment become unavailable, for example because its drive is not connected, **Set up ML** installs the shared one. A remembered environment that belongs to another project is never deleted. In Docker the environment stays in the project folder as before.

## Fixes

- The hint "This working folder is still empty" disappears once the project contains images.

## Good to know

- Imported images do not carry scene information. Next to frames from real videos they are used for training only. In a project made of imported images alone, they are split at random, so validation and test scores are not independent of the training data. For a fair comparison, use the **Independent Model Test** with a game that is not part of the dataset.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.26.0 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.26.0 update. This release updates the interface and workers; the launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.26.0.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.26.0.zip`
- **Linux:** `Reco.Trainer.Linux.0.26.0.zip`
- **Docker:** `Reco.Trainer.Docker.0.26.0.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
