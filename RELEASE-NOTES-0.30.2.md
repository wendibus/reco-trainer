# Reco Trainer 0.30.2 — Remove a field boundary

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Remove a field boundary:** once a boundary is saved, **Edit field boundaries** shows **Remove field boundary**. It deletes the boundary of the current video, or of every video when **Use for all videos** is ticked. The first click turns the button red ("Really remove?"); only the second click deletes. `project.json` is backed up to `backups/` first. Auto-label then no longer filters those videos by field.

## Fixed

- On Windows, the hardware line showed "? GB" for memory. It now reads the installed memory, which also sets the training batch size correctly.
- Detection on AMD and Intel graphics (DirectML) is no longer labelled "experimental"; it is confirmed on a Radeon RX 7800 XT.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.30.2 through the in-app updater. For a fresh installation, use the [signed and notarized 0.25.1 launcher](https://github.com/wendibus/reco-trainer/releases/download/v0.25.1/Reco-Trainer-Mac-0.25.1.dmg), then accept the 0.30.2 update. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.30.2.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.30.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.30.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.30.2.zip`

`SHA256SUMS.txt` contains checksums for all attached update and platform packages.
