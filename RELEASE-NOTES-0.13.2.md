# Reco Trainer 0.13.2 — Ball-Tracking Simulation Startup Fix

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Fixed:** the ball-tracking simulation (introduced in 0.13.0) could fail immediately with "The data couldn't be read because it isn't in the correct format." The Mac app captures a local ML process's combined stdout and stderr as one stream; PyTorch/RF-DETR sometimes print their own warnings to that stream while loading a model, which ended up mixed into the same output as the simulation's result and broke reading it. Only the output's last line is used for that now, making it robust to any such warnings.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.13.2.dmg`
- **Windows:** `Reco.Trainer.Windows.0.13.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.13.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.13.2.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
