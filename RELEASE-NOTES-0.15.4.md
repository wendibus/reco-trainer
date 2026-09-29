# Reco Trainer 0.15.4 — Ball-Tracking Simulation: Playback and Progress (Windows/Linux/Docker)

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Fixed:** in the Windows/Linux/Docker web interface, the ball-tracking simulation clip never played past its first second. The interface re-reads the worker status every 900 ms, and each read produced a "new" simulation object, which restarted the player at frame 1. The player now keys off the simulation's identity, so the clip plays through its full length before looping.
- **Improved:** the simulation now shows a visible progress bar with a status line and percentage. Frame extraction and ball detection report real progress (the bar previously sat at 1 % and then 40 % for the entire run, which looked like there was no progress bar at all).

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.15.4.dmg`
- **Windows:** `Reco.Trainer.Windows.0.15.4.zip`
- **Linux:** `Reco.Trainer.Linux.0.15.4.zip`
- **Docker:** `Reco.Trainer.Docker.0.15.4.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
