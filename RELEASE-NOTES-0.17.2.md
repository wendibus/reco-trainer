# Reco Trainer 0.17.2 — Full-Resolution Frames on Windows, Linux and Docker

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Changed (Windows/Linux/Docker):** the 1280 px width limit for extracted frames is removed. Frames keep the full resolution of the source video, as the Mac app already did. Small objects - for example players in the far corner of a full-field 11v11 panorama - keep their pixels.
- Projects that were prepared with an earlier version keep their old, smaller frames. Prepare the videos again to get full-resolution frames.
- Larger frames need more disk space and make labeling and training slower and more memory-hungry.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.17.2.dmg`
- **Windows:** `Reco.Trainer.Windows.0.17.2.zip`
- **Linux:** `Reco.Trainer.Linux.0.17.2.zip`
- **Docker:** `Reco.Trainer.Docker.0.17.2.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
