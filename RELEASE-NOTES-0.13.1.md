# Reco Trainer 0.13.1 — Folder Browser Hang Fix

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Fixed:** the folder browser for "Unabhängiger Modelltest" / "Independent model test" (introduced in 0.12.15) could look hung when browsing a folder containing many recording subfolders, each holding many files — for example GoPro recordings split into several chapter files. The new "already used" marker recursively searched each visible subfolder's *entire* contents before the listing could even appear; it now only checks each folder's immediate contents, matching how camera dumps are actually organized (videos directly inside each session folder, not deeply nested).
- Reading each video's duration before extraction now shows progress ("checking video 3/14 …") instead of pausing silently, which could otherwise look like hanging on its own with many files.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.13.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.13.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.13.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.13.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
