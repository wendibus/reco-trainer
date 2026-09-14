# Reco Trainer 0.12.14 — Windows Training Crash Fix, Update Check for the Web UI

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. This remains work-in-progress alpha software.

## Changes

- **Windows fix:** "Train locally" crashed partway through the first epoch with a raw Unicode encoding error, because a subprocess whose output is piped (not a real terminal) falls back to the OS locale's preferred encoding — on Windows typically a legacy codepage, not UTF-8 — and RF-DETR's rich-rendered metrics tables use characters that don't fit in it. Fixed by forcing UTF-8 for the local ML process's output on both ends. This affects the Windows/Linux/Docker builds' local ML worker.
- The Windows/Linux/Docker web interface now checks at launch whether a newer version has been published on GitHub, showing a dismissible banner with a download link — matching what the Mac app has had since 0.12.9. Previously there was no such notice there at all, so relaunching an outdated copy gave no hint that a newer one existed.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.12.14.dmg`
- **Windows:** `Reco.Trainer.Windows.0.12.14.zip`
- **Linux:** `Reco.Trainer.Linux.0.12.14.zip`
- **Docker:** `Reco.Trainer.Docker.0.12.14.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints. A `.recomodel` exchange package contains weights, checksums, and aggregate metadata only.
