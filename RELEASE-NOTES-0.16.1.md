# Reco Trainer 0.16.1 — Translations and Type Cleanup

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer. Ready for real-world use; not yet bug-free - please report problems.

## Changes

- **Fixed:** 47 labels and status messages in the Mac app existed only in German and English and showed up in English for Spanish and French users - among them the language and sport headings, "Improve model", "Train from scratch", "Minimum confidence", the annotation tool labels and the progress messages while training, exporting and importing. They are now complete in all four languages.
- **Fixed:** the TypeScript type errors in the web interface (all 9) and a lint error in the ball-tracking player; the player now restarts only when a different simulation arrives.
- **Docs:** removed the outdated "alpha" wording from the old forum announcement.

## Downloads

- **macOS Apple Silicon:** `Reco-Trainer-Mac-0.16.1.dmg`
- **Windows:** `Reco.Trainer.Windows.0.16.1.zip`
- **Linux:** `Reco.Trainer.Linux.0.16.1.zip`
- **Docker:** `Reco.Trainer.Docker.0.16.1.zip`
- **Integrity:** `SHA256SUMS.txt`

## Privacy

Training remains local. Reco Trainer does not upload videos, frames, annotations, or model checkpoints.
