# Reco Trainer 0.11 — Versioned Local Model Library

Reco Trainer 0.11 prevents successive training runs from becoming indistinguishable or silently overwriting the model a user wanted to keep. This is still work-in-progress alpha software and every model should be evaluated on representative, previously unseen footage before real use.

## What changed

- Every completed training run is archived as an immutable local model revision.
- The mutable checkpoint created by an older Reco Trainer version is preserved before the first new training run.
- The model library displays a readable name, stable package ID, model size, creation time and independent test mAP when available, with validation mAP as a fallback.
- Models can be renamed and activated. Inactive revisions can be deleted after confirmation; the active model is protected.
- The best comparable result is marked automatically, preferring the independent test score over validation.
- A newly trained revision is activated only when its comparable quality score improves on the active revision. Worse runs remain available for inspection and benchmarking.
- The benchmark continues to compare all compatible library entries on the same frozen local test set.
- Native macOS and browser-based Windows, Linux and Docker interfaces expose the model library.

## Privacy

The library is stored below `.reco-training/models/library/` in the selected local project. It contains model weights and aggregate metadata only. Videos, extracted frames and annotations are not copied into model entries and are never uploaded by Reco Trainer.

## Installation

- **macOS:** Open `Reco-Trainer-Mac-0.11.0.dmg`, copy Reco Trainer to Applications and open it. The alpha build is ad-hoc signed and not Apple-notarized.
- **Windows:** Extract `Reco Trainer Windows 0.11.zip` and run `Start Reco Trainer Windows.bat`.
- **Linux:** Extract `Reco Trainer Linux 0.11.zip`, make `Start Reco Trainer Linux.sh` executable and run it.
- **Docker:** Extract `Reco Trainer Docker 0.11.zip`, configure the local video-folder mount and start Docker Compose.

Keep a backup of the complete `.reco-training` folder before upgrading important projects. Import `.recomodel` packages only from trusted publishers.

## Verification

SHA-256 checksums are attached to the GitHub release as `SHA256SUMS.txt`.
