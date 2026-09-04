# Reco Trainer 0.12: Safer editing and clearer model comparison (work in progress)

Reco Trainer is an experimental, privacy-first tool for improving sports-camera detection models without uploading sensitive match footage. Videos, extracted frames, labels, training runs and benchmark data stay on the user's own computer.

Version 0.12 adds **ten-step undo and redo** for annotation changes. The controls work independently for each image and support `Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z`.

Creating a `.recomodel` exchange package now starts with a name prompt. The chosen readable name is stored in the package and used for its portable file name. The package still contains no videos, frames, local paths or video file names.

The model comparison now contains an expandable, plain-language glossary explaining quality score, mAP@0.50, precision, recall, F1, false positives, false negatives, inference time and confidence threshold.

The versioned local model library introduced in 0.11 remains available. Every completed training run is stored as a separate model instead of silently overwriting the previous result. Reco Trainer marks the active and best evaluated models and protects the active version against accidental deletion.

If a newly trained model performs worse than the active model on comparable evaluation data, it is still preserved for inspection, but it is not activated automatically. You can rename models, activate a different version, or delete inactive versions from the interface. The active version is protected against accidental deletion.

The comparison prefers results from the independent test set. If no test result is available, validation mAP is shown and clearly identified as such. For a meaningful ranking, use representative images from different recordings that were not included in training.

Reco Trainer also includes resumable training from the latest checkpoint, permanently visible training subfolders, OpenCV-assisted box checks, Futsal support, safer video-folder scanning, reviewed active learning from new videos, and the local multi-model benchmark.

This remains **work in progress / alpha software**. Keep backups, verify automatically generated labels, and evaluate every model on complete matches before using it in a production camera workflow.

## Downloads and installation

Download the files from the [Reco Trainer 0.12 release page](https://github.com/wendibus/reco-trainer/releases/tag/v0.12.0):

- **macOS:** `Reco-Trainer-Mac-0.12.0.dmg` — open the image, drag Reco Trainer to Applications and launch it. The app is locally/ad-hoc signed and not Apple-notarized.
- **Windows:** `Reco.Trainer.Windows.0.12.zip` — extract the complete archive and run `Start Reco Trainer Windows.bat`.
- **Linux:** `Reco.Trainer.Linux.0.12.zip` — extract the archive, make `Start Reco Trainer Linux.sh` executable and run it.
- **Docker:** `Reco.Trainer.Docker.0.12.zip` — extract the archive and follow `START-HERE.md` to mount a private local working folder and start Docker Compose.
- **Checksums:** use `SHA256SUMS.txt` to verify that your download is complete and unchanged.

The [GitHub repository](https://github.com/wendibus/reco-trainer) contains the source code, installation instructions and privacy notes. See the [0.12 release notes](https://github.com/wendibus/reco-trainer/blob/main/RELEASE-NOTES-0.12.md) for the detailed changes.

## Suggested model workflow

1. Select a sport and a private local working folder.
2. Prepare frames, remove unsuitable images and correct the target boxes.
3. Train the model; Reco Trainer archives the result as a new version.
4. Review the displayed test or validation score and give the model a useful name.
5. Use **Test models** with a frozen, previously unseen image set to compare all compatible models fairly.
6. Activate the winner only after checking false positives, false negatives and real-match behaviour.
7. Add reviewed examples from new videos and repeat the cycle without losing older models.

The interface and walkthrough are available in German, English, Spanish and French. Supported sports include basketball, football (soccer), Futsal, handball, hockey, rugby, lacrosse and American football.

Please test version 0.12 and report your operating system, hardware, sport, model size and the full error message. **Do not upload private footage, extracted frames, labels or `.reco-training` folders.** Synthetic or redacted examples are preferable.
