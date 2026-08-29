# Reco Trainer 0.7: Private local training and automatic model comparison (work in progress)

Reco Trainer is an experimental, privacy-first tool for improving sports-camera detection models without uploading sensitive match footage. Videos, extracted frames, corrected labels, training runs and benchmark reports stay on the user's own computer.

Version 0.7 adds a local model test platform. You define the correct answers by reviewing and freezing the annotations in a known image set. Reco Trainer then runs every compatible downloaded `.recomodel` package against exactly the same images and creates an automatic ranking.

The ranking includes mAP@0.50, precision, recall, F1, mean IoU, false positives, false negatives and inference time per image. Its quality score uses 70% mAP@0.50 and 30% F1; latency is only a tie-breaker. Models run sequentially to prevent them from competing for the same GPU, Apple Neural Engine or memory.

For an honest comparison, the benchmark directory should contain representative images that were not used to train any of the compared models.

This remains **work in progress / alpha software**. Please keep backups, verify labels manually and do not treat the highest-ranked model as production-ready without testing it on complete matches.

## Supported systems and downloads

Download the files from the [Reco Trainer 0.7 release page](https://github.com/wendibus/reco-trainer/releases/tag/v0.7.0):

- **macOS:** `Reco.Trainer.Mac.0.7.dmg` — open it, drag Reco Trainer to Applications and start it. The alpha is locally/ad-hoc signed and not Apple-notarized.
- **Windows:** `Reco.Trainer.Windows.0.7.zip` — extract it and run `Start Reco Trainer Windows.bat`.
- **Linux:** `Reco.Trainer.Linux.0.7.zip` — extract it, make `Start Reco Trainer Linux.sh` executable and run it.
- **Docker:** `Reco.Trainer.Docker.0.7.zip` — extract it, set the local video-folder mount and run Docker Compose as described in `START-HERE.md`.

The source code, full installation instructions and privacy notes are in the [GitHub repository](https://github.com/wendibus/reco-trainer). Detailed changes and checksums are in the [0.7 release notes](https://github.com/wendibus/reco-trainer/blob/main/RELEASE-NOTES-0.7.md).

## Suggested test workflow

1. Select the sport and a private local folder.
2. Prepare frames and correct all target boxes.
3. Remove incorrect automatic suggestions.
4. Import the compatible models you want to compare.
5. Open **Test models** and freeze the correct answers.
6. Run the benchmark and inspect both the ranking and the FP/FN counts.

The interface and walkthrough are available in German, English, Spanish and French. Supported sports currently include basketball, football (soccer), handball, hockey, rugby, lacrosse and American football.

Feedback is very welcome, especially for Windows/Linux installation, different Apple chips, large test sets, ranking reproducibility and cases where a numerically better model performs worse in a real match. Please include your operating system, hardware, sport, model size and the full error message—but **do not attach private footage, extracted frames or `.reco-training` folders**. Synthetic or redacted examples are preferable.
