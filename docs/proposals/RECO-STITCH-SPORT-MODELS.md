# Feature request: sport-specific model profiles and RF-DETR model import

## Motivation

Ball appearance, scale, motion, camera angle and the most common false positives differ substantially between basketball, football, futsal, handball, hockey, rugby, lacrosse and American football. A single generic detector and one shared confidence setting are unlikely to be optimal for every sport.

I would like to propose sport-specific model profiles for Reco Stitch. As a first public experiment, I am sharing a locally trained basketball ball detector created with [Reco Trainer](https://github.com/wendibus/reco-trainer).

## Proposed user experience

1. The user selects a sport when creating or opening a project.
2. Reco Stitch offers compatible local models for that sport and clearly labels experimental models.
3. Each model declares its architecture, input size, class mapping, post-processor, checksum, license status and measured domain.
4. Per-sport defaults can set confidence, crop strategy and tracking behavior without hiding advanced controls.
5. Users can import a local model package and can always fall back to the current generic model.
6. Model selection remains local and never uploads source footage.

## Architecture proposal

Reco Stitch currently exposes a YOLO-oriented model chooser and YOLO output parsing. The first shared model is RF-DETR, so changing only the file extension would not be sufficient. A small detector adapter layer could select both the runtime and the correct pre-/post-processing implementation.

Suggested manifest fields:

- model ID, revision and display name
- sport and detection classes
- architecture and post-processor (`yolo`, `rf-detr`, or future adapters)
- artifact format and input resolution
- class-index mapping
- recommended confidence range
- model-weight license and upstream architecture license
- SHA-256 checksum
- validation/test metrics and dataset-domain notes
- minimum compatible Reco Stitch version

The first integration question is the preferred interoperable contract for RF-DETR: direct support for a documented ONNX export plus RF-DETR post-processing, or a small conversion/import tool that produces a Reco Stitch model bundle.

## First WIP basketball model

**Basketball Ball RF-DETR Small — WIP 0.1** is an RF-DETR Small checkpoint trained locally for one `ball` class.

Independent-test metrics on 64 frames with 50 labelled balls:

| Metric | Result | Plain-language interpretation |
| --- | ---: | --- |
| mAP@0.50:0.95 | 0.443 | Overall detection and localization quality across strict overlap thresholds |
| mAP@0.50 | 0.793 | Detection quality with a relatively forgiving box-overlap requirement |
| mAP@0.75 | 0.401 | Localization quality with a stricter overlap requirement |
| Precision | 0.909 | About 91% of reported detections were correct at the F1-selected operating point |
| Recall | 0.800 | About 80% of labelled balls were found at that operating point |
| F1 | 0.851 | Combined balance of precision and recall |
| mAR | 0.560 | Average recall across the evaluator's settings |

These figures are promising enough for rapid continued fine-tuning, but the test set is too small and narrow to claim production readiness. The model is an object detector rather than a complete temporal tracker, and it is not directly compatible with Reco Stitch's current YOLO post-processing path.

Model package, checksum, full results and limitations: [Basketball Ball RF-DETR Small — WIP 0.1](https://github.com/wendibus/reco-trainer/releases/tag/model-basketball-rfdetr-small-wip-0.1)

## Privacy-first community improvement

[Reco Trainer](https://github.com/wendibus/reco-trainer) extracts frames, reviews labels, trains, tests and compares models locally. Videos and images stay on the contributor's computer. Only an explicitly exported `.recomodel` package contains weights, checksums and aggregate metadata.

Please try the model with your own locally held data, correct its mistakes and retrain it for different venues, cameras and lighting. Keep a frozen independent test set, report aggregate metrics, and never publish private footage just to contribute to model development.

## Suggested acceptance criteria

- A detector interface supports architecture-specific preprocessing and post-processing.
- A model manifest validates architecture, classes, input format, checksum and compatibility.
- The project sport filters or recommends models without preventing manual selection.
- Imported models and footage remain local by default.
- Unsupported or incompatible packages fail with a clear explanation.
- A repeatable benchmark can compare candidate models on the same labelled test set.
