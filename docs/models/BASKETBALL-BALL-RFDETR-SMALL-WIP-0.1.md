# Basketball Ball RF-DETR Small — WIP 0.1

> **Work in progress.** This checkpoint is not a production-ready tracking model. It is an early basketball ball detector intended for evaluation, local fine-tuning and discussion of sport-specific model support in Reco Stitch.

## Download and integrity

- Package: `Basketball-Ball-RF-DETR-Small-WIP-0.1-20260904-190754.recomodel`
- Format: Reco Trainer `.recomodel` package
- Architecture: RF-DETR Small 1.9.4
- Detection class: `ball`
- SHA-256: `e1fab9cc7e299555300fbe6027f47220456f29688afd77fb434163bb2c295145`

The package contains the checkpoint, a manifest, aggregate training metadata and integrity hashes. It contains no videos, extracted frames, local paths or source file names.

## Training snapshot

The checkpoint was trained locally with Reco Trainer on:

- 1,456 reviewed frames
- 1,050 ball boxes
- 1,328 training frames
- 64 validation frames with 32 labelled balls
- 64 independent test frames with 50 labelled balls

The test split was not used for training or model selection. It is small, so these results are an early signal rather than proof of broad real-world performance.

## Results

| Metric | Validation | Independent test |
| --- | ---: | ---: |
| mAP@0.50:0.95 | 0.460 | 0.443 |
| mAP@0.50 | 0.901 | 0.793 |
| mAP@0.75 | 0.370 | 0.401 |
| mAR | 0.591 | 0.560 |
| F1 | 0.889 | 0.851 |
| Precision | 0.903 | 0.909 |
| Recall | 0.875 | 0.800 |

The independent test result is the more important estimate. At the F1-selected operating point, precision 0.909 means that roughly 91% of reported detections matched a labelled ball, while recall 0.800 means that roughly 80% of labelled balls were found. F1 0.851 summarizes that precision/recall balance.

mAP@0.50 0.793 shows reasonably strong detection when a predicted and labelled box need at least 50% overlap. The lower mAP@0.50:0.95 of 0.443 averages increasingly strict overlap requirements and shows that exact box localization still needs improvement. mAP@0.75 0.401 is another stricter localization measure. mAR 0.560 averages recall over the evaluator's settings and must not be confused with the single F1-selected recall value of 0.800.

## Limitations

- The validation and test sets are small.
- The images come from a limited set of cameras, halls, lighting conditions, ball designs and viewpoints.
- Similar frames from the same source can make results look better than performance on a completely new venue or recording setup.
- This is an object detector, not a complete temporal tracker. Stable tracking also needs motion logic, confidence handling and recovery after occlusion.
- High mAP@0.50 together with lower strict-IoU metrics indicates that ball presence is learned better than exact box placement.
- Reco Stitch currently uses a YOLO-oriented inference and post-processing path. This RF-DETR checkpoint is not a drop-in Reco Stitch model yet.

## Recommended community workflow

1. Install [Reco Trainer](https://github.com/wendibus/reco-trainer) and keep all source videos local.
2. Import this `.recomodel` package as a starting checkpoint.
3. Add and review data from your own basketball cameras, venues and lighting conditions.
4. Include difficult examples: small balls, motion blur, partial occlusion, reflections, scoreboards and empty frames.
5. Keep a separate, frozen test set that is never used for training or model selection.
6. Retrain, compare revisions on exactly the same test set and share aggregate results.
7. Export only the model package. Do not upload private footage or extracted training frames.

Community reports should include hardware, input resolution, confidence threshold, dataset size, independent-test metrics and a description of the recording domain. Please do not attach private sports footage.

## Licensing status

RF-DETR itself is distributed under Apache-2.0. This Reco Trainer repository currently grants no separate software or model-weight license. The package is therefore published as an experimental evaluation artifact; users must verify the applicable rights before redistribution or commercial use. A clear model-weight license should be agreed before treating it as a reusable production asset.

