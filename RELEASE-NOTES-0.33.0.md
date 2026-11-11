# Reco Trainer 0.33.0 — find the images the ball tracking fails on, Reco export fits Reco's threshold

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Gaps from the ball-tracking simulation become review candidates.** Below each player in the ball-tracking simulation there is a new button, **Add gaps as review candidates**. A gap is a short stretch (up to 1.5 seconds) in which the ball is missing between two detections. Reco Trainer looks at those images again in a close-up around the place where the ball should be and adds them to the project as review candidates, with a suggested box when the ball was found there. You then label exactly the images the model fails on instead of random ones.
  - Up to three images per gap and 60 per run. Images added before are skipped, and the project is backed up first.
  - Videos that have images in the independent test are refused, so the test stays independent.
  - Until you click this button, the simulation still stores nothing in the project.
- **Export for Reco fits the confidences to Reco's threshold (RF-DETR models).** Without a setting, Reco accepts every detection from 10 % confidence. RF-DETR models report many false balls and players at that level. After writing the file, the export now runs it the way Reco does on up to 200 validation images, finds the best threshold for each class, and rescales the confidences so that this threshold sits at 10 % in the file. The order of the detections does not change. For the ball (and puck), finding it counts twice as much as avoiding a false one.
  - The export reports the thresholds it found and stores them in the file's metadata. Leave Reco's threshold at 0.10.
  - A class with fewer than ten labelled objects in the validation images stays as it is. Without validation images nothing is changed and the export says so.
  - YOLO exports are unchanged.
- **Export for Reco follows the most common image shape.** The file is built for the image shape most project images have (before: the median of the first 40 images), and the export warns when more than 10 % of the images have another shape. Reco sees the raw camera images, so a project for it should hold images of one camera shape.

## Measured

On 128 camera images of one football project at Reco's default threshold of 0.10 (14.5 labelled persons per image):

| Model | File | Balls found | False balls per image | Persons per image |
|---|---|---|---|---|
| RF-DETR Small | before | 86 % | 0.47 | 23.6 |
| RF-DETR Small | fitted | 75 % | 0.05 | 12.0 |
| RF-DETR Nano | before | 84 % | 0.61 | 32.4 |
| RF-DETR Nano | fitted | 61 % | 0.08 | 11.5 |

## Notes

- Fitting trades some found balls for far fewer false ones. If you want more balls, lower Reco's threshold below 0.10.
- The thresholds come from the validation images of the model's own project; for footage from another venue they tend to be too strict.
- Not yet tested inside the Reco app itself. The gap search is covered by tests with constructed data; it has not been tried on a real match video yet.
