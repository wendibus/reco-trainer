# Reco Trainer 0.30.5 — Empty images only when confirmed

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## Fixed

- Images without boxes always went into training, including video frames nobody had labelled yet. That taught the model that the players and the ball on them were background. Now an image without boxes only trains after **Image is empty** was clicked in the editor. Candidates answered with "no ball" and images imported with a label file count as checked. Training reports how many images were left out, and the summary shows them as "without boxes, not in training".

## New

- **Image is empty** button in the editor, for empty halls and other scenes with nothing to mark. Such background images help the model tell lamps, logos and goals from balls and players. Mark visible hoops on them as usual.
- Augmentation Studio: the role "Static" is now called **Stays in place**. Hoops and goals are not moved but keep their class and box.

Note: because left-out images can change which videos exist for the split, the train/validation/test split may differ from 0.30.4.
