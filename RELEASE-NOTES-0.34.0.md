# Reco Trainer 0.34.0 — ball-tracking simulation: clear progress, cancel with a choice, notable scenes

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Progress in three steps.** The ball-tracking simulation now says what it is doing: extract images, load the model, find the ball, each with a counter, the running time and the time left. Before, the bar simply stood still while the model was loading.
- **Cancel with a choice.** When you cancel a simulation you are asked what to do with the images: keep simulating, cancel and delete the images of this run, or cancel and keep what is analysed. Keeping shows the finished images as a shorter simulation ("Cancelled run: 56 of 240 images analysed").
- **Notable scenes.** After a run the simulation lists the stretches where the model is probably wrong, with a strip along the clip and a list. A click jumps to the scene and pauses.
  - **Gap:** the ball is briefly missing between two detections.
  - **Jump:** two detections lie further apart than a ball travels in that time; at least one of them is something else.
  - **Unsure:** the ball is found for several images, but only with low confidence.
  - **Lost:** no ball for more than 1.5 seconds.
- **Choose what goes into the project.** Tick the scenes you want; gaps and jumps are ticked to begin with. **Add scenes** puts a few images of each ticked scene into the image list as review candidates, where you check every image yourself. For a gap, the ball is first looked for again in a close-up; for a jump or an unsure stretch the image carries the box the model reported; a lost stretch comes without a box. This replaces "Add gaps as review candidates".

## Fixed

- A cancelled or failed simulation no longer leaves the previous simulation on screen with its images already deleted.
- A simulation is only shown in the project it was made in.

## Notes

- For a jump, the suggested box is usually the wrong detection. Remove it in the image list instead of confirming it, so the image teaches the model what is not a ball.
- The simulation is kept in memory only: after restarting the app, run it again.
- Checked with a real model on a 20-second clip (240 images, 19 scenes; cancel with keep and with delete; 29 images from 13 scenes added). Not yet tried on a full match or in other sports.
