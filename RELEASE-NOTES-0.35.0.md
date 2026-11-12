# Reco Trainer 0.35.0 — simulation with a section of the video, further balls, straight on to the next image

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **The simulation always looks at 12 images per second.** A long video is no longer thinned out. You choose where the section starts (seconds) and how long it is (25, 60, 120 or 180 seconds); the panel shows how many images that gives. A shorter video is taken whole. With **Same video as A**, the right side uses exactly the section of the left side. Scene times are times in the video.
- **Further balls are suggested.** When you add scenes as review candidates, every image is looked at once more for balls the simulation did not follow, for example at the edge or on the floor. They come in as suggested boxes. **Ball** confirms all ball boxes of the image, so check the suggestions.
- **Straight on to the next image.** After **Ball**, **No ball**, **Skip** or confirming an independent test image, the next image in the queue opens.

## Fixed

- Neighbouring simulation images are close in time again, so the states (interpolated, held) and the notable scenes mean what they say. Before, a video longer than 25 seconds was thinned out and the scenes were unreliable.
- After saving a decision, the selection could jump back to the start of the queue instead of the next image.

## Notes

- A section of 180 seconds is 2,160 images at the video's own resolution, about 1 GB of temporary space for Full HD per side.
- If the second look for further balls fails, the candidates are still added, without the extra boxes.
- Checked with a real model (RF-DETR nano, basketball): 60 seconds (720 images, 19 s) and 180 seconds (2,160 images, 49 s) from a one-hour video; side B on the same section; the second look on five real images. The interface was built and type-checked but the end-to-end browser test was not run.
- More than 40 scenes in a long section are cut without a notice.
