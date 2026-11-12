# Reco Trainer 0.35.0 — simulation with a section and a choice of density, further balls, straight on to the next image

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Choose the section and how dense the images are.** The simulation no longer has a fixed limit of 300 images. You choose where the section starts (seconds), how long it is, and whether to look at all 12 images per second or only every 2nd, 3rd, 4th, 6th or 12th. One run holds up to 2,160 images: 180 seconds with all images, 540 seconds with every 3rd. The panel shows how many images that gives. A shorter video is taken whole. With **Same video as A**, the right side uses exactly the section of the left side. Scene times are times in the video.
- **Look-ahead in seconds.** The slider "How far to look directly into the future" is now set in seconds, so it means the same at every density of images.
- **Hints and checks.** From every 4th image on, the panel says that gaps and jumps are only seen roughly. Before a run it checks that the images fit on the disk and, if not, suggests a sparser density or a shorter section. The list of notable scenes grows with the length of the section (40 per minute), and says how many more scenes are not shown.
- **Further balls are suggested.** When you add scenes as review candidates, every image is looked at once more for balls the simulation did not follow, for example at the edge or on the floor. They come in as suggested boxes. **Ball** confirms all ball boxes of the image, so check the suggestions.
- **Straight on to the next image.** After **Ball**, **No ball**, **Skip** or confirming an independent test image, the next image in the queue opens.

## Fixed

- The images of a simulation lie evenly spaced in time again, so the states (interpolated, held) and the notable scenes mean what they say. Before, a video longer than 25 seconds was thinned out and the scenes were unreliable.
- After saving a decision, the selection could jump back to the start of the queue instead of the next image.

## Notes

- A section of 180 seconds is 2,160 images at the video's own resolution, about 1 GB of temporary space for Full HD per side.
- If the second look for further balls fails, the candidates are still added, without the extra boxes.
- Checked with a real model (RF-DETR nano, basketball): 60 seconds (720 images, 19 s) and 180 seconds (2,160 images, 49 s) from a one-hour video; side B on the same section; the second look on five real images. The interface was built and type-checked but the end-to-end browser test was not run.
- Start, length and density are remembered per project, in this browser only.
