# Reco Trainer 0.38.1 — find pending automatic boxes, remove unlabelled images

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Auto pending filter.** Next to the role chips above the image list, a new chip shows only the images with at least one automatic box nobody has confirmed or corrected yet. It switches back to **All** once nothing is pending.
- **Next pending image.** A button jumps from the selected image to the next one with an unconfirmed automatic box and starts again at the front after the last one.

- **Remove unlabelled images.** With the **Not in training** chip active, a button removes all images without boxes that are not confirmed as empty (for example half time or camera set-up), after a confirmation. Labelled images, images confirmed empty, review candidates and the independent test are kept, and source videos are never touched.

## Notes

- Tested with the type check and the existing tests; the interface was not opened in a browser.
