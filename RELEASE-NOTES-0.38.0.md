# Reco Trainer 0.38.0 — YOLO on AMD, field boundary before pre-labelling, only trained classes in the independent test

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **YOLO on AMD cards that stumble over MIOpen.** On some AMD cards on Windows, YOLO training stopped with a MIOpen error ("'type_traits' file not found"). The trainer now recognises it and starts the training over once without MIOpen, which is slower but works, and says so in the log. After setting up AMD training, the check also tries a BatchNorm step and remembers the result, so YOLO can start without MIOpen right away. RF-DETR was never affected.
- **Field boundary before pre-labelling.** Before **Auto-label**, **Expand dataset** and the **Independent model test**, the app offers to set the field boundary first when there is none yet. With one, pre-labelling only marks people standing on the field. **Apply to all videos** covers new videos too.
- **Independent model test: only trained classes.** The test is pre-labelled only with the classes your model was trained on, not with every class of the sport.
- **Extend dataset is easier to find.** Once a model has been trained, the button is highlighted with the note "Next step: review new videos so the model keeps learning."

## Fixed

- YOLO's check of mixed precision no longer downloads a model into whatever folder the app was started from; it goes into the app's own data folder.
- In a batch, a step that had already finished and was stopped afterwards (for example RF-DETR Medium) now says "result saved … not an error" instead of sounding like a failure.
- On Windows with an NVIDIA card, **Set up ML** now checks that PyTorch really uses the card and installs the matching CUDA build when it ended up with a CPU-only one.

## Notes

- The MIOpen workaround and the NVIDIA check were tested with simulated failures only, not on an AMD card or on a Windows computer with an NVIDIA card.
- Checked by 547 automatic tests; the interface was built and type-checked but not tried in a browser.
