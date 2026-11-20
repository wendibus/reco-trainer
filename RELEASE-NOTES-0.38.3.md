# Reco Trainer 0.38.3 — search, filter and sort the image list, use review candidates for training

Videos, frames, labels, checkpoints, and metrics remain entirely on the local computer.

## New

- **Search, filter and sort above the image list.** Search by video name or time (for example `12:30`), filter by the number of boxes (**Boxes from … to …**, so 1 to 3 finds sparsely labelled images and 0 to 0 finds empty ones), find images **with** or **without** a class (for example with ball, without player), and sort by fewest or most boxes. The filters combine with the existing chips. The arrow keys follow the filtered list, the open image stays listed while you fix it, and your choice is restored on the next start.
- **Class missing.** A new filter shows images with boxes that lack a class which at least 60 % of the labelled images have (from 8 labelled images on), typically only the ball marked and no players. Missing players can throw off ball tracking when such images go into training.
- **Use for training.** Review candidates that already have boxes (for example from ball tracking) can be put into training with one click. Their boxes count as confirmed afterwards.
- **Class dots.** Each image in the list shows a small dot in the class colour for every class it contains.
- The hint on **Not in training** now explains that review candidates with boxes are listed there until you confirm them.

## Notes

- Tested with the type check, the existing tests and a script for the filter logic; the interface was not opened in a browser. The 60 % threshold of **Class missing** was not tried on real project data.

## Downloads

- **macOS Apple Silicon:** existing installations receive 0.38.3 through the in-app updater. The launcher is unchanged.
- **Mac update package:** `Reco.Trainer.Payload.0.38.3.zip` + `.sig` (installed automatically by the app)
- **Windows:** `Reco.Trainer.Windows.0.38.3.zip`
- **Linux:** `Reco.Trainer.Linux.0.38.3.zip`
- **Docker:** `Reco.Trainer.Docker.0.38.3.zip`

`SHA256SUMS-0.38.3.txt` contains checksums for all attached update and platform packages.
