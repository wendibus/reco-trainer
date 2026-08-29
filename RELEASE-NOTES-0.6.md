# Reco Trainer 0.6 — Work in Progress Alpha

## Package descriptions

- **Reco Trainer Mac 0.6** — Native Apple-silicon application for private, local sports-video annotation, RF-DETR training, Core ML export, and validated `.recomodel` exchange.
- **Reco Trainer Windows 0.6** — Local Windows package with native folder selection, frame review, CPU-based RF-DETR training, model export, and validated model exchange.
- **Reco Trainer Linux 0.6** — Local Linux package with native folder selection where supported, frame review, CPU-based RF-DETR training, model export, and validated model exchange.
- **Reco Trainer Docker 0.6** — Portable CPU-based container package for macOS, Windows, and Linux with local volume mounting and no media upload endpoint.

## Highlights

- First-launch language selection: German, English, Spanish, or French.
- Six-step walkthrough in the selected language with visible back and forward arrows.
- Basketball, football, handball, hockey, rugby, lacrosse, and American football schemas.
- Local frame extraction, annotation correction, RF-DETR training, and model exchange.
- New `.recomodel` manifests contain a canonical English package description.
- Model packages contain weights and aggregate metadata only; videos and frames remain local.
- Imported model packages are checked for structure, checksum, sport, RF-DETR size, and classes.
- Next.js 16.3.3; the production dependency audit reported zero known vulnerabilities when this release was built.

## SHA-256

```text
fe81fcd5f2a76eeed10d9abeb4f8ffd17387c75dad537870db1bcdcc6f9b9311  Reco.Trainer.Mac.0.6.dmg
61c808e67590f0a0f30eb697a07ca1d239fc79c36c95756a826671c5b64c5d7e  Reco.Trainer.Windows.0.6.zip
ade97a43ad6aac1346fe384cabf7a37f1d24d14677280d6214c6316703007030  Reco.Trainer.Linux.0.6.zip
b9cfcdd1ce4e38a94a556295ae90cf09b2d0909dd15a388954dcc5dfb71fb06a  Reco.Trainer.Docker.0.6.zip
```

The Mac build is locally/ad-hoc signed and not Apple-notarized. This is alpha software. Keep backups, review all automatic labels, and import `.recomodel` files only from trusted publishers.
