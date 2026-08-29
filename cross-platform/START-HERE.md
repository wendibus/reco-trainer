# Reco Trainer Local 0.7

Diese Version ist keine reine Vorschau mehr. Der Browser dient als Oberfläche; ein ausschließlich an `127.0.0.1` gebundener lokaler Worker verarbeitet die ausgewählten Videos auf demselben Mac.

## Start

1. `Start Reco Trainer.command` doppelklicken.
2. Das Terminalfenster geöffnet lassen.
3. In Reco Trainer die Sportart wählen.
4. „Videoordner auswählen“ anklicken und den Ordner im Finder bestätigen.
5. „Videos lokal vorbereiten“ anklicken. Nun werden echte JPEG-Frames unter `.reco-training/frames/` im gewählten Ordner erzeugt. Eine spätere Aktualisierung übernimmt vorhandene Markierungen und legt vorher eine lokale Sicherung an.
6. „ML einrichten“ einmalig ausführen. Dabei wird eine isolierte Python-Umgebung unter `.reco-training/.runtime/` eingerichtet und RF-DETR heruntergeladen.
7. Automatisch markieren, die Mindest-Sicherheit einstellen und Markierungen prüfen. Automatische Vorschläge können pro Frame übernommen oder verworfen werden.
8. Boxen lassen sich auswählen, verschieben, an vier Ecken ändern und löschen. Rückgängig/Wiederholen sowie Pfeiltasten für den Framewechsel sind verfügbar. Pinch oder Mausrad zoomen; „Verschieben“ bewegt die Ansicht.
9. RF-DETR Nano oder Small auswählen und lokal trainieren. Nach dem Training zeigt Reco Trainer die Datensatzaufteilung und – sofern von der installierten RF-DETR-Version geliefert – die Validierungsqualität an.
10. Das trainierte Apple-Modell als Core ML exportieren.
11. „Austauschpaket“ erstellt eine `.recomodel`-Datei mit Modellgewichten, Prüfsummen und aggregierten Angaben. Sie enthält keine Videos, Frames, Pfade oder Videodateinamen.
12. „Modell importieren“ prüft ein erhaltenes `.recomodel`-Paket und aktiviert es nur, wenn Sportart, Modellgröße, Klassen, Paketstruktur und Prüfsumme zum Projekt passen. Die lokale Bibliothek liegt unter `.reco-training/models/`.
13. „Modelle testen“ öffnet den lokalen Benchmark. Zuerst alle automatischen Vorschläge prüfen und die richtigen Antworten festlegen. Danach werden alle kompatiblen importierten Modelle nacheinander auf denselben Bildern bewertet. Das Ranking nutzt 70 % mAP@0.50 und 30 % F1; die Geschwindigkeit entscheidet nur bei Gleichstand. Referenz, Vorhersagen und Berichte bleiben unter `.reco-training/benchmarks/`.

Zum Beenden `Stop Reco Preview.command` doppelklicken oder im Terminal `Ctrl+C` drücken.

## Voraussetzungen

- macOS 14 oder neuer
- Node.js 22 oder neuer
- Xcode Command Line Tools
- FFmpeg ist optional. Falls es vorhanden ist, wird es verwendet; andernfalls baut Reco Trainer lokal einen kleinen AVFoundation-Extraktor aus dem enthaltenen Swift-Quelltext.

## Datenschutz

- Der Worker lauscht ausschließlich auf `127.0.0.1:8766`.
- Er besitzt keinen Upload-Endpunkt.
- Ordner werden über den nativen macOS-Dialog ausgewählt.
- Videos, Frames, Markierungen, Trainingsläufe und Modelle liegen ausschließlich unter `.reco-training/` im ausgewählten Videoordner.
- Vor Änderungen werden bis zu 30 lokale Projekt-Sicherungen unter `.reco-training/backups/` aufbewahrt.
- Das erstmalige Einrichten von RF-DETR benötigt Internetzugriff für die ML-Pakete und Modellgewichte. Die Sportvideos werden dabei nicht übertragen.

## Windows und Linux

- Das Paket **Windows** benötigt Node.js 22, Python 3.11/3.12 und FFmpeg. Danach `Start Reco Trainer Windows.bat` starten.
- Das Paket **Linux** benötigt Node.js 22, Python 3.11/3.12, FFmpeg sowie Zenity oder KDialog. Danach `Start Reco Trainer Linux.sh` starten.
- Auf beiden Plattformen läuft der Worker ausschließlich lokal. Ohne CUDA wird das universelle CPU-Training verwendet.
- Alternativ kann der Docker-Modus mit einem explizit eingebundenen Videoordner verwendet werden. Auch dort werden beide Ports nur an `127.0.0.1` des Host-Rechners veröffentlicht.
- Im Docker-Modus kann ein zu importierendes Paket als einzige `.recomodel`-Datei in den eingebundenen Videoordner oder nach `.reco-training/inbox/` gelegt werden.

Der Ablauf für Modellpakete und GitHub Releases steht in `MODEL-EXCHANGE.md`.

---

# Reco Trainer Local 0.7

This is no longer a UI-only preview. The browser provides the interface while a loopback-only worker processes selected videos on the same Mac.

Double-click `Start Reco Trainer.command`, select a sport and folder, and choose “Prepare videos locally”. Real JPEG frames are written to `.reco-training/frames/` inside the selected folder. “Set up ML” installs an isolated RF-DETR environment; auto-labeling, local training, and Core ML export then use the same local worker as the native Mac prototype.

Version 0.7 asks for German, English, Spanish, or French on first launch and then opens a seven-step walkthrough in the selected language with clearly visible back and forward navigation. Rugby, lacrosse, and American football join basketball, football, handball, and hockey with sport-specific class schemas. Newly created `.recomodel` packages include a canonical English description while videos, frames, paths, and file names remain excluded. Imported models are accepted only when their checksum, package structure, sport, RF-DETR size, and classes match the local project.

“Test models” opens the local benchmark platform. Review every annotation first, freeze the correct answers as ground truth, and then run every compatible imported model against the same images. Reco Trainer ranks detection quality using 70% mAP@0.50 and 30% F1, with mean inference latency used only as a tie-breaker. False positives, missed objects, per-class results, and local prediction JSON are retained below `.reco-training/benchmarks/`; no image data is uploaded.

The service binds only to `127.0.0.1:8766`, provides no upload endpoint, and keeps all media and project artifacts below `.reco-training/` in the selected folder.
