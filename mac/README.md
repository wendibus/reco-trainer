# Reco Trainer for Mac

Ein lokaler macOS-Prototyp zum Erstellen sportartspezifischer Objekterkennungsmodelle. Originalvideos und extrahierte Frames bleiben im gewählten Ordner. Die Anwendung enthält keinen Upload-Befehl.

## Enthalten

- native SwiftUI-Oberfläche für macOS 14+
- Sportarten Fußball, Basketball, Handball und Hockey
- lokale Videoauswahl und Frame-Extraktion über AVFoundation
- visueller Editor für Ball/Puck, Spieler, Torwart, Schiedsrichter und sportartspezifische Objekte
- Pinch-Zoom bis 8× sowie Markieren-/Verschieben-Modus
- umschaltbare deutsche, englische, spanische und französische Oberfläche
- RF-DETR Nano/Small unter Apache 2.0
- automatische Vorbeschriftung mit Korrekturworkflow
- automatische Vorbeschriftung nur für die aktuell gewählte Klasse
- Training auf Apple Silicon über PyTorch MPS, mit CPU-Fallback
- automatisches Apple-Silicon-Leistungsprofil anhand von gemeinsamem Speicher und CPU-Kernen
- speicherabhängige Batch-Größe, parallele Datenlader, Prefetching und dauerhaft laufende Worker
- Export als universelles ONNX-CPU-Modell
- FP16-Export als Apple-Core-ML-Paket für eine effiziente Ausführung auf CPU, GPU und Neural Engine
- COCO-Datensatz mit Train-/Valid-/Test-Splits
- lokale Projektsicherungen vor Änderungen (maximal 30)
- datenschutzfreundliche `.recomodel`-Austauschpakete ohne Videos, Frames, Pfade oder Videodateinamen
- lokaler Modell-Benchmark mit eingefrorener Ground Truth, mAP@0.50, Präzision, Recall, F1, Fehlalarmen, übersehenen Objekten und Zeit pro Bild

#

Voraussetzungen: macOS 14 oder neuer sowie die Xcode Command Line Tools.

```bash
cd /Pfad/zu/reco-trainer-mac
./scripts/run-mac.sh
```

Alternativ erzeugt dieser Befehl eine lokal signierte `.app`:

```bash
./scripts/package-app.sh
open "dist/Reco Trainer.app"
```

Ein installierbares DMG wird so erstellt:

```bash
./scripts/package-dmg.sh
```

Ohne Apple-Developer-ID wird die App lokal/ad-hoc signiert. Für eine öffentliche, notarierte Veröffentlichung werden `RECO_SIGN_IDENTITY` und für `scripts/notarize.sh` zusätzlich ein mit `notarytool` gespeichertes `RECO_NOTARY_PROFILE` benötigt.

Beim ersten Start:

1. Sportart wählen.
2. Einen Ordner mit `.mov`, `.mp4` oder `.m4v`-Videos wählen.
3. **Videos lokal analysieren** anklicken.
4. **ML einrichten** anklicken. Dieser Schritt legt eine isolierte Python-Umgebung unter `.reco-training/.runtime/` im gewählten Ordner an und lädt die ML-Abhängigkeiten.
5. Optional **Automatisch markieren** verwenden und die Boxen korrigieren.
6. **Lokal trainieren** starten.
7. **CPU-Modell (ONNX)** beziehungsweise **Apple-Modell (Core ML)** exportieren.
8. Optional **Paket erstellen** anklicken. Die erzeugte `.recomodel`-Datei kann separat geprüft und anschließend bewusst in ein GitHub Release hochgeladen werden.
9. Ein erhaltenes Paket kann über **Modell importieren** ausgewählt werden. Reco Trainer prüft Struktur, Prüfsumme, Sportart, RF-DETR-Größe und Klassen, installiert es unter `.reco-training/models/library/` und aktiviert es lokal.
10. **Modelle testen** öffnen, alle automatischen Vorschläge vorher prüfen und die richtigen Antworten festlegen. Danach bewertet Reco Trainer alle kompatiblen importierten Modelle nacheinander auf denselben Bildern und erstellt ein lokales Ranking.

Alle Projektdaten liegen unter:

```text
GEWÄHLTER_VIDEOORDNER/.reco-training/
├── project.json
├── frames/
├── dataset/
│   ├── train/
│   ├── valid/
│   └── test/
├── runs/
├── exports/
│   └── share/
├── backups/
├── models/
│   ├── active.json
│   └── library/
├── benchmarks/
│   ├── ground-truth.json
│   ├── latest.json
│   └── runs/
└── .runtime/
```

## Das CPU-Modell

Der ONNX-Export läuft bewusst über den CPU-Pfad. Die erzeugte Datei ist nicht an Apple Silicon gebunden und kann später mit ONNX Runtime auf macOS, Windows und Linux verwendet werden. Hardwareoptimierte Laufzeit-Backends können darauf aufbauen, ohne das Modell neu zu trainieren.

Core ML ist ein zusätzlicher Apple-spezifischer Export für CPU, GPU und – soweit die Modelloperatoren unterstützt werden – Neural Engine.
Der Export verwendet die von RF-DETR unterstützte Standardbenennung. Optionale Parameter wie FP16 werden versionsabhängig nur dann übergeben, wenn die installierte Export-API sie ausdrücklich anbietet.

## Apple-Silicon-Leistung

Beim Training verwendet PyTorch die Apple-GPU über Metal/MPS. Die App bestimmt aus dem gemeinsamen Speicher und der Zahl der CPU-Kerne automatisch eine geeignete Batch-Größe. Mehrere dauerhafte Datenlader bereiten Trainingsbeispiele parallel vor, damit die GPU möglichst wenig warten muss. Die effektive Batch-Größe bleibt durch Gradienten-Akkumulation stabil.

Mehrere vollständige Trainings gleichzeitig zu starten ist auf einem einzelnen Mac nicht sinnvoll: Sie konkurrieren um dieselbe GPU und denselben gemeinsamen Speicher. Die Apple Neural Engine wird von diesem PyTorch-Training nicht direkt angesprochen. Sie kommt beim exportierten Core-ML-Modell für die spätere Erkennung zum Einsatz, sofern Core ML alle Modelloperationen dort ausführen kann.

Das gewählte Leistungsprofil wird vor jedem Training auf Deutsch oder Englisch protokolliert. Es werden keine unsicheren Speichergrenzen gesetzt; macOS behält genügend Spielraum für das System.

## Tests

```bash
swift test
./scripts/test-worker.sh
```

## Datenschutz und Weitergabe

Lokale Verarbeitung allein klärt nicht automatisch die Rechte am Trainingsmaterial. Vor einer Weitergabe trainierter Gewichte muss geprüft werden, ob die Aufnahmen für Training und Modellveröffentlichung verwendet werden dürfen. Ein `.recomodel`-Paket enthält ausschließlich `manifest.json` und die Modellgewichte. Das Manifest verwendet aggregierte Trainingsangaben und Prüfsummen; Quellpfade, Videos, Frames, Annotationen und Videodateinamen werden nicht aufgenommen.

## Bekannte Grenzen des ersten MVP

- Die automatische Vorbeschriftung des allgemeinen Basismodells erkennt vor allem `person` und `sports ball`. Körbe, Tore und sportartspezifische Rollen müssen zunächst manuell ergänzt werden.
- Bereits manuell markierte Frames werden bei der automatischen Vorbeschriftung bewusst nicht überschrieben.
- Bei nur einem Video wird ein zeitlicher Split verwendet. Belastbare Qualitätsmessungen benötigen mehrere unabhängige Spiele.
- Gewichtsdateien werden nicht automatisch auf GitHub zusammengeführt. Austauschpakete sollten erst offline validiert, auf einem unabhängigen Referenzsatz verglichen und danach manuell als Release freigegeben werden.
- Der erste Download der RF-DETR-Gewichte benötigt Internetzugriff; danach kann das Training offline laufen.
