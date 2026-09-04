# Änderungen

## 0.12.1

- Abgebrochene Trainingsläufe werden vollständig mit Optimizer, Lernratenplan, EMA und Callback-Zuständen fortgesetzt.
- Abgeschlossene Modelle starten neue Feinabstimmungszyklen vom besten Checkpoint mit reduzierter Lernrate und Cosine-Scheduler.
- Geduldigeres Early Stopping reduziert verfrühte Abbrüche bei schwankenden Validierungswerten.
- Vorherige Messprotokolle werden vor einem neuen Lauf automatisch in einem eindeutigen Verlaufsordner archiviert.
- Kleine Validierungs- und Testsätze werden transparent als vorläufig gekennzeichnet.
- Unnötige TensorBoard-Warnungen sind im lokalen Standardprofil deaktiviert.
- Alle neuen Meldungen stehen auf Deutsch, Englisch, Spanisch und Französisch bereit.

## 0.12.0

- Zehnstufiges Rückgängig/Wiederholen für Boxänderungen, inklusive Tastenkürzeln.
- Namensabfrage vor der Paketerstellung; der Name wird in Manifest und Dateiname übernommen.
- Aufklappbare Erklärungen zu Qualität, mAP, Präzision, Recall, F1, FP/FN, Geschwindigkeit und Schwelle im Modellvergleich.
- Alle neuen Texte stehen auf Deutsch, Englisch, Spanisch und Französisch bereit.

## 0.11.0

- Jeder lokale Trainingslauf wird als unveränderlicher Modellstand in der Modellbibliothek archiviert.
- Bestehende Checkpoints älterer Versionen werden vor dem nächsten Training automatisch gesichert.
- Verständliche Modellnamen, Datum, Größe, Validierungs-mAP sowie Aktiv-/Bestmarkierung in der Oberfläche.
- Modelle können umbenannt, aktiviert und – sofern inaktiv – gelöscht werden.
- Schlechtere Trainingsläufe bleiben vergleichbar, ersetzen aber nicht automatisch das aktive bessere Modell.
- Dieselbe Modellverwaltung ist in der Windows-, Linux- und Docker-Oberfläche verfügbar.

## 0.9.0

- Neuer Active-Learning-Workflow „Neue Videos prüfen“ mit getrenntem Videoordner.
- Lokale Kandidatenerkennung mit niedriger Schwelle und nach Unsicherheit sortierter Prüfwarteschlange.
- Ein-Klick-Entscheidungen für Ball, kein Ball und Überspringen; Boxen bleiben korrigierbar.
- Ungeprüfte Kandidaten sind technisch vom Training, Export und Benchmark ausgeschlossen.
- Bereits verwendete Videos werden nicht erneut importiert.

## 0.8.0

- Die Frame-Anzahl wird jetzt je Video gewählt; zusätzliche Videos liefern zusätzliche Trainingsbilder.
- Ungeeignete extrahierte Bilder lassen sich entfernen, ohne das Quellvideo zu verändern.
- Beim Entfernen werden abgeleitete Datensatzkopien bereinigt und veraltete Benchmark-Referenzen verworfen.

## 0.7.0

- Vollständig lokaler Modell-Benchmark für alle kompatiblen importierten `.recomodel`-Pakete.
- Geprüfte manuelle Boxen können als unveränderliche Ground-Truth-Referenz festgelegt werden; Bilder ohne Box gelten dabei ausdrücklich als negative Beispiele.
- Automatisches Ranking aus 70 % mAP@0.50 und 30 % F1; mittlere Inferenzzeit dient nur als Tie-Breaker.
- Präzision, Recall, F1, False Positives, False Negatives, mittlere IoU und Zeit pro Bild werden pro Modell lokal ausgewiesen.
- Modellvorhersagen und Benchmark-Berichte bleiben unter `.reco-training/benchmarks/`; Bilder und Videos werden nicht übertragen.
- Benchmark-Oberfläche und siebter Walkthrough-Schritt auf Deutsch, Englisch, Spanisch und Französisch.

## 0.6.0

- Sprachauswahl beim ersten Start mit Deutsch, Englisch, Spanisch und Französisch.
- Interaktiver Walkthrough in der gewählten Sprache mit deutlich sichtbarer Vorwärts- und Zurücknavigation.
- Rugby, Lacrosse und American Football als spezialisierte Sportarten ergänzt; Hockey bleibt als Puck-Sportart erhalten.
- Neue Austauschpakete enthalten eine einheitliche englische Modellbeschreibung.
- Paketimport und Klassenprüfung unterstützen alle sieben Sportarten.

## 0.5.0

- Spanische und französische Oberfläche zusätzlich zu Deutsch und Englisch.
- Datenschutzsichere `.recomodel`-Pakete können jetzt über einen nativen Dateidialog importiert werden.
- Vor der Installation werden Prüfsumme, Paketstruktur, Sportart, Modellgröße und Klassen geprüft.
- Importierte Modelle liegen in einer lokalen Modellbibliothek und werden für Vorbeschriftung oder weiteres Training aktiviert.
- Der Austausch benötigt keine eingerichtete RF-DETR-Umgebung und überträgt keine Medien.

## 0.4.0

- Sichere lokale Projektsicherungen und Erhalt vorhandener Markierungen bei erneuter Frame-Extraktion.
- Einstellbare Erkennungsschwelle für automatische Vorschläge.
- Datenschutzsicheres `.recomodel`-Austauschpaket mit Gewichten, Prüfsummen und aggregierten Metadaten; keine Videos, Frames, Pfade oder Dateinamen.
- Lokale Validierungsmetriken und Trainingshistorie werden im Projekt gespeichert.
- DMG-Erzeugung, optionale Developer-ID-Signierung und vorbereiteter Apple-Notarisierungsablauf.
- Austauschpakete sind für einen kontrollierten GitHub-Release-Prozess vorbereitet.

## 0.3.1

- Core-ML- und ONNX-Export mit älteren und neueren RF-DETR-Exportsignaturen kompatibel gemacht.
- Nicht überall unterstützten Parameter `output_name` entfernt; RF-DETR vergibt den Artefaktnamen selbst.
- Optionale Exportparameter werden nur noch übergeben, wenn sie ausdrücklich in der installierten API vorhanden sind.
- Core-ML-Präzisionswert auf das offizielle `float16` korrigiert.
- ML-Einrichtung setzt für den nativen Core-ML-Export mindestens RF-DETR 1.9.0 voraus.
- Regressionstests für alte und neue Exportsignaturen ergänzt.

## 0.3.0

- Automatisches Apple-Silicon-Leistungsprofil anhand des gemeinsamen Speichers und der CPU-Kerne.
- Größere, speicherabhängige MPS-Batches statt einer festen Batch-Größe von 1.
- Parallele, dauerhafte Datenlader mit Prefetching, damit die Apple-GPU kontinuierlicher arbeitet.
- Stabile effektive Batch-Größe durch automatisch angepasste Gradienten-Akkumulation.
- Speicheroptimiertes Gradient Checkpointing nur noch, wenn Modellgröße und verfügbarer Speicher es erfordern.
- Hardwareanzeige ergänzt um gemeinsamen Speicher und Zahl der Datenlader.
- Vorbeschriftung nutzt auf leistungsfähigeren Macs ebenfalls größere Batches.
- FP16-Core-ML-Export, sofern die installierte RF-DETR-Version ihn unterstützt.
- Deutsche und englische Erläuterung, welche Aufgaben GPU und Neural Engine übernehmen.

## 0.2.0

- Pinch-Zoom von 1× bis 8× im Bildeditor.
- Umschaltbarer Markieren-/Verschieben-Modus für vergrößerte Bilder.
- Plus-, Minus- und Zurücksetzen-Bedienelemente für den Zoom.
- Automatische Vorbeschriftung berücksichtigt nur noch die aktuell gewählte Klasse.
- Bei Ball/Puck werden Personen vollständig ignoriert.
- Vorhandene Boxen anderer Klassen bleiben beim automatischen Markieren erhalten.
- Umschaltbare deutsche und englische Oberfläche einschließlich Worker-Rückmeldungen.
- Sichtbare Warnung bei generischer Personenerkennung als „Player“.

## 0.1.1

- Markierungen bleiben jetzt ausschließlich in dem Frame, in dem sie angelegt wurden.
- Der Editor synchronisiert sich beim Framewechsel und nach automatischer Vorbeschriftung neu.
- Automatische Vorbeschriftung meldet die Zahl der erzeugten Boxen und geprüften Frames.
- Die Erkennungsschwelle für kleine Sportbälle wurde von 0,35 auf 0,25 gesenkt.
- Bereits manuell markierte Frames werden nicht überschrieben.

## 0.1.0

- Erster nativer macOS-MVP.
