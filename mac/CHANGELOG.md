# Änderungen

## 0.38.0

YOLO auf AMD-Karten mit MIOpen-Problem, Spielfeld vor dem Vormarkieren anbieten, nur trainierte Klassen im unabhängigen Test; stilles Update für NVIDIA unter Windows.

- YOLO auf ROCm/Windows: MIOpen kompiliert den BatchNorm-Kernel zur Laufzeit und scheitert auf manchen Karten (`'type_traits' file not found`, `miopenStatusUnknownError`, gfx1101). `yolo_training.train` erkennt das (`is_miopen_failure`), schaltet MIOpen aus (`torch.backends.cudnn.enabled = False`) und startet das Training einmal neu, mit Hinweis im Protokoll (langsamer, aber ohne Laufzeit-Compile). Der Prüfschritt nach der AMD-Einrichtung (`VERIFY_SCRIPT`) probiert jetzt auch einen BatchNorm-Schritt und merkt sich `batchNorm`; ist er fehlgeschlagen, setzt `rocm_training_env()` `RECO_NO_MIOPEN=1`, und YOLO startet gleich ohne MIOpen. RF-DETR ist nicht betroffen, der Schritt entscheidet nicht über „available“. Auf einer AMD-Karte nicht nachgestellt, nur mit nachgebautem Fehler getestet.
- Ultralytics-Gewichte: Der Mixed-Precision-Check lud `yolo26n.pt` in einen relativen Ordner `weights\` im Startverzeichnis. `settings.weights_dir` zeigt jetzt auf den App-Ordner `yolo-weights` (`yolo_training.weights_folder`), dort liegen auch die Basismodelle.
- Meldung im Batch: Hat ein Schritt sein Ergebnis schon ausgegeben und wird danach beendet (z. B. RF-DETR Medium), steht „Ergebnis gesichert … kein Fehler“ statt „Der Schritt endete nicht von selbst“ (`localized_step_stopped(finished=True)`).
- Spielfeld: Vor „Automatisch markieren“, „Datensatz erweitern“ und dem unabhängigen Test fragt ein Dialog, ob zuerst das Spielfeld festgelegt werden soll, wenn es noch keins gibt (einmal je Sitzung abweisbar). Mit „Für alle Videos übernehmen“ gilt es auch für neue Videos.
- Unabhängiger Test: Vormarkiert werden nur die Klassen, die das aktive Modell (sonst das letzte Training) kennt (`trained_classes`); ohne Angabe wie bisher alle Klassen der Sportart.
- Datensatz erweitern: Sobald ein Modell trainiert ist, ist die Schaltfläche hervorgehoben (Hauptschaltfläche, Rand, Satz „Nächster Schritt“).
- Stilles Update, NVIDIA unter Windows (keine Neuigkeit im Dialog): Nach „ML einrichten“ prüft `ensure_cuda_torch`, ob PyTorch die NVIDIA-Karte wirklich nutzt. Ist es eine CPU-Version (RF-DETR hatte die CUDA-Version von cu121 überschrieben, das nur bis torch 2.5 reicht), wird dieselbe Version mit passendem torchvision aus cu126, sonst cu128 neu installiert (`--force-reinstall --no-deps`). Hat PyTorch CUDA, nutzt die Karte aber nicht, steht der Grund im Protokoll. Nie fatal: sonst bleibt es auf der CPU. Auf Windows nicht nachgestellt.
- Geprüft: 547 Tests. Nicht geprüft: AMD-Karte, NVIDIA-Windows-Rechner und die Oberfläche im Browser.
- Offen: Der Hardware-Bericht mit Leistungsstufen bei „ML einrichten“ ist weiter ein Vorschlag ohne Antwort.

## 0.37.0

Rolle je Video wählbar; YOLO26-Modelle im Model Hub teilen und laden (AGPL-3.0, gleiches Sicherheitskonzept wie RF-DETR).

- Rollen: Unter den Chips der Bilderliste legt „Rollen der Videos“ je Video Training, Validierung oder Test fest (`/api/video-role`, `project.json` `videoRoles`, bleibt beim Aktualisieren der Bilder erhalten). `dataset_split.split_frames(frames, language, video_roles)`: Ein Video bleibt ganz in seiner Rolle; fehlt Validierung oder Test, wird sie aus den übrigen Videos gewählt, solange eines zum Training bleibt (Training wählen hält ein Video heraus). Alle fünf Trainings- und Exportpfade in `ml_worker.py` lesen `videoRoles`. Ohne Wahl unverändert.
- Hub, YOLO26: `model_hub.py` teilt und lädt YOLO26 (Nano, Small, Medium) als safetensors, nur Gewichte. Teilen: `sanitize_yolo_checkpoint` liest das selbst trainierte `best.pt` mit Ultralytics, speichert die Gewichte in halber Genauigkeit und entfernt damit Trainingsargumente und lokale Pfade; Größe und Klassenzahl kommen aus dem Kopf der Datei (`inspect_yolo_header`: Breite der ersten Faltung, drei Klassenköpfe). Nur lokal trainierte Modelle (`source: local-training`, mit PyTorch-Gewichten); importierte ONNX-Modelle bleiben ausgeschlossen. Laden: sha256 prüfen, Kopf prüfen, `rebuild_yolo_checkpoint` baut das Modell aus der eigenen Definition (`yolo26<n|s|m>.yaml`) und lädt die Gewichte mit `strict=True` (passen sie nicht, Abbruch), dann ONNX-Export (`yolo_training.export_onnx`); die Bibliothek bekommt ein `onnx-yolo`-Paket mit `trainWeights`, das als Elternmodell für das nächste YOLO-Training dient (`parent` im Hub-Eintrag).
- Lizenz: Einträge tragen `architecture: yolo26`, `license: AGPL-3.0`, `framework` (Ultralytics-Version), Rechte-Erklärung Version 2 und `compat.trainer_min` mindestens 0.37.0, damit ältere Trainer sie nicht listen. Die Oberfläche zeigt die Lizenz, verlangt die Zustimmung vor dem Laden (und installiert Ultralytics bei Bedarf) und beim Hochladen eine zweite Bestätigung.
- Hub-Repo: Der Prüfcode im Hub-Repo (`scripts/hub_tools.py`) muss YOLO26 kennen, bevor jemand hochlädt: Pull Request wendibus/reco-models#3 (Merge vor dem ersten YOLO-Upload).
- Geprüft: 524 Tests. Echter Durchlauf mit Ultralytics 8.4.180 für Nano, Small und Medium: Gewichte bereinigen (kein lokaler Pfad in der Datei), Prüfung mit dem Hub-Code, lokal neu aufbauen, ONNX-Export und Lauf in onnxruntime (Ausgabe 1×300×6); falsche Größe oder Klassenzahl wird abgelehnt. Nicht getestet: ein echter Upload oder Download über GitHub, ein mit echten Daten trainiertes YOLO-Modell, die Oberfläche im Browser.
- Offen: Der Hub hat noch kein Modell; der Fall „Hub-Modell laden, weitertrainieren, zurückteilen“ (`parent`) ist nur durch Tests abgedeckt.

## 0.36.0

Simulation mit freier Länge und Anteilen, Frage nach neuem Ausschnitt; Test-Anteil für neue Bilder; „Nicht im Training“ nachlabeln.

- Länge: Das Feld ist eine freie Zahl in Sekunden; ¼, ½, ¾ und „ganz“ nehmen diesen Anteil der Videolänge (`lengthFraction`, im Worker aus der echten Dauer gerechnet, ab Start am Ende abgeschnitten). Die Grenze von 2160 Bildern je Lauf entfällt; nur mehr als 20000 Bilder lehnt `simulation_section` ab (die App schickt mit jedem Status alle Bilder der Simulation), mit Vorschlag für eine dünnere Dichte oder einen kürzeren Abschnitt. Die Platzprüfung bleibt. Das Ergebnis trägt `videoDuration` und `request` (was gewünscht war).
- Neuer Ausschnitt: Ändert man Start, Länge oder Dichte bei angezeigter Simulation, fragt das Panel nach 0,7 s „Jetzt analysieren?“. Der Worker nimmt dasselbe Video mit `again` ohne Dateidialog (`simulation_video`). Im Vergleichsmodus nur für Seite A.
- Test-Anteil: `dataset_split` kennt „neue Bilder“ (bestätigte Kandidaten mit `reviewedAt` nach `lastTraining.completedAt`; `review_candidate` setzt `reviewedAt`). Der Dialog „Lokal trainieren“ fragt, wie viele davon Test sein sollen (keine, 10, 20, 30, 50 %; Voreinstellung keine). `ml_action` ruft vor `train` und `train-batch` `pin_new_images_as_test` auf: Auswahl über einen stabilen Hash der ID, mindestens ein Bild, gespeichert als `testHold`, dauerhaft Test (`split_frames`). `preview` meldet `newImages`. Warnung: Bilder aus demselben Video ähneln sich, die Testwerte fallen zu gut aus.
- Nachlabeln: Der Trainingsdialog zeigt „N Bilder ohne Markierung nachlabeln“ (ohne Kandidaten und unabhängige Testbilder); der Knopf schließt den Dialog, stellt den Chip „Nicht im Training“ ein und springt zum ersten Bild ohne Box.
- Geprüft: 506 Tests. Echter Lauf: ¼ eines Ein-Stunden-Videos mit jedem 12. Bild, 905 Bilder in 48 s; neuer Ausschnitt ohne neue Dateiwahl (900 s, jedes 2. Bild). Oberfläche nicht im Browser angesehen.
- Offen: Ein echtes Training mit gesetztem Test-Anteil wurde nicht gestartet; die Auswahl ist durch Tests abgedeckt. Vor dieser Version bestätigte Kandidaten haben kein `reviewedAt` und zählen nicht als neu.

## 0.35.0

Simulation mit wählbarem Abschnitt und wählbarer Bilddichte, ohne feste Obergrenze von 300 Bildern; weitere Bälle beim Übernehmen vormerken; nach „Ball“ / „Kein Ball“ sofort das nächste Bild.

- Bildrate: Die Simulation nimmt 12 Bilder pro Sekunde oder nur jedes n-te davon (n = 1, 2, 3, 4, 6, 12). Bisher verteilte `extract_frames_for_video` höchstens 300 Bilder ungleich dicht über das ganze Video, bei langen Videos lagen Sekunden zwischen zwei Bildern; Zustände (interpoliert/gehalten) und `teachers.tracking_scenes` setzen dichte Bilder voraus und lieferten dort Unsinn.
- Abschnitt: `simulation_section` wählt Start, Länge und `every` (Voreinstellung 0, 25 s, alle Bilder). Es gibt keine feste Obergrenze mehr, nur höchstens 2160 Bilder je Lauf: 180 s bei 12 Bildern pro Sekunde, 540 s bei jedem 3. Bild (`fps` der Simulation ist dann 4); ein kürzeres Video wird ganz genommen, ein Start hinter dem Videoende ist ein klarer Fehler). Oberfläche: Start (Sekunden), Länge, Bilddichte und die Zahl der Bilder. „Gleiches Video wie A“ nimmt den Abschnitt von A (`STATE.simulation_sections`).
- Zeitstempel: `frames[].timestamp` ist die echte Videozeit (Start + Index/fps). Daraus entstehen IDs und Dateinamen der Prüfkandidaten (`gap-<videoid>-<ms>.jpg`), Abschnitte desselben Videos kollidieren nicht mehr. Die Szenenzeiten sind Videozeiten.
- Extraktion: ffmpeg mit `-ss`/`-t` vor der Eingabe und `fps=12/n`; `frame_extractor.swift` nimmt optional `START STEP` (sonst unverändert).
- Weitere Bälle: Beim Übernehmen von Szenen sucht der neue Befehl `ml_worker find-extra-balls` in jedem übernommenen Bild alle Bälle ab Sicherheit 0,2 (die Simulation behält nur einen je Bild). Neue Treffer, die sich nicht mit vorhandenen Boxen überdecken (`extra_ball_boxes`), kommen als Vorschläge (Quelle „auto“) ins Bild. Scheitert der Durchlauf, werden die Kandidaten ohne die Zusatzboxen übernommen.
- Prüfen: Nach „Ball“, „Kein Ball“, „Überspringen“ und der Bestätigung unabhängiger Testbilder wird das nächste Bild der Warteschlange vor dem Speichern gewählt. Vorher konnte der Effekt, der den Anfang der Warteschlange wählt, die Auswahl nach dem Speichern überschreiben.
- Vorausschau in Sekunden: Der Regler „Wie weit in die Zukunft schauen“ zählt in Sekunden (Voreinstellung 0,7 s) und rechnet je Simulation in Bilder um, damit er bei jeder Dichte dasselbe bedeutet.
- Dünne Dichte: Ab jedem 4. Bild weist die Oberfläche darauf hin, dass Lücken, Sprünge und verlorene Stücke nur grob erkannt werden.
- Szenen: Das Limit wächst mit der Länge (`teachers.scene_limit`: 40 je Minute, mindestens 40); sind es mehr, steht „N weitere Szenen werden nicht angezeigt“ (`scenesTotal`).
- Speicherplatz: `check_simulation_space` lehnt einen Lauf ab, dessen Bilder nicht auf die Platte passen (etwa 0,22 Byte je Pixel und Bild), und schlägt jedes n-te Bild oder einen kürzeren Abschnitt vor.
- Gemerkt: Start, Länge und Dichte merkt sich die Oberfläche je Projekt (`localStorage`, nur in diesem Browser; ohne Speicher gelten die Voreinstellungen).
- Geprüft: 496 Tests. Echter Lauf mit jedem 3. Bild: 300 s ab 1000 s, 1200 Bilder in 35 s, Abstand 0,25 s. Echter Lauf mit RF-DETR nano (Basketball): 60 s ab 600 s (720 Bilder, 19 s) und 180 s ab 1200 s (2160 Bilder, 49 s); Seite B mit „Gleiches Video wie A“ gleicher Abschnitt; `find-extra-balls` an 5 echten Bildern. Nicht im Browser angesehen, der End-to-End-Test der Oberfläche lief nicht (kein Playwright).
- Offen: 2160 Bilder belegen bei Full HD etwa 1 GB temporär je Seite.

## 0.34.0

Balltracking-Simulation: stimmiger Fortschritt, Abbruch mit Wahl, auffällige Szenen statt „alle Lücken".

- Fortschritt: Der Lauf meldet drei Schritte über `stage` (Bilder extrahieren, Modell laden, Ball suchen) mit Zähler. `ml_worker simulate-ball-tracking` meldet nach dem Laden des Modells `RECO_PROGRESS 0 N`; vorher stand der Balken während des Ladens bei 40 %. Anteile jetzt 30 % Extraktion, 5 % Modell, 65 % Erkennung.
- Abbruch: `/api/simulation-cancel` mit `keep`. Die Erkennung schreibt jedes fertige Bild zusätzlich als JSON-Zeile nach `detections.jsonl` (`--partial`). Mit `keep` wird daraus eine kürzere Simulation (`partial: {done, total}`), die übrigen extrahierten Bilder werden gelöscht; ohne `keep` wird der Ordner des Laufs entfernt. Der Auftragsdialog stellt die Frage statt der allgemeinen Rückfrage; „behalten" gibt es ab Schritt 3.
- Behoben: Ein abgebrochener oder fehlgeschlagener Lauf ließ die vorige Simulation stehen, deren Bilder beim Start schon gelöscht waren. Jetzt wird die Simulation des Slots beim Start geleert. Außerdem trägt eine Simulation den Projektordner (`root`) und wird in einem anderen Projekt nicht mehr angezeigt.
- Szenen: `teachers.tracking_scenes` findet in den Erkennungen eines Clips vier Arten: `gap` (Ball fehlt bis 1,5 s zwischen zwei nahen Erkennungen), `jump` (Abstand größer als 12 % der Bildbreite plus 60 % je Sekunde; Sprünge innerhalb einer halben Sekunde sind eine Szene), `weak` (mindestens drei Bilder in Folge unter 0,4 Sicherheit), `lost` (länger als 1,5 s kein Ball, auch am Anfang und Ende). Je Szene höchstens drei Bilder, höchstens 40 Szenen.
- Oberfläche: Leiste über die Länge des Clips, Liste zum Anspringen, Auswahl je Szene, „vorige/nächste Szene". Lücken und Sprünge sind vorab angekreuzt.
- Übernahme: `/api/simulation-gaps` nimmt `scenes` (IDs). Lücken werden wie bisher im Ausschnitt nachgesucht; bei Sprung und Unsicher kommt die vom Modell gemeldete Box als Vorschlag mit (die Erkennung liefert dafür jetzt Breite und Höhe); Verloren kommt leer. Herkunft `ball-gap`, `ball-jump`, `ball-weak`, `ball-lost`. Ohne `scenes` gilt das alte Verhalten (alle Lücken). Höchstens 60 Bilder je Übernahme.
- Geprüft mit einem echten Modell (RF-DETR Small, Basketball) an einem 20-Sekunden-Ausschnitt: 240 Bilder, 19 Szenen (9 Sprünge, 5 Lücken, 5 unsichere); Abbruch mit Behalten (56 von 240) und mit Löschen; Übernahme von 29 Bildern aus 13 Szenen, in allen 6 Lücken-Bildern fand die Nachsuche den Ball.
- Offen: Die Simulation liegt nur im Arbeitsspeicher und ist nach einem Neustart weg. Bei Sprüngen ist die mitgegebene Box meist der Fehlfund und muss gelöscht werden. Schwelle 0,25 und 300 Bilder sind fest.

## 0.33.0

Zwei Schritte auf dem Weg zu besserem Balltracking in der Reco-App: die Bilder finden, an denen das Modell scheitert, und das Modell so exportieren, dass Recos Standardschwelle passt.

- Lückensuche in der Balltracking-Simulation: Knopf „Lücken als Prüfkandidaten übernehmen“ (`app/BallTrackingPlayer.tsx`, `/api/simulation-gaps`, `ml_worker recover-ball-gaps`, `teachers.ball_gaps`).
  - Eine Lücke sind Bilder ohne Ball zwischen zwei Erkennungen, höchstens 1,5 s lang und mit einem Sprung von höchstens einem Viertel der Bildbreite. Je Lücke bis zu drei Bilder, je Lauf höchstens 60.
  - Um die erwartete Position wird ein Ausschnitt (mindestens 384 px) noch einmal mit demselben Modell durchsucht (Schwelle 0,15). Ein Fund wird als automatischer Vorschlag eingetragen.
  - Die Bilder landen als `frames/gap-<Video>-<ms>.jpg` im Projekt, Status Prüfkandidat, Herkunft `ball-gap`. Doppelte werden über eine feste ID je Video und Zeitpunkt übersprungen, vorher wird das Projekt gesichert.
  - Videos, die Bilder im unabhängigen Test haben, werden abgelehnt, damit der Test unabhängig bleibt.
  - Der Einleitungstext der Simulation sagt jetzt, dass Bilder nur über diesen Knopf ins Projekt kommen.
- Reco-Export, Sicherheiten (nur RF-DETR): Reco nimmt jeden Fund ab 0,10. Nach dem Schreiben läuft die Datei wie in Reco auf bis zu 200 Validierungsbildern; je Klasse wird die beste Schwelle bestimmt (F1, bei Ball und Puck F2, also Trefferquote doppelt gewichtet) und die Sicherheit stückweise linear so umgerechnet, dass diese Schwelle bei 0,10 liegt. Die Reihenfolge der Funde bleibt gleich. Klassen mit weniger als zehn markierten Objekten bleiben unverändert. Die Schwellen stehen in den Metadaten (`reco_trainer_thresholds`, `reco_trainer_calibrated_to`). `--raw-scores` auf der Kommandozeile schaltet das ab; in der Oberfläche gibt es dafür keinen Schalter.
- Reco-Export, Seitenverhältnis: Es zählt das häufigste Bildformat des Projekts statt des Medians der ersten 40 Bilder. Haben mehr als 10 % der Bilder ein anderes Format, warnt der Export.
- Gemessen an 128 Kamerabildern eines Fußballprojekts bei Recos Schwelle 0,10 (markiert: 14,5 Personen je Bild): RF-DETR Small vorher 86 % der Bälle, 0,47 falsche Bälle und 23,6 Personen je Bild; umgerechnet 75 %, 0,05 und 12,0. Nano vorher 84 %, 0,61 und 32,4; umgerechnet 61 %, 0,08 und 11,5.
- Grenzen: Die Schwellen stammen aus Validierungsbildern des eigenen Projekts und sind für fremde Hallen eher zu streng; wer mehr Bälle will, senkt Recos Schwelle. Beides ist nicht in der Reco-App selbst erprobt, die Lückensuche nur mit Tests und nachgestellten Daten, nicht mit einem echten Video.

## 0.32.0

Nach einem Forumsbericht: Ein in Reco Trainer trainiertes und als ONNX exportiertes Modell (kombiniert aus Ball, Spieler, Tor) erkannte in der Reco-App nichts.

- Ursache 1: Reco (Stitcher/Autocam, `reco-detect`) liest jeden Detektor als YOLO-End-to-End-Export: quadratischer Eingang mit Letterbox (Grau 114, /255), genau eine Ausgabe `[1, N, 6]` (x1, y1, x2, y2 in Eingangspixeln, Sicherheit, Klasse) und Klassennamen als Python-Dict in den Metadaten `names`. Bei mehr als einer Ausgabe nimmt Reco den Decoder für einen externen Balldetektor. Der bisherige Export „CPU-Modell (ONNX)“ schreibt RF-DETRs eigenes Format mit zwei Ausgaben (`dets`, `labels`) ohne `names`, Reco deutet ihn also falsch.
- Ursache 2: Dieser Export nimmt per `newest_checkpoint()` nur eine einzelne Gewichtsdatei der Modellgröße. Ein kombiniertes Modell hat keine einzelne Datei; exportiert wurde still der letzte Trainingslauf dieser Größe.
- Neu: „Für Reco exportieren“ (`app/RecoExportDialog.tsx`, `/api/export-reco`, `ml_worker export-reco`, Modul `reco_export.py`). Modell aus der Bibliothek wählen, Klassen ankreuzen (voreingestellt nur Ball), Datei nach `exports/reco/`.
  - RF-DETR (auch kombiniert): ein Torch-Modul schneidet die grauen Ränder ab (Seitenverhältnis = Median der Projektbilder), staucht auf die Modellauflösung wie im Training, normalisiert, rechnet alle Mitglieder, nimmt je Query die beste gewählte Klasse und gibt die 300 besten Treffer als `[1, 300, 6]` aus. Eingang 960 px (bis Auflösung 640), sonst 1280 px. Export mit `torch.onnx.export` (Opset 17, ohne Dynamo).
  - YOLO26 (`onnx-yolo`): an die vorhandene Ausgabe werden ONNX-Knoten gehängt, die Klassen umnummerieren und nicht gewählte Klassen auf Sicherheit 0 setzen.
  - `player` heißt in der Datei `person`, weil Reco Spieler unter diesem Namen sucht.
  - Danach rechnet ein Probelauf die Datei wie Reco (Letterbox, ONNX Runtime CPU) auf bis zu acht markierten Projektbildern und meldet, auf wie vielen die markierten Objekte ab 25 % gefunden wurden.
- Geprüft auf echten Basketball-Modellen (RF-DETR Nano einzeln und kombiniert, Ball/Spieler): Boxen fast pixelgleich mit `predict()`, Sicherheiten weichen wegen Letterbox gegenüber Stauchung leicht ab. Klassenfilter an YOLO26n (COCO) geprüft. In der Reco-App selbst nicht getestet, nur gegen deren Quellcode (v0.5.4).
- Hinweis am alten ONNX-Export: nicht für die Reco-App.

## 0.31.8

Nach dem Bericht eines Testers, der Videos auf eine externe Platte verschoben und dabei einige hundert markierte Bilder verloren hatte (über den Ordner `dataset` hat er sie wiederhergestellt).

- Behoben: „Frames sicher aktualisieren“ (`extract_project`) baute `frames/` nur aus den Videos neu, die gerade im Ordner lagen. Bilder und Markierungen aller anderen Videos gingen verloren, ebenso Kandidaten aus „Datensatz erweitern“ und dem unabhängigen Test. Nur importierte Bilder blieben. `carry_over_frames` übernimmt jetzt jedes frühere Bild, das dieser Lauf nicht neu extrahiert (per Hardlink, sonst Kopie). `reusable_video_frames` übernimmt Videos mit gleicher Bildzahl, deren Bilder noch vollständig da sind, statt sie neu zu extrahieren. Die Meldung nennt übernommene Videos und Videos, die nicht mehr im Ordner liegen.
- Neu: `dataset_split.py` (nur Standardbibliothek) enthält `trains_on`, `split_frames` und `preview`. `ml_worker` und `local_worker` nutzen dasselbe Modul. Der Status enthält `splits` (Rolle pro Bild, Anzahlen, Warnung, Probleme); im Studio-Modus ist das `null`. In der Oberfläche (`app/SplitView.tsx`) filtern Chips über der Bildliste nach Training / Validierung / Test / „Nicht im Training“, jedes Bild hat ein Rollen-Tag, und der Trainingsdialog zeigt die Aufteilung samt Problemen (keine Testbilder, weniger als max(10, 5 %) Validierungsbilder).
- Neu: `import-labels` erkennt Datensätze aus Reco Trainer (`metadata.json` mit `generator: "Reco Trainer"` oder COCO-`info.description` „Local Reco Trainer dataset“) und übernimmt die Rolle aus den Ordnern train/valid/val/test als `frame.split`. `split_frames` legt solche importierten Bilder direkt in ihre Rolle. Ein erneuter Import setzt fehlende Rollen bei schon importierten Bildern (`rolesSet`). Andere Datensätze bleiben Trainingsbilder.
- Tests: Die Batch-Tests hingen vom freien Platz der Testmaschine ab (Batch überspringt Modelle unter 5 GB frei); `disk_usage` ist jetzt gemockt.

## 0.31.7

- Behoben: Das Batch-Training auf der AMD-Karte blieb auch mit 0.31.6 nach „Confidence prüfen: 53/53“ von Medium stehen (Tester, RX 7800 XT). Das harte Beenden in 0.31.6 hat also nicht gereicht. Wahrscheinlich hält unter Windows ein vom Prüfschritt gestarteter Prozess die Ausgabe-Pipe offen, und `run_batch_step` las mit `for line in process.stdout` bis EOF. `run_batch_step` liest jetzt auf einem Thread. Ist der Prozess beendet, wartet es höchstens 10 s auf das Ende der Pipe. Hat ein Schritt sein Ergebnis (JSON) geliefert und endet nicht, beendet es ihn nach 60 s samt Unterprozessen (`taskkill /T /F`). Die optionale Confidence-Prüfung wird nach 600 s ohne Ausgabe abgebrochen und als `sweepError` vermerkt.
- Der Sweep schreibt vor seinem Ergebnis „Beste Confidence: x (F1 y)“ ins Protokoll, damit man sieht, wie weit er gekommen ist.

## 0.31.6

- Behoben (vermutlich): Beim Tester (RX 7800 XT, 0.31.5) blieb ein Batch aus RF-DETR Medium und YOLO26 Medium nach Medium stehen. Angezeigt wurde zuletzt „Comprobar la confianza: 53/53“, 99 %, YOLO begann nie, der Batch musste abgebrochen werden. Nach dem letzten Bild rechnet der Sweep nur noch kurz und gibt sein Ergebnis aus, also endete entweder dieser Prozess nicht, oder YOLO hing vor seiner ersten Meldung. `ml_worker.main` beendet Prozesse der ROCm-Umgebung unter Windows (`ROCM_SDK_TARGET_FAMILY=custom`) nach getaner Arbeit jetzt mit `os._exit(0)`. Ausgaben werden vorher geleert, alle Dateien sind dann schon atomar geschrieben, und der Treiber wird nicht mehr aufgeräumt.
- `train-batch` meldet jetzt vor jedem Modell „<Modell>: wird gestartet …“ und setzt den Fortschritt zurück. Bleibt ein Schritt hängen, bevor er sich selbst meldet, sieht man das jetzt.

## 0.31.5

- Behoben: Jeder Prozess in der ROCm-Umgebung, auch jeder Datenlader, schrieb zwei Zeilen „Unknown command line argument … offload-arch.exe“ ins Protokoll. AMDs `rocm_sdk` (`_dist_info.discover_current_target_family`) ruft `offload-arch` über den Starter im `Scripts`-Ordner auf, und der reicht den Pfad unter Windows ohne Anführungszeichen weiter. Mit Leerzeichen im Pfad scheitert das immer (`Reco Trainer`, beim Tester auch der Benutzername). Danach nahm `rocm_sdk` ohnehin seine Vorgabe `custom`, die einzige Zielfamilie in AMDs Windows-Paketen. `ROCM_TRAINING_ENV` und `ROCM_STEP_ENV` setzen jetzt `ROCM_SDK_TARGET_FAMILY=custom`, ebenso die Prüfungen nach der Einrichtung und vor YOLO. Bestätigt vom Tester: 0.31.4 trainiert RF-DETR und YOLO auf der RX 7800 XT.

## 0.31.4

- Neu: YOLO trainiert auf der AMD-Karte (Windows, ROCm). Bei einem Batch mit YOLO installiert `local_worker.ensure_yolo_rocm` Ultralytics in derselben gepinnten Version zusätzlich in die ROCm-Umgebung: `--no-deps`, die übrigen Abhängigkeiten unter `rocm-env/constraints.txt`, damit pip AMDs torch behält, und `opencv-python-headless` nur, wenn `cv2` fehlt. Danach läuft der ganze Batch mit dem Python der ROCm-Umgebung, wie ein Batch ohne YOLO. `ml_worker` erkennt die Karte als `cuda`, YOLO trainiert dann mit AMP und der Batchgröße für die Grafikkarte.
- Schlägt die Installation oder die Importprüfung fehl, bleibt es beim Weg aus 0.31.3: RF-DETR auf der Karte, YOLO auf der CPU, mit einem Hinweis im Protokoll. Auf echter AMD-Hardware noch nicht geprüft.

## 0.31.3

- Behoben: Ein Batch-Training mit RF-DETR und YOLO lief komplett in der normalen ML-Umgebung, unter Windows mit AMD-Karte also auf der CPU, obwohl „Auf der AMD-Karte trainieren“ an war. Gemeldet von einem Tester mit RX 7800 XT (RF-DETR Medium + YOLO26 Medium): rund 4 Minuten pro Epoche, CPU 90 %, „Compute“ der Karte 0 %, 0,8 GB Grafikspeicher. `local_worker` gibt dem Batch jetzt `--rfdetr-python` mit; `ml_worker train-batch` startet die RF-DETR-Schritte mit dem Python der ROCm-Umgebung und `RECO_ONNX_INFERENCE=off`, YOLO und der Konfidenz-Sweep bleiben in der normalen Umgebung.

## 0.31.2

- Behoben: Das Augmentation Studio machte bei Datensätzen aus „Datensatz exportieren“ aus jedem Bild eine eigene Session. Der Export benennt Bilder absichtlich `frame-<id>.jpg`, ohne Videonamen, und `session_hint` behielt die Id. Gemeldet mit „Basketball Cleaned“: 4.900 Bilder ergaben 4.900 Sessions, keine Hallenplatte aus mehreren Bildern, Aufteilung nur nach den Export-Ordnern. Jetzt fallen reine Ids weg (auch Roboflows `_jpg.rf.<hash>`), und Bildgröße und Hintergrund entscheiden. Auf demselben Datensatz ergibt das rund 1.500 Sessions; 30 davon mit mindestens 20 Bildern decken zusammen rund 1.850 Bilder ab.
- Dabei aufgefallen: Im selben Export lagen Bilder derselben Halle und Kamera in Training, Validierung und Test, weil das Projekt aus einem importierten Datensatz ohne Videonamen stammt. Das Studio teilt solche Hallen jetzt als Ganzes neu auf.
- Oberfläche: Das Studio zeigt höchstens 12 Sessions vollständig (die größten, ab 5 Bildern). Die übrigen stehen in einer zugeklappten Liste, die erst beim Aufklappen gerendert wird. Vorher wären bei diesem Datensatz rund 30.000 Vorschaubilder auf einer Seite gelandet.

## 0.31.1

- Neu: Schritt-für-Schritt-Anleitung im Augmentation Studio (`app/AugmentGuide.tsx`), sieben Schritte in Deutsch, Englisch, Spanisch und Französisch: was das Studio macht und was es braucht (mindestens 300, besser 500 geprüfte Bilder aus mehreren Aufnahmen), Quellordner und Rollen, Einstellungen (1.500–2.000 Variationen für einen ersten Test), Analyse und Hallen, Teams, Erzeugen, Training und Vergleich mit „Nur echt, gleiche Aufteilung“.
- Sie öffnet sich bei jedem Start des Studios, bis „Nicht mehr anzeigen“ angehakt wird (`localStorage` `reco-augment-guide-hidden-v1`). Der Knopf „Anleitung“ im Studio öffnet sie wieder; dort lässt sich der Haken auch wieder entfernen. Pfeiltasten blättern, Esc schließt.
- Die Bildschirmbilder (`public/augment-guide/<sprache>-<schritt>.jpg`) zeigen das echte Studio in der jeweiligen Sprache. `tests/augment-guide-shots.mjs` erzeugt sie aus dem Build gegen einen nachgebildeten Worker; alle Hallen, Spieler und Vorschauen darin sind als SVG gezeichnet, es stecken keine echten Trainingsbilder darin.
- Tests: `tests/test_augment_guide.py` prüft, dass alle 28 Bilder da und klein sind und jede Sprache sieben Schritte hat; `tests/editor-ui.mjs` prüft Öffnen, Ausblenden, Wiederholen und das Laden der Bilder.

## 0.31.0

Training auf AMD-Grafikkarten unter Windows (ROCm), nach dem erfolgreichen Test auf einer RX 7800 XT (eine Epoche 16,4 s auf der Karte statt 711 s auf der CPU).

- Neu: `amd_training.py` (nur Standardbibliothek). Eigene Umgebung `<App-Daten>/rocm-env/venv` mit Python 3.12, AMDs ROCm 7.2.1 mit torch 2.9.1 von repo.radeon.com und `rfdetr[train,onnx]` in derselben Version wie die normale ML-Umgebung. Ein Constraints-File hält AMDs torch fest. Nach der Installation prüft ein kleiner Test die Karte (Matrixprodukt gegen CPU, Faltung mit Rückwärtsschritt). Erst wenn er besteht, wird die Umgebung genutzt. Der Zustand steht in `state.json` (`verified`, `enabled`, `problem`).
- Angeboten nur unter Windows x64 ohne NVIDIA und mit einer Karte von AMDs Liste: gfx1100/1101 (RX 7700–7900, PRO W7700–W7900) und gfx1200/1201 (RX 9060/9070, AI PRO R9700).
- `local_worker`: `/api/rocm-setup` (Job `rocm-setup`), `/api/rocm-enabled`, `rocm` im Status. „Lokal trainieren“ (`train`) und Batch-Training ohne YOLO starten `ml_worker` mit dem Python der ROCm-Umgebung und `RECO_ONNX_INFERENCE=off`. Die Karte erscheint in PyTorch als `cuda`, `ml_worker` bleibt unverändert. Ein Absturz dort startet keinen DirectML-Bericht, sondern nennt den Schalter zum Zurückwechseln auf die CPU. Die Hardware-Zeile zeigt `rocm-directml`.
- Oberfläche: „AMD-Training einrichten“ (mit Rückfrage, braucht „ML einrichten“) und danach der Schalter „Auf der AMD-Karte trainieren“.
- Behoben, bevor es auslieferbar war: Das neue Modul fehlte in den Dateilisten von `build-payload.sh` und `Dockerfile`. Neuer Test `PackagedModulesTests` prüft jetzt jedes eigene Modul, das die Worker importieren.

## 0.30.6

- Behoben: `tools/amd_training_probe.py` brach auf Windows nach jeder Trainingsepoche ab. RF-DETR 1.11.2 gibt seine Metrik-Tabellen mit Rahmenzeichen aus, und die Konsole mit Codepage cp1252 kann sie nicht kodieren (`UnicodeEncodeError`). Jetzt läuft der Bench-Prozess mit `PYTHONIOENCODING=utf-8`/`PYTHONUTF8=1` und stellt stdout/stderr selbst auf UTF-8 um. Bei einem Fehler wird `secondsUntilError` mitgeschrieben. Auf dem Mac mit `PYTHONIOENCODING=cp1252` nachgestellt und geprüft. Die App selbst war nicht betroffen, sie startet ihre Worker schon mit UTF-8.
- Erster Bericht von einer RX 7800 XT (gfx1101, Adrenalin 32.0.32015.7029): ROCm 7.2.1 mit torch 2.9.1 erkennt die Karte, Matrixmultiplikation stimmt, RF-DETR Nano trainiert eine Epoche auf der Karte bis zur Validierung.

## 0.30.5

- Behoben: Bilder ohne Boxen gingen immer ins Training, auch noch nicht markierte Videobilder. Das brachte dem Modell bei, dass Spieler und Ball darauf Hintergrund sind. Neu ist `ml_worker.trains_on`: Ein Bild ohne Boxen zählt nur mit `emptyConfirmed` (Knopf „Bild ist leer“ im Editor) oder `reviewStatus: reviewed` („kein Ball“ im Kandidaten-Check, importierte Labeldateien). Das gilt für `build_dataset`, den Datensatz-Export, die Split-Statistik nach dem Training und den Schwellen-Sweep. Das Training meldet, wie viele Bilder ausgelassen wurden, die Übersicht zeigt `unconfirmedEmpty`. Wer Boxen speichert, setzt `emptyConfirmed` zurück.
- Augmentation Studio: Die Rolle „Statisch“ heißt jetzt „Bleibt an seinem Platz“ (DE/EN/ES/FR).
- Hinweis: Weil ausgelassene Bilder einen Video-Split verändern können, kann sich die Aufteilung in Training, Validierung und Test gegenüber 0.30.4 verschieben.

## 0.30.4

- Behoben: „Quellordner wählen“ im Augmentation Studio öffnete ohne vorbereiteten Arbeitsordner oder ohne ML-Umgebung kein Fenster, nur eine kleine Fehlerzeile oben. Jetzt ist der Knopf dann gesperrt und daneben steht, was fehlt (`needs: project | ml` in `/api/augment/state`). Das Studio lädt seinen Zustand neu, wenn ein anderer Arbeitsordner geöffnet oder ML eingerichtet wird.

## 0.30.3

Nach dem zweiten Bericht von der RX 7800 XT: kein Grafikproblem, sondern zwei Fehler in der Konsistenzprüfung. Dazu ein Test für AMD-Training.

- Behoben: „Vorschläge berechnen“ meldete „Keine markierten Bilder“, obwohl die Prüfliste markierte Bilder zeigte. Die App zählt markierte Bilder des unabhängigen Testsets mit, der Worker ließ sie weg. Jetzt nehmen beide dieselben Bilder (markiert, kein Kandidat, Testbilder eingeschlossen). Der zweite Detektor ist ein untrainiertes COCO-Modell, Testbilder fließen also nicht ins Training.
- Behoben: Hinweise des ML-Workers (`SystemExit` mit Text, z. B. „Keine markierten Bilder“) endeten mit Code 1. Die App zeigte „mit Code 1 beendet“ und startete bei eingeschalteter AMD-/Intel-Grafik einen Diagnosebericht. Jetzt gibt der Worker solche Hinweise als `RECO_USER_ERROR` mit Code 3 aus, die App zeigt den Text und startet keine Diagnose.
- Neu: `tools/Test AMD Training.bat` (Windows). Installiert AMDs ROCm 7.2.1 mit PyTorch 2.9.1 (Python 3.12) und RF-DETR in einen eigenen Ordner (`%LOCALAPPDATA%\Reco Trainer\amd-training-probe`), trainiert RF-DETR Nano eine Epoche auf 64 erzeugten Bildern auf Karte und CPU und legt `amd-training-report.txt` auf den Desktop. Die ML-Umgebung der App bleibt unberührt. Grundlage für AMD-Training in einer späteren Version.

## 0.30.2

Spielfeldgrenze entfernen (Wunsch eines Testers).

- Neu: „Spielfeldgrenze entfernen“ im Dialog „Spielfeld bearbeiten“, sobald eine Grenze gespeichert ist. Löscht die Grenze dieses Videos oder, mit Häkchen „Für alle Videos übernehmen“, die aller Videos (auch die Vorgabe für neue Videos). Der erste Klick färbt den Knopf rot („Wirklich entfernen?“), erst der zweite löscht. Vorher wird `project.json` in `backups/` gesichert. Automatisch markieren filtert für diese Videos danach nicht mehr nach Spielfeld.
- Behoben: Unter Windows zeigte die Hardware-Zeile „? GB“, weil es dort kein `os.sysconf` gibt. Der Arbeitsspeicher wird jetzt über `GlobalMemoryStatusEx` gelesen, auch für die Wahl der Trainings-Batchgröße.
- Geändert: Die Hardware-Zeile nennt die DirectML-Erkennung nicht mehr „experimentell“ (bestätigt auf einer RX 7800 XT).

## 0.30.1

Kleines Bugfix-Release nach dem ersten Bericht von einer AMD-Karte.

- Bestätigt: Erkennung auf AMD-Grafik unter Windows. Auf einer Radeon RX 7800 XT (Windows 11, 16 GB) läuft RF-DETR Nano per DirectML in 6,7 ms statt 110,7 ms auf der CPU (etwa 16-mal schneller), größte Box-Abweichung 0,00018. Vulkan sieht die Karte ebenfalls. Training läuft auf AMD weiterhin auf der CPU.
- Behoben: Der Diagnosebericht warnte, wenn dxdiag „Driver Problem Code: Unknown“ (bzw. „Unbekannt“) meldet. Das heißt nur, dass der Treiber keinen Code liefert, und gilt jetzt als in Ordnung.
- Behoben: Unter Windows und Linux stand im Bericht „recoTrainer: unknown“. Die Pakete enthalten jetzt eine `version.json`; das Paketskript bricht ab, wenn seine Version nicht zu `package.json` passt.
- Behoben: Nach einem YOLO-Training blieben leere Ordner `runs/detect/val*` im Arbeitsordner zurück (auch in den 0.30.0-Paketen für Windows/Linux/Docker). Die Auswertung landet jetzt im Trainingslauf (`runs/<Größe>/eval-val`, `eval-test`), und das Paketskript lässt `runs/` aus.

## 0.30.0

Neuer Look und optionales YOLO-Training.

- Neu: Die Oberfläche sieht aus wie eine Mac-App, hell und dunkel. Systemschrift in lesbaren Größen, Milchglas-Kopfzeile und -Seitenleiste, macOS-Knöpfe und Segmentschalter, ruhige Karten, Dialoge als Sheets (kurze Einblendung, abgeschaltet bei „Bewegung reduzieren“), Hinweise und Job-Anzeige als Glas-HUD. Das Bild steht auf einer dunklen Graphit-Bühne mit Glasleiste, die in jeder Sprache einzeilig bleibt. Ballorange markiert App-Symbol, Prüfkandidaten und den Ball. Der Dunkelmodus folgt der Systemeinstellung. Aufbau, Knöpfe und Abläufe sind unverändert.
- Neu: YOLO trainieren (optional). „Lokal trainieren“ bietet neben RF-DETR jetzt YOLO26 Nano, Small und Medium (Ultralytics), auch beide Familien in einem Vergleichslauf. Ultralytics steht unter AGPL-3.0 und wird erst nach Zustimmung im Dialog in die ML-Umgebung installiert (Version 8.4.180 fest, ohne zweites OpenCV-Paket); die Basisgewichte sind per SHA-256 festgelegt und nicht Teil der Apache/MIT-Modellliste. Nutzungsstatistik und automatische Paketinstallationen von Ultralytics sind abgeschaltet. Bildgröße 640, 960 oder 1280 px wählbar. Das trainierte Modell wird nach ONNX (NMS eingebaut) exportiert und als Bibliotheksmodell abgelegt; es markiert automatisch, läuft im Modelltest und in der Balltracking-Simulation, und der nächste YOLO-Lauf derselben Größe setzt darauf auf. Core ML, kombinierte Modelle und `.recomodel`-Teilen bleiben RF-DETR vorbehalten. Batch-Training und Confidence-Ermittlung kennen YOLO.

## 0.29.0

Annotieren von sehr kleinen Bällen (viel Zoomen und Verschieben) und eine Diagnose für AMD-/Intel-Grafik.

- Neu: Das Bildfenster nutzt auf Laptops fast die ganze Höhe (1366×768: 537 statt 204 px hoch, 1440×900: 669 statt 277 px). Seitenleiste und Kopfzeile drücken es nicht mehr zusammen; der Trainingsbereich liegt darunter, die Seite scrollt. Auch für Monitore im Hochformat angepasst.
- Neu: Tastenkürzel. Leertaste gedrückt halten = verschieben, beim Loslassen ist das vorherige Werkzeug (z. B. „Neue Box“) wieder aktiv. B = Neue Box, 1–9 = Klasse 1–9 in der Reihenfolge der Klassenliste, ←/→ wie bisher = Bild wechseln. Kein Kürzel löst aus, solange ein Eingabefeld, eine Liste oder ein Dialog den Fokus hat. Die Kürzel stehen an den Werkzeug-Knöpfen, in der Klassenliste („Ball (1)“), in den Hinweisen und unter dem Bild.
- Neu: Das Mausrad zoomt zu dem Punkt unter dem Mauszeiger statt zur Bildmitte; der Bildpunkt unter dem Cursor bleibt stehen. Das gilt in jedem Werkzeug, auch während „Neue Box“ aktiv ist. Das Mausrad scrollt dabei nicht mehr die Seite.
- Neu: Maximaler Zoom 20× statt 5×, in gleichmäßigen, multiplikativen Schritten (drei Klicks verdoppeln). Ab etwa 3× werden die echten Bildpunkte gezeigt.
- Neu: Box-Rahmen (1,5 px, ausgewählt 2,5 px), Anfasser und Beschriftungen haben bei jedem Zoom dieselbe Größe auf dem Bildschirm. Bild und Boxen liegen dafür in einem SVG mit Ausschnitt (viewBox) statt in einer per CSS skalierten Ebene. Die gespeicherten Box-Koordinaten ändern sich nicht; Tests belegen, dass dieselbe Box bei jedem Zoom dieselben Koordinaten bekommt.
- Neu: Spielfeldgrenze pro Video bzw. Kameraposition. Der Editor nennt das Video und kann die Grenze „für alle Videos übernehmen“. Bestehende Projekte übernehmen die bisherige globale Grenze beim Öffnen automatisch für alle ihre Videos (vorher Sicherung in `backups/`, der alte Wert bleibt als `fieldGeometryBeforePerVideo` erhalten). Automatisch markieren prüft jedes Bild gegen die Grenze seines Videos. Geprüft an einer Kopie eines echten Projekts (2493 Bilder, 41 Videos): Bilder unverändert, 49.860 Filterentscheidungen vorher und nachher identisch. Die Grenze wird weiterhin nur beim automatischen Markieren genutzt.
- Neu: „Modelle testen“ sagt, warum der Vergleich noch nicht geht (keine Bilder, ungeprüfte Auto-Vorschläge, noch kein bestätigtes Testset, noch kein trainiertes Modell für die Sportart, anderer Vorgang läuft) und was als Nächstes zu tun ist. Deaktivierte Knöpfe nennen den Grund als Tooltip.
- Neu: AMD-/Intel-Grafik (Windows, experimentell): Ohne NVIDIA-GPU fragt Reco Trainer, ob die Grafikkarte versucht werden soll; bis zum Ja läuft alles auf der CPU. Nach dem Ja installiert ein Test `onnxruntime-directml` und prüft die Karte. Scheitert sie (Rückfall auf die CPU oder ein abgestürzter Job), öffnet sich ein Diagnosebericht (System, Treiber, DirectX, Pakete, DirectML-Minimaltest, RF-DETR auf Karte und CPU, Vulkan) zum Kopieren oder Speichern.
- Neu: „Was ist neu“ erscheint nach einem Update einmal; ein Klick auf die Versionsnummer öffnet es wieder.
- Behoben: „Der lokale ML-Worker wurde mit Code 1 beendet“ nennt jetzt die letzte Fehlermeldung des Workers (z. B. eine fehlende DLL).

## 0.28.0

AMD- und Intel-Grafik sind **experimentell**: noch nicht auf echter AMD-/Intel-Hardware getestet; bei Problemen läuft alles wie bisher auf der CPU.

- Neu: AMD- und Intel-Grafikkarten unter Windows beschleunigen die Erkennung (Vorlabeln, zweiter Detektor der Konsistenzprüfung, Benchmark, Balltracking-Simulation). RF-DETR läuft dort über ONNX Runtime mit DirectML; „ML einrichten“ installiert `onnxruntime-directml`, wenn keine NVIDIA-GPU vorhanden ist. Das Modell wird je Gewichtsdatei einmal nach ONNX exportiert und bei den gemeinsamen Modellen abgelegt. Ergebnisse sind mit PyTorch identisch (gleiche Boxen, Klassen und Scores, Abweichung unter 0,05 px). Fällt die Grafikkarte aus, läuft die Erkennung mit PyTorch auf der CPU weiter. Die Hardwarezeile zeigt dann „Training auf der CPU · Erkennung auf der Grafikkarte (DirectML)“. Abschaltbar mit `RECO_ONNX_INFERENCE=off`.
- Neu: Die VLM der Konsistenzprüfung nutzt unter Windows und Linux (x64) ohne NVIDIA-GPU den Vulkan-Build von llama.cpp (b11985, MIT) und damit AMD- und Intel-Grafik. Ohne Vulkan-Treiber läuft er wie bisher auf der CPU; scheitert der Start auf der Grafikkarte, startet der Server einmal neu auf der CPU. `RECO_VLM_RUNTIME=cpu|vulkan|cuda` legt den Build fest.
- Neu: Die Hardware-Stufe erkennt AMD- und Intel-Grafikkarten unter Windows (aus der Registry, ohne PyTorch). Ab 8 GB eigenem Grafikspeicher gilt der Rechner als „Mittel“ (RF-DETR Medium als Vorschlagsmodell, VLM wählbar), integrierte Grafik bleibt „Schwach“. Die Hardwarezeile nennt die Karte, das Job-Log Provider, Karte und Speicher.

## 0.27.0

- Neu: Ergebnisse der Konsistenzprüfung Bild für Bild durchgehen. Über der Bildliste startet „markierte Bilder prüfen“ (dazu „Nur markierte zeigen“), im Ergebnisfenster „Der Reihe nach prüfen“. Unter dem Bild stehen Grund, VLM-Kommentar und ein konkreter Vorschlag; im Bild sind messbar doppelte Boxen rot und vom zweiten Detektor gefundene, noch fehlende Boxen grün gestrichelt. Pro Bild: Vorschlag übernehmen, nachbessern, ablehnen oder Bild löschen. Meldet die VLM doppelte Boxen, die sich nicht messen lassen, weist die App auf einen vermutlichen Fehlalarm hin.
- Neu: Die Konsistenzprüfung speichert, welche Boxen der zweite Detektor ergänzen würde. Für frühere Prüfungen holt „Vorschläge berechnen“ das nach, ohne Markierungen zu ändern.
- Neu: Balltracking-Simulation mit zwei Modellen vergleichen. Zwei Seiten A und B mit eigenem Video und eigenem Modell (aktives Modell oder jedes Bibliotheksmodell mit Ball-Klasse, auch ONNX-YOLO), gemeinsame Zeitleiste, Trefferstatistik je Seite; „Gleiches Video wie A“ vergleicht beide Modelle auf denselben Bildern. Das aktive Modell bleibt dabei unverändert.
- Behoben: Die Balltracking-Simulation nutzte immer die Größe „nano“, auch wenn ein anderes Modell eingestellt war.

## 0.26.0

- Neu: „Datensatz importieren“ direkt unter „Arbeitsordner auswählen“. Ein Ordner mit Bildern und COCO- oder YOLO-Labels lässt sich übernehmen, sobald ein Arbeitsordner gewählt ist, auch wenn dort noch nichts vorbereitet wurde. Das Projekt wird dann mit dem Datensatz angelegt; die Bilder werden kopiert, das Original bleibt unverändert.
- Neu: Die ML-Umgebung wird einmal pro Rechner eingerichtet und in jedem Arbeitsordner wiederverwendet. Eine ML-Umgebung, die ein Projekt aus einer früheren Version schon hat, wird weiter genutzt und für neue Arbeitsordner gemerkt; „ML einrichten“ ist dafür nicht noch einmal nötig. In Docker bleibt die Umgebung im Projektordner.
- Behoben: Der Hinweis „Dieser Arbeitsordner ist noch leer“ verschwindet, sobald Bilder im Projekt sind.

## 0.25.2

- Neu: Training und Modellverwaltung unter dem Labelbild lassen sich einklappen. Der frei werdende Platz vergrößert die Vorschau; die Einstellung bleibt gespeichert. Der aufgeklappte Bereich ist bei wenig Platz scrollbar.
- Neu: Eine vorhandene Box anklicken und oben ihre Klasse ändern, zum Beispiel Spieler zu Schiedsrichter. Die Änderung wird als manuelle Korrektur gespeichert, entfernt den alten Confidence-Wert und unterstützt Rückgängig/Wiederholen. Reines Auswählen erzeugt keinen Speichervorgang und keinen Undo-Schritt.

- Behoben: „ML einrichten“ prüft jetzt, ob sich PyTorch wirklich laden lässt. Unter Windows mit NVIDIA-Karte konnte die CUDA-Version sauber installiert werden und trotzdem nicht laden (`WinError 127` bei `c10_cuda.dll`, zum Beispiel durch einen zu alten NVIDIA-Treiber oder fehlende Visual-C++-Laufzeit); das Training brach dann mit einem unverständlichen Fehler ab. Jetzt wechselt „ML einrichten“ in diesem Fall automatisch zur CPU-Version von PyTorch. Wer betroffen ist, klickt erneut auf „ML einrichten“; die Umgebung wird repariert, Videos, Labels und Modelle bleiben unberührt. Meldungen beim Training erklären den Fehler jetzt.
- Behoben: Der nach dem Training gespeicherte Qualitätswert wird jetzt auf dem besten Checkpoint gemessen, nicht auf der letzten Epoche. Ein früh optimales Modell wurde zuvor zu schlecht bewertet, auch bei der automatischen Wahl des besten Modells.

## 0.25.1

- Model Hub: Das Repository ist online und „Im Hub teilen“ freigeschaltet. Der Hub enthält noch keine Modelle, und über das echte GitHub wurde noch kein Modell geteilt.

## 0.25.0

- Neu: „Lokal trainieren“ fragt jetzt, welche Modelle trainiert werden: das aktuelle oder zusätzlich Nano, Small, Medium und Large. Sie laufen nacheinander, das kleinste zuerst. Scheitert eines, zum Beispiel an zu wenig Speicher, geht es mit dem nächsten weiter.
- Neu: Automatische Epochenzahl: Das Training läuft, bis sich der Wert über eine einstellbare Zahl von Epochen nicht mehr verbessert (Standard 10). Die Lernrate halbiert sich dabei, sobald der Wert stagniert.
- Neu: Nach dem Training wird die beste Confidence je Modell und Klasse auf den Validierungsbildern ermittelt, ohne erneutes Training. Eine Tabelle zeigt alle Modelle mit Empfehlung; die Confidence lässt sich übernehmen.
- Neu: RF-DETR Medium und Large sind trainierbar.
- Der Computer bleibt während des Trainings wach (macOS, Windows, Linux mit systemd).
- Neu: Model Hub – fertige Modelle anderer Nutzer laden und eigene über einen Pull Request teilen. Es werden nur bereinigte Gewichte hochgeladen, nie Bilder. Beim Weitertrainieren misst der Trainer vorher und nachher auf den eigenen Validierungsbildern. Der Hub startet ohne Modelle; „Im Hub teilen“ erscheint erst, wenn der GitHub-Login eingerichtet ist.
- Datenschutz: `.recomodel`-Pakete enthielten die Trainingsargumente des Checkpoints mit lokalen Pfaden, also auch dem Benutzernamen. Neue Pakete enthalten nur noch Gewichte und Klassennamen. Bereits weitergegebene Pakete bleiben davon unberührt.
- Korrektur: Die Confidence-Messung nutzt mit Augmentation Studio dessen Validierungsbilder.
- Studio: Die Aufteilung des Quellordners in train, valid und test wird übernommen; Torhüter verfälschen die Teamerkennung nicht mehr und bleiben auf ihrer Position; Tore behalten ihr Label aus der Box; beim Entfernen von Personen werden auch die Füße unter zu kurzen Boxen erfasst.

## 0.24.0

- Neu: Augmentation Studio – erzeugt aus einem Ordner mit Bildern und COCO-Labels einen synthetischen Trainingsdatensatz: Masken mit SAM 2 (Zwischenspeicher), leere Hallenplatte per Pixel-Median ab 15 Bildern, sonst Inpainting (LaMa, ersatzweise OpenCV), Personen werden mit passender Größe und Neigung neu platziert, Labels kommen aus der sichtbaren Maske. Feste Regeln: höchstens zwei Teams, 5 Spieler je Team, 3 Schiedsrichter, 1 Ball, keine Klone. Ein Prüfer erzeugt verletzende Bilder automatisch neu. Validierung und Test bleiben echt und nach Sessions getrennt. Gleicher Seed ergibt bitgleiche Bilder, ein abgebrochener Lauf lässt sich fortsetzen.
- Studio: Ohne Kalibrierung und bei bewegter Kamera (weniger als 15 Bilder je Session) bleiben Personen nah an ihrer Originalposition, weil der Hallenboden nur dort bekannt ist. Als Ball markierte Objekte, die größer als eine halbe Person sind (z. B. ein Senderlogo), werden ignoriert und protokolliert.
- Neu: Fährt man mit der Maus über eine Taste, erscheint ein kurzer Hinweis in der eingestellten Sprache (DE, EN, ES, FR). Die Taste „Hinweise“ oben neben den Systemen schaltet das ab; die Einstellung bleibt gespeichert.
- Neu: Aus dem Videoordner wird der Arbeitsordner. Videos werden wie bisher zu Frames, Bilder werden zum Beschriften übernommen, und liegen Labels dabei (YOLO oder COCO), werden sie mitübernommen. Ist der Ordner leer, erscheint ein Hinweis, dass Videos, Bilder oder Datensätze hineingehören. Liegen nur Bilder darin, bleiben früher vorbereitete Video-Frames erhalten.
- Korrigiert: Ein Projekt nur aus importierten Bildern wird nicht mehr nach Dateireihenfolge geteilt, sondern gemischt und reproduzierbar. Beim Fußball-Datensatz bestand der Test-Split vorher nur aus v7-Ausschnitten. Der Hinweis nennt die Qualitätswerte jetzt ausdrücklich „nicht unabhängig“, weil gleiche Szenen in Training und Test liegen können.
- Korrigiert: Neben echten Videos dienen importierte Bilder nur noch dem Training; Validierung und Test kommen aus den Videos. Bisher konnte der ganze Import in einem Split landen und ein Test mit Importbildern als unabhängig gelten. Die Splits solcher Projekte ändern sich dadurch, Qualitätswerte sind mit älteren Trainings nicht direkt vergleichbar.

## 0.23.0

- Neu: „Konsistenz prüfen“ – ein zweiter Detektor vergleicht alle geprüften Bilder mit den Labels (Übereinstimmung ab einstellbarem IoU, Standard 0,5), und der zeitliche Verlauf meldet plötzlich fehlende Spieler und springende Boxen. Auffällige Bilder kommen in eine Prüfliste, die sich Bild für Bild öffnen lässt.
- Neu: Lokales Sprach-Bild-Modell (VLM, Qwen3.5-4B oder -9B, Apache-2.0) – schaut sich nur die auffälligen Bilder und optional eine Stichprobe an. Bestätigte Probleme werden als „Hard Example“ markiert und im Training doppelt verwendet; unbegründete Warnungen werden entfernt.
- Neu: Das VLM ist fest in die App eingebaut, auf jeder Plattform – ohne Ollama oder andere Zusatzinstallation. Beim ersten Einsatz lädt die App eine feste llama.cpp-Version (`llama-server`, MIT) für das System (macOS Apple Silicon mit Metal, Windows und Linux mit CUDA oder CPU, Linux arm64) und das Modell, prüft die Prüfsummen und legt beides einmal pro Rechner ab. Der Server läuft nur während des Jobs auf 127.0.0.1 und gibt danach den Grafikspeicher frei.
- Neu: Beim anonymisierten Export kann das VLM eine Stichprobe vergrößerter Kopfausschnitte auf noch erkennbare Gesichter prüfen; Treffer kommen auf die Prüfliste statt in den Export.
- Hardware-Stufen: Auf „stark“ ist das VLM standardmäßig an (9B wählbar), auf „mittel“ optional, auf „schwach“ (nur CPU) nicht angeboten.
- Geändert (Docker): Das Image basiert auf Debian 13, weil die llama.cpp-Versionen für Linux arm64 und CUDA glibc 2.38 brauchen. Ältere Linux-Systeme nutzen automatisch die CPU-Version.

## 0.22.0

- Neu: Hardware-Stufen (stark, mittel, schwach) – erkannt über NVIDIA-VRAM bzw. den Arbeitsspeicher des Apple-Chips, nur CPU = schwach. Sie bestimmen, welche Modelle genutzt werden, und lassen sich unter „Erweiterte Optionen“ festlegen.
- Neu: Vorschlagsmodell (Teacher) – solange kein eigenes Modell eine Klasse kennt, schlägt das größte RF-DETR-Basismodell vor, das die Hardware zulässt (L, M oder S; Apache-2.0). Wählbar unter „Erweiterte Optionen“.
- Neu: Zweite Meinung mit Florence-2 (MIT) – ist kein zweites eigenes Modell installiert, prüft Florence-2 Personen und Ball und markiert Bilder mit Abweichungen zur Prüfung. Es schreibt selbst keine Boxen.
- Neu: Boxen verfeinern mit SAM 2 (Apache-2.0) als wählbare, experimentelle Methode; es übernimmt nur kleine Korrekturen, die Originalkoordinaten bleiben gespeichert. OpenCV bleibt Standard – in einem ersten Vergleich war SAM 2 nicht messbar besser.
- Neu: Fehlt der Ball zwischen zwei nahen Bildern eines Videos (höchstens 1 s Abstand), wird seine Position interpoliert und als Vorschlag zur Prüfung eingetragen.
- Neu: „Erweiterte Optionen“ zeigt alle Modelle mit Zweck, Lizenz und Ladezustand. SAM 2 und Florence-2 werden bei Bedarf auf eine feste Version festgelegt geladen.

## 0.21.0

- Neu: Jeder längere Vorgang läuft als Job mit Fortschrittsfenster – Schritt („Schritt 2/3“), Balken (unbestimmt, wenn die Menge unbekannt ist), Prozent, verstrichene und geschätzte Restzeit, „Abbrechen“ mit Rückfrage. Das Fenster lässt sich zu einer Statusleiste minimieren und übersteht ein Neuladen der Seite; weitere Vorgänge warten in einer Warteschlange. Der Fortschritt kommt per Server-Sent Events.
- Neu: Abbrechen beendet den ganzen Prozessbaum (ml_worker mit Datenladern, pip, FFmpeg), entfernt halbfertige Dateien und lässt das Projekt unverändert. Ein abgebrochenes Training behält seinen letzten Checkpoint; das nächste Training setzt dort fort. Beim Beenden der App laufen keine Prozesse weiter.
- Neu: Das Training zeigt „Epoche n von m“.
- Neu: „Datensatz exportieren“ – alle geprüften Bilder mit Labels als YOLO und/oder COCO, nach Video aufgeteilt, mit neutralen Dateinamen, `metadata.json` (Version, Methode, Modelle mit Lizenzen) und optional als ZIP.
- Neu: Anonymisierter Export – Köpfe und Gesichter werden weichgezeichnet oder verpixelt (keine Balken), auch Zuschauer und Bank; Trikotnummern und Anzeigetafel optional. Ball und Labels bleiben unverändert. Bilder, in denen danach noch ein Gesicht erkennbar ist, werden nicht exportiert, sondern zur Prüfung aufgelistet.
- Neu: „Anonymisierte Trainingsbilder“ – ein Anteil der Trainingsbilder wird mit weichgezeichneten Köpfen trainiert.
- Neu: Modell-Registry mit Lizenzen (nur Apache-2.0 und MIT) und geprüften Downloads (SHA-256).
- Geändert: „Spielfeld festlegen“ braucht keine Breite und Länge mehr. Mindestens vier Punkte der Reihe nach am Rand, verbunden durch gerade Linien; liegt das Feld nicht ganz im Bild, kommen Punkte auf den Bildrand, so sind beliebige Formen möglich. Geprüft wird immer im Bild. Bestehende Spielfelder mit Maßen gelten als Umriss weiter. Umrisse mit sich kreuzenden Kanten lassen sich nicht speichern.
- Geändert: Jede Klasse hat eine eigene, deutlich unterscheidbare Boxfarbe (Ball rot, Spieler blau, Schiedsrichter braun, Torwart grün, Tor/Torpfosten lila, Korb orange). Automatische Vorschläge sind gestrichelt statt andersfarbig; neben der Klassenauswahl zeigt ein Punkt die Farbe.

## 0.20.0

- Behoben: Der nach dem Training gespeicherte Qualitätswert stammte von den Gewichten der letzten Epoche, obwohl der beste Checkpoint gespeichert wird. Ein früh optimales Modell (zum Beispiel Epoche 4 von 24) wurde dadurch zu schlecht bewertet, auch bei der automatischen Wahl des besten Modells. Jetzt wird der beste Checkpoint bewertet.
- Geändert (Windows, Linux, Docker): Die Weboberfläche wird fertig gebaut mitgeliefert und vom Python-Worker selbst ausgeliefert. Node.js und npm sind nicht mehr nötig; die Oberfläche läuft unter http://localhost:8766/ (ein Port statt zwei). Start-Skripte starten nur noch den Worker und öffnen den Browser.
- Geändert (Entwicklung): Vite + React statt Next.js/vinext/Cloudflare-Werkzeugen; `npm run typecheck` ersetzt `lint`. CI baut und prüft die Oberfläche.
- Geändert (Mac): Die App ist jetzt ein Starter um die Weboberfläche (WKWebView) und startet den Python-Worker. Die SwiftUI-Oberfläche entfällt; Funktionen und Aussehen sind auf allen Systemen identisch. Neue Versionen von Oberfläche und Workern installiert die App selbst (Ed25519-signiertes Paket aus dem GitHub-Release, automatischer Rückfall auf die vorherige Version). Ein vorkompilierter Frame-Extraktor ist enthalten. Projekte bleiben unverändert nutzbar.

## 0.19.2

- Behoben (Mac): Die Schaltflächen unter dem Bild standen alle in einer Zeile; bei normaler Fensterbreite wurden die letzten – darunter „Modell importieren“ und „Labels importieren“ – abgeschnitten und waren unsichtbar. Sie stehen jetzt in zwei Zeilen (Arbeiten · Modelle und Daten) und brechen bei schmalem Fenster um.
- Neu: YOLO-Modelle im ONNX-Format (mit eingebautem NMS) lassen sich direkt importieren – auf dem Mac über „Modell importieren“ (die Auswahl zeigt jetzt auch `.onnx`-Dateien), in der Weboberfläche über „ONNX-Modell importieren“. Ein Formular fragt nach Name, Klassen (Reihenfolge der Klassen-IDs), Eingabegröße und Lizenz; Reco Trainer erstellt daraus das Paket (abgelegt in `inbox/`), installiert und aktiviert es. Übliche Klassennamen wie „person“ werden zugeordnet. Eine ML-Umgebung ist dafür nicht nötig.
- Neuer Worker-Befehl `ml_worker.py install-onnx`.
- „Neu in Reco Trainer“ zeigt jetzt die Neuerungen von 0.19.x.

## 0.19.1

- Neu: Schaltfläche „Labels importieren“ in der Mac-App (neben „Modell importieren“) und in der Weboberfläche. Einen Ordner mit Bildern und YOLO-txt- oder COCO-Labels wählen; Reco Trainer übernimmt sie ins geöffnete Projekt und meldet die Anzahl der Bilder und Boxen sowie nicht übernommene Klassen. Über das Menü der Schaltfläche (Mac) bzw. „Labels importieren: Als Vorschläge“ (Web) landen die Boxen stattdessen ungeprüft in der Prüfwarteschlange.
- Für den Import ist keine ML-Umgebung nötig. Klassennamen kommen aus `data.yaml`, `classes.txt` oder der COCO-Datei; für eine eigene Klassenzuordnung (`--class-map`) bleibt der Befehl `ml_worker.py import-labels`.
- Das Projekt wird nach dem Import automatisch neu geladen.

## 0.19.0

- Neu: Beschriftete Bilder aus anderen Quellen lassen sich ins Projekt importieren – YOLO-txt (auch Ultralytics-Ordner mit `images/` und `labels/`, Polygon-Zeilen werden zur Box) und COCO-JSON. Aufruf: `ml_worker.py import-labels --project <Projektordner> --source <Ordner>` (siehe README). Klassen werden auf die Reco-Klassen der Sportart abgebildet (z. B. person → player), unbekannte Klassen werden übersprungen und gemeldet.
- Importierte Bilder landen in `frames/imported/` und gelten als geprüft (mit `--candidates` als Vorschläge). Doppelte Bilder werden erkannt. Windows/Linux/Docker behalten importierte Frames beim erneuten Vorbereiten der Videos.
- Noch ohne Schaltfläche in der Oberfläche.

## 0.18.0

- Neu: Externe ONNX-Detektoren (YOLO mit eingebautem NMS, Ausgabe 1 × N × 6) lassen sich als `.recomodel`-Paket über „Modell importieren“ einbinden. Sie dienen zum automatischen Vorbeschriften und für den Modellvergleich (Benchmark). Trainieren oder Feinabstimmen ist mit ihnen nicht möglich; das Training läuft weiter mit RF-DETR.
- Neues Werkzeug `cross-platform/tools/make_onnx_package.py` erzeugt ein solches Paket aus einer `.onnx`-Datei (Name, Sportart, Klassen in Reihenfolge der Klassen-IDs, Eingabegröße). `validate_model_package.py` prüft auch diese Pakete.
- ONNX-Pakete sind unabhängig von der gewählten RF-DETR-Größe (nano/small) und lassen sich nicht zu kombinierten Modellen zusammenfassen.

## 0.17.2

- Windows/Linux/Docker: Die Größenbegrenzung der extrahierten Bilder (bisher höchstens 1280 Pixel Breite) ist entfernt. Bilder behalten jetzt die volle Auflösung des Videos, wie bereits in der Mac-App. Kleine Objekte, etwa Spieler in der entfernten Ecke eines 11-gegen-11-Vollfeldbildes, behalten so ihre Pixel. Bereits vorbereitete Projekte behalten ihre alten Bilder; für volle Auflösung die Videos neu vorbereiten. Größere Bilder brauchen mehr Speicherplatz und Rechenzeit.

## 0.17.1

- „ML einrichten“ scheitert nicht mehr mit „No matching distribution found for rfdetr[…]>=1.9.0“, wenn nur das ältere System-Python (3.9) gefunden wurde. Reco Trainer sucht jetzt selbst ein Python ab 3.10 (3.12 und 3.11 zuerst, auch Homebrew und python.org), prüft dessen Version und legt die ML-Umgebung damit an.
- Eine früher mit Python 3.9 angelegte ML-Umgebung (`.runtime/venv`) wird erkannt und automatisch neu erstellt.
- Ist kein passendes Python installiert, erscheint eine klare Meldung mit dem Hinweis auf „brew install python@3.12“ statt des pip-Fehlers.
- Windows/Linux/Docker: dieselbe Python-Auswahl und Erkennung veralteter Umgebungen; die Python-Warnung erscheint nur noch, wenn wirklich kein Python 3.10+ vorhanden ist.

## 0.17.0

- Spielfeld festlegen (Mac-App und Weboberfläche): nicht mehr auf vier Ecken beschränkt. Weitere Punkte rund um den Rand setzen (bis zu 32, z. B. für runde oder L-förmige Felder), Punkte per Ziehen verschieben und den jeweils letzten Punkt wieder entfernen. Bei genau vier Punkten bleibt alles wie bisher (echte Maße nötig); bei mehr Punkten sind die Maße optional und „Automatisch markieren“ prüft, ob die Füße innerhalb der Umrisse im Bild liegen.

## 0.16.1

- Mac-App: 47 Beschriftungen und Statusmeldungen gab es nur auf Deutsch und Englisch und erschienen für Spanisch und Französisch auf Englisch (u. a. Sprache/Sportart, „Modell verbessern“, „Von Grund auf neu trainieren“, „Mindest-Sicherheit“, Werkzeug-Beschriftungen, Fortschrittsmeldungen). Jetzt vollständig in allen vier Sprachen.
- Weboberfläche: alle 9 TypeScript-Typfehler und ein Lint-Fehler im Balltracking-Player behoben.

## 0.16.0

- Windows/Linux/Docker: Die Weboberfläche hat jetzt denselben Funktionsumfang wie die Mac-App - „Boxen mit OpenCV verfeinern“, „Spielfeld festlegen/bearbeiten“ (inkl. einmaligem Hinweis), Export „CPU-Modell (ONNX)“ und „Trainingsordner anzeigen“. Der Core-ML-Export steht dort nur auf einem Mac zur Verfügung.
- Neue Voreinstellung: **60 Bilder je Video** (bisher 240) in Mac-App und Weboberfläche; weiterhin von 4 bis 5000 einstellbar, bestehende Projekte behalten ihren Wert.
- Weboberfläche: Schaltflächen im Primär-Stil in Dialogen (z. B. „Loslegen“ im Video-Dialog) waren unsichtbar - behoben. Die Trainings-Schaltflächen umbrechen jetzt in eine eigene Zeile.
- Mac-App: die restlichen Beschriftungen im Trainings-Bereich gibt es jetzt auch auf Spanisch und Französisch.

## 0.15.4

- Windows/Linux/Docker: Der Clip der Balltracking-Simulation spielte nie über seine erste Sekunde hinaus, weil die Oberfläche den Status alle 900 ms neu lädt und der Player dadurch jedes Mal bei Bild 1 neu startete. Jetzt läuft der Clip in voller Länge.
- Windows/Linux/Docker: Die Balltracking-Simulation zeigt jetzt einen gut sichtbaren Fortschrittsbalken mit Statuszeile und Prozentangabe; Bildextraktion und Ballerkennung melden echten Fortschritt.

## 0.15.3

- Windows/Linux/Docker: Die Balltracking-Simulation konnte mit „Expecting ',' delimiter: line 1 column 6 (char 5)“ fehlschlagen, weil eine Log-Zeile der ML-Bibliotheken vor dem Ergebnis ausgegeben wurde und die gesamte Ausgabe als ein JSON-Dokument gelesen wurde. Es wird jetzt nur noch die letzte Ausgabezeile ausgewertet (wie in der Mac-App bereits seit 0.14.x).

## 0.15.2

- Die Video-Einführung stürzte beim Start des Videos ab (SIGABRT in Apples privatem _AVKit_SwiftUI-Framework, einer bekannten Schwachstelle von SwiftUIs `VideoPlayer` unter macOS 27). Die Wiedergabe läuft jetzt über AVKits klassische `AVPlayerView` statt über SwiftUIs `VideoPlayer` und umgeht damit das betroffene Framework vollständig.

## 0.15.1

- Der in 0.15.0 neu eingeführte Video-Einführung konnte im gepackten App-Bundle nicht gefunden werden ("Video nicht gefunden") - die Ressourcen-Suche prüfte den falschen Pfad innerhalb des Bundles. Behoben.
- Der Dialog „Neu in Reco Trainer“ zeigte bei jedem Versions-Update die gesamte Änderungshistorie vergangener Versionen statt nur der aktuellen - jetzt wirklich nur die Neuerungen der jeweils installierten Version.

## 0.15.0

- Neue Video-Einführung beim ersten Start: Nach der Sprachwahl kannst du zwischen einem rund einminütigen Video und der bisherigen Schritt-für-Schritt-Anleitung wählen. Das Video liegt in allen vier Sprachen mit passenden Untertiteln vor, läuft vollständig lokal und zeigt bewusst KI-generiertes Beispielmaterial statt echter Aufnahmen. Jederzeit erneut über „Ablauf erklären“ aufrufbar.

## 0.14.4

- Windows/Linux/Docker: neues Feld zum Einstellen der Trainings-Epochen (bisher fest auf 20) - Startwert jetzt 100, wie in der Mac-App, deren eigener Startwert ebenfalls angehoben wurde.
- Windows/Linux/Docker: viele Status- und Fehlermeldungen, die bisher unabhängig von der gewählten Sprache fest auf Deutsch standen, sind jetzt korrekt in allen vier Sprachen - unter anderem die Hardware-Anzeige, Modell-Testwerte, die Paketbeschreibung eines aktiven Modells und praktisch jede Statuszeile während eines Vorgangs.

## 0.14.3

- Die Code-Sicherheitsprüfung für Modellpakete läuft jetzt zusätzlich verpflichtend unmittelbar vor jeder tatsächlichen Nutzung einer Gewichtsdatei (Training, automatisches Markieren, Modellvergleich, Export) - nicht mehr nur beim Import, wo sie bei fehlender ML-Umgebung übersprungen werden konnte. Zip-Einträge, die als Symlink markiert sind, werden jetzt ebenfalls grundsätzlich abgelehnt.

## 0.14.2

- Modellpakete (.recomodel) werden beim Import (und beim Erstellen/Kombinieren) jetzt zusätzlich mit PyTorchs sicherem Lademodus (weights_only) geprüft: Gewichtsdateien, die mehr als reine Modelldaten enthalten - etwa ausführbaren Code -, werden erkannt und abgelehnt, nicht nur per Prüfsumme verglichen. Ohne eingerichtete ML-Umgebung wird weiterhin nur die Prüfsumme geprüft.

## 0.14.1

- Neuer Schalter „Von Grund auf neu trainieren“: ignoriert für einen Lauf jeden vorhandenen Checkpoint und startet vom Basismodell. RF-DETR erweitert beim Fortsetzen von einem Checkpoint mit weniger Klassen (z. B. nur Ball) den Klassifikations-Kopf nicht auf neue Klassen.
- Ein neues Modell mit anderen Klassen als das aktive wird nicht mehr allein wegen eines niedrigeren Gesamtwerts als schlechter behandelt (die Werte sind nicht vergleichbar); es wird archiviert, mit Hinweis auf den Modellvergleich pro Kategorie.

## 0.14.0

- Neue Prüf-Priorität für automatisch beschriftete Bilder: ein optionaler zweiter Modelldurchlauf markiert Bilder, bei denen sich zwei installierte Modelle uneinig sind (Schalter „Zweites Modell zur Unsicherheits-Prüfung nutzen“), und eine kostenlose Prüfung markiert Bilder, deren Position stark von den Nachbar-Bildern desselben Videos abweicht. Markierte Bilder erscheinen zuerst in der Prüfwarteschlange, mit Warnsymbol und Begründung. Betrifft nur die Reihenfolge der Prüfung, nie das Training selbst.

## 0.13.2

- Die Balltracking-Simulation konnte mit „Die Daten konnten nicht geöffnet werden, da sie nicht das korrekte Format haben“ fehlschlagen: Warnungen von PyTorch/RF-DETR beim Laden des Modells landeten in derselben Ausgabe wie das Ergebnis. Nur noch die letzte Zeile der Ausgabe wird dafür verwendet.

## 0.13.1

- Der Ordner-Browser für „Unabhängiger Modelltest“ hing bei vielen Videos (z. B. GoPro-Aufnahmen mit mehreren Kapitel-Dateien): die neue „bereits verwendet“-Markierung durchsuchte jeden sichtbaren Unterordner vollständig rekursiv statt nur die direkte Ebene.
- Das Einlesen der Videolängen vor der Extraktion zeigt jetzt Fortschritt an, statt kommentarlos zu pausieren.

## 0.13.0

- Neuer Bereich „Balltracking simulieren“: ein kurzes Video wählen und die Ballerkennung des aktiven Modells Bild für Bild abspielen - erkannt, interpoliert, gehalten oder verloren, mit einem Live-Regler für den Vorhersage-Horizont.
- Windows: „ML einrichten“ erkennt jetzt automatisch eine NVIDIA-Grafikkarte und installiert PyTorch mit CUDA-Unterstützung - vorher landete dort standardmäßig die CPU-only-Version. Abschaltbar über einen neuen Schalter, falls das Probleme macht.
- Die Hardware-Anzeige zeigt unter Windows/Linux jetzt korrekt an, ob CUDA genutzt wird, statt immer nur „CPU“ - unabhängig von der obigen automatischen Einrichtung.
- Beim Vorbereiten vieler Videos (z. B. GoPro-Aufnahmen mit mehreren Kapitel-Dateien) zeigt die Weboberfläche jetzt den Fortschritt beim Einlesen der Videometadaten an, statt scheinbar zu hängen.

## 0.12.15

- Neuer Bereich „Unabhängiger Modelltest“: einen Videoordner prüfen, der nie zum Training verwendet wurde, und daraus - mit Unterstützung über alle Kategorien hinweg - ein echtes, unabhängiges Testset erstellen.
- Der Modellvergleich verlangt jetzt unabhängige Testbilder für die Referenz (nicht mehr beliebige geprüfte Trainingsbilder) - eine bestehende Referenz muss einmal neu festgelegt werden.
- Neue Aktion „Kombiniertes Modell erstellen“ beim Modellvergleich: aus mehreren installierten Modellen pro Kategorie das stärkste auswählen und daraus ein neues Modell backen (Ensemble zur Laufzeit, kein echtes Gewichte-Merging).
- Die plattformübergreifende Weboberfläche zeigt jetzt ebenfalls den Vergleich je Kategorie (mAP@.50), wie die Mac-App bereits seit 0.12.9.

## 0.12.14

- Windows: „Lokal trainieren“ stürzte während der ersten Epoche mit einem Kodierungsfehler ab, weil die Konsolenausgabe dort standardmäßig nicht auf UTF-8 steht. Behoben.
- Die Windows/Linux/Docker-Weboberfläche prüft jetzt beim Start ebenfalls, ob eine neuere Version veröffentlicht wurde (wie die Mac-App bereits seit 0.12.9).

## 0.12.13

- Jeder Modelleintrag in der Modellverwaltung zeigt jetzt direkt, welche Klassen das Modell tatsächlich kann.
- Windows: „ML einrichten“ (und jeder folgende lokale ML-Schritt) scheiterte mit „Die angegebene Datei wurde nicht gefunden“, weil die App intern einen reinen macOS/Linux-Dateipfad für die virtuelle Python-Umgebung erwartet hat. Behoben.

## 0.12.12

- Das „Modell“-Dropdown bei „4 · Modell verbessern“ zeigte den vollen Wert nicht an (zu schmal). Behoben.
- Die Trainingsbilder-Liste unterstützt jetzt ⌘-Klick/Umschalt-Klick für Mehrfachauswahl, mit einem eigenen „Entfernen“-Knopf für die Auswahl.
- Die Bestätigung beim Entfernen eines Trainingsbilds hat einen „Nicht mehr fragen“-Haken bekommen.
- Neue Klassen-Chips bei „4 · Modell verbessern“ lassen Dich einzelne Klassen von einem Trainingslauf ausschließen (z. B. nur Ball und Schiedsrichter trainieren).

## 0.12.11

- Der „Schiedsrichter“-Chip bei „Automatisch markieren“ ist jetzt für jede Sportart nutzbar, auch ohne eigens trainiertes Modell: Personen mit typischer Schiedsrichter-Kleidung (z. B. komplett Schwarz im Fußball, Grau/Schwarz im Basketball, Schwarz-Weiß gestreift im Eishockey) werden automatisch als „Schiedsrichter“ vorgeschlagen - weiterhin zur Prüfung markiert, keine Garantie. Setzt voraus, dass „Spieler“ ebenfalls ausgewählt ist.

## 0.12.10

- Neuer Button „Spielfeld festlegen“: Feldmaße eintragen und die vier Eckpunkte im Bild anklicken. „Automatisch markieren“ berücksichtigt danach bei Spieler/Schiedsrichter/Torwart nur noch Personen, die mit den Füßen tatsächlich auf dem markierten Feld stehen - Zuschauer, Bank und Betreuer werden ignoriert. Bestehende Projekte ohne Markierung funktionieren unverändert weiter.

## 0.12.9

- Reco Trainer prüft jetzt beim Start einmal, ob auf GitHub eine neuere Version veröffentlicht wurde, und zeigt bei Bedarf einen Hinweis mit Download-Link (nur die öffentliche GitHub-Releases-API wird dafür abgefragt, keine Projektdaten).
- „Boxen mit OpenCV verfeinern“ (vormals „Ball-Boxen mit OpenCV prüfen“) zieht jetzt auf Wunsch auch von Hand gezeichnete Boxen nach, nicht mehr nur automatische Vorschläge.
- „Spieler“ bleibt bei „Automatisch markieren“ auswählbar, auch wenn das aktive Modell nur andere Klassen (z. B. „Ball“) gelernt hat - Reco Trainer nutzt dafür zusätzlich das allgemeine Basismodell.

## 0.12.8

- Ein aktiviertes Modell, das z. B. nur mit „Ball“ trainiert wurde, ließ „Automatisch markieren“ trotzdem eine volle Erkennung für „Schiedsrichter“ oder „Spieler“ durchlaufen und meldete am Ende nur ein pauschales „0 Boxen“. Reco Trainer prüft jetzt die tatsächlichen Klassen des aktivierten Modells und überspringt automatisch, was es nachweislich nicht kennt - mit einer klaren Erklärung je Klasse.

## 0.12.7

- Ein neu geladenes Projekt wählt beim automatischen Markieren jetzt jede aktuell vom Modell unterstützte Klasse vor (beim Basismodell: Ball und Spieler). Da jede Person allgemein als „Spieler“ erkannt wird, muss nur noch der Schiedsrichter einzeln angeklickt und umbenannt werden.
- Der Modellvergleich zeigt jetzt zusätzlich eine Tabelle mit mAP@.50 je Kategorie und Modell, um z. B. zu erkennen, dass ein Modell besser bei Bällen und ein anderes besser bei Schiedsrichtern ist.

## 0.12.6

- Nach dem Überspringen eines nicht mehr passenden Fortsetzungsversuchs konnte ein neuer Feinabstimmungslauf trotzdem noch mit der veralteten, kleineren Klassenanzahl des Checkpoints starten und später mit einem Absturz tief in der Trainingslogik enden. Reco Trainer verwendet jetzt immer die aktuelle Klassenanzahl des Projekts für den Modellaufbau.

## 0.12.5

- Ein abgebrochener Trainingslauf ließ sich nicht mehr fortsetzen, wenn seitdem über automatisches Markieren eine neue Klasse hinzugekommen war: Training stürzte mit einem "size mismatch"-Fehler ab, bevor eine einzige Epoche lief. Reco Trainer prüft jetzt vor dem Fortsetzen, ob die Klassenanzahl noch zum abgebrochenen Checkpoint passt, und startet andernfalls automatisch einen neuen Feinabstimmungslauf statt abzustürzen.

## 0.12.4

- "Automatisch markieren" erkennt jetzt mehrere ausgewählte Klassen in einem Durchlauf, statt für jede Klasse einen eigenen Durchlauf zu benötigen. Nicht vom aktuellen Modell unterstützte Klassen sind bei der Auswahl ausgegraut.
- Die OpenCV-Nachprüfung automatischer Boxen ist von Ball/Puck auf alle Sport-Klassen (Spieler, Schiedsrichter, Torwart, Korb, Tor, Torpfosten) erweitert.
- Boxen im Annotations-Editor lassen sich jetzt anklicken, auswählen, verschieben, löschen und auf eine andere Klasse ändern (z. B. Spieler ↔ Schiedsrichter) - vorher ging das nur durch Löschen und komplettes Neuzeichnen.
- "Videos lokal vorbereiten" liest jedes Video in einem Durchgang statt für jeden Frame einzeln neu zu springen; das beschleunigt die Extraktion spürbar.

## 0.12.3

- Der Windows-Starter (`Start Reco Trainer Windows.bat`) öffnete sich kurz und schloss sich sofort wieder, ausgelöst durch die in 0.12.2 hinzugefügte Python-Versionsprüfung. Die Prüfung läuft jetzt in Reco Trainer selbst statt im Batch-Skript und wird über die Statuszeile angezeigt.
- Automatische Vorschläge lassen sich jetzt in der Mac-App direkt annehmen oder verwerfen (bisher gab es dafür keinen Mechanismus, wodurch Schritt 1 des Modellvergleichs nach jedem "Automatisch markieren" dauerhaft blockiert war).

## 0.12.2

- Der lokale Modellvergleich berechnet die Boxüberlappung (IoU) jetzt korrekt; Qualitätswerte, mAP, Präzision und Recall sind zuverlässiger.
- Offene Prüfkandidaten aus der Datensatzerweiterung blockieren das Festlegen der Referenz nicht mehr fälschlich.
- Beide Aktionen im Modellvergleich zeigen jetzt einen Hinweis, welche Voraussetzung noch fehlt.
- Die Windows- und Linux-Startskripte warnen deutlich, wenn die gefundene Python-Version nicht 3.11 oder 3.12 ist.
- Interne Code-Dopplungen bei Checkpoint-Suche und Frame-Extraktion entfernt; automatische Tests laufen jetzt bei jedem Commit.

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
