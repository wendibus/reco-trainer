# Reco Trainer für Mac

Die Mac-App ist ein schlanker **Starter**: ein natives Fenster um die Weboberfläche, die auf Windows, Linux und Docker identisch aussieht, plus der Code, der den Python-Worker startet und Updates installiert. Die gesamte Funktion (Oberfläche, Frames, Auto-Label, Training, Benchmark, Modellaustausch) liegt in `cross-platform/` und ist für alle Systeme dieselbe. Videos, Frames und Modelle bleiben im gewählten Ordner; die App hat keinen Upload.

## Wie es zusammenhängt

- **App** (`Reco Trainer.app`, signiert und notarisiert): ändert sich selten. Sie enthält den Starter und eine Kopie der **Nutzlast** (Payload).
- **Payload**: Weboberfläche (`ui/`), `local_worker.py`, `ml_worker.py`, `frame_extractor.swift`. Beim ersten Start wird sie nach `~/Library/Application Support/Reco Trainer/payloads/<Version>/` kopiert und von dort gestartet.
- **Updates**: Beim Start (und über „Nach Updates suchen …“) prüft die App `releases/latest` auf GitHub. Enthält das Release `Reco.Trainer.Payload.<Version>.zip` und `.zip.sig`, wird nach Rückfrage heruntergeladen, die **Ed25519-Signatur** geprüft und die neue Version installiert; der Worker startet neu. Startet eine neue Version nicht, fällt die App automatisch auf die vorherige zurück. Es bleiben die zwei neuesten Versionen erhalten.
- Braucht ein Release eine neue App (neue Mindestversion des Starters), öffnet die App stattdessen die Download-Seite.
- Der Worker läuft auf `127.0.0.1:8766` (bei Belegung der nächste freie Port). Sein Protokoll liegt in `~/Library/Logs/Reco Trainer/worker.log` (Menü Hilfe → Protokoll zeigen).
- Voraussetzung auf dem Mac: **Python 3** (empfohlen 3.12, zum Beispiel `brew install python@3.12`). „ML einrichten“ legt wie bisher eine private Umgebung unter `.reco-training/.runtime/` im Videoordner an.

## Entwickeln

Voraussetzungen: macOS 14+, Xcode Command Line Tools, Node.js 22 (nur zum Bauen der Oberfläche).

```bash
cd mac
./scripts/run-mac.sh          # baut die Payload und startet den Starter
swift test                    # Unit-Tests (Versionen, Payload-Verwaltung, Signaturen, Update-Planung)
./scripts/package-app.sh      # lokal signierte .app (Payload und vorkompilierter Frame-Extraktor sind enthalten)
./scripts/package-dmg.sh      # DMG
```

Für eine öffentliche Version `RECO_SIGN_IDENTITY` (Developer ID) und für `scripts/notarize.sh` zusätzlich `RECO_NOTARY_PROFILE` setzen.

## Update-Signierung (einmalig einrichten)

```bash
swift mac/tools/payload-signing.swift keygen
```

Das legt den privaten Schlüssel unter `~/.reco-trainer/payload-signing.key` an (geheim halten, sichern, nie einchecken) und gibt den öffentlichen Schlüssel aus. Diesen in `mac/Sources/LauncherCore/PayloadKey.swift` eintragen und die App neu bauen. **Solange er leer ist, installiert die App keine Downloads**, sondern nur die eingebaute Payload.

## Release ohne neue App (nur Oberfläche/Worker geändert)

```bash
mac/scripts/release-payload.sh
```

Das erzeugt `mac/dist/Reco.Trainer.Payload.<Version>.zip` und `.zip.sig`. Beides an das GitHub-Release hängen (Tag `v<Version>`, die Version steht in `cross-platform/package.json`). Wird `minLauncher` in `cross-platform/scripts/build-payload.sh` (Variable `RECO_PAYLOAD_MIN_LAUNCHER`) höher gesetzt als die installierte App, verlangt das Update eine neue App.
