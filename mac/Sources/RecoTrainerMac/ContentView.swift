import AppKit
import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var app: AppState
    @AppStorage("recoWalkthroughCompleteV1") private var walkthroughComplete = false
    @AppStorage("recoPreferredLanguageV1") private var preferredLanguage = ""
    @State private var showWalkthrough = false
    @State private var showLanguagePicker = false
    @State private var walkthroughIndex = 0
    @State private var showBenchmark = false
    @State private var confirmFrameRemoval = false

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .frame(minWidth: 1120, minHeight: 720)
        .alert(app.tr("Hinweis", "Notice"), isPresented: Binding(
            get: { app.errorMessage != nil },
            set: { if !$0 { app.errorMessage = nil } }
        )) {
            Button("OK") { app.errorMessage = nil }
        } message: {
            Text(app.errorMessage ?? "")
        }
        .confirmationDialog(
            app.tr("Trainingsbild entfernen?", "Remove training image?", "¿Quitar imagen de entrenamiento?", "Retirer l’image d’entraînement ?"),
            isPresented: $confirmFrameRemoval,
            titleVisibility: .visible
        ) {
            Button(app.tr("Bild entfernen", "Remove image", "Quitar imagen", "Retirer l’image"), role: .destructive) { app.removeSelectedFrame() }
            Button(app.tr("Abbrechen", "Cancel", "Cancelar", "Annuler"), role: .cancel) {}
        } message: {
            Text(app.tr("Nur der extrahierte Trainingsframe wird entfernt. Das Quellvideo bleibt unverändert.", "Only the extracted training frame is removed. The source video remains unchanged.", "Solo se quita el fotograma extraído. El vídeo original no se modifica.", "Seule l’image extraite est retirée. La vidéo source reste inchangée."))
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                Label("Reco Trainer", systemImage: "lock.shield.fill")
                    .font(.headline)
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showBenchmark.toggle()
                } label: {
                    Label(
                        showBenchmark
                            ? app.tr("Training", "Training", "Entrenamiento", "Entraînement")
                            : app.tr("Modelle testen", "Test models", "Probar modelos", "Tester les modèles"),
                        systemImage: showBenchmark ? "arrow.left" : "chart.bar.xaxis"
                    )
                }
            }
            ToolbarItem(placement: .primaryAction) {
                Button {
                    walkthroughIndex = 0
                    showWalkthrough = true
                } label: {
                    Label(
                        app.tr("Ablauf erklären", "Show workflow", "Mostrar el flujo", "Afficher le parcours"),
                        systemImage: "questionmark.circle"
                    )
                }
            }
        }
        .onAppear {
            if let saved = AppLanguage(rawValue: preferredLanguage) {
                app.language = saved
                if !walkthroughComplete { showWalkthrough = true }
            } else {
                showLanguagePicker = true
            }
        }
        .onChange(of: app.language) { _, language in
            if !showLanguagePicker { preferredLanguage = language.rawValue }
        }
        .sheet(isPresented: onboardingPresented) {
            OnboardingContainer(
                showLanguagePicker: showLanguagePicker,
                language: app.language,
                walkthroughIndex: $walkthroughIndex,
                selectLanguage: { language in
                    app.language = language
                    preferredLanguage = language.rawValue
                    showLanguagePicker = false
                    walkthroughIndex = 0
                    showWalkthrough = true
                },
                finish: {
                    walkthroughComplete = true
                    showWalkthrough = false
                }
            )
        }
    }

    private var onboardingPresented: Binding<Bool> {
        Binding(
            get: { showLanguagePicker || showWalkthrough },
            set: { presented in
                if !presented {
                    showLanguagePicker = false
                    showWalkthrough = false
                }
            }
        )
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 16) {
            Picker(app.tr("Sprache", "Language"), selection: $app.language) {
                ForEach(AppLanguage.allCases) { language in Text(language.title).tag(language) }
            }
            .pickerStyle(.segmented)

            VStack(alignment: .leading, spacing: 8) {
                Text(app.tr("1 · Sportart", "1 · Sport")).font(.headline)
                Picker(app.tr("Sportart", "Sport"), selection: $app.sport) {
                    ForEach(Sport.allCases) { sport in Text(sport.title(language: app.language)).tag(sport) }
                }
                .labelsHidden()
                .disabled(app.project != nil || app.isWorking)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text("2 · Videos").font(.headline)
                Button {
                    app.chooseFolder()
                } label: {
                    Label(app.tr("Videoordner auswählen", "Select video folder"), systemImage: "folder")
                }
                Text(app.selectedFolder?.path(percentEncoded: false) ?? app.tr("Noch kein Ordner gewählt", "No folder selected"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                Stepper(
                    app.tr("\(app.framesPerVideo) Bilder je Video", "\(app.framesPerVideo) images per video", "\(app.framesPerVideo) imágenes por vídeo", "\(app.framesPerVideo) images par vidéo"),
                    value: $app.framesPerVideo,
                    in: 4...5000,
                    step: 20
                )
                .font(.caption)
                Button {
                    app.analyzeVideos()
                } label: {
                    Label(app.tr("Videos lokal analysieren", "Analyze videos locally"), systemImage: "film.stack")
                }
                .buttonStyle(.borderedProminent)
                .disabled(app.selectedFolder == nil || app.isWorking)
            }

            if let project = app.project {
                Divider()
                Text(app.tr("3 · Trainingsbilder", "3 · Training images")).font(.headline)
                List(project.frames, selection: $app.selectedFrameID) { frame in
                    HStack(spacing: 8) {
                        if let store = app.store, let image = NSImage(contentsOf: store.frameURL(for: frame)) {
                            Image(nsImage: image)
                                .resizable()
                                .scaledToFill()
                                .frame(width: 56, height: 36)
                                .clipped()
                                .cornerRadius(4)
                        }
                        VStack(alignment: .leading) {
                            Text(frame.videoName).lineLimit(1)
                            Text(String(
                                format: app.tr("%02d:%02d · %d Markierungen", "%02d:%02d · %d annotations"),
                                Int(frame.timestamp) / 60,
                                Int(frame.timestamp) % 60,
                                frame.annotations.count
                            ))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .tag(frame.id)
                }
                .listStyle(.sidebar)
            }

            Spacer(minLength: 0)
            Label(app.tr("Keine Videos oder Frames werden hochgeladen.", "No videos or frames are uploaded."), systemImage: "lock.fill")
                .font(.caption)
                .foregroundStyle(.green)
        }
        .padding()
        .navigationSplitViewColumnWidth(min: 280, ideal: 320)
    }

    @ViewBuilder
    private var detail: some View {
        if showBenchmark {
            benchmarkPanel
        } else if let frame = app.selectedFrame, let store = app.store {
            VStack(spacing: 14) {
                annotationToolbar
                AnnotationEditor(
                    imageURL: store.frameURL(for: frame),
                    frame: frame,
                    selectedCategory: app.selectedCategory,
                    language: app.language
                ) { annotations in
                    app.updateAnnotations(for: frame.id, annotations)
                }
                .frame(minHeight: 420)
                trainingPanel
            }
            .padding(18)
        } else {
            welcome
        }
    }

    private var benchmarkPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 5) {
                    Label(app.tr("Lokaler Modellvergleich", "Local model benchmark", "Comparación local de modelos", "Comparaison locale des modèles"), systemImage: "chart.bar.xaxis")
                        .font(.largeTitle.bold())
                    Text(app.tr(
                        "Alle kompatiblen Modelle erhalten dieselben geprüften Bilder. Bilder und Vorhersagen verlassen diesen Mac nicht.",
                        "Every compatible model receives the same reviewed images. Images and predictions never leave this Mac.",
                        "Todos los modelos compatibles reciben las mismas imágenes revisadas. Las imágenes y predicciones nunca salen de este Mac.",
                        "Tous les modèles compatibles reçoivent les mêmes images vérifiées. Les images et prédictions ne quittent jamais ce Mac."
                    ))
                    .foregroundStyle(.secondary)
                }

                HStack(alignment: .top, spacing: 12) {
                    GroupBox(app.tr("1 · Richtige Antworten", "1 · Correct answers", "1 · Respuestas correctas", "1 · Bonnes réponses")) {
                        VStack(alignment: .leading, spacing: 10) {
                            Text(app.tr(
                                "Prüfe alle Boxen. Bilder ohne Box gelten nach dem Festlegen bewusst als negative Beispiele.",
                                "Review every box. Images without boxes become intentional negative examples when frozen.",
                                "Revisa cada cuadro. Las imágenes sin cuadros se convierten en ejemplos negativos intencionados.",
                                "Vérifiez chaque boîte. Les images sans boîte deviennent des exemples négatifs intentionnels."
                            )).font(.caption).foregroundStyle(.secondary)
                            if let reference = app.benchmarkGroundTruth {
                                Label("\(reference.frames.count) \(app.tr("Bilder", "images", "imágenes", "images")) · \(reference.annotationCount) \(app.tr("Boxen", "boxes", "cuadros", "boîtes"))", systemImage: "checkmark.seal.fill")
                                    .foregroundStyle(.green)
                            }
                            Button(app.tr("Geprüfte Antworten festlegen", "Freeze reviewed answers", "Fijar respuestas revisadas", "Figer les réponses vérifiées"), action: app.freezeBenchmarkGroundTruth)
                                .buttonStyle(.borderedProminent)
                                .disabled(app.project == nil || app.isWorking || (app.project?.frames.flatMap(\.annotations).contains { $0.source == "auto" } ?? false))
                        }.frame(maxWidth: .infinity, minHeight: 125, alignment: .topLeading)
                    }
                    GroupBox(app.tr("2 · Lokale Modelle", "2 · Local models", "2 · Modelos locales", "2 · Modèles locaux")) {
                        VStack(alignment: .leading, spacing: 10) {
                            Text("\(app.installedModelCount) " + app.tr("installierte Austauschmodelle", "installed exchange models", "modelos de intercambio instalados", "modèles d’échange installés"))
                                .font(.title3.bold())
                            Text(app.tr("Nur Modelle derselben Sportart werden bewertet.", "Only models for the same sport are evaluated.", "Solo se evalúan modelos del mismo deporte.", "Seuls les modèles du même sport sont évalués."))
                                .font(.caption).foregroundStyle(.secondary)
                        }.frame(maxWidth: .infinity, minHeight: 125, alignment: .topLeading)
                    }
                    GroupBox(app.tr("3 · Vergleich starten", "3 · Start comparison", "3 · Iniciar comparación", "3 · Lancer la comparaison")) {
                        VStack(alignment: .leading, spacing: 9) {
                            HStack { Text(app.tr("Schwelle", "Threshold", "Umbral", "Seuil")); Spacer(); Text("\(Int(app.benchmarkThreshold * 100)) %").monospacedDigit() }
                            Slider(value: $app.benchmarkThreshold, in: 0.01...0.5, step: 0.01)
                            Button(app.tr("Alle Modelle lokal testen", "Test every model locally", "Probar todos los modelos localmente", "Tester tous les modèles localement"), action: app.benchmarkModels)
                                .buttonStyle(.borderedProminent)
                                .disabled(app.benchmarkGroundTruth == nil || app.installedModelCount == 0 || app.isWorking)
                            if app.isWorking { ProgressView() }
                        }.frame(maxWidth: .infinity, minHeight: 125, alignment: .topLeading)
                    }
                }

                GroupBox(app.tr("Automatisches Ranking", "Automatic ranking", "Clasificación automática", "Classement automatique")) {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(app.tr(
                            "Qualität = 70 % mAP@0.50 + 30 % F1. Die Geschwindigkeit entscheidet nur bei Gleichstand.",
                            "Quality = 70% mAP@0.50 + 30% F1. Speed is only the tie-breaker.",
                            "Calidad = 70 % mAP@0.50 + 30 % F1. La velocidad solo desempata.",
                            "Qualité = 70 % mAP@0.50 + 30 % F1. La vitesse départage seulement les égalités."
                        )).font(.caption).foregroundStyle(.secondary)
                        if let report = app.benchmarkReport, app.benchmarkReportIsCurrent {
                            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
                                GridRow {
                                    Text("#").bold(); Text(app.tr("Modell", "Model", "Modelo", "Modèle")).bold(); Text(app.tr("Qualität", "Quality", "Calidad", "Qualité")).bold(); Text("mAP@.50").bold(); Text(app.tr("Präzision", "Precision", "Precisión", "Précision")).bold(); Text("Recall").bold(); Text("F1").bold(); Text("ms/" + app.tr("Bild", "image", "imagen", "image")).bold(); Text("FP/FN").bold()
                                }
                                Divider().gridCellColumns(9)
                                ForEach(report.results) { result in
                                    GridRow {
                                        Text(result.rank.map { "#\($0)" } ?? "–").bold()
                                        VStack(alignment: .leading) { Text(result.packageID).lineLimit(1); Text(result.modelSize?.uppercased() ?? "").font(.caption2).foregroundStyle(.secondary) }
                                        Text(result.metrics.map { String(format: "%.1f", $0.qualityScore) } ?? "–").foregroundStyle(result.rank == 1 ? .green : .primary).bold()
                                        Text(percent(result.metrics?.mAP50)); Text(percent(result.metrics?.precision)); Text(percent(result.metrics?.recall)); Text(percent(result.metrics?.f1))
                                        Text(result.meanLatencyMs.map { String(format: "%.0f", $0) } ?? "–")
                                        Text(result.metrics.map { "\($0.falsePositives)/\($0.falseNegatives)" } ?? "–")
                                    }
                                    Divider().gridCellColumns(9)
                                }
                            }
                            .font(.caption)
                        } else {
                            ContentUnavailableView(
                                app.benchmarkReport == nil ? app.tr("Noch kein Vergleich", "No benchmark yet", "Aún no hay comparación", "Aucune comparaison") : app.tr("Referenz geändert", "Ground truth changed", "La referencia ha cambiado", "La référence a changé"),
                                systemImage: "chart.bar",
                                description: Text(app.tr("Lege die richtigen Antworten fest und starte den Vergleich.", "Freeze the correct answers and start the comparison.", "Fija las respuestas correctas e inicia la comparación.", "Figez les bonnes réponses et lancez la comparaison."))
                            )
                            .frame(minHeight: 190)
                        }
                    }
                }
                Text(app.status).font(.callout).foregroundStyle(.secondary)
                if !app.log.isEmpty { Text(app.log).font(.system(.caption, design: .monospaced)).textSelection(.enabled).padding(10).frame(maxWidth: .infinity, alignment: .leading).background(.black.opacity(0.05), in: RoundedRectangle(cornerRadius: 8)) }
            }
            .padding(22)
        }
    }

    private func percent(_ value: Double?) -> String {
        value.map { String(format: "%.1f%%", $0 * 100) } ?? "–"
    }

    private var annotationToolbar: some View {
        HStack {
            Text(app.tr("Objekt markieren:", "Annotate object:")).font(.headline)
            Picker(app.tr("Klasse", "Class"), selection: $app.selectedCategory) {
                ForEach(app.sport.categories, id: \.self) { Text(app.language.category($0)).tag($0) }
            }
            .frame(width: 180)
            Spacer()
            if let frame = app.selectedFrame {
                Text("\(frame.width) × \(frame.height) · \(frame.videoName)")
                    .foregroundStyle(.secondary)
                Button(role: .destructive) { confirmFrameRemoval = true } label: {
                    Label(app.tr("Bild aus Training entfernen", "Remove image from training", "Quitar imagen del entrenamiento", "Retirer l’image de l’entraînement"), systemImage: "trash")
                }
                .disabled(app.isWorking)
            }
        }
    }

    private var trainingPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(app.tr("4 · Modell verbessern", "4 · Improve model")).font(.headline)
                Picker(app.tr("Modell", "Model"), selection: $app.modelSize) {
                    ForEach(ModelSize.allCases) { Text($0.title(language: app.language)).tag($0) }
                }
                .frame(width: 110)
                Stepper(app.tr("\(app.epochs) Epochen", "\(app.epochs) epochs"), value: $app.epochs, in: 1...200)
                    .frame(width: 150)
                Spacer()
                if let hardware = app.hardware {
                    Label(hardwareSummary(hardware), systemImage: hardware.mpsAvailable ? "cpu.fill" : "desktopcomputer")
                        .foregroundStyle(hardware.mpsAvailable ? .green : .secondary)
                        .help(app.tr(
                            "MPS nutzt die Apple-GPU für das Training. Parallele Datenlader halten sie ausgelastet.",
                            "MPS uses the Apple GPU for training. Parallel data loaders keep it supplied."
                        ))
                }
            }

            HStack {
                Button(app.tr("Hardware prüfen", "Check hardware"), action: app.checkHardware)
                Button(app.tr("ML einrichten", "Set up ML"), action: app.prepareEnvironment)
                Button(
                    app.tr(
                        "Automatisch: \(app.language.category(app.selectedCategory))",
                        "Auto-detect: \(app.language.category(app.selectedCategory))"
                    ),
                    action: app.autoLabel
                )
                    .disabled(app.project == nil)
                Button(app.tr("Lokal trainieren", "Train locally"), action: app.train)
                    .buttonStyle(.borderedProminent)
                Divider().frame(height: 22)
                Button(app.tr("CPU-Modell (ONNX)", "CPU model (ONNX)"), action: app.exportCPU)
                Button(app.tr("Apple-Modell (Core ML)", "Apple model (Core ML)"), action: app.exportCoreML)
                Button(app.tr("Paket erstellen", "Exchange package"), action: app.packageModel)
                Button(app.tr("Modell importieren", "Import model"), action: app.importModelPackage)
            }
            .disabled(app.isWorking)

            HStack {
                Text(app.tr("Mindest-Sicherheit", "Minimum confidence"))
                Slider(value: $app.confidenceThreshold, in: 0.05...0.9, step: 0.05)
                    .frame(maxWidth: 220)
                Text("\(Int(app.confidenceThreshold * 100)) %")
                    .font(.system(.caption, design: .monospaced))
            }

            if let packageID = app.activeModelPackageID {
                VStack(alignment: .leading, spacing: 3) {
                    Label(
                        "\(app.tr("Aktives Austauschmodell", "Active exchange model")): \(packageID)",
                        systemImage: "checkmark.shield.fill"
                    )
                    .foregroundStyle(.green)
                    if let description = app.activeModelDescription {
                        Text(description)
                            .foregroundStyle(.secondary)
                    }
                }
                .font(.caption)
            }

            Text(app.tr(
                "Nur Modellpakete aus einer vertrauenswürdigen Quelle importieren.",
                "Import model packages only from a trusted source."
            ))
            .font(.caption2)
            .foregroundStyle(.orange)

            if app.selectedCategory == "player" {
                Text(app.tr(
                    "Hinweis: Das allgemeine Basismodell kennt nur „Person“ und kann Spieler, Schiedsrichter und Zuschauer noch nicht sicher unterscheiden.",
                    "Note: The generic base model only knows “person” and cannot yet reliably distinguish players, referees, and spectators."
                ))
                .font(.caption)
                .foregroundStyle(.orange)
            }

            if app.isWorking {
                ProgressView(value: app.progress == 0 ? nil : app.progress)
            }
            Text(app.status).font(.callout).foregroundStyle(.secondary)
            if !app.log.isEmpty {
                ScrollView {
                    Text(app.log)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 100)
                .padding(8)
                .background(.black.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding(14)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func hardwareSummary(_ hardware: HardwareStatus) -> String {
        var parts = [hardware.recommendedDevice.uppercased()]
        if let memory = hardware.memoryGB { parts.append("\(memory) GB") }
        if let workers = hardware.dataWorkers {
            parts.append(app.tr("\(workers) Loader", "\(workers) loaders"))
        }
        return parts.joined(separator: " · ")
    }

    private var welcome: some View {
        VStack(spacing: 18) {
            Image(systemName: "figure.basketball.circle.fill")
                .font(.system(size: 72))
                .foregroundStyle(.blue.gradient)
            Text(app.tr("Lokales Training für Reco Cam", "Local training for Reco Cam"))
                .font(.largeTitle.bold())
            Text(app.tr(
                "Sportart wählen, Videoordner öffnen und Fehler direkt auf dem Mac korrigieren.\nDie Originalaufnahmen verlassen den Rechner nicht.",
                "Choose a sport, open a video folder, and correct errors directly on your Mac.\nThe original recordings never leave the computer."
            ))
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button(app.tr("Videoordner auswählen", "Select video folder"), action: app.chooseFolder)
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
        }
        .padding(40)
    }
}

private struct OnboardingContainer: View {
    let showLanguagePicker: Bool
    let language: AppLanguage
    @Binding var walkthroughIndex: Int
    let selectLanguage: (AppLanguage) -> Void
    let finish: () -> Void

    @ViewBuilder
    var body: some View {
        if showLanguagePicker {
            LanguageSelectionSheet(select: selectLanguage)
        } else {
            WalkthroughSheet(language: language, index: $walkthroughIndex, finish: finish)
        }
    }
}

private struct WalkthroughSheet: View {
    let language: AppLanguage
    @Binding var index: Int
    let finish: () -> Void

    private var steps: [WalkthroughStep] { language.walkthroughSteps }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack {
                Text(language.text("Schritt", "Step", "Paso", "Étape") + " \(index + 1) / \(steps.count)")
                    .font(.caption.bold())
                    .foregroundStyle(.blue)
                Spacer()
                Button(language.text("Schließen", "Close", "Cerrar", "Fermer"), action: finish)
            }

            Text(language.walkthroughTitle)
                .font(.title.bold())

            HStack(spacing: 5) {
                ForEach(steps.indices, id: \.self) { step in
                    Capsule()
                        .fill(step <= index ? Color.blue : Color.secondary.opacity(0.2))
                        .frame(height: 5)
                }
            }

            HStack(alignment: .top, spacing: 18) {
                Text("\(index + 1)")
                    .font(.title2.bold())
                    .foregroundStyle(.white)
                    .frame(width: 48, height: 48)
                    .background(.blue, in: RoundedRectangle(cornerRadius: 13))
                VStack(alignment: .leading, spacing: 8) {
                    Text(steps[index].title).font(.title3.bold())
                    Text(steps[index].detail)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 130, alignment: .topLeading)
            .padding(20)
            .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))

            HStack {
                Button {
                    index = max(0, index - 1)
                } label: {
                    Label(language.text("Zurück", "Back", "Atrás", "Retour"), systemImage: "arrow.left")
                }
                .disabled(index == 0)
                Spacer()
                Button {
                    if index < steps.count - 1 { index += 1 } else { finish() }
                } label: {
                    Label(
                        index < steps.count - 1
                            ? language.text("Weiter", "Next", "Siguiente", "Suivant")
                            : language.text("Loslegen", "Get started", "Empezar", "Commencer"),
                        systemImage: index < steps.count - 1 ? "arrow.right" : "checkmark"
                    )
                    .labelStyle(.titleAndIcon)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(30)
        .frame(width: 620, height: 440)
    }
}

private struct LanguageSelectionSheet: View {
    let select: (AppLanguage) -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "globe")
                .font(.system(size: 42, weight: .semibold))
                .foregroundStyle(.blue)
            VStack(spacing: 5) {
                Text("Sprache wählen · Choose your language").font(.title2.bold())
                Text("Elige tu idioma · Choisissez votre langue")
                    .foregroundStyle(.secondary)
            }
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                languageButton(.de, "Deutsch")
                languageButton(.en, "English")
                languageButton(.es, "Español")
                languageButton(.fr, "Français")
            }
            Text("Videos and images always remain on this computer.")
                .font(.caption)
                .foregroundStyle(.green)
        }
        .padding(30)
        .frame(width: 540, height: 390)
    }

    private func languageButton(_ language: AppLanguage, _ title: String) -> some View {
        Button {
            select(language)
        } label: {
            HStack {
                Text(title).font(.headline)
                Spacer()
                Text(language.title).font(.caption.bold()).foregroundStyle(.secondary)
                Image(systemName: "arrow.right.circle.fill")
                    .font(.title3)
                    .foregroundStyle(.blue)
            }
            .padding(8)
            .frame(maxWidth: .infinity, minHeight: 48)
        }
        .buttonStyle(.bordered)
    }
}
