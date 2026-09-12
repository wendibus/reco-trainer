import AppKit
import SwiftUI

struct ContentView: View {
    private static var currentRelease: String { AppState.appVersion }

    @EnvironmentObject private var app: AppState
    @AppStorage("recoWalkthroughCompleteV1") private var walkthroughComplete = false
    @AppStorage("recoPreferredLanguageV1") private var preferredLanguage = ""
    @AppStorage("recoLastSeenReleaseV1") private var lastSeenRelease = ""
    @AppStorage("recoDismissedUpdateVersionV1") private var dismissedUpdateVersion = ""
    @AppStorage("recoSkipFrameRemovalConfirmationV1") private var skipFrameRemovalConfirmation = false
    @State private var showWalkthrough = false
    @State private var showLanguagePicker = false
    @State private var walkthroughIndex = 0
    @State private var showBenchmark = false
    @State private var confirmFrameRemoval = false
    @State private var frameIDsPendingRemoval: Set<UUID> = []
    @State private var showReleaseNotes = false
    @State private var modelNameDrafts: [String: String] = [:]
    @State private var modelPendingDeletion: ManagedModelRecord?
    @State private var showModelLibrary = false
    @State private var showPackageNamePrompt = false
    @State private var packageNameDraft = ""
    @State private var showFieldGeometryEditor = false
    @AppStorage("recoDismissedFieldGeometryNudgeV1") private var dismissedFieldGeometryNudge = false

    var body: some View {
        NavigationSplitView {
            sidebar
        } detail: {
            detail
        }
        .frame(minWidth: 1120, minHeight: 720)
        .safeAreaInset(edge: .top, spacing: 0) {
            VStack(spacing: 0) {
                updateBanner
                fieldGeometryNudge
            }
        }
        .alert(app.tr("Hinweis", "Notice"), isPresented: Binding(
            get: { app.errorMessage != nil },
            set: { if !$0 { app.errorMessage = nil } }
        )) {
            Button("OK") { app.errorMessage = nil }
        } message: {
            Text(app.errorMessage ?? "")
        }
        .alert(
            app.tr("Modellpaket benennen", "Name model package", "Nombrar el paquete del modelo", "Nommer le paquet du modèle"),
            isPresented: $showPackageNamePrompt
        ) {
            TextField(app.tr("Paketname", "Package name", "Nombre del paquete", "Nom du paquet"), text: $packageNameDraft)
            Button(app.tr("Paket erstellen", "Create package", "Crear paquete", "Créer le paquet")) {
                let name = packageNameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
                if !name.isEmpty { app.packageModel(name: name) }
            }
            Button(app.tr("Abbrechen", "Cancel", "Cancelar", "Annuler"), role: .cancel) {}
        } message: {
            Text(app.tr(
                "Dieser Name erscheint in der Modellbibliothek und im Dateinamen. Videos oder Bilder werden nicht in das Paket übernommen.",
                "This name appears in the model library and file name. Videos and images are not included in the package.",
                "Este nombre aparece en la biblioteca y en el archivo. El paquete no contiene vídeos ni imágenes.",
                "Ce nom apparaît dans la bibliothèque et le fichier. Le paquet ne contient ni vidéos ni images."
            ))
        }
        .sheet(isPresented: $confirmFrameRemoval) {
            FrameRemovalConfirmationSheet(
                count: frameIDsPendingRemoval.count,
                language: app.language,
                onConfirm: { dontAskAgain in
                    if dontAskAgain { skipFrameRemovalConfirmation = true }
                    app.removeFrames(frameIDsPendingRemoval)
                    confirmFrameRemoval = false
                    frameIDsPendingRemoval = []
                },
                onCancel: {
                    confirmFrameRemoval = false
                    frameIDsPendingRemoval = []
                }
            )
        }
        .confirmationDialog(
            app.tr("Modell dauerhaft löschen?", "Delete model permanently?", "¿Eliminar el modelo permanentemente?", "Supprimer définitivement le modèle ?"),
            isPresented: Binding(get: { modelPendingDeletion != nil }, set: { if !$0 { modelPendingDeletion = nil } }),
            titleVisibility: .visible
        ) {
            Button(app.tr("Modell löschen", "Delete model", "Eliminar modelo", "Supprimer le modèle"), role: .destructive) {
                if let model = modelPendingDeletion { app.deleteModel(model) }
                modelPendingDeletion = nil
            }
            Button(app.tr("Abbrechen", "Cancel", "Cancelar", "Annuler"), role: .cancel) { modelPendingDeletion = nil }
        } message: {
            Text(modelPendingDeletion?.displayName ?? modelPendingDeletion?.packageID ?? "")
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
            app.checkForUpdates()
            if let saved = AppLanguage(rawValue: preferredLanguage) {
                app.language = saved
                if !walkthroughComplete {
                    showWalkthrough = true
                } else if lastSeenRelease != Self.currentRelease {
                    showReleaseNotes = true
                }
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
        .sheet(isPresented: $showReleaseNotes, onDismiss: {
            lastSeenRelease = Self.currentRelease
        }) {
            WhatsNewSheet(language: app.language) {
                lastSeenRelease = Self.currentRelease
                showReleaseNotes = false
            }
        }
        .sheet(isPresented: $showFieldGeometryEditor) {
            if let frame = app.selectedFrame ?? app.project?.frames.first, let store = app.store {
                FieldGeometryEditor(
                    imageURL: store.frameURL(for: frame),
                    frameWidth: frame.width,
                    frameHeight: frame.height,
                    language: app.language,
                    existing: app.project?.fieldGeometry,
                    onSave: { geometry in
                        app.updateFieldGeometry(geometry)
                        showFieldGeometryEditor = false
                    },
                    onCancel: { showFieldGeometryEditor = false }
                )
            }
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

    @ViewBuilder
    private var updateBanner: some View {
        if let update = app.availableUpdate, update.version != dismissedUpdateVersion {
            HStack(spacing: 10) {
                Image(systemName: "arrow.down.circle.fill").foregroundStyle(.blue)
                Text(app.tr(
                    "Neue Version \(update.version) verfügbar.",
                    "New version \(update.version) available.",
                    "Nueva versión \(update.version) disponible.",
                    "Nouvelle version \(update.version) disponible."
                ))
                Spacer()
                Button(app.tr("Herunterladen", "Download", "Descargar", "Télécharger")) {
                    if let url = URL(string: update.url) { NSWorkspace.shared.open(url) }
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                Button(app.tr("Nicht jetzt", "Not now", "Ahora no", "Pas maintenant")) {
                    dismissedUpdateVersion = update.version
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .font(.caption)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.blue.opacity(0.12))
        }
    }

    /// Nudges toward marking field boundaries as soon as there's something to
    /// mark them on, matching the request to ask for this early rather than
    /// leaving it to be discovered. Purely a suggestion - "Spielfeld festlegen"
    /// stays available in the toolbar at any time either way.
    @ViewBuilder
    private var fieldGeometryNudge: some View {
        if let project = app.project, !project.frames.isEmpty, project.fieldGeometry == nil, !dismissedFieldGeometryNudge {
            HStack(spacing: 10) {
                Image(systemName: "sportscourt").foregroundStyle(.green)
                Text(app.tr(
                    "Spielfeld markieren, damit automatisches Markieren nur Personen auf dem Feld berücksichtigt.",
                    "Mark the field boundaries so auto-label only considers people standing on the field.",
                    "Marca los límites del campo para que el marcado automático solo considere a personas dentro del campo.",
                    "Marquez les limites du terrain pour que le marquage automatique ne prenne en compte que les personnes sur le terrain."
                ))
                Spacer()
                Button(app.tr("Spielfeld festlegen", "Set field boundaries", "Definir el campo", "Définir le terrain")) {
                    showFieldGeometryEditor = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                Button(app.tr("Nicht jetzt", "Not now", "Ahora no", "Pas maintenant")) {
                    dismissedFieldGeometryNudge = true
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .font(.caption)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.green.opacity(0.12))
        }
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
                Button {
                    app.startActiveLearning()
                } label: {
                    Label(app.tr("Neue Videos prüfen", "Review new videos", "Revisar vídeos nuevos", "Vérifier de nouvelles vidéos"), systemImage: "sparkles.rectangle.stack")
                }
                .disabled((app.project?.frames.isEmpty ?? true) || app.isWorking)
                if !app.reviewCandidates.isEmpty {
                    Text("\(app.reviewCandidates.count) \(app.tr("Prüfkandidaten", "review candidates", "candidatos", "candidats"))")
                        .font(.caption.bold()).foregroundStyle(.orange)
                }
            }

            if let project = app.project {
                Divider()
                HStack {
                    Text(app.tr("3 · Trainingsbilder", "3 · Training images")).font(.headline)
                    Spacer()
                    if app.selectedFrameIDs.count > 1 {
                        Text("\(app.selectedFrameIDs.count) \(app.tr("ausgewählt", "selected", "seleccionadas", "sélectionnées"))")
                            .font(.caption).foregroundStyle(.secondary)
                        Button(role: .destructive) { requestFrameRemoval(app.selectedFrameIDs) } label: {
                            Image(systemName: "trash")
                        }
                        .disabled(app.isWorking)
                        .help(app.tr("Ausgewählte Trainingsbilder entfernen", "Remove selected training images", "Quitar imágenes de entrenamiento seleccionadas", "Retirer les images d’entraînement sélectionnées"))
                    }
                }
                List(project.frames, selection: $app.selectedFrameIDs) { frame in
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
                        if frame.reviewStatus == "candidate" { Image(systemName: "questionmark.diamond.fill").foregroundStyle(.orange) }
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
        if let purpose = app.localPickerPurpose {
            LocalFilePicker(
                purpose: purpose,
                language: app.language,
                initialURL: app.pickerInitialURL(for: purpose),
                select: app.completeLocalPicker,
                cancel: app.cancelLocalPicker
            )
        } else if showBenchmark {
            benchmarkPanel
        } else if app.selectedFrameIDs.count > 1 {
            multiFrameSelectionPanel
        } else if let frame = app.selectedFrame, let store = app.store {
            VStack(spacing: 14) {
                annotationToolbar
                AnnotationEditor(
                    imageURL: store.frameURL(for: frame),
                    frame: frame,
                    selectedCategory: app.selectedCategory,
                    categories: app.sport.categories,
                    language: app.language
                ) { annotations in
                    app.updateAnnotations(for: frame.id, annotations)
                }
                .id(frame.id)
                .frame(minHeight: 420)
                if frame.reviewStatus == "candidate" { candidateReviewBar }
                trainingPanel
            }
            .padding(18)
        } else {
            welcome
        }
    }

    private var multiFrameSelectionPanel: some View {
        VStack(spacing: 16) {
            Spacer()
            Label(
                "\(app.selectedFrameIDs.count) \(app.tr("Trainingsbilder ausgewählt", "training images selected", "imágenes de entrenamiento seleccionadas", "images d’entraînement sélectionnées"))",
                systemImage: "checklist"
            ).font(.title3.bold())
            Text(app.tr(
                "Wähle mit ⌘-Klick oder Umschalt-Klick weitere Bilder aus, oder klicke ein einzelnes Bild an, um es zu bearbeiten.",
                "Use Cmd-click or Shift-click to select more images, or click a single image to edit it.",
                "Usa Cmd+clic o Mayús+clic para seleccionar más imágenes, o haz clic en una sola para editarla.",
                "Utilisez Cmd-clic ou Maj-clic pour sélectionner d’autres images, ou cliquez sur une seule pour la modifier."
            ))
                .font(.callout).foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 420)
            Button(role: .destructive) { requestFrameRemoval(app.selectedFrameIDs) } label: {
                Label(
                    "\(app.tr("Auswahl entfernen", "Remove selection", "Quitar selección", "Retirer la sélection")) (\(app.selectedFrameIDs.count))",
                    systemImage: "trash"
                )
            }
            .disabled(app.isWorking)
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(18)
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
                                .disabled(app.project == nil || app.isWorking || app.hasUnreviewedTrainingAnnotations)
                            if !app.isWorking && app.hasUnreviewedTrainingAnnotations {
                                Label(app.tr("Automatische Vorschläge zuerst übernehmen, korrigieren oder verwerfen.", "Accept, correct, or reject automatic suggestions first.", "Primero acepta, corrige o rechaza las sugerencias automáticas.", "Acceptez, corrigez ou refusez d’abord les suggestions automatiques."), systemImage: "exclamationmark.triangle")
                                    .font(.caption).foregroundStyle(.orange)
                            }
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
                            if app.isWorking {
                                ProgressView()
                            } else if app.benchmarkGroundTruth == nil {
                                Label(app.tr("Zuerst Schritt 1 abschließen.", "Finish step 1 first.", "Primero completa el paso 1.", "Terminez d’abord l’étape 1."), systemImage: "exclamationmark.triangle")
                                    .font(.caption).foregroundStyle(.orange)
                            } else if app.installedModelCount == 0 {
                                Label(app.tr("Noch kein Modell trainiert oder importiert.", "No model trained or imported yet.", "Aún no se ha entrenado ni importado ningún modelo.", "Aucun modèle entraîné ou importé pour l’instant."), systemImage: "exclamationmark.triangle")
                                    .font(.caption).foregroundStyle(.orange)
                            }
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
                        DisclosureGroup(app.tr(
                            "Was bedeuten die Werte?", "What do the metrics mean?", "¿Qué significan los valores?", "Que signifient les valeurs ?"
                        )) {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(Array(benchmarkMetricExplanations.enumerated()), id: \.offset) { _, item in
                                    HStack(alignment: .top, spacing: 8) {
                                        Text(item.0).bold().frame(width: 92, alignment: .leading)
                                        Text(item.1).foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .font(.caption)
                            .padding(.top, 6)
                        }
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
                            perClassComparison(report: report)
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

    private func requestFrameRemoval(_ ids: Set<UUID>) {
        guard !ids.isEmpty else { return }
        if skipFrameRemovalConfirmation {
            app.removeFrames(ids)
        } else {
            frameIDsPendingRemoval = ids
            confirmFrameRemoval = true
        }
    }

    private func percent(_ value: Double?) -> String {
        value.map { String(format: "%.1f%%", $0 * 100) } ?? "–"
    }

    /// mAP@.50 broken down per category instead of only the overall aggregate, so it's
    /// visible at a glance when one model is strong at one class (e.g. ball) and another
    /// is strong at a different one (e.g. referee) rather than just an overall winner.
    @ViewBuilder
    private func perClassComparison(report: BenchmarkReport) -> some View {
        let successfulResults = report.results.filter { $0.status == "completed" }
        if report.classes.count > 1 && successfulResults.count > 1 {
            Divider().padding(.vertical, 4)
            Text(app.tr(
                "Vergleich je Kategorie (mAP@.50)",
                "Per-category comparison (mAP@.50)",
                "Comparación por categoría (mAP@.50)",
                "Comparaison par catégorie (mAP@.50)"
            )).font(.caption.bold())
            Text(app.tr(
                "Zeigt z. B., ob ein Modell besser bei Bällen und ein anderes besser bei Schiedsrichtern ist. Bestwert je Zeile grün.",
                "Shows e.g. whether one model is better at balls and another better at referees. Best value per row in green.",
                "Muestra, por ejemplo, si un modelo es mejor con los balones y otro con los árbitros. El mejor valor de cada fila en verde.",
                "Montre par exemple si un modèle est meilleur pour les ballons et un autre pour les arbitres. Meilleure valeur de chaque ligne en vert."
            )).font(.caption2).foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                    GridRow {
                        Text(app.tr("Kategorie", "Category", "Categoría", "Catégorie")).bold().frame(width: 90, alignment: .leading)
                        ForEach(successfulResults) { result in
                            Text(result.packageID).bold().lineLimit(1).frame(width: 120, alignment: .leading)
                        }
                    }
                    Divider().gridCellColumns(successfulResults.count + 1)
                    ForEach(report.classes, id: \.self) { category in
                        let bestValue = successfulResults.compactMap { $0.metrics?.perClass[category]?.ap50 }.max()
                        GridRow {
                            Text(app.language.category(category)).frame(width: 90, alignment: .leading)
                            ForEach(successfulResults) { result in
                                let value = result.metrics?.perClass[category]?.ap50
                                let isBest = value != nil && bestValue != nil && value! == bestValue!
                                Text(percent(value))
                                    .foregroundStyle(isBest ? .green : .primary)
                                    .bold(isBest)
                                    .frame(width: 120, alignment: .leading)
                            }
                        }
                    }
                }
                .font(.caption)
            }
        }
    }

    private var benchmarkMetricExplanations: [(String, String)] {
        [
            (app.tr("Qualität", "Quality", "Calidad", "Qualité"), app.tr("Gesamtrang: 70 % mAP@0.50 und 30 % F1. Höher ist besser.", "Overall ranking: 70% mAP@0.50 and 30% F1. Higher is better.", "Clasificación total: 70 % mAP@0.50 y 30 % F1. Un valor mayor es mejor.", "Classement global : 70 % mAP@0.50 et 30 % F1. Plus élevé est meilleur.")),
            ("mAP@.50", app.tr("Misst die Erkennung über viele Sicherheitsschwellen. Eine Box zählt ab 50 % Überlappung als richtig.", "Measures detection across many confidence levels. A box counts as correct from 50% overlap.", "Mide la detección con muchos niveles de confianza. Un cuadro cuenta como correcto desde un 50 % de solapamiento.", "Mesure la détection à plusieurs niveaux de confiance. Une boîte est correcte dès 50 % de chevauchement.")),
            (app.tr("Präzision", "Precision", "Precisión", "Précision"), app.tr("Anteil der gemeldeten Treffer, die wirklich richtig sind. Hoch bedeutet wenige Fehlalarme.", "Share of reported detections that are correct. High means fewer false alarms.", "Proporción de detecciones notificadas que son correctas. Un valor alto significa menos falsas alarmas.", "Part des détections signalées qui sont correctes. Une valeur élevée signifie moins de fausses alertes.")),
            ("Recall", app.tr("Anteil der vorhandenen Objekte, die gefunden wurden. Hoch bedeutet weniger übersehene Bälle.", "Share of real objects that were found. High means fewer missed balls.", "Proporción de objetos reales encontrados. Un valor alto significa menos balones omitidos.", "Part des objets réels trouvés. Une valeur élevée signifie moins de ballons manqués.")),
            ("F1", app.tr("Gemeinsamer Ausgleich von Präzision und Recall.", "Balance of precision and recall.", "Equilibrio entre precisión y cobertura.", "Équilibre précision et rappel.")),
            ("FP / FN", app.tr("Falsche Treffer / übersehene echte Objekte. Bei beiden ist weniger besser.", "False detections / missed real objects. Lower is better for both.", "Detecciones falsas / objetos reales omitidos. Menos es mejor.", "Fausses détections / objets réels manqués. Moins est meilleur.")),
            (app.tr("Zeit/Bild", "Time/image", "Tiempo/imagen", "Temps/image"), app.tr("Mittlere Rechenzeit pro Bild. Niedriger ist schneller.", "Average processing time per image. Lower is faster.", "Tiempo medio por imagen. Un valor menor es más rápido.", "Temps moyen par image. Plus bas est plus rapide.")),
            (app.tr("Schwelle", "Threshold", "Umbral", "Seuil"), app.tr("Minimale Sicherheit. Niedriger findet mehr, erzeugt aber meist mehr Fehlalarme.", "Minimum confidence. Lower finds more but usually creates more false alarms.", "Confianza mínima. Un valor menor encuentra más, pero suele producir más falsas alarmas.", "Confiance minimale. Plus bas trouve davantage, mais produit souvent plus de fausses alertes.")),
        ]
    }

    private var annotationToolbar: some View {
        HStack {
            Text(app.tr("Objekt markieren:", "Annotate object:")).font(.headline)
            Picker(app.tr("Klasse", "Class"), selection: $app.selectedCategory) {
                ForEach(app.sport.categories, id: \.self) { Text(app.language.category($0)).tag($0) }
            }
            .frame(width: 180)
            Spacer()
            if app.automaticAnnotationCountOnSelectedFrame > 0 {
                Label("\(app.automaticAnnotationCountOnSelectedFrame)", systemImage: "sparkles")
                    .foregroundStyle(.orange)
                    .help(app.tr("Automatische Vorschläge auf diesem Bild", "Automatic suggestions on this image", "Sugerencias automáticas en esta imagen", "Suggestions automatiques sur cette image"))
                Button(app.tr("Übernehmen", "Accept auto", "Aceptar auto", "Accepter auto")) { app.acceptAllAutomaticAnnotations() }
                    .tint(.green)
                Button(app.tr("Verwerfen", "Reject auto", "Rechazar auto", "Rejeter auto"), role: .destructive) { app.rejectAllAutomaticAnnotations() }
            }
            if let frame = app.selectedFrame {
                Text("\(frame.width) × \(frame.height) · \(frame.videoName)")
                    .foregroundStyle(.secondary)
                Button(role: .destructive) { requestFrameRemoval([frame.id]) } label: {
                    Label(app.tr("Bild aus Training entfernen", "Remove image from training", "Quitar imagen del entrenamiento", "Retirer l’image de l’entraînement"), systemImage: "trash")
                }
                .disabled(app.isWorking)
            }
        }
    }

    private var candidateReviewBar: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(app.tr("Ball oder kein Ball?", "Ball or no ball?", "¿Balón o no?", "Ballon ou non ?")).font(.headline)
                Text(app.tr("Box prüfen oder korrigieren. Nur bestätigte Bilder gelangen ins Training.", "Review or correct the box. Only confirmed images enter training.", "Revisa o corrige el cuadro. Solo se entrenan imágenes confirmadas.", "Vérifiez ou corrigez la boîte. Seules les images confirmées sont entraînées.")).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button(app.tr("Ball", "Ball", "Balón", "Ballon")) { app.reviewSelectedCandidate(asBall: true) }.buttonStyle(.borderedProminent).tint(.green)
            Button(app.tr("Kein Ball", "No ball", "No es balón", "Pas de ballon")) { app.reviewSelectedCandidate(asBall: false) }.buttonStyle(.bordered)
            Button(app.tr("Überspringen", "Skip", "Omitir", "Ignorer"), role: .destructive) { app.removeSelectedFrame() }
        }
        .padding(10)
        .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
    }

    private var trainingPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(app.tr("4 · Modell verbessern", "4 · Improve model")).font(.headline)
                Picker(app.tr("Modell", "Model"), selection: $app.modelSize) {
                    ForEach(ModelSize.allCases) { Text($0.title(language: app.language)).tag($0) }
                }
                .fixedSize()
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

            HStack(spacing: 6) {
                Text(app.tr("Automatisch markieren:", "Auto-label:", "Marcado automático:", "Marquage automatique :"))
                    .font(.subheadline).foregroundStyle(.secondary)
                ForEach(app.sport.categories, id: \.self) { category in
                    autoLabelCategoryChip(category)
                }
            }

            if let classes = app.project?.classes, classes.count > 1 {
                HStack(spacing: 6) {
                    Text(app.tr("Training beschränken auf:", "Restrict training to:", "Restringir entrenamiento a:", "Limiter l’entraînement à :"))
                        .font(.subheadline).foregroundStyle(.secondary)
                    ForEach(classes, id: \.self) { category in
                        trainingCategoryChip(category)
                    }
                }
                .help(app.tr(
                    "Abgewählte Klassen werden bei diesem Trainingslauf komplett ausgeschlossen - keine Garantie, dass ihre bisherige Erkennungsqualität dadurch unverändert bleibt.",
                    "Deselected classes are entirely excluded from this training run - no guarantee that their existing detection quality stays unchanged as a result.",
                    "Las clases no seleccionadas quedan totalmente excluidas de este entrenamiento - no hay garantía de que su calidad de detección actual permanezca igual.",
                    "Les classes désélectionnées sont entièrement exclues de cet entraînement - aucune garantie que leur qualité de détection actuelle reste inchangée."
                ))
            }

            HStack {
                Button(app.tr("Hardware prüfen", "Check hardware"), action: app.checkHardware)
                Button(app.tr("ML einrichten", "Set up ML"), action: app.prepareEnvironment)
                Button(
                    app.project?.fieldGeometry == nil
                        ? app.tr("Spielfeld festlegen", "Set field boundaries", "Definir el campo", "Définir le terrain")
                        : app.tr("Spielfeld bearbeiten", "Edit field boundaries", "Editar el campo", "Modifier le terrain")
                ) { showFieldGeometryEditor = true }
                    .disabled(app.project == nil || (app.selectedFrame ?? app.project?.frames.first) == nil)
                    .help(app.tr(
                        "Markiere die vier Eckpunkte des Spielfelds, damit „Automatisch markieren“ nur Personen berücksichtigt, die mit den Füßen auf dem Feld stehen.",
                        "Mark the field's four corners so \"Auto-label\" only considers people whose feet are standing on the field.",
                        "Marca las cuatro esquinas del campo para que «Marcado automático» solo considere a personas con los pies sobre el campo.",
                        "Marquez les quatre coins du terrain pour que « Marquage automatique » ne prenne en compte que les personnes ayant les pieds sur le terrain."
                    ))
                Button(
                    app.tr("Automatisch markieren", "Auto-label"),
                    action: app.autoLabel
                )
                    .disabled(app.project == nil || app.autoLabelCategories.isEmpty)
                Button(
                    app.tr(
                        "Boxen mit OpenCV verfeinern",
                        "Refine boxes with OpenCV",
                        "Refinar cuadros con OpenCV",
                        "Affiner les boîtes avec OpenCV"
                    ),
                    action: app.refineBoxes
                )
                    .disabled(app.project == nil)
                    .help(app.tr(
                        "Zieht Ball-, Spieler-, Schiedsrichter- und weitere unterstützte Boxen lokal nach, auch von Hand gezeichnete. Nur plausible, eng anliegende Anpassungen werden übernommen; die ursprünglichen Koordinaten bleiben je Box gespeichert.",
                        "Locally tightens ball, player, referee, and other supported boxes, including hand-drawn ones. Only plausible, closely-matching adjustments are applied; each box's original coordinates remain stored.",
                        "Ajusta localmente cuadros de balón, jugador, árbitro y otras clases compatibles, incluidos los dibujados a mano. Solo se aplican ajustes plausibles y cercanos; las coordenadas originales de cada cuadro permanecen guardadas.",
                        "Resserre localement les boîtes ballon, joueur, arbitre et autres classes prises en charge, y compris dessinées à la main. Seuls les ajustements plausibles et proches sont appliqués ; les coordonnées d’origine de chaque boîte restent enregistrées."
                    ))
                Button(
                    app.continuationCheckpointName == nil
                        ? app.tr("Lokal trainieren", "Train locally", "Entrenar localmente", "Entraîner localement")
                        : app.tr("Letztes Modell weitertrainieren", "Continue latest model", "Continuar el último modelo", "Continuer le dernier modèle"),
                    action: app.train
                )
                    .buttonStyle(.borderedProminent)
                Divider().frame(height: 22)
                Button(app.tr("CPU-Modell (ONNX)", "CPU model (ONNX)"), action: app.exportCPU)
                Button(app.tr("Apple-Modell (Core ML)", "Apple model (Core ML)"), action: app.exportCoreML)
                Button(app.tr("Paket erstellen", "Exchange package")) {
                    packageNameDraft = "\(app.sport.title(language: app.language)) · \(app.modelSize.title(language: app.language))"
                    showPackageNamePrompt = true
                }
                Button(app.tr("Modell importieren", "Import model"), action: app.importModelPackage)
            }
            .disabled(app.isWorking)

            if app.selectedFolder != nil {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Button(action: app.revealTrainingFolder) {
                        Label(
                            app.tr("Trainingsordner anzeigen", "Show training folder", "Mostrar carpeta de entrenamiento", "Afficher le dossier d’entraînement"),
                            systemImage: "folder"
                        )
                    }
                    Text("frames · dataset · runs · exports · backups · benchmarks · models · inbox")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                if let checkpoint = app.continuationCheckpointName {
                    Text(app.tr(
                        "Das nächste Training setzt automatisch bei \(checkpoint) fort.",
                        "The next training run automatically continues from \(checkpoint).",
                        "El próximo entrenamiento continúa automáticamente desde \(checkpoint).",
                        "Le prochain entraînement reprend automatiquement depuis \(checkpoint)."
                    ))
                    .font(.caption2)
                    .foregroundStyle(.green)
                }
            }

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

            modelLibraryPanel

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

    private var modelLibraryPanel: some View {
        GroupBox {
            DisclosureGroup(isExpanded: $showModelLibrary) {
                if app.managedModels.isEmpty {
                    Text(app.tr(
                    "Nach dem nächsten Training wird jeder Modellstand automatisch hier archiviert.",
                    "Every model revision will be archived here automatically after the next training run.",
                    "Cada versión del modelo se archivará aquí automáticamente tras el próximo entrenamiento.",
                    "Chaque version du modèle sera automatiquement archivée ici après le prochain entraînement."
                    ))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                } else {
                    ScrollView {
                        VStack(spacing: 8) {
                            ForEach(app.managedModels) { model in
                            HStack(alignment: .center, spacing: 10) {
                                VStack(alignment: .leading, spacing: 3) {
                                    HStack(spacing: 5) {
                                        Text(model.displayName ?? model.packageID).font(.caption.bold()).lineLimit(1)
                                        if model.isActive == true { Text(app.tr("AKTIV", "ACTIVE", "ACTIVO", "ACTIF")).font(.system(size: 9, weight: .bold)).foregroundStyle(.green) }
                                        if model.isBest == true { Text(app.tr("BESTES", "BEST", "MEJOR", "MEILLEUR")).font(.system(size: 9, weight: .bold)).foregroundStyle(.blue) }
                                    }
                                    Text("\(model.modelSize.uppercased()) · \(model.comparisonScore.map { String(format: model.testScore == nil ? "Val mAP %.4f" : "Test mAP %.4f", $0) } ?? "mAP –") · \(model.createdAt ?? "")")
                                        .font(.system(.caption2, design: .monospaced))
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    Text(model.packageID).font(.system(size: 9, design: .monospaced)).foregroundStyle(.tertiary).lineLimit(1)
                                }
                                Spacer(minLength: 6)
                                TextField(
                                    app.tr("Eigener Name", "Custom name", "Nombre propio", "Nom personnalisé"),
                                    text: Binding(
                                        get: { modelNameDrafts[model.packageID] ?? "" },
                                        set: { modelNameDrafts[model.packageID] = $0 }
                                    )
                                )
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 145)
                                Button(app.tr("Umbenennen", "Rename", "Renombrar", "Renommer")) {
                                    let name = modelNameDrafts[model.packageID] ?? ""
                                    app.renameModel(model, to: name)
                                    modelNameDrafts[model.packageID] = nil
                                }
                                .disabled((modelNameDrafts[model.packageID] ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || app.isWorking)
                                Button(app.tr("Aktivieren", "Activate", "Activar", "Activer")) { app.activateModel(model) }
                                    .disabled(model.isActive == true || app.isWorking)
                                Button(role: .destructive) { modelPendingDeletion = model } label: { Image(systemName: "trash") }
                                    .disabled(model.isActive == true || app.isWorking)
                            }
                            .padding(7)
                            .background(model.isActive == true ? Color.green.opacity(0.08) : Color.secondary.opacity(0.06), in: RoundedRectangle(cornerRadius: 8))
                            }
                        }
                    }
                    .frame(maxHeight: 205)
                }
            } label: {
                Label(
                    "\(app.tr("Modellverwaltung", "Model library", "Biblioteca de modelos", "Bibliothèque de modèles")) (\(app.managedModels.count))",
                    systemImage: "shippingbox.and.arrow.backward"
                )
                .font(.caption.bold())
            }
        }
    }

    /// Categories the generic COCO "person" class covers visually but that a sport
    /// schema still tracks separately, and that have no fallback of their own
    /// (unlike "player", generic base-model detection, and "referee", the clothing
    /// heuristic) - the disabled-chip tooltip points at the "select player, then
    /// relabel by hand" shortcut instead of just saying "unsupported".
    private static let personShapedCategories: Set<String> = ["goalkeeper"]

    @ViewBuilder
    private func autoLabelCategoryChip(_ category: String) -> some View {
        let isSelected = app.autoLabelCategories.contains(category)
        let isSupported = app.autoLabelSupportedCategories.contains(category)
        let toggle = {
            if isSelected {
                app.autoLabelCategories.remove(category)
            } else {
                app.autoLabelCategories.insert(category)
            }
        }
        let refereeHelp = category == "referee" ? app.tr(
            "Personen mit typischer Schiedsrichter-Kleidung für diese Sportart werden automatisch als „Schiedsrichter“ vorgeschlagen (weiterhin zur Prüfung markiert, keine Garantie). Dafür muss „Spieler“ ebenfalls ausgewählt sein.",
            "People wearing attire typical for this sport's referees are suggested automatically as \"referee\" (still flagged for review, not a guarantee). This requires \"player\" to be selected too.",
            "Las personas con ropa típica de árbitro para este deporte se sugieren automáticamente como «árbitro» (siguen marcadas para revisión, no es una garantía). Para ello, «jugador» también debe estar seleccionado.",
            "Les personnes portant une tenue typique d’arbitre pour ce sport sont suggérées automatiquement comme « arbitre » (toujours signalées pour vérification, ce n’est pas une garantie). Cela nécessite que « joueur » soit également sélectionné."
        ) : nil
        if isSelected {
            Button(app.language.category(category), action: toggle)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .help(refereeHelp ?? "")
        } else {
            Button(app.language.category(category), action: toggle)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(!isSupported)
                .opacity(isSupported ? 1 : 0.4)
                .help(isSupported ? (refereeHelp ?? "") : (
                    Self.personShapedCategories.contains(category)
                        ? app.tr(
                            "Das Basismodell erkennt diese Klasse nicht einzeln, aber jede Person allgemein als „Spieler“. Markiere mit „Spieler“ automatisch, klicke dann einzelne \(app.language.category(category))-Boxen an und ordne sie per Klick um.",
                            "The base model can't tell this class apart on its own, but it does detect every person generically as \"player\". Auto-label with \"player\", then click individual \(app.language.category(category)) boxes to relabel them.",
                            "El modelo base no distingue esta clase por sí solo, pero detecta a toda persona genéricamente como \"jugador\". Marca automáticamente con \"jugador\" y luego haz clic en los cuadros de \(app.language.category(category)) para reasignarlos.",
                            "Le modèle de base ne distingue pas cette classe à lui seul, mais détecte chaque personne génériquement comme « joueur ». Marquez automatiquement avec « joueur », puis cliquez sur les boîtes de \(app.language.category(category)) pour les réattribuer."
                        )
                        : app.tr(
                            "Das allgemeine Basismodell kennt diese Klasse nicht. Zuerst manuell markieren und ein eigenes Modell trainieren.",
                            "The generic base model does not know this class. Annotate it manually first and train a custom model.",
                            "El modelo base general no conoce esta clase. Anótala manualmente primero y entrena un modelo propio.",
                            "Le modèle de base générique ne connaît pas cette classe. Annotez-la d’abord manuellement puis entraînez un modèle personnalisé."
                        )
                ))
        }
    }

    @ViewBuilder
    private func trainingCategoryChip(_ category: String) -> some View {
        let isIncluded = !app.trainingExcludedCategories.contains(category)
        let toggle = {
            if isIncluded {
                app.trainingExcludedCategories.insert(category)
            } else {
                app.trainingExcludedCategories.remove(category)
            }
        }
        if isIncluded {
            Button(app.language.category(category), action: toggle)
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
        } else {
            Button(app.language.category(category), action: toggle)
                .buttonStyle(.bordered)
                .controlSize(.small)
                .opacity(0.5)
        }
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

private struct LocalFileEntry: Identifiable, Sendable {
    let url: URL
    let isDirectory: Bool
    var id: String { url.path }
}

private struct LocalDirectorySnapshot: Sendable {
    let entries: [LocalFileEntry]
    let error: String?
}

private struct LocalFilePicker: View {
    let purpose: LocalPickerPurpose
    let language: AppLanguage
    let select: (URL) -> Void
    let cancel: () -> Void

    @State private var currentURL: URL
    @State private var pathText: String
    @State private var entries: [LocalFileEntry] = []
    @State private var selectedFile: URL?
    @State private var isLoading = false
    @State private var loadError: String?

    init(
        purpose: LocalPickerPurpose,
        language: AppLanguage,
        initialURL: URL,
        select: @escaping (URL) -> Void,
        cancel: @escaping () -> Void
    ) {
        self.purpose = purpose
        self.language = language
        self.select = select
        self.cancel = cancel
        let normalized = initialURL.standardizedFileURL
        _currentURL = State(initialValue: normalized)
        _pathText = State(initialValue: normalized.path)
    }

    private var title: String {
        switch purpose {
        case .trainingFolder:
            language.text("Ordner mit Sportvideos auswählen", "Select folder containing sports videos", "Seleccionar carpeta con vídeos deportivos", "Sélectionner le dossier des vidéos sportives")
        case .activeLearningFolder:
            language.text("Ordner mit neuen Videos auswählen", "Select folder with new videos", "Seleccionar carpeta con vídeos nuevos", "Sélectionner le dossier des nouvelles vidéos")
        case .modelPackage:
            language.text("Reco-Modellpaket auswählen", "Select Reco model package", "Seleccionar paquete de modelo Reco", "Sélectionner le paquet de modèle Reco")
        }
    }

    private var shortcuts: [(String, String, URL)] {
        let manager = FileManager.default
        return [
            (language.text("Laufwerke", "Drives", "Unidades", "Volumes"), "externaldrive", URL(filePath: "/Volumes", directoryHint: .isDirectory)),
            (language.text("Persönlicher Ordner", "Home", "Carpeta personal", "Dossier personnel"), "house", manager.homeDirectoryForCurrentUser),
            (language.text("Filme", "Movies", "Películas", "Films"), "film", manager.urls(for: .moviesDirectory, in: .userDomainMask).first ?? manager.homeDirectoryForCurrentUser),
            (language.text("Downloads", "Downloads", "Descargas", "Téléchargements"), "arrow.down.circle", manager.urls(for: .downloadsDirectory, in: .userDomainMask).first ?? manager.homeDirectoryForCurrentUser),
            (language.text("Schreibtisch", "Desktop", "Escritorio", "Bureau"), "desktopcomputer", manager.urls(for: .desktopDirectory, in: .userDomainMask).first ?? manager.homeDirectoryForCurrentUser)
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Label(title, systemImage: purpose == .modelPackage ? "shippingbox" : "folder")
                    .font(.title2.bold())
                Spacer()
                Button(language.text("Abbrechen", "Cancel", "Cancelar", "Annuler"), action: cancel)
            }

            Text(language.text(
                "Dieser lokale Browser ersetzt den blockierenden macOS-Dateidialog. Es werden keine Dateien hochgeladen.",
                "This local browser replaces the blocking macOS file dialog. No files are uploaded.",
                "Este navegador local sustituye al diálogo bloqueado de macOS. No se carga ningún archivo.",
                "Ce navigateur local remplace la boîte de dialogue macOS bloquante. Aucun fichier n’est envoyé."
            ))
            .font(.callout)
            .foregroundStyle(.secondary)

            HStack(spacing: 8) {
                Button {
                    navigate(to: currentURL.deletingLastPathComponent())
                } label: {
                    Label(language.text("Zurück", "Up", "Subir", "Parent"), systemImage: "arrow.up")
                }
                .disabled(currentURL.path == "/")

                TextField(language.text("Ordnerpfad", "Folder path", "Ruta de carpeta", "Chemin du dossier"), text: $pathText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { navigate(to: URL(filePath: pathText, directoryHint: .isDirectory)) }
                Button(language.text("Öffnen", "Open", "Abrir", "Ouvrir")) {
                    navigate(to: URL(filePath: pathText, directoryHint: .isDirectory))
                }
            }

            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(shortcuts, id: \.2.path) { shortcut in
                        Button {
                            navigate(to: shortcut.2)
                        } label: {
                            Label(shortcut.0, systemImage: shortcut.1)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .buttonStyle(.plain)
                        .padding(.vertical, 5)
                    }
                    Spacer()
                }
                .frame(width: 170)
                .padding(10)
                .background(.quaternary.opacity(0.45), in: RoundedRectangle(cornerRadius: 10))

                Group {
                    if isLoading {
                        VStack(spacing: 12) {
                            ProgressView()
                            Text(language.text("Ordner wird lokal gelesen …", "Reading folder locally …", "Leyendo carpeta localmente …", "Lecture locale du dossier …"))
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if let loadError {
                        ContentUnavailableView(
                            language.text("Ordner nicht verfügbar", "Folder unavailable", "Carpeta no disponible", "Dossier indisponible"),
                            systemImage: "exclamationmark.folder",
                            description: Text(loadError)
                        )
                    } else if entries.isEmpty {
                        ContentUnavailableView(
                            language.text("Keine Unterordner", "No subfolders", "Sin subcarpetas", "Aucun sous-dossier"),
                            systemImage: "folder",
                            description: Text(purpose == .modelPackage
                                ? language.text("Keine .recomodel-Datei in diesem Ordner.", "No .recomodel file in this folder.", "No hay ningún archivo .recomodel.", "Aucun fichier .recomodel dans ce dossier.")
                                : language.text("Dieser Ordner kann trotzdem ausgewählt werden.", "This folder can still be selected.", "Esta carpeta se puede seleccionar igualmente.", "Ce dossier peut tout de même être sélectionné."))
                        )
                    } else {
                        List(entries, selection: $selectedFile) { entry in
                            Button {
                                if entry.isDirectory {
                                    navigate(to: entry.url)
                                } else {
                                    selectedFile = entry.url
                                }
                            } label: {
                                Label(entry.url.lastPathComponent, systemImage: entry.isDirectory ? "folder.fill" : "shippingbox.fill")
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .buttonStyle(.plain)
                            .tag(entry.isDirectory ? nil as URL? : entry.url)
                        }
                        .listStyle(.inset)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }

            HStack {
                Image(systemName: "lock.fill").foregroundStyle(.green)
                Text(currentURL.path).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                Spacer()
                Button(selectionTitle) {
                    if purpose == .modelPackage {
                        if let selectedFile { select(selectedFile) }
                    } else {
                        select(currentURL)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(purpose == .modelPackage && selectedFile == nil)
            }
        }
        .padding(22)
        .task(id: currentURL) { await loadCurrentDirectory() }
    }

    private var selectionTitle: String {
        purpose == .modelPackage
            ? language.text("Modell importieren", "Import model", "Importar modelo", "Importer le modèle")
            : language.text("Diesen Ordner verwenden", "Use this folder", "Usar esta carpeta", "Utiliser ce dossier")
    }

    private func navigate(to url: URL) {
        let normalized = url.standardizedFileURL
        selectedFile = nil
        pathText = normalized.path
        currentURL = normalized
    }

    @MainActor
    private func loadCurrentDirectory() async {
        let requestedURL = currentURL
        isLoading = true
        loadError = nil
        let requestedPurpose = purpose
        let snapshot = await Task.detached(priority: .userInitiated) {
            Self.readDirectory(at: requestedURL, purpose: requestedPurpose)
        }.value
        guard currentURL == requestedURL else { return }
        entries = snapshot.entries
        loadError = snapshot.error
        isLoading = false
    }

    nonisolated private static func readDirectory(at url: URL, purpose: LocalPickerPurpose) -> LocalDirectorySnapshot {
        do {
            let urls = try FileManager.default.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            )
            let entries = urls.compactMap { candidate -> LocalFileEntry? in
                let isDirectory = (try? candidate.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if isDirectory { return LocalFileEntry(url: candidate, isDirectory: true) }
                guard purpose == .modelPackage, candidate.pathExtension.lowercased() == "recomodel" else { return nil }
                return LocalFileEntry(url: candidate, isDirectory: false)
            }.sorted {
                if $0.isDirectory != $1.isDirectory { return $0.isDirectory }
                return $0.url.lastPathComponent.localizedStandardCompare($1.url.lastPathComponent) == .orderedAscending
            }
            return LocalDirectorySnapshot(entries: entries, error: nil)
        } catch {
            return LocalDirectorySnapshot(entries: [], error: error.localizedDescription)
        }
    }
}

private struct FrameRemovalConfirmationSheet: View {
    let count: Int
    let language: AppLanguage
    let onConfirm: (Bool) -> Void
    let onCancel: () -> Void

    @State private var dontAskAgain = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(count == 1
                ? language.text("Trainingsbild entfernen?", "Remove training image?", "¿Quitar imagen de entrenamiento?", "Retirer l’image d’entraînement ?")
                : language.text(
                    "\(count) Trainingsbilder entfernen?", "Remove \(count) training images?",
                    "¿Quitar \(count) imágenes de entrenamiento?", "Retirer \(count) images d’entraînement ?"
                )
            ).font(.headline)
            Text(count == 1
                ? language.text(
                    "Nur der extrahierte Trainingsframe wird entfernt. Das Quellvideo bleibt unverändert.",
                    "Only the extracted training frame is removed. The source video remains unchanged.",
                    "Solo se quita el fotograma extraído. El vídeo original no se modifica.",
                    "Seule l’image extraite est retirée. La vidéo source reste inchangée."
                )
                : language.text(
                    "Nur die extrahierten Trainingsframes werden entfernt. Die Quellvideos bleiben unverändert.",
                    "Only the extracted training frames are removed. The source videos remain unchanged.",
                    "Solo se quitan los fotogramas extraídos. Los vídeos originales no se modifican.",
                    "Seules les images extraites sont retirées. Les vidéos sources restent inchangées."
                )
            ).font(.callout).foregroundStyle(.secondary)
            Toggle(
                language.text("Nicht mehr fragen", "Don't ask again", "No volver a preguntar", "Ne plus demander"),
                isOn: $dontAskAgain
            )
            HStack {
                Spacer()
                Button(language.text("Abbrechen", "Cancel", "Cancelar", "Annuler"), action: onCancel)
                Button(role: .destructive) {
                    onConfirm(dontAskAgain)
                } label: {
                    Text(count == 1
                        ? language.text("Bild entfernen", "Remove image", "Quitar imagen", "Retirer l’image")
                        : language.text("Entfernen", "Remove", "Quitar", "Retirer")
                    )
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(24)
        .frame(width: 420)
    }
}

private struct WhatsNewSheet: View {
    let language: AppLanguage
    let dismiss: () -> Void

    private var changes: [(String, String)] {
        [
            (
                language.text("Schiedsrichter werden per Kleidung automatisch vorgeschlagen", "Referees are automatically suggested by their clothing", "Los árbitros se sugieren automáticamente por su vestimenta", "Les arbitres sont automatiquement suggérés grâce à leur tenue"),
                language.text("Der „Schiedsrichter“-Chip bei „Automatisch markieren“ ist jetzt für jede Sportart nutzbar, auch ohne eigens trainiertes Modell: Personen mit typischer Schiedsrichter-Kleidung (z. B. komplett Schwarz im Fußball, Grau/Schwarz im Basketball, Schwarz-Weiß gestreift im Eishockey) werden automatisch als „Schiedsrichter“ vorgeschlagen - weiterhin zur Prüfung markiert, keine Garantie. Setzt voraus, dass „Spieler“ ebenfalls ausgewählt ist.", "The \"referee\" chip for auto-label now works for every sport, even without a custom-trained model: people wearing attire typical for that sport's referees (e.g. all black in football, grey/black in basketball, black-and-white stripes in ice hockey) are automatically suggested as \"referee\" - still flagged for review, not a guarantee. Requires \"player\" to be selected too.", "El chip «árbitro» del marcado automático ahora funciona en todos los deportes, incluso sin un modelo entrenado propio: las personas con vestimenta típica de árbitro para ese deporte (p. ej. todo negro en fútbol, gris/negro en baloncesto, rayas blanco y negro en hockey sobre hielo) se sugieren automáticamente como «árbitro» - siguen marcadas para revisión, no es una garantía. Requiere que «jugador» también esté seleccionado.", "La puce « arbitre » du marquage automatique fonctionne désormais pour tous les sports, même sans modèle personnalisé entraîné : les personnes portant une tenue typique des arbitres de ce sport (p. ex. tout en noir en football, gris/noir en basketball, rayures noir et blanc en hockey sur glace) sont automatiquement suggérées comme « arbitre » - toujours signalées pour vérification, ce n’est pas une garantie. Nécessite que « joueur » soit également sélectionné.")
            )
        ]
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            Label(language.text("Neu in Reco Trainer", "What’s new in Reco Trainer", "Novedades de Reco Trainer", "Nouveautés de Reco Trainer"), systemImage: "sparkles")
                .font(.title.bold())
                .foregroundStyle(.blue)
            ForEach(Array(changes.enumerated()), id: \.offset) { _, change in
                VStack(alignment: .leading, spacing: 5) {
                    Text(change.0).font(.headline)
                    Text(change.1).foregroundStyle(.secondary)
                }
            }
            Text(language.text("Videos und Trainingsbilder bleiben weiterhin vollständig auf diesem Rechner.", "Videos and training images continue to remain entirely on this computer.", "Los vídeos y las imágenes siguen permaneciendo por completo en este equipo.", "Les vidéos et images d’entraînement restent entièrement sur cet ordinateur."))
                .font(.caption)
                .foregroundStyle(.green)
            HStack {
                Spacer()
                Button(language.text("Verstanden", "Got it", "Entendido", "Compris"), action: dismiss)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(30)
        .frame(width: 640)
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
