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
    @State private var showBallTracking = false
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
                    if showBenchmark { showBallTracking = false }
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
                    showBallTracking.toggle()
                    if showBallTracking { showBenchmark = false }
                } label: {
                    Label(
                        showBallTracking
                            ? app.tr("Training", "Training", "Entrenamiento", "Entraînement")
                            : app.tr("Balltracking simulieren", "Simulate ball tracking", "Simular seguimiento del balón", "Simuler le suivi du ballon"),
                        systemImage: showBallTracking ? "arrow.left" : "scope"
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
                Button {
                    app.startIndependentValidation()
                } label: {
                    Label(app.tr("Unabhängiger Modelltest", "Independent model test", "Prueba de modelo independiente", "Test de modèle indépendant"), systemImage: "checkmark.shield")
                }
                .disabled((app.project?.frames.isEmpty ?? true) || app.isWorking)
                .help(app.tr(
                    "Videoordner prüfen, der nie zum Training verwendet wurde - nötig, um den Modellvergleich ehrlich zu testen.",
                    "Review a video folder that was never used for training - needed for an honest model comparison.",
                    "Revisa una carpeta de vídeos que nunca se usó para entrenar - necesario para una comparación de modelos honesta.",
                    "Vérifiez un dossier vidéo jamais utilisé pour l’entraînement - nécessaire pour une comparaison de modèles honnête."
                ))
                if !app.independentValidationCandidates.isEmpty {
                    Text("\(app.independentValidationCandidates.count) \(app.tr("unabhängige Prüfkandidaten", "independent review candidates", "candidatos independientes", "candidats indépendants"))")
                        .font(.caption.bold()).foregroundStyle(.blue)
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
                        if frame.reviewStatus == "candidate" {
                            Image(systemName: frame.heldOut == true ? "checkmark.shield" : "questionmark.diamond.fill")
                                .foregroundStyle(frame.heldOut == true ? .blue : .orange)
                        } else if frame.heldOut == true {
                            Image(systemName: "checkmark.shield.fill").foregroundStyle(.blue)
                        }
                        if let flags = frame.reviewFlags, !flags.isEmpty {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.yellow)
                                .help(flags.map(app.reviewFlagLabel).joined(separator: ", "))
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
        if let purpose = app.localPickerPurpose {
            LocalFilePicker(
                purpose: purpose,
                language: app.language,
                initialURL: app.pickerInitialURL(for: purpose),
                knownVideoIDs: Set((app.project?.frames ?? []).map(\.videoID)),
                select: app.completeLocalPicker,
                cancel: app.cancelLocalPicker
            )
        } else if showBenchmark {
            benchmarkPanel
        } else if showBallTracking {
            ballTrackingPanel
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
                if frame.reviewStatus == "candidate" {
                    if frame.heldOut == true { heldOutReviewBar } else { candidateReviewBar }
                }
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

    private var ballTrackingPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 5) {
                    Label(app.tr("Balltracking-Simulation", "Ball-tracking simulation", "Simulación de seguimiento del balón", "Simulation de suivi du ballon"), systemImage: "scope")
                        .font(.largeTitle.bold())
                    Text(app.tr(
                        "Kurzes Video wählen und die Ballerkennung des aktiven Modells Bild für Bild abspielen. Das Video wird nur lokal verarbeitet, nie gespeichert oder trainiert.",
                        "Pick a short clip and play back the active model's ball detection frame by frame. The video is only processed locally, never saved or trained on.",
                        "Elige un vídeo corto y reproduce la detección del balón del modelo activo cuadro a cuadro. El vídeo solo se procesa localmente, nunca se guarda ni se entrena con él.",
                        "Choisissez une courte vidéo et regardez la détection du ballon du modèle actif image par image. La vidéo n’est traitée que localement, jamais enregistrée ni utilisée pour l’entraînement."
                    ))
                    .foregroundStyle(.secondary)
                }

                HStack(spacing: 10) {
                    Button {
                        app.startBallTrackingSimulation()
                    } label: {
                        Label(app.tr("Video auswählen", "Select video", "Seleccionar vídeo", "Sélectionner une vidéo"), systemImage: "film")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(app.store == nil || app.isWorking)

                    Picker(app.tr("Modell", "Model"), selection: $app.modelSize) {
                        ForEach(ModelSize.allCases) { Text($0.title(language: app.language)).tag($0) }
                    }
                    .fixedSize()
                    .disabled(app.isWorking)

                    if app.isWorking {
                        ProgressView(value: app.progress)
                        Text(app.status).font(.caption).foregroundStyle(.secondary)
                    }
                }

                if let simulation = app.ballTrackingSimulation {
                    BallTrackingPlayer(
                        simulation: simulation,
                        fieldGeometry: app.project?.fieldGeometry,
                        language: app.language,
                        lookaheadFrames: $app.ballTrackingLookaheadFrames
                    )
                    .id(simulation.id)
                } else if !app.isWorking {
                    ContentUnavailableView(
                        app.tr("Noch keine Simulation", "No simulation yet", "Aún no hay simulación", "Aucune simulation pour l’instant"),
                        systemImage: "scope",
                        description: Text(app.tr(
                            "Wähle oben ein kurzes Video aus, um zu starten.",
                            "Select a short video above to get started.",
                            "Selecciona un vídeo corto arriba para empezar.",
                            "Sélectionnez une courte vidéo ci-dessus pour commencer."
                        ))
                    )
                    .frame(minHeight: 200)
                }

                if let errorMessage = app.errorMessage {
                    Label(errorMessage, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .font(.caption)
                }
            }
            .padding(24)
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
                                .disabled(app.project == nil || app.isWorking || app.hasUnreviewedTrainingAnnotations || !app.hasReviewedHeldOutFrames)
                            if !app.isWorking && !app.hasReviewedHeldOutFrames {
                                Label(app.tr("Zuerst unter „Unabhängiger Modelltest“ einen nie trainierten Videoordner prüfen.", "First review a never-trained video folder under “Independent model test”.", "Primero revisa una carpeta de vídeos nunca entrenada en “Prueba de modelo independiente”.", "Vérifiez d’abord un dossier vidéo jamais entraîné sous « Test de modèle indépendant »."), systemImage: "exclamationmark.triangle")
                                    .font(.caption).foregroundStyle(.orange)
                            } else if !app.isWorking && app.hasUnreviewedTrainingAnnotations {
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

    private var heldOutReviewBar: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(app.tr("Unabhängiges Testbild prüfen", "Review independent test image", "Revisar imagen de prueba independiente", "Vérifier l’image de test indépendante")).font(.headline)
                Text(app.tr(
                    "Boxen oben direkt anklicken, verschieben, umbenennen oder löschen. Danach übernehmen.",
                    "Click, move, relabel, or delete boxes directly above. Then confirm.",
                    "Haz clic, mueve, renombra o elimina los cuadros arriba. Luego confirma.",
                    "Cliquez, déplacez, renommez ou supprimez les boîtes ci-dessus. Puis confirmez."
                )).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            Button(app.tr("Übernehmen", "Confirm", "Confirmar", "Confirmer")) { app.markHeldOutCandidateReviewed() }.buttonStyle(.borderedProminent).tint(.blue)
            Button(app.tr("Überspringen", "Skip", "Omitir", "Ignorer"), role: .destructive) { app.removeSelectedFrame() }
        }
        .padding(10)
        .background(.blue.opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
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
                Toggle(app.tr("Von Grund auf neu trainieren", "Train from scratch"), isOn: $app.trainFreshStart)
                    .help(app.tr(
                        "Ignoriert für diesen Lauf jeden vorhandenen Checkpoint und startet garantiert vom Apache-Basismodell. Sinnvoll, wenn seit dem letzten Modell neue Kategorien dazugekommen sind - RF-DETR erweitert einen fortgesetzten Checkpoint sonst nicht automatisch auf mehr Klassen.",
                        "Ignores any existing checkpoint for this run and guarantees a start from the Apache base model. Useful when new categories were added since the last model - RF-DETR doesn't automatically expand a continued checkpoint to more classes."
                    ))
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

            Toggle(isOn: $app.useEnsembleDisagreement) {
                Text(app.tr(
                    "Zweites Modell zur Unsicherheits-Prüfung nutzen (langsamer)",
                    "Use a second model to flag uncertain frames (slower)",
                    "Usar un segundo modelo para señalar cuadros inciertos (más lento)",
                    "Utiliser un second modèle pour signaler les images incertaines (plus lent)"
                ))
            }
            .help(app.tr(
                "Führt zusätzlich ein weiteres installiertes Modell aus und markiert Bilder, bei denen sich die Modelle uneinig sind, zur bevorzugten Prüfung.",
                "Also runs another installed model and flags frames where the models disagree for priority review.",
                "Ejecuta además otro modelo instalado y marca para revisión prioritaria los cuadros en los que los modelos no coinciden.",
                "Exécute aussi un autre modèle installé et signale pour une révision prioritaire les images où les modèles sont en désaccord."
            ))

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
                                    if !model.classes.isEmpty {
                                        Text(model.classes.map { app.language.category($0) }.joined(separator: ", "))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }
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
    /// True when this is a folder whose videos (checked recursively) were
    /// already extracted into the current project - "Unabhängiger
    /// Modelltest" only, so the folder browser can tell already-used
    /// folders apart from genuinely fresh ones instead of listing them
    /// indistinguishably.
    var isAlreadyUsed: Bool = false
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
    /// videoIDs already present in the current project's frames - used only
    /// for .independentValidationFolder, to mark subfolders whose videos
    /// were already extracted so they're not confused with fresh ones.
    let knownVideoIDs: Set<String>

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
        knownVideoIDs: Set<String> = [],
        select: @escaping (URL) -> Void,
        cancel: @escaping () -> Void
    ) {
        self.purpose = purpose
        self.language = language
        self.knownVideoIDs = knownVideoIDs
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
        case .independentValidationFolder:
            language.text("Ordner mit nie trainierten Videos auswählen", "Select folder with never-trained videos", "Seleccionar carpeta con vídeos nunca entrenados", "Sélectionner le dossier des vidéos jamais entraînées")
        case .modelPackage:
            language.text("Reco-Modellpaket auswählen", "Select Reco model package", "Seleccionar paquete de modelo Reco", "Sélectionner le paquet de modèle Reco")
        case .simulationVideo:
            language.text("Kurzes Video für die Balltracking-Simulation auswählen", "Select a short clip for the ball-tracking simulation", "Seleccionar un vídeo corto para la simulación de seguimiento del balón", "Sélectionner une courte vidéo pour la simulation de suivi du ballon")
        }
    }

    /// Purposes where the confirm button accepts a chosen file (not the
    /// current folder) - currently .recomodel packages and simulation clips.
    private var isFileSelection: Bool {
        purpose == .modelPackage || purpose == .simulationVideo
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
                Label(title, systemImage: purpose == .modelPackage ? "shippingbox" : purpose == .simulationVideo ? "film" : "folder")
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
                                : purpose == .simulationVideo
                                ? language.text("Kein Video in diesem Ordner.", "No video in this folder.", "No hay ningún vídeo en esta carpeta.", "Aucune vidéo dans ce dossier.")
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
                                HStack {
                                    Label(entry.url.lastPathComponent, systemImage: entry.isDirectory ? "folder.fill" : purpose == .simulationVideo ? "film.fill" : "shippingbox.fill")
                                        .foregroundStyle(entry.isAlreadyUsed ? .secondary : .primary)
                                    if entry.isAlreadyUsed {
                                        Spacer()
                                        Label(language.text("bereits verwendet", "already used", "ya usado", "déjà utilisé"), systemImage: "checkmark.circle")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
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
                    if isFileSelection {
                        if let selectedFile { select(selectedFile) }
                    } else {
                        select(currentURL)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(isFileSelection && selectedFile == nil)
            }
        }
        .padding(22)
        .task(id: currentURL) { await loadCurrentDirectory() }
    }

    private var selectionTitle: String {
        if purpose == .modelPackage {
            return language.text("Modell importieren", "Import model", "Importar modelo", "Importer le modèle")
        }
        if purpose == .simulationVideo {
            return language.text("Video verwenden", "Use this video", "Usar este vídeo", "Utiliser cette vidéo")
        }
        return language.text("Diesen Ordner verwenden", "Use this folder", "Usar esta carpeta", "Utiliser ce dossier")
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
        let requestedKnownIDs = knownVideoIDs
        let snapshot = await Task.detached(priority: .userInitiated) {
            Self.readDirectory(at: requestedURL, purpose: requestedPurpose, knownVideoIDs: requestedKnownIDs)
        }.value
        guard currentURL == requestedURL else { return }
        entries = snapshot.entries
        loadError = snapshot.error
        isLoading = false
    }

    nonisolated private static func readDirectory(at url: URL, purpose: LocalPickerPurpose, knownVideoIDs: Set<String>) -> LocalDirectorySnapshot {
        do {
            let urls = try FileManager.default.contentsOfDirectory(
                at: url,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants]
            )
            let entries = urls.compactMap { candidate -> LocalFileEntry? in
                let isDirectory = (try? candidate.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
                if isDirectory {
                    let isAlreadyUsed = purpose == .independentValidationFolder && !knownVideoIDs.isEmpty
                        && containsAlreadyUsedVideo(in: candidate, knownVideoIDs: knownVideoIDs)
                    return LocalFileEntry(url: candidate, isDirectory: true, isAlreadyUsed: isAlreadyUsed)
                }
                if purpose == .modelPackage, candidate.pathExtension.lowercased() == "recomodel" {
                    return LocalFileEntry(url: candidate, isDirectory: false)
                }
                if purpose == .simulationVideo, FrameExtractor.supportedExtensions.contains(candidate.pathExtension.lowercased()) {
                    return LocalFileEntry(url: candidate, isDirectory: false)
                }
                return nil
            }.sorted {
                if $0.isDirectory != $1.isDirectory { return $0.isDirectory }
                if $0.isAlreadyUsed != $1.isAlreadyUsed { return !$0.isAlreadyUsed }
                return $0.url.lastPathComponent.localizedStandardCompare($1.url.lastPathComponent) == .orderedAscending
            }
            return LocalDirectorySnapshot(entries: entries, error: nil)
        } catch {
            return LocalDirectorySnapshot(entries: [], error: error.localizedDescription)
        }
    }

    /// Recursively checks whether any video under `folder` matches an
    /// already-extracted videoID (see FrameExtractor.stableID, which hashes
    /// a video's absolute path) - so "Unabhängiger Modelltest" can mark a
    /// subfolder as already used without re-extracting anything.
    /// Deliberately shallow (this folder's immediate children only), not a
    /// recursive descent - a previous version used FileManager.enumerator
    /// (recursive) here, which meant every folder listing while browsing for
    /// "Unabhängiger Modelltest" had to walk the *entire* subtree of every
    /// visible subfolder before the list could even appear. For someone with
    /// many GoPro session folders, each holding a dozen-plus chapter files,
    /// that made the picker look hung. Camera dumps are essentially always
    /// flat (videos directly inside each session folder), so a shallow check
    /// already covers the common case at a cost in line with the rest of
    /// this picker's browsing (a single directory listing, not a full walk).
    nonisolated private static func containsAlreadyUsedVideo(in folder: URL, knownVideoIDs: Set<String>) -> Bool {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: folder,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return false }
        for candidate in entries {
            guard FrameExtractor.supportedExtensions.contains(candidate.pathExtension.lowercased()) else { continue }
            if knownVideoIDs.contains(FrameExtractor.stableID(for: candidate.path)) { return true }
        }
        return false
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
                language.text("Modellpakete: Schutz vor ausführbarem Code", "Model packages: protection against executable code", "Paquetes de modelo: protección contra código ejecutable", "Paquets de modèle : protection contre le code exécutable"),
                language.text("Eine Prüfsumme allein beweist nur, dass eine Gewichtsdatei zu ihrem eigenen Manifest passt - nicht, dass sie sicher ist, denn wer ein Paket erstellt, kontrolliert auch dessen Prüfsumme. Modellpakete (.recomodel) werden beim Import (und zur Sicherheit auch beim Erstellen/Kombinieren) jetzt zusätzlich mit PyTorchs eigenem sicheren Lademodus (weights_only) geprüft, der jede Gewichtsdatei ablehnt, die mehr als reine Modelldaten enthält - etwa ausführbaren Code, der beim Laden automatisch ausgeführt würde. Ist die ML-Umgebung noch nicht eingerichtet, wird die Prüfsumme wie bisher trotzdem verifiziert, nur diese zusätzliche Prüfung dann übersprungen.", "A matching checksum alone only proves a weights file matches its own manifest - not that it's safe, since whoever builds a package also controls its checksum. Model packages (.recomodel) are now additionally checked, on import (and, for safety, also when creating/combining one), with PyTorch's own safe loading mode (weights_only), which rejects any weights file that contains more than plain model data - such as executable code that would run automatically on load. If the ML environment isn't set up yet, the checksum is still verified as before, only this extra check is skipped.", "Una suma de verificación por sí sola solo demuestra que un archivo de pesos coincide con su propio manifiesto, no que sea seguro, ya que quien crea un paquete también controla su suma de verificación. Los paquetes de modelo (.recomodel) ahora también se comprueban, al importarlos (y, por seguridad, también al crearlos o combinarlos), con el modo de carga seguro propio de PyTorch (weights_only), que rechaza cualquier archivo de pesos que contenga más que datos de modelo puros, como código ejecutable que se ejecutaría automáticamente al cargarlo. Si el entorno de ML aún no está configurado, la suma de verificación se sigue comprobando como antes, solo se omite esta comprobación adicional.", "Une somme de contrôle correspondante prouve seulement qu’un fichier de poids correspond à son propre manifeste, pas qu’il est sûr, car qui crée un paquet en contrôle aussi la somme de contrôle. Les paquets de modèle (.recomodel) sont désormais aussi vérifiés, à l’import (et, par sécurité, aussi lors de la création/combinaison), avec le mode de chargement sûr propre à PyTorch (weights_only), qui rejette tout fichier de poids contenant davantage que de simples données de modèle - par exemple du code exécutable qui s’exécuterait automatiquement au chargement. Si l’environnement ML n’est pas encore configuré, la somme de contrôle est tout de même vérifiée comme avant, seule cette vérification supplémentaire est ignorée.")
            ),
            (
                language.text("Training von Grund auf neu", "Train from scratch", "Entrenar desde cero", "Entraîner à partir de zéro"),
                language.text("Neuer Schalter „Von Grund auf neu trainieren“: ignoriert für einen Lauf jeden vorhandenen Checkpoint und startet garantiert vom Basismodell. Nötig, weil RF-DETR beim Fortsetzen von einem Checkpoint mit weniger Klassen (z. B. nur Ball) den Klassifikations-Kopf nicht auf die neuen Klassen (Spieler, Schiedsrichter …) erweitert - das Modell konnte sie dann gar nicht lernen. Außerdem wird ein neues Modell mit anderen Klassen als das aktive nicht mehr allein wegen eines niedrigeren Gesamtwerts als „schlechter“ behandelt: Die Werte sind dann nicht vergleichbar, das Modell wird archiviert und mit einem Hinweis auf den Modellvergleich pro Kategorie versehen.", "New „Train from scratch“ switch: ignores any existing checkpoint for one run and guarantees a start from the base model. Needed because when continuing from a checkpoint with fewer classes (e.g. ball only), RF-DETR does not expand the classification head to the new classes (player, referee …) - the model could not learn them at all. Also, a new model with different classes than the active one is no longer treated as „worse“ just because of a lower overall score: the scores are not comparable then, so it is archived with a pointer to the per-category model comparison.", "Nuevo interruptor „Entrenar desde cero“: ignora cualquier checkpoint existente en una ejecución y garantiza el inicio desde el modelo base. Necesario porque, al continuar desde un checkpoint con menos clases (p. ej. solo balón), RF-DETR no amplía la cabeza de clasificación a las nuevas clases (jugador, árbitro …), por lo que el modelo no podía aprenderlas. Además, un modelo nuevo con clases distintas a las del activo ya no se considera „peor“ solo por una puntuación global menor: las puntuaciones no son comparables, así que se archiva con una indicación de la comparación de modelos por categoría.", "Nouvel interrupteur « Entraîner à partir de zéro » : ignore tout checkpoint existant pour une exécution et garantit un démarrage depuis le modèle de base. Nécessaire car, en reprenant un checkpoint avec moins de classes (p. ex. ballon seul), RF-DETR n’étend pas la tête de classification aux nouvelles classes (joueur, arbitre …) - le modèle ne pouvait donc pas les apprendre. De plus, un nouveau modèle avec d’autres classes que le modèle actif n’est plus jugé « moins bon » à cause d’un score global plus bas : les scores ne sont pas comparables, il est donc archivé avec un renvoi vers la comparaison de modèles par catégorie.")
            ),
            (
                language.text("Prüf-Priorität: Modell-Uneinigkeit und zeitliche Ausreißer", "Review priority: model disagreement and temporal outliers", "Prioridad de revisión: desacuerdo entre modelos y valores atípicos temporales", "Priorité de révision : désaccord des modèles et anomalies temporelles"),
                language.text("Zwei neue Signale markieren automatisch beschriftete Bilder zur bevorzugten Prüfung: ein optionaler zweiter Modelldurchlauf zeigt Bilder an, bei denen sich zwei installierte Modelle uneinig sind (Schalter „Zweites Modell zur Unsicherheits-Prüfung nutzen“, etwas langsamer), und eine kostenlose Prüfung erkennt Bilder, deren Position stark von den Nachbar-Bildern desselben Videos abweicht. Markierte Bilder erscheinen zuerst in der Prüfwarteschlange, mit einem Warnsymbol und Begründung. Betrifft nur die Reihenfolge der Prüfung, nie das Training selbst.", "Two new signals automatically flag auto-labeled frames for priority review: an optional second model pass shows frames where two installed models disagree (toggle „Use a second model to flag uncertain frames“, a bit slower), and a free check flags frames whose position deviates sharply from neighboring frames in the same video. Flagged frames appear first in the review queue, with a warning icon and reason. Only affects review order, never training itself.", "Dos nuevas señales marcan automáticamente los cuadros etiquetados para revisión prioritaria: un segundo paso de modelo opcional muestra cuadros en los que dos modelos instalados no coinciden (interruptor „Usar un segundo modelo para señalar cuadros inciertos“, algo más lento), y una comprobación gratuita detecta cuadros cuya posición se desvía mucho de los cuadros vecinos del mismo vídeo. Los cuadros marcados aparecen primero en la cola de revisión, con un icono de advertencia y el motivo. Solo afecta al orden de revisión, nunca al entrenamiento en sí.", "Deux nouveaux signaux marquent automatiquement les images auto-étiquetées pour une révision prioritaire : un second passage de modèle optionnel affiche les images où deux modèles installés sont en désaccord (interrupteur « Utiliser un second modèle pour signaler les images incertaines », un peu plus lent), et une vérification gratuite repère les images dont la position dévie fortement des images voisines de la même vidéo. Les images signalées apparaissent en premier dans la file de révision, avec une icône d’avertissement et le motif. N’affecte que l’ordre de révision, jamais l’entraînement lui-même.")
            ),
            (
                language.text("Balltracking-Simulation: Fehlermeldung beim Start behoben", "Ball-tracking simulation: fixed an error on start", "Simulación de seguimiento del balón: corregido un error al iniciar", "Simulation de suivi du ballon : correction d’une erreur au démarrage"),
                language.text("Die Balltracking-Simulation konnte mit „Die Daten konnten nicht geöffnet werden, da sie nicht das korrekte Format haben“ fehlschlagen: Warnungen, die PyTorch/RF-DETR beim Laden des Modells manchmal ausgibt, landeten in derselben Ausgabe wie das Ergebnis und störten dessen Verarbeitung. Nur noch die letzte Zeile der Ausgabe wird dafür verwendet.", "The ball-tracking simulation could fail with “The data couldn’t be read because it isn’t in the correct format”: warnings PyTorch/RF-DETR sometimes print while loading the model ended up mixed into the same output as the result and interfered with reading it. Only the output’s last line is used for that now.", "La simulación de seguimiento del balón podía fallar con „Los datos no se pudieron abrir porque no tienen el formato correcto“: las advertencias que PyTorch/RF-DETR a veces muestra al cargar el modelo se mezclaban con la misma salida que el resultado e interferían con su lectura. Ahora solo se usa la última línea de la salida para eso.", "La simulation de suivi du ballon pouvait échouer avec « Les données n’ont pas pu être ouvertes, car leur format n’est pas correct » : les avertissements que PyTorch/RF-DETR affiche parfois lors du chargement du modèle se mélangeaient à la même sortie que le résultat et perturbaient sa lecture. Seule la dernière ligne de la sortie est désormais utilisée pour cela.")
            ),
            (
                language.text("Ordner-Browser für „Unabhängiger Modelltest“ hing bei vielen Videos", "Folder browser for “Independent model test” hung with many videos", "El navegador de carpetas de “Prueba de modelo independiente” se bloqueaba con muchos vídeos", "Le navigateur de dossiers pour « Test de modèle indépendant » se bloquait avec de nombreuses vidéos"),
                language.text("Die neue „bereits verwendet“-Markierung im Ordner-Browser hat beim Durchsuchen jeden sichtbaren Unterordner vollständig rekursiv nach Videos durchsucht - bei vielen Aufnahme-Ordnern mit jeweils vielen Dateien (z. B. GoPro-Aufnahmen mit mehreren Kapitel-Dateien) sah das wie Hängen aus. Prüft jetzt nur noch die jeweils direkte Ordnerebene. Außerdem zeigt das Einlesen der Videolängen vor der Extraktion jetzt Fortschritt an, statt kommentarlos zu pausieren.", "The new “already used” marker in the folder browser recursively searched every visible subfolder's entire contents for videos - with many recording folders each holding many files (e.g. GoPro recordings split into several chapter files), this looked like hanging. Now only checks each folder's immediate contents. Reading video durations before extraction also now shows progress instead of pausing silently.", "La nueva marca de “ya usado” en el navegador de carpetas buscaba vídeos de forma recursiva en todo el contenido de cada subcarpeta visible - con muchas carpetas de grabación con muchos archivos cada una (p. ej. grabaciones GoPro divididas en varios capítulos), esto parecía un bloqueo. Ahora solo revisa el contenido directo de cada carpeta. Además, la lectura de la duración de los vídeos antes de la extracción ahora muestra progreso en lugar de pausarse sin avisar.", "Le nouveau marqueur « déjà utilisé » du navigateur de dossiers recherchait récursivement les vidéos dans tout le contenu de chaque sous-dossier visible - avec de nombreux dossiers d’enregistrement contenant chacun de nombreux fichiers (p. ex. des enregistrements GoPro divisés en plusieurs chapitres), cela ressemblait à un blocage. Ne vérifie désormais que le contenu direct de chaque dossier. La lecture de la durée des vidéos avant l’extraction affiche aussi désormais une progression au lieu de faire une pause silencieuse.")
            ),
            (
                language.text("Balltracking-Simulation", "Ball-tracking simulation", "Simulación de seguimiento del balón", "Simulation de suivi du ballon"),
                language.text("Neuer Bereich „Balltracking simulieren“: ein kurzes Video wählen und die Ballerkennung des aktiven Modells Bild für Bild abspielen - erkannt (grün), zwischen zwei echten Erkennungen interpoliert (orange), auf der letzten bekannten Position gehalten (gelb) oder verloren. Ein Regler bestimmt live, wie viele Bilder weit nach vorn und zurück geschaut werden darf, ohne dass die Erkennung neu laufen muss. Das Video wird nur lokal verarbeitet, nie gespeichert oder trainiert.", "New „Simulate ball tracking“ section: pick a short clip and play back the active model's ball detection frame by frame - detected (green), interpolated between two real detections (orange), held at the last known position (yellow), or lost. A live slider controls how many frames ahead and behind it's allowed to look, without rerunning detection. The video is only processed locally, never saved or trained on.", "Nueva sección „Simular seguimiento del balón“: elige un vídeo corto y reproduce la detección del balón del modelo activo cuadro a cuadro - detectado (verde), interpolado entre dos detecciones reales (naranja), mantenido en la última posición conocida (amarillo) o perdido. Un control deslizante en vivo determina cuántos cuadros hacia delante y atrás se puede mirar, sin volver a ejecutar la detección. El vídeo solo se procesa localmente, nunca se guarda ni se entrena con él.", "Nouvelle section « Simuler le suivi du ballon » : choisissez une courte vidéo et regardez la détection du ballon du modèle actif image par image - détecté (vert), interpolé entre deux détections réelles (orange), maintenu à la dernière position connue (jaune) ou perdu. Un curseur en direct détermine de combien d’images on peut regarder en avant et en arrière, sans relancer la détection. La vidéo n’est traitée que localement, jamais enregistrée ni utilisée pour l’entraînement.")
            ),
            (
                language.text("Windows: automatische GPU-Einrichtung", "Windows: automatic GPU setup", "Windows: configuración automática de GPU", "Windows : configuration GPU automatique"),
                language.text("„ML einrichten“ erkennt unter Windows jetzt eine vorhandene NVIDIA-Grafikkarte und installiert PyTorch automatisch mit CUDA-Unterstützung - vorher landete dort standardmäßig die CPU-only-Version, ohne dass die App das erkennen oder anzeigen konnte. Ein neuer Schalter („GPU-Beschleunigung automatisch einrichten“) erlaubt es, das bei Problemen auszuschalten. Die Hardware-Anzeige zeigt jetzt korrekt an, ob CUDA genutzt wird, statt immer nur „CPU“.", "„Set up ML“ now detects an NVIDIA GPU on Windows and installs PyTorch with CUDA support automatically - previously the CPU-only build was installed by default there, with no way for the app to detect or show that. A new switch („Automatically set up GPU acceleration“) lets you turn this off if it causes trouble. The hardware display now correctly shows whether CUDA is being used, instead of always just „CPU“.", "„Configurar ML“ ahora detecta una GPU NVIDIA en Windows e instala PyTorch con soporte CUDA automáticamente - antes se instalaba allí la versión solo para CPU por defecto, sin que la app pudiera detectarlo ni mostrarlo. Un nuevo interruptor („Configurar automáticamente la aceleración por GPU“) permite desactivarlo si causa problemas. La indicación de hardware ahora muestra correctamente si se usa CUDA, en lugar de mostrar siempre „CPU“.", "« Configurer le ML » détecte désormais une GPU NVIDIA sous Windows et installe PyTorch avec le support CUDA automatiquement - auparavant, la version CPU uniquement y était installée par défaut, sans que l’application puisse le détecter ni l’afficher. Un nouvel interrupteur (« Configurer automatiquement l’accélération GPU ») permet de le désactiver en cas de problème. L’affichage matériel indique désormais correctement si CUDA est utilisé, au lieu d’afficher toujours « CPU ».")
            ),
            (
                language.text("Unabhängiger Modelltest", "Independent model test", "Prueba de modelo independiente", "Test de modèle indépendant"),
                language.text("Neuer Bereich „Unabhängiger Modelltest“: einen Videoordner prüfen, der nie zum Training verwendet wurde, und daraus - mit Unterstützung über alle Kategorien hinweg - ein echtes, unabhängiges Testset erstellen. Der Modellvergleich verlangt jetzt genau solche unabhängigen Bilder für die Referenz, statt beliebiger geprüfter Trainingsbilder - damit ein Modell nicht einfach gut abschneidet, weil es genau diese Aufnahmen schon kannte.", "New „Independent model test“ section: review a video folder that was never used for training, and turn it - with app assistance across every category - into a genuinely independent test set. The model benchmark now requires exactly this kind of independent images for its reference instead of any reviewed training images, so a model can't simply score well because it already knew that exact footage.", "Nueva sección „Prueba de modelo independiente“: revisa una carpeta de vídeos que nunca se usó para entrenar y conviértela - con ayuda en todas las categorías - en un conjunto de prueba realmente independiente. La comparación de modelos ahora exige precisamente este tipo de imágenes independientes como referencia, en lugar de cualquier imagen de entrenamiento revisada, para que un modelo no puntúe bien solo por conocer ya exactamente ese material.", "Nouvelle section « Test de modèle indépendant » : vérifiez un dossier vidéo jamais utilisé pour l’entraînement et transformez-le - avec l’aide de l’appli sur toutes les catégories - en un vrai jeu de test indépendant. La comparaison de modèles exige désormais ce type d’images indépendantes comme référence, plutôt que n’importe quelle image d’entraînement vérifiée, pour qu’un modèle ne réussisse pas simplement parce qu’il connaissait déjà exactement ces images.")
            ),
            (
                language.text("Kombiniertes Modell erstellen", "Create a combined model", "Crear un modelo combinado", "Créer un modèle combiné"),
                language.text("Beim Modellvergleich lässt sich jetzt aus mehreren installierten Modellen ein neues, kombiniertes Modell „backen“: pro Kategorie das stärkste Modell wählen (z. B. ein Modell für Schiedsrichter, ein anderes für den Ball) - kein echtes Zusammenführen der Gewichte (bei RF-DETR nicht möglich), sondern jedes Modell übernimmt zur Laufzeit nur seine zugeordneten Kategorien. Erscheint danach wie jedes andere Modell in der Bibliothek.", "The model benchmark can now „bake“ a new combined model from several installed models: pick the strongest model per category (e.g. one model for referees, another for the ball) - not real weight merging (not possible with RF-DETR), each model just handles its assigned categories at runtime. Appears in the library afterward like any other model.", "La comparación de modelos ahora puede „crear“ un nuevo modelo combinado a partir de varios modelos instalados: elige el modelo más fuerte por categoría (p. ej. un modelo para árbitros, otro para el balón) - no es una fusión real de pesos (no es posible con RF-DETR), cada modelo solo gestiona sus categorías asignadas en tiempo de ejecución. Después aparece en la biblioteca como cualquier otro modelo.", "La comparaison de modèles peut désormais « créer » un nouveau modèle combiné à partir de plusieurs modèles installés : choisissez le modèle le plus fort par catégorie (p. ex. un modèle pour les arbitres, un autre pour le ballon) - pas une vraie fusion de poids (impossible avec RF-DETR), chaque modèle ne gère que ses catégories assignées au moment de l’exécution. Apparaît ensuite dans la bibliothèque comme n’importe quel autre modèle.")
            ),
            (
                language.text("Windows: Trainingsabsturz durch Konsolen-Kodierung behoben", "Windows: fixed a training crash caused by console encoding", "Windows: corregido un fallo de entrenamiento por la codificación de la consola", "Windows : correction d’un plantage d’entraînement lié à l’encodage de la console"),
                language.text("„Lokal trainieren“ stürzte unter Windows während der ersten Epoche mit einem Kodierungsfehler ab, weil die Konsolenausgabe dort standardmäßig nicht auf UTF-8 steht. Behoben, indem der lokale ML-Vorgang jetzt ausdrücklich UTF-8 verwendet.", "\"Train locally\" crashed on Windows during the first epoch with an encoding error, because console output there does not default to UTF-8. Fixed by having the local ML process use UTF-8 explicitly.", "«Entrenar localmente» fallaba en Windows durante la primera época con un error de codificación, ya que la salida de la consola no usa UTF-8 por defecto allí. Corregido haciendo que el proceso local de ML use UTF-8 explícitamente.", "« Entraîner localement » plantait sous Windows pendant la première époque avec une erreur d’encodage, la sortie console n’y étant pas UTF-8 par défaut. Corrigé en forçant le processus ML local à utiliser UTF-8 explicitement.")
            ),
            (
                language.text("Hinweis auf neue Versionen auch in der plattformübergreifenden Oberfläche", "Notice for new versions in the cross-platform interface too", "Aviso de nuevas versiones también en la interfaz multiplataforma", "Avis de nouvelles versions aussi dans l’interface multiplateforme"),
                language.text("Die Windows/Linux/Docker-Weboberfläche prüft jetzt beim Start ebenfalls, ob eine neuere Version veröffentlicht wurde (wie die Mac-App bereits seit 0.12.9) - bisher gab es dort gar keinen Hinweis darauf.", "The Windows/Linux/Docker web interface now also checks at launch whether a newer version has been published (like the Mac app already has since 0.12.9) - previously there was no such notice there at all.", "La interfaz web de Windows/Linux/Docker ahora también comprueba al iniciar si se ha publicado una versión más reciente (como ya hace la app de Mac desde la 0.12.9) - antes no existía ese aviso ahí.", "L’interface web Windows/Linux/Docker vérifie désormais aussi au démarrage si une version plus récente a été publiée (comme l’application Mac le fait déjà depuis la 0.12.9) - il n’y avait auparavant aucun avis de ce type.")
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
