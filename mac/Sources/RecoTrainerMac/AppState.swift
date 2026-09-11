import AppKit
import Combine
import CryptoKit
import Foundation

enum LocalPickerPurpose: String, Identifiable, Sendable {
    case trainingFolder
    case activeLearningFolder
    case modelPackage

    var id: String { rawValue }
}

private struct FolderLoadSnapshot: Sendable {
    var project: ProjectDocument?
    var activeModel: ActiveModelRecord?
    var installedModelCount: Int
    var models: [ManagedModelRecord]
    var continuationCheckpoints: [ModelSize: String]
    var benchmarkGroundTruth: BenchmarkGroundTruthRecord?
    var benchmarkReport: BenchmarkReport?
    var errorMessage: String?
}

@MainActor
final class AppState: ObservableObject {
    @Published var language: AppLanguage = .de
    @Published var sport: Sport = .football
    @Published var selectedFolder: URL?
    @Published var project: ProjectDocument?
    @Published var selectedFrameID: UUID?
    @Published var selectedCategory = "ball"
    /// Which classes "Automatisch markieren" should detect in one pass. Separate from
    /// selectedCategory, which stays single-valued for hand-drawing a new box and for
    /// the active-learning Ball/No-ball review - both are inherently one category at a time.
    @Published var autoLabelCategories: Set<String> = ["ball"]
    @Published var modelSize: ModelSize = .nano
    @Published var epochs = 20
    @Published var confidenceThreshold = 0.35
    @Published var framesPerVideo = 240
    @Published var progress = 0.0
    @Published var status = "Videoordner auswählen, um zu beginnen."
    @Published var log = ""
    @Published var isWorking = false
    @Published var hardware: HardwareStatus?
    @Published var activeModelPackageID: String?
    @Published var activeModelDescription: String?
    @Published var benchmarkThreshold = 0.05
    @Published var benchmarkGroundTruth: BenchmarkGroundTruthRecord?
    @Published var benchmarkReport: BenchmarkReport?
    @Published var errorMessage: String?
    @Published var localPickerPurpose: LocalPickerPurpose?
    @Published private(set) var installedModelCount = 0
    @Published var managedModels: [ManagedModelRecord] = []
    @Published private var continuationCheckpoints: [ModelSize: String] = [:]

    var store: ProjectStore? {
        selectedFolder.map(ProjectStore.forSourceFolder)
    }

    var selectedFrame: FrameRecord? {
        guard let id = selectedFrameID else { return project?.frames.first }
        return project?.frames.first(where: { $0.id == id })
    }

    var reviewCandidates: [FrameRecord] {
        project?.frames.filter { $0.reviewStatus == "candidate" } ?? []
    }

    /// Mirrors freezeBenchmarkGroundTruth()'s own frame filter: only training
    /// frames (not pending active-learning candidates, which freezing ignores
    /// entirely) matter for whether ground truth can be frozen.
    var hasUnreviewedTrainingAnnotations: Bool {
        (project?.frames.filter { $0.reviewStatus != "candidate" } ?? [])
            .flatMap(\.annotations)
            .contains { $0.source == "auto" }
    }

    var continuationCheckpointName: String? {
        continuationCheckpoints[modelSize]
    }

    /// Categories the current model (base or custom-trained) can actually detect
    /// right now. Mirrors ml_worker.py's auto_label()/active_model_classes() check:
    /// the Apache-2.0 base model only knows COCO's "sports ball" (-> ball/puck) and
    /// "person" (-> player) classes.
    ///
    /// If a library model is activated for the selected size, its own manifest is
    /// authoritative - it may have been trained back when the project only had
    /// "ball" annotated, so it can't just be assumed to cover every category the
    /// sport schema lists now. Previously that wrong assumption ("any checkpoint at
    /// all -> every category is fair game") made a fresh project preselect e.g.
    /// "referee" for auto-label even though the active model had never seen it,
    /// wasting a full inference pass and reporting a single misleading "0 boxes".
    /// Only a raw, not-yet-packaged runs/<size> checkpoint (no manifest to consult)
    /// still falls back to that optimistic assumption.
    var autoLabelSupportedCategories: Set<String> {
        if let activeID = activeModelPackageID,
           let active = managedModels.first(where: { $0.packageID == activeID }),
           active.modelSize == modelSize.rawValue {
            return Set(sport.categories).intersection(active.classes)
        }
        guard continuationCheckpointName == nil else { return Set(sport.categories) }
        return Set(sport.categories).intersection(["ball", "puck", "player"])
    }

    var benchmarkReportIsCurrent: Bool {
        benchmarkReport?.datasetID == benchmarkGroundTruth?.datasetID
    }

    /// Auto-label chips a freshly loaded project should start with pre-checked.
    ///
    /// Previously this defaulted to just the sport's first category (typically "ball"),
    /// so getting person detections for free from the base model's generic COCO "person"
    /// class required remembering to also tick "player" - referees then had to be drawn
    /// by hand from scratch instead of being generically detected as "player" and just
    /// relabeled via the annotation editor's click-to-relabel. Preselecting every
    /// currently-supported category makes that free detection the default instead of an
    /// easy-to-miss opt-in.
    nonisolated static func defaultAutoLabelCategories(selectedCategory: String, supported: Set<String>) -> Set<String> {
        supported.isEmpty ? [selectedCategory] : supported
    }

    func tr(_ german: String, _ english: String, _ spanish: String? = nil, _ french: String? = nil) -> String {
        language.text(german, english, spanish, french)
    }

    func chooseFolder() {
        localPickerPurpose = .trainingFolder
    }

    func pickerInitialURL(for purpose: LocalPickerPurpose) -> URL {
        switch purpose {
        case .trainingFolder:
            return selectedFolder ?? URL(filePath: "/Volumes", directoryHint: .isDirectory)
        case .activeLearningFolder:
            return URL(filePath: "/Volumes", directoryHint: .isDirectory)
        case .modelPackage:
            return FileManager.default.urls(for: .downloadsDirectory, in: .userDomainMask).first
                ?? FileManager.default.homeDirectoryForCurrentUser
        }
    }

    func cancelLocalPicker() {
        localPickerPurpose = nil
    }

    func completeLocalPicker(with url: URL) {
        guard let purpose = localPickerPurpose else { return }
        localPickerPurpose = nil
        switch purpose {
        case .trainingFolder:
            selectedFolder = url
            loadExistingProjectIfPresent()
        case .activeLearningFolder:
            guard let selectedFolder, let store else { return }
            startActiveLearning(from: url, selectedFolder: selectedFolder, store: store)
        case .modelPackage:
            guard let store else { return }
            importModelPackage(from: url, store: store)
        }
    }

    func loadExistingProjectIfPresent() {
        guard let selectedFolder else { return }
        let requestedFolder = selectedFolder.standardizedFileURL
        isWorking = true
        errorMessage = nil
        project = nil
        selectedFrameID = nil
        installedModelCount = 0
        managedModels = []
        continuationCheckpoints = [:]
        status = tr(
            "Lade vorhandenes Trainingsprojekt …",
            "Loading existing training project …",
            "Cargando el proyecto de entrenamiento …",
            "Chargement du projet d’entraînement …"
        )

        Task {
            let snapshot = await Task.detached(priority: .userInitiated) {
                Self.loadFolderSnapshot(from: requestedFolder)
            }.value
            guard self.selectedFolder?.standardizedFileURL == requestedFolder else { return }

            self.project = snapshot.project
            self.installedModelCount = snapshot.installedModelCount
            self.managedModels = snapshot.models
            self.continuationCheckpoints = snapshot.continuationCheckpoints
            self.benchmarkGroundTruth = snapshot.benchmarkGroundTruth
            self.benchmarkReport = snapshot.benchmarkReport
            self.activeModelPackageID = snapshot.activeModel?.packageID
            self.activeModelDescription = snapshot.activeModel?.description

            if let loaded = snapshot.project {
                self.sport = loaded.sport
                self.selectedCategory = loaded.sport.categories.first ?? "ball"
                self.selectedFrameID = loaded.frames.first?.id
                self.framesPerVideo = loaded.framesPerVideo ?? 240
                if let activeSize = snapshot.activeModel.flatMap({ ModelSize(rawValue: $0.modelSize) }) {
                    self.modelSize = activeSize
                } else if let lastModel = loaded.lastTraining?.model,
                          let lastModelSize = ModelSize(rawValue: lastModel) {
                    self.modelSize = lastModelSize
                }
                self.autoLabelCategories = Self.defaultAutoLabelCategories(
                    selectedCategory: self.selectedCategory,
                    supported: self.autoLabelSupportedCategories
                )
                self.status = self.tr(
                    "Projekt geladen: \(loaded.frames.count) Frames.",
                    "Project loaded: \(loaded.frames.count) frames.",
                    "Proyecto cargado: \(loaded.frames.count) fotogramas.",
                    "Projet chargé : \(loaded.frames.count) images."
                )
            } else if let loadError = snapshot.errorMessage {
                self.errorMessage = loadError
                self.selectedCategory = self.sport.categories.first ?? "ball"
                self.autoLabelCategories = Self.defaultAutoLabelCategories(
                    selectedCategory: self.selectedCategory,
                    supported: self.autoLabelSupportedCategories
                )
                self.status = self.tr(
                    "Trainingsprojekt konnte nicht geladen werden. Die vorhandenen Daten wurden nicht verändert.",
                    "The training project could not be loaded. Existing data was not changed.",
                    "No se pudo cargar el proyecto. Los datos existentes no se modificaron.",
                    "Le projet n’a pas pu être chargé. Les données existantes n’ont pas été modifiées."
                )
            } else {
                self.selectedCategory = self.sport.categories.first ?? "ball"
                self.autoLabelCategories = Self.defaultAutoLabelCategories(
                    selectedCategory: self.selectedCategory,
                    supported: self.autoLabelSupportedCategories
                )
                self.status = self.tr(
                    "Ordner gewählt. Jetzt Videos analysieren.",
                    "Folder selected. Analyze the videos next.",
                    "Carpeta seleccionada. Analiza ahora los vídeos.",
                    "Dossier sélectionné. Analysez maintenant les vidéos."
                )
            }
            self.isWorking = false
        }
    }

    nonisolated private static func loadFolderSnapshot(from folder: URL) -> FolderLoadSnapshot {
        let store = ProjectStore.forSourceFolder(folder)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601

        let project: ProjectDocument?
        var loadError: String?
        do {
            project = try store.load()
        } catch ProjectStoreError.noProject {
            project = nil
        } catch {
            project = nil
            loadError = error.localizedDescription
        }

        let activeURL = store.rootURL.appending(path: "models/active.json")
        let activeModel = (try? Data(contentsOf: activeURL)).flatMap {
            try? decoder.decode(ActiveModelRecord.self, from: $0)
        }

        let libraryURL = store.rootURL.appending(path: "models/library", directoryHint: .isDirectory)
        let modelDirectories = (try? FileManager.default.contentsOfDirectory(
            at: libraryURL,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        var models = modelDirectories.compactMap { directory -> ManagedModelRecord? in
            let manifest = directory.appending(path: "manifest.json")
            guard let data = try? Data(contentsOf: manifest),
                  var record = try? decoder.decode(ManagedModelRecord.self, from: data) else { return nil }
            record.isActive = record.packageID == activeModel?.packageID
            return record
        }
        if let bestID = models.compactMap({ model in model.comparisonScore.map { ($0, model.packageID) } }).max(by: { $0.0 < $1.0 })?.1 {
            models = models.map { model in
                var updated = model
                updated.isBest = model.packageID == bestID
                return updated
            }
        }
        models.sort {
            if ($0.isActive ?? false) != ($1.isActive ?? false) { return $0.isActive ?? false }
            if ($0.isBest ?? false) != ($1.isBest ?? false) { return $0.isBest ?? false }
            return ($0.createdAt ?? "") > ($1.createdAt ?? "")
        }
        let installedCount = models.count

        var checkpoints: [ModelSize: String] = [:]
        for size in ModelSize.allCases {
            if let checkpoint = store.preferredTrainingCheckpoint(for: size) {
                checkpoints[size] = checkpoint.lastPathComponent
            }
        }
        if let activeModel,
           let size = ModelSize(rawValue: activeModel.modelSize),
           checkpoints[size] == nil {
            checkpoints[size] = URL(filePath: activeModel.weights).lastPathComponent
        }

        let benchmarkURL = store.rootURL.appending(path: "benchmarks", directoryHint: .isDirectory)
        let groundTruth = (try? Data(contentsOf: benchmarkURL.appending(path: "ground-truth.json"))).flatMap {
            try? decoder.decode(BenchmarkGroundTruthRecord.self, from: $0)
        }
        let report = (try? Data(contentsOf: benchmarkURL.appending(path: "latest.json"))).flatMap {
            try? decoder.decode(BenchmarkReport.self, from: $0)
        }
        if groundTruth?.datasetID == report?.datasetID,
           let benchmarkBestID = report?.results.first(where: { $0.rank == 1 })?.packageID {
            models = models.map { model in
                var updated = model
                updated.isBest = model.packageID == benchmarkBestID
                return updated
            }
            models.sort {
                if ($0.isActive ?? false) != ($1.isActive ?? false) { return $0.isActive ?? false }
                if ($0.isBest ?? false) != ($1.isBest ?? false) { return $0.isBest ?? false }
                return ($0.createdAt ?? "") > ($1.createdAt ?? "")
            }
        }

        return FolderLoadSnapshot(
            project: project,
            activeModel: activeModel,
            installedModelCount: installedCount,
            models: models,
            continuationCheckpoints: checkpoints,
            benchmarkGroundTruth: groundTruth,
            benchmarkReport: report,
            errorMessage: loadError
        )
    }

    private func refreshModelState() async {
        guard let folder = selectedFolder?.standardizedFileURL else { return }
        let snapshot = await Task.detached(priority: .utility) {
            Self.loadFolderSnapshot(from: folder)
        }.value
        guard selectedFolder?.standardizedFileURL == folder else { return }
        installedModelCount = snapshot.installedModelCount
        managedModels = snapshot.models
        continuationCheckpoints = snapshot.continuationCheckpoints
        activeModelPackageID = snapshot.activeModel?.packageID
        activeModelDescription = snapshot.activeModel?.description
    }

    func revealTrainingFolder() {
        guard let store else { return }
        do {
            try store.prepare()
            NSWorkspace.shared.activateFileViewerSelecting([store.visibleRootURL])
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func analyzeVideos() {
        guard let folder = selectedFolder, let store else { return }
        isWorking = true
        progress = 0
        errorMessage = nil
        status = tr("Suche Videos …", "Searching for videos …", "Buscando vídeos …", "Recherche des vidéos …")

        Task {
            do {
                let extractor = FrameExtractor()
                let videos = extractor.discoverVideos(in: folder)
                let extractedFrames = try await extractor.extract(videos: videos, into: store, framesPerVideo: framesPerVideo) { value, videoName in
                    await MainActor.run {
                        self.progress = value
                        self.status = self.tr(
                            "Extrahiere lokale Trainingsbilder: \(videoName)",
                            "Extracting local training images: \(videoName)",
                            "Extrayendo imágenes de entrenamiento locales: \(videoName)",
                            "Extraction des images d’entraînement locales : \(videoName)"
                        )
                    }
                }
                let previousByPath = Dictionary(
                    (project?.frames ?? []).map { ($0.relativePath, $0) },
                    uniquingKeysWith: { current, _ in current }
                )
                let frames = extractedFrames.map { fresh -> FrameRecord in
                    guard let previous = previousByPath[fresh.relativePath] else { return fresh }
                    var preserved = fresh
                    preserved.id = previous.id
                    preserved.annotations = previous.annotations
                    return preserved
                }
                var document = ProjectDocument(
                    name: folder.lastPathComponent,
                    sport: sport,
                    sourceFolder: folder.path
                )
                document.frames = frames
                document.framesPerVideo = framesPerVideo
                document.lastTraining = project?.lastTraining
                document.trainingHistory = project?.trainingHistory
                try store.save(document)
                project = document
                selectedFrameID = frames.first?.id
                selectedCategory = sport.categories.first ?? "ball"
                autoLabelCategories = [selectedCategory]
                progress = 1
                status = tr(
                    "\(frames.count) Frames erstellt. Videos und Bilder bleiben lokal.",
                    "Created \(frames.count) frames. Videos and images remain local.",
                    "Se crearon \(frames.count) fotogramas. Los vídeos y las imágenes permanecen localmente.",
                    "\(frames.count) images créées. Les vidéos et images restent locales."
                )
            } catch {
                errorMessage = error.localizedDescription
                status = tr("Analyse fehlgeschlagen.", "Analysis failed.", "El análisis ha fallado.", "Échec de l’analyse.")
            }
            isWorking = false
        }
    }

    var automaticAnnotationCountOnSelectedFrame: Int {
        selectedFrame?.annotations.filter { $0.source == "auto" }.count ?? 0
    }

    /// Marks every automatic suggestion on the selected frame as reviewed. Moving or
    /// resizing an individual box is not yet possible in this editor, so accepting a
    /// whole frame at once is the only way to clear source == "auto" without deleting
    /// and manually redrawing every box - which would defeat the point of auto-labeling.
    func acceptAllAutomaticAnnotations() {
        guard let frame = selectedFrame else { return }
        let updated = frame.annotations.map { annotation -> BoxAnnotation in
            var accepted = annotation
            if accepted.source == "auto" { accepted.source = "manual" }
            return accepted
        }
        updateAnnotations(for: frame.id, updated)
    }

    func rejectAllAutomaticAnnotations() {
        guard let frame = selectedFrame else { return }
        updateAnnotations(for: frame.id, frame.annotations.filter { $0.source != "auto" })
    }

    func updateAnnotations(for frameID: UUID, _ annotations: [BoxAnnotation]) {
        guard var document = project,
              let index = document.frames.firstIndex(where: { $0.id == frameID }),
              let store else { return }
        document.frames[index].annotations = annotations
        project = document
        do {
            try store.save(document)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func removeSelectedFrame() {
        guard var document = project,
              let frameID = selectedFrameID,
              let index = document.frames.firstIndex(where: { $0.id == frameID }),
              let store else { return }
        let frame = document.frames[index]
        do {
            let benchmarkInvalidated = try store.removeDerivedFrame(frame)
            document.frames.remove(at: index)
            try store.save(document)
            project = document
            selectedFrameID = document.frames.isEmpty ? nil : document.frames[min(index, document.frames.count - 1)].id
            if benchmarkInvalidated {
                benchmarkGroundTruth = nil
                benchmarkReport = nil
            }
            status = tr(
                "Trainingsbild entfernt. Das Quellvideo bleibt unverändert.",
                "Training image removed. The source video remains unchanged.",
                "Imagen de entrenamiento eliminada. El vídeo original no se modifica.",
                "Image d’entraînement retirée. La vidéo source reste inchangée."
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func checkHardware() { runWorkerAction(tr("Prüfe Mac-Hardware …", "Checking Mac hardware …")) { worker, output in
        let status = try await worker.doctor()
        await MainActor.run { self.hardware = status }
        await output(self.tr(
            "Hardware erkannt: \(status.recommendedDevice.uppercased())\n",
            "Hardware detected: \(status.recommendedDevice.uppercased())\n"
        ))
    }}

    func prepareEnvironment() { runWorkerAction(tr("Richte lokale ML-Umgebung ein …", "Setting up local ML environment …")) { worker, output in
        try await worker.prepareEnvironment(onOutput: output)
    }}

    func autoLabel() {
        guard !autoLabelCategories.isEmpty else { return }
        let categories = Array(autoLabelCategories)
        let categoryLabel = categories.map { language.category($0) }.sorted().joined(separator: ", ")
        runWorkerAction(
            tr(
                "Erkenne „\(categoryLabel)“ …",
                "Detecting “\(categoryLabel)” …",
                "Detectando «\(categoryLabel)» …",
                "Détection de « \(categoryLabel) » …"
            ),
            reloadProject: true,
            reloadedCategories: autoLabelCategories
        ) { worker, output in
            try await worker.autoLabel(
                modelSize: self.modelSize,
                categories: categories,
                threshold: self.confidenceThreshold,
                language: self.language,
                onOutput: output
            )
        }
    }

    func refineBoxes() { runWorkerAction(
        tr(
            "Prüfe automatische Ball- und Puck-Boxen lokal mit OpenCV …",
            "Reviewing automatic ball and puck boxes locally with OpenCV …",
            "Revisando localmente con OpenCV los cuadros automáticos de balón y disco …",
            "Vérification locale avec OpenCV des boîtes automatiques ballon et palet …"
        ),
        reloadProject: true
    ) { worker, output in
        try await worker.refineBoxes(language: self.language, onOutput: output)
    }}

    func startActiveLearning() {
        guard selectedFolder != nil, store != nil else { return }
        localPickerPurpose = .activeLearningFolder
    }

    private func startActiveLearning(from folder: URL, selectedFolder: URL, store: ProjectStore) {
        guard folder.standardizedFileURL != selectedFolder.standardizedFileURL else {
            errorMessage = tr("Bitte einen anderen Ordner als den bisherigen Trainingsordner wählen.", "Choose a folder different from the existing training folder.", "Elige una carpeta distinta de la carpeta de entrenamiento.", "Choisissez un dossier différent du dossier d’entraînement.")
            return
        }
        isWorking = true
        progress = 0
        errorMessage = nil
        status = tr("Neue Videos werden lokal vorbereitet …", "Preparing new videos locally …", "Preparando vídeos nuevos localmente …", "Préparation locale des nouvelles vidéos …")
        Task {
            do {
                let extractor = FrameExtractor()
                let videos = extractor.discoverVideos(in: folder)
                var fresh = try await extractor.extract(videos: videos, into: store, framesPerVideo: min(framesPerVideo, 500)) { value, name in
                    await MainActor.run {
                        self.progress = value * 0.45
                        self.status = self.tr("Extrahiere Prüfkandidaten: \(name)", "Extracting review candidates: \(name)", "Extrayendo candidatos: \(name)", "Extraction des candidats : \(name)")
                    }
                }
                var document = project ?? ProjectDocument(name: selectedFolder.lastPathComponent, sport: sport, sourceFolder: selectedFolder.path)
                let existingPaths = Set(document.frames.map(\.relativePath))
                fresh.removeAll { existingPaths.contains($0.relativePath) }
                guard !fresh.isEmpty else {
                    throw NSError(domain: "RecoActiveLearning", code: 1, userInfo: [NSLocalizedDescriptionKey: tr("Alle gewählten Videos wurden bereits verwendet. Bitte neue Videos auswählen.", "All selected videos were already used. Choose new videos.", "Todos los vídeos seleccionados ya se utilizaron. Elige vídeos nuevos.", "Toutes les vidéos sélectionnées ont déjà été utilisées. Choisissez de nouvelles vidéos.")])
                }
                for index in fresh.indices { fresh[index].reviewStatus = "candidate" }
                document.frames.append(contentsOf: fresh)
                try store.save(document)
                project = document
                status = tr("Lokale Ballerkennung läuft …", "Running local ball detection …", "Ejecutando detección local …", "Détection locale du ballon …")
                let worker = MLWorker(projectRoot: store.rootURL)
                try await worker.autoLabel(modelSize: modelSize, categories: [selectedCategory], threshold: 0.12, candidateOnly: true, language: language) { chunk in
                    await MainActor.run { self.log += chunk }
                }
                let loaded = try store.load()
                project = loaded
                selectedFrameID = loaded.frames.first(where: { $0.reviewStatus == "candidate" })?.id
                progress = 1
                status = tr("Prüfwarteschlange bereit. Nur bestätigte Bilder gelangen ins Training.", "Review queue ready. Only confirmed images enter training.", "Cola lista. Solo las imágenes confirmadas pasan al entrenamiento.", "File prête. Seules les images confirmées entrent dans l’entraînement.")
            } catch {
                errorMessage = error.localizedDescription
                status = tr("Datensatzerweiterung fehlgeschlagen.", "Dataset expansion failed.", "Error al ampliar el conjunto.", "Échec de l’extension du jeu.")
            }
            isWorking = false
        }
    }

    func reviewSelectedCandidate(asBall: Bool) {
        guard var document = project,
              let frameID = selectedFrameID,
              let index = document.frames.firstIndex(where: { $0.id == frameID && $0.reviewStatus == "candidate" }),
              let store else { return }
        if asBall {
            guard document.frames[index].annotations.contains(where: { $0.category == selectedCategory }) else {
                errorMessage = tr("Zuerst eine passende Ball-Box einzeichnen oder korrigieren.", "Draw or correct a matching ball box first.", "Primero dibuja o corrige un cuadro del balón.", "Dessinez ou corrigez d’abord une boîte du ballon.")
                return
            }
            document.frames[index].annotations = document.frames[index].annotations.map { annotation in
                var accepted = annotation
                if accepted.category == selectedCategory { accepted.source = "manual" }
                return accepted
            }
        } else {
            document.frames[index].annotations.removeAll { $0.category == selectedCategory }
        }
        document.frames[index].reviewStatus = "reviewed"
        do {
            try store.save(document)
            project = document
            selectedFrameID = document.frames.first(where: { $0.reviewStatus == "candidate" })?.id ?? frameID
            status = tr("Geprüftes Bild wurde übernommen.", "Reviewed image added to training.", "Imagen revisada añadida al entrenamiento.", "Image vérifiée ajoutée à l’entraînement.")
        } catch { errorMessage = error.localizedDescription }
    }

    func train() { runWorkerAction(tr("Trainiere lokal …", "Training locally …")) { worker, output in
        try await worker.train(modelSize: self.modelSize, epochs: self.epochs, language: self.language, onOutput: output)
    }}

    func exportCPU() { runWorkerAction(tr("Exportiere universelles CPU-Modell …", "Exporting universal CPU model …")) { worker, output in
        try await worker.exportCPU(modelSize: self.modelSize, language: self.language, onOutput: output)
    }}

    func exportCoreML() { runWorkerAction(tr("Exportiere Core-ML-Modell …", "Exporting Core ML model …")) { worker, output in
        try await worker.exportCoreML(modelSize: self.modelSize, language: self.language, onOutput: output)
    }}

    func packageModel(name: String) { runWorkerAction(tr("Erstelle datenschutzsicheres Austauschpaket …", "Creating privacy-safe exchange package …")) { worker, output in
        try await worker.packageModel(modelSize: self.modelSize, name: name, language: self.language, onOutput: output)
    }}

    func freezeBenchmarkGroundTruth() {
        guard let project, let store else { return }
        do {
            let reviewedProjectFrames = project.frames.filter { $0.reviewStatus != "candidate" }
            guard !reviewedProjectFrames.isEmpty else {
                throw NSError(domain: "RecoBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: tr("Das Projekt enthält keine Testbilder.", "The project contains no test images.", "El proyecto no contiene imágenes de prueba.", "Le projet ne contient aucune image de test.")])
            }
            guard !hasUnreviewedTrainingAnnotations else {
                throw NSError(domain: "RecoBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: tr("Vor dem Modelltest alle automatischen Vorschläge übernehmen, korrigieren oder verwerfen.", "Accept, correct, or reject every automatic suggestion before benchmarking.", "Acepta, corrige o rechaza todas las sugerencias automáticas antes de comparar.", "Acceptez, corrigez ou refusez toutes les suggestions automatiques avant la comparaison.")])
            }
            let frames = reviewedProjectFrames.map { frame in
                BenchmarkFrameRecord(
                    id: frame.id.uuidString,
                    relativePath: frame.relativePath,
                    width: frame.width,
                    height: frame.height,
                    annotations: frame.annotations.map {
                        BenchmarkAnnotationRecord(category: $0.category, x: $0.x, y: $0.y, width: $0.width, height: $0.height)
                    }
                )
            }
            let count = frames.reduce(0) { $0 + $1.annotations.count }
            guard count > 0 else {
                throw NSError(domain: "RecoBenchmark", code: 3, userInfo: [NSLocalizedDescriptionKey: tr("Mindestens eine richtige Box muss festgelegt sein.", "At least one correct box must be defined.", "Debe definirse al menos un cuadro correcto.", "Au moins une boîte correcte doit être définie.")])
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let identity = try encoder.encode(BenchmarkDatasetIdentity(sport: project.sport.rawValue, frames: frames))
            let digest = SHA256.hash(data: identity).map { String(format: "%02x", $0) }.joined()
            let reference = BenchmarkGroundTruthRecord(
                schemaVersion: 1,
                createdAt: ISO8601DateFormatter().string(from: Date()),
                sport: project.sport.rawValue,
                datasetID: digest,
                frames: frames,
                reviewStatement: "Every frame was explicitly frozen as ground truth; frames without boxes are intentional negatives."
            )
            let directory = store.rootURL.appending(path: "benchmarks", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let prettyEncoder = JSONEncoder()
            prettyEncoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try prettyEncoder.encode(reference).write(to: directory.appending(path: "ground-truth.json"), options: .atomic)
            benchmarkGroundTruth = reference
            status = tr("Referenz mit \(frames.count) Bildern und \(count) Boxen festgelegt.", "Ground truth frozen with \(frames.count) images and \(count) boxes.", "Referencia fijada con \(frames.count) imágenes y \(count) cuadros.", "Référence figée avec \(frames.count) images et \(count) boîtes.")
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    func benchmarkModels() { runWorkerAction(tr("Vergleiche alle lokalen Modelle …", "Comparing every local model …", "Comparando todos los modelos locales …", "Comparaison de tous les modèles locaux …")) { worker, output in
        try await worker.benchmark(threshold: self.benchmarkThreshold, language: self.language, onOutput: output)
        await MainActor.run { self.loadBenchmarkState() }
    }}

    func importModelPackage() {
        guard store != nil else { return }
        localPickerPurpose = .modelPackage
    }

    private func importModelPackage(from fileURL: URL, store: ProjectStore) {
        isWorking = true
        errorMessage = nil
        status = tr("Prüfe und importiere Modellpaket …", "Validating and importing model package …")
        Task {
            do {
                let imported = try await MLWorker(projectRoot: store.rootURL)
                    .installModelPackage(fileURL: fileURL, language: language)
                activeModelPackageID = imported.active.packageID
                activeModelDescription = imported.active.description
                if let importedSize = ModelSize(rawValue: imported.active.modelSize) {
                    modelSize = importedSize
                }
                await refreshModelState()
                switch language {
                case .de: status = "Modell „\(imported.active.packageID)“ geprüft, installiert und aktiviert."
                case .en: status = "Model “\(imported.active.packageID)” validated, installed, and activated."
                case .es: status = "Modelo «\(imported.active.packageID)» validado, instalado y activado."
                case .fr: status = "Modèle « \(imported.active.packageID) » validé, installé et activé."
                }
            } catch {
                errorMessage = error.localizedDescription
                status = tr("Modellimport fehlgeschlagen.", "Model import failed.")
            }
            isWorking = false
        }
    }

    func activateModel(_ model: ManagedModelRecord) {
        guard let store else { return }
        isWorking = true
        errorMessage = nil
        Task {
            do {
                let response = try await MLWorker(projectRoot: store.rootURL).activateModel(packageID: model.packageID, language: language)
                if let size = ModelSize(rawValue: response.active.modelSize) { modelSize = size }
                await refreshModelState()
                status = tr("Modell aktiviert: \(model.displayName ?? model.packageID)", "Model activated: \(model.displayName ?? model.packageID)", "Modelo activado: \(model.displayName ?? model.packageID)", "Modèle activé : \(model.displayName ?? model.packageID)")
            } catch {
                errorMessage = error.localizedDescription
            }
            isWorking = false
        }
    }

    func renameModel(_ model: ManagedModelRecord, to name: String) {
        guard let store else { return }
        isWorking = true
        errorMessage = nil
        Task {
            do {
                try await MLWorker(projectRoot: store.rootURL).renameModel(packageID: model.packageID, name: name, language: language)
                await refreshModelState()
                status = tr("Modell umbenannt.", "Model renamed.", "Modelo renombrado.", "Modèle renommé.")
            } catch { errorMessage = error.localizedDescription }
            isWorking = false
        }
    }

    func deleteModel(_ model: ManagedModelRecord) {
        guard let store else { return }
        isWorking = true
        errorMessage = nil
        Task {
            do {
                try await MLWorker(projectRoot: store.rootURL).deleteModel(packageID: model.packageID, language: language)
                await refreshModelState()
                status = tr("Modell gelöscht.", "Model deleted.", "Modelo eliminado.", "Modèle supprimé.")
            } catch { errorMessage = error.localizedDescription }
            isWorking = false
        }
    }

    private func runWorkerAction(
        _ initialStatus: String,
        reloadProject: Bool = false,
        reloadedCategories: Set<String>? = nil,
        action: @escaping (MLWorker, @escaping @Sendable (String) async -> Void) async throws -> Void
    ) {
        guard let store else { return }
        isWorking = true
        errorMessage = nil
        status = initialStatus
        log = ""
        let worker = MLWorker(projectRoot: store.rootURL)
        Task {
            do {
                try await action(worker) { chunk in
                    await MainActor.run {
                        self.log += chunk
                        if self.log.count > 30_000 { self.log.removeFirst(self.log.count - 30_000) }
                    }
                }
                await refreshModelState()
                if reloadProject, let loaded = try? store.load() {
                    project = loaded
                    let automaticCount = loaded.frames
                        .flatMap(\.annotations)
                        .filter { $0.source == "auto" && (reloadedCategories == nil || reloadedCategories!.contains($0.category)) }
                        .count
                    let categoryLabel = (reloadedCategories ?? []).map { language.category($0) }.sorted().joined(separator: ", ")
                    status = automaticCount > 0
                        ? tr(
                            "\(automaticCount) automatische Markierungen für „\(categoryLabel)“ geladen. Bitte prüfen.",
                            "Loaded \(automaticCount) automatic “\(categoryLabel)” annotations. Please review them.",
                            "Se cargaron \(automaticCount) anotaciones automáticas de «\(categoryLabel)». Revísalas.",
                            "\(automaticCount) annotations automatiques « \(categoryLabel) » chargées. Veuillez les vérifier."
                        )
                        : tr(
                            "Keine automatischen Treffer für diese Klasse gefunden. Details stehen im Protokoll.",
                            "No automatic detections found for this class. See the log for details.",
                            "No se encontraron detecciones automáticas para esta clase. Consulta el registro.",
                            "Aucune détection automatique pour cette classe. Consultez le journal."
                        )
                } else {
                    status = tr("Fertig.", "Done.")
                }
            } catch {
                errorMessage = error.localizedDescription
                status = tr("Vorgang fehlgeschlagen.", "Operation failed.")
            }
            isWorking = false
        }
    }

    private func loadBenchmarkState() {
        guard let root = store?.rootURL.appending(path: "benchmarks", directoryHint: .isDirectory) else { return }
        let decoder = JSONDecoder()
        if let data = try? Data(contentsOf: root.appending(path: "ground-truth.json")) {
            benchmarkGroundTruth = try? decoder.decode(BenchmarkGroundTruthRecord.self, from: data)
        } else {
            benchmarkGroundTruth = nil
        }
        if let data = try? Data(contentsOf: root.appending(path: "latest.json")) {
            benchmarkReport = try? decoder.decode(BenchmarkReport.self, from: data)
        } else {
            benchmarkReport = nil
        }
    }
}
