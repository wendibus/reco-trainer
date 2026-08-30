import AppKit
import Combine
import CryptoKit
import Foundation
import UniformTypeIdentifiers

@MainActor
final class AppState: ObservableObject {
    @Published var language: AppLanguage = .de
    @Published var sport: Sport = .football
    @Published var selectedFolder: URL?
    @Published var project: ProjectDocument?
    @Published var selectedFrameID: UUID?
    @Published var selectedCategory = "ball"
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

    var store: ProjectStore? {
        selectedFolder.map(ProjectStore.forSourceFolder)
    }

    var selectedFrame: FrameRecord? {
        guard let id = selectedFrameID else { return project?.frames.first }
        return project?.frames.first(where: { $0.id == id })
    }

    var installedModelCount: Int {
        guard let root = store?.rootURL.appending(path: "models/library", directoryHint: .isDirectory),
              let directories = try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { return 0 }
        return directories.filter { FileManager.default.fileExists(atPath: $0.appending(path: "manifest.json").path) }.count
    }

    var benchmarkReportIsCurrent: Bool {
        benchmarkReport?.datasetID == benchmarkGroundTruth?.datasetID
    }

    func tr(_ german: String, _ english: String, _ spanish: String? = nil, _ french: String? = nil) -> String {
        language.text(german, english, spanish, french)
    }

    func chooseFolder() {
        let panel = NSOpenPanel()
        panel.title = tr("Ordner mit Sportvideos auswählen", "Select folder containing sports videos")
        panel.prompt = tr("Ordner verwenden", "Use folder")
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        selectedFolder = url
        loadExistingProjectIfPresent()
    }

    func loadExistingProjectIfPresent() {
        guard let store else { return }
        if let loaded = try? store.load() {
            project = loaded
            sport = loaded.sport
            selectedCategory = loaded.sport.categories.first ?? "ball"
            selectedFrameID = loaded.frames.first?.id
            framesPerVideo = loaded.framesPerVideo ?? 240
            let activeURL = store.rootURL.appending(path: "models/active.json")
            if let data = try? Data(contentsOf: activeURL),
               let active = try? JSONDecoder().decode(ActiveModelRecord.self, from: data) {
                activeModelPackageID = active.packageID
                activeModelDescription = active.description
                modelSize = ModelSize(rawValue: active.modelSize) ?? modelSize
            } else {
                activeModelPackageID = nil
                activeModelDescription = nil
            }
            loadBenchmarkState()
            status = tr(
                "Projekt geladen: \(loaded.frames.count) Frames.",
                "Project loaded: \(loaded.frames.count) frames.",
                "Proyecto cargado: \(loaded.frames.count) fotogramas.",
                "Projet chargé : \(loaded.frames.count) images."
            )
        } else {
            project = nil
            selectedFrameID = nil
            selectedCategory = sport.categories.first ?? "ball"
            benchmarkGroundTruth = nil
            benchmarkReport = nil
            status = tr("Ordner gewählt. Jetzt Videos analysieren.", "Folder selected. Analyze the videos next.", "Carpeta seleccionada. Analiza ahora los vídeos.", "Dossier sélectionné. Analysez maintenant les vidéos.")
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

    func autoLabel() { runWorkerAction(
        tr(
            "Erkenne nur die Klasse „\(language.category(selectedCategory))“ …",
            "Detecting only the “\(language.category(selectedCategory))” class …",
            "Detectando únicamente la clase «\(language.category(selectedCategory))» …",
            "Détection de la classe « \(language.category(selectedCategory)) » uniquement …"
        ),
        reloadProject: true,
        reloadedCategory: selectedCategory
    ) { worker, output in
        try await worker.autoLabel(
            modelSize: self.modelSize,
            category: self.selectedCategory,
            threshold: self.confidenceThreshold,
            language: self.language,
            onOutput: output
        )
    }}

    func train() { runWorkerAction(tr("Trainiere lokal …", "Training locally …")) { worker, output in
        try await worker.train(modelSize: self.modelSize, epochs: self.epochs, language: self.language, onOutput: output)
    }}

    func exportCPU() { runWorkerAction(tr("Exportiere universelles CPU-Modell …", "Exporting universal CPU model …")) { worker, output in
        try await worker.exportCPU(modelSize: self.modelSize, language: self.language, onOutput: output)
    }}

    func exportCoreML() { runWorkerAction(tr("Exportiere Core-ML-Modell …", "Exporting Core ML model …")) { worker, output in
        try await worker.exportCoreML(modelSize: self.modelSize, language: self.language, onOutput: output)
    }}

    func packageModel() { runWorkerAction(tr("Erstelle datenschutzsicheres Austauschpaket …", "Creating privacy-safe exchange package …")) { worker, output in
        try await worker.packageModel(modelSize: self.modelSize, language: self.language, onOutput: output)
    }}

    func freezeBenchmarkGroundTruth() {
        guard let project, let store else { return }
        do {
            guard !project.frames.isEmpty else {
                throw NSError(domain: "RecoBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: tr("Das Projekt enthält keine Testbilder.", "The project contains no test images.", "El proyecto no contiene imágenes de prueba.", "Le projet ne contient aucune image de test.")])
            }
            guard !project.frames.flatMap(\.annotations).contains(where: { $0.source == "auto" }) else {
                throw NSError(domain: "RecoBenchmark", code: 2, userInfo: [NSLocalizedDescriptionKey: tr("Vor dem Modelltest alle automatischen Vorschläge übernehmen, korrigieren oder verwerfen.", "Accept, correct, or reject every automatic suggestion before benchmarking.", "Acepta, corrige o rechaza todas las sugerencias automáticas antes de comparar.", "Acceptez, corrigez ou refusez toutes les suggestions automatiques avant la comparaison.")])
            }
            let frames = project.frames.map { frame in
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
        guard let store else { return }
        let panel = NSOpenPanel()
        panel.title = tr("Reco-Modellpaket auswählen", "Select Reco model package")
        panel.prompt = tr("Modell importieren", "Import model")
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "recomodel") ?? .data]
        guard panel.runModal() == .OK, let fileURL = panel.url else { return }

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

    private func runWorkerAction(
        _ initialStatus: String,
        reloadProject: Bool = false,
        reloadedCategory: String? = nil,
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
                if reloadProject, let loaded = try? store.load() {
                    project = loaded
                    let automaticCount = loaded.frames
                        .flatMap(\.annotations)
                        .filter { $0.source == "auto" && (reloadedCategory == nil || $0.category == reloadedCategory) }
                        .count
                    status = automaticCount > 0
                        ? tr(
                            "\(automaticCount) automatische Markierungen für „\(language.category(reloadedCategory ?? ""))“ geladen. Bitte prüfen.",
                            "Loaded \(automaticCount) automatic “\(language.category(reloadedCategory ?? ""))” annotations. Please review them.",
                            "Se cargaron \(automaticCount) anotaciones automáticas de «\(language.category(reloadedCategory ?? ""))». Revísalas.",
                            "\(automaticCount) annotations automatiques « \(language.category(reloadedCategory ?? "")) » chargées. Veuillez les vérifier."
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
