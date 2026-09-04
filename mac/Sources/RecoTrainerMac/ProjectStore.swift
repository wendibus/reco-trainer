import Foundation

enum ProjectStoreError: LocalizedError {
    case noProject
    case invalidProject

    var errorDescription: String? {
        switch self {
        case .noProject: "Es wurde noch kein Projekt angelegt."
        case .invalidProject: "Die Projektdatei konnte nicht gelesen werden."
        }
    }
}

struct ProjectStore: Sendable {
    static let projectDirectoryName = ".reco-training"
    static let visibleProjectDirectoryName = "Reco Training"
    static let projectFileName = "project.json"

    let rootURL: URL

    var framesURL: URL { rootURL.appending(path: "frames", directoryHint: .isDirectory) }
    var datasetURL: URL { rootURL.appending(path: "dataset", directoryHint: .isDirectory) }
    var runsURL: URL { rootURL.appending(path: "runs", directoryHint: .isDirectory) }
    var exportsURL: URL { rootURL.appending(path: "exports", directoryHint: .isDirectory) }
    var backupsURL: URL { rootURL.appending(path: "backups", directoryHint: .isDirectory) }
    var benchmarksURL: URL { rootURL.appending(path: "benchmarks", directoryHint: .isDirectory) }
    var modelsURL: URL { rootURL.appending(path: "models", directoryHint: .isDirectory) }
    var inboxURL: URL { rootURL.appending(path: "inbox", directoryHint: .isDirectory) }
    var visibleRootURL: URL {
        rootURL.deletingLastPathComponent().appending(path: Self.visibleProjectDirectoryName, directoryHint: .isDirectory)
    }
    var projectFileURL: URL { rootURL.appending(path: Self.projectFileName) }

    static func forSourceFolder(_ folder: URL) -> ProjectStore {
        ProjectStore(rootURL: folder.appending(path: projectDirectoryName, directoryHint: .isDirectory))
    }

    func prepare() throws {
        let manager = FileManager.default
        for directory in [rootURL, framesURL, datasetURL, runsURL, exportsURL, backupsURL, benchmarksURL, modelsURL, inboxURL] {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        try createVisibleProjectShortcutIfNeeded()
    }

    /// Finder hides `.reco-training` because its name starts with a dot. Keep the
    /// stable internal path for compatibility and expose a permanent, visible
    /// shortcut beside it instead of changing a system-wide Finder preference.
    private func createVisibleProjectShortcutIfNeeded() throws {
        guard rootURL.lastPathComponent == Self.projectDirectoryName else { return }
        let manager = FileManager.default
        if manager.fileExists(atPath: visibleRootURL.path) {
            return
        }
        try manager.createSymbolicLink(
            atPath: visibleRootURL.path,
            withDestinationPath: Self.projectDirectoryName
        )
    }

    func preferredTrainingCheckpoint(for modelSize: ModelSize) -> URL? {
        let modelRunURL = runsURL.appending(path: modelSize.rawValue, directoryHint: .isDirectory)
        let preferredNames = ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint.pth", "last.ckpt"]
        for name in preferredNames {
            let candidate = modelRunURL.appending(path: name)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: modelRunURL,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { return nil }
        return files
            .filter { ["pth", "ckpt"].contains($0.pathExtension.lowercased()) }
            .max {
                let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return left < right
            }
    }

    func save(_ project: ProjectDocument) throws {
        try prepare()
        try backupCurrentProject()
        var current = project
        current.updatedAt = Date()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(current)
        try data.write(to: projectFileURL, options: .atomic)
    }

    private func backupCurrentProject() throws {
        let manager = FileManager.default
        guard manager.fileExists(atPath: projectFileURL.path) else { return }
        try manager.createDirectory(at: backupsURL, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        let destination = backupsURL.appending(path: "project-\(formatter.string(from: Date()))-\(UUID().uuidString.prefix(8)).json")
        try manager.copyItem(at: projectFileURL, to: destination)
        let backups = try manager.contentsOfDirectory(at: backupsURL, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("project-") && $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
        for old in backups.dropFirst(30) { try? manager.removeItem(at: old) }
    }

    func load() throws -> ProjectDocument {
        guard FileManager.default.fileExists(atPath: projectFileURL.path) else {
            throw ProjectStoreError.noProject
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        do {
            return try decoder.decode(ProjectDocument.self, from: Data(contentsOf: projectFileURL))
        } catch {
            throw ProjectStoreError.invalidProject
        }
    }

    func frameURL(for frame: FrameRecord) -> URL {
        rootURL.appending(path: frame.relativePath)
    }

    func removeDerivedFrame(_ frame: FrameRecord) throws -> Bool {
        let manager = FileManager.default
        let target = frameURL(for: frame).standardizedFileURL
        guard target.path.hasPrefix(framesURL.standardizedFileURL.path + "/") else {
            throw ProjectStoreError.invalidProject
        }
        if manager.fileExists(atPath: target.path) { try manager.removeItem(at: target) }
        for split in ["train", "valid", "test"] {
            let copy = datasetURL.appending(path: split, directoryHint: .isDirectory).appending(path: target.lastPathComponent)
            if manager.fileExists(atPath: copy.path) { try manager.removeItem(at: copy) }
        }
        var invalidated = false
        if frame.reviewStatus != "candidate" {
            for name in ["ground-truth.json", "latest.json"] {
                let file = rootURL.appending(path: "benchmarks", directoryHint: .isDirectory).appending(path: name)
                if manager.fileExists(atPath: file.path) {
                    try manager.removeItem(at: file)
                    invalidated = true
                }
            }
        }
        return invalidated
    }
}
