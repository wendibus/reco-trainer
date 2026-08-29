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

struct ProjectStore {
    static let projectDirectoryName = ".reco-training"
    static let projectFileName = "project.json"

    let rootURL: URL

    var framesURL: URL { rootURL.appending(path: "frames", directoryHint: .isDirectory) }
    var datasetURL: URL { rootURL.appending(path: "dataset", directoryHint: .isDirectory) }
    var runsURL: URL { rootURL.appending(path: "runs", directoryHint: .isDirectory) }
    var exportsURL: URL { rootURL.appending(path: "exports", directoryHint: .isDirectory) }
    var backupsURL: URL { rootURL.appending(path: "backups", directoryHint: .isDirectory) }
    var projectFileURL: URL { rootURL.appending(path: Self.projectFileName) }

    static func forSourceFolder(_ folder: URL) -> ProjectStore {
        ProjectStore(rootURL: folder.appending(path: projectDirectoryName, directoryHint: .isDirectory))
    }

    func prepare() throws {
        let manager = FileManager.default
        for directory in [rootURL, framesURL, datasetURL, runsURL, exportsURL] {
            try manager.createDirectory(at: directory, withIntermediateDirectories: true)
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
}
