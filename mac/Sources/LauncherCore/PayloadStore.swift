import Foundation

public struct PayloadManifest: Codable, Equatable, Sendable {
    public let version: String
    /// Oldest launcher (app) version that may run this payload.
    public let minLauncher: String

    public init(version: String, minLauncher: String) {
        self.version = version
        self.minLauncher = minLauncher
    }
}

public struct Payload: Equatable, Sendable {
    public let manifest: PayloadManifest
    public let directory: URL

    public var version: String { manifest.version }
    public var workerScript: URL { directory.appendingPathComponent("local_worker.py") }
}

public enum PayloadError: Error, LocalizedError, Equatable {
    case extractionFailed(String)
    case invalidPayload(String)
    case launcherTooOld(required: String)
    case unsafeContent(String)

    public var errorDescription: String? {
        switch self {
        case .extractionFailed(let detail): return "The update archive could not be unpacked. \(detail)"
        case .invalidPayload(let detail): return "The update is incomplete: \(detail)"
        case .launcherTooOld(let required): return "This update needs Reco Trainer \(required) or newer."
        case .unsafeContent(let detail): return "The update contains unsafe content: \(detail)"
        }
    }
}

/// Keeps the downloaded "payload" (web interface + Python workers) in
/// `~/Library/Application Support/Reco Trainer/payloads/<version>/`, always runnable from a
/// writable location, and seeds it from the copy bundled inside the app.
public final class PayloadStore {
    public let installRoot: URL
    public let bundledRoot: URL?
    public let launcherVersion: String
    private let fileManager = FileManager.default

    /// Files a folder must contain to count as a payload (mirrored by scripts/build-payload.sh and its test).
    static let requiredFiles = ["payload.json", "local_worker.py", "ml_worker.py", "ui/index.html"]

    public init(installRoot: URL, bundledRoot: URL?, launcherVersion: String) {
        self.installRoot = installRoot
        self.bundledRoot = bundledRoot
        self.launcherVersion = launcherVersion
    }

    public static var defaultInstallRoot: URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        return support
            .appendingPathComponent("Reco Trainer", isDirectory: true)
            .appendingPathComponent("payloads", isDirectory: true)
    }

    // MARK: Reading

    /// Returns the payload in `directory`, or nil when it is incomplete or its manifest is not valid.
    public func loadPayload(at directory: URL) -> Payload? {
        for relative in Self.requiredFiles
        where !fileManager.fileExists(atPath: directory.appendingPathComponent(relative).path) {
            return nil
        }
        guard
            let data = try? Data(contentsOf: directory.appendingPathComponent("payload.json")),
            let manifest = try? JSONDecoder().decode(PayloadManifest.self, from: data),
            Self.isSafeVersion(manifest.version),
            Self.isSafeVersion(manifest.minLauncher)
        else { return nil }
        return Payload(manifest: manifest, directory: directory)
    }

    static func isSafeVersion(_ value: String) -> Bool {
        value.range(of: #"^[0-9]+(\.[0-9]+){1,3}$"#, options: .regularExpression) != nil
    }

    /// Valid installed payloads, newest first.
    public func installedPayloads() -> [Payload] {
        let urls = (try? fileManager.contentsOfDirectory(
            at: installRoot,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        return urls
            .compactMap { loadPayload(at: $0) }
            .filter { $0.directory.lastPathComponent == $0.version }
            .sorted { isNewerVersion($0.version, than: $1.version) }
    }

    /// Installed payloads this launcher may run: new enough launcher, not marked as failing.
    public func usablePayloads() -> [Payload] {
        let bad = badMarks()
        return installedPayloads().filter {
            !isNewerVersion($0.manifest.minLauncher, than: launcherVersion)
                && !bad.contains(badKey(version: $0.version))
        }
    }

    public func activePayload() -> Payload? { usablePayloads().first }

    // MARK: Failing versions

    private var badFile: URL { installRoot.appendingPathComponent(".bad-versions") }

    private func badKey(version: String) -> String { "\(launcherVersion)|\(version)" }

    private func badMarks() -> Set<String> {
        guard let text = try? String(contentsOf: badFile, encoding: .utf8) else { return [] }
        return Set(text.split(separator: "\n").map(String.init))
    }

    /// Remember that `version` failed to start with this launcher, so the next start falls back
    /// to an older payload. The mark is tied to the launcher version, so a new app gives it a fresh chance.
    public func markBad(version: String) {
        try? fileManager.createDirectory(at: installRoot, withIntermediateDirectories: true)
        var marks = badMarks()
        marks.insert(badKey(version: version))
        try? marks.sorted().joined(separator: "\n").write(to: badFile, atomically: true, encoding: .utf8)
    }

    // MARK: Installing

    /// Copies the payload bundled in the app into the install folder unless an equal or newer one is installed.
    @discardableResult
    public func seedBundledPayloadIfNeeded() throws -> Payload? {
        guard let bundledRoot, let bundled = loadPayload(at: bundledRoot) else { return nil }
        if let newest = installedPayloads().first, !isNewerVersion(bundled.version, than: newest.version) {
            return nil
        }
        try fileManager.createDirectory(at: installRoot, withIntermediateDirectories: true)
        let staging = installRoot.appendingPathComponent(".seed-\(UUID().uuidString)", isDirectory: true)
        try fileManager.copyItem(at: bundledRoot, to: staging)
        return try moveIntoPlace(content: staging, version: bundled.version)
    }

    /// Unpacks a (signature-checked) payload archive and installs it as `<version>/`.
    public func install(zip: URL) throws -> Payload {
        try fileManager.createDirectory(at: installRoot, withIntermediateDirectories: true)
        let staging = installRoot.appendingPathComponent(".staging-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: staging) }

        try Self.unzip(zip, to: staging)
        try rejectLinks(in: staging)

        var content = staging
        if !fileManager.fileExists(atPath: staging.appendingPathComponent("payload.json").path) {
            let children = (try? fileManager.contentsOfDirectory(at: staging, includingPropertiesForKeys: nil)) ?? []
            if children.count == 1 { content = children[0] }
        }
        guard let payload = loadPayload(at: content) else {
            throw PayloadError.invalidPayload("payload.json, local_worker.py, ml_worker.py or ui/index.html is missing.")
        }
        if isNewerVersion(payload.manifest.minLauncher, than: launcherVersion) {
            throw PayloadError.launcherTooOld(required: payload.manifest.minLauncher)
        }
        return try moveIntoPlace(content: content, version: payload.version)
    }

    private func moveIntoPlace(content: URL, version: String) throws -> Payload {
        let destination = installRoot.appendingPathComponent(version, isDirectory: true)
        if fileManager.fileExists(atPath: destination.path) { try fileManager.removeItem(at: destination) }
        try fileManager.moveItem(at: content, to: destination)
        guard let payload = loadPayload(at: destination) else {
            throw PayloadError.invalidPayload("The installed copy could not be read back.")
        }
        return payload
    }

    /// Keeps the `keeping` newest payloads (plus any in `protecting`) and removes the rest.
    public func prune(keeping: Int, protecting: Set<String> = []) {
        for payload in installedPayloads().dropFirst(keeping) where !protecting.contains(payload.version) {
            try? fileManager.removeItem(at: payload.directory)
        }
    }

    private func rejectLinks(in directory: URL) throws {
        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isSymbolicLinkKey],
            options: []
        ) else { return }
        for case let url as URL in enumerator {
            if (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true {
                throw PayloadError.unsafeContent(url.lastPathComponent)
            }
        }
    }

    static func unzip(_ zip: URL, to directory: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        process.arguments = ["-x", "-k", zip.path, directory.path]
        let errors = Pipe()
        process.standardError = errors
        try process.run()
        let detail = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw PayloadError.extractionFailed(String(data: detail, encoding: .utf8) ?? "")
        }
    }
}
