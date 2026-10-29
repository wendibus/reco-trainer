import Foundation
import LauncherCore

enum UpdaterError: Error, LocalizedError {
    case badSignature
    case notConfigured

    var errorDescription: String? {
        switch self {
        case .badSignature:
            return L("Die Signatur des Updates ist ungültig. Es wurde nicht installiert.", "The update's signature is invalid. It was not installed.")
        case .notConfigured:
            return L("Automatische Updates sind in dieser App-Version nicht aktiviert.", "Automatic updates are not enabled in this build of the app.")
        }
    }
}

/// Fetches the newest GitHub release and installs its signed payload. Only GitHub's public releases
/// API and the release files are contacted - no project data is sent.
enum Updater {
    static var isConfigured: Bool { !PayloadKey.publicKeyBase64.isEmpty }

    /// For local update tests only: RECO_UPDATE_FEED points at a release JSON (file:// or https://)
    /// whose assets may then be file:// URLs. The Ed25519 signature check is the same as for GitHub.
    static let feedOverride: URL? = ProcessInfo.processInfo.environment["RECO_UPDATE_FEED"].flatMap(URL.init(string:))
    static var allowsFileURLs: Bool { feedOverride != nil }

    static func fetchLatestRelease() async throws -> Release {
        if let feed = feedOverride, feed.isFileURL {
            return try JSONDecoder().decode(Release.self, from: Data(contentsOf: feed))
        }
        var request = URLRequest(url: feedOverride ?? URL(string: "https://api.github.com/repos/wendibus/reco-trainer/releases/latest")!)
        request.setValue("RecoTrainerMac", forHTTPHeaderField: "User-Agent")
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(Release.self, from: data)
    }

    /// Downloads the payload and its signature, verifies the signature, then installs it.
    static func install(zip: URL, signature: URL, into store: PayloadStore) async throws -> Payload {
        guard isConfigured else { throw UpdaterError.notConfigured }
        let zipData = try await download(zip)
        let signatureData = try await download(signature)
        guard
            let signatureText = String(data: signatureData, encoding: .utf8),
            PayloadSignature.verify(data: zipData, signatureBase64: signatureText, publicKeyBase64: PayloadKey.publicKeyBase64)
        else { throw UpdaterError.badSignature }

        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("reco-payload-\(UUID().uuidString).zip")
        try zipData.write(to: temporary)
        defer { try? FileManager.default.removeItem(at: temporary) }
        return try store.install(zip: temporary)
    }

    private static func download(_ url: URL) async throws -> Data {
        if allowsFileURLs, url.isFileURL { return try Data(contentsOf: url) }
        guard url.scheme == "https" else { throw URLError(.unsupportedURL) }
        var request = URLRequest(url: url)
        request.setValue("RecoTrainerMac", forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = 120
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        return data
    }
}
