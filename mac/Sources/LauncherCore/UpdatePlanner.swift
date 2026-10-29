import Foundation

public struct ReleaseAsset: Decodable, Equatable, Sendable {
    public let name: String
    public let browserDownloadURL: URL

    enum CodingKeys: String, CodingKey {
        case name
        case browserDownloadURL = "browser_download_url"
    }

    public init(name: String, browserDownloadURL: URL) {
        self.name = name
        self.browserDownloadURL = browserDownloadURL
    }
}

public struct Release: Decodable, Equatable, Sendable {
    public let tagName: String
    public let htmlURL: URL
    public let assets: [ReleaseAsset]

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case htmlURL = "html_url"
        case assets
    }

    public init(tagName: String, htmlURL: URL, assets: [ReleaseAsset]) {
        self.tagName = tagName
        self.htmlURL = htmlURL
        self.assets = assets
    }
}

public enum UpdateDecision: Equatable, Sendable {
    case upToDate
    /// A newer release with a signed payload: download `zip` and verify it against `signature`.
    case available(version: String, zip: URL, signature: URL)
    /// A newer release without a payload (it needs a new app download).
    case needsNewApp(version: String, releasePage: URL)
}

public enum UpdatePlanner {
    public static func payloadAssetName(version: String) -> String { "Reco.Trainer.Payload.\(version).zip" }

    public static func decide(release: Release, installedVersion: String) -> UpdateDecision {
        let version = normalizedVersion(release.tagName)
        guard isNewerVersion(version, than: installedVersion) else { return .upToDate }
        let zipName = payloadAssetName(version: version)
        guard
            let zip = release.assets.first(where: { $0.name == zipName }),
            let signature = release.assets.first(where: { $0.name == zipName + ".sig" }),
            zip.browserDownloadURL.scheme == "https",
            signature.browserDownloadURL.scheme == "https"
        else { return .needsNewApp(version: version, releasePage: release.htmlURL) }
        return .available(version: version, zip: zip.browserDownloadURL, signature: signature.browserDownloadURL)
    }
}
