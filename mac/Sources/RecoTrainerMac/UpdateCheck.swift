import Foundation

struct AvailableUpdate: Equatable {
    let version: String
    let url: String
}

struct GitHubRelease: Decodable {
    let tagName: String
    let htmlURL: String

    enum CodingKeys: String, CodingKey {
        case tagName = "tag_name"
        case htmlURL = "html_url"
    }
}

/// Splits a version string like "v0.12.8" or "0.12.8" into [0, 12, 8].
/// A non-numeric component (a stray suffix) falls back to 0 rather than crashing.
func parseVersionComponents(_ raw: String) -> [Int] {
    var value = raw
    if value.hasPrefix("v") || value.hasPrefix("V") { value.removeFirst() }
    return value.split(separator: ".").map { Int($0) ?? 0 }
}

/// True when `remote` denotes a strictly newer version than `local`, comparing
/// major/minor/patch (and beyond) component by component. A missing trailing
/// component is treated as 0, so "0.13" counts as newer than "0.12.9".
func isNewerVersion(_ remote: String, than local: String) -> Bool {
    let remoteParts = parseVersionComponents(remote)
    let localParts = parseVersionComponents(local)
    let count = max(remoteParts.count, localParts.count)
    for index in 0..<count {
        let remoteValue = index < remoteParts.count ? remoteParts[index] : 0
        let localValue = index < localParts.count ? localParts[index] : 0
        if remoteValue != localValue { return remoteValue > localValue }
    }
    return false
}

private let latestReleaseURL = URL(string: "https://api.github.com/repos/wendibus/reco-trainer/releases/latest")!

/// Fetches the tag and URL of the latest published GitHub release. Contacts only
/// GitHub's public releases API - no project data (videos, frames, annotations,
/// models) is ever part of this request, and a failure here is silently ignored
/// by the caller since this is a best-effort, non-essential check.
func fetchLatestRelease() async throws -> GitHubRelease {
    var request = URLRequest(url: latestReleaseURL)
    request.setValue("RecoTrainerMac", forHTTPHeaderField: "User-Agent")
    request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
    let (data, response) = try await URLSession.shared.data(for: request)
    guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
        throw URLError(.badServerResponse)
    }
    return try JSONDecoder().decode(GitHubRelease.self, from: data)
}
