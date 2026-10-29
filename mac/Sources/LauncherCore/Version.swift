import Foundation

/// Splits a version string like "v0.12.8", "0.12.8" or the historic stray-dot tag
/// "v.0.17.2" into [0, 12, 8]. A non-numeric component falls back to 0.
public func parseVersionComponents(_ raw: String) -> [Int] {
    var value = raw.trimmingCharacters(in: .whitespacesAndNewlines)
    if value.hasPrefix("v") || value.hasPrefix("V") { value.removeFirst() }
    if value.hasPrefix(".") { value.removeFirst() }
    return value.split(separator: ".").map { Int($0) ?? 0 }
}

/// True when `remote` is strictly newer than `local`. Missing trailing components count as 0.
public func isNewerVersion(_ remote: String, than local: String) -> Bool {
    let remoteParts = parseVersionComponents(remote)
    let localParts = parseVersionComponents(local)
    for index in 0..<max(remoteParts.count, localParts.count) {
        let remoteValue = index < remoteParts.count ? remoteParts[index] : 0
        let localValue = index < localParts.count ? localParts[index] : 0
        if remoteValue != localValue { return remoteValue > localValue }
    }
    return false
}

/// Normalised "0.20.1" form of any tag spelling.
public func normalizedVersion(_ raw: String) -> String {
    parseVersionComponents(raw).map(String.init).joined(separator: ".")
}
