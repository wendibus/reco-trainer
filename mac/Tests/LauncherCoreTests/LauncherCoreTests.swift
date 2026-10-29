import CryptoKit
import Foundation
import Testing
@testable import LauncherCore

// MARK: Versions

@Test func versionParsingAcceptsEverySpellingOfATag() {
    #expect(parseVersionComponents("v0.12.8") == [0, 12, 8])
    #expect(parseVersionComponents("0.12.8") == [0, 12, 8])
    #expect(parseVersionComponents("v.0.17.2") == [0, 17, 2])
    #expect(normalizedVersion("v.0.17.2") == "0.17.2")
}

@Test func versionComparison() {
    #expect(isNewerVersion("0.20.1", than: "0.20.0"))
    #expect(isNewerVersion("0.13", than: "0.12.9"))
    #expect(!isNewerVersion("0.20.0", than: "0.20.0"))
    #expect(!isNewerVersion("0.19.9", than: "0.20.0"))
    #expect(isNewerVersion("v.0.20.1", than: "0.20.0"))
}

// MARK: Update planning

private func release(tag: String, assets: [String]) -> Release {
    Release(
        tagName: tag,
        htmlURL: URL(string: "https://github.com/wendibus/reco-trainer/releases/tag/\(tag)")!,
        assets: assets.map { ReleaseAsset(name: $0, browserDownloadURL: URL(string: "https://example.com/\($0)")!) }
    )
}

@Test func planFindsPayloadAndSignature() {
    let decision = UpdatePlanner.decide(
        release: release(tag: "v0.20.1", assets: ["Reco.Trainer.Payload.0.20.1.zip", "Reco.Trainer.Payload.0.20.1.zip.sig", "x.dmg"]),
        installedVersion: "0.20.0"
    )
    #expect(decision == .available(
        version: "0.20.1",
        zip: URL(string: "https://example.com/Reco.Trainer.Payload.0.20.1.zip")!,
        signature: URL(string: "https://example.com/Reco.Trainer.Payload.0.20.1.zip.sig")!
    ))
}

@Test func planWithoutSignatureNeedsNewApp() {
    let decision = UpdatePlanner.decide(
        release: release(tag: "v0.21.0", assets: ["Reco.Trainer.Payload.0.21.0.zip"]),
        installedVersion: "0.20.0"
    )
    guard case .needsNewApp(let version, _) = decision else { Issue.record("expected needsNewApp"); return }
    #expect(version == "0.21.0")
}

@Test func planIgnoresOlderAndEqualReleases() {
    #expect(UpdatePlanner.decide(release: release(tag: "v0.20.0", assets: []), installedVersion: "0.20.0") == .upToDate)
    #expect(UpdatePlanner.decide(release: release(tag: "v0.19.2", assets: []), installedVersion: "0.20.0") == .upToDate)
}

@Test func planHandlesStrayDotTag() {
    let decision = UpdatePlanner.decide(
        release: release(tag: "v.0.20.1", assets: ["Reco.Trainer.Payload.0.20.1.zip", "Reco.Trainer.Payload.0.20.1.zip.sig"]),
        installedVersion: "0.20.0"
    )
    guard case .available(let version, _, _) = decision else { Issue.record("expected available"); return }
    #expect(version == "0.20.1")
}

// MARK: Signatures

@Test func signatureVerification() throws {
    let key = Curve25519.Signing.PrivateKey()
    let publicKey = key.publicKey.rawRepresentation.base64EncodedString()
    let data = Data("payload bytes".utf8)
    let signature = try key.signature(for: data).base64EncodedString()

    #expect(PayloadSignature.verify(data: data, signatureBase64: signature, publicKeyBase64: publicKey))
    #expect(PayloadSignature.verify(data: data, signatureBase64: signature + "\n", publicKeyBase64: publicKey))
    #expect(!PayloadSignature.verify(data: Data("tampered".utf8), signatureBase64: signature, publicKeyBase64: publicKey))
    #expect(!PayloadSignature.verify(data: data, signatureBase64: "not base64!", publicKeyBase64: publicKey))
    #expect(!PayloadSignature.verify(data: data, signatureBase64: signature, publicKeyBase64: ""))
    let otherKey = Curve25519.Signing.PrivateKey().publicKey.rawRepresentation.base64EncodedString()
    #expect(!PayloadSignature.verify(data: data, signatureBase64: signature, publicKeyBase64: otherKey))
}

// MARK: Python lookup

@Test func pythonLocatorPrefersNewestHomebrewAndSkipsStubWithoutCommandLineTools() {
    let all = Set(["/opt/homebrew/bin/python3.12", "/opt/homebrew/bin/python3.11", "/usr/bin/python3"])
    #expect(PythonLocator.firstAvailable(isExecutable: { all.contains($0) }) == "/opt/homebrew/bin/python3.12")
    #expect(PythonLocator.firstAvailable(isExecutable: { $0 == "/usr/bin/python3" }, commandLineToolsInstalled: { false }) == nil)
    #expect(PythonLocator.firstAvailable(isExecutable: { $0 == "/usr/bin/python3" }, commandLineToolsInstalled: { true }) == "/usr/bin/python3")
}

// MARK: Payload store

private func makeTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("reco-launcher-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

@discardableResult
private func writePayload(at directory: URL, version: String, minLauncher: String = "0.20.0") throws -> URL {
    let fileManager = FileManager.default
    try fileManager.createDirectory(at: directory.appendingPathComponent("ui"), withIntermediateDirectories: true)
    try Data("<html></html>".utf8).write(to: directory.appendingPathComponent("ui/index.html"))
    try Data("# worker".utf8).write(to: directory.appendingPathComponent("local_worker.py"))
    try Data("# ml".utf8).write(to: directory.appendingPathComponent("ml_worker.py"))
    let manifest = "{\"version\": \"\(version)\", \"minLauncher\": \"\(minLauncher)\"}"
    try Data(manifest.utf8).write(to: directory.appendingPathComponent("payload.json"))
    return directory
}

private func zip(directory: URL, to archive: URL) throws {
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
    process.arguments = ["-c", "-k", directory.path, archive.path]
    try process.run()
    process.waitUntilExit()
    #expect(process.terminationStatus == 0)
}

@Test func seedingCopiesBundledPayloadOnceAndNeverDowngrades() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let bundled = try writePayload(at: base.appendingPathComponent("bundled"), version: "0.20.0")
    let store = PayloadStore(installRoot: base.appendingPathComponent("installed"), bundledRoot: bundled, launcherVersion: "0.20.0")

    #expect(try store.seedBundledPayloadIfNeeded()?.version == "0.20.0")
    #expect(store.activePayload()?.version == "0.20.0")
    #expect(try store.seedBundledPayloadIfNeeded() == nil)

    try writePayload(at: store.installRoot.appendingPathComponent("0.20.3"), version: "0.20.3")
    #expect(store.activePayload()?.version == "0.20.3")
    #expect(try store.seedBundledPayloadIfNeeded() == nil)
}

@Test func installUnpacksArchiveAndIgnoresPayloadsNeedingNewerLauncher() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let store = PayloadStore(installRoot: base.appendingPathComponent("installed"), bundledRoot: nil, launcherVersion: "0.20.0")

    let good = try writePayload(at: base.appendingPathComponent("good"), version: "0.20.1")
    let goodZip = base.appendingPathComponent("good.zip")
    try zip(directory: good, to: goodZip)
    #expect(try store.install(zip: goodZip).version == "0.20.1")
    #expect(store.activePayload()?.version == "0.20.1")

    let tooNew = try writePayload(at: base.appendingPathComponent("new"), version: "0.30.0", minLauncher: "0.30.0")
    let tooNewZip = base.appendingPathComponent("new.zip")
    try zip(directory: tooNew, to: tooNewZip)
    #expect(throws: PayloadError.launcherTooOld(required: "0.30.0")) { try store.install(zip: tooNewZip) }
    #expect(store.activePayload()?.version == "0.20.1")
}

@Test func installRejectsIncompleteArchives() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let store = PayloadStore(installRoot: base.appendingPathComponent("installed"), bundledRoot: nil, launcherVersion: "0.20.0")
    let broken = try writePayload(at: base.appendingPathComponent("broken"), version: "0.20.1")
    try FileManager.default.removeItem(at: broken.appendingPathComponent("ml_worker.py"))
    let brokenZip = base.appendingPathComponent("broken.zip")
    try zip(directory: broken, to: brokenZip)
    #expect(throws: PayloadError.self) { try store.install(zip: brokenZip) }
    #expect(store.activePayload() == nil)
}

@Test func installRejectsSymbolicLinks() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let store = PayloadStore(installRoot: base.appendingPathComponent("installed"), bundledRoot: nil, launcherVersion: "0.20.0")
    let linked = try writePayload(at: base.appendingPathComponent("linked"), version: "0.20.1")
    try FileManager.default.createSymbolicLink(at: linked.appendingPathComponent("evil"), withDestinationURL: URL(fileURLWithPath: "/etc"))
    let linkedZip = base.appendingPathComponent("linked.zip")
    try zip(directory: linked, to: linkedZip)
    #expect(throws: PayloadError.self) { try store.install(zip: linkedZip) }
}

@Test func failingVersionFallsBackToOlderPayloadUntilLauncherChanges() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let root = base.appendingPathComponent("installed")
    try writePayload(at: root.appendingPathComponent("0.20.0"), version: "0.20.0")
    try writePayload(at: root.appendingPathComponent("0.20.1"), version: "0.20.1")
    let store = PayloadStore(installRoot: root, bundledRoot: nil, launcherVersion: "0.20.0")

    #expect(store.activePayload()?.version == "0.20.1")
    store.markBad(version: "0.20.1")
    #expect(store.activePayload()?.version == "0.20.0")

    let newLauncher = PayloadStore(installRoot: root, bundledRoot: nil, launcherVersion: "0.20.2")
    #expect(newLauncher.activePayload()?.version == "0.20.1")
}

@Test func pruneKeepsNewestAndProtectedVersions() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let root = base.appendingPathComponent("installed")
    for version in ["0.20.0", "0.20.1", "0.20.2", "0.20.3"] {
        try writePayload(at: root.appendingPathComponent(version), version: version)
    }
    let store = PayloadStore(installRoot: root, bundledRoot: nil, launcherVersion: "0.20.0")
    store.prune(keeping: 2, protecting: ["0.20.0"])
    #expect(store.installedPayloads().map(\.version) == ["0.20.3", "0.20.2", "0.20.0"])
}

@Test func payloadWithUnsafeVersionIsIgnored() throws {
    let base = try makeTemporaryDirectory()
    defer { try? FileManager.default.removeItem(at: base) }
    let root = base.appendingPathComponent("installed")
    try writePayload(at: root.appendingPathComponent("evil"), version: "../../escape")
    let store = PayloadStore(installRoot: root, bundledRoot: nil, launcherVersion: "0.20.0")
    #expect(store.installedPayloads().isEmpty)
}

@Test func planRejectsFileURLsUnlessAllowedForLocalTests() {
    let zip = URL(fileURLWithPath: "/tmp/feed/Reco.Trainer.Payload.0.21.0.zip")
    let local = Release(
        tagName: "v0.21.0",
        htmlURL: URL(string: "https://github.com/wendibus/reco-trainer/releases/tag/v0.21.0")!,
        assets: [
            ReleaseAsset(name: zip.lastPathComponent, browserDownloadURL: zip),
            ReleaseAsset(name: zip.lastPathComponent + ".sig", browserDownloadURL: zip.appendingPathExtension("sig")),
        ]
    )
    guard case .needsNewApp = UpdatePlanner.decide(release: local, installedVersion: "0.20.0") else {
        Issue.record("file URLs must not be accepted from a normal release"); return
    }
    guard case .available(let version, _, _) = UpdatePlanner.decide(release: local, installedVersion: "0.20.0", allowFileURLs: true) else {
        Issue.record("expected available with allowFileURLs"); return
    }
    #expect(version == "0.21.0")
}
