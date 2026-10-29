// Signs update payloads with Ed25519 so the Mac app only installs updates you published.
//
//   swift mac/tools/payload-signing.swift keygen [keyfile]    one-time: create the key pair
//   swift mac/tools/payload-signing.swift sign <file> [keyfile]   writes <file>.sig
//   swift mac/tools/payload-signing.swift verify <file> <publicKeyBase64>
//
// The private key (default ~/.reco-trainer/payload-signing.key) never goes into the repository.
import CryptoKit
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

let arguments = CommandLine.arguments
let defaultKeyURL = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".reco-trainer/payload-signing.key")

func loadPrivateKey(_ path: String?) -> Curve25519.Signing.PrivateKey {
    let url = path.map { URL(fileURLWithPath: $0) } ?? defaultKeyURL
    guard
        let text = try? String(contentsOf: url, encoding: .utf8),
        let raw = Data(base64Encoded: text.trimmingCharacters(in: .whitespacesAndNewlines)),
        let key = try? Curve25519.Signing.PrivateKey(rawRepresentation: raw)
    else { fail("Cannot read the private key at \(url.path). Create one with: keygen") }
    return key
}

switch arguments.dropFirst().first {
case "keygen":
    let url = arguments.count > 2 ? URL(fileURLWithPath: arguments[2]) : defaultKeyURL
    if FileManager.default.fileExists(atPath: url.path) { fail("A key already exists at \(url.path). Not overwriting it.") }
    let key = Curve25519.Signing.PrivateKey()
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try key.rawRepresentation.base64EncodedString().write(to: url, atomically: true, encoding: .utf8)
    chmod(url.path, 0o600)
    print("Private key written to \(url.path) - keep it secret, back it up, never commit it.")
    print("Public key (paste into mac/Sources/LauncherCore/PayloadKey.swift):")
    print(key.publicKey.rawRepresentation.base64EncodedString())

case "sign":
    guard arguments.count >= 3 else { fail("usage: sign <file> [keyfile]") }
    let file = URL(fileURLWithPath: arguments[2])
    let key = loadPrivateKey(arguments.count > 3 ? arguments[3] : nil)
    let data = try Data(contentsOf: file)
    let signature = try key.signature(for: data).base64EncodedString()
    let output = URL(fileURLWithPath: arguments[2] + ".sig")
    try signature.write(to: output, atomically: true, encoding: .utf8)
    print(output.path)

case "verify":
    guard arguments.count >= 4 else { fail("usage: verify <file> <publicKeyBase64>") }
    let file = URL(fileURLWithPath: arguments[2])
    guard
        let raw = Data(base64Encoded: arguments[3]),
        let publicKey = try? Curve25519.Signing.PublicKey(rawRepresentation: raw),
        let signatureText = try? String(contentsOf: URL(fileURLWithPath: arguments[2] + ".sig"), encoding: .utf8),
        let signature = Data(base64Encoded: signatureText.trimmingCharacters(in: .whitespacesAndNewlines))
    else { fail("Cannot read the public key or the .sig file.") }
    let data = try Data(contentsOf: file)
    if publicKey.isValidSignature(signature, for: data) { print("Signature OK") } else { fail("Signature INVALID") }

default:
    fail("usage: keygen [keyfile] | sign <file> [keyfile] | verify <file> <publicKeyBase64>")
}
