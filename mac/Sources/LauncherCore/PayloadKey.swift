/// Public Ed25519 key (base64, 32 raw bytes) that update payloads must be signed with.
/// Generate the key pair once with `swift mac/tools/payload-signing.swift keygen` and paste the
/// printed public key here. While this is empty the launcher never installs downloaded updates -
/// it only runs the payload bundled inside the app.
public enum PayloadKey {
    public static let publicKeyBase64 = ""
}
