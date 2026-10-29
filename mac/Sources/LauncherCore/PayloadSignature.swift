import CryptoKit
import Foundation

public enum PayloadSignature {
    /// Verifies an Ed25519 signature (base64) over `data` against a base64 raw public key.
    /// Any malformed input counts as "not valid".
    public static func verify(data: Data, signatureBase64: String, publicKeyBase64: String) -> Bool {
        let trimSet = CharacterSet.whitespacesAndNewlines
        guard
            let keyData = Data(base64Encoded: publicKeyBase64.trimmingCharacters(in: trimSet)),
            let signature = Data(base64Encoded: signatureBase64.trimmingCharacters(in: trimSet)),
            let key = try? Curve25519.Signing.PublicKey(rawRepresentation: keyData)
        else { return false }
        return key.isValidSignature(signature, for: data)
    }
}
