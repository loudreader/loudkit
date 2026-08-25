import Foundation

/// A packed loudkit checkpoint, opened lazily. The embedded manifest is the
/// authority on every weight-borne algorithm value — same rule as the Python
/// loader, and the reason `AlgorithmConfig.fromManifest` refuses a pack that
/// does not carry the window recipe.
public final class Checkpoint {
    public let url: URL
    public let store: Safetensors
    public let manifest: [String: Any]

    public init(url: URL) throws {
        self.url = url
        store = try Safetensors(url: url)
        guard let manifestJSON = store.metadata["manifest"],
              let parsed = try JSONSerialization.jsonObject(
                with: Data(manifestJSON.utf8)) as? [String: Any] else {
            throw LoudKitError.manifest("\(url.lastPathComponent): no embedded manifest — not a loudkit checkpoint")
        }
        guard parsed["format"] as? String == "loudkit-checkpoint",
              (parsed["format_version"] as? NSNumber)?.intValue == 1 else {
            throw LoudKitError.manifest("\(url.lastPathComponent): unknown format/version")
        }
        manifest = parsed
    }

    public func algorithm() throws -> AlgorithmConfig {
        try AlgorithmConfig.fromManifest(manifest)
    }

    /// The text tokenizer ships beside the checkpoint under this name.
    public var tokenizerURL: URL {
        url.deletingLastPathComponent().appendingPathComponent("tokenizer.json")
    }

    /// Default CoreML asset directory: `coreml/` beside the checkpoint —
    /// where `tools/export_coreml.py` writes.
    public var coremlAssetsURL: URL {
        url.deletingLastPathComponent().appendingPathComponent("coreml")
    }
}
