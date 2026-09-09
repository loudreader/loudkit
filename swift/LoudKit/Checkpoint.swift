import Foundation

/// A packed loudkit checkpoint, opened lazily. The embedded manifest is the
/// authority on every weight-borne algorithm value, same rule as the Python
/// loader, and the reason `AlgorithmConfig.fromManifest` reads every key it
/// carries rather than defaulting past one it cannot read. A pack that names
/// no window is the one thing it does default: all five ports read that as the
/// ragged window, so it is a declared shape and not a guess.
public final class Checkpoint {
    /// Where the pack was opened from. The tokenizer and the CoreML packages
    /// are resolved relative to it.
    public let url: URL
    /// The safetensors store the weights are read out of, tensor by tensor.
    public let store: Safetensors
    /// The embedded manifest, parsed. Authority on every weight-borne value.
    public let manifest: [String: Any]

    /// Manifest format versions this build understands.
    public static let supportedFormatVersions = [1, 2]

    /// Decode loops this build implements.
    public static let supportedDecodeModes = ["single", "fusion_mtp2"]

    /// Open a pack and check its manifest, before any weight is read.
    ///
    /// - Throws: `LoudKitError.manifest` when the file carries no manifest, a
    ///   format version this build does not read, or a decode mode it does not
    ///   implement.
    public init(url: URL) throws {
        self.url = url
        store = try Safetensors(url: url)
        // `ManifestReader.json`, not `JSONSerialization` directly: Foundation
        // keeps the first of two members with the same name and the reference
        // keeps the last, so `"sample_rate":24000,"sample_rate":48000` ran four
        // engines at 48 kHz and this one at 24.
        guard let manifestJSON = store.metadata["manifest"] else {
            throw LoudKitError.manifest("\(url.lastPathComponent): no embedded manifest, not a loudkit checkpoint")
        }
        let raw = try ManifestReader.json(
            Data(manifestJSON.utf8), "\(url.lastPathComponent): bad manifest JSON")
        guard let parsed = raw as? [String: Any] else {
            throw LoudKitError.manifest(
                "\(url.lastPathComponent): manifest is "
                + "\(ManifestReader.jsonTypePhrase(raw)), expected a JSON object")
        }
        guard parsed["format"] as? String == "loudkit-checkpoint" else {
            throw LoudKitError.manifest(
                "\(url.lastPathComponent): no embedded manifest, not a loudkit checkpoint")
        }
        // Read the same way `AlgorithmConfig.fromManifest` reads them, not a
        // second time with a looser rule. A boolean is an `NSNumber` once
        // `JSONSerialization` is done with it, so `format_version: true` used
        // to read as version 1 and load; a `decode` block of the wrong type, or
        // a mode that was not a string, read as "single" here and were refused
        // one layer down, which put two rules for one key in one package.
        let version = try ManifestReader.formatVersion(parsed, "format_version", -1)
        guard Checkpoint.supportedFormatVersions.contains(version) else {
            throw LoudKitError.manifest(
                "\(url.lastPathComponent): manifest format_version \(version); "
                + "this build reads \(Checkpoint.supportedFormatVersions)")
        }
        let decode = try ManifestReader.block(parsed, "decode")
        let mode = try ManifestReader.text(decode, "manifest['decode']", "mode") ?? "single"
        guard mode != "fusion_mtp2" || version >= 2 else {
            throw LoudKitError.manifest("fusion_mtp2 requires format_version 2")
        }
        guard Checkpoint.supportedDecodeModes.contains(mode) else {
            throw LoudKitError.manifest(
                "\(url.lastPathComponent): manifest declares decode.mode \(mode); "
                + "this build decodes \(Checkpoint.supportedDecodeModes)")
        }
        manifest = parsed
    }

    /// The algorithm this pack declares, range-checked. Recomputed on each
    /// call rather than stored, so a caller cannot hold a config the manifest
    /// no longer describes.
    public func algorithm() throws -> AlgorithmConfig {
        try AlgorithmConfig.fromManifest(manifest)
    }

    /// The text tokenizer ships beside the checkpoint under this name.
    public var tokenizerURL: URL {
        url.deletingLastPathComponent().appendingPathComponent("tokenizer.json")
    }

    /// Default CoreML asset directory: `coreml/` beside the checkpoint,
    /// where `tools/export_coreml.py` writes.
    public var coremlAssetsURL: URL {
        url.deletingLastPathComponent().appendingPathComponent("coreml")
    }
    /// The renderer packages must describe this checkpoint and algorithm.
    func verifyCoreMLExport(at assets: URL, algorithm: AlgorithmConfig) throws {
        let record = assets.appendingPathComponent("export.json")
        guard FileManager.default.fileExists(atPath: record.path) else {
            FileHandle.standardError.write(Data("loudkit: \(assets.path) has no export.json; re-export to record provenance\n".utf8))
            return
        }
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: record)) as? [String: Any],
              let packages = root["packages"] as? [String: [String: Any]] else {
            throw LoudKitError.asset("\(record.path): unreadable export record")
        }
        let digest = try LoudKit.fileSHA256(url)
        let keys = ["checkpoint_sha256", "algorithm_fingerprint", "euler_steps", "estimator_sha256"]
        var agreed: NSDictionary?
        for stem in ["flow_encoder", "flow_estimator", "vocoder"] {
            guard let entry = packages[stem + ".mlpackage"] else {
                throw LoudKitError.asset("\(record.path) does not record \(stem).mlpackage")
            }
            let identity = NSDictionary(dictionary: entry.filter { keys.contains($0.key) })
            if let agreed, !agreed.isEqual(identity) {
                throw LoudKitError.asset("\(record.path): renderer packages came from different exports")
            }
            agreed = identity
            guard entry["checkpoint_sha256"] as? String == digest,
                  entry["algorithm_fingerprint"] as? String == algorithm.fingerprint(),
                  (entry["euler_steps"] as? NSNumber)?.intValue == algorithm.eulerSteps else {
                throw LoudKitError.asset("\(record.path): renderer was exported from a different engine")
            }
            if let sources = manifest["sources"] as? [String: [String: Any]],
               let estimator = sources.values.first(where: { $0["role"] as? String == "estimator" }),
               let expected = estimator["sha256"] as? String,
               let actual = entry["estimator_sha256"] as? String, expected != actual {
                throw LoudKitError.asset("\(record.path): renderer uses a different estimator")
            }
        }
    }

}
