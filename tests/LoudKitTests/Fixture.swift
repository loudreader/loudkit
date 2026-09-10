import Foundation
import XCTest

/// Locate the shared conformance fixture and the optional weight assets.
///
/// The fixture (`tests/data/conformance`) is committed and always present in
/// a checkout, the weight-free tests never skip. The checkpoint and CoreML
/// packages are resolved exactly like the Python side's `tests/assets.py`:
/// environment variable first, developer-machine default second, and a named
/// skip when absent (`LOUDKIT_REQUIRE_ASSETS=1` turns those skips into
/// failures, so CI cannot go green by losing its weights).
enum Fixture {
    static var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Fixture.swift
            .deletingLastPathComponent()  // LoudKitTests
            .deletingLastPathComponent()  // Tests
    }

    static var conformanceDir: URL {
        repoRoot.appendingPathComponent("tests/data/conformance")
    }

    static func vectors() throws -> [String: Any] {
        try shared("vectors.json")
    }

    /// One of the JSON fixtures under `tests/data/conformance`.
    static func shared(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: conformanceDir.appendingPathComponent(name))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoudKitTestError.fixture("\(name) is not an object")
        }
        return root
    }

    static var requireAssets: Bool {
        ["1", "true", "yes"].contains(
            (ProcessInfo.processInfo.environment["LOUDKIT_REQUIRE_ASSETS"] ?? "").lowercased())
    }

    static var checkpointURL: URL {
        if let env = ProcessInfo.processInfo.environment["LOUDKIT_CHECKPOINT"] {
            return URL(fileURLWithPath: env)
        }
        // The repository's own `assets/` (gitignored), derived from this
        // file's own location so that it resolves on any checkout. The layout
        // is flat at the root, the same one `tests/assets.py` resolves. Set
        // LOUDKIT_ASSET_ROOT to read them from anywhere else.
        let root = ProcessInfo.processInfo.environment["LOUDKIT_ASSET_ROOT"]
            ?? repoRoot.appendingPathComponent("assets").path
        return URL(fileURLWithPath: root)
            .appendingPathComponent("loudr-1.safetensors")
    }

    static var coremlAssetsURL: URL? {
        if let env = ProcessInfo.processInfo.environment["LOUDKIT_COREML_ASSETS"] {
            return URL(fileURLWithPath: env)
        }
        return nil  // Engine.load defaults to <checkpoint dir>/coreml
    }

    /// Skip (or fail, under LOUDKIT_REQUIRE_ASSETS) when the checkpoint is
    /// not on this machine.
    static func requireCheckpoint() throws {
        if !FileManager.default.fileExists(atPath: checkpointURL.path) {
            if requireAssets {
                XCTFail("LOUDKIT_REQUIRE_ASSETS is set but checkpoint is missing: \(checkpointURL.path)")
            }
            throw XCTSkip("checkpoint not present at \(checkpointURL.path): set LOUDKIT_CHECKPOINT")
        }
    }
}

enum LoudKitTestError: Error {
    case fixture(String)
}

func asDoubles(_ any: Any?) -> [Double]? {
    (any as? [NSNumber])?.map(\.doubleValue)
}

func asInts(_ any: Any?) -> [Int]? {
    (any as? [NSNumber])?.map(\.intValue)
}

/// Pearson correlation, on the explicit condition that the two are the same
/// length.
///
/// Correlating `min(a.count, b.count)` samples scores a truncated render
/// perfectly against the prefix it did produce, and in that case the length
/// *is* the finding. Asserting the condition here rather than leaving it to
/// each caller is what keeps a copy of this helper from scoring 1.0 on a short
/// mel.
func correlation(_ a: [Float], _ b: [Float],
                 file: StaticString = #filePath, line: UInt = #line) -> Double {
    XCTAssertEqual(a.count, b.count,
                   "length mismatch: correlating a prefix would hide a truncated render",
                   file: file, line: line)
    let n = min(a.count, b.count)
    var ma = 0.0, mb = 0.0
    for i in 0..<n {
        ma += Double(a[i])
        mb += Double(b[i])
    }
    ma /= Double(n)
    mb /= Double(n)
    var cov = 0.0, va = 0.0, vb = 0.0
    for i in 0..<n {
        let da = Double(a[i]) - ma
        let db = Double(b[i]) - mb
        cov += da * db
        va += da * da
        vb += db * db
    }
    return cov / (va.squareRoot() * vb.squareRoot())
}

/// The RMS ratio of a render to its reference, in dB.
///
/// Correlation subtracts the mean and divides by the deviation, so it reports
/// 1.0 for a render at half volume, at twenty times volume, or with a DC
/// offset. Level is exactly what that normalisation discards, so it is the one
/// amplitude fact worth its own gate.
func levelDB(_ a: [Float], _ b: [Float],
             file: StaticString = #filePath, line: UInt = #line) -> Double {
    XCTAssertEqual(a.count, b.count, "length mismatch", file: file, line: line)
    var sa = 0.0, sb = 0.0
    for i in 0..<min(a.count, b.count) {
        sa += Double(a[i]) * Double(a[i])
        sb += Double(b[i]) * Double(b[i])
    }
    XCTAssertGreaterThan(sa, 0.0, "rendered silence", file: file, line: line)
    return 20.0 * log10((sa / sb).squareRoot())
}

/// The loudest sample, against the `[-1, 1]` a waveform is declared to occupy.
/// Everything downstream clips to that range, so a render outside it is audibly
/// wrong and needs no tolerance to say so.
func peakOf(_ a: [Float]) -> Double {
    a.reduce(0.0) { Swift.max($0, Double(abs($1))) }
}

extension Fixture {
    static var fusionCheckpointURL: URL {
        if let path = ProcessInfo.processInfo.environment["LOUDKIT_FUSION_CHECKPOINT"] {
            return URL(fileURLWithPath: path)
        }
        let root = ProcessInfo.processInfo.environment["LOUDKIT_ASSET_ROOT"]
            ?? repoRoot.appendingPathComponent("assets").path
        return URL(fileURLWithPath: root).appendingPathComponent("loudr-1-turbo.safetensors")
    }

    static var fusionCoremlURL: URL? {
        ProcessInfo.processInfo.environment["LOUDKIT_FUSION_COREML_ASSETS"]
            .map { URL(fileURLWithPath: $0) }
    }

    static func requireFusionCheckpoint() throws {
        guard FileManager.default.fileExists(atPath: fusionCheckpointURL.path) else {
            if requireAssets { XCTFail("fusion checkpoint missing: \(fusionCheckpointURL.path)") }
            throw XCTSkip("unpublished fusion checkpoint missing; set LOUDKIT_FUSION_CHECKPOINT")
        }
    }
}
