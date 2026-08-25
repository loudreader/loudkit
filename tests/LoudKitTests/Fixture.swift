import Foundation
import XCTest

/// Locate the shared conformance fixture and the optional weight assets.
///
/// The fixture (`tests/data/conformance`) is committed and always present in
/// a checkout — the weight-free tests never skip. The checkpoint and CoreML
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
        let data = try Data(contentsOf: conformanceDir.appendingPathComponent("vectors.json"))
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoudKitTestError.fixture("vectors.json is not an object")
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
        // The repository's own `assets/` (gitignored), derived from this file's
        // own location — not an author's home directory, which is what used to
        // be written here and which worked on exactly one machine while telling
        // every reader whose machine it was. The default used to be a sibling
        // `chatterbox-apple` checkout; that checkout is gone and its artefacts
        // now live in `assets/`, flat at the root, the same layout
        // `tests/assets.py` resolves. Set LOUDKIT_ASSET_ROOT anywhere else.
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
            throw XCTSkip("checkpoint not present at \(checkpointURL.path) — set LOUDKIT_CHECKPOINT")
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
/// This used to correlate `min(a.count, b.count)` samples, which scores a
/// truncated render perfectly against the prefix it did produce — the length
/// *is* the finding in that case. Both current callers assert the count first,
/// so it was defended in practice and a trap for the next caller: the demo's
/// `ConformanceRunner` copied it verbatim and did not assert, and its mel check
/// scored 1.0 on a short mel. Asserted here so the trap cannot be copied again.
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
