import CryptoKit
import XCTest

@testable import LoudKitText

/// The grammar digest is what makes data drift between the five implementations
/// a startup failure instead of an audible surprise: each hashes its own copy,
/// and a copy that has fallen behind produces a different fingerprint.
final class GrammarDigestTests: XCTestCase {
    func testMatchesTheReference() {
        // Update deliberately, in the same commit as the data change, and only
        // after checking every port ships the same bytes.
        XCTAssertEqual(
            Numbers.grammarDigest, "71ad748c2caf97dd",
            "this bundle's numbers.json has drifted from the Python reference")
    }

    func testTheLexiconAndTheDigestReadTheSameFile() {
        /// The digest is only a claim about the funnel if it hashes the bytes
        /// the funnel used. `grammarDigest` read `Bundle.module` unconditionally
        /// while `LexicalRespelling` preferred `ChatterboxAssets`, so an
        /// application shipping its own `pl_en_respell.json` respelled from one
        /// file and reported the digest of another, different speech under one
        /// fingerprint, which is the single thing the digest exists to prevent.
        ///
        /// Both go through `Numbers.resourceBytes` now. Asserted by hashing
        /// what that returns and checking the digest is a prefix of it, because
        /// the alternative, comparing two paths, passes whenever both happen
        /// to be nil.
        let grammar = Numbers.resourceBytes("numbers")
        let respell = Numbers.resourceBytes("pl_en_respell")
        // The numeral fold table too: it decides the words a numeral becomes,
        // and it pins the Unicode version the fold uses, so it hashes with the
        // other two.
        let numerals = Numbers.resourceBytes("numerals")
        XCTAssertNotNil(grammar, "numbers.json did not resolve")
        XCTAssertNotNil(respell, "pl_en_respell.json did not resolve")
        XCTAssertNotNil(numerals, "numerals.json did not resolve")

        var combined = Data()
        combined.append(grammar!)
        combined.append(respell!)
        combined.append(numerals!)
        let expected = SHA256.hash(data: combined)
            .map { String(format: "%02x", $0) }.joined().prefix(16).description
        XCTAssertEqual(Numbers.grammarDigest, expected)
    }

    func testTheNumeralTableAndTheDigestReadTheSameFile() throws {
        // The same defect one file over, and the sharper one: `Numerals`
        // resolved `Bundle.module` directly while the digest resolved
        // `ChatterboxAssets` first, so an application shipping its own
        // `numerals.json` folded numerals by the packaged table and reported
        // the digest of the override. The fingerprint described a file that was
        // not in use.
        //
        // Checked by putting a table on the search path that says something no
        // packaged table says -- `2` reads as `two hundred` -- and asking the
        // parser what it read.
        let work = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("loudkit-numerals-\(ProcessInfo.processInfo.globallyUniqueString)")
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: work)
            ChatterboxAssets.searchPaths = ChatterboxAssets.defaultSearchPaths
        }
        let override = #"{"decimal_zeros": [], "spelled": {"178": "two hundred"}}"#
        try Data(override.utf8).write(to: work.appendingPathComponent("numerals.json"))

        ChatterboxAssets.searchPaths = [work] + ChatterboxAssets.defaultSearchPaths
        XCTAssertEqual(
            Numerals.load().spelled[178], "two hundred",
            "the fold read the packaged table while the digest hashed the override")
        XCTAssertEqual(
            Numbers.resourceBytes("numerals"), Data(override.utf8),
            "the digest read the packaged table while the fold read the override")
    }
}

