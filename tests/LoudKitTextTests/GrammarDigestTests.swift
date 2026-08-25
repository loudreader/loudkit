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
            Numbers.grammarDigest, "d10073beca3c0f03",
            "this bundle's numbers.json has drifted from the Python reference")
    }

    func testTheLexiconAndTheDigestReadTheSameFile() {
        /// The digest is only a claim about the funnel if it hashes the bytes
        /// the funnel used. `grammarDigest` read `Bundle.module` unconditionally
        /// while `LexicalRespelling` preferred `ChatterboxAssets`, so an
        /// application shipping its own `pl_en_respell.json` respelled from one
        /// file and reported the digest of another — different speech under one
        /// fingerprint, which is the single thing the digest exists to prevent.
        ///
        /// Both go through `Numbers.resourceBytes` now. Asserted by hashing
        /// what that returns and checking the digest is a prefix of it, because
        /// the alternative — comparing two paths — passes whenever both happen
        /// to be nil.
        let grammar = Numbers.resourceBytes("numbers")
        let respell = Numbers.resourceBytes("pl_en_respell")
        XCTAssertNotNil(grammar, "numbers.json did not resolve")
        XCTAssertNotNil(respell, "pl_en_respell.json did not resolve")

        var combined = Data()
        combined.append(grammar!)
        combined.append(respell!)
        let expected = SHA256.hash(data: combined)
            .map { String(format: "%02x", $0) }.joined().prefix(16).description
        XCTAssertEqual(Numbers.grammarDigest, expected)
    }
}

