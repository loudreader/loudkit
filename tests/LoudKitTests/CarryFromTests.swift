import XCTest

@testable import LoudKit

/// Cross-call prosody context: the tail of one call conditioning the next.
///
/// The engine already carries the last `chunking.prefixTokens` of a chunk into
/// the next one inside a passage; `previousTokens` is that same carry, seeded
/// from a *previous call*. There is deliberately no second mechanism, so what
/// is left to test is the slice and its refusals.
///
/// Tested through `Engine.carryFrom(_:prefixTokens:startSpeechToken:)` rather
/// than through `synthesize`, because an `Engine` cannot exist without the
/// checkpoint and the CoreML packages, and on a machine without them the whole
/// feature would go untested. The static helper is the honest unit here: it is
/// the entirety of the arithmetic, and both call sites in `Engine` are one line
/// each that forward this engine's config into it.
final class CarryFromTests: XCTestCase {
    /// The shipped values: six tokens of context, acoustic ids below 6561.
///
/// What this therefore does NOT cover is the wiring: if the helper's result
/// stopped being handed to the generator's prefix, every assertion here would
/// still pass. That half is pinned in Python, by
/// tests/test_engine.py::TestCrossRequestContext, against a fake generator
/// that records the context it was given — building an equivalent seam in four
/// more languages would cost four engine refactors to re-assert one fact.
    private let prefixTokens = ChunkConfig().prefixTokens
    private let startSpeechToken = AlgorithmConfig().startSpeechToken

    func testTheCarryIsTheTailOfWhatWasHandedIn() throws {
        let previous = Array(100..<120)
        let carry = try Engine.carryFrom(
            previous, prefixTokens: prefixTokens, startSpeechToken: startSpeechToken)
        XCTAssertEqual(carry, Array(previous.suffix(prefixTokens)))
    }

    func testALongHistoryIsSlicedRatherThanRefused() throws {
        // Any length is accepted because only the tail is used, so
        // `previousTokens: result.tokens` is the intended call and a caller
        // never has to know the prefix length to make it.
        let previous = (0..<5_000).map { $0 % 6_000 }
        let carry = try Engine.carryFrom(
            previous, prefixTokens: prefixTokens, startSpeechToken: startSpeechToken)
        XCTAssertEqual(carry.count, prefixTokens)
        XCTAssertEqual(carry, Array(previous.suffix(prefixTokens)))
    }

    func testAHistoryShorterThanThePrefixIsUsedWhole() throws {
        let carry = try Engine.carryFrom(
            [7, 8], prefixTokens: prefixTokens, startSpeechToken: startSpeechToken)
        XCTAssertEqual(carry, [7, 8])
    }

    func testAbsentIsNoContextAtAll() throws {
        // The default, and byte-for-byte the behaviour that existed before this
        // parameter did.
        XCTAssertEqual(
            try Engine.carryFrom(
                nil, prefixTokens: prefixTokens, startSpeechToken: startSpeechToken),
            [])
        XCTAssertEqual(
            try Engine.carryFrom(
                [], prefixTokens: prefixTokens, startSpeechToken: startSpeechToken),
            [])
    }

    func testZeroPrefixCarriesNothingRatherThanEverything() throws {
        // The bug this guards: a naive `tokens[-wanted:]` at zero is the whole
        // list rather than nothing, which would condition on the entire
        // previous utterance at exactly the setting that means "chunks are
        // independent".
        let carry = try Engine.carryFrom(
            Array(0..<50), prefixTokens: 0, startSpeechToken: startSpeechToken)
        XCTAssertEqual(carry, [])
    }

    func testAnIdOutsideTheAcousticCodebookIsRefused() {
        for bad in [-1, 6_561, 6_562, 99_999] {
            XCTAssertThrowsError(
                try Engine.carryFrom(
                    [1, 2, bad], prefixTokens: prefixTokens,
                    startSpeechToken: startSpeechToken),
                "\(bad) was accepted as a speech token"
            ) {
                XCTAssertTrue(
                    "\($0)".contains("\(bad)") && "\($0)".contains("6561"),
                    "the message has to name the id and the bound: \($0)")
            }
        }
    }

    func testTheWholeInputIsCheckedAndNotOnlyTheSliceThatIsUsed() {
        // An id out of range means the sequence was built wrong. Reporting that
        // only when it happens to land in the last six tokens would make the
        // failure depend on how long the caller's text happened to be — the
        // same bad input passing or failing for no reason the caller can see.
        var previous = Array(repeating: 42, count: 100)
        previous[0] = startSpeechToken + 1
        XCTAssertThrowsError(
            try Engine.carryFrom(
                previous, prefixTokens: prefixTokens, startSpeechToken: startSpeechToken))
    }

    func testTheBoundsOfTheCodebookAreThemselvesAllowed() throws {
        let edge = [0, startSpeechToken - 1]
        XCTAssertEqual(
            try Engine.carryFrom(
                edge, prefixTokens: prefixTokens, startSpeechToken: startSpeechToken),
            edge)
    }
}
