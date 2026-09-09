import XCTest

@testable import LoudKit
import LoudKitText

/// The refusals on the way into a render, tested through the static seams
/// `Engine` uses so they run without a checkpoint.
///
/// Each one used to be a silent success: `synthesizeWindow` on text the funnel
/// empties tokenised the bare language tag and rendered babble, a window that
/// generated nothing came back as an empty row, and `synthesizeTokens` filtered
/// a caller's out-of-range ids away and rendered what was left.
final class EngineDoorsTests: XCTestCase {
    private let limit = AlgorithmConfig().startSpeechToken

    private func message(_ body: () throws -> Void) -> String? {
        do {
            try body()
            XCTFail("the input was accepted")
            return nil
        } catch let error as LoudKitError {
            return error.description
        } catch {
            XCTFail("unexpected error \(error)")
            return nil
        }
    }

    /// The funnel really does empty these, so the guard is not theoretical.
    func testTextTheFunnelEmptiesIsRefusedRatherThanRendered() {
        for text in ["[12]", "[1]", "\u{1F600}", "***", "   ", "\u{2063}"] {
            let prepared = SpeechText.prepared(text, languageId: "en")
            XCTAssertTrue(prepared.isEmpty, "the funnel kept \(prepared.debugDescription)")
            XCTAssertEqual(
                message { try Engine.requireSomethingToSpeak(prepared) },
                "shape: nothing to speak: the text funnel removed every character. "
                + "Footnote markers, emoji and symbols with no word in the render "
                + "language are dropped, and this input was only those.",
                text.debugDescription)
        }
    }

    func testTextTheFunnelKeepsPasses() {
        for text in ["Hello", "\u{B7}", "1999"] {
            let prepared = SpeechText.prepared(text, languageId: "en")
            XCTAssertNoThrow(try Engine.requireSomethingToSpeak(prepared),
                             text.debugDescription)
        }
    }

    func testAWindowThatGeneratedNothingIsRefused() {
        XCTAssertEqual(
            message { try Engine.requireSpeechProduced([]) },
            "shape: generation produced no speech tokens: the stop token was accepted "
            + "immediately. Set sampling.min_tokens_floor above 0 to refuse that "
            + "during sampling.")
        XCTAssertNoThrow(try Engine.requireSpeechProduced([7]))
    }

    /// A caller's ids are checked at both ends, not filtered at one: an id at
    /// or above the limit is a control token the caller does not own, and
    /// dropping it renders something that was never asked for.
    func testACallersTokensAreRefusedAtBothEnds() {
        for bad in [-1, limit, limit + 1, 8193] {
            XCTAssertEqual(
                message {
                    try Engine.validateSpeechTokens([1, 2, bad], limit: limit, field: "tokens")
                },
                "shape: tokens contains \(bad), which is not an acoustic speech token "
                + "(expected 0 <= id < \(limit)). Pass `Result.tokens` from an earlier "
                + "call; the generator's own control tokens are already stripped from it.")
        }
        XCTAssertNoThrow(
            try Engine.validateSpeechTokens([0, 42, limit - 1], limit: limit, field: "tokens"))
    }

    /// One function behind both doors, so the two fields refuse the same values
    /// and differ only in the name they print.
    func testTheTwoDoorsShareOneSentence() {
        let tokens = message {
            try Engine.validateSpeechTokens([6561], limit: limit, field: "tokens")
        }
        let previous = message {
            _ = try Engine.carryFrom([6561], prefixTokens: 6, startSpeechToken: limit)
        }
        XCTAssertEqual(tokens?.replacingOccurrences(of: "tokens contains", with: "X"),
                       previous?.replacingOccurrences(of: "previousTokens contains", with: "X"))
    }
}
