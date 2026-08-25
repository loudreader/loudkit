import Foundation
import XCTest

import LoudKitText

/// The funnel's last pass, which nothing in this package exercised.
///
/// `pl_en_respell.json` was reachable only through `ChatterboxAssets`, a channel
/// `swift test` does not populate. So in every test run the lexicon was absent,
/// `LexicalRespelling` fell back to an empty table, logged one line, and turned
/// itself off — while `SpeechFunnelTests` went on passing, because the shared
/// conformance fixture carries no case where an English word inside Polish text
/// has to change. A silent no-op is the failure mode a bundling mistake has, and
/// it is invisible to a suite that only checks what it happens to cover.
///
/// The expectations are the Python funnel's output for the same input, read off
/// `loudkit.frontend.polish.speech_text`. If a port disagrees, one of them is
/// wrong and this says which words.
final class RespellingTests: XCTestCase {
    /// English loanwords Polish readers say with English values, in text that is
    /// otherwise Polish. Each of these changes; a build without the lexicon
    /// returns them untouched, which is exactly what this catches.
    func testEnglishInsidePolishIsRespelled() {
        let cases: [(String, String)] = [
            ("Mam weekend i laptop.", "Mam łikend i laptop."),
            ("To jest deadline na backup.", "To jest dedlajn na bekap."),
            ("Nowy software w chmurze.", "Nowy softłer w chmurze."),
        ]
        for (input, expected) in cases {
            XCTAssertEqual(
                SpeechText.prepared(input, languageId: "pl"),
                expected,
                "respelling did not fire for \(input) — is pl_en_respell.json bundled?"
            )
        }
    }

    /// The pass is Polish-only: the same words read as English text are already
    /// pronounced correctly by an English voice, and rewriting them there would
    /// be damage. This is what keeps the lexicon from leaking into eleven other
    /// languages.
    func testEnglishTextIsLeftAlone() {
        XCTAssertEqual(
            SpeechText.prepared("I had a weekend with a laptop.", languageId: "en"),
            "I had a weekend with a laptop."
        )
    }

    /// A word absent from the lexicon survives the pass unchanged, rather than
    /// being guessed at. The lexicon is a lookup, not a transliterator.
    func testUnknownWordsAreNotInvented() {
        let sentence = "Poszedłem do sklepu."
        XCTAssertEqual(SpeechText.prepared(sentence, languageId: "pl"), sentence)
    }
}
