import Foundation
import XCTest

@testable import LoudKitText

/// The funnel's last pass, which nothing in this package exercised.
///
/// `pl_en_respell.json` was reachable only through `ChatterboxAssets`, a channel
/// `swift test` does not populate. So in every test run the lexicon was absent,
/// `LexicalRespelling` fell back to an empty table, logged one line, and turned
/// itself off, while `SpeechFunnelTests` went on passing, because the shared
/// conformance fixture carries no case where an English word inside Polish text
/// has to change. A silent no-op is the failure mode a bundling mistake has, and
/// it is invisible to a suite that only checks what it happens to cover.
///
/// The expectations are the Python funnel's output for the same input, read off
/// `loudkit.frontend.speechtext.speech_text`. If a port disagrees, one of them is
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
                "respelling did not fire for \(input): is pl_en_respell.json bundled?"
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

    /// The bundled copy of `pl_respell_rules.json` is the canonical file, byte
    /// for byte.
    ///
    /// The curated tables used to be literals in this source, which made the
    /// drift invisible by construction: this port could gain a word the Python
    /// funnel does not have and nothing would say so. As a copied file the
    /// drift is at least *possible* to see, and this is what sees it. Not
    /// `grammar_digest`, which does not cover this file: an edit here moves the
    /// spoken words under the same sixteen hex digits, so a byte comparison is
    /// the only gate there is.
    func testTheBundledRulesAreTheCanonicalBytes() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // LoudKitTextTests
            .deletingLastPathComponent()  // tests
            .deletingLastPathComponent()  // repository root
        let canonical = try Data(
            contentsOf: root.appendingPathComponent(
                "python/loudkit/models/data/pl_respell_rules.json"))
        let bundled = try XCTUnwrap(
            Numbers.resourceBytes("pl_respell_rules"),
            "pl_respell_rules.json is not bundled; copy it with tools/sync_grammar.py")
        XCTAssertEqual(
            bundled, canonical,
            "the bundled Polish respelling rules differ from the canonical file; "
                + "re-run tools/sync_grammar.py")
    }

    /// Every table the funnel reads came out of that file, and none of them is
    /// empty.
    ///
    /// A decoder that silently produced empty tables would leave Polish text
    /// read with Polish letter values for every English word in it, which is
    /// the state this whole pass exists to end, and the three cases above only
    /// reach the lexicon.
    func testEveryCuratedTableLoaded() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let raw = try JSONSerialization.jsonObject(
            with: Data(contentsOf: root.appendingPathComponent(
                "python/loudkit/models/data/pl_respell_rules.json")))
        let file = try XCTUnwrap(raw as? [String: Any])

        XCTAssertEqual(LexicalRespelling.phrases.count,
                       (file["phrases"] as? [[String]])?.count)
        XCTAssertEqual(LexicalRespelling.phrases.first.map { [$0.0, $0.1] },
                       (file["phrases"] as? [[String]])?.first,
                       "the phrase pass is ordered; the file's order is the order")
        XCTAssertEqual(LexicalRespelling.lexicon, file["lexicon"] as? [String: String])
        XCTAssertEqual(LexicalRespelling.keepPolish,
                       Set(try XCTUnwrap(file["keep_polish"] as? [String])))
        XCTAssertEqual(LexicalRespelling.polishFunctionWords,
                       Set(try XCTUnwrap(file["function_words"] as? [String])))
        XCTAssertEqual(LexicalRespelling.polishEndings,
                       Set(try XCTUnwrap(file["endings"] as? [String])))
        XCTAssertFalse(LexicalRespelling.lexicon.isEmpty)
    }

    /// One entry from each table, through the funnel, so a table that parsed
    /// and is not consulted still fails.
    func testEveryCuratedTableIsConsulted() {
        // `lexicon`, and `endings` with it: "deadline" declines to "dedlajnu".
        XCTAssertEqual(SpeechText.prepared("deadline'u", languageId: "pl"), "dedlajnu")
        // `phrases`: word by word "notes" is the Polish notebook and stays.
        XCTAssertEqual(SpeechText.prepared("release notes", languageId: "pl"), "rilis nołc")
        // `keep_polish`: the same word alone is Polish and must not move.
        XCTAssertEqual(SpeechText.prepared("notes", languageId: "pl"), "notes")
        // `function_words`: "to" is Polish here, and must not join the span.
        XCTAssertEqual(SpeechText.prepared("to weekend", languageId: "pl"), "to łikend")
    }
}
