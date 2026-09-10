import XCTest

@testable import LoudKit
@testable import LoudKitText

/// Every text pass compares code points, because the reference does.
///
/// Swift's `String.contains` and `String.replacingOccurrences` compare extended
/// grapheme clusters, so `"✔️"` (U+2714 U+FE0F) is not `"✔"` and `"50%\u{FE0F}"`
/// does not contain `"%"`. Python's `in` and `str.replace` compare code points.
/// The funnel's symbol table therefore never fired on a symbol carrying a
/// variation selector or a combining mark, and the punctuation pass turned the
/// unrecognised cluster into a space: the word was deleted from the utterance.
/// U+FE0F is what a chat client or a Markdown document emits for a tick, a
/// cross or an arrow, so this was ordinary pasted text on every Apple render.
///
/// The expectations below are the reference's answers, measured with
/// `loudkit.frontend.speechtext.speech_text` and
/// `loudkit.frontend.text.GraphemeTextFrontend.encode`, not this port's.
final class GraphemeComparisonTests: XCTestCase {

    /// U+FE0F after a table symbol, which is the spelling real text carries.
    func testVariationSelectorStillReachesTheSymbolTable() {
        let cases: [(String, String, String)] = [
            ("Discount 50%\u{FE0F} today", "en", "Discount fifty percent today"),
            ("Done \u{2714}\u{FE0F} and not done.", "en", "Done yes and not done."),
            ("Step one \u{2192}\u{FE0F} step two", "en", "Step one, step two"),
            ("Temperature 21\u{00B0}\u{FE0F} today", "en", "Temperature twenty-one degrees today"),
            ("Coffee @\u{FE0F} home", "en", "Coffee at home"),
            ("Salt &\u{FE0F} pepper", "en", "Salt and pepper"),
            ("5 \u{00D7}\u{FE0F} 3", "en", "five times three"),
            ("Nope \u{2717}\u{FE0F}", "en", "Nope no"),
        ]
        for (text, language, want) in cases {
            XCTAssertEqual(SpeechText.prepared(text, languageId: language), want,
                           "the symbol table did not fire on: \(text)")
        }
    }

    /// A combining mark does the same thing to the cluster, and was the form
    /// the fuzzer found first.
    func testCombiningMarkStillReachesTheSymbolTable() {
        XCTAssertEqual(SpeechText.prepared("a&\u{0301}b", languageId: "en"), "a and b")
        XCTAssertEqual(SpeechText.prepared("50%\u{0301} and 5%", languageId: "en"),
                       "fifty percent and five percent")
    }

    /// The currency pass is the same comparison, and its wording is
    /// per-language: a dropped mark loses the unit word, not just a symbol.
    func testMarkedCurrencyKeepsItsUnitWord() {
        XCTAssertEqual(SpeechText.prepared("Cena: \u{20AC}\u{FE0F}25 za sztuk\u{119}.",
                                           languageId: "pl"),
                       "Cena: euro dwadzie\u{15B}cia pi\u{119}\u{107} za sztuk\u{119}.")
    }

    /// A symbol with no mark was never broken; it must stay unbroken.
    func testBareSymbolsAreUnchanged() {
        XCTAssertEqual(SpeechText.prepared("Discount 50% today", languageId: "en"),
                       "Discount fifty percent today")
        XCTAssertEqual(SpeechText.prepared("Done \u{2714} and not done.", languageId: "en"),
                       "Done yes and not done.")
    }

    private func frontend() throws -> TextFrontend {
        try TextFrontend(tokenizerURL: Fixture.conformanceDir
            .appendingPathComponent("tokenizer.json"))
    }

    /// The comment above the bracket pass says brackets never reach the
    /// tokenizer. A bracket carrying a combining mark used to.
    func testMarkedBracketsDoNotReachTheTokenizer() throws {
        let fe = try frontend()
        XCTAssertEqual(try fe.encode("a]\u{0301}b", language: "en"), [708, 14, 2, 764, 15])
        XCTAssertEqual(try fe.encode("a[\u{0301}b", language: "en"), [708, 14, 2, 764, 15])
        XCTAssertEqual(try fe.encode("a[\u{FE0F}b", language: "en"), [708, 14, 2, 1, 15])
    }

    /// The same law three lines down: a space carrying a combining mark has to
    /// become `[SPACE]` like every other space.
    func testMarkedSpaceBecomesTheSpaceToken() throws {
        let fe = try frontend()
        XCTAssertEqual(try fe.encode("a \u{0301}b", language: "en"), [708, 14, 2, 764, 15])
    }

    /// Word-final capital sigma lowercases to the final form. `String
    /// .lowercased()` skips the rule and answered `\u{3C3}`; the reference,
    /// Rust and JS all answer `\u{3C2}`.
    func testFinalSigmaLowercasesToTheFinalForm() throws {
        let fe = try frontend()
        XCTAssertEqual(try fe.encode("a\u{3A3}", language: "en").last, 1039)  // ς
        XCTAssertEqual(try fe.encode("\u{39F}\u{394}\u{39F}\u{3A3}", language: "en").last, 1039)
        // Word-medial is the non-final form, and always was.
        XCTAssertEqual(try fe.encode("a\u{3A3}b", language: "en").dropLast().last, 1040)  // σ
        // A lone sigma has no cased letter before it, so it is not final.
        XCTAssertEqual(try fe.encode("\u{3A3}", language: "en").last, 1040)
    }
}
