import XCTest
@testable import LoudKitText

/// The abbreviation pass looks at its boundaries rather than consuming them.
///
/// A pattern that eats the character on either side leaves the next occurrence
/// with no boundary to start from, because the matches are non-overlapping. Of
/// 440 cases built from every abbreviation in all twelve languages, 176 came
/// out different from the reference: any repeat expanded the first and left the
/// second written, which is ordinary prose in every language that has a list.
final class AbbreviationBoundaryTests: XCTestCase {
    func testARepeatedAbbreviationIsExpandedEveryTime() {
        for (text, language, want) in [
            ("Apples, etc. etc. and pears.", "en", "Apples, etcetera etcetera and pears."),
            ("Kot, pies, tzn. tzn. zwierzeta.", "pl", "Kot, pies, to znaczy to znaczy zwierzeta."),
            ("z.B. z.B. hier", "de", "zum Beispiel zum Beispiel hier"),
        ] {
            XCTAssertEqual(SpeechText.prepared(text, languageId: language), want)
        }
    }

    /// The boundary still has to be there: an abbreviation glued to a word is
    /// part of that word.
    func testAnAbbreviationInsideAWordIsLeftAlone() {
        XCTAssertEqual(SpeechText.prepared("xetc. y", languageId: "en"), "xetc. y")
        XCTAssertEqual(SpeechText.prepared("a etc.b", languageId: "en"), "a etc.b")
    }
}
