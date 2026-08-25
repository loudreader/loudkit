import XCTest

@testable import LoudKitText

/// A digit run with two or more separators is a version, an address or a date —
/// never a number.
///
/// All three used to be read as one: with the comma as the decimal mark the dots
/// were treated as thousands grouping and the segments concatenated, so
/// `192.168.0.1` was spoken as "nineteen million two hundred sixteen thousand
/// eight hundred one". The Python reference additionally crashed on these, which
/// is how the class was found.
///
/// Regression tests: every literal below is one that shipped wrong.
final class NotQuantityTests: XCTestCase {
    private let notQuantities = [
        "1.2.3", "1.2.3.4", "192.168.0.1", "12.03.2026", "10.0.0.255",
    ]

    func testDigitsThatAreNotQuantitiesAreLeftAlone() {
        for lang in Numbers.supportedLanguages {
            for literal in notQuantities {
                XCTAssertEqual(
                    Numbers.expandNumbers(literal, language: lang), literal,
                    "\(lang): \(literal) was read as a number")
            }
        }
    }

    func testRealNumbersStillRead() {
        // The guard must not buy correctness by refusing everything.
        for lang in Numbers.supportedLanguages {
            for literal in ["7", "2,5", "2.5"] {
                XCTAssertNotEqual(
                    Numbers.expandNumbers(literal, language: lang), literal,
                    "\(lang): \(literal) was left as digits")
            }
        }
    }

    func testGroupedThousandsAreStillANumber() {
        // The rule is "three digits after the first separator", not "at most
        // one separator" — a guard that refused every multi-separator run would
        // stop reading grouped thousands, a regression dressed as a fix.
        for lang in Numbers.supportedLanguages where lang != "en" {
            XCTAssertNotEqual(
                Numbers.expandNumbers("1.234.567", language: lang), "1.234.567", lang)
        }
        XCTAssertNotEqual(Numbers.expandNumbers("1,234,567", language: "en"), "1,234,567")
    }

    func testATimeIsNotPartOfADate() {
        // `12.03` matched inside `12.03.2026`, so the ordinary written date of
        // five of the twelve languages was spoken as a clock time with the year
        // trailing behind it.
        for lang in Numbers.supportedLanguages {
            for literal in ["12.03.2026", "am 05.11.2025 kam"] {
                XCTAssertEqual(
                    Numbers.expandTimes(literal, language: lang), literal, "\(lang): \(literal)")
            }
            // A dotted time reads only where the dot is not the decimal
            // point: `14.30` is half past two in eleven of these languages and
            // a number in the twelfth. Asserting it for all twelve made every
            // English decimal with two fraction digits a clock time.
            if lang != "en" {
                XCTAssertNotEqual(Numbers.expandTimes("14.30", language: lang), "14.30",
                                  "\(lang): 14.30")
            } else {
                XCTAssertEqual(Numbers.expandTimes("14.30", language: lang), "14.30",
                               "\(lang): 14.30")
            }
            for literal in ["14:30", "at 14:30."] {
                XCTAssertNotEqual(
                    Numbers.expandTimes(literal, language: lang), literal, "\(lang): \(literal)")
            }
        }
    }
}
