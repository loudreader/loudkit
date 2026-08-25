import XCTest

@testable import LoudKitText

/// The number verbalizer against both shared corpora: the hand-written fixture
/// (expectations from each language's own reference description) and the CLDR
/// differential (1300 rows Unicode wrote; disputed rows skipped with reasons).
final class NumbersTests: XCTestCase {
    private func fixture(_ name: String) throws -> [String: Any] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // LoudKitTextTests
            .deletingLastPathComponent()  // tests
            .appendingPathComponent("data/conformance/\(name)")
        let data = try Data(contentsOf: url)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testCardinalMatchesTheHandFixture() throws {
        let fx = try fixture("numbers.json")
        let cardinals = try XCTUnwrap(fx["cardinals"] as? [String: [[String: Any]]])
        XCTAssertFalse(cardinals.isEmpty, "nothing was compared")
        for (lang, cases) in cardinals {
            for kase in cases {
                let value = (kase["value"] as! NSNumber).int64Value
                let got = Numbers.cardinal(value, language: lang)
                XCTAssertEqual(got, kase["expect"] as? String, "\(lang) \(value)")
            }
        }
        for kase in try XCTUnwrap(fx["gendered"] as? [[String: Any]]) {
            let lang = kase["language"] as! String
            let value = (kase["value"] as! NSNumber).int64Value
            let gender = kase["gender"] as! String
            let got = Numbers.cardinal(value, language: lang, gender: gender)
            XCTAssertEqual(got, kase["expect"] as? String, "\(lang) \(value) g=\(gender)")
        }
    }

    func testCardinalMatchesCLDR() throws {
        let fx = try fixture("numbers_cldr.json")
        let all = try XCTUnwrap(fx["cases"] as? [String: [[String: Any]]])
        var checked = 0
        for (lang, cases) in all {
            for kase in cases {
                if kase["disputed"] != nil { continue }
                let value = (kase["value"] as! NSNumber).int64Value
                let gender = kase["gender"] as? String ?? ""
                // Past our scale: the refusal (nil) is the declared behaviour.
                guard let got = Numbers.cardinal(value, language: lang, gender: gender) else {
                    continue
                }
                checked += 1
                XCTAssertEqual(got, kase["expect"] as? String, "\(lang) \(value) g=\(gender)")
            }
        }
        XCTAssertGreaterThan(checked, 1000, "only \(checked) CLDR rows ran")
    }

    func testExpandNumbersInRunningText() {
        let cases: [(String, String, String)] = [
            ("I have 21 apples.", "en", "I have twenty-one apples."),
            ("3.5", "en", "three point five"),
            ("1,200", "en", "one thousand two hundred"),
            ("3,5", "pl", "trzy przecinek pięć"),
            ("Es kostet 250 Euro.", "de", "Es kostet zweihundertfünfzig Euro."),
            ("21 apples", "xx", "21 apples"),
            ("no numbers here", "en", "no numbers here"),
        ]
        for (text, lang, want) in cases {
            XCTAssertEqual(Numbers.expandNumbers(text, language: lang), want, text)
        }
    }

    /// The digit-run glue rules, against the minimal repros the parity fuzzer
    /// reduced seeds 1 and 2 to (`tools/fuzz_parity.py`, 41 of 600 cases).
    ///
    /// Every row is a string at least one of the five ports read differently
    /// from the other four, so every row is a rule that was decided rather than
    /// observed. The rule is `docs/reference/preprocess.md`: a maximal run of
    /// digits and separators that does not reduce to a single readable number is
    /// left written — never half-spoken, never welded to a word.
    ///
    /// Swift was not in the fuzzer's `--ports` list when these were found; it is
    /// architecturally closest to Python, and shares exactly one of the eight
    /// families with it (the ragged run below).
    func testDigitRunGlueAgainstTheFuzzMinimalRepros() {
        let cases: [(String, String, String, String)] = [
            // A ragged run that reaches a letter is one token, and the token is
            // left written. The backtracking engines match a bare `1` here,
            // where Go and Rust bind `1 002` and walk into the `R` from inside
            // the digits; reading the `1` alone said "en 0023R" — half the run
            // spoken and the rest welded to a word.
            ("ragged run into a letter", "1 0023R", "da", "1 0023R"),
            ("same, with an exponent behind it", "200 0001e-3", "en", "200 0001e-3"),
            // The same run with no letter in it is not refused: the space stops
            // grouping, and each side of it is its own number.
            ("ragged run, nothing glued", "1 0023", "da", "en nul nul to tre"),
            // A fraction belongs to the segment it is written against. Go and
            // Rust scanned digits and grouping spaces only and emitted `.5`
            // verbatim behind a spoken cardinal.
            (
                "fraction on the last segment", "4 5672.5", "es",
                "cuatro cinco mil seiscientos setenta y dos coma cinco"
            ),
            ("fraction on the first", "1 000.0 3", "nl", "duizend komma nul drie"),
            // The forward walk crosses a thousands space and stops at the next
            // number: four digits behind the space are not a group, so the `e`
            // two tokens away is not glued to anything.
            ("exponent two tokens away", "1000 5.1e+3", "en", "one thousand 5.1e+3"),
            // A one-character slice is not a thousands group. Python's
            // `text[i:i+3].isdigit()` is True for `"5"`, which crossed the space
            // and refused a number nothing was glued to; the bounds check here
            // asks all three positions.
            ("single digit is not a group", "R2 2", "en", "R2 two"),
            ("same", "R2 5", "en", "R2 five"),
            // A match the lookbehind refuses must not consume the run behind it.
            // Go and Rust post-checked the lookbehind and skipped the whole
            // region, losing the `1000`.
            ("refused match, next run intact", "e3 1000", "sv", "e3 ettusen"),
            // Both walks cross a grouping space, so a group glued to a word
            // through one is refused whole rather than read from the middle.
            ("backward walk crosses the space", "C0200 000", "it", "C0200 000"),
            ("forward walk crosses it", "200 000x", "en", "200 000x"),
            ("backward, one-digit first group", "a1 000 000", "en", "a1 000 000"),
            ("backward, three-digit first group", "x200 000", "en", "x200 000"),
            // …and does not cross an ordinary one: nothing is glued to `Sold`.
            ("word, space, grouped number", "Sold 200 000", "en", "Sold two hundred thousand"),
            // A non-ASCII letter glues exactly as an ASCII one does. JS read
            // this as "étwo" for want of a `u` flag.
            ("non-ASCII letter in front", "é2", "en", "é2"),
            ("non-ASCII letter behind", "1 234 567.é", "de", "1 234 567.é"),
            // A grouped run that cannot reach a boundary drops back to being one
            // number per segment — the shape `+1 202 555 0199` arrives in when
            // the phone pass has already declined it for want of a plus.
            (
                "grouped run, no boundary", "1 202 555 0199", "en",
                "one two hundred and two five hundred and fifty-five zero one nine nine"
            ),
        ]
        for (why, text, lang, want) in cases {
            XCTAssertEqual(Numbers.expandNumbers(text, language: lang), want, "\(why): \(text)")
        }
    }

    /// The infix is consumed only where there is one to consume. Go called the
    /// consumer unconditionally and, with an empty infix, ate the whitespace
    /// behind the time in every language but German: `3.14 é` came out
    /// *três catorzeé*.
    func testAClockTimeDoesNotEatTheSpaceBehindIt() {
        XCTAssertEqual(Numbers.expandTimes("3.14 é", language: "pt"), "três catorze é")
        XCTAssertEqual(Numbers.expandTimes("3.14 é", language: "de"), "drei Uhr vierzehn é")
    }

    /// A day past the last of the month is not a date. Go's guards passed the
    /// regex class `\d.,:/-` to a literal rune test, so no digit ever matched
    /// one and `42.3.2026` was read as *4zweite März …* — a day invented out of
    /// the second digit of a number that was never a date.
    func testAnImpossibleDayIsLeftWritten() {
        XCTAssertEqual(SpeechText.prepared("42.3.2026", languageId: "de"), "42.3.2026")
        XCTAssertEqual(
            SpeechText.prepared("12.03.2026", languageId: "de"),
            "zwölfte März zweitausendsechsundzwanzig")
    }

    /// German writes the time with the word the spoken form also carries: the
    /// reading puts the infix between hour and minutes, so the written "Uhr"
    /// behind the digits is that same token and is consumed, not duplicated.
    func testAWrittenInfixIsNotSaidTwice() {
        let cases: [(String, String)] = [
            ("um 14:30 Uhr", "um vierzehn Uhr dreißig"),
            // A tab before the word consumes exactly like a space.
            ("um 14:30\tUhr", "um vierzehn Uhr dreißig"),
            ("um 24:00 Uhr an.", "um vierundzwanzig Uhr an."),
            // The dotted form runs through the second pattern.
            ("Termin um 14.30 Uhr.", "Termin um vierzehn Uhr dreißig."),
            // Without the word nothing changes.
            ("um 14:30", "um vierzehn Uhr dreißig"),
            // The noun on its own is not part of any time.
            (
                "Es ist 14:30 Uhr und die Uhr tickt.",
                "Es ist vierzehn Uhr dreißig und die Uhr tickt."
            ),
            // Infix inside a longer word keeps its head.
            ("Die Uhrzeit ist 14:30.", "Die Uhrzeit ist vierzehn Uhr dreißig."),
        ]
        for (text, want) in cases {
            XCTAssertEqual(Numbers.expandTimes(text, language: "de"), want, text)
        }
        // Eleven of the twelve grammars carry an empty infix: nothing to consume.
        XCTAssertEqual(
            Numbers.expandTimes("at 14:30 sharp", language: "en"), "at fourteen thirty sharp")
    }
}
