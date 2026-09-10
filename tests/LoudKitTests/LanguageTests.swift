import XCTest

@testable import LoudKit
@testable import LoudKitText

/// A tag the tokenizer knows is not a language the kit can speak.
///
/// The vocabulary carries tags for 31 languages; the text layer is written for
/// twelve. While this was a blacklist of zh/ja/he/ko/ru the other 26 went
/// straight through, so `encode(text, language: "bg")` NFKD-mangled Cyrillic
/// into ids the model reads as sounds it never learned, no error,
/// plausible-sounding audio, wrong language.
///
/// The roster is asserted against the number grammars rather than a literal
/// list: `numbers.json` is the one authority, and a port with its own copy is a
/// port that disagrees with Python the next time a grammar is added. Checked on
/// the roster rather than through `encode` because a `TextFrontend` needs the
/// tokenizer asset; the accepting path is covered by the conformance run.
final class FrontendRosterTests: XCTestCase {
    func testTheRosterIsTheTwelveInNumbersJSON() {
        let roster = TextFrontend.supportedLanguages
        XCTAssertEqual(roster, Numbers.supportedLanguages)
        XCTAssertEqual(roster.count, 12, "roster: \(roster)")
    }

    func testOffRosterTagsAreRefused() {
        let roster = TextFrontend.supportedLanguages
        for lang in ["en", "pl", "sv"] {
            XCTAssertTrue(roster.contains(lang), "\(lang) is on the roster")
        }
        // Cyrillic and Czech are tokenizer tags, never languages this build speaks.
        for lang in ["bg", "cs", "zh"] {
            XCTAssertFalse(roster.contains(lang), "\(lang) must be refused")
        }
    }
}

/// The obvious call must not be the wrong one.
///
/// `engine.synthesize("Cześć", voice: polishVoice)` used to run Polish text
/// through the English frontend, because `language` defaulted to `"en"` and a
/// profile's own `language`, recorded at enrollment, and until now not even
/// parsed out of the safetensors header by this port, was never consulted. The
/// chain is now argument, then voice, then `"en"`, and these are its links.
///
/// Tested against the resolver rather than through `synthesize` because this
/// package has no weight-free engine seam: `Engine`'s four components are
/// concrete final classes needing the checkpoint and CoreML models, and there
/// are no protocols to substitute. The resolver is the whole of the new
/// behaviour; `testTheHeaderLanguageIsRead` below covers the other half, that
/// the field arrives from the file at all.
final class LanguageResolutionTests: XCTestCase {
    private func profile(language: String) -> VoiceProfile {
        VoiceProfile(
            name: "fake",
            speakerEmbedding: [Float](repeating: 0.0625, count: 256),
            flowEmbedding: [Float](repeating: 0.0625, count: 192),
            promptTokens: [1, 2, 3],
            promptMel: [Float](repeating: 0.1, count: 80 * 4),
            promptMelFrames: 4,
            condPromptTokens: [1, 2, 3],
            language: language)
    }

    func testAPolishVoiceReadsPolishByDefault() {
        XCTAssertEqual(
            Engine.resolveLanguage(nil, voice: profile(language: "pl")), "pl")
    }

    func testAnExplicitLanguageOverridesTheProfile() {
        XCTAssertEqual(
            Engine.resolveLanguage("en", voice: profile(language: "pl")), "en")
    }

    /// A hand-built profile can carry an empty language, and an empty language
    /// id is not a language, `TextFrontend.encode` would tag the text `[]`
    /// with it. A header that simply omits the key loads as `"en"` instead, so
    /// it never reaches this branch.
    func testAProfileWithoutALanguageFallsBackToEnglish() {
        XCTAssertEqual(
            Engine.resolveLanguage(nil, voice: profile(language: "")), "en")
    }
}

/// The funnel passes whose patterns are built from grammar data actually run,
/// in every language that declares the data.
///
/// They used to compile with `try?`, so a pattern that failed to build skipped
/// its pass in silence: a date stayed written as digits, an ordinal stayed
/// "1st", a price lost its currency word. Nothing said so, and the render was
/// plausible. A `try!` is the refusal; these assertions are what notices.
final class FunnelPatternTests: XCTestCase {
    /// The day-first pattern with the month written out, which every language
    /// on the roster reads. The dotted form is deliberately refused in some, so
    /// it would not tell a skipped pass from a rule.
    func testEveryLanguagesDatePassRuns() {
        for language in Dates.supportedLanguages {
            guard let march = Dates.monthName(3, language: language) else {
                XCTFail("\(language) declares no month names")
                continue
            }
            let written = "12 \(march) 2026"
            XCTAssertNotEqual(Dates.expandDates(written, language: language), written,
                              "the date pass did not run for \(language)")
        }
    }

    func testEveryLanguagesOrdinalPassRuns() {
        // Only the languages whose rules declare suffixes have a pass to run.
        var ran = 0
        for language in Dates.supportedLanguages {
            for probe in ["1st", "1.", "1er", "1º", "1:a", "1e"] {
                if Dates.expandOrdinals(probe, language: language) != probe {
                    ran += 1
                    break
                }
            }
        }
        XCTAssertGreaterThan(ran, 0, "no language expanded an ordinal")
    }

    func testThePricePassesRunInBothOrders() {
        XCTAssertNotEqual(SpeechText.prepared("$0.49", languageId: "en"), "$0.49")
        XCTAssertNotEqual(SpeechText.prepared("2.50 \u{20AC}", languageId: "en"), "2.50 \u{20AC}")
    }

    /// 24:00 is ISO 8601 end of day and reads; 24 with any other minute is not
    /// a time in any convention and has to survive the pass untouched.
    ///
    /// Untouched means byte for byte. The guard returned to the loop after the
    /// run up to the match had already been appended and without moving the
    /// cursor past it, so the text before an admitted-then-rejected match came
    /// out twice: "at 24:30 today" read as *at at 24:30 today*, and one such
    /// match poisoned everything before it. Python and Go answer these four
    /// probes with the input unchanged.
    func testAnHourOfTwentyFourWithMinutesIsLeftExactlyAsWritten() {
        for probe in ["at 24:30 today", "x 24:45 y 24:50 z", "24:01", "the 24:59 train"] {
            XCTAssertEqual(Numbers.expandTimes(probe, language: "en"), probe,
                           "the time pass rewrote a non-time")
        }
        XCTAssertEqual(Numbers.expandTimes("at 24:00 today", language: "en"),
                       "at twenty-four today", "24:00 is end of day and reads")
        XCTAssertEqual(Numbers.expandTimes("at 14:30 today", language: "en"),
                       "at fourteen thirty today", "an ordinary time still reads")
    }
}
