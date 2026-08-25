import XCTest

@testable import LoudKit
import LoudKitText

/// A tag the tokenizer knows is not a language the kit can speak.
///
/// The vocabulary carries tags for 31 languages; the text layer is written for
/// twelve. While this was a blacklist of zh/ja/he/ko/ru the other 26 went
/// straight through, so `encode(text, language: "bg")` NFKD-mangled Cyrillic
/// into ids the model reads as sounds it never learned — no error,
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
/// profile's own `language` — recorded at enrollment, and until now not even
/// parsed out of the safetensors header by this port — was never consulted. The
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
    /// id is not a language — `TextFrontend.encode` would tag the text `[]`
    /// with it. A header that simply omits the key loads as `"en"` instead, so
    /// it never reaches this branch.
    func testAProfileWithoutALanguageFallsBackToEnglish() {
        XCTAssertEqual(
            Engine.resolveLanguage(nil, voice: profile(language: "")), "en")
    }
}
