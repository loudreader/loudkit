import Foundation
import XCTest

import LoudKitText

/// The speech funnel, against the fixture every port is checked with.
///
/// `SpeechText` and `LexicalRespelling` are the shipped
/// implementations that the Python, Go, Rust and JS funnels are described as
/// bit-parity ports *of* — so the one implementation nothing verified was the
/// reference the others are measured against.
///
/// Hand-written cases in five languages are five tests of five different
/// things. `tests/data/conformance/speechtext.json` is one test of one thing,
/// and a disagreement names itself.
final class SpeechFunnelTests: XCTestCase {
    private struct Case: Decodable {
        let text: String
        let language: String?
        let expected: String
    }

    private struct Fixture: Decodable {
        let cases: [Case]
    }

    /// `tests/data/conformance/`, found by walking up from this file rather
    /// than from the process CWD, which `swift test` does not promise.
    ///
    /// `LOUDKIT_FIXTURE_DIR` overrides it, which is how `tools/fuzz_parity.py`
    /// points all five implementations at a generated fixture.
    ///
    /// Only `_DIR` is read. The two names carry two meanings across every port:
    /// `LOUDKIT_FIXTURE_DIR` is the directory, `LOUDKIT_FIXTURE` is the
    /// `vectors.json` file inside it. Accepting the file path here as if it
    /// were a directory would build `vectors.json/speechtext.json` and fail
    /// with an error that names neither variable.
    private static var conformanceDir: URL {
        if let dir = ProcessInfo.processInfo.environment["LOUDKIT_FIXTURE_DIR"], !dir.isEmpty {
            return URL(fileURLWithPath: dir)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // SpeechFunnelTests.swift
            .deletingLastPathComponent()  // LoudKitTextTests
            .deletingLastPathComponent()  // Tests
            .appendingPathComponent("tests/data/conformance")
    }

    private func fixture() throws -> Fixture {
        let url = Self.conformanceDir.appendingPathComponent("speechtext.json")
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }

    func testFunnelMatchesTheSharedFixture() throws {
        let cases = try fixture().cases
        XCTAssertFalse(cases.isEmpty, "the fixture is empty; nothing was compared")

        var mismatches: [String] = []
        for c in cases {
            let got = SpeechText.prepared(c.text, languageId: c.language)
            if got != c.expected {
                mismatches.append(
                    """
                    text:     \(c.text.debugDescription)
                    language: \(c.language.map { "\"\($0)\"" } ?? "nil")
                    expected: \(c.expected.debugDescription)
                    got:      \(got.debugDescription)
                    """
                )
            }
        }
        XCTAssertTrue(
            mismatches.isEmpty,
            "the funnel disagrees with the shared fixture in "
                + "\(mismatches.count)/\(cases.count) cases:\n\n"
                + mismatches.joined(separator: "\n\n")
        )
    }

    /// The funnel must not care about the case of the language tag.
    ///
    /// The tokenizer lowercases its own tag, so a caller passing `"PL"` used
    /// to get Polish *tokens* with English spelling — half the utterance read
    /// one way and half the other, with nothing to indicate it.
    func testLanguageIdIsCaseInsensitive() {
        XCTAssertEqual(
            SpeechText.prepared("download", languageId: "PL"),
            SpeechText.prepared("download", languageId: "pl")
        )
    }

    /// Unicode digits are digits.
    ///
    /// Pinned because a port got this wrong in the other direction: the Rust
    /// funnel used `is_ascii_digit`, so Arabic-Indic numerals fell through as
    /// letters and were read as a word.
    func testUnicodeDigitsAreRecognised() {
        let out = SpeechText.prepared("١٢٣", languageId: "pl")
        XCTAssertNotEqual(out, "١٢٣", "Arabic-Indic digits were passed through as letters")
    }
}
