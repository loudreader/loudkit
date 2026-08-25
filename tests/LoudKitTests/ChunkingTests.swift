import Foundation
import XCTest

@testable import LoudKit

/// The splitter, against the fixture the other four ports are held to.
///
/// Where the splits fall is audible, so it is algorithm rather than
/// convenience: a caller who splits differently gets different chunk
/// boundaries, different derived seeds, and different audio from every other
/// port — while `AlgorithmConfig.fingerprint()` goes on declaring the chunking
/// recipe. This module did not exist in Swift, which meant every Swift caller
/// with a paragraph was that caller.
final class ChunkingTests: XCTestCase {
    private struct Case: Decodable {
        let config: String
        let maxTokens: Int
        let prefixTokens: Int
        let splitOn: [String]
        let text: String
        let chunks: [String]

        enum CodingKeys: String, CodingKey {
            case config
            case maxTokens = "max_tokens"
            case prefixTokens = "prefix_tokens"
            case splitOn = "split_on"
            case text, chunks
        }
    }

    private struct Fixture: Decodable {
        let chunking: [Case]
    }

    /// `tests/data/conformance/`, found by walking up from this file rather
    /// than from the process CWD, which `swift test` does not promise.
    private static var conformanceDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ChunkingTests.swift
            .deletingLastPathComponent()  // LoudKitTests
            .deletingLastPathComponent()  // tests
            .appendingPathComponent("tests/data/conformance")
    }

    func testSplitsWhereTheSharedFixtureSays() throws {
        let url = Self.conformanceDir.appendingPathComponent("speechtext.json")
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertFalse(fixture.chunking.isEmpty, "the fixture has no chunking cases")

        var mismatches: [String] = []
        for c in fixture.chunking {
            var config = ChunkConfig()
            config.maxTokens = c.maxTokens
            config.prefixTokens = c.prefixTokens
            config.splitOn = c.splitOn
            let got = Chunking.splitText(c.text, config: config)
            if got != c.chunks {
                mismatches.append(
                    """
                    config:   \(c.config)  max=\(c.maxTokens) prefix=\(c.prefixTokens)
                    text:     \(c.text.prefix(70).debugDescription)
                    expected: \(c.chunks.map { String($0.prefix(30)) })
                    got:      \(got.map { String($0.prefix(30)) })
                    """)
            }
        }
        XCTAssertTrue(
            mismatches.isEmpty,
            "the splitter disagrees with the shared fixture in "
                + "\(mismatches.count)/\(fixture.chunking.count) cases:\n\n"
                + mismatches.joined(separator: "\n\n"))
    }

    /// A character is a code point, not a UTF-16 unit and not a grapheme
    /// cluster. Swift's `String.count` counts the third of those, which is a
    /// third different answer — JS counting UTF-16 units cut surrogate pairs in
    /// half, and that bug reached `frontend.encode`.
    func testEstimateCountsCodePoints() {
        // Four code points either way — one emoji is one code point, not the
        // two UTF-16 units it occupies.
        XCTAssertEqual(Chunking.estimateTokens("abcd"), Chunking.estimateTokens("😀😀😀😀"))
        XCTAssertNotEqual(Chunking.estimateTokens("abcd"), Chunking.estimateTokens("😀😀"))
        // A family emoji is one grapheme cluster and several scalars; counting
        // clusters would under-estimate and overflow the window.
        XCTAssertGreaterThan(Chunking.estimateTokens("👨‍👩‍👧‍👦"), Chunking.estimateTokens("ab"))
    }
}

/// Concatenating mels along time, which is the part of long-form that is easy
/// to get wrong and silent when you do.
///
/// A mel is `(bins, frames)` row-major. Appending the flat arrays end to end
/// puts the second mel's first bin after the first mel's last bin, which is not
/// a spectrogram — and it still renders, into audio that sounds like a fault in
/// the model rather than a fault in the join. Three ports had this bug.
final class MelConcatenationTests: XCTestCase {
    func testFramesAreJoinedPerBinNotEndToEnd() {
        // Two 3-bin mels: values encode (bin * 100 + frame) so a wrong join is
        // readable rather than merely unequal.
        let bins = 3
        let leftFrames = 2, rightFrames = 3
        var left: [Float] = [], right: [Float] = []
        for b in 0..<bins {
            for f in 0..<leftFrames { left.append(Float(b * 100 + f)) }
        }
        for b in 0..<bins {
            for f in 0..<rightFrames { right.append(Float(b * 100 + 50 + f)) }
        }

        let joined = Engine.appendMelAlongTime(left, leftFrames, right, rightFrames)
        XCTAssertEqual(joined.count, bins * (leftFrames + rightFrames))
        for b in 0..<bins {
            let row = Array(joined[b * 5..<(b + 1) * 5])
            XCTAssertEqual(
                row, [Float(b * 100), Float(b * 100 + 1), Float(b * 100 + 50),
                      Float(b * 100 + 51), Float(b * 100 + 52)],
                "bin \(b) is not this bin's frames in order")
        }
    }

    func testAnEmptyLeftIsTheRight() {
        XCTAssertEqual(Engine.appendMelAlongTime([], 0, [1, 2, 3], 3), [1, 2, 3])
    }
}
