import Foundation
import XCTest

@testable import LoudKit

/// The splitter, against the fixture the other four ports are held to.
///
/// Where the splits fall is audible, so it is algorithm rather than
/// convenience: a caller who splits differently gets different chunk
/// boundaries, different derived seeds, and different audio from every other
/// port, while `AlgorithmConfig.fingerprint()` goes on declaring the chunking
/// recipe. This module did not exist in Swift, which meant every Swift caller
/// with a paragraph was that caller.
final class ChunkingTests: XCTestCase {
    private struct Case: Decodable {
        let config: String
        let maxTokens: Int
        let prefixTokens: Int
        let splitOn: [String]
        let abbreviations: [String]
        let midSentencePeriod: String
        let text: String
        let chunks: [String]

        enum CodingKeys: String, CodingKey {
            case config
            case maxTokens = "max_tokens"
            case prefixTokens = "prefix_tokens"
            case splitOn = "split_on"
            case abbreviations
            case midSentencePeriod = "mid_sentence_period"
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
            config.abbreviations = c.abbreviations
            config.midSentencePeriod = MidSentencePeriod(rawValue: c.midSentencePeriod)!
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
    /// third different answer, JS counting UTF-16 units cut surrogate pairs in
    /// half, and that bug reached `frontend.encode`.
    func testEstimateCountsCodePoints() {
        // Four code points either way, one emoji is one code point, not the
        // two UTF-16 units it occupies.
        XCTAssertEqual(Chunking.estimateTokens("abcd"), Chunking.estimateTokens("😀😀😀😀"))
        XCTAssertNotEqual(Chunking.estimateTokens("abcd"), Chunking.estimateTokens("😀😀"))
        // A family emoji is one grapheme cluster and several scalars; counting
        // clusters would under-estimate and overflow the window.
        XCTAssertGreaterThan(Chunking.estimateTokens("👨‍👩‍👧‍👦"), Chunking.estimateTokens("ab"))
    }

    /// A period that does not end a sentence must not end a chunk.
    ///
    /// `"But Mr. Smith went home"` used to break after the title and hand the
    /// renderer a seven-character chunk: its own utterance, its own derived
    /// seed, and a token ceiling proportional to seven characters. It is the
    /// only chunk in a 9920-row rendered census that hit that ceiling.
    /// Surveyed over 1200 passages in ten languages, 59 of 3773 cuts landed on
    /// a period inside a sentence; under this law, one.
    func testMidSentencePeriodsAreNotBoundaries() {
        let title =
            "But Mr. Smith went home to the house on the hill where he had lived "
            + "for forty years without ever once complaining about any of it at all."
        XCTAssertNotEqual(
            Chunking.splitText(title, config: ChunkConfig())[0], "But Mr.",
            "the hold did not fire")

        // The old law, kept namable so a pack can say what it was measured under.
        var old = ChunkConfig()
        old.midSentencePeriod = .brk
        XCTAssertEqual(Chunking.splitText(title, config: old)[0], "But Mr.")

        // The half of the law that no list could do: `speech_text` folds a
        // mid-sentence ellipsis to a single period, and the next word is lower
        // case. It is the dominant cause in Polish, which has no abbreviation
        // cuts at all.
        let ellipsis =
            "Grzeja sie i swieca. ciep\u{142}em ktore pamietaja z lata i z kazdej "
            + "innej pory roku na swiecie, a potem gasna powoli i nikt juz nie pamieta."
        XCTAssertFalse(
            Chunking.splitText(ellipsis, config: ChunkConfig())[0].hasSuffix("swieca."))

        // Gated on the period. A comma is followed by a lower-case word almost
        // every time it is written, so a rule that did not gate would veto
        // every comma in the language.
        let commas =
            "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu, "
            + "nu xi omicron pi rho, sigma tau upsilon phi chi psi omega at the end."
        XCTAssertEqual(
            Chunking.splitText(commas, config: ChunkConfig()),
            Chunking.splitText(commas, config: old))

        // "NASA" ends in "A", and "A" is a listed initial; the guard on the
        // character in front of the match is what keeps this one breaking.
        let nasa =
            "The rocket that carried them up there was built by NASA. And the rest "
            + "of the afternoon went by without anybody saying much about it to anyone."
        XCTAssertTrue(
            Chunking.splitText(nasa, config: ChunkConfig())[0].hasSuffix("by NASA."))

        // Every sentence end in the window is held, so the split falls through
        // to the latest comma. A comma break is heard; a chunk of "Mr." is
        // heard worse.
        let norrell =
            "Mr. Norrell, who had been waiting in the hall for the better part of an "
            + "hour, said nothing at all to either of them about what he had seen there."
        XCTAssertTrue(
            Chunking.splitText(norrell, config: ChunkConfig())[0].hasSuffix("hour,"))

        // An empty list is the old behaviour expressed as data. It is not the
        // same statement as naming the law, which is why both spellings exist.
        var bare = ChunkConfig()
        bare.abbreviations = []
        XCTAssertEqual(Chunking.splitText(title, config: bare)[0], "But Mr.")
        XCTAssertFalse(
            Chunking.splitText(ellipsis, config: bare)[0].hasSuffix("swieca."))
    }

    /// The default algorithm hashes to what the reference says it does.
    ///
    /// The production fingerprint is checked against the real manifest in
    /// `AlgorithmConformanceTests`, which needs the checkpoint. This one needs
    /// nothing, and it is the pin that moves when a *default* changes: the
    /// reference asserts the same value in `tests/test_chunking.py`.
    func testTheDefaultFingerprintMatchesTheReference() {
        XCTAssertEqual(AlgorithmConfig().fingerprint(), "6c1c79aab72fa0cb")
    }

    /// The shipping list must be the one the reference ships, sorted.
    func testTheShippingListMatchesTheReference() {
        let entries = ChunkConfig().abbreviations
        XCTAssertEqual(entries, entries.sorted())
        XCTAssertEqual(entries.count, Set(entries).count)
        XCTAssertEqual(entries.count, 24)
        XCTAssertTrue(entries.contains("\u{15B}w"))
    }
}

/// Concatenating mels along time, which is the part of long-form that is easy
/// to get wrong and silent when you do.
///
/// A mel is `(bins, frames)` row-major. Appending the flat arrays end to end
/// puts the second mel's first bin after the first mel's last bin, which is not
/// a spectrogram, and it still renders, into audio that sounds like a fault in
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

        let joined = Engine.joinMelsAlongTime(
            [(mel: left, frames: leftFrames), (mel: right, frames: rightFrames)],
            frames: leftFrames + rightFrames)
        XCTAssertEqual(joined.count, bins * (leftFrames + rightFrames))
        for b in 0..<bins {
            let row = Array(joined[b * 5..<(b + 1) * 5])
            XCTAssertEqual(
                row, [Float(b * 100), Float(b * 100 + 1), Float(b * 100 + 50),
                      Float(b * 100 + 51), Float(b * 100 + 52)],
                "bin \(b) is not this bin's frames in order")
        }
    }

    func testASingleChunkIsItself() {
        XCTAssertEqual(
            Engine.joinMelsAlongTime([(mel: [1, 2, 3], frames: 3)], frames: 3), [1, 2, 3])
    }

    func testNoChunksIsNoMel() {
        XCTAssertEqual(Engine.joinMelsAlongTime([], frames: 0), [])
    }

    // MARK: - The repair for a chunk the window could not hold
    //
    // The Python reference's TestSplitInHalf, case for case.

    func testItHalvesAtAWordBoundary() throws {
        let halves = try XCTUnwrap(Chunking.splitInHalf("one two three four five six"))
        XCTAssertEqual("\(halves.0) \(halves.1)", "one two three four five six")
    }

    func testItDoesNotSeekPunctuation() throws {
        // A weaker separator sits nearer the middle than the comma does, and
        // the comma has no pull of its own: of 27 re-splits taken at the
        // separator nearest the middle, every seam over a second fell on one.
        let halves = try XCTUnwrap(Chunking.splitInHalf("aa bb, cc dddddddddddd ee"))
        XCTAssertEqual(halves.0, "aa bb, cc")
        XCTAssertEqual(halves.1, "dddddddddddd ee")
    }

    func testTheWordBoundaryFallbackBreaksOnANonBreakingSpace() {
        // `splitText`'s last resort, when a window holds no punctuation. The
        // boundary table was introduced for `splitInHalf` and this fallback,
        // ten lines away, was left matching U+0020 alone, and it used
        // `range(of:)`, a grapheme-cluster search, so a space followed by a
        // combining mark was invisible to it too.
        let words = (0..<30).map { String(format: "ord%02d", $0) }
        let chunks = Chunking.splitText(
            words.joined(separator: "\u{00a0}"), config: ChunkConfig())
        XCTAssertGreaterThan(chunks.count, 1, "the case needs to cross a window")
        for c in chunks {
            let last = String(c.split(separator: "\u{00a0}").last ?? "")
            XCTAssertTrue(words.contains(last), "cut mid-word: \(last)")
        }
    }

    func testItCountsScalarsNotGraphemeClusters() throws {
        // CRLF is one grapheme cluster and two Unicode scalars. `Array(text)`
        // yields clusters, so this halved one word later than the other four
        // ports until `splitInHalf` was switched to `unicodeScalars`. The
        // funnel passes CRLF through verbatim, so real source text reaches it.
        let halves = try XCTUnwrap(Chunking.splitInHalf("xx\r\nxx xx xxxxx"))
        XCTAssertEqual(halves.0, "xx\r\nxx")
        XCTAssertEqual(halves.1, "xx xxxxx")
    }

    func testItCutsOnBoundariesTheFunnelKeeps() throws {
        // NBSP and tab survive the funnel and are word boundaries. Matching
        // only U+0020 made a capped chunk whose separators were all
        // non-breaking come back unsplittable, so it shipped its truncation.
        for sep in ["\u{00a0}", "\t", "\u{202f}"] {
            let halves = try XCTUnwrap(
                Chunking.splitInHalf("alpha\(sep)beta\(sep)gamma"), sep)
            XCTAssertEqual(halves.0, "alpha\(sep)beta", sep)
            XCTAssertEqual(halves.1, "gamma", sep)
        }
    }

    func testASingleUnbrokenRunIsRefused() {
        // Splitting it would have to cut a word, which is worse than the
        // truncation it would be repairing.
        XCTAssertNil(Chunking.splitInHalf("omringden."))
        XCTAssertNil(Chunking.splitInHalf(""))
        // Trimming can empty a half the scan thought was interior. Unreachable
        // through the engine, which trims first, but this is public.
        XCTAssertNil(Chunking.splitInHalf("x \t"))
    }

    func testNeitherHalfIsEmpty() throws {
        for text in ["a bb", "aaaaaaaa b", "a bbbbbbbb"] {
            let halves = try XCTUnwrap(Chunking.splitInHalf(text), text)
            XCTAssertFalse(halves.0.isEmpty, text)
            XCTAssertFalse(halves.1.isEmpty, text)
        }
    }

    func testCapResplitDefaultsToTheLaw() {
        XCTAssertEqual(ChunkConfig().capResplit, .word)
        XCTAssertNil(CapResplit(rawValue: "halve"))
    }

    // MARK: - The re-split plan, against the shared fixture
    //
    // Holds this port to the `cap_resplit` law without weights. The token
    // streams need a model; the plan does not. Which window carries which
    // index, where the split falls and which seed each window draws are
    // computable from the fixture alone, and they are exactly the three things
    // that went wrong while this law was written: a second half seeded from the
    // next chunk's stream, a split counting grapheme clusters, and a queue that
    // did not advance. A port that gets any of them wrong produces different
    // audio while every other test still passes.
    //
    // Whether a chunk overran needs the model, so that one bit is read from the
    // fixture; everything the port decides given it is recomputed and compared.

    private func deriveSeed(_ seed: UInt64, _ stream: UInt64) -> UInt64 {
        seed &* 0x9E37_79B9_7F4A_7C15 &+ stream &* 0xBF58_476D_1CE4_E5B9
    }

    func testTheResplitPlanMatchesTheSharedFixture() throws {
        try checkResplitPlan("vectors.json", decode: "single")
    }

    func testFusionResplitPlanMatchesFixture() throws {
        try checkResplitPlan("vectors_fusion_mtp2.json", decode: "fusion_mtp2")
    }

    private func checkResplitPlan(_ file: String, decode: String) throws {
        let url = Fixture.conformanceDir.appendingPathComponent(file)
        guard let data = try? Data(contentsOf: url),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let section = json["resplit"] as? [String: Any]
        else {
            // Not a skip. `vectors.json` and its `resplit` section are
            // committed, so absence means the fixture moved or lost a section,
            // and skipping retires the law this test exists for while the
            // suite still reports success.
            XCTFail("vectors.json has no resplit section; nothing was compared")
            throw XCTSkip("no resplit section")
        }

        let resplitStream = UInt64(section["resplit_stream"] as! Int)
        let chunkBase = UInt64(section["chunk_stream_base"] as! Int)
        let prefixTokens = section["prefix_tokens"] as! Int

        for c in section["cases"] as! [[String: Any]] {
            let name = c["name"] as! String
            let seed = UInt64(c["seed"] as! Int)
            // The case moves the window, so the chunk budget moves with it: the
            // config refuses a budget larger than the window, and the engine
            // gates the re-split on the window rather than on any cap.
            var config = ChunkConfig()
            config.maxTokens = c["window"] as! Int
            XCTAssertEqual(config.capResplit, .word, "the fixture pins the law")
            let prepared = c["prepared"] as! String
            let windows = c["windows"] as! [[String: Any]]
            let texts = Chunking.splitText(prepared, config: config)

            func tail(_ w: [String: Any]) -> [Int] {
                let all = w["tokens"] as! [Int]
                return try! Engine.carryFrom(all, prefixTokens: prefixTokens,
                                             startSpeechToken: 6561, decode: decode)
            }
            func check(_ at: Int, _ text: String, _ sd: UInt64, _ pre: [Int],
                       _ split: Bool, _ index: Int) {
                let w = windows[at]
                XCTAssertEqual(w["index"] as! Int, index,
                    "\(name) window \(at): a moved index moves every later chunk's seed")
                XCTAssertEqual(w["text"] as! String, text, "\(name) window \(at): text")
                XCTAssertEqual(w["split"] as! Bool, split, "\(name) window \(at): split")
                XCTAssertEqual(w["seed"] as! String, "0x" + String(sd, radix: 16),
                    "\(name) window \(at): the second half draws from its own stream")
                XCTAssertEqual(w["prefix"] as! [Int], pre, "\(name) window \(at): carry")
            }

            var wi = 0
            var carry: [Int] = []
            for (index, chunkText) in texts.enumerated() {
                // Chunk 0 draws the caller's seed itself; the base applies
                // from chunk 1 up.
                let chunkSeed =
                    index == 0 ? seed : deriveSeed(seed, chunkBase + UInt64(index))
                let wasSplit = windows[wi]["split"] as! Bool
                if wasSplit {
                    let halves = try XCTUnwrap(
                        Chunking.splitInHalf(chunkText),
                        "\(name) chunk \(index): fixture split it, this port cannot")
                    check(wi, halves.0, chunkSeed, carry, true, index)
                    let pre = tail(windows[wi])
                    wi += 1
                    check(wi, halves.1, deriveSeed(chunkSeed, resplitStream), pre, true, index)
                    carry = tail(windows[wi])
                    wi += 1
                } else {
                    check(wi, chunkText, chunkSeed, carry, false, index)
                    carry = tail(windows[wi])
                    wi += 1
                }
            }
            XCTAssertEqual(wi, windows.count, "\(name): window count")
        }
    }
}
