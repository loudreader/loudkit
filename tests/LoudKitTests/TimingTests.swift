import XCTest

@testable import LoudKit

/// Timestamps: exact at the chunk, estimated at the word.
///
/// The whole value of the feature is that a reading app can trust the first
/// tier and is told, loudly, not to trust the second in the same way. So the
/// tests are split the same way: the chunk assertions are equalities, the word
/// assertions are invariants (monotonic, inside the chunk, every word present)
/// and nothing here claims a word lands where a listener would say it does.
///
/// All of it is arithmetic over sample counts, so none of it needs the
/// checkpoint or the CoreML packages — it runs in every checkout, which is the
/// point of keeping the two tiers in a module of their own.
final class TimingTests: XCTestCase {
    private static let sampleRate = 24_000

    // MARK: the exact tier

    func testChunksAreAdjacentToTheLastBit() {
        // A highlight that switches on `time >= start` flickers on a gap and
        // double-lights on an overlap, and both are invisible to a comparison
        // with a tolerance. Offsets accumulate as integer samples for exactly
        // this reason, so these are `==` and not `accuracy:`.
        let spans = [
            ChunkSpan(text: "a b", samples: 7_001, tokens: 3),
            ChunkSpan(text: "c d e", samples: 13_337, tokens: 5),
        ]
        let got = Timing.timeline(spans, sampleRate: Self.sampleRate)
        XCTAssertEqual(got[0].start, 0.0)
        XCTAssertEqual(got[1].start, got[0].end)
        XCTAssertEqual(got[1].end, Double(7_001 + 13_337) / Double(Self.sampleRate))
    }

    func testTheSpansCoverTheWholeRenderWithNothingLeftOver() {
        let spans = [
            ChunkSpan(text: "one", samples: 100, tokens: 1),
            ChunkSpan(text: "two", samples: 200, tokens: 2),
            ChunkSpan(text: "three", samples: 300, tokens: 3),
        ]
        let got = Timing.timeline(spans, sampleRate: Self.sampleRate)
        XCTAssertEqual(
            got.reduce(0.0) { $0 + $1.duration }, 600.0 / Double(Self.sampleRate), accuracy: 1e-12)
        XCTAssertEqual(got.map(\.tokens), [1, 2, 3])
    }

    func testAnEmptyRenderIsAnEmptyTimeline() {
        XCTAssertTrue(Timing.timeline([], sampleRate: Self.sampleRate).isEmpty)
    }

    func testShiftingMovesTheWordsWithTheChunk() {
        // What a caller stitching a stream does: every delivered chunk starts
        // at zero, and only the caller knows how much audio it has queued.
        let span = Timing.timeline(
            [ChunkSpan(text: "a bb", samples: 240, tokens: 2)], sampleRate: Self.sampleRate)[0]
        let moved = span.shifted(by: 1.0)
        XCTAssertEqual(moved.start, span.start + 1.0)
        XCTAssertEqual(moved.end, span.end + 1.0)
        XCTAssertEqual(moved.words.map(\.start), span.words.map { $0.start + 1.0 })
        XCTAssertEqual(moved.words.map(\.text), span.words.map(\.text))
    }

    // MARK: the estimated tier

    func testWordsTileTheChunkWithoutGaps() {
        let words = Timing.estimateWords("alpha beta gamma", start: 1.0, end: 4.0)
        XCTAssertEqual(words.map(\.text), ["alpha", "beta", "gamma"])
        XCTAssertEqual(words.first?.start, 1.0)
        XCTAssertEqual(words.last?.end, 4.0)
        for (left, right) in zip(words, words.dropFirst()) {
            XCTAssertEqual(left.end, right.start)
        }
    }

    func testTimesAreMonotonicAndInsideTheChunk() {
        let words = Timing.estimateWords("a bb ccc dddd e", start: 2.5, end: 3.25)
        for word in words {
            XCTAssertLessThanOrEqual(2.5, word.start)
            XCTAssertLessThanOrEqual(word.start, word.end)
            XCTAssertLessThanOrEqual(word.end, 3.25)
        }
        XCTAssertEqual(words.map(\.start), words.map(\.start).sorted())
    }

    func testALongerWordIsGivenLonger() {
        // The whole content of the estimate: characters stand in for seconds.
        // Nothing else here knows how long a word takes.
        let words = Timing.estimateWords("hi internationalisation", start: 0.0, end: 1.0)
        XCTAssertEqual(words.count, 2)
        XCTAssertGreaterThan(words[1].end - words[1].start, words[0].end - words[0].start)
    }

    func testPunctuationStaysWithItsWord() {
        // A caller highlighting `"end."` wants the full stop lit with the word,
        // and a caller matching back against their own text needs the substring
        // to be a substring.
        let words = Timing.estimateWords("Hello, world!", start: 0.0, end: 1.0)
        XCTAssertEqual(words.map(\.text), ["Hello,", "world!"])
    }

    func testNoTextIsNoWordsRatherThanADivisionByZero() {
        XCTAssertTrue(Timing.estimateWords("   ", start: 0.0, end: 1.0).isEmpty)
        XCTAssertTrue(Timing.estimateWords("", start: 0.0, end: 1.0).isEmpty)
        XCTAssertTrue(Timing.estimateWords("\n\t ", start: 0.0, end: 1.0).isEmpty)
    }

    func testLengthIsCountedInCodePointsNotBytesOrGraphemes() {
        // The other four ports count code points too (`RuneCountInString`,
        // `chars().count`, `[...w].length`). A byte count would give Polish and
        // Japanese text different word weights in Swift than in Python for text
        // that reads identically, and `String.count` — grapheme clusters —
        // would give a third answer again.
        let words = Timing.estimateWords("aaaa żółć", start: 0.0, end: 1.0)
        XCTAssertEqual(words.count, 2)
        XCTAssertEqual(
            words[0].end - words[0].start, words[1].end - words[1].start, accuracy: 1e-12)
    }

    func testAChunkOfOneWordSpansTheWholeChunk() {
        let words = Timing.estimateWords("word", start: 0.5, end: 0.75)
        XCTAssertEqual(words.count, 1)
        XCTAssertEqual(words[0].start, 0.5)
        XCTAssertEqual(words[0].end, 0.75)
    }

    func testTheWordsOfAChunkStartAndEndWithIt() {
        // The estimate may be wrong about the interior; it may not be wrong
        // about the edges, because those are the exact tier.
        let span = Timing.timeline(
            [ChunkSpan(text: "one two three four", samples: 30_000, tokens: 9)],
            sampleRate: Self.sampleRate)[0]
        XCTAssertEqual(span.words.first?.start, span.start)
        XCTAssertEqual(span.words.last?.end, span.end)
    }
}
