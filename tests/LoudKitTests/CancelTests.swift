import XCTest

@testable import LoudKit

/// The cancellation contract every port shares: cancelled at a decode step
/// inside the second chunk, `synthesize` throws `LoudKitError.cancelled` and
/// returns nothing, and `stream` delivers the first chunk exactly as an
/// uncancelled run does and nothing after, without throwing. Needs the packed
/// checkpoint and the CoreML packages; skips without them.
final class CancelTests: XCTestCase {
    // Comfortably past one window, so the second chunk exists to be cancelled in.
    private let text =
        "The first sentence sets the scene and runs on for a while. "
        + "The second sentence follows it and is no shorter than the first one was. "
        + "The third sentence exists so that the splitter has somewhere to breathe. "
        + "The fourth sentence closes the passage without hurrying."

    /// A `shouldCancel` that counts its polls and fires past `at`.
    private final class Counter {
        var polls = 0
        let at: Int
        init(at: Int) { self.at = at }
        func poll() -> Bool {
            polls += 1
            return polls > at
        }
    }

    func testCancelAtADecodeStep() throws {
        try Fixture.requireCheckpoint()
        let engine = try Engine.load(
            checkpoint: Fixture.checkpointURL, coremlAssets: Fixture.coremlAssetsURL)
        try checkCancelAtADecodeStep(engine)
    }

    func testFusionCancelAtADecodeStep() throws {
        try checkCancelAtADecodeStep(EndToEndConformanceTests.loadFusionEngine())
    }

    private func checkCancelAtADecodeStep(_ engine: Engine) throws {
        let voice = try LongFormTests.referenceVoice()

        // Uncancelled first, counting the polls: the step to cancel at has to
        // land inside the second chunk's decode loop.
        let counting = Counter(at: Int.max)
        var pollsAtChunk: [Int] = []
        var first: [Int] = []
        try engine.stream(text, voice: voice, seed: 7, language: "en", shouldCancel: counting.poll) {
            chunk in
            pollsAtChunk.append(counting.polls)
            if first.isEmpty { first = chunk.tokens }
            return true
        }
        XCTAssertGreaterThanOrEqual(pollsAtChunk.count, 2, "the passage must split")
        let step = pollsAtChunk[0] + 5
        XCTAssertLessThan(step, pollsAtChunk[1], "step \(step) is not inside chunk 1")

        // stream: the first chunk as it was, nothing after, no throw.
        var got: [[Int]] = []
        try engine.stream(
            text, voice: voice, seed: 7, language: "en", shouldCancel: Counter(at: step).poll
        ) { chunk in
            got.append(chunk.tokens)
            return true
        }
        XCTAssertEqual(got.count, 1, "only the chunk before the cancel arrives")
        XCTAssertEqual(got.first, first, "the chunk before the cancel is what an uncancelled run delivered")

        // synthesize: nothing returned, .cancelled.
        XCTAssertThrowsError(
            try engine.synthesize(
                text, voice: voice, seed: 7, language: "en", shouldCancel: Counter(at: step).poll)
        ) { error in
            guard case LoudKitError.cancelled = error else {
                return XCTFail("synthesize threw \(error), want LoudKitError.cancelled")
            }
        }

        // synthesizeWindow, three steps in: the same signal.
        XCTAssertThrowsError(
            try engine.synthesizeWindow(
                "Hello from loudkit.", voice: voice, seed: 7, language: "en",
                shouldCancel: Counter(at: 3).poll)
        ) { error in
            guard case LoudKitError.cancelled = error else {
                return XCTFail("synthesizeWindow threw \(error), want LoudKitError.cancelled")
            }
        }
    }

    /// The site that fires: `TokenGenerator.generate` throws at the poll that
    /// returned true and nothing polls after it. A generator that `break`s
    /// instead returns a partial row here, and the engine's later poll would
    /// hide that behind the same `.cancelled`.
    func testFusionCancellationAndRepeatability() throws {
        let engine = try EndToEndConformanceTests.loadFusionEngine()
        let voice = try LongFormTests.referenceVoice()
        let ids = try engine.frontend.encode("Hello from loudkit.", language: "en")
        func generate() throws -> [Int] {
            try engine.tokenGenerator.generate(textTokens: ids, voice: voice,
                sampler: LRSamplerV1(config: engine.algorithm.sampling, seed: 7)).rawTokens
        }
        XCTAssertEqual(try generate(), try generate())
        let counter = Counter(at: 3)
        XCTAssertThrowsError(try engine.tokenGenerator.generate(textTokens: ids, voice: voice,
            sampler: LRSamplerV1(config: engine.algorithm.sampling, seed: 7), shouldCancel: counter.poll)) { error in
            guard case LoudKitError.cancelled = error else { return XCTFail("\(error)") }
        }
        XCTAssertEqual(counter.polls, 4)
    }

    func testTheGeneratorThrowsAtThePollThatFires() throws {
        try Fixture.requireCheckpoint()
        let engine = try Engine.load(
            checkpoint: Fixture.checkpointURL, coremlAssets: Fixture.coremlAssetsURL)
        let voice = try VoiceProfile.load(
            url: Fixture.conformanceDir.appendingPathComponent(
                "../reference/testvoice.voice.safetensors"))
        let ids = try engine.frontend.encode("Hello from loudkit.", language: "en")
        let counter = Counter(at: 3)
        XCTAssertThrowsError(
            try engine.tokenGenerator.generate(
                textTokens: ids, voice: voice,
                sampler: LRSamplerV1(config: engine.algorithm.sampling, seed: 7),
                shouldCancel: counter.poll)
        ) { error in
            guard case LoudKitError.cancelled = error else {
                return XCTFail("generate threw \(error), want LoudKitError.cancelled")
            }
        }
        XCTAssertEqual(counter.polls, 4, "the poll that fired is the last one made")
    }
}
