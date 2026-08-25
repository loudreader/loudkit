import XCTest

@testable import LoudKit

/// A generator-only stopwatch, off by default (`LOUDKIT_BENCH=1` runs it).
///
/// The renderer is CoreML and measures 11.84x real time; the generator is
/// native Swift and is the half that decides the end-to-end figure, so it is
/// timed on its own — no mel, no vocoder, nothing to average the number with.
/// Fixed text, fixed voice, fixed seed and a fixed cap, so two runs differ
/// only in wall time.
final class GeneratorBenchTests: XCTestCase {
    func testDecodeThroughput() throws {
        guard ProcessInfo.processInfo.environment["LOUDKIT_BENCH"] == "1" else {
            throw XCTSkip("set LOUDKIT_BENCH=1 to time the generator")
        }
        try Fixture.requireCheckpoint()
        let checkpoint = try Checkpoint(url: Fixture.checkpointURL)
        let algorithm = try checkpoint.algorithm()
        let generator = try TokenGenerator(checkpoint: checkpoint, config: algorithm)
        let frontend = try TextFrontend(tokenizerURL: checkpoint.tokenizerURL)
        let voice = try VoiceProfile.load(
            url: Fixture.conformanceDir.appendingPathComponent(
                "../reference/testvoice.voice.safetensors"))
        let text = "The quick brown fox jumps over the lazy dog, and then it turned "
            + "around and did the whole thing again, twice, for no reason at all."
        let textTokens = try frontend.encode(text, language: "en")

        // One untimed pass: the first decode pays for page faults on a 26 MB
        // weight file that the timed run should not be charged for.
        _ = generator.generate(
            textTokens: textTokens, voice: voice,
            sampler: LRSamplerV1(config: algorithm.sampling, seed: 4242), maxNewTokens: 8)

        let cap = 200
        let t0 = Date()
        let gen = generator.generate(
            textTokens: textTokens, voice: voice,
            sampler: LRSamplerV1(config: algorithm.sampling, seed: 4242), maxNewTokens: cap)
        let dt = -t0.timeIntervalSinceNow
        let n = gen.rawTokens.count
        print(String(format: "BENCH generator: %d tokens in %.3f s = %.1f tok/s (%.2f ms/token)",
                     n, dt, Double(n) / dt, dt * 1000 / Double(n)))
        print("BENCH checksum: \(gen.rawTokens.prefix(12))")
        XCTAssertGreaterThan(n, 0)
    }
}
