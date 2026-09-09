import XCTest

@testable import LoudKit

/// Opt-in warm generator timing, excluding renderer and checkpoint load.
final class GeneratorBenchTests: XCTestCase {
    func testDecodeThroughput() throws {
        guard ProcessInfo.processInfo.environment["LOUDKIT_BENCH"] == "1" else {
            throw XCTSkip("set LOUDKIT_BENCH=1 to time the generator")
        }
        try Fixture.requireCheckpoint()
        try Fixture.requireFusionCheckpoint()
        try benchmark(Fixture.checkpointURL)
        try benchmark(Fixture.fusionCheckpointURL)
    }

    private func benchmark(_ url: URL) throws {
        let checkpoint = try Checkpoint(url: url)
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
        _ = try generator.generate(
            textTokens: textTokens, voice: voice,
            sampler: LRSamplerV1(config: algorithm.sampling, seed: 4242), maxNewTokens: 8)

        let cap = 200
        for prefix in [[], [10, 20, 30, 40, 50, 60]] {
            let t0 = Date()
            let gen = try generator.generate(
                textTokens: textTokens, voice: voice,
                sampler: LRSamplerV1(config: algorithm.sampling, seed: 4242), maxNewTokens: cap, prefix: prefix)
            let dt = -t0.timeIntervalSinceNow
            let n = gen.rawTokens.count
            print("BENCH mode=\(algorithm.decode) prefix=\(prefix.count)")
            print(String(format: "BENCH generator: %d tokens in %.3f s = %.1f tok/s (%.2f ms/token)",
                         n, dt, Double(n) / dt, dt * 1000 / Double(n)))
            print("BENCH checksum: \(gen.rawTokens.prefix(12))")
            XCTAssertGreaterThan(n, 0)
        }
    }
}
