import XCTest

@testable import LoudKit

/// Sampling values a manifest can carry, and the ones Python's
/// `SamplingConfig.__post_init__` refuses.
///
/// A manifest one port refuses and another accepts is two renders under one
/// fingerprint, and every one of these failure modes is silent: temperature 0
/// divides by zero, min_p 1 empties the candidate set, a negative EOS floor
/// lets a chunk stop on its first token.
final class SamplingManifestTests: XCTestCase {
    /// Swift refuses an un-amended pack, so every case carries the window and
    /// EOS blocks it requires.
    private func amended(
        sampling: [String: Any] = [:], eos: [String: Any] = [:]
    ) -> [String: Any] {
        var floor: [String: Any] = ["min_tokens_floor": 10, "min_tokens_text_ratio": 1.2]
        for (k, v) in eos { floor[k] = v }
        return [
            "window": [
                "max_speech_tokens": 255, "static_length": 255,
                "pad_token_id": 4254, "static_prompt_tokens": 238,
            ],
            "eos_floor": floor,
            "sampling_defaults": sampling,
        ]
    }

    func testTemperatureIsRefused() {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(sampling: ["temperature": 0.0])))
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(sampling: ["temperature": 4.5])))
    }

    func testRepetitionPenaltyBelowOneIsRefused() {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(sampling: ["repetition_penalty": 0.9])))
    }

    func testMinPOutOfRangeIsRefused() {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(sampling: ["min_p": -0.1])))
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(sampling: ["min_p": 1.0])))
    }

    func testNegativeEOSFloorIsRefused() {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(eos: ["min_tokens_floor": -1])))
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(eos: ["min_tokens_text_ratio": -0.5])))
    }

    /// Zero is a configuration, not a typo: it disables the floor.
    func testZeroEOSFloorLoads() throws {
        let cfg = try AlgorithmConfig.fromManifest(
            amended(eos: ["min_tokens_floor": 0, "min_tokens_text_ratio": 0.0]))
        XCTAssertEqual(cfg.sampling.minTokensFloor, 0)
        XCTAssertEqual(cfg.sampling.minTokensTextRatio, 0.0)
    }
}
