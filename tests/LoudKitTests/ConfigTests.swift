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

    func testMaxNewTokensBelowOneIsRefused() throws {
        // The one range check the port did not mirror. `Engine` refuses a
        // *caller's* `maxNewTokens` below 1 and let the manifest's own through:
        // zero generated an empty row and handed it back as a result, and a
        // negative reached `0..<cap` in the token generator. The other four
        // ports refuse the pack at load.
        for bad in [0, -1] {
            XCTAssertThrowsError(
                try AlgorithmConfig.fromManifest(amended(sampling: ["max_new_tokens": bad])),
                "a manifest declaring max_new_tokens \(bad) loaded"
            ) { error in
                XCTAssertTrue(
                    "\(error)".contains("max_new_tokens must be positive"),
                    "refused for the wrong reason: \(error)")
            }
        }
        // One is a legal, if useless, configuration: the refusal is of values
        // that are not a count of tokens at all.
        let cfg = try AlgorithmConfig.fromManifest(amended(sampling: ["max_new_tokens": 1]))
        XCTAssertEqual(cfg.sampling.maxNewTokens, 1)
    }

    /// Zero is a configuration, not a typo: it disables the floor.
    func testZeroEOSFloorLoads() throws {
        let cfg = try AlgorithmConfig.fromManifest(
            amended(eos: ["min_tokens_floor": 0, "min_tokens_text_ratio": 0.0]))
        XCTAssertEqual(cfg.sampling.minTokensFloor, 0)
        XCTAssertEqual(cfg.sampling.minTokensTextRatio, 0.0)
    }
}

/// What this engine declares it can read, and why one number is not enough.
///
/// Both implemented decoder modes load; unknown formats and inconsistent
/// version/decode declarations are refused before any model computation.
final class CheckpointSupportTests: XCTestCase {
    func testSupportedFormatVersionsAreTheOnesImplemented() {
        XCTAssertEqual(Checkpoint.supportedFormatVersions, [1, 2])
    }

    func testSupportedDecodeModesAreTheOnesImplemented() {
        XCTAssertEqual(Checkpoint.supportedDecodeModes, ["single", "fusion_mtp2"])
    }

    /// A safetensors file carrying nothing but a manifest.
    ///
    /// Small enough to write in a test and complete enough for `Checkpoint`'s
    /// initialiser to reach every check it makes, which is the point:
    /// asserting on the two constants alone left both guards deletable with
    /// the tests still green.
    private func manifestOnly(_ manifest: String) throws -> URL {
        let header = try JSONSerialization.data(
            withJSONObject: ["__metadata__": ["manifest": manifest]])
        var bytes = withUnsafeBytes(of: UInt64(header.count).littleEndian) { Data($0) }
        bytes.append(header)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try bytes.write(to: url)
        return url
    }

    /// The loader, not the constant.
    ///
    func testTheLoaderRefusesWhatTheConstantsDeclare() throws {
        let cases: [(name: String, manifest: String, refusal: String?)] = [
            (
                "version 1 loads",
                #"{"format":"loudkit-checkpoint","format_version":1}"#,
                nil
            ),
            (
                "an explicit single is the same loop",
                #"{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"single"}}"#,
                nil
            ),
            (
                "version 2 paired decoder loads",
                #"{"format":"loudkit-checkpoint","format_version":2,"decode":{"mode":"fusion_mtp2"}}"#,
                nil
            ),
            (
                "fusion requires format version 2",
                #"{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"fusion_mtp2"}}"#,
                "requires format_version 2"
            ),
            (
                "a future format is refused",
                #"{"format":"loudkit-checkpoint","format_version":3}"#,
                "format_version 3"
            ),
            (
                "an unknown decoder is refused",
                #"{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"unknown"}}"#,
                "decode.mode"
            ),
        ]
        for (name, manifest, refusal) in cases {
            let url = try manifestOnly(manifest)
            defer { try? FileManager.default.removeItem(at: url) }
            if let refusal {
                XCTAssertThrowsError(try Checkpoint(url: url), name) { error in
                    XCTAssertTrue(
                        "\(error)".contains(refusal),
                        "\(name): the refusal does not name \(refusal): \(error)")
                }
            } else {
                XCTAssertNoThrow(try Checkpoint(url: url), name)
            }
        }
    }
}

final class FusionIdentityTests: XCTestCase {
    func testFingerprintMatchesIndependentFixture() throws {
        for file in ["vectors.json", "vectors_fusion_mtp2.json"] {
        let fixture = try Fixture.shared(file)
        let expected = fixture["algorithm"] as! [String: Any]
        let canonical = expected["canonical_form"] as! String
        let parsed = try JSONSerialization.jsonObject(with: Data(canonical.utf8)) as! [String: Any]
        let reference = parsed["algorithm"] as! [String: Any]
        let sampling = reference["sampling"] as! [String: Any]
        let postprocess = reference["postprocess"] as! [String: Any]
        var algorithm = AlgorithmConfig()
        algorithm.sampling.minTokensFloor = 10
        algorithm.sampling.minTokensTextRatio = 1.2
        algorithm.sampling.silenceTokenIds = asInts(sampling["silence_token_ids"])!
        algorithm.postprocess.quietRenderIds = asInts(postprocess["quiet_render_ids"])!
        algorithm.postprocess.silenceRenderIds = asInts(postprocess["silence_render_ids"])!
        algorithm.window.staticLength = 255
        algorithm.window.padTokenId = 4254
        algorithm.window.staticPromptTokens = 238
        algorithm.decode = file == "vectors.json" ? "single" : "fusion_mtp2"
        algorithm.eulerSteps = (reference["euler_steps"] as! NSNumber).intValue
        algorithm.guidanceRate = Double(reference["guidance_rate"] as! String)!
        XCTAssertEqual(algorithm.fingerprint(), expected["fingerprint"] as? String)
        XCTAssertEqual(algorithm.canonicalForm(), expected["canonical_form"] as? String)
        }
    }
}


final class EdgeFadeIdentityTests: XCTestCase {
    /// The 20 ms default is explicit in the fingerprint.
    func testTheDefaultRampIsExplicitInIdentity() throws {
        let base = AlgorithmConfig()
        var explicit = AlgorithmConfig()
        explicit.edgeFadeSeconds = 0.02
        XCTAssertEqual(explicit.canonicalForm(), base.canonicalForm())
        XCTAssertTrue(base.canonicalForm().contains("edge_fade_seconds"))
    }

    func testAnotherRampMovesTheFingerprintWhereEveryPortPutsIt() throws {
        let base = AlgorithmConfig()
        var other = AlgorithmConfig()
        other.edgeFadeSeconds = 0.008
        let want = base.canonicalForm().replacingOccurrences(
            of: "\"edge_fade_seconds\":\"0.02\"", with: "\"edge_fade_seconds\":\"0.008\"")
        XCTAssertEqual(other.canonicalForm(), want)
        XCTAssertNotEqual(other.fingerprint(), base.fingerprint())
    }

    func testTheManifestKeyIsRead() throws {
        let manifest: [String: Any] = [
            "window": [
                "max_speech_tokens": 255, "static_length": 255,
                "pad_token_id": 4254, "static_prompt_tokens": 238,
            ],
            "eos_floor": ["min_tokens_floor": 10, "min_tokens_text_ratio": 1.2],
            "sampling_defaults": [:],
            "edge_fade_seconds": 0.008,
        ]
        XCTAssertEqual(try AlgorithmConfig.fromManifest(manifest).edgeFadeSeconds, 0.008)
    }
}

/// Every range check Python runs in a dataclass initialiser, run here on the
/// manifest that carries the value.
///
/// A manifest one port refuses and another accepts is two renders under one
/// fingerprint. Three of these were traps rather than refusals: zero Euler
/// steps makes `timeGrid()` return `[nan]`, a negative count traps in
/// `0...eulerSteps`, `repetition_min_cycles: 0` divides by zero in
/// `repetitionCut`, and a NaN ceiling ratio traps in `Int(...)` in `ceiling`.
final class AlgorithmManifestRangeTests: XCTestCase {
    private func amended(
        _ extra: [String: Any] = [:], window: [String: Any] = [:],
        chunking: [String: Any]? = nil, postprocess: [String: Any]? = nil
    ) -> [String: Any] {
        var win: [String: Any] = [
            "max_speech_tokens": 255, "static_length": 255,
            "pad_token_id": 4254, "static_prompt_tokens": 238,
        ]
        for (k, v) in window { win[k] = v }
        var manifest: [String: Any] = [
            "window": win,
            "eos_floor": ["min_tokens_floor": 10, "min_tokens_text_ratio": 1.2],
            "sampling_defaults": [:],
        ]
        if let chunking { manifest["chunking"] = chunking }
        if let postprocess { manifest["postprocess"] = postprocess }
        for (k, v) in extra { manifest[k] = v }
        return manifest
    }

    private func refuses(_ manifest: [String: Any], _ fragment: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try AlgorithmConfig.fromManifest(manifest),
                             "the manifest loaded", file: file, line: line) { error in
            XCTAssertTrue("\(error)".contains(fragment),
                          "refused for the wrong reason: \(error)", file: file, line: line)
        }
    }

    func testEulerStepsBelowOneIsRefusedRatherThanTrapped() {
        refuses(amended(["n_cfm_timesteps": 0]), "euler_steps must be >= 1: 0")
        refuses(amended(["n_cfm_timesteps": -2]), "euler_steps must be >= 1: -2")
    }

    func testTheEulerGridIsReadAndItsShapeChecked() throws {
        let cfg = try AlgorithmConfig.fromManifest(
            amended(["n_cfm_timesteps": 2, "euler_grid": [0.0, 0.5, 1.0]]))
        XCTAssertEqual(cfg.eulerGrid, [0.0, 0.5, 1.0])
        XCTAssertEqual(cfg.timeGrid(), [0.0, 0.5, 1.0])
        refuses(amended(["euler_grid": [0.0, 1.0]]), "euler_grid has 2 points, expected 3")
        refuses(amended(["euler_grid": [0.0, 0.5, 0.5]]), "strictly increasing")
        refuses(amended(["euler_grid": [0.1, 0.5, 1.0]]), "must run from 0.0 to 1.0")
        refuses(amended(["euler_grid": "cosine"]), "must be a list of floats or null")
    }

    func testTheTokenRateIsReadAndRefusedWhenItIsNotARate() throws {
        XCTAssertEqual(try AlgorithmConfig.fromManifest(amended(["token_rate_hz": 12.5]))
                        .tokenRateHz, 12.5)
        refuses(amended(["token_rate_hz": 0.0]), "token_rate_hz must be > 0")
    }

    func testTheSpeechCodebookBoundsAreChecked() {
        refuses(amended(["speech_vocab_size": 0]), "speech_vocab_size must be >= 1")
        refuses(amended(["speech_vocab_size": 100]),
                "start_speech_token must be in [0, 100): 6561")
        refuses(amended(["speech_tokens": ["start": 6561, "stop": 6561]]),
                "must differ: both are 6561")
    }

    func testTheEdgeFadeBandIsChecked() {
        refuses(amended(["edge_fade_seconds": 0.0005]), "edge_fade_seconds must be in")
        refuses(amended(["edge_fade_seconds": 0.06]), "edge_fade_seconds must be in")
    }

    func testGuidanceAndItsRateHaveToAgree() {
        refuses(amended(["guidance_rate": 0.7]), "guidance_rate must be 0.0 in single_path")
        // cfg_dual_path is now refused at the manifest door, before the rate
        // pairing is looked at, because this port cannot run it whatever the
        // rate says. The pairing law is still the pairing law, and is checked
        // on its own below.
        refuses(amended(["guidance": "cfg_dual_path"]), "does not implement")
        refuses(amended(["guidance": "cfg_dual_path", "guidance_rate": 0.7]),
                "does not implement")
    }

    /// The rate pairing, reached the only way it still can be: a config built
    /// in code rather than read from a manifest.
    func testTheGuidanceRatePairingStillHolds() {
        var config = AlgorithmConfig()
        config.guidance = .cfgDualPath
        config.guidanceRate = 0.0
        XCTAssertThrowsError(try config.validate(), "a zero rate loaded") { error in
            XCTAssertTrue("\(error)".contains("twice the work for nothing"),
                          "refused for the wrong reason: \(error)")
        }
        config.guidance = .singlePath
        config.guidanceRate = 0.7
        XCTAssertThrowsError(try config.validate(), "a single-path rate loaded") { error in
            XCTAssertTrue("\(error)".contains("guidance_rate must be 0.0 in single_path"),
                          "refused for the wrong reason: \(error)")
        }
    }

    func testTheThreeTokenBudgetsHaveToAgree() {
        refuses(amended(chunking: ["max_tokens": 300]), "exceeds the render window (255)")
        refuses(amended(["sampling_defaults": ["max_new_tokens": 300]]),
                "sampling.max_new_tokens 300 exceeds the render window (255)")
    }

    func testTheSplitRecipeIsChecked() {
        refuses(amended(chunking: ["prefix_tokens": 255]), "prefix_tokens must be in [0, ")
        refuses(amended(chunking: ["split_on": [String]()]), "split_on cannot be empty")
        refuses(amended(chunking: ["max_tokens": 1]), "no character budget to split on")
    }

    /// A refusal the fingerprint cannot stand in for.
    ///
    /// `ChunkConfig` here carries no first-chunk cap, so the canonical form has
    /// no slot for one: a manifest that sets `first_chunk_max_tokens` and a
    /// manifest that omits it hash identically, while Python cuts the first
    /// chunk short and this splitter does not. Ignoring it is a different
    /// reading under one recipe version with nothing to report it. Go, Rust and
    /// JS refuse it by this name too.
    func testAnUnhonouredChunkingKeyIsRefused() {
        refuses(amended(chunking: ["first_chunk_max_tokens": 8]),
                "is not implemented by this port")
        // Null is still the key being set, and Python reads it as "no cap": the
        // refusal is about the key this port cannot honour, not about its value.
        refuses(amended(chunking: ["first_chunk_max_tokens": NSNull()]),
                "is not implemented by this port")
        // And the keys it does honour still load.
        XCTAssertEqual(
            try? AlgorithmConfig.fromManifest(amended(chunking: ["max_tokens": 128]))
                .chunking.maxTokens, 128)
    }

    func testTheFramingRecipeIsChecked() {
        refuses(amended(window: ["static_length": 128]),
                "static_length 128 cannot be shorter than max_speech_tokens 255")
        refuses(amended(window: ["static_prompt_tokens": 0]),
                "static_prompt_tokens must be positive")
    }

    /// `repetition_min_cycles: 0` divided by zero and a NaN ratio trapped in
    /// `Int(...)`; both now name the field.
    func testThePostprocessPresetIsChecked() {
        refuses(amended(postprocess: ["repetition_min_cycles": 0]),
                "repetition_min_cycles must be at least 2: 0")
        refuses(amended(postprocess: ["ceiling_speech_per_text_token": Double.nan]),
                "ceiling_speech_per_text_token must be a finite number")
        refuses(amended(postprocess: ["pacing_tolerance": Double.infinity]),
                "pacing_tolerance must be a finite number")
        refuses(amended(postprocess: ["stall_run_tokens": 0]),
                "stall_run_tokens must be positive")
        refuses(amended(postprocess: ["repetition_max_period": 0]),
                "repetition_max_period must be positive")
        refuses(amended(postprocess: ["repetition_min_span": 2]),
                "repetition_min_span (2) must be at least repetition_min_cycles (3)")
        refuses(amended(postprocess: ["retry_max_attempts": 8]),
                "retry_max_attempts must be in [0, 8)")
        refuses(amended(postprocess: ["retry_max_attempts": -1]),
                "retry_max_attempts must be in [0, 8)")
        // The counts the reference's own list omitted until it was measured: a
        // negative band floor closes the band the field exists to open, and a
        // negative dropout minimum leaves no row short enough to be one.
        refuses(amended(postprocess: ["desperation_band_floor": -1]),
                "desperation_band_floor must be >= 0: -1")
        refuses(amended(postprocess: ["dropout_min_tokens": -1]),
                "dropout_min_tokens must be >= 0: -1")
        refuses(amended(postprocess: ["ceiling_slack_tokens": -1]),
                "ceiling_slack_tokens must be >= 0: -1")
        refuses(amended(postprocess: ["echo_weak_max_tail": -1]),
                "echo_weak_max_tail must be >= 0: -1")
        refuses(amended(postprocess: ["filler_min_eos_probability": -0.1]),
                "filler_min_eos_probability out of range")
        refuses(amended(postprocess: ["echo_strong_min_position_pct": 101]),
                "echo_strong_min_position_pct")
        refuses(amended(postprocess: ["ceiling_speech_per_text_token": 0.0]),
                "ceiling_speech_per_text_token must be positive")
        refuses(amended(postprocess: ["desperation_speech_per_text_token": 3.0]),
                "must exceed ceiling_speech_per_text_token")
        refuses(amended(postprocess: ["desperation_min_keep_per_text_token": 3.0]),
                "must not exceed desperation_band_ratio")
        refuses(amended(postprocess: ["trailing_filler_threshold": 1.5]),
                "trailing_filler_threshold must be in (0, 1]")
        refuses(amended(postprocess: ["filler_min_eos_probability": 1.0]),
                "filler_min_eos_probability out of range")
        refuses(amended(postprocess: ["ended_tail_keep": -1]), "ended_tail_keep must be >= 0")
        refuses(amended(postprocess: ["echo_weak_min_position_pct": 101]),
                "echo_weak_min_position_pct is a percentage: 101")
    }

    /// The shipped preset passes its own validator, so none of the above can
    /// start refusing a released pack.
    func testTheShippingManifestStillLoads() throws {
        let cfg = try AlgorithmConfig.fromManifest(amended())
        XCTAssertEqual(cfg.eulerSteps, 2)
        XCTAssertEqual(cfg.tokenRateHz, 25.0)
        XCTAssertNil(cfg.eulerGrid)
        XCTAssertNoThrow(try cfg.validate())
        XCTAssertNoThrow(try PostprocessConfig().validate())
    }
}

/// `LoudKitError.code` against the frozen catalog in `docs/reference/errors.md`.
///
/// Pinned because the codes are the vocabulary a transport sends and a caller in
/// any language branches on, so they are frozen the way a wire value is: a code
/// that changes silently is a caller that starts mishandling the failure it was
/// written for. Python asserts the same strings on its own classes.
final class ErrorCodeTests: XCTestCase {
    func testTheCodesAreTheCatalogsCodes() {
        XCTAssertEqual(LoudKitError.cancelled.code, "cancelled")
        XCTAssertEqual(LoudKitError.windowOverflow(tokens: 300, window: 255).code,
                       "window_overflow")
        // A message that names no condition falls back the way
        // `loudkit.errors.error_code` does, so a failure neither side can name
        // classifies the same on both rather than differently.
        XCTAssertEqual(LoudKitError.manifest("x").code, "invalid_request")
        XCTAssertEqual(LoudKitError.asset("x").code, "invalid_request")
        XCTAssertEqual(LoudKitError.shape("x").code, "invalid_request")
        XCTAssertEqual(LoudKitError.prediction("x").code, "invalid_request")
    }

    /// The four conditions the raising sites name, read back off the sentence
    /// each site writes.
    ///
    /// Through the factories the sites raise, not through a hand-built message:
    /// a site that stopped using its factory would write a sentence this
    /// classifier no longer recognises, and the code would silently fall back
    /// to `invalid_request` with nothing to show for it.
    func testTheNamedConditionsCarryTheirOwnCode() {
        XCTAssertEqual(
            LoudKitError.voiceNotFound(name: "nobody", directory: "/r", shipped: ["ana"]).code,
            "voice_not_found")
        XCTAssertEqual(
            LoudKitError.voiceNotFound(name: "nobody", directory: "/r", shipped: []).code,
            "voice_not_found")
        XCTAssertEqual(
            LoudKitError.languageUnsupported("bg", why: "is not on the roster",
                                             roster: ["en", "pl"]).code,
            "unsupported_language")
        XCTAssertEqual(
            LoudKitError.invalidToken(field: "tokens", token: 99_999, limit: 6561).code,
            "invalid_tokens")
        XCTAssertEqual(LoudKitError.noTokensToRender.code, "invalid_tokens")
    }

    /// Reading the condition out of the message moved no message.
    ///
    /// The bytes below are what the four sites produced before `code` could
    /// tell them apart, character for character: the factories exist so the
    /// sentence is written once, not so it could be reworded.
    func testTheMessagesAreUnchanged() {
        XCTAssertEqual(LoudKitError.asset("gone").description, "asset: gone")
        XCTAssertEqual(LoudKitError.shape("wrong").description, "shape: wrong")
        XCTAssertEqual(LoudKitError.manifest("bad").description, "manifest: bad")
        XCTAssertEqual(LoudKitError.prediction("failed").description, "prediction: failed")
        XCTAssertEqual(LoudKitError.cancelled.description,
                       "cancelled before the audio was rendered")
        XCTAssertEqual(
            LoudKitError.voiceNotFound(name: "nobody", directory: "/r", shipped: ["ana", "bo"])
                .description,
            "asset: nobody: no such voice in /r. This release ships ana, bo.")
        XCTAssertEqual(
            LoudKitError.voiceNotFound(name: "nobody", directory: "/r", shipped: []).description,
            "asset: nobody: no such voice in /r. This release ships none.")
        XCTAssertEqual(
            LoudKitError.languageUnsupported(
                "bg", why: "is not one of the languages this build's text layer is written for",
                roster: ["en", "pl"]).description,
            "asset: language bg is not one of the languages this build's text layer "
                + "is written for. Supported: en, pl")
        XCTAssertEqual(
            LoudKitError.invalidToken(field: "tokens", token: 99_999, limit: 6561).description,
            "shape: tokens contains 99999, which is not an acoustic speech token "
                + "(expected 0 <= id < 6561). Pass `Result.tokens` from an earlier call; "
                + "the generator's own control tokens are already stripped from it.")
        XCTAssertEqual(LoudKitError.noTokensToRender.description,
                       "shape: tokens is empty: there is nothing to render.")
    }

    /// The sites still raise those exact errors.
    ///
    /// The factory and the classifier agreeing proves nothing on its own if the
    /// engine no longer goes through the factory, so this drives the real
    /// entry points.
    func testTheRaisingSitesStillCarryTheCodes() {
        XCTAssertThrowsError(
            try Engine.validateSpeechTokens([1, 99_999], limit: 6561, field: "tokens")
        ) { error in
            XCTAssertEqual((error as? LoudKitError)?.code, "invalid_tokens")
        }
        XCTAssertThrowsError(
            try Engine.carryFrom([-1], prefixTokens: 2, startSpeechToken: 6561)
        ) { error in
            XCTAssertEqual((error as? LoudKitError)?.code, "invalid_tokens")
        }
        // `stripSpecials` refuses a negative id with its own range clause and
        // the shared sentence, so it carries the code too.
        XCTAssertEqual(
            LoudKitError.shape("tokens contains -1\(LoudKitError.notASpeechToken) "
                               + "(expected 0 to 6560)").code,
            "invalid_tokens")
    }
}
