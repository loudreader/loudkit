import XCTest

@testable import LoudKit

/// The postprocess layer, against the shared conformance fixture.
///
/// Every case in `tests/data/conformance/postprocess.json` is a regression from
/// the shipped reader or a named device trace, and every port runs the same
/// file. A rule that drifts in one language fails in one language.
final class PostprocessConformanceTests: XCTestCase {

    private func fixture() throws -> [String: Any] {
        let url = Fixture.conformanceDir.appendingPathComponent("postprocess.json")
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("postprocess.json is not an object")
        }
        return json
    }

    private func cases(_ fx: [String: Any], _ key: String) throws -> [[String: Any]] {
        guard let list = fx[key] as? [[String: Any]], !list.isEmpty else {
            XCTFail("fixture section \(key) is missing or empty; nothing was compared")
            throw XCTSkip("no cases")
        }
        return list
    }

    /// The fixture's token-shape builder, spelled out in its header.
    private func build(_ shape: [[Any]]) -> [Int] {
        var out: [Int] = []
        for segment in shape {
            guard let kind = segment[0] as? String,
                  let count = (segment[1] as? NSNumber)?.intValue else { continue }
            if kind == "cycle" {
                // `count` is the period here; the third element the repeats.
                let repeats = (segment[2] as? NSNumber)?.intValue ?? 0
                let cycle = (0..<count).map { 20 + $0 % 60 }
                for _ in 0..<repeats { out.append(contentsOf: cycle) }
                continue
            }
            if kind == "cycle_mixed" {
                // Second half silence: the word-then-pause stutter.
                let repeats = (segment[2] as? NSNumber)?.intValue ?? 0
                let half = count / 2
                let cycle = (0..<(count - half)).map { 20 + $0 } + (0..<half).map { $0 % 8 }
                for _ in 0..<repeats { out.append(contentsOf: cycle) }
                continue
            }
            for i in 0..<count {
                out.append(kind == "speech" ? 20 + i % 60 : i % 8)
            }
        }
        return out
    }

    private func silence(_ fx: [String: Any]) -> Set<Int> {
        Set((fx["silence_token_ids"] as? [NSNumber] ?? []).map(\.intValue))
    }

    /// Build the detector config out of the fixture, so the numbers these tests
    /// run on are the ones the fixture declares rather than this port's own
    /// defaults — which is the whole point of a shared file.
    private func config(_ fx: [String: Any], mode: String? = nil) throws -> PostprocessConfig {
        guard let c = fx["config"] as? [String: Any] else {
            throw XCTSkip("fixture has no config block")
        }
        func d(_ key: String) -> Double { (c[key] as! NSNumber).doubleValue }
        func i(_ key: String) -> Int { (c[key] as! NSNumber).intValue }

        var cfg = PostprocessConfig()
        cfg.mode = Postprocess.Mode(rawValue: mode ?? (c["mode"] as! String))!
        cfg.ceilingSpeechPerTextToken = d("ceiling_speech_per_text_token")
        cfg.ceilingSlackTokens = i("ceiling_slack_tokens")
        cfg.trailingFillerThreshold = d("trailing_filler_threshold")
        cfg.trailingSilenceRunTokens = i("trailing_silence_run_tokens")
        cfg.fillerMinEosProbability = d("filler_min_eos_probability")
        cfg.fillerMaxSpeechAfterRun = i("filler_max_speech_after_run")
        cfg.desperationSpeechPerTextToken = d("desperation_speech_per_text_token")
        cfg.desperationMinTextTokens = i("desperation_min_text_tokens")
        cfg.endedTailSilenceRun = i("ended_tail_silence_run")
        cfg.endedTailBlipMax = i("ended_tail_blip_max")
        cfg.endedTailWordMax = i("ended_tail_word_max")
        cfg.endedTailKeep = i("ended_tail_keep")
        cfg.echoStrongEosProbability = d("echo_strong_eos_probability")
        cfg.echoStrongMaxTail = i("echo_strong_max_tail")
        cfg.echoStrongMinPositionPct = i("echo_strong_min_position_pct")
        cfg.echoWeakEosProbability = d("echo_weak_eos_probability")
        cfg.echoWeakMaxTail = i("echo_weak_max_tail")
        cfg.echoWeakMinPositionPct = i("echo_weak_min_position_pct")
        cfg.repetitionMaxPeriod = i("repetition_max_period")
        cfg.repetitionMinCycles = i("repetition_min_cycles")
        cfg.repetitionMinSpan = i("repetition_min_span")
        cfg.dropoutMinTokens = i("dropout_min_tokens")
        cfg.retryMaxAttempts = i("retry_max_attempts")
        cfg.pacingTolerance = d("pacing_tolerance")
        return cfg
    }

    /// The fixture's nullable `expect`, as this port's optional.
    private func want(_ raw: Any?) -> Int? { (raw as? NSNumber)?.intValue }

    /// The shipping constants are the fixture's, or the cases below prove
    /// nothing about what actually runs.
    func testShippingDefaultsMatchTheFixture() throws {
        let fx = try fixture()
        XCTAssertEqual(
            PostprocessConfig(), try config(fx),
            "PostprocessConfig() has drifted from the conformance fixture")
    }

    func testCeiling() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        for kase in try cases(fx, "ceiling") {
            let got = Postprocess.ceiling(
                forTextTokens: (kase["text_tokens"] as! NSNumber).intValue,
                config: cfg,
                window: (kase["window"] as! NSNumber).intValue)
            XCTAssertEqual(got, (kase["expect"] as! NSNumber).intValue,
                           "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    func testTrailingFiller() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        let sil = silence(fx)
        for kase in try cases(fx, "trailing_filler") {
            let got = Postprocess.isTrailingFiller(
                build(kase["shape"] as! [[Any]]),
                from: (kase["from"] as! NSNumber).intValue,
                silence: sil, config: cfg)
            XCTAssertEqual(got, kase["expect"] as! Bool, "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    func testDesperation() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        let sil = silence(fx)
        for kase in try cases(fx, "desperation") {
            let got = Postprocess.desperationCut(
                build(kase["shape"] as! [[Any]]),
                textTokenCount: (kase["text_tokens"] as! NSNumber).intValue,
                minTokens: (kase["min_tokens"] as! NSNumber).intValue,
                eosPeakAt: (kase["eos_peak_at"] as! NSNumber).intValue,
                silence: sil, config: cfg,
                peakAllowed: kase["peak_allowed"] as! Bool)
            XCTAssertEqual(got, want(kase["expect"]), "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    func testEndedTail() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        let sil = silence(fx)
        for kase in try cases(fx, "ended_tail") {
            let got = Postprocess.endedTailTrim(
                build(kase["shape"] as! [[Any]]),
                silence: sil, config: cfg,
                isTerminal: kase["is_terminal"] as! Bool)
            XCTAssertEqual(got, want(kase["expect"]), "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    func testTerminalEcho() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        for kase in try cases(fx, "terminal_echo") {
            let got = Postprocess.terminalEchoCut(
                tokenCount: (kase["token_count"] as! NSNumber).intValue,
                eosPeakAt: (kase["eos_peak_at"] as! NSNumber).intValue,
                eosPeakProb: (kase["eos_peak_prob"] as! NSNumber).doubleValue,
                minTokens: (kase["min_tokens"] as! NSNumber).intValue,
                isTerminal: kase["is_terminal"] as! Bool,
                hitCeiling: kase["hit_ceiling"] as! Bool,
                config: cfg)
            XCTAssertEqual(got, want(kase["expect"]), "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    /// The precedence, which is the part a caller cannot get right by itself.
    func testResolve() throws {
        let fx = try fixture()
        let sil = silence(fx)
        for kase in try cases(fx, "resolve") {
            let cfg = try config(fx, mode: kase["mode"] as? String)
            let request = Postprocess.Request(
                textTokenCount: (kase["text_tokens"] as! NSNumber).intValue,
                minTokens: (kase["min_tokens"] as! NSNumber).intValue,
                eosPeakAt: (kase["eos_peak_at"] as! NSNumber).intValue,
                eosPeakProb: (kase["eos_peak_prob"] as! NSNumber).doubleValue,
                ended: kase["ended"] as! Bool,
                isTerminal: kase["is_terminal"] as! Bool,
                hitCeiling: kase["hit_ceiling"] as! Bool)
            let got = Postprocess.inspect(
                build(kase["shape"] as! [[Any]]),
                request: request, silence: sil, config: cfg)
            let expect = kase["expect"] as! [String: Any]
            let why = "\(kase["name"]!): \(kase["why"]!)"
            XCTAssertEqual(got.keep, (expect["keep"] as! NSNumber).intValue, why)
            XCTAssertEqual(got.reason.rawValue, expect["reason"] as! String, why)
            XCTAssertEqual(got.suspect, expect["suspect"] as! Bool, why)
        }
    }

    /// Long-form drift, report-only, in the same integer-derived domain.
    func testPacingMatchesTheFixture() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        guard let section = fx["pacing"] as? [String: Any],
              let cases = section["cases"] as? [[String: Any]], !cases.isEmpty else {
            XCTFail("the fixture has no pacing cases")
            return
        }
        for kase in cases {
            let ratios = (kase["ratios"] as! [NSNumber]).map(\.doubleValue)
            let want = (kase["expect"] as! [NSNumber]).map(\.intValue)
            XCTAssertEqual(Postprocess.pacingOutliers(ratios, config: cfg), want,
                           "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    /// Early truncation — the failure a listener cannot hear.
    ///
    /// Every other rule says the end of the row is wrong. This one says the row
    /// is incomplete, which is why it reports rather than cuts.
    func testDropoutMatchesTheFixture() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        guard let section = fx["dropout"] as? [String: Any],
              let cases = section["cases"] as? [[String: Any]], !cases.isEmpty else {
            XCTFail("the fixture has no dropout cases; nothing was compared")
            return
        }
        for kase in cases {
            let got = Postprocess.isDropout(
                (kase["tokens"] as! NSNumber).intValue,
                (kase["text_tokens"] as! NSNumber).intValue, config: cfg)
            XCTAssertEqual(got, kase["expect"] as! Bool, "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    /// The loop the tail rules cannot see, because it happens mid-row.
    ///
    /// Every other rule reads the end of the chunk. A stuck decoder repeats
    /// inside it, and the literature puts that failure first or second in every
    /// ranking of what goes wrong with autoregressive speech models.
    func testRepetitionMatchesTheFixture() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        let sil = silence(fx)
        guard let cases = fx["repetition"] as? [[String: Any]], !cases.isEmpty else {
            XCTFail("the fixture has no repetition cases; nothing was compared")
            return
        }
        var negatives = 0
        for kase in cases {
            let expect = want(kase["expect"])
            if expect == nil { negatives += 1 }
            let got = Postprocess.repetitionCut(
                build(kase["shape"] as! [[Any]]), silence: sil, config: cfg)
            XCTAssertEqual(got, expect, "\(kase["name"]!): \(kase["why"]!)")
        }
        // A mid-sequence cut is the most destructive thing this layer can do,
        // so the cases that must NOT fire carry more weight than those that must.
        XCTAssertGreaterThanOrEqual(negatives, 6, "too few negative cases for a mid-row cut")
    }
}

/// The manifest side: which recipe a checkpoint is running, and why.
final class PostprocessManifestTests: XCTestCase {
    private func amended(_ extra: [String: Any]) -> [String: Any] {
        // Swift refuses an un-amended pack, so every case here carries the
        // window and EOS blocks it requires.
        var manifest: [String: Any] = [
            "window": [
                "max_speech_tokens": 255, "static_length": 255,
                "pad_token_id": 4254, "static_prompt_tokens": 238,
            ],
            "eos_floor": ["min_tokens_floor": 10, "min_tokens_text_ratio": 1.2],
        ]
        for (k, v) in extra { manifest[k] = v }
        return manifest
    }

    /// The detectors default on when the block is absent; the tag does not
    /// move for it: there is one recipe, and a manifest that omits a block
    /// left a shipping default unstated.
    func testAbsentBlockDefaultsTheDetectorsOn() throws {
        let cfg = try AlgorithmConfig.fromManifest(
            amended(["recipe_version": "loudkit-1", "chunking": [String: Any]()]))
        XCTAssertEqual(cfg.recipeVersion, "loudkit-1")
        XCTAssertEqual(cfg.postprocess.mode, .trim)
    }

    func testDeclaredBlockIsRead() throws {
        let cfg = try AlgorithmConfig.fromManifest(
            amended([
                "recipe_version": "loudkit-1",
                "chunking": [String: Any](),
                "postprocess": ["mode": "report", "trailing_silence_run_tokens": 13],
            ]))
        XCTAssertEqual(cfg.recipeVersion, "loudkit-1")
        XCTAssertEqual(cfg.postprocess.mode, .report)
        XCTAssertEqual(cfg.postprocess.trailingSilenceRunTokens, 13)
    }

    /// One recipe means one accepted value, and the error names what the
    /// manifest declared. Believing a foreign tag would fingerprint it;
    /// defaulting it would claim this recipe for a checkpoint that named
    /// another. All five ports refuse it identically.
    func testAForeignRecipeVersionIsRefusedByName() {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(["recipe_version": "loudkit-9"]))
        ) { error in
            XCTAssertTrue("\(error)".contains("loudkit-9"), "error must name the tag: \(error)")
        }
        // Not even a string: refused, not defaulted. A manifest one port
        // misreads while another defaults is the divergence this library
        // exists to prevent.
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(["recipe_version": 9])))
    }

    func testUnknownModeIsRefused() {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(amended(["postprocess": ["mode": "shave"]])))
    }
}

/// The stop-token observation the postprocess layer reads.
///
/// Pinned across languages because it is hand-written in five of them and it is
/// *audible*: two of the detector rules compare it against a threshold, so a
/// port that computes it differently cuts a chunk somewhere else. The quantity
/// has two subtleties either of which a reimplementation gets wrong silently —
/// the numerator is the stop token's weight taken BEFORE the min_p cutoff, and
/// the peak is recorded only PAST the floor.
final class EOSPeakConformanceTests: XCTestCase {
    func testMatchesTheSharedFixture() throws {
        let fixture = try Fixture.vectors()
        guard let section = fixture["eos_peak"] as? [String: Any],
              let cases = section["cases"] as? [[String: Any]], !cases.isEmpty
        else {
            XCTFail("the fixture has no eos_peak cases; nothing was compared")
            return
        }
        let rtol = (section["prob_rtol"] as! NSNumber).doubleValue

        for kase in cases {
            let cfgMap = kase["config"] as! [String: Any]
            var sampling = SamplingConfig()
            sampling.temperature = (cfgMap["temperature"] as! NSNumber).doubleValue
            sampling.repetitionPenalty = (cfgMap["repetition_penalty"] as! NSNumber).doubleValue
            sampling.minP = (cfgMap["min_p"] as! NSNumber).doubleValue
            sampling.silenceTokenIds = (cfgMap["silence_token_ids"] as! [NSNumber]).map(\.intValue)

            let sampler = LRSamplerV1(
                config: sampling, seed: (kase["seed"] as! NSNumber).uint64Value)
            sampler.observeEOS(
                stopToken: (kase["stop_token"] as! NSNumber).intValue,
                floor: (kase["eos_floor"] as! NSNumber).intValue)

            let r = kase["logits_recipe"] as! [String: Any]
            let vocab = (r["vocab"] as! NSNumber).intValue
            let scale = (r["scale"] as! NSNumber).doubleValue
            let offset = (r["offset"] as! NSNumber).doubleValue
            var seen = [Bool](repeating: false, count: vocab)
            for step in 0..<(r["steps"] as! NSNumber).intValue {
                let u = Philox.uniforms(
                    seed: (r["seed"] as! NSNumber).uint64Value,
                    stream: (r["stream"] as! NSNumber).uint32Value,
                    step0: UInt32(step), nSteps: 1, width: vocab)
                let row = (0..<vocab).map { Float(u[$0] * scale + offset) }
                seen[sampler.sample(logits: row, step: step, seen: seen)] = true
            }
            let peak = sampler.eosPeak
            let wantProb = (kase["expected_prob"] as! NSNumber).doubleValue
            XCTAssertEqual(peak.at, (kase["expected_at"] as! NSNumber).intValue,
                           kase["name"] as! String)
            XCTAssertLessThanOrEqual(
                abs(peak.probability - wantProb), rtol * abs(wantProb),
                "\(kase["name"]!): peak prob \(peak.probability), want \(wantProb)")
        }
    }
}

/// The ceiling was settled on English traces; nine languages ship.
///
/// Speech tokens per *text* token is a property of the orthography, so a
/// constant tuned on one language is an assumption everywhere else — and the
/// expensive direction of that assumption is a guard that truncates correct
/// speech in a language nobody measured. Measured with one voice held constant
/// across nine language tags, because the voice-to-voice spread on a single
/// sentence is larger than the language-to-language spread.
final class LanguageGuardConformanceTests: XCTestCase {
    func testMatchesTheFixture() throws {
        let url = Fixture.conformanceDir.appendingPathComponent("postprocess.json")
        let fx = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        guard let section = fx["language_guard"] as? [String: Any],
              let cases = section["cases"] as? [[String: Any]], !cases.isEmpty else {
            XCTFail("the fixture has no language_guard cases; nothing was compared")
            return
        }
        let c = fx["config"] as! [String: Any]
        var cfg = PostprocessConfig()
        cfg.ceilingSpeechPerTextToken =
            (c["ceiling_speech_per_text_token"] as! NSNumber).doubleValue
        cfg.ceilingSlackTokens = (c["ceiling_slack_tokens"] as! NSNumber).intValue

        var stopped: [String] = []
        for kase in cases {
            let name = kase["name"] as! String
            let ceiling = Postprocess.ceiling(
                forTextTokens: (kase["text_tokens"] as! NSNumber).intValue,
                config: cfg,
                window: (kase["window"] as! NSNumber).intValue)
            XCTAssertEqual(ceiling, (kase["expect"] as! NSNumber).intValue,
                           "\(name): \(kase["why"]!)")
            let hit = (kase["measured_speech_tokens"] as! NSNumber).intValue >= ceiling
            XCTAssertEqual(hit, kase["expect_stopped_by_ceiling"] as! Bool,
                           "\(name) changed side of the ceiling: \(kase["why"]!)")
            if hit { stopped.append(name) }
        }
        // One row belongs here and it is not a false positive: a Spanish
        // three-word phrase whose decoder never emitted a stop token. The guard
        // caught a runaway; it did not cut a legitimate read.
        XCTAssertEqual(stopped, ["es_short"],
                       "a new entry is a language being truncated by an English-tuned constant")
    }
}
