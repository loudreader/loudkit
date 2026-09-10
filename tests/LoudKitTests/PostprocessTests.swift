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
            XCTFail("postprocess.json is not an object; nothing can be compared")
            throw XCTSkip("unreadable fixture")
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
            if kind == "cycle_dead" {
                // The stutter with its pause on a census-only id.
                let repeats = (segment[2] as? NSNumber)?.intValue ?? 0
                let half = count / 2
                let cycle = (0..<(count - half)).map { 20 + $0 }
                    + [Int](repeating: 12, count: half)
                for _ in 0..<repeats { out.append(contentsOf: cycle) }
                continue
            }
            let value: (Int) -> Int
            switch kind {
            case "speech": value = { 20 + $0 % 60 }
            case "quiet": value = { $0 % 8 }
            // True digital silence in the stall section's two-class scheme.
            case "sil": value = { $0 % 4 }
            // The contextually-quiet family: extends a dead-air run without
            // counting toward its gate.
            case "breath": value = { 4 + $0 % 4 }
            // True silence only the render census knows (the 6405 class):
            // outside the fixture's sampler list, inside its silence census.
            case "dead": value = { _ in 12 }
            // Breath only the quiet census knows.
            case "sigh": value = { _ in 13 }
            default:
                // Refused loudly: an unknown kind silently built as something
                // else is a fixture section this port only appears to run.
                XCTFail("unknown segment kind \(kind)")
                continue
            }
            for i in 0..<count { out.append(value(i)) }
        }
        return out
    }

    private func silence(_ fx: [String: Any]) -> Set<Int> {
        Set((fx["silence_token_ids"] as? [NSNumber] ?? []).map(\.intValue))
    }

    /// Build the detector config out of the fixture, so the numbers these tests
    /// run on are the ones the fixture declares rather than this port's own
    /// defaults, which is the whole point of a shared file.
    private func config(_ fx: [String: Any], mode: String? = nil) throws -> PostprocessConfig {
        guard let c = fx["config"] as? [String: Any] else {
            // Without this block every case below would run on this port's own
            // defaults, which is the one thing a shared fixture exists to stop.
            XCTFail("postprocess.json has no config block; the port would test its own defaults")
            throw XCTSkip("no config block")
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

    /// `config(_:)` plus the stall section's render-id censuses. The fallback
    /// arm (`renderIds` false) runs without them, a checkpoint packed before
    /// the census, where only the run trigger fires.
    private func stallConfig(_ fx: [String: Any], renderIds: Bool) throws -> PostprocessConfig {
        var cfg = try config(fx)
        if renderIds {
            let section = fx["stall"] as! [String: Any]
            cfg.silenceRenderIds =
                (section["silence_render_ids"] as! [NSNumber]).map(\.intValue)
            cfg.quietRenderIds = (section["quiet_render_ids"] as! [NSNumber]).map(\.intValue)
        }
        return cfg
    }

    /// `config(_:)` plus the repetition_silence section's render-id censuses,
    /// the ids only the censuses know, which the loop exemption must union
    /// in under the acoustic family.
    private func repSilenceConfig(_ fx: [String: Any]) throws -> PostprocessConfig {
        var cfg = try config(fx)
        let section = fx["repetition_silence"] as! [String: Any]
        cfg.silenceRenderIds = (section["silence_render_ids"] as! [NSNumber]).map(\.intValue)
        cfg.quietRenderIds = (section["quiet_render_ids"] as! [NSNumber]).map(\.intValue)
        return cfg
    }

    /// A fixture section shaped `{"cases": [...]}` rather than a bare list.
    private func sectionCases(_ fx: [String: Any], _ key: String) throws -> [[String: Any]] {
        guard let section = fx[key] as? [String: Any],
              let list = section["cases"] as? [[String: Any]], !list.isEmpty else {
            XCTFail("fixture section \(key) is missing or empty; nothing was compared")
            throw XCTSkip("no cases")
        }
        return list
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

    /// Early truncation, the failure a listener cannot hear.
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

    /// The loop exemption keys on acoustic silence, not the sampler list.
    ///
    /// The specimen: kathleen/en0023 seed 1234 parked a mid-chunk pause on
    /// ids 6486 (x7) then 6405 (x24), both render true silence, both in the
    /// manifest's `silence_render_ids`, neither in the sampler's list the
    /// exemption used to read. The period-1 run fired as a loop and the cut
    /// deleted the pause plus two whole sentences of correctly-read speech
    /// behind it, verdict repetition, not suspect, audibly fluent. Six of the
    /// checkpoint's eight truly-silent ids sit outside the sampler list, so
    /// the exemption was blind on most real pauses. The law now unions the
    /// sampler list with both render censuses for this one rule; the tail
    /// rules keep the list they were calibrated against.
    func testRepetitionSilenceMatchesTheFixture() throws {
        let fx = try fixture()
        let sil = silence(fx)
        let cfg = try repSilenceConfig(fx)
        for kase in try sectionCases(fx, "repetition_silence") {
            let got = Postprocess.repetitionCut(
                build(kase["shape"] as! [[Any]]), silence: sil, config: cfg)
            XCTAssertEqual(got, want(kase["loop"]), "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    /// The cascade is part of the contract: a declined loop falls through to
    /// stall, which condemns the specimen's pause into the retry ladder
    /// instead of shipping the cut.
    func testRepetitionSilenceResolverMatchesTheFixture() throws {
        let fx = try fixture()
        let sil = silence(fx)
        let cfg = try repSilenceConfig(fx)
        for kase in try sectionCases(fx, "repetition_silence") {
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

    func testSamplingNamesTheOldLaw() throws {
        // The pre-amendment behaviour stays nameable, a checkpoint measured
        // under it can declare what it measured, and this is what it did:
        // cut at the pause and delete everything behind it.
        let fx = try fixture()
        var cfg = try repSilenceConfig(fx)
        cfg.repetitionSilence = .sampling
        let kase = try sectionCases(fx, "repetition_silence")[0]
        let got = Postprocess.repetitionCut(
            build(kase["shape"] as! [[Any]]), silence: silence(fx), config: cfg)
        XCTAssertEqual(got, 31, "the old law cut one token past the pause's start")
    }

    func testWithoutCensusesTheUnionIsTheSamplerList() throws {
        // A checkpoint packed before the censuses changes nothing: nothing on
        // such a build knows id 12 is silent, so the run still reads as a
        // loop there, under either field value.
        let fx = try fixture()
        let kase = try sectionCases(fx, "repetition_silence")[0]
        let got = Postprocess.repetitionCut(
            build(kase["shape"] as! [[Any]]), silence: silence(fx), config: try config(fx))
        XCTAssertEqual(got, 31)
    }

    private func resumeRequest(_ kase: [String: Any]) -> Postprocess.Request {
        Postprocess.Request(
            textTokenCount: (kase["text_tokens"] as! NSNumber).intValue,
            minTokens: (kase["min_tokens"] as! NSNumber).intValue,
            eosPeakAt: (kase["eos_peak_at"] as! NSNumber).intValue,
            eosPeakProb: (kase["eos_peak_prob"] as! NSNumber).doubleValue,
            ended: kase["ended"] as! Bool,
            isTerminal: kase["is_terminal"] as! Bool,
            hitCeiling: kase["hit_ceiling"] as! Bool)
    }

    /// A loop the decoder resumed from is condemned, never cut.
    ///
    /// The census fix (`repetitionSilence`) needs a manifest that names the
    /// silent ids, and the published pack has none: on it the en0023 pause
    /// fired again as a period-1 loop and the cut kept 52 of 206 tokens,
    /// deleting two sentences of correctly-read speech, verdict repetition,
    /// no retry. The guard here needs no silence knowledge at all: a genuine
    /// lock-up runs its cycle to the end of the row (a ceiling truncates at
    /// most one incomplete copy, `period - 1` tokens), so a qualifying loop
    /// followed by a full period or more of other content is a decoder that
    /// resumed, and a decoder that resumed was never locked. Such a row is
    /// handed back whole, suspect, into the retry ladder. The section
    /// configures no censuses on purpose: it is the arm `repetitionSilence`
    /// cannot reach.
    func testRepetitionResumeMatchesTheFixture() throws {
        // The bare rule still reports the loop; the law lives in the resolver.
        let fx = try fixture()
        let sil = silence(fx)
        let cfg = try config(fx)
        for kase in try sectionCases(fx, "repetition_resume") {
            let got = Postprocess.repetitionCut(
                build(kase["shape"] as! [[Any]]), silence: sil, config: cfg)
            XCTAssertEqual(got, want(kase["loop"]), "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    func testRepetitionResumeResolverMatchesTheFixture() throws {
        let fx = try fixture()
        let sil = silence(fx)
        let cfg = try config(fx)
        for kase in try sectionCases(fx, "repetition_resume") {
            let got = Postprocess.inspect(
                build(kase["shape"] as! [[Any]]),
                request: resumeRequest(kase), silence: sil, config: cfg)
            let expect = kase["expect"] as! [String: Any]
            let why = "\(kase["name"]!): \(kase["why"]!)"
            XCTAssertEqual(got.keep, (expect["keep"] as! NSNumber).intValue, why)
            XCTAssertEqual(got.reason.rawValue, expect["reason"] as! String, why)
            XCTAssertEqual(got.suspect, expect["suspect"] as! Bool, why)
        }
    }

    func testCutNamesTheOldLaw() throws {
        // The pre-amendment behaviour stays nameable, a checkpoint measured
        // under it can declare what it measured, and this is what it did:
        // cut at the pause and delete everything behind it.
        let fx = try fixture()
        var cfg = try config(fx)
        cfg.repetitionResume = .cut
        let kase = try sectionCases(fx, "repetition_resume")[0]
        let got = Postprocess.inspect(
            build(kase["shape"] as! [[Any]]),
            request: resumeRequest(kase), silence: silence(fx), config: cfg)
        XCTAssertEqual(got.reason, .repetition)
        XCTAssertEqual(got.keep, want(kase["loop"]), "the old law shipped the specimen's cut")
        XCTAssertFalse(got.suspect)
    }

    func testTheCensusArmIsUntouched() throws {
        // With the censuses configured the specimen's pause is exempt from
        // the loop rule entirely and stall condemns it, the
        // repetition_silence contract, byte for byte, guard or no guard.
        let fx = try fixture()
        let cfg = try repSilenceConfig(fx)
        let kase = try sectionCases(fx, "repetition_silence")[0]
        let got = Postprocess.inspect(
            build(kase["shape"] as! [[Any]]),
            request: resumeRequest(kase), silence: silence(fx), config: cfg)
        let expect = kase["expect"] as! [String: Any]
        XCTAssertEqual(got.reason, .stall)
        XCTAssertEqual(got.keep, (expect["keep"] as! NSNumber).intValue)
    }

    /// The decoder trapped in silence, the failure no tail rule can see.
    ///
    /// Every mute chunk and every mid-row hole in the interior-stall study
    /// shipped as clean, because all six other rules anchor on the tail. The
    /// stall rule condemns instead of cutting: the failure is a hole, and the
    /// fix is the retry ladder. Detection is two-class (a true-silence gate, a
    /// quiet-family continuation), which the fixture pins because single-set
    /// counting was measured broken.
    func testStallMatchesTheFixture() throws {
        let fx = try fixture()
        let sil = silence(fx)
        for kase in try sectionCases(fx, "stall") {
            let cfg = try stallConfig(fx, renderIds: kase["render_ids"] as! Bool)
            let got = Postprocess.isStalled(
                build(kase["shape"] as! [[Any]]),
                hitCeiling: kase["hit_ceiling"] as! Bool,
                silence: sil, config: cfg)
            XCTAssertEqual(got, kase["stalled"] as! Bool, "\(kase["name"]!): \(kase["why"]!)")
        }
    }

    /// The wiring is part of the contract: after repetition, before every tail
    /// rescue, condemned like dropout.
    func testStallResolverMatchesTheFixture() throws {
        let fx = try fixture()
        let sil = silence(fx)
        for kase in try sectionCases(fx, "stall") {
            let cfg = try stallConfig(fx, renderIds: kase["render_ids"] as! Bool)
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

    func testStallNeverCuts() throws {
        let fx = try fixture()
        let cfg = try stallConfig(fx, renderIds: true)
        let row = build([["speech", 30], ["sil", 30], ["speech", 30]])
        let got = Postprocess.inspect(
            row,
            request: Postprocess.Request(
                textTokenCount: 40, minTokens: 48, eosPeakAt: -1, eosPeakProb: 0.0,
                ended: true, isTerminal: true, hitCeiling: false),
            silence: silence(fx), config: cfg)
        XCTAssertEqual(got.reason, .stall)
        XCTAssertEqual(
            got.keep, row.count,
            "a stalled row must be handed back whole; the hole is mid-row and no "
            + "cut can remove it")
        XCTAssertTrue(got.suspect, "the caller has to be told, since nothing was changed")
    }

    /// A row that both loops and stalls answers to the loop: an exactly
    /// repeated cycle pins where the failure began. Here the decoder resumed
    /// after the region, so the loop condemns rather than cuts, but it still
    /// outranks the stall's condemnation, and the verdict names the anchor
    /// that was found.
    func testRepetitionOutranksStall() throws {
        let fx = try fixture()
        let cfg = try stallConfig(fx, renderIds: true)
        let row = build([["cycle", 4, 8], ["sil", 30], ["speech", 20]])
        let got = Postprocess.inspect(
            row,
            request: Postprocess.Request(
                textTokenCount: 40, minTokens: 48, eosPeakAt: -1, eosPeakProb: 0.0,
                ended: true, isTerminal: true, hitCeiling: false),
            silence: silence(fx), config: cfg)
        XCTAssertEqual(got.reason, .repetition, "the exact anchor outranks the condemnation")
    }

    /// A cap-hit desperation cut that keeps less than any full read.
    ///
    /// The one row that survived the stall fix: soren/da0028 chunk 4 burned
    /// 132 tokens to the ceiling and the seam cut kept 36, 1.44 s in which 33
    /// of the 36 kept tokens render near-silent through ids outside both
    /// manifest censuses, invisible to every set-membership rule. The keep's
    /// *length* is the only evidence there is: a keep under
    /// `desperationMinKeepPerTextToken` per text token cannot hold a full
    /// read, so the verdict is condemned into the retry ladder. The cut stands
    /// as the keep, an exhausted ladder ships the trim, flagged, rather than
    /// the untrimmed babble. No censuses configured here on purpose: the
    /// trigger is a length test and must fire identically on a checkpoint
    /// packed before them.
    func testStarvedRescueMatchesTheFixture() throws {
        let fx = try fixture()
        let cfg = try config(fx)
        let sil = silence(fx)
        for kase in try sectionCases(fx, "starved_rescue") {
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

    func testStarvedRescueFloorIsExclusive() throws {
        // keep == floor ships: `<`, not `<=`, so the pinned law has no
        // ambiguity at the boundary for a port to resolve differently. Text 20
        // puts the floor at exactly 34.0.
        let fx = try fixture()
        let cfg = try config(fx)
        let row = build([["speech", 34], ["sil", 12], ["speech", 90]])
        let got = Postprocess.inspect(
            row,
            request: Postprocess.Request(
                textTokenCount: 20, minTokens: 24, eosPeakAt: -1, eosPeakProb: 0.0,
                ended: false, isTerminal: true, hitCeiling: true),
            silence: silence(fx), config: cfg)
        XCTAssertEqual(got.reason, .desperation)
        XCTAssertEqual(got.keep, 34)
        XCTAssertFalse(got.suspect, "a keep exactly at the floor is not under it")
    }

    func testStarvedRescueZeroDisablesTheTrigger() throws {
        let fx = try fixture()
        var cfg = try config(fx)
        cfg.desperationMinKeepPerTextToken = 0.0
        let kase = try sectionCases(fx, "starved_rescue")[0]
        let got = Postprocess.inspect(
            build(kase["shape"] as! [[Any]]),
            request: Postprocess.Request(
                textTokenCount: (kase["text_tokens"] as! NSNumber).intValue,
                minTokens: (kase["min_tokens"] as! NSNumber).intValue,
                eosPeakAt: (kase["eos_peak_at"] as! NSNumber).intValue,
                eosPeakProb: (kase["eos_peak_prob"] as! NSNumber).doubleValue,
                ended: kase["ended"] as! Bool,
                isTerminal: kase["is_terminal"] as! Bool,
                hitCeiling: kase["hit_ceiling"] as! Bool),
            silence: silence(fx), config: cfg)
        XCTAssertEqual(got.reason, .desperation)
        XCTAssertFalse(got.suspect, "zero must disable the trigger")
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

    /// A law this port does not implement must be refused, not defaulted:
    /// the resolver would cut where the manifest said to condemn. `cut`
    /// names the pre-amendment law and is read.
    func testUnknownRepetitionResumeIsRefused() throws {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(
                amended(["postprocess": ["repetition_resume": "maybe"]])))
        let cfg = try AlgorithmConfig.fromManifest(
            amended(["postprocess": ["repetition_resume": "cut"]]))
        XCTAssertEqual(cfg.postprocess.repetitionResume, .cut)
    }

    /// A family this port does not implement must be refused, not defaulted:
    /// the loop exemption would read one silence list under a manifest
    /// declaring another. `sampling` names the pre-amendment law and is read.
    func testUnknownRepetitionSilenceIsRefused() throws {
        XCTAssertThrowsError(
            try AlgorithmConfig.fromManifest(
                amended(["postprocess": ["repetition_silence": "both"]])))
        let cfg = try AlgorithmConfig.fromManifest(
            amended(["postprocess": ["repetition_silence": "sampling"]]))
        XCTAssertEqual(cfg.postprocess.repetitionSilence, .sampling)
    }

    /// The render censuses ride the manifest top level, beside
    /// `silence_token_ids`; declaring them inside the postprocess block would
    /// give one value two homes in one file, so it is refused by name, the
    /// same way Python's block reader refuses it.
    func testRenderIdsInsideTheBlockAreRefused() {
        for key in ["silence_render_ids", "quiet_render_ids"] {
            XCTAssertThrowsError(
                try AlgorithmConfig.fromManifest(
                    amended(["postprocess": [key: [1, 2, 3]]]))
            ) { error in
                XCTAssertTrue("\(error)".contains(key), "error must name the key: \(error)")
            }
        }
    }

    /// The top-level censuses reach the detectors, with or without a
    /// postprocess block, a manifest with no detector overrides still
    /// carries the properties of its weights.
    func testTopLevelCensusesAreRead() throws {
        let cfg = try AlgorithmConfig.fromManifest(
            amended(["silence_render_ids": [7, 8], "quiet_render_ids": [9]]))
        XCTAssertEqual(cfg.postprocess.silenceRenderIds, [7, 8])
        XCTAssertEqual(cfg.postprocess.quietRenderIds, [9])

        let withBlock = try AlgorithmConfig.fromManifest(
            amended([
                "silence_render_ids": [7, 8],
                "postprocess": ["stall_run_tokens": 30],
            ]))
        XCTAssertEqual(withBlock.postprocess.silenceRenderIds, [7, 8])
        XCTAssertEqual(withBlock.postprocess.stallRunTokens, 30)
    }
}

/// The stop-token observation the postprocess layer reads.
///
/// Pinned across languages because it is hand-written in five of them and it is
/// *audible*: two of the detector rules compare it against a threshold, so a
/// port that computes it differently cuts a chunk somewhere else. The quantity
/// has two subtleties either of which a reimplementation gets wrong silently,
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
/// constant tuned on one language is an assumption everywhere else, and the
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
