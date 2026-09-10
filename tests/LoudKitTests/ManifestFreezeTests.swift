import Foundation
import XCTest

@testable import LoudKit

/// The shipped 0.1.1 manifest, read to the shipped fingerprint, through the
/// reader rather than around it.
///
/// The other fingerprint checks in this package build the configuration by hand
/// and compare the canonical form, which pins the *writer*: it holds even if
/// `fromManifest` ignores every key and answers its defaults, and for most of
/// this manifest the defaults are the shipped values. The one check that does
/// go manifest to fingerprint needs the 1.27 GB checkpoint and skips without
/// it, so on a machine with no weights nothing pinned the reader at all.
///
/// `go/config/freeze_test.go` and `rust/tests/freeze.rs` are the same test in
/// the other two ports, over the same manifest.
final class ManifestFreezeTests: XCTestCase {
    /// `7cd75498ad4e7531` is the algorithm this build implements, recorded in
    /// CHANGELOG.md and in the conformance vectors. A published voice records
    /// the fingerprint it was rendered under, which is a fact about that
    /// artefact and does not move with this one.
    static let shippedFingerprint = "7cd75498ad4e7531"

    /// The manifest 0.1.1 ships, as `tools/amend_manifest.py` writes it over
    /// the values the packed checkpoint already carried.
    ///
    /// Written out rather than assembled from the shipping constants, because
    /// those are what the reader is being checked against: a manifest built
    /// from them would agree with a reader that ignored every key and answered
    /// its defaults, which is the exact failure this pins.
    static func shipped011() -> [String: Any] {
        [
            "format": "loudkit-checkpoint",
            "format_version": 1,
            "edge_fade_seconds": 0.02,
            "guidance": "single_path",
            "guidance_rate": 0.0,
            "recipe_version": "loudkit-1",
            "postprocess": [
                "mode": "trim",
                "ceiling_speech_per_text_token": 4.0,
                "ceiling_slack_tokens": 40,
                "trailing_filler_threshold": 0.7,
                "trailing_silence_run_tokens": 12,
                "filler_min_eos_probability": 0.05,
                "filler_max_speech_after_run": 10,
                "desperation_speech_per_text_token": 4.5,
                "desperation_min_text_tokens": 10,
                "ended_tail_silence_run": 6,
                "ended_tail_blip_max": 2,
                "ended_tail_word_max": 10,
                "ended_tail_keep": 5,
                "echo_strong_eos_probability": 0.1,
                "echo_strong_max_tail": 30,
                "echo_strong_min_position_pct": 68,
                "echo_weak_eos_probability": 0.003,
                "echo_weak_max_tail": 16,
                "echo_weak_min_position_pct": 85,
            ] as [String: Any],
            "window": [
                "max_speech_tokens": 255,
                "static_length": 255,
                "pad_token_id": 4254,
                "static_prompt_tokens": 238,
            ],
            "eos_floor": ["min_tokens_floor": 10, "min_tokens_text_ratio": 1.2] as [String: Any],
            "chunking": [
                "enabled": true,
                "max_tokens": 255,
                "prefix_tokens": 6,
                "split_on": [". ", "! ", "? ", "; ", ", "],
                "abbreviations": [
                    "A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M",
                    "Mr", "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs",
                    "prof", "\u{15b}w",
                ],
                "mid_sentence_period": "hold",
            ] as [String: Any],
            "sample_rate": 24_000,
            "token_rate_hz": 25.0,
            "speech_vocab_size": 8194,
            "n_cfm_timesteps": 2,
            "speech_tokens": ["start": 6561, "stop": 6562],
            "sampling_defaults": [
                "temperature": 0.8,
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "max_new_tokens": 255,
            ] as [String: Any],
            "silence_token_ids": [
                1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927,
                3928, 3930, 4008, 4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173,
                4174, 4218, 4245, 4251, 4252, 4254, 4255, 4260, 4282,
            ],
            "silence_render_ids": [4137, 4215, 4218, 4299, 6162, 6324, 6405, 6486],
            "quiet_render_ids": [
                1458, 1461, 1488, 1701, 1704, 1707, 1716, 1731, 1785, 1788, 1869,
                1947, 1950, 1951, 1959, 1978, 2028, 2031, 2040, 2058, 2076, 2112,
                2139, 3645, 3648, 3651, 3704, 3888, 3894, 4188, 5838, 6081, 6183,
                6537,
            ],
        ]
    }

    /// The manifest, round-tripped through `JSONSerialization`.
    ///
    /// A dictionary literal hands the reader Swift's own `Int`, `Double` and
    /// `Bool`; a checkpoint hands it `__NSCFNumber` and `__NSCFBoolean`, which
    /// bridge differently and are the whole reason the reader needs a rule for
    /// telling a number from a flag. Reading the literal alone would test a
    /// shape no user ever supplies.
    static func decoded(_ manifest: [String: Any]) throws -> [String: Any] {
        let data = try JSONSerialization.data(withJSONObject: manifest)
        return try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testTheShippedManifestStillReadsToTheShippedFingerprint() throws {
        let config = try AlgorithmConfig.fromManifest(Self.decoded(Self.shipped011()))

        // The blob first: a mismatch there names the field that drifted, while
        // a mismatch in the hash alone says only that something did. The
        // fixture is written by Python, so this is the byte-for-byte comparison
        // against the reference rather than against this port's own writer.
        let vectors = try Fixture.vectors()
        let algorithm = try XCTUnwrap(vectors["algorithm"] as? [String: Any])
        XCTAssertEqual(
            config.canonicalForm(), algorithm["canonical_form"] as? String,
            "the shipped manifest no longer reads to the canonical form Python writes")
        XCTAssertEqual(config.fingerprint(), Self.shippedFingerprint)

        // Named separately because it is the block whose sub-keys are easiest
        // to drop silently: the window recipe is the whole measured deviation
        // between two backends' renders, and a dropped static_length frames it
        // ragged.
        XCTAssertEqual(config.window.maxSpeechTokens, 255)
        XCTAssertEqual(config.window.staticLength, 255)
        XCTAssertEqual(config.window.staticPromptTokens, 238)
        XCTAssertEqual(config.window.padTokenId, 4254)
    }

    /// The manifest is read, not recognised.
    ///
    /// Without this, a reader that answered its defaults for everything would
    /// pass the test above whenever the defaults happened to be the shipped
    /// values.
    func testAMovedKeyMovesTheFingerprint() throws {
        var manifest = Self.shipped011()
        var window = try XCTUnwrap(manifest["window"] as? [String: Any])
        window["pad_token_id"] = 4253
        manifest["window"] = window
        let config = try AlgorithmConfig.fromManifest(Self.decoded(manifest))
        XCTAssertEqual(config.window.padTokenId, 4253)
        XCTAssertNotEqual(config.fingerprint(), Self.shippedFingerprint)
    }

    /// A value with no reading here is refused by the key's name, not replaced
    /// by the constant next to it.
    ///
    /// `"255"` is the case the reference coerces and this port refuses, for the
    /// reason `ManifestReader.asNumber` gives; the rest have no reading
    /// anywhere. Read as the default, each frames some part of the algorithm
    /// the manifest never declared, and the canonical form then records the
    /// fallback as if the manifest had asked for it.
    func testAnUnreadableSubKeyIsRefusedByName() throws {
        let cases: [(String, String, Any)] = [
            ("window", "static_length", "255"),
            ("window", "static_length", "abc"),
            ("window", "static_length", true),
            ("window", "static_length", [255]),
            ("window", "max_speech_tokens", NSNull()),
            ("sampling_defaults", "temperature", "0.8"),
            ("sampling_defaults", "max_new_tokens", "255"),
            ("eos_floor", "min_tokens_floor", "10"),
            ("eos_floor", "min_tokens_text_ratio", true),
            ("speech_tokens", "start", "6561"),
            ("chunking", "max_tokens", "255"),
            ("chunking", "prefix_tokens", "6"),
            ("chunking", "enabled", 0),
            ("chunking", "mid_sentence_period", 5),
            ("chunking", "cap_resplit", 5),
            ("postprocess", "stall_run_tokens", "30"),
            ("postprocess", "echo_weak_max_tail", true),
            ("postprocess", "mode", 5),
            ("postprocess", "repetition_resume", 5),
            ("postprocess", "repetition_silence", 5),
        ]
        for (blockName, key, bad) in cases {
            var manifest = Self.shipped011()
            var block = try XCTUnwrap(manifest[blockName] as? [String: Any])
            block[key] = bad
            manifest[blockName] = block
            XCTAssertThrowsError(
                try AlgorithmConfig.fromManifest(Self.decoded(manifest)),
                "\(blockName).\(key) = \(bad) must be refused"
            ) { error in
                XCTAssertTrue(
                    "\(error)".contains(key), "the refusal must name the key: \(error)")
            }
        }
    }

    /// The same rule at the manifest's top level, including the two censuses
    /// and the silence list.
    ///
    /// A census read short is the measured one: the list used to become empty
    /// on a single element of the wrong type, and an empty silence set is a
    /// stall detector that never fires while the canonical form records the
    /// empty set as if the manifest had declared it.
    func testAnUnreadableTopLevelKeyIsRefusedByName() throws {
        let cases: [(String, Any)] = [
            ("sample_rate", "24000"),
            ("sample_rate", true),
            ("token_rate_hz", true),
            ("n_cfm_timesteps", "2"),
            ("guidance_rate", "0.0"),
            ("edge_fade_seconds", "0.02"),
            ("speech_vocab_size", "8194"),
            ("guidance", 5),
            ("silence_token_ids", ["1731"]),
            ("silence_token_ids", "1731"),
            ("silence_render_ids", ["4137"]),
            ("quiet_render_ids", [true]),
            ("euler_grid", ["0.5"]),
            ("window", 255),
            ("chunking", 255),
            ("postprocess", 255),
            ("sampling_defaults", [1, 2]),
            ("speech_tokens", "6561"),
        ]
        for (key, bad) in cases {
            var manifest = Self.shipped011()
            manifest[key] = bad
            XCTAssertThrowsError(
                try AlgorithmConfig.fromManifest(Self.decoded(manifest)),
                "\(key) = \(bad) must be refused"
            ) { error in
                XCTAssertTrue(
                    "\(error)".contains(key), "the refusal must name the key: \(error)")
            }
        }
    }

    /// The manifest's own spellings of "unset" still load.
    ///
    /// Only the three window lengths have a null reading, and it is the ragged
    /// one; `edge_fade_seconds: null` is the 5 ms the pre-key releases shipped.
    /// `max_speech_tokens` has no null reading at all, and is covered above.
    func testTheNullReadingsTheReferenceHas() throws {
        var manifest = Self.shipped011()
        var window = try XCTUnwrap(manifest["window"] as? [String: Any])
        window["static_length"] = NSNull()
        manifest["window"] = window
        manifest["edge_fade_seconds"] = NSNull()
        manifest["euler_grid"] = NSNull()
        let config = try AlgorithmConfig.fromManifest(Self.decoded(manifest))
        XCTAssertNil(config.window.staticLength)
        XCTAssertEqual(config.window.maxSpeechTokens, 255)
        XCTAssertEqual(config.edgeFadeSeconds, 0.005)
        XCTAssertNil(config.eulerGrid)
    }

    /// A zero a manifest declares out loud is kept, not read as absence.
    func testAnExplicitZeroIsKept() throws {
        var manifest = Self.shipped011()
        var defaults = try XCTUnwrap(manifest["sampling_defaults"] as? [String: Any])
        defaults["min_p"] = 0
        manifest["sampling_defaults"] = defaults
        var eos = try XCTUnwrap(manifest["eos_floor"] as? [String: Any])
        eos["min_tokens_floor"] = 0
        manifest["eos_floor"] = eos
        let config = try AlgorithmConfig.fromManifest(Self.decoded(manifest))
        XCTAssertEqual(config.sampling.minP, 0)
        XCTAssertEqual(config.sampling.minTokensFloor, 0)
    }

    /// A whole number written as a float still reads as that number.
    ///
    /// `max_tokens: 255.0` is what a manifest round-tripped through a language
    /// with one number type carries, and it used to fall through the `as? Int`
    /// cast to the shipping default. Only the fraction is refused; the number
    /// written twice is the number.
    func testAWholeNumberWrittenAsAFloatIsRead() throws {
        var manifest = Self.shipped011()
        var chunking = try XCTUnwrap(manifest["chunking"] as? [String: Any])
        chunking["max_tokens"] = 200.0
        chunking["prefix_tokens"] = 4.0
        manifest["chunking"] = chunking
        let config = try AlgorithmConfig.fromManifest(Self.decoded(manifest))
        XCTAssertEqual(config.chunking.maxTokens, 200)
        XCTAssertEqual(config.chunking.prefixTokens, 4)
    }

    /// A manifest key that counts things takes a whole number.
    ///
    /// Truncating 2.7 to 2 turns a packer's arithmetic mistake into a chunker
    /// that breathes in a different place, a window framed to a different
    /// length or a census naming a token nobody wrote, under a
    /// `recipe_version` saying the five implementations agree. The sentence is
    /// `manifest._int`'s, word for word, so the same bad file reads the same
    /// in both.
    func testAFractionalCountIsRefused() throws {
        let cases: [([String: Any], String)] = [
            (["chunking": ["max_tokens": 200.5]],
             "manifest['chunking']['max_tokens'] must be a whole number, got 200.5"),
            (["chunking": ["prefix_tokens": 2.7]],
             "manifest['chunking']['prefix_tokens'] must be a whole number, got 2.7"),
            (["window": ["max_speech_tokens": 255.5]],
             "manifest['window']['max_speech_tokens'] must be a whole number, got 255.5"),
            (["window": ["static_length": 255.5]],
             "manifest['window']['static_length'] must be a whole number, got 255.5"),
            (["window": ["pad_token_id": 2.7]],
             "manifest['window']['pad_token_id'] must be a whole number, got 2.7"),
            (["window": ["static_prompt_tokens": 238.5]],
             "manifest['window']['static_prompt_tokens'] must be a whole number, got 238.5"),
            (["speech_tokens": ["start": 6561.5]],
             "manifest['speech_tokens']['start'] must be a whole number, got 6561.5"),
            (["speech_tokens": ["stop": 6562.5]],
             "manifest['speech_tokens']['stop'] must be a whole number, got 6562.5"),
            (["sampling_defaults": ["max_new_tokens": 255.5]],
             "manifest['sampling_defaults']['max_new_tokens'] must be a whole number, got 255.5"),
            (["eos_floor": ["min_tokens_floor": 10.5]],
             "manifest['eos_floor']['min_tokens_floor'] must be a whole number, got 10.5"),
            (["n_cfm_timesteps": 2.7],
             "manifest['n_cfm_timesteps'] must be a whole number, got 2.7"),
            (["sample_rate": 24000.5],
             "manifest['sample_rate'] must be a whole number, got 24000.5"),
            (["speech_vocab_size": 8194.5],
             "manifest['speech_vocab_size'] must be a whole number, got 8194.5"),
            (["postprocess": ["stall_run_tokens": 2.7]],
             "manifest['postprocess']['stall_run_tokens'] must be a whole number, got 2.7"),
            (["silence_token_ids": [4137, 4218.5]],
             "manifest['silence_token_ids'][1] must be a whole number, got 4218.5"),
            (["silence_render_ids": [2.7]],
             "manifest['silence_render_ids'][0] must be a whole number, got 2.7"),
            (["quiet_render_ids": [2.7]],
             "manifest['quiet_render_ids'][0] must be a whole number, got 2.7"),
            // The fraction is named before the magnitude and before the sign,
            // as the reference names only the fraction.
            (["window": ["pad_token_id": -1.5]],
             "manifest['window']['pad_token_id'] must be a whole number, got -1.5"),
        ]
        for (overlay, sentence) in cases {
            var manifest = Self.shipped011()
            for (key, value) in overlay {
                if var block = manifest[key] as? [String: Any],
                   let patch = value as? [String: Any] {
                    for (k, v) in patch { block[k] = v }
                    manifest[key] = block
                } else {
                    manifest[key] = value
                }
            }
            XCTAssertThrowsError(
                try AlgorithmConfig.fromManifest(Self.decoded(manifest)),
                "\(overlay) must be refused"
            ) { error in
                // `LoudKitError.manifest` names its own kind first; what
                // follows is the reference's sentence.
                XCTAssertEqual(
                    (error as? LoudKitError)?.errorDescription, "manifest: \(sentence)")
            }
        }
    }

    /// And the fields that hold a rate, a threshold or a probability still
    /// take one. Getting the boundary wrong in this direction refuses a
    /// manifest that is correct, which is the same defect facing the other
    /// way.
    func testAFractionalRateIsAValue() throws {
        var manifest = Self.shipped011()
        manifest["token_rate_hz"] = 25.5
        manifest["edge_fade_seconds"] = 0.03
        var defaults = try XCTUnwrap(manifest["sampling_defaults"] as? [String: Any])
        defaults["temperature"] = 0.75
        defaults["repetition_penalty"] = 1.25
        defaults["min_p"] = 0.055
        manifest["sampling_defaults"] = defaults
        var eos = try XCTUnwrap(manifest["eos_floor"] as? [String: Any])
        eos["min_tokens_text_ratio"] = 1.25
        manifest["eos_floor"] = eos
        var postprocess = try XCTUnwrap(manifest["postprocess"] as? [String: Any])
        postprocess["pacing_tolerance"] = 0.17
        manifest["postprocess"] = postprocess
        let config = try AlgorithmConfig.fromManifest(Self.decoded(manifest))
        XCTAssertEqual(config.tokenRateHz, 25.5)
        XCTAssertEqual(config.edgeFadeSeconds, 0.03)
        XCTAssertEqual(config.sampling.temperature, 0.75)
        XCTAssertEqual(config.sampling.repetitionPenalty, 1.25)
        XCTAssertEqual(config.sampling.minP, 0.055)
        XCTAssertEqual(config.sampling.minTokensTextRatio, 1.25)
        XCTAssertEqual(config.postprocess.pacingTolerance, 0.17)
    }

    /// The declared format version is the one number the reference still
    /// truncates, and this port follows it there.
    ///
    /// `checkpoint._read_manifest` reads it with a bare `int()`, one module
    /// away from the reader that refuses a fractional count, so a file saying
    /// `format_version: 2.7` opens in all five. Applying the count rule here
    /// would refuse a checkpoint every other implementation loads, which is
    /// the same defect as truncating, facing the other way.
    func testTheFormatVersionIsTruncatedTheWayTheReferenceTruncatesIt() throws {
        XCTAssertEqual(
            try ManifestReader.formatVersion(["format_version": 2.7], "format_version", -1), 2)
        XCTAssertEqual(
            try ManifestReader.formatVersion(["format_version": 1], "format_version", -1), 1)
        XCTAssertEqual(try ManifestReader.formatVersion([:], "format_version", -1), -1)
    }
}
