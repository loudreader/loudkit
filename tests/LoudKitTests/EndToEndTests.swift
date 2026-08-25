import XCTest

import LoudKitText

@testable import LoudKit

/// The gate from PLAN step 10: Swift and Python render the same tokens from
/// the same seed, and their waveforms agree at the band from step 2. Needs
/// the packed checkpoint and the exported CoreML packages; skips with a
/// named reason without them (LOUDKIT_REQUIRE_ASSETS=1 makes that a failure).
final class EndToEndConformanceTests: XCTestCase {
    static var engine: Engine?

    private func loadEngine() throws -> Engine {
        if let engine = Self.engine { return engine }
        try Fixture.requireCheckpoint()
        let engine = try Engine.load(
            checkpoint: Fixture.checkpointURL, coremlAssets: Fixture.coremlAssetsURL)
        Self.engine = engine
        return engine
    }

    private func voiceProfile(_ relative: String) throws -> VoiceProfile {
        try VoiceProfile.load(url: Fixture.conformanceDir.appendingPathComponent(relative))
    }

    func testTokensAndWaveformAgainstPython() throws {
        let fixture = try Fixture.vectors()
        guard let cases = fixture["end_to_end"] as? [[String: Any]], !cases.isEmpty else {
            throw XCTSkip("fixture has no end_to_end section")
        }
        let engine = try loadEngine()
        let voice = try voiceProfile(cases[0]["voice"] as! String)
        for kase in cases {
            let name = kase["name"] as! String
            let text = kase["text"] as! String
            let language = kase["language"] as! String
            let seed = UInt64((kase["seed"] as! NSNumber).uint64Value)
            let wantTokens = asInts(kase["tokens"])!
            let gates = kase["gates"] as! [String: NSNumber]

            let result = try engine.synthesize(text, voice: voice, seed: seed, language: language)
            XCTAssertFalse(result.hitTokenCap, "\(name) hit the token cap")
            XCTAssertEqual(result.tokens, wantTokens,
                           "\(name): Swift and Python sampled different tokens from seed \(seed)")

            let melMeta = kase["mel"] as! [String: Any]
            let shape = asInts(melMeta["shape"])!
            let melRef = try loadFloats(melMeta["file"] as! String)
            XCTAssertEqual(result.mel.count, shape[0] * shape[1], "\(name) mel shape")
            let melCorr = correlation(result.mel, melRef)
            XCTAssertGreaterThanOrEqual(melCorr, gates["mel_corr"]!.doubleValue,
                                        "\(name) mel corr \(melCorr)")

            let wavMeta = kase["wav"] as! [String: Any]
            let wavRef = try loadFloats(wavMeta["file"] as! String)
            XCTAssertEqual(result.audio.count, (wavMeta["samples"] as! NSNumber).intValue,
                           "\(name) sample count")
            let waveCorr = correlation(result.audio, wavRef)
            XCTAssertGreaterThanOrEqual(waveCorr, gates["wave_corr"]!.doubleValue,
                                        "\(name) wave corr \(waveCorr)")
            var maxDiff: Float = 0
            for i in 0..<min(result.audio.count, wavRef.count) {
                maxDiff = max(maxDiff, abs(result.audio[i] - wavRef[i]))
            }
            print("conformance \(name): tokens \(result.tokens.count)/\(wantTokens.count) exact, "
                  + String(format: "mel corr %.9f, wave corr %.9f, wave max|d| %.3e",
                           melCorr, waveCorr, maxDiff))
        }
    }

    func testRerenderIsBitIdentical() throws {
        let fixture = try Fixture.vectors()
        guard let cases = fixture["end_to_end"] as? [[String: Any]], !cases.isEmpty else {
            throw XCTSkip("fixture has no end_to_end section")
        }
        let engine = try loadEngine()
        let voice = try voiceProfile(cases[0]["voice"] as! String)
        let kase = cases[0]
        let tokens = asInts(kase["tokens"])!
        let seed = UInt64((kase["seed"] as! NSNumber).uint64Value)
        let a = try engine.synthesizeTokens(tokens, voice: voice, seed: seed)
        let b = try engine.synthesizeTokens(tokens, voice: voice, seed: seed)
        XCTAssertEqual(a.audio, b.audio, "I-2: same seed, same build, bit-identical waveform")
    }

    private func loadFloats(_ file: String) throws -> [Float] {
        let data = try Data(contentsOf: Fixture.conformanceDir.appendingPathComponent(file))
        return data.withUnsafeBytes { buf in
            Array(buf.bindMemory(to: Float32.self))
        }
    }
}

/// Long-form composition, on the real checkpoint.
///
/// `swift/LoudKit` had none: `synthesize` renders one window (~127 characters
/// of prepared text) and refuses anything longer, so a caller with a paragraph
/// split it themselves — and a caller who splits differently gets different
/// chunk boundaries, different derived seeds, and different audio from every
/// other port, while the fingerprint goes on declaring the chunking recipe.
final class LongFormTests: XCTestCase {
    /// Shared with `EndToEndConformanceTests`: loading the CoreML packages
    /// twice in one process is minutes, not seconds.
    static func loadSharedEngine() throws -> Engine {
        if let engine = EndToEndConformanceTests.engine { return engine }
        try Fixture.requireCheckpoint()
        let engine = try Engine.load(
            checkpoint: Fixture.checkpointURL, coremlAssets: Fixture.coremlAssetsURL)
        EndToEndConformanceTests.engine = engine
        return engine
    }

    /// The same voice the end-to-end fixture names, resolved the same way.
    static func referenceVoice() throws -> VoiceProfile {
        try VoiceProfile.load(
            url: Fixture.conformanceDir.appendingPathComponent(
                "../reference/testvoice.voice.safetensors"))
    }

    /// The whole-passage path is the streaming path with the chunks joined. If
    /// they ever become two loops they will drift, and the drift is inaudible
    /// until a join lands somewhere different — so the equality is asserted.
    func testStreamAndSynthesizeLongAreOneLoop() throws {
        let engine = try Self.loadSharedEngine()
        let voice = try Self.referenceVoice()

        // Comfortably past one window: the budget is
        // floor(max_tokens * charsPerToken) = 127 characters, so a passage that
        // merely feels long can still arrive as a single chunk and make this
        // test assert nothing.
        let text =
            "The first sentence sets the scene and runs on for a while. "
            + "The second sentence follows it and is no shorter than the first one was. "
            + "The third sentence exists so that the splitter has somewhere to breathe. "
            + "The fourth sentence closes the passage without hurrying."

        let whole = try engine.synthesizeLong(text, voice: voice, seed: 7)

        var pieces: [[Float]] = []
        var tokens: [Int] = []
        try engine.stream(text, voice: voice, seed: 7) { chunk in
            pieces.append(chunk.audio)
            tokens.append(contentsOf: chunk.tokens)
            return true
        }
        XCTAssertGreaterThan(pieces.count, 1, "the passage must actually split")

        let streamed = pieces.flatMap { $0 }
        XCTAssertEqual(streamed.count, whole.audio.count, "streamed and joined lengths differ")
        XCTAssertEqual(tokens, whole.tokens, "streamed and joined tokens differ")
        XCTAssertEqual(streamed, whole.audio, "streamed and joined samples differ")
    }

    /// A passage too long for one window, chunk by chunk, against Python.
    ///
    /// `EndToEndConformanceTests` above is single-window work with an empty
    /// prefix, and with an empty prefix `prefix.count + step + 1` and
    /// `step + 1` are the same number and a repetition mask seeded from the
    /// prefix is the empty one. Three ports wrote the shorter form of both and
    /// the fixture passed throughout. A carried prefix is what separates them,
    /// and this port has always indexed it the long way — so a failure here
    /// means the fixture case is wrong, not that Swift is.
    ///
    /// Every chunk is asserted on its own rather than on the concatenation: a
    /// divergence inside chunk *k* shifts every token after it, so a
    /// whole-passage comparison reports one enormous mismatch instead of
    /// naming the chunk.
    ///
    /// The generator's own tokens, before the postprocess trim the streaming
    /// path applies to the terminal chunk — the same layer the fixture's
    /// `end_to_end` tokens are taken at.
    func testLongFormChunkTokensAgainstPython() throws {
        let fixture = try Fixture.vectors()
        guard let section = fixture["long_form"] as? [String: Any],
            let cases = section["cases"] as? [[String: Any]], !cases.isEmpty
        else {
            throw XCTSkip("fixture has no long_form section")
        }
        let engine = try Self.loadSharedEngine()
        let voice = try Self.referenceVoice()
        let prefixTokens = (section["prefix_tokens"] as! NSNumber).intValue
        XCTAssertEqual(engine.algorithm.chunking.prefixTokens, prefixTokens,
                       "this port carries a different number of tokens across a join")

        for kase in cases {
            let name = kase["name"] as! String
            let language = kase["language"] as! String
            // Funnel first, then split — the order the engine uses, and the
            // order the character budget assumes.
            let prepared = SpeechText.prepared(kase["text"] as! String, languageId: language)
            XCTAssertEqual(prepared, kase["prepared"] as! String,
                           "\(name): the speech funnel drifted")
            let chunks = kase["chunks"] as! [[String: Any]]
            XCTAssertGreaterThan(chunks.count, 1,
                                 "\(name) is a single window and proves nothing")
            XCTAssertEqual(
                Chunking.splitText(prepared, config: engine.algorithm.chunking),
                chunks.map { $0["text"] as! String },
                "\(name): the split moved, so every chunk below is asking about different text")

            for chunk in chunks {
                let index = (chunk["index"] as! NSNumber).intValue
                let prefix = asInts(chunk["prefix"]) ?? []
                let want = asInts(chunk["tokens"])!
                // The chain the streaming path walks: chunk k is conditioned on
                // the tail of chunk k-1. Spelled out in the fixture so a
                // mismatch names the carry rather than the tokens that followed
                // from it.
                if index > 0 {
                    let previous = asInts(chunks[index - 1]["tokens"])!
                    XCTAssertEqual(prefix, Array(previous.suffix(prefixTokens)),
                                   "\(name) chunk \(index): carry")
                }
                // Hex: a derived 64-bit seed does not survive a JSON double.
                let seed = UInt64((chunk["seed"] as! String).dropFirst(2), radix: 16)!
                let ids = try engine.frontend.encode(chunk["text"] as! String, language: language)
                let sampler = LRSamplerV1(config: engine.algorithm.sampling, seed: seed)
                let generation = engine.tokenGenerator.generate(
                    textTokens: ids, voice: voice, sampler: sampler, prefix: prefix)
                let got = generation.rawTokens.filter { $0 < engine.algorithm.startSpeechToken }
                XCTAssertEqual(got, want, "\(name) chunk \(index)")
            }
            print("conformance \(name): long-form tokens exact across \(chunks.count) chunks")
        }
    }

    /// Stopping early must not change what was already produced: each chunk
    /// draws from `derive(seed, 16 + index)`, so its audio does not depend on
    /// how many chunks follow it.
    func testStoppingEarlyLeavesEarlierChunksUnchanged() throws {
        let engine = try Self.loadSharedEngine()
        let voice = try Self.referenceVoice()
        let text =
            "The first sentence sets the scene and runs on for a while. "
            + "The second sentence follows it and is no shorter than the first one was. "
            + "The third sentence exists so that the splitter has somewhere to breathe."

        var all: [[Float]] = []
        try engine.stream(text, voice: voice, seed: 7) { chunk in
            all.append(chunk.audio)
            return true
        }
        XCTAssertGreaterThan(all.count, 1)

        var first: [[Float]] = []
        try engine.stream(text, voice: voice, seed: 7) { chunk in
            first.append(chunk.audio)
            return false  // stop after the first
        }
        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(first[0], all[0], "the first chunk changed when the rest was dropped")
    }
}
