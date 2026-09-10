import Foundation
import XCTest

@testable import LoudKit

/// The enrollment port, gated against the shared fixture: the same reference
/// clip must yield the fixture's prompt tokens exactly and its embeddings to
/// cosine > 0.9999. Needs the exported enrollment CoreML packages; skips with
/// a named reason otherwise.
final class EnrollmentTests: XCTestCase {
    private static var coremlDir: URL? {
        if let env = ProcessInfo.processInfo.environment["LOUDKIT_COREML_ASSETS"] {
            return URL(fileURLWithPath: env)
        }
        let ckpt = Fixture.checkpointURL
        let dir = ckpt.deletingLastPathComponent().appendingPathComponent("coreml")
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("s3_tokenizer.mlpackage").path) ? dir : nil
    }

    private static func requireCoreml() throws -> URL {
        guard let dir = coremlDir else {
            if Fixture.requireAssets {
                XCTFail("LOUDKIT_REQUIRE_ASSETS is set but enrollment CoreML packages are missing")
            }
            throw XCTSkip("enrollment CoreML packages not found: run tools/export_enroll_coreml.py")
        }
        return dir
    }

    private static var fixtureDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // EnrollmentTests.swift
            .deletingLastPathComponent()  // LoudKitTests
            .deletingLastPathComponent()  // tests
            .appendingPathComponent("tests/data/enrollment")
    }

    private func readF32(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: Self.fixtureDir.appendingPathComponent(name))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    private func readI64(_ name: String) throws -> [Int] {
        let data = try Data(contentsOf: Self.fixtureDir.appendingPathComponent(name))
        return data.withUnsafeBytes { buf in
            Array(buf.bindMemory(to: Int64.self)).map(Int.init)
        }
    }

    private func cos(_ a: [Float], _ b: [Float]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0..<a.count {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
        }
        return dot / (na.squareRoot() * nb.squareRoot())
    }

    /// One enrollment, shared by every test in this class.
    ///
    /// Enrolling costs about six minutes here, and the three tests below check
    /// three properties of one enrollment rather than three enrollments. A
    /// `static` and not a `lazy var`: XCTest builds a fresh instance per test,
    /// so an instance-level cache would never be reused. The failure is cached
    /// too, so a broken setup is reported by each test instead of being retried
    /// three times.
    private static var cached: Result<EnrolledVoice, Error>?

    private func enroll() throws -> EnrolledVoice {
        if let cached = Self.cached { return try cached.get() }
        let outcome: Result<EnrolledVoice, Error>
        do {
            let dir = try Self.requireCoreml()
            let enroller = try Enrollment.Enroller(coremlDir: dir)
            let audio = try readF32("ref_audio.f32")
            outcome = .success(try enroller.enroll(audio, sampleRate: 24_000))
        } catch {
            outcome = .failure(error)
        }
        Self.cached = outcome
        return try outcome.get()
    }

    func testPromptTokensExact() throws {
        let voice = try enroll()
        let want = try readI64("prompt_tokens.i64")
        XCTAssertEqual(voice.promptTokens, want, "prompt tokens must match exactly")
    }

    func testCondTokensExact() throws {
        let voice = try enroll()
        let want = try readI64("cond_prompt_tokens.i64")
        XCTAssertEqual(voice.condPromptTokens, want, "cond tokens must match exactly")
    }

    func testEmbeddingsMatch() throws {
        let voice = try enroll()
        let flow = try readF32("flow_embedding.f32")
        let speaker = try readF32("speaker_embedding.f32")
        XCTAssertGreaterThan(cos(voice.flowEmbedding, flow), 0.9999, "flow embedding cosine")
        XCTAssertGreaterThan(cos(voice.speakerEmbedding, speaker), 0.9999, "speaker embedding cosine")
    }
    func testCloneOnceSpeaksWithBothModelsWithoutChangingProfile() throws {
        try Fixture.requireFusionCheckpoint()
        let enrolled = try enroll()
        let voice = VoiceProfile(name: "shared", speakerEmbedding: enrolled.speakerEmbedding,
            flowEmbedding: enrolled.flowEmbedding, promptTokens: enrolled.promptTokens,
            promptMel: enrolled.promptMel, promptMelFrames: enrolled.promptMelFrames,
            condPromptTokens: enrolled.condPromptTokens, language: "en")
        let path = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: path) }
        try voice.save(to: path)
        let before = try Data(contentsOf: path)
        for engine in [try LongFormTests.loadSharedEngine(), try EndToEndConformanceTests.loadFusionEngine()] {
            let result = try engine.synthesize("Hello from one shared voice.",
                                               voice: VoiceProfile.load(url: path), seed: 7)
            XCTAssertFalse(result.audio.isEmpty)
            XCTAssertTrue(result.audio.allSatisfy(\.isFinite))
            XCTAssertEqual(try Data(contentsOf: path), before)
        }
    }

}

/// The enrollment DSP seams that need no model: the shared DFT basis and the
/// bundled float32 tables.
final class EnrollmentDSPTests: XCTestCase {
    /// The basis is built once per call site and handed to every frame, so a
    /// frame must score exactly as it did against a basis of its own.
    func testTheSharedBasisScoresAFrameLikeAPerFrameOne() {
        let nfft = 16
        var frame = [Double](repeating: 0, count: nfft)
        for i in 0..<nfft { frame[i] = Foundation.sin(Double(i) * 0.7) + 0.25 * Double(i % 3) }
        let (cosT, sinT) = Enrollment.basis(nfft)
        let shared = Enrollment.powerSpectrum(frame, nfft, cosT, sinT)

        var want = [Double](repeating: 0, count: nfft / 2 + 1)
        for k in 0..<want.count {
            var re = 0.0, im = 0.0
            for n in 0..<nfft {
                let (perFrameCos, perFrameSin) = Enrollment.basis(nfft)
                let a = perFrameCos[k][n], b = perFrameSin[k][n]
                re += a * frame[n]
                im += b * frame[n]
            }
            want[k] = re * re + im * im
        }
        XCTAssertEqual(shared, want)
    }

    /// A table that is not in the package is an error at the missing file, not
    /// an index trap several hundred lines later.
    func testAMissingTableIsRefusedByName() {
        XCTAssertThrowsError(try Enrollment.table("no_such_filterbank")) { error in
            XCTAssertEqual(
                (error as? LoudKitError)?.description,
                "asset: no_such_filterbank.f32 is missing from the package resources; "
                + "the enrollment filterbanks cannot be built without it")
        }
    }

    func testTheBundledTablesLoad() throws {
        for name in ["s3_hann400", "s3_mel_filters", "matcha_hann1920", "matcha_mel_filters",
                     "kaldi_mel_filters", "kaldi_povey400", "voiceenc_hann400",
                     "voiceenc_mel_filters"] {
            XCTAssertFalse(try Enrollment.table(name).isEmpty, name)
        }
    }
}

/// The five refusals `validate_reference_audio` makes on the Python side, run
/// here without weights. Each one used to be a trap, a padded enrollment or a
/// silently accepted five-minute recording.
final class ReferenceAudioValidationTests: XCTestCase {
    private static let goodInput =
        "A good input is 5 to 10 seconds of one person speaking, clean, "
        + "without music or a second voice."

    private func refusal(_ audio: [Float], sampleRate: Int,
                         file: StaticString = #filePath, line: UInt = #line) -> String? {
        do {
            try Enrollment.validateReferenceAudio(audio, sampleRate: sampleRate)
            XCTFail("the recording was accepted", file: file, line: line)
            return nil
        } catch let error as LoudKitError {
            return error.description
        } catch {
            XCTFail("unexpected error \(error)", file: file, line: line)
            return nil
        }
    }

    func testANonPositiveRateIsRefused() {
        XCTAssertEqual(refusal([Float](repeating: 0.5, count: 24_000), sampleRate: 0),
                       "shape: sample rate must be positive, got 0")
        XCTAssertEqual(refusal([Float](repeating: 0.5, count: 24_000), sampleRate: -24_000),
                       "shape: sample rate must be positive, got -24000")
    }

    func testNaNAndInfSamplesAreRefused() {
        var audio = [Float](repeating: 0.5, count: 24_000 * 2)
        audio[1234] = .nan
        XCTAssertEqual(refusal(audio, sampleRate: 24_000),
                       "shape: the recording contains NaN or Inf samples, so no voice can "
                       + "be derived from it. Re-export the file. " + Self.goodInput)
        audio[1234] = .infinity
        XCTAssertEqual(refusal(audio, sampleRate: 24_000),
                       "shape: the recording contains NaN or Inf samples, so no voice can "
                       + "be derived from it. Re-export the file. " + Self.goodInput)
    }

    /// The clip that killed the process: 720 samples is one short of the reflect
    /// padding `matchaMel` reads, so it used to index past the end of the array.
    func testAClipShorterThanASecondIsRefusedBeforeTheReflectPadding() {
        XCTAssertEqual(refusal([Float](repeating: 0.5, count: 720), sampleRate: 24_000),
                       "shape: the recording is 0.03 s: too short to enroll a speaker from "
                       + "(minimum 1 s). " + Self.goodInput)
        XCTAssertEqual(refusal([], sampleRate: 24_000),
                       "shape: the recording is 0.00 s: too short to enroll a speaker from "
                       + "(minimum 1 s). " + Self.goodInput)
        // Half a second: no trap, but the utterance encoder pads it out to its
        // 1.6 s first partial and enrolls mostly padding.
        XCTAssertEqual(refusal([Float](repeating: 0.5, count: 12_000), sampleRate: 24_000),
                       "shape: the recording is 0.50 s: too short to enroll a speaker from "
                       + "(minimum 1 s). " + Self.goodInput)
    }

    func testARecordingLongerThanThirtySecondsIsRefused() {
        XCTAssertEqual(refusal([Float](repeating: 0.5, count: 24_000 * 31), sampleRate: 24_000),
                       "shape: the recording is 31.0 s. Only the first 10 s become the voice "
                       + "prompt, and the whole clip shapes the speaker embedding, so a long "
                       + "recording enrolls something the prompt does not carry. Trim it to "
                       + "the best 5 to 10 seconds (at most 30 s). " + Self.goodInput)
    }

    func testASilentRecordingIsRefused() {
        XCTAssertEqual(refusal([Float](repeating: 0, count: 24_000 * 2), sampleRate: 24_000),
                       "shape: the recording is silent (peak 0.0e+00); there is no voice in "
                       + "it to enroll. " + Self.goodInput)
        XCTAssertEqual(refusal([Float](repeating: 5e-5, count: 24_000 * 2), sampleRate: 24_000),
                       "shape: the recording is silent (peak 5.0e-05); there is no voice in "
                       + "it to enroll. " + Self.goodInput)
    }

    /// The bounds are inclusive at both ends and the fixture clip sits inside
    /// them, so tightening the guard cannot start refusing what already enrolls.
    func testTheBandItselfIsAccepted() throws {
        for count in [24_000, 24_000 * 15, 24_000 * 30] {
            var audio = [Float](repeating: 0, count: count)
            audio[0] = 1e-4
            XCTAssertNoThrow(try Enrollment.validateReferenceAudio(audio, sampleRate: 24_000),
                             "\(count) samples at 24 kHz")
        }
    }
}
