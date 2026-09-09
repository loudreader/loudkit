import Foundation
import XCTest

@testable import LoudKit

/// Discovery over a release directory, built here rather than downloaded.
///
/// None of these files are real weights: what is under test is the layout rules
/// and the sentences they produce, and both are decided before a byte of a
/// checkpoint is read.
final class BundleTests: XCTestCase {
    private var root = URL(fileURLWithPath: NSTemporaryDirectory())

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("loudkit-bundle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: layout

    private func write(_ relative: String, _ contents: String = "x") throws {
        let url = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(contents.utf8).write(to: url)
    }

    private func package(_ relative: String) throws {
        try write("\(relative)/Manifest.json", "{}")
        try write("\(relative)/Data/com.apple.CoreML/model.mlmodel", "weights")
    }

    /// A complete CoreML release, minus whatever the caller leaves out.
    private func release(cloning: Bool = false, checkpoint: String = LoudKit.checkpointName) throws {
        try write(checkpoint)
        try write("tokenizer.json", "{}")
        try write("manifest.json", "{}")
        try write("voices/joe.safetensors")
        try write("voices/ada.safetensors")
        for name in LoudKit.coremlSynthesis { try package(name) }
        if cloning { for name in LoudKit.coremlEnrollment { try package(name) } }
    }

    // MARK: what it finds

    func testFindsEveryPieceFromTheDirectoryAlone() throws {
        try release()
        let bundle = try ModelBundle(directory: root)
        XCTAssertEqual(bundle.checkpoint.lastPathComponent, LoudKit.checkpointName)
        XCTAssertEqual(bundle.tokenizer.lastPathComponent, "tokenizer.json")
        XCTAssertEqual(bundle.coreml.lastPathComponent, "coreml")
        XCTAssertEqual(bundle.voiceNames, ["ada", "joe"])
    }

    /// The compiled form is what an app bundle ships, and `Engine.load` already
    /// accepts it. Discovery has to agree, or an app that skipped the on-device
    /// compile is told its release is incomplete.
    func testAPrecompiledStageCountsAsThePackage() throws {
        try release()
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("coreml/vocoder.mlpackage"))
        try write("coreml/vocoder.mlmodelc/model.espresso.net", "compiled")
        XCTAssertNoThrow(try ModelBundle(directory: root))
    }

    func testNamesEverythingMissingAtOnce() throws {
        try release()
        try FileManager.default.removeItem(at: root.appendingPathComponent("tokenizer.json"))
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("coreml/flow_estimator.mlpackage"))
        XCTAssertThrowsError(try ModelBundle(directory: root)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("tokenizer.json"), message)
            XCTAssertTrue(message.contains("coreml/flow_estimator.mlpackage"), message)
        }
    }

    /// The shortfall a pattern fetch actually produces: the package directory
    /// arrives, its weights do not. "A directory with something in it" passes
    /// for that, which is why the rule is `Manifest.json` beside a non-empty
    /// `Data/`.
    func testAnEmptyPackageIsNotAPackage() throws {
        try release()
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("coreml/vocoder.mlpackage/Data"))
        XCTAssertThrowsError(try ModelBundle(directory: root)) { error in
            XCTAssertTrue(String(describing: error).contains("coreml/vocoder.mlpackage"),
                          String(describing: error))
        }
    }

    func testAFileIsNotABundle() throws {
        try release()
        XCTAssertThrowsError(
            try ModelBundle(directory: root.appendingPathComponent(LoudKit.checkpointName)))
    }

    // MARK: which checkpoint

    /// `ve.safetensors` sits at the root of every cloning-capable release. The
    /// counting rule that did not know its name answered an ordinary release
    /// with "2 checkpoints, name the one you mean".
    func testTheVoiceEncoderIsNotACandidate() throws {
        try release(checkpoint: "some-model.safetensors")
        try write("ve.safetensors")
        let bundle = try ModelBundle(directory: root)
        XCTAssertEqual(bundle.checkpoint.lastPathComponent, "some-model.safetensors")
    }

    func testTwoUnnamedCheckpointsAreAmbiguous() throws {
        try release(checkpoint: "one.safetensors")
        try write("two.safetensors")
        XCTAssertThrowsError(try ModelBundle(directory: root)) { error in
            XCTAssertTrue(String(describing: error).contains("2 checkpoints"),
                          String(describing: error))
        }
    }

    func testVoicesAreNotCheckpoints() throws {
        try release(checkpoint: "only.safetensors")
        XCTAssertEqual(try ModelBundle.findCheckpoint(in: root).lastPathComponent,
                       "only.safetensors")
    }

    /// They are different releases with different decode loops, so returning
    /// the one whose name is checked first returns a model nobody asked for.
    func testBothModelsUnderTheirOwnNamesIsNotAPickToMake() throws {
        try release()
        try write(LoudKit.turboCheckpointName)
        XCTAssertThrowsError(try ModelBundle.findCheckpoint(in: root)) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("2 models here"), message)
            XCTAssertTrue(message.contains("no right one to pick"), message)
        }
    }

    // MARK: voices and enrollment

    func testAWrongVoiceNameListsTheOnesThatExist() throws {
        try release()
        let bundle = try ModelBundle(directory: root)
        XCTAssertThrowsError(try bundle.voice(named: "jo")) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("ada, joe"), message)
        }
    }

    func testEnrollmentSaysWhichPackagesTheFetchLeftOut() throws {
        try release()
        let bundle = try ModelBundle(directory: root)
        XCTAssertThrowsError(try bundle.enroller()) { error in
            let message = String(describing: error)
            XCTAssertTrue(message.contains("coreml/camp.mlpackage"), message)
            XCTAssertTrue(message.contains("cloning: true"), message)
        }
    }

    /// The mirror is the fetch's business, not the loader's: the checkpoint
    /// embeds the manifest the engine actually reads, so a directory without
    /// `manifest.json` loads.
    func testTheManifestMirrorIsNotRequiredToLoad() throws {
        try release()
        try FileManager.default.removeItem(at: root.appendingPathComponent("manifest.json"))
        XCTAssertNoThrow(try ModelBundle(directory: root))
        XCTAssertThrowsError(try LoudKit.verifyInventory(at: root))
    }

    /// `Engine.load(_:)`'s rule is the dumb one from `hub.py`: anything that
    /// exists on disk is a path. This directory exists and is not a release,
    /// so the complaint must be the bundle's.
    func testRepoOrDirectoryTakesAPathBeforeTheNetwork() async throws {
        try write("nothing-useful.txt")
        do {
            _ = try await Engine.load(root.path)
            XCTFail("an empty directory loaded as a release")
        } catch {
            let message = String(describing: error)
            XCTAssertTrue(message.contains("no *.safetensors here"), message)
        }
    }

    func testRepoOrDirectoryRefusesWhatIsNeither() async throws {
        do {
            _ = try await Engine.load("not-a-model")
            XCTFail("an unknown name was accepted")
        } catch {
            let message = String(describing: error)
            XCTAssertTrue(message.contains("not a Hugging Face repo id"), message)
        }
    }

    // MARK: the whole first mile, when the weights are here

    /// Download-shaped directory to hello.wav, which is the claim the README
    /// makes. Assembled from the local assets by symlink rather than fetched:
    /// what is under test is that discovery hands `Engine.load` the same three
    /// paths a person would have typed.
    private func linkedRelease() throws -> ModelBundle {
        try Fixture.requireCheckpoint()
        let assets = Fixture.checkpointURL.deletingLastPathComponent()
        let fm = FileManager.default
        for name in [Fixture.checkpointURL.lastPathComponent, "tokenizer.json",
                     "coreml", ModelBundle.voiceDirectory] {
            let source = assets.appendingPathComponent(name)
            try XCTSkipUnless(fm.fileExists(atPath: source.path),
                              "\(name) is not beside the checkpoint")
            try fm.createSymbolicLink(at: root.appendingPathComponent(name),
                                      withDestinationURL: source)
        }
        return try ModelBundle(directory: root)
    }

    func testABundleShapedDirectoryProducesAWavFile() throws {
        let bundle = try linkedRelease()
        let engine = try Engine.load(bundle: bundle)
        // The voices come off the engine, which remembers the release.
        XCTAssertEqual(engine.voiceNames, bundle.voiceNames)
        XCTAssertFalse(engine.voiceNames.isEmpty)
        let voice = try engine.voice(named: try XCTUnwrap(engine.voiceNames.first))
        let result = try engine.synthesize("Hello from loudkit.", voice: voice, seed: 7)
        let wav = root.appendingPathComponent("hello.wav")
        try result.saveWav(to: wav)

        let written = try Data(contentsOf: wav)
        XCTAssertEqual(written.count, 44 + result.audio.count * 2)
        XCTAssertEqual([UInt8](written[0..<4]), Array("RIFF".utf8))
        XCTAssertGreaterThan(result.duration, 0.5, "a file this short is not the sentence")
    }

    /// An enrolled voice becomes a `VoiceProfile` with its tensors intact.
    ///
    /// The memberwise initialiser is internal, so before this wrapper existed
    /// `Enroller.enroll` returned five tensors and no way to speak with them.
    ///
    /// The whole file-to-profile path over the real graphs was run once by hand
    /// against the local packages and produced a 256-d speaker vector, a 192-d
    /// flow vector and a prompt mel of `80 * promptMelFrames`. It is not
    /// committed: the enrollment DSP is a naive DFT, and eleven seconds of
    /// reference, the length `camp.mlpackage`'s fixed geometry requires, takes
    /// six minutes in a debug build.
    func testAnEnrolledVoiceBecomesAProfile() {
        // Distinct fills on the two float fields and the two token fields, so
        // a pair swapped in the initialiser is caught.
        let enrolled = EnrolledVoice(
            speakerEmbedding: [Float](repeating: 0.1, count: 256),
            flowEmbedding: [Float](repeating: 0.2, count: 192),
            promptTokens: [1, 2, 3],
            promptMel: [Float](repeating: 0.3, count: 80 * 4),
            promptMelFrames: 4,
            condPromptTokens: [4, 5])
        let voice = VoiceProfile(enrolled, name: "mine", language: "pl", sourceSampleRate: 16_000)
        XCTAssertEqual(voice.speakerEmbedding, enrolled.speakerEmbedding)
        XCTAssertEqual(voice.flowEmbedding, enrolled.flowEmbedding)
        XCTAssertEqual(voice.promptTokens, enrolled.promptTokens)
        XCTAssertEqual(voice.condPromptTokens, enrolled.condPromptTokens)
        XCTAssertEqual(voice.sourceSampleRate, 16_000)
        XCTAssertEqual(VoiceProfile(enrolled, name: "x").language, "en", "the default")
    }

    /// A saved profile reads back as itself, with the header Python writes.
    func testASavedProfileReadsBackAsItself() throws {
        let enrolled = EnrolledVoice(
            speakerEmbedding: (0..<256).map { Float($0 % 7) * 0.125 },
            flowEmbedding: (0..<192).map { Float($0 % 5) * 0.25 - 0.5 },
            promptTokens: [1, 2, 3],
            promptMel: (0..<(80 * 3)).map { Float($0) * 0.01 },
            promptMelFrames: 3,
            condPromptTokens: [4, 5])
        let voice = VoiceProfile(enrolled, name: "mine", language: "pl", sourceSampleRate: 16_000)
        let url = root.appendingPathComponent("mine.safetensors")
        try voice.save(to: url)
        let back = try VoiceProfile.load(url: url)
        XCTAssertEqual(back.name, "mine")
        XCTAssertEqual(back.language, "pl")
        XCTAssertEqual(back.sourceSampleRate, 16_000)
        XCTAssertEqual(back.speakerEmbedding, voice.speakerEmbedding)
        XCTAssertEqual(back.flowEmbedding, voice.flowEmbedding)
        XCTAssertEqual(back.promptTokens, voice.promptTokens)
        XCTAssertEqual(back.promptMel, voice.promptMel)
        XCTAssertEqual(back.promptMelFrames, 3)
        XCTAssertEqual(back.condPromptTokens, voice.condPromptTokens)
        // The header keys are the ones python/loudkit/voice.py writes.
        let store = try Safetensors(url: url)
        let header = try JSONSerialization.jsonObject(
            with: Data(try XCTUnwrap(store.metadata["voice"]).utf8)) as? [String: Any]
        XCTAssertEqual(Set(try XCTUnwrap(header).keys),
                       ["format_version", "name", "source_sample_rate", "language", "enrolment"])
    }

    // MARK: the fetch's own receipt

    func testInventoryPassesOnACompleteFetch() throws {
        try release(cloning: true)
        XCTAssertNoThrow(try LoudKit.verifyInventory(at: root, cloning: true))
    }

    func testInventoryRefusesAVoicelessFetch() throws {
        try release()
        try FileManager.default.removeItem(at: root.appendingPathComponent("voices"))
        XCTAssertThrowsError(try LoudKit.verifyInventory(at: root)) { error in
            XCTAssertTrue(String(describing: error).contains("voices/*.safetensors"),
                          String(describing: error))
        }
    }

    func testInventoryRefusesASynthesisFetchMissingAPackage() throws {
        try release()
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("coreml/flow_encoder.mlpackage"))
        XCTAssertThrowsError(try LoudKit.verifyInventory(at: root)) { error in
            XCTAssertTrue(String(describing: error).contains("coreml/flow_encoder.mlpackage"),
                          String(describing: error))
        }
    }

    /// Enrollment packages are only required when they were asked for:
    /// requiring what the plan did not fetch turns a correct download into an
    /// error naming files the caller was right not to have.
    func testInventoryDoesNotAskForEnrollmentUnlessCloning() throws {
        try release()
        XCTAssertNoThrow(try LoudKit.verifyInventory(at: root))
        XCTAssertThrowsError(try LoudKit.verifyInventory(at: root, cloning: true))
    }
}
