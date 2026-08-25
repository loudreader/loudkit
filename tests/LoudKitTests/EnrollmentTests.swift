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
            throw XCTSkip("enrollment CoreML packages not found — run tools/export_enroll_coreml.py")
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
}
