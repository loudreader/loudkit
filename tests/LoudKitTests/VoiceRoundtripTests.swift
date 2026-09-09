import Foundation
import XCTest
@testable import LoudKit

final class VoiceRoundtripTests: XCTestCase {
    func testLegacyPythonProfileAndPauseEnrolmentRoundtrip() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("data/enrollment/profile.safetensors")
        var profile = try VoiceProfile.load(url: source)
        XCTAssertEqual(profile.enrolment, "first-10s")
        for strategy in ["first-10s", "first-10s-pause"] {
            profile.enrolment = strategy
            let path = FileManager.default.temporaryDirectory
                .appendingPathComponent("loudkit-voice-roundtrip-\(UUID().uuidString).safetensors")
            defer { try? FileManager.default.removeItem(at: path) }
            try profile.save(to: path)
            let out = try VoiceProfile.load(url: path)
            XCTAssertEqual(out.enrolment, strategy)
            XCTAssertEqual(out.speakerEmbedding, profile.speakerEmbedding)
            XCTAssertEqual(out.flowEmbedding, profile.flowEmbedding)
            XCTAssertEqual(out.promptTokens, profile.promptTokens)
            XCTAssertEqual(out.promptMel, profile.promptMel)
            XCTAssertEqual(out.condPromptTokens, profile.condPromptTokens)
        }
    }
}
