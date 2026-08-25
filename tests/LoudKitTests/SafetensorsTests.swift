import Foundation
import XCTest

@testable import LoudKit

/// The safetensors reader against files that are not safetensors.
///
/// This parser is the boundary between a checkpoint or voice profile — a file
/// that gets copied, mailed and downloaded — and `cblas_sgemm`, which reads
/// exactly what it is told to read with no bounds check anywhere on the path.
/// The header was trusted about how many elements a tensor holds, so a profile
/// declaring `shape: [256]` over four bytes of payload was not a bad tensor but
/// an out-of-bounds read in C.
///
/// Every case here is a header a corrupt or hostile file can contain.
final class SafetensorsTests: XCTestCase {
    /// Build a safetensors file byte by byte from a header and a payload.
    private func write(header: [String: Any], payload: Data) throws -> URL {
        let json = try JSONSerialization.data(withJSONObject: header)
        var out = Data()
        withUnsafeBytes(of: UInt64(json.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(json)
        out.append(payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("st-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }

    private func entry(_ dtype: String, _ shape: [Int], _ from: Int, _ to: Int) -> [String: Any] {
        ["dtype": dtype, "shape": shape, "data_offsets": [from, to]]
    }

    func testWellFormedFileLoads() throws {
        let payload = Data(repeating: 0, count: 16)
        let url = try write(header: ["x": entry("F32", [4], 0, 16)], payload: payload)
        defer { try? FileManager.default.removeItem(at: url) }

        let st = try Safetensors(url: url)
        XCTAssertEqual(try st.floats("x").count, 4)
    }

    func testShapeMustAccountForTheBytes() throws {
        // 256 floats declared, 16 bytes present: the value a caller sizes a
        // matrix multiply from, against the memory that actually exists.
        let url = try write(header: ["x": entry("F32", [256], 0, 16)], payload: Data(count: 16))
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try Safetensors(url: url)) { error in
            XCTAssertTrue(
                "\(error)".contains("does not describe the payload"),
                "unhelpful error: \(error)")
        }
    }

    func testNegativeDimensionIsRefused() throws {
        let url = try write(header: ["x": entry("F32", [-4], 0, 16)], payload: Data(count: 16))
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try Safetensors(url: url))
    }

    /// `[1<<40, 1<<40]` multiplies to something small and plausible if the
    /// product is allowed to wrap.
    func testOverflowingShapeIsRefused() throws {
        let huge = 1 << 40
        let url = try write(
            header: ["x": entry("F32", [huge, huge, huge], 0, 16)], payload: Data(count: 16))
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try Safetensors(url: url))
    }

    func testUnknownDtypeIsRefusedAtLoad() throws {
        let url = try write(header: ["x": entry("F8_E4M3", [16], 0, 16)], payload: Data(count: 16))
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try Safetensors(url: url)) { error in
            XCTAssertTrue("\(error)".contains("dtype"), "unhelpful error: \(error)")
        }
    }

    /// `Int.max` in `data_offsets` overflowed the addition of `payloadOffset`,
    /// and Swift traps on integer overflow — so the process died inside the
    /// bounds check written to refuse exactly this. The same shape as the
    /// `headerLen` conversion above it, which had already been fixed for the
    /// same reason; the offsets were still doing it.
    func testOffsetsThatWouldOverflowAreRefused() throws {
        for offsets in [(Int.max, Int.max), (0, Int.max), (Int.max, 0)] {
            let url = try write(
                header: ["x": entry("F32", [4], offsets.0, offsets.1)],
                payload: Data(count: 16))
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try Safetensors(url: url), "offsets \(offsets) did not throw")
        }
    }

    func testOffsetsPastTheEndAreRefused() throws {
        let url = try write(header: ["x": entry("F32", [1024], 0, 4096)], payload: Data(count: 16))
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try Safetensors(url: url))
    }

    func testHeaderLongerThanTheFileIsRefused() throws {
        var out = Data()
        withUnsafeBytes(of: UInt64(1 << 40).littleEndian) { out.append(contentsOf: $0) }
        out.append(Data(count: 8))
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("st-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        XCTAssertThrowsError(try Safetensors(url: url)) { error in
            XCTAssertTrue("\(error)".contains("overruns"), "unhelpful error: \(error)")
        }
    }
}

/// A header length no `Int` can hold.
///
/// `payloadOffset = 8 + Int(headerLen)` converted a `UInt64` read straight
/// from the file, and `Int(UInt64)` **traps** for anything >= 2^63 — so a
/// corrupt or hostile file did not reach the "header overruns file" refusal
/// below it; it killed the process with SIGTRAP, inside the check meant to
/// protect against it. Rust, Go and JS all survive the same bytes.
///
/// `testHeaderLongerThanTheFileIsRefused` used `UInt64(1 << 40)`, which is
/// under `Int.max` and therefore converts cleanly — the adjacent case.
final class SafetensorsHugeHeaderTests: XCTestCase {
    private func write(headerLen: UInt64) throws -> URL {
        var bytes = withUnsafeBytes(of: headerLen.littleEndian) { Data($0) }
        bytes.append(contentsOf: [0x7b, 0x7d])  // "{}"
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("huge-\(headerLen).safetensors")
        try bytes.write(to: url)
        return url
    }

    func testAHeaderLengthBeyondIntMaxIsRefusedRatherThanTrapping() throws {
        for headerLen: UInt64 in [0x1000_0000_0000, 0x8000_0000_0000_0000, UInt64.max] {
            let url = try write(headerLen: headerLen)
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try Safetensors(url: url), "headerLen \(headerLen)") { error in
                XCTAssertTrue(
                    "\(error)".contains("overruns"),
                    "expected the documented refusal, got \(error)")
            }
        }
    }
}

/// A voice profile is data from an untrusted file, and this module says so.
///
/// Swift checked `format_version` and that the mel had 80 rows. Python, Rust,
/// Go and JS have all checked embedding widths, finiteness, a minimum norm,
/// negative token ids since the degenerate-profile fix;
/// Rust and Go carry a comment saying "the ports accepted anything shaped like
/// floats". Swift was still in that state — and `Renderer` divides by that zero
/// norm and indexes `spkWeight[r * k + c]` with `k = emb.count`, which for a
/// wrong-width embedding runs past the array and traps.
final class VoiceProfileValidationTests: XCTestCase {
    /// Writes a safetensors file by hand, so a profile no enroller would
    /// produce can be handed to the loader.
    private func writeProfile(
        speaker: [Float], flow: [Float], promptTokens: [Int64], mel: [Float],
        melFrames: Int, language: String? = nil
    ) throws -> URL {
        var offset = 0
        var header: [String: Any] = [:]
        var payload = Data()

        func add(_ name: String, _ floats: [Float], _ shape: [Int]) {
            let bytes = floats.withUnsafeBufferPointer { Data(buffer: $0) }
            header[name] = ["dtype": "F32", "shape": shape,
                            "data_offsets": [offset, offset + bytes.count]]
            payload.append(bytes)
            offset += bytes.count
        }
        func addInts(_ name: String, _ ints: [Int64], _ shape: [Int]) {
            let bytes = ints.withUnsafeBufferPointer { Data(buffer: $0) }
            header[name] = ["dtype": "I64", "shape": shape,
                            "data_offsets": [offset, offset + bytes.count]]
            payload.append(bytes)
            offset += bytes.count
        }

        add("speaker_embedding", speaker, [speaker.count])
        add("flow_embedding", flow, [flow.count])
        addInts("prompt_tokens", promptTokens, [promptTokens.count])
        add("prompt_mel", mel, [80, melFrames])
        addInts("cond_prompt_tokens", promptTokens, [promptTokens.count])
        // A nil language writes a header with **no** language key at all, which
        // is the shape every profile file written before this port read the
        // field back has. That branch is the one real profiles take, so it must
        // stay reachable from a test: writing the key unconditionally left
        // `?? "en"` in VoiceProfile.load with no coverage.
        let languageField = language.map { #", "language": "\#($0)""# } ?? ""
        header["__metadata__"] = [
            "voice": #"{"format_version": 1, "name": "evil"\#(languageField)}"#
        ]

        let headerData = try JSONSerialization.data(withJSONObject: header)
        var out = withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { Data($0) }
        out.append(headerData)
        out.append(payload)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("evil-\(UUID().uuidString).safetensors")
        try out.write(to: url)
        return url
    }

    private func expectRefusal(
        _ what: String, speaker: [Float]? = nil, flow: [Float]? = nil,
        promptTokens: [Int64]? = nil, melNaN: Bool = false
    ) throws {
        let mel = melNaN
            ? [Float](repeating: .nan, count: 80 * 4)
            : [Float](repeating: 0.1, count: 80 * 4)
        let url = try writeProfile(
            speaker: speaker ?? [Float](repeating: 0.0625, count: 256),
            flow: flow ?? [Float](repeating: 0.0625, count: 192),
            promptTokens: promptTokens ?? [1, 2, 3],
            mel: mel, melFrames: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertThrowsError(try VoiceProfile.load(url: url), what)
    }

    func testAnHonestProfileStillLoads() throws {
        let url = try writeProfile(
            speaker: [Float](repeating: 0.0625, count: 256),
            flow: [Float](repeating: 0.0625, count: 192),
            promptTokens: [1, 2, 3],
            mel: [Float](repeating: 0.1, count: 80 * 4), melFrames: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        let profile = try VoiceProfile.load(url: url)
        XCTAssertEqual(profile.speakerEmbedding.count, 256)
        XCTAssertEqual(profile.promptMelFrames, 4)
    }

    /// The header key Python has always written and this port used to drop.
    ///
    /// Without it every profile arrived as English and the engine's language
    /// chain had nothing to consult, so a Polish voice read Polish text through
    /// the English frontend.
    func testTheHeaderLanguageIsRead() throws {
        let url = try writeProfile(
            speaker: [Float](repeating: 0.0625, count: 256),
            flow: [Float](repeating: 0.0625, count: 192),
            promptTokens: [1, 2, 3],
            mel: [Float](repeating: 0.1, count: 80 * 4), melFrames: 4,
            language: "pl")
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(try VoiceProfile.load(url: url).language, "pl")
    }

    /// A header with no language key at all reads as `"en"`.
    ///
    /// The branch every profile file written before this port read the field
    /// back actually takes, and the reason the language chain does not retrofit
    /// them: they load as English, not as blank, so they inherit nothing.
    /// Matches `loudkit.voice.VoiceProfile.load`.
    func testAHeaderWithoutALanguageReadsAsEnglish() throws {
        let url = try writeProfile(
            speaker: [Float](repeating: 0.0625, count: 256),
            flow: [Float](repeating: 0.0625, count: 192),
            promptTokens: [1, 2, 3],
            mel: [Float](repeating: 0.1, count: 80 * 4), melFrames: 4)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(try VoiceProfile.load(url: url).language, "en")
    }

    func testTheProfilesEveryOtherPortRefuses() throws {
        try expectRefusal("wrong speaker width", speaker: [Float](repeating: 0.5, count: 8))
        try expectRefusal("wrong flow width", flow: [Float](repeating: 0.5, count: 8))
        try expectRefusal("zero speaker norm", speaker: [Float](repeating: 0, count: 256))
        try expectRefusal(
            "non-finite speaker", speaker: [Float](repeating: .nan, count: 256))
        try expectRefusal("negative token id", promptTokens: [-5, 1, 2])
        try expectRefusal("NaN mel", melNaN: true)
    }
}

/// Over-window token sequences are refused, not sliced.
///
/// `stripSpecials` ended in `.prefix(maxSpeechTokens)` and `Renderer.decode`
/// sliced again independently, so the end of a passage simply did not exist
/// while the audio still sounded perfectly fine — silent data loss, noticed
/// only by a listener who knows the text. Python raises (engine.py:466) and
/// Rust, Go and JS all return an error, each with a comment saying so; Swift
/// was the one that still truncated, and this module has no `synthesizeLong`,
/// so a caller handing it a paragraph had nothing raised anywhere.
final class OverWindowRefusalTests: XCTestCase {
    func testAPassageLongerThanTheWindowIsRefused() throws {
        XCTAssertNoThrow(try Windowing.requireFits(255, 255), "an exact fit must pass")
        XCTAssertNoThrow(try Windowing.requireFits(0, 255))

        XCTAssertThrowsError(try Windowing.requireFits(273, 255)) { error in
            let message = "\(error)"
            XCTAssertTrue(
                message.contains("exceed the 255-token window by 18"),
                "the refusal must name the window and the overflow: \(message)")
            XCTAssertTrue(
                message.contains("split the text first"),
                "the refusal must say what to do instead: \(message)")
        }
    }
}
