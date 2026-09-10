import Foundation
import XCTest

@testable import LoudKit

/// What the manifest, container and voice readers accept, refuse, and say.
///
/// Four separate ways this port disagreed with the other four about a file
/// none of them had trouble with: it kept the first of two members with the
/// same name where they keep the last, it refused a manifest that named no
/// window where they default, it let a profile it could not read load with
/// every default, and it reported a raw Foundation error that named neither
/// the file nor what was being read.
final class ReaderDoorTests: XCTestCase {

    // MARK: building files by hand

    /// A safetensors file with one four-byte tensor and the given metadata,
    /// so a manifest no packer would write can be handed to the reader.
    private func pack(manifest: String?, voice: String? = nil,
                      headerOverride: String? = nil) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("door-\(UUID().uuidString).safetensors")
        var header: String
        if let headerOverride {
            header = headerOverride
        } else {
            var metadata: [String] = []
            if let manifest { metadata.append("\"manifest\":\(quoted(manifest))") }
            if let voice { metadata.append("\"voice\":\(quoted(voice))") }
            header = "{\"__metadata__\":{\(metadata.joined(separator: ","))},"
                + "\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}"
        }
        let headerData = Data(header.utf8)
        var out = withUnsafeBytes(of: UInt64(headerData.count).littleEndian) { Data($0) }
        out.append(headerData)
        out.append(Data(repeating: 0, count: 4))
        try out.write(to: url)
        return url
    }

    /// A JSON string literal holding `text`, written by Foundation so the
    /// escaping is not this test's opinion.
    private func quoted(_ text: String) -> String {
        let data = try! JSONSerialization.data(
            withJSONObject: text, options: [.fragmentsAllowed])
        return String(decoding: data, as: UTF8.self)
    }

    /// The smallest manifest `Checkpoint` will open, as raw text so a
    /// duplicate member survives into the file.
    private func manifestText(_ body: String) -> String {
        "{\"format\":\"loudkit-checkpoint\",\"format_version\":1,"
            + "\"recipe_version\":\"loudkit-1\"\(body.isEmpty ? "" : ",")\(body)}"
    }

    private func fingerprint(_ body: String) throws -> String {
        let url = try pack(manifest: manifestText(body))
        defer { try? FileManager.default.removeItem(at: url) }
        return try Checkpoint(url: url).algorithm().fingerprint()
    }

    // MARK: duplicate members

    /// `json.loads`, `encoding/json`, `serde_json` and `JSON.parse` all keep
    /// the last member with a given name. Foundation keeps the first, and has
    /// no option to do otherwise, so the reader drops the earlier member
    /// before Foundation sees the bytes.
    func testTheLastOfTwoMembersWins() throws {
        XCTAssertEqual(try fingerprint("\"sample_rate\":24000,\"sample_rate\":48000"),
                       try fingerprint("\"sample_rate\":48000"))
        XCTAssertEqual(try fingerprint("\"n_cfm_timesteps\":10,\"n_cfm_timesteps\":4"),
                       try fingerprint("\"n_cfm_timesteps\":4"))
        // Three deep, and the last still wins.
        XCTAssertEqual(try fingerprint("\"sample_rate\":1,\"sample_rate\":2,"
                                       + "\"sample_rate\":48000"),
                       try fingerprint("\"sample_rate\":48000"))
    }

    /// Nested objects too: the scan is per object, not per document.
    func testTheLastOfTwoNestedMembersWins() throws {
        XCTAssertEqual(
            try fingerprint("\"chunking\":{\"max_tokens\":200,\"max_tokens\":100}"),
            try fingerprint("\"chunking\":{\"max_tokens\":100}"))
    }

    /// A `{`, a `,` or a `"` inside a string is text, not structure, and a
    /// document with no duplicate must come through untouched.
    func testStringsAndArraysAreNotMistakenForStructure() throws {
        let tricky = "\"chunking\":{\"split_on\":[\"\\\", \",\"} {\",\"a\\\\\"],"
            + "\"abbreviations\":[\"Dr\"]}"
        XCTAssertEqual(try fingerprint(tricky), try fingerprint(tricky))
        let config = try AlgorithmConfig.fromManifest(
            try XCTUnwrap(ManifestReader.json(
                Data(manifestText(tricky).utf8), "probe") as? [String: Any]))
        XCTAssertEqual(config.chunking.splitOn, ["\", ", "} {", "a\\"])
    }

    // MARK: the window door

    /// A manifest naming no window, or naming it null, is the ragged window in
    /// every other port. This one used to refuse it and send the caller to
    /// tools/amend_manifest.py, so the five disagreed about which checkpoints
    /// load at all.
    func testAnAbsentOrNullWindowIsTheRaggedDefault() throws {
        // `algorithm_from({})`, measured. Not `AlgorithmConfig().fingerprint()`:
        // a manifest that never names `edge_fade_seconds` gets the 5 ms those
        // releases shipped rather than today's field default, which is the one
        // deliberate exception `algorithm_from` documents.
        let ragged = "2f7468a9a48fa2ab"
        XCTAssertEqual(try fingerprint(""), ragged)
        XCTAssertEqual(try fingerprint("\"window\":null"), ragged)
        XCTAssertEqual(try fingerprint("\"eos_floor\":{\"min_tokens_floor\":0}"), ragged)
        XCTAssertEqual(
            try fingerprint("\"window\":null,\"eos_floor\":{\"min_tokens_floor\":0}"),
            ragged)

        // And it is the ragged window, not merely a hash that agrees.
        let url = try pack(manifest: manifestText("\"window\":null"))
        defer { try? FileManager.default.removeItem(at: url) }
        let window = try Checkpoint(url: url).algorithm().window
        XCTAssertEqual(window.maxSpeechTokens, 255)
        XCTAssertNil(window.staticLength)
        XCTAssertNil(window.padTokenId)
        XCTAssertNil(window.staticPromptTokens)
    }

    /// Null is the only non-object reading. Anything else is still refused,
    /// exactly as `_window_from` refuses it.
    func testAWindowThatIsNeitherObjectNorNullIsStillRefused() throws {
        for bad in ["[1,2,3]", "\"ragged\"", "3", "true"] {
            XCTAssertThrowsError(try fingerprint("\"window\":\(bad)"),
                                 "window: \(bad) loaded") { error in
                XCTAssertTrue("\(error)".contains("must be a mapping"),
                              "refused for the wrong reason: \(error)")
            }
        }
    }

    // MARK: the guidance door

    /// Refused where the fingerprint is computed, not at the first Euler step:
    /// a hash published for an algorithm the port cannot run is worse than no
    /// hash. Go, Rust and JS all refuse it in their manifest readers.
    func testCfgDualPathIsRefusedAtTheManifest() throws {
        for body in ["\"guidance\":\"cfg_dual_path\"",
                     "\"guidance\":\"cfg_dual_path\",\"guidance_rate\":0.7"] {
            XCTAssertThrowsError(try fingerprint(body), "\(body) loaded") { error in
                XCTAssertTrue("\(error)".contains("does not implement"),
                              "refused for the wrong reason: \(error)")
            }
        }
        XCTAssertNoThrow(try fingerprint("\"guidance\":\"single_path\""))
    }

    // MARK: what the refusal says

    /// A message that names neither the file nor what was being read is not a
    /// diagnosis. Go says `%s: bad manifest JSON: %w` and Rust
    /// `{path}: bad manifest JSON: {e}`.
    func testAnUnreadableDocumentIsNamed() throws {
        let cases: [(String, URL, String)] = [
            ("zero-length header", try pack(manifest: nil, headerOverride: ""),
             "bad header JSON"),
            ("truncated header",
             try pack(manifest: nil, headerOverride: "{\"a\":"), "bad header JSON"),
            ("header is an array",
             try pack(manifest: nil, headerOverride: "[1,2,3]"), "header is an array"),
            ("unparseable manifest", try pack(manifest: "{oops"), "bad manifest JSON"),
            ("manifest is an array", try pack(manifest: "[1,2]"), "manifest is an array"),
        ]
        for (what, url, fragment) in cases {
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try Checkpoint(url: url), "\(what) loaded") { error in
                let message = "\(error)"
                XCTAssertTrue(message.contains(fragment),
                              "\(what): said \(message)")
                XCTAssertTrue(message.contains(url.lastPathComponent),
                              "\(what) does not name the file: \(message)")
                XCTAssertFalse(message.contains("NSCocoaErrorDomain"),
                               "\(what) is a raw Foundation error: \(message)")
            }
        }
    }

    // MARK: the voice header

    private func voice(_ header: String?) throws -> URL {
        try pack(manifest: nil, voice: header)
    }

    /// The version guard used to sit inside the header read, so a profile with
    /// no `voice` metadata, or with `voice` holding an array or a string,
    /// skipped it and loaded with every default: the filename as the name,
    /// English, 24 kHz. The reference reads `meta.get("voice", "{}")` and then
    /// `int(header.get("format_version", 0))`, so an absent header is version
    /// zero and refused.
    func testAProfileWithNoReadableHeaderIsRefused() throws {
        for (what, header) in [("absent", nil as String?), ("empty object", "{}"),
                               ("array", "[1,2,3]"), ("string", "\"hello\""),
                               ("number", "42"), ("null", "null"),
                               ("not JSON", "{oops")] {
            let url = try voice(header)
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try VoiceProfile.load(url: url),
                                 "a \(what) voice header loaded") { error in
                XCTAssertTrue("\(error)".contains(url.lastPathComponent),
                              "\(what) does not name the file: \(error)")
                XCTAssertFalse("\(error)".contains("NSCocoaErrorDomain"),
                               "\(what) is a raw Foundation error: \(error)")
            }
        }
    }

    /// How the prompt was cut decides the timbre, so a strategy this build
    /// does not implement is refused rather than cut this build's way. Only
    /// the reference and Go used to say so.
    func testAnUnknownEnrolmentLawIsRefused() throws {
        for law in ["first-30s-nonsense", "", "FIRST-10S", "first-10s "] {
            let url = try voice(
                "{\"format_version\":1,\"name\":\"ref\",\"enrolment\":\"\(law)\"}")
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try VoiceProfile.load(url: url),
                                 "enrolment \(law) loaded") { error in
                XCTAssertTrue(
                    "\(error)".contains("would speak in a different voice under the same"),
                    "refused for the wrong reason: \(error)")
            }
        }
    }

    /// The two laws this build does implement still have to get past the
    /// guard, and they fail later, on the tensors this hand-built file lacks.
    func testTheKnownEnrolmentLawsPassTheGuard() throws {
        for law in ["first-10s", "first-10s-pause"] {
            let url = try voice(
                "{\"format_version\":1,\"name\":\"ref\",\"enrolment\":\"\(law)\"}")
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try VoiceProfile.load(url: url)) { error in
                XCTAssertFalse("\(error)".contains("enrolment strategy"),
                               "\(law) was refused as unknown: \(error)")
            }
        }
    }

    /// A header value of the wrong JSON type is refused by name, not coerced.
    ///
    /// `as? NSNumber` is true of a JSON boolean and its `intValue` is 1, so
    /// `format_version: true` loaded here as a version-1 voice, and
    /// `language: 5` took the English default. Language selects the text
    /// funnel, so a voice whose header says 5 was read through whichever
    /// funnel the fallback landed on.
    func testAHeaderValueOfTheWrongTypeIsRefused() throws {
        let cases: [(String, String)] = [
            ("{\"format_version\":1,\"language\":5}",
             "voice header 'language' must be a string, got 5"),
            ("{\"format_version\":1,\"language\":null}",
             "voice header 'language' must be a string, got None"),
            ("{\"format_version\":1,\"language\":true}",
             "voice header 'language' must be a string, got True"),
            ("{\"format_version\":1,\"language\":[\"en\"]}",
             "voice header 'language' must be a string, got ['en']"),
            ("{\"format_version\":1,\"name\":5}",
             "voice header 'name' must be a string, got 5"),
            ("{\"format_version\":1,\"enrolment\":5}",
             "voice header 'enrolment' must be a string, got 5"),
            ("{\"format_version\":true}",
             "voice header 'format_version' must be a number, got True"),
            ("{\"format_version\":\"1\"}",
             "voice header 'format_version' must be a number, got '1'"),
            ("{\"format_version\":1.9}",
             "voice header 'format_version' must be a whole number, got 1.9"),
            ("{\"format_version\":1,\"source_sample_rate\":\"24000\"}",
             "voice header 'source_sample_rate' must be a number, got '24000'"),
            ("{\"format_version\":1,\"source_sample_rate\":24000.5}",
             "voice header 'source_sample_rate' must be a whole number, got 24000.5"),
        ]
        for (header, want) in cases {
            let url = try voice(header)
            defer { try? FileManager.default.removeItem(at: url) }
            XCTAssertThrowsError(try VoiceProfile.load(url: url),
                                 "\(header) loaded") { error in
                XCTAssertTrue("\(error)".contains(want),
                              "\(header) refused as \(error), want \(want)")
            }
        }
    }
}
