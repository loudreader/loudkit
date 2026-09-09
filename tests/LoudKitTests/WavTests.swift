import Foundation
import XCTest

@testable import LoudKit

/// The 16-bit writer, against the bytes Python writes.
///
/// The header and the quantisation come from the shared fixtures, so the file
/// a Swift user gets is the file the Python server returns for the same
/// floats. A round trip alone would pass for a writer with a private format.
final class WavTests: XCTestCase {

    /// A probe is a number, or the string "nan", which JSON cannot spell.
    private func probe(_ value: Any) -> Float {
        if let s = value as? String, s == "nan" { return .nan }
        return (value as! NSNumber).floatValue
    }

    func testQuantisationIsTheSharedRule() throws {
        let cases = try Fixture.shared("wav_quantise.json")["cases"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(cases.count, 10, "the fixture holds probes")
        for c in cases {
            let x = probe(c["x"]!)
            XCTAssertEqual(Wav.pcm16(x), (c["pcm16"] as! NSNumber).int16Value, "\(c["x"]!)")
        }
        // The two probes a fixture of round numbers cannot tell apart: floor,
        // not round and not truncate.
        XCTAssertEqual(Wav.pcm16(-0.00001), -1)
        XCTAssertEqual(Wav.pcm16(.infinity), 32767)
        XCTAssertEqual(Wav.pcm16(-.infinity), -32768)
    }

    func testTheWavBytesAreTheFixtures() throws {
        let cases = try Fixture.shared("wav_header.json")["cases"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(cases.count, 3, "the fixture holds cases")
        for c in cases {
            let samples = (c["samples"] as! [Any]).map(probe)
            let rate = (c["sample_rate"] as! NSNumber).intValue
            let data = try Wav.data(samples, sampleRate: rate)
            XCTAssertEqual(data.map { String(format: "%02x", $0) }.joined(), c["hex"] as! String,
                           "\(samples.count) samples at \(rate) Hz")
        }
    }

    /// The writer and the reader are the two halves of "clone from a file", so
    /// they are checked against each other: what `AudioFile.read` hands the
    /// enroller must be the samples that were written, to within the one LSB
    /// the 16-bit container costs.
    func testAudioFileReadsBackWhatTheWriterWrote() throws {
        let samples = (0..<1_000).map { Float(sin(Double($0) * 0.05)) * 0.8 }
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("loudkit-roundtrip-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Wav.data(samples, sampleRate: 16_000).write(to: url)

        let (read, rate) = try AudioFile.read(url)
        XCTAssertEqual(rate, 16_000)
        XCTAssertEqual(read.count, samples.count)
        for (i, expected) in samples.enumerated() {
            XCTAssertEqual(read[i], expected, accuracy: 1.0 / 32_768,
                           "sample \(i) came back changed")
        }
    }

    func testAudioFileRefusesWhatIsNotAudio() throws {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("loudkit-notaudio-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("this is not a RIFF file".utf8).write(to: url)
        XCTAssertThrowsError(try AudioFile.read(url))
    }
}
