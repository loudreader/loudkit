import Foundation

/// 16-bit PCM WAV, which is what everything else opens.
///
/// Foundation rather than `AVAudioFile`: the quantisation rule below has to be
/// this one and not an encoder's opinion of it, so the file a Swift user gets
/// is the file the Python server returns for the same floats.
public enum Wav {

    /// Float samples to int16, by the rule `loudkit.synthesis._quantise` uses:
    /// `floor(x * 32768)`, clipped to the int16 range.
    ///
    /// Chosen rather than invented, it is bit for bit what libsndfile's WAV
    /// writer does, so this file and the Python server's carry the same
    /// samples. Clipping rather than scaling by 32767: the engine's contract is
    /// samples in [-1, 1], and the positive rail being one code short of the
    /// negative one is not a reason to move every other sample.
    /// A NaN sample becomes silence rather than a trap: `Int16(nan)` kills the
    /// process, and a single bad sample is not worth a crash in a file writer.
    /// An infinity clips to the rail, which is what Python's clip does with it.
    public static func pcm16(_ sample: Float) -> Int16 {
        let scaled = (Double(sample) * 32768.0).rounded(.down)
        guard !scaled.isNaN else { return 0 }
        return Int16(min(32767.0, max(-32768.0, scaled)))
    }

    /// A complete 16-bit PCM WAV file: 44-byte canonical header, mono.
    public static func data(_ samples: [Float], sampleRate: Int) throws -> Data {
        let byteCount = samples.count * 2
        // RIFF carries its sizes in 32 bits. `UInt32(...)` on a larger value
        // does not truncate in Swift, it traps: the process dies with no
        // message and no file written.
        guard 36 + byteCount <= Int(UInt32.max) else {
            throw LoudKitError.shape(
                "\(samples.count) samples is \(byteCount) bytes of audio; a WAV header "
                + "cannot describe more than \(UInt32.max): split the passage and write "
                + "several files")
        }
        var data = Data(capacity: 44 + byteCount)
        func append<T>(_ value: T) {
            var v = value
            withUnsafeBytes(of: &v) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + byteCount).littleEndian)
        data.append(contentsOf: Array("WAVE".utf8))
        data.append(contentsOf: Array("fmt ".utf8))
        append(UInt32(16).littleEndian)
        append(UInt16(1).littleEndian)  // PCM
        append(UInt16(1).littleEndian)  // mono
        append(UInt32(sampleRate).littleEndian)
        append(UInt32(sampleRate * 2).littleEndian)  // byte rate
        append(UInt16(2).littleEndian)  // block align
        append(UInt16(16).littleEndian)
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(byteCount).littleEndian)
        for sample in samples { append(pcm16(sample).littleEndian) }
        return data
    }
}

extension Engine.Result {
    /// This result as a 16-bit PCM WAV, in memory.
    public func wavData() throws -> Data {
        try Wav.data(audio, sampleRate: sampleRate)
    }

    /// Write a 16-bit PCM WAV, the file to hand a person or a player.
    /// `saveFloat32Wav(to:)` writes the same audio as float32, for the
    /// conformance harness.
    public func saveWav(to url: URL) throws {
        try wavData().write(to: url)
    }

    /// `saveWav(to:)` with a path.
    public func saveWav(_ path: String) throws {
        try saveWav(to: URL(fileURLWithPath: path))
    }
}
