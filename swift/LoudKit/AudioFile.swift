import AVFoundation
import Foundation

/// Reading a recording off disk, so a clone starts from a file.
///
/// `Enrollment.Enroller.enroll(_:sampleRate:)` takes samples, which is the
/// right shape for a microphone and the wrong one for a person with a voice
/// memo: the reader was theirs to write, in a package whose whole subject is
/// audio. AVFoundation rather than a WAV parser, because it opens CAF, AIFF,
/// m4a and mp3 as well, and a recording made on the phone is a CAF.
public enum AudioFile {

    /// Mono float samples and the file's own sample rate.
    ///
    /// Channels are averaged. The rate is not converted: the enroller resamples
    /// from whatever it is given, by the one portable law all five ports share,
    /// and a second resampler here would be a second answer.
    public static func read(_ url: URL) throws -> (samples: [Float], sampleRate: Int) {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw LoudKitError.asset("\(url.lastPathComponent): cannot be read as audio (\(error))")
        }
        let format = file.processingFormat
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0 else {
            throw LoudKitError.shape("\(url.lastPathComponent): no audio in it")
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw LoudKitError.asset("\(url.lastPathComponent): cannot hold \(frames) frames")
        }
        try file.read(into: buffer)
        guard let channels = buffer.floatChannelData else {
            throw LoudKitError.asset(
                "\(url.lastPathComponent): not float samples after decoding")
        }
        let count = Int(buffer.frameLength)
        let channelCount = Int(format.channelCount)
        var samples = [Float](repeating: 0, count: count)
        for channel in 0..<channelCount {
            let data = channels[channel]
            for i in 0..<count { samples[i] += data[i] }
        }
        if channelCount > 1 {
            let scale = 1 / Float(channelCount)
            for i in 0..<count { samples[i] *= scale }
        }
        return (samples, Int(format.sampleRate.rounded()))
    }
}

extension Enrollment.Enroller {
    /// Clone a voice from a recording on disk.
    ///
    /// Five to ten seconds of clean speech is the input this was tuned for.
    /// Everything past ten seconds is used for the speaker embedding and cut
    /// from the prompt, which holds about 9.5 seconds.
    public func enroll(contentsOf url: URL, name: String = "", language: String = "en")
        throws -> VoiceProfile {
        let (samples, sampleRate) = try AudioFile.read(url)
        let enrolled = try enroll(samples, sampleRate: sampleRate)
        return VoiceProfile(
            enrolled,
            name: name.isEmpty ? url.deletingPathExtension().lastPathComponent : name,
            language: language, sourceSampleRate: sampleRate)
    }
}

extension VoiceProfile {
    /// Wrap an enrolled voice. `language` is what the engine reads a
    /// synthesis as when the caller names none; `sourceSampleRate` is the
    /// recording's own, for provenance.
    public init(_ enrolled: EnrolledVoice, name: String, language: String = "en",
                sourceSampleRate: Int = 24_000) {
        self.init(
            name: name,
            speakerEmbedding: enrolled.speakerEmbedding,
            flowEmbedding: enrolled.flowEmbedding,
            promptTokens: enrolled.promptTokens,
            promptMel: enrolled.promptMel,
            promptMelFrames: enrolled.promptMelFrames,
            condPromptTokens: enrolled.condPromptTokens,
            language: language,
            sourceSampleRate: sourceSampleRate)
    }
}
