import CoreML
import Foundation

/// Enrollment: reference audio to a `VoiceProfile` — a bit-parity port of
/// `loudkit.models.enroll` over the exported enrollment CoreML graphs.
///
/// The DSP (resampler, filterbanks, trim) is implemented here and held to the
/// enrollment fixture; the model stages run through `s3_tokenizer.mlpackage`,
/// `camp.mlpackage` and `voice_encoder.mlpackage`. The filter tables and
/// windows are the same float32 data every port loads (see
/// `tools/gen_dsp_assets.py`).
public enum Enrollment {

    // ------------------------------------------------------------ constants

    static let melSR = 24_000
    static let s3SR = 16_000
    static let maxRefSeconds = 10
    static let condSeconds = 6

    static let frame400 = 400
    static let hop160 = 160
    static let kaldiFFT = 512
    static let matchaNFFT = 1920
    static let matchaHop = 480

    static let partialFrames = 160
    static let partialStep = 77

    // ---------------------------------------------------------------- tables

    private static func table(_ name: String) -> [Float] {
        guard let url = Bundle.module.url(
            forResource: name, withExtension: "f32", subdirectory: "Resources"
        ), let data = try? Data(contentsOf: url) else {
            return []
        }
        // `loadUnaligned`, like `Safetensors.floats`. `Data` from a bundled file
        // gives no four-byte alignment guarantee, and `bindMemory` requires
        // one: it works on ARM and is undefined behaviour by Swift's own rules,
        // because the optimiser is entitled to assume the alignment it was
        // promised. These tables are packaged rather than downloaded, which
        // makes it unlikely to bite and does not make it defined.
        return data.withUnsafeBytes { buf in
            (0..<(buf.count / MemoryLayout<Float>.size)).map {
                buf.loadUnaligned(fromByteOffset: $0 * MemoryLayout<Float>.size, as: Float.self)
            }
        }
    }

    // ------------------------------------------------------------- resampler

    /// The one portable Hann-windowed-sinc resampler, a bit-parity port of
    /// `loudkit.models.resample`: float64 kernel rounded to float32 once, FIR
    /// accumulated left to right in float32 with no fused multiply-add.
    public static func resample(_ waveform: [Float], origFreq: Int, newFreq: Int) -> [Float] {
        if origFreq == newFreq { return waveform }
        let g = gcd(origFreq, newFreq)
        let orig = origFreq / g
        let new = newFreq / g

        let (kernel, width) = sincHannKernel(orig: orig, new: new)
        let taps = kernel[0].count

        var padded = [Float](repeating: 0, count: width + waveform.count + width + orig)
        for (i, v) in waveform.enumerated() { padded[width + i] = v }

        let nOut = (padded.count - taps) / orig + 1
        var out = [Float](repeating: 0, count: nOut * new)
        for i in 0..<nOut {
            let base = i * orig
            for phase in 0..<new {
                var acc: Float = 0
                for c in 0..<taps {
                    acc += kernel[phase][c] * padded[base + c]
                }
                out[i * new + phase] = acc
            }
        }

        let target = (new * waveform.count + orig - 1) / orig
        return Array(out.prefix(target))
    }

    private static func sincHannKernel(orig: Int, new: Int) -> ([[Float]], Int) {
        let base = Double(min(orig, new)) * 0.99
        let width = Int((6.0 * Double(orig) / base).rounded(.up))

        var kernel = [[Float]](repeating: [Float](repeating: 0, count: 2 * width + orig), count: new)
        for phase in 0..<new {
            for idx in 0..<(2 * width + orig) {
                var t = -Double(phase) / Double(new) + Double(idx - width) / Double(orig)
                t *= base
                t = min(6.0, max(-6.0, t))
                let window = pow(cos(t * .pi / 6.0 / 2.0), 2)
                let tt = t * .pi
                let sinc = tt == 0 ? 1.0 : sin(tt) / tt
                kernel[phase][idx] = Float(sinc * window * (base / Double(orig)))
            }
        }
        return (kernel, width)
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var a = a, b = b
        while b != 0 { (a, b) = (b, a % b) }
        return a
    }

    // ------------------------------------------------------------ filterbanks

    private static func basis(_ nfft: Int) -> ([[Double]], [[Double]]) {
        let bins = nfft / 2 + 1
        var cosTable = [[Double]](repeating: [Double](repeating: 0, count: nfft), count: bins)
        var sinTable = [[Double]](repeating: [Double](repeating: 0, count: nfft), count: bins)
        for k in 0..<bins {
            for n in 0..<nfft {
                let a = -2 * .pi * Double(k) * Double(n) / Double(nfft)
                cosTable[k][n] = cos(a)
                sinTable[k][n] = sin(a)
            }
        }
        return (cosTable, sinTable)
    }

    private static func powerSpectrum(_ frame: [Double], _ nfft: Int) -> [Double] {
        let (cosT, sinT) = basis(nfft)
        let bins = nfft / 2 + 1
        var out = [Double](repeating: 0, count: bins)
        for k in 0..<bins {
            var re = 0.0, im = 0.0
            for n in 0..<nfft {
                re += cosT[k][n] * frame[n]
                im += sinT[k][n] * frame[n]
            }
            out[k] = re * re + im * im
        }
        return out
    }

    private static func melMultiply(_ filters: [Float], _ rows: Int, _ bins: Int,
                                    _ spectra: [[Double]], _ frames: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * frames)
        for r in 0..<rows {
            for f in 0..<frames {
                var acc: Float = 0
                for b in 0..<bins {
                    acc += filters[r * bins + b] * Float(spectra[b][f])
                }
                out[r * frames + f] = acc
            }
        }
        return out
    }

    private static func centredPowerSpectra(_ samples: [Double], _ window: [Float],
                                            dropLast: Bool) -> [[Double]] {
        let nfft = window.count
        let half = nfft / 2
        var padded = [Double](repeating: 0, count: samples.count + nfft)
        for i in 0..<half { padded[i] = samples[half - i] }
        for (i, v) in samples.enumerated() { padded[half + i] = v }
        for i in 0..<half { padded[half + samples.count + i] = samples[samples.count - 2 - i] }

        var frames = samples.count / hop160 + 1
        if dropLast { frames -= 1 }
        let bins = nfft / 2 + 1
        var out = [[Double]](repeating: [Double](repeating: 0, count: frames), count: bins)
        for f in 0..<frames {
            let start = f * hop160
            var frame = [Double](repeating: 0, count: nfft)
            for i in 0..<nfft {
                frame[i] = padded[start + i] * Double(window[i])
            }
            let sp = powerSpectrum(frame, nfft)
            for k in 0..<bins { out[k][f] = sp[k] }
        }
        return out
    }

    private static func tokenizerMel(_ samples: [Double]) -> (values: [Float], frames: Int) {
        let s3hann = table("s3_hann400")
        let s3mel = table("s3_mel_filters")
        let spectra = centredPowerSpectra(samples, s3hann, dropLast: true)
        let frames = spectra[0].count
        var mel = melMultiply(s3mel, 128, 201, spectra, frames)

        var peak: Float = -.infinity
        for v in mel {
            let x = Float(log10(Double(max(v, 1e-10))))
            if x > peak { peak = x }
        }
        let ceiling = peak - 8
        for i in 0..<mel.count {
            var v = Float(log10(Double(max(mel[i], 1e-10))))
            if v < ceiling { v = ceiling }
            mel[i] = (v + 4) * 0.25
        }
        return (mel, frames)
    }

    private static func matchaMel(_ samples: [Double]) -> [Float] {
        let matchaHann = table("matcha_hann1920")
        let matchaMelF = table("matcha_mel_filters")
        let pad = (matchaNFFT - matchaHop) / 2
        var padded = [Double](repeating: 0, count: samples.count + 2 * pad)
        for i in 0..<pad { padded[i] = samples[pad - i] }
        for (i, v) in samples.enumerated() { padded[pad + i] = v }
        for i in 0..<pad { padded[pad + samples.count + i] = samples[samples.count - 2 - i] }

        let frames = (padded.count - matchaNFFT) / matchaHop + 1
        let bins = matchaNFFT / 2 + 1
        var spectra = [[Double]](repeating: [Double](repeating: 0, count: frames), count: bins)
        for f in 0..<frames {
            let start = f * matchaHop
            var frame = [Double](repeating: 0, count: matchaNFFT)
            for i in 0..<matchaNFFT {
                frame[i] = padded[start + i] * Double(matchaHann[i])
            }
            var sp = powerSpectrum(frame, matchaNFFT)
            for i in 0..<sp.count { sp[i] = (sp[i] + 1e-9).squareRoot() }
            for k in 0..<bins { spectra[k][f] = sp[k] }
        }

        var mel = melMultiply(matchaMelF, 80, bins, spectra, frames)
        for i in 0..<mel.count { mel[i] = Float(log(Double(max(mel[i], 1e-5)))) }
        return mel
    }

    private static func kaldiFbank(_ samples: [Double]) -> [Float] {
        let kaldiMel = table("kaldi_mel_filters")
        let kaldiPovey = table("kaldi_povey400")
        let frames = (samples.count - frame400) / hop160 + 1
        let bins = kaldiFFT / 2 + 1
        var spectra = [[Double]](repeating: [Double](repeating: 0, count: frames), count: bins)

        for f in 0..<frames {
            let start = f * hop160
            var frame = [Double](repeating: 0, count: kaldiFFT)
            var mean = 0.0
            for i in 0..<frame400 {
                frame[i] = samples[start + i]
                mean += frame[i]
            }
            mean /= Double(frame400)
            for i in 0..<frame400 { frame[i] -= mean }
            let prev = frame[0]
            for i in stride(from: frame400 - 1, through: 1, by: -1) {
                frame[i] -= 0.97 * frame[i - 1]
            }
            frame[0] -= 0.97 * prev
            for i in 0..<frame400 { frame[i] *= Double(kaldiPovey[i]) }
            let sp = powerSpectrum(frame, kaldiFFT)
            for k in 0..<bins { spectra[k][f] = sp[k] }
        }

        var mel = melMultiply(kaldiMel, 80, 256, spectra, frames)
        let epsilon: Float = 1.1920928955078125e-07
        for i in 0..<mel.count { mel[i] = Float(log(Double(max(mel[i], epsilon)))) }
        for b in 0..<80 {
            var m = 0.0
            for f in 0..<frames { m += Double(mel[b * frames + f]) }
            m /= Double(frames)
            for f in 0..<frames { mel[b * frames + f] -= Float(m) }
        }
        var out = [Float](repeating: 0, count: mel.count)
        for f in 0..<frames {
            for b in 0..<80 { out[f * 80 + b] = mel[b * frames + f] }
        }
        return out
    }

    private static func voiceEncoderMel(_ samples: [Double]) -> (values: [Float], frames: Int) {
        let veHann = table("voiceenc_hann400")
        let veMel = table("voiceenc_mel_filters")
        let spectra = centredPowerSpectra(samples, veHann, dropLast: false)
        let frames = spectra[0].count
        let binMajor = melMultiply(veMel, 40, 201, spectra, frames)
        var out = [Float](repeating: 0, count: frames * 40)
        for f in 0..<frames {
            for b in 0..<40 { out[f * 40 + b] = binMajor[b * frames + f] }
        }
        return (out, frames)
    }

    private static func trim(_ samples: [Double]) -> [Double] {
        let frame = 2048, hop = 512
        let half = frame / 2
        var padded = [Double](repeating: 0, count: samples.count + frame)
        for i in 0..<half { padded[i] = samples[half - i] }
        for (i, v) in samples.enumerated() { padded[half + i] = v }
        for i in 0..<half { padded[half + samples.count + i] = samples[samples.count - 2 - i] }

        let nFrames = 1 + samples.count / hop
        var rms = [Double](repeating: 0, count: nFrames)
        var peak = 0.0
        for f in 0..<nFrames {
            let start = f * hop
            var sum = 0.0
            for i in start..<(start + frame) { sum += padded[i] * padded[i] }
            let r = (sum / Double(frame)).squareRoot()
            rms[f] = r
            if r > peak { peak = r }
        }
        var first = -1, last = -1
        for (f, r) in rms.enumerated() where r > 0.1 * peak {
            if first == -1 { first = f }
            last = f
        }
        guard first != -1 else { return samples }
        let start = first * hop
        let end = min(last * hop + hop, samples.count)
        return end <= start ? samples : Array(samples[start..<end])
    }

    // --------------------------------------------------------------- enroller

    /// An enrollment pipeline over the three exported graphs.
    public final class Enroller {
        private let tokenizer: MLModel
        private let camp: MLModel
        private let ve: MLModel

        public init(coremlDir: URL) throws {
            tokenizer = try MLHelpers.loadModel(
                packageURL: coremlDir.appendingPathComponent("s3_tokenizer.mlpackage"),
                computeUnits: .cpuOnly)
            camp = try MLHelpers.loadModel(
                packageURL: coremlDir.appendingPathComponent("camp.mlpackage"),
                computeUnits: .cpuOnly)
            ve = try MLHelpers.loadModel(
                packageURL: coremlDir.appendingPathComponent("voice_encoder.mlpackage"),
                computeUnits: .cpuOnly)
        }

        /// An enrolled voice, before wrapping in a `VoiceProfile`.
        public func enroll(_ audio: [Float], sampleRate: Int) throws -> EnrolledVoice {
            // A non-positive rate reaches the resampler as a division by zero, which
            // traps here and kills the process. Go refused it at this point, Python raised
            // from inside a kernel calculation, and this and Rust died. Same sentence as
            // Go's, at the same place.
            guard sampleRate > 0 else {
                throw LoudKitError.shape("sample rate must be positive, got \(sampleRate)")
            }
            let wav = audio.map(Double.init)
            let wav24Full = sampleRate == Enrollment.melSR
                ? wav
                : Enrollment.resample(audio, origFreq: sampleRate, newFreq: Enrollment.melSR).map(Double.init)
            let maxSamples = Enrollment.maxRefSeconds * Enrollment.melSR
            let wav24 = Array(wav24Full.prefix(maxSamples))

            let wav16Flow = Enrollment.resample(wav24.map(Float.init), origFreq: Enrollment.melSR,
                                                newFreq: Enrollment.s3SR).map(Double.init)
            let wav16T3 = Enrollment.resample(wav24Full.map(Float.init), origFreq: Enrollment.melSR,
                                              newFreq: Enrollment.s3SR).map(Double.init)

            let promptMel = Enrollment.matchaMel(wav24)
            let promptMelFrames = promptMel.count / 80

            let (tokMel, _) = Enrollment.tokenizerMel(wav16Flow)
            let tokens = try tokenize(tokMel)
            let nTok = min(tokens.count, promptMelFrames / 2)
            let promptTokens = Array(tokens.prefix(nTok))
            let promptMelOut = Array(promptMel.prefix(80 * 2 * nTok))

            let condSamples = min(Enrollment.condSeconds * Enrollment.s3SR, wav16T3.count)
            let (condMel, _) = Enrollment.tokenizerMel(Array(wav16T3.prefix(condSamples)))
            let condTokens = try tokenizeCapped(condMel, cap: 150)

            let fbank = Enrollment.kaldiFbank(wav16Flow)
            let flowEmbedding = try camEmbedding(fbank)

            let speakerEmbedding = try speakerEmbedding(wav16T3)

            return EnrolledVoice(
                speakerEmbedding: speakerEmbedding,
                flowEmbedding: flowEmbedding,
                promptTokens: promptTokens,
                promptMel: promptMelOut,
                promptMelFrames: 2 * nTok,
                condPromptTokens: condTokens)
        }

        private func tokenize(_ mel: [Float]) throws -> [Int] {
            let frames = mel.count / 128
            let input = try MLHelpers.floatArray([1, 128, frames], mel)
            let output = try MLHelpers.predict(tokenizer, ["mel": input], stage: "tokenizer")
            let h = output  // [T, 8] codes, flattened; T = frames / 4
            return Self.foldCodes(h, nFrames: frames / 4)
        }

        private func tokenizeCapped(_ mel: [Float], cap: Int) throws -> [Int] {
            if mel.count / 128 > cap * 4 {
                return try tokenize(Array(mel.prefix(128 * cap * 4)))
            }
            return try tokenize(mel)
        }

        /// The FSQ base-3 fold: 8 dims of {0,1,2} to a token id. Mirrors
        /// `_fold_codes` in the export tool — the graph emits the codes, the
        /// host owns the integer encode.
        static func foldCodes(_ codes: [Float], nFrames: Int) -> [Int] {
            var out = [Int](repeating: 0, count: nFrames)
            for t in 0..<nFrames {
                var token = 0
                var power = 1
                for d in 0..<8 {
                    let code = Int(codes[t * 8 + d].rounded())
                    token += code * power
                    power *= 3
                }
                out[t] = token
            }
            return out
        }

        private func camEmbedding(_ fbank: [Float]) throws -> [Float] {
            let frames = fbank.count / 80
            var transposed = [Float](repeating: 0, count: fbank.count)
            for f in 0..<frames {
                for b in 0..<80 { transposed[b * frames + f] = fbank[f * 80 + b] }
            }
            let input = try MLHelpers.floatArray([1, 80, frames], transposed)
            return try MLHelpers.predict(camp, ["fbank": input], stage: "camp")
        }

        private func speakerEmbedding(_ wav16T3: [Double]) throws -> [Float] {
            let trimmed = Enrollment.trim(wav16T3)
            let (mel, frames) = Enrollment.voiceEncoderMel(trimmed)

            var nWins = 0
            var rem = 0
            let span = mel.count / 40
            if span > Enrollment.partialFrames - Enrollment.partialStep {
                nWins = (span - Enrollment.partialFrames + Enrollment.partialStep) / Enrollment.partialStep
                rem = (span - Enrollment.partialFrames + Enrollment.partialStep) % Enrollment.partialStep
            }
            if nWins == 0 ||
                Double(rem + (Enrollment.partialFrames - Enrollment.partialStep)) / Double(Enrollment.partialFrames) >= 0.8 {
                nWins += 1
            }
            let target = Enrollment.partialFrames + Enrollment.partialStep * (nWins - 1)
            var melOut = mel
            if target > frames {
                melOut.append(contentsOf: [Float](repeating: 0, count: (target - frames) * 40))
            }

            var partials = [Float](repeating: 0, count: nWins * Enrollment.partialFrames * 40)
            for i in 0..<nWins {
                let start = i * Enrollment.partialStep * 40
                for j in 0..<(Enrollment.partialFrames * 40) {
                    partials[i * Enrollment.partialFrames * 40 + j] = melOut[start + j]
                }
            }

            let input = try MLHelpers.floatArray([nWins, Enrollment.partialFrames, 40], partials)
            let perPartial = try MLHelpers.predict(ve, ["partials": input], stage: "voice_encoder")

            var pooled = [Float](repeating: 0, count: 256)
            for i in 0..<nWins {
                for d in 0..<256 { pooled[d] += perPartial[i * 256 + d] }
            }
            var norm = 0.0
            for v in pooled { norm += Double(v) * Double(v) }
            norm = norm.squareRoot()
            if norm > 0 {
                for i in 0..<pooled.count { pooled[i] = Float(Double(pooled[i]) / norm) }
            }
            return pooled
        }
    }
}

/// An enrolled voice's five tensors, before wrapping in a `VoiceProfile`.
public struct EnrolledVoice {
    public let speakerEmbedding: [Float]
    public let flowEmbedding: [Float]
    public let promptTokens: [Int]
    public let promptMel: [Float]
    public let promptMelFrames: Int
    public let condPromptTokens: [Int]
}
