// Stage-by-stage probe of the Swift decode loop, for cross-port comparison.
//
// The same stages in the same order as tools/decode_probe/probe.py, so a
// comparison stops at the first stage that disagrees instead of reporting
// noise from everything downstream of a divergence.
//
//   out/probe-swift BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR
//
// Compile against the release LoudKit/LoudKitText objects; see
// tools/decode_probe/swift.sh. Writes OUTDIR/swift.json and
// OUTDIR/swift.<stage>.bin (raw little-endian float32).
//
// This port runs CoreML, not ONNX, so it reads `coreml/` out of the same
// bundle the other four read `onnx/` out of.
import Foundation
import CryptoKit
import LoudKit

let args = CommandLine.arguments
guard args.count == 7 else {
    FileHandle.standardError.write(Data("usage: probe BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR\n".utf8))
    exit(2)
}
let bundlePath = args[1], voicePath = args[2], text = args[3]
guard let seed = UInt64(args[4]) else {
    FileHandle.standardError.write(Data("bad seed\n".utf8))
    exit(2)
}
let language = args[5], outdir = args[6]

func f32Data(_ a: [Float]) -> Data {
    var out = Data(capacity: a.count * 4)
    for x in a { withUnsafeBytes(of: x.bitPattern.littleEndian) { out.append(contentsOf: $0) } }
    return out
}
func shaOf(_ a: [Float]) -> String {
    SHA256.hash(data: f32Data(a)).map { String(format: "%02x", $0) }.joined()
}
func head(_ a: [Float], _ n: Int) -> [Float] { Array(a.prefix(n)) }
func jsonInts(_ a: [Int]) -> String { "[" + a.map(String.init).joined(separator: ",") + "]" }
func jsonFloats(_ a: [Float]) -> String {
    "[" + a.map { String(format: "%.9g", $0) }.joined(separator: ",") + "]"
}
func quote(_ s: String) -> String {
    let escaped = s.replacingOccurrences(of: "\\", with: "\\\\")
        .replacingOccurrences(of: "\"", with: "\\\"")
        .replacingOccurrences(of: "\n", with: "\\n")
    return "\"\(escaped)\""
}

// Which binary actually answered.
FileHandle.standardError.write(Data("swift probe: \(args[0])\n".utf8))

try FileManager.default.createDirectory(
    atPath: outdir, withIntermediateDirectories: true)
let bundle = URL(fileURLWithPath: bundlePath)
let engine = try Engine.load(bundle: bundle)
FileHandle.standardError.write(Data("\(engine.describe())\n".utf8))
let voice = try VoiceProfile.load(url: URL(fileURLWithPath: voicePath))
let cfg = engine.algorithm

// 1. text tokens.
let textTokens = try engine.frontend.encode(text, language: language)

// 3. the token sequence.
let sampler = LRSamplerV1(config: cfg.sampling, seed: seed)
let generation = try engine.tokenGenerator.generate(
    textTokens: textTokens, voice: voice, sampler: sampler)
let raw = generation.rawTokens
let stripped = raw.filter { $0 < cfg.startSpeechToken }

// 4. the mel frames.
let (mel, melFrames) = try engine.melDecoder.decode(
    tokens: stripped, voice: voice, seed: seed)
try f32Data(mel).write(to: URL(fileURLWithPath: outdir).appendingPathComponent("swift.mel.bin"))

// 5. the rendered samples.
let audio = try engine.vocoder.synthesize(mel: mel, frames: melFrames, seed: seed)
try f32Data(audio).write(to: URL(fileURLWithPath: outdir).appendingPathComponent("swift.audio.bin"))

// 6. the long-form path.
let out = try engine.synthesize(text, voice: voice, seed: seed, language: language)
try f32Data(out.audio).write(
    to: URL(fileURLWithPath: outdir).appendingPathComponent("swift.longform.bin"))

let record = """
{
 "port": "swift",
 "loaded_from": \(quote(args[0])),
 "bundle": \(quote(bundlePath)),
 "voice": \(quote(voicePath)),
 "text": \(quote(text)),
 "seed": \(seed),
 "language": \(quote(language)),
 "decode": \(quote(cfg.decode)),
 "fingerprint": \(quote(engine.describe())),
 "sample_rate": \(cfg.sampleRate),
 "text_tokens": \(jsonInts(textTokens)),
 "speech_tokens_raw": \(jsonInts(raw)),
 "speech_tokens": \(jsonInts(stripped)),
 "mel": {"len": \(mel.count), "frames": \(melFrames), "sha": \(quote(shaOf(mel))), "head": \(jsonFloats(head(mel, 8)))},
 "audio": {"len": \(audio.count), "sha": \(quote(shaOf(audio))), "head": \(jsonFloats(head(audio, 8)))},
 "longform": {"tokens": \(jsonInts(out.tokens)), "audio_len": \(out.audio.count), "audio_sha": \(quote(shaOf(out.audio))), "audio_head": \(jsonFloats(head(out.audio, 8))), "n_chunks": \(out.chunks.count), "hit_token_cap": \(out.hitTokenCap)}
}
"""
try record.write(
    to: URL(fileURLWithPath: outdir).appendingPathComponent("swift.json"),
    atomically: true, encoding: .utf8)
let done = "wrote \(outdir)/swift.json\n"
FileHandle.standardError.write(Data(done.utf8))
