// Compile with the release LoudKit/LoudKitText objects; see README.md.
import Foundation
import LoudKit
let args = CommandLine.arguments
// A trap here would be the first thing a reader of README.md meets, and this
// file is built with `-O`, where a failed `precondition` prints nothing at all:
// the shell reports "Illegal instruction: 4" and a status of 132. The other
// three runners answer a wrong argument count with the usage line and a status,
// so this one does too.
guard args.count == 4 else {
    FileHandle.standardError.write(Data("usage: bench BUNDLE VOICE TEXT\n".utf8))
    exit(2)
}
let bundle = URL(fileURLWithPath: args[1])
var start = ProcessInfo.processInfo.systemUptime
let engine = try Engine.load(checkpoint: bundle.appendingPathComponent(bundle.lastPathComponent + ".safetensors"))
let load = ProcessInfo.processInfo.systemUptime - start
// The checkpoint's rate, not a literal 24000: at any other rate a hardcoded
// divisor reports the wrong audio duration, and so the wrong RTF, for every
// run. The Rust runner reads it from the engine for the same reason.
let sampleRate = Double(engine.algorithm.sampleRate)
let voice = try VoiceProfile.load(url: URL(fileURLWithPath: args[2]))
var runs: [[String: Any]] = []
for run in 0..<4 {
    start = ProcessInfo.processInfo.systemUptime
    var first = 0.0, samples = 0, tokens = 0, chunks = 0
    try engine.stream(args[3], voice: voice, seed: 7) { chunk in
        if chunks == 0 { first = ProcessInfo.processInfo.systemUptime - start }
        chunks += 1; samples += chunk.audio.count; tokens += chunk.tokens.count
        return true
    }
    runs.append(["run": run, "seconds": ProcessInfo.processInfo.systemUptime - start,
                 "ttfa_s": first, "audio_s": Double(samples) / sampleRate,
                 "tokens": tokens, "chunks": chunks])
}
let output: [String: Any] = ["runtime": "swift", "bundle": args[1], "load_s": load,
                            "text": args[3], "seed": 7, "runs": runs]
print(String(data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]), encoding: .utf8)!)
