// Package.swift to hello.wav: `swift run Hello`. Model files are cached locally.
import Foundation
import LoudKit

let engine = try await Engine.load("loudreader/loudr-1")
let result = try engine.synthesize("Hello from loudkit.", voice: engine.voice(named: "joe"), seed: 7)
try result.saveWav("hello.wav")
print("hello.wav: \(String(format: "%.2f", result.duration))s")
