// swift-tools-version: 5.9
// LoudKit — the Swift implementation of the loudkit engine.
//
// `Package.swift` sits at the repo root because SwiftPM cannot consume a
// package from a subdirectory — a git dependency reads the manifest from the
// root or not at all. The *sources* do not have to be there, so they are not:
// every target names its path under `swift/`, beside `python/`, `go/`, `rust/`
// and `js/`. Five peer implementations, one directory each, and the two
// manifests that have to be in the root are in the root.
//
// The conformance fixture (tests/data/conformance) is read by `pytest` and
// `swift test` alike, which is why the test targets point back into `tests/`.
import PackageDescription

let package = Package(
    name: "LoudKit",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "LoudKit", targets: ["LoudKit"]),
        // The text funnel alone, for consumers (like the LoudReader app) that
        // want the language-aware verbalization without the CoreML engine.
        .library(name: "LoudKitText", targets: ["LoudKitText"]),
    ],
    targets: [
        // Pure text: the speech funnel and the Polish respelling lexicon,
        // with no CoreML and no engine behind them.
        //
        // Its own target because both engines need it and neither owns it. The
        // funnel decides what the weights are asked to say — "Rabat 15% na
        // weekend!" becomes "Rabat piętnaście procent na łikend!" before a
        // single token is emitted — so an engine that cannot reach it produces
        // different speech from every other port while every fingerprint
        // agrees.
        // Resources are bundled (not excluded) since the number grammars are read
        // from the same numbers.json every other implementation reads — the
        // one-file-five-readers rule that keeps twelve languages from drifting.
        // `pl_en_respell.json` rides along for the same reason. It is the last
        // pass of the funnel, and it was the one funnel asset resolved solely
        // through `ChatterboxAssets` — a channel `swift test` does not populate
        // — so respelling switched itself off, logged a line nobody reads, and
        // the package spoke the English inside Polish text with Polish letter
        // values while every fingerprint still claimed parity.
        .target(
            name: "LoudKitText",
            path: "swift/LoudKitText",
            resources: [
                .copy("Resources/numbers.json"),
                .copy("Resources/pl_en_respell.json"),
            ]
        ),
        // `LoudKit` runs the CoreML graphs and takes its compute-unit
        // placement as an `ExecutionConfig` parameter: it exposes the knob and
        // holds no opinion about where each stage should run. Deciding that —
        // the tuned per-stage placement and warm-up order that make the Neural
        // Engine fast — is product rather than kit, and is not part of this
        // package. What ships is the measurement of what the ANE can do
        // (`docs/platforms/apple.md`), not the recipe for getting there.
        .target(name: "LoudKit", dependencies: ["LoudKitText"], path: "swift/LoudKit",
                resources: [.copy("Resources")]),
        // Explicit lowercase path: on this repo's case-insensitive dev machine
        // "Tests" and the Python "tests" directory are one and the same, so the
        // Swift tests live at tests/LoudKitTests and the manifest must say so
        // for a case-sensitive checkout to build.
        .testTarget(name: "LoudKitTests", dependencies: ["LoudKit"], path: "tests/LoudKitTests"),
        // The funnel's own tests, in the target that owns the funnel.
        // `SpeechText` and `LexicalRespelling` are the implementations the
        // Python, Go, Rust and JS ports are bit-parity ports *of*, so this is
        // the reference all the others are measured against.
        .testTarget(name: "LoudKitTextTests", dependencies: ["LoudKitText"],
                    path: "tests/LoudKitTextTests"),
    ]
)
