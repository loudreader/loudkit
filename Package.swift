// swift-tools-version: 5.9
// LoudKit, the Swift implementation of the loudkit engine.
//
// `Package.swift` sits at the repo root because SwiftPM cannot consume a
// package from a subdirectory, a git dependency reads the manifest from the
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
        // The text funnel alone, for a consumer that wants the language-aware
        // verbalization without the CoreML engine.
        .library(name: "LoudKitText", targets: ["LoudKitText"]),
    ],
    targets: [
        // Pure text: the speech funnel and the Polish respelling lexicon,
        // with no CoreML and no engine behind them.
        //
        // Its own target because both engines need it and neither owns it. The
        // funnel decides what the weights are asked to say, "Rabat 15% na
        // weekend!" becomes "Rabat piętnaście procent na łikend!" before a
        // single token is emitted, so an engine that cannot reach it produces
        // different speech from every other port while every fingerprint
        // agrees.
        // Resources are bundled rather than excluded, because the number
        // grammars are read from the same numbers.json every other
        // implementation reads: the one-file-five-readers rule that keeps twelve
        // languages from drifting. `pl_en_respell.json` and `numerals.json` ride
        // along for the same reason. A funnel asset reachable only through
        // `ChatterboxAssets`, a channel `swift test` does not populate, makes
        // the pass that reads it switch itself off in the package while every
        // fingerprint still claims parity.
        .target(
            name: "LoudKitText",
            path: "swift/LoudKitText",
            resources: [
                .copy("Resources/numbers.json"),
                .copy("Resources/pl_en_respell.json"),
                .copy("Resources/numerals.json"),
                // The curated half of Polish respelling, the phrases and word
                // lists chosen by ear. It sits below `grammar_digest` rather
                // than in it, so unlike the three above it is one file five
                // readers must hold identical with nothing to catch a drift:
                // `tools/sync_grammar.py` copies it, and `RespellingTests`
                // compares what this port parsed against the shipped bytes.
                .copy("Resources/pl_respell_rules.json"),
            ]
        ),
        // `LoudKit` runs the CoreML graphs and takes its compute-unit
        // placement as an `ExecutionConfig` parameter: it exposes the knob and
        // holds no opinion about where each stage should run. Deciding that,
        // the tuned per-stage placement and warm-up order that make the Neural
        // Engine fast, is product rather than kit, and is not part of this
        // package. What ships is the measurement of what the ANE can do
        // (`docs/platforms/apple.md`), not the recipe for getting there.
        .target(name: "LoudKit", dependencies: ["LoudKitText"], path: "swift/LoudKit",
                resources: [.copy("Resources")]),
        // Compile the quickstart from `swift/README.md` as an executable
        // target to check the documented API; it is not an exported product.
        .executableTarget(name: "Hello", dependencies: ["LoudKit"], path: "swift/Examples/Hello"),
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
