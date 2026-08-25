import XCTest

@testable import LoudKit

/// The weight-free half of the conformance contract: Philox bits, sampler
/// choices, frontend ids, seed derivation. These vectors were produced by the
/// Python implementation; if this file passes, the two implementations agree
/// on every decision that shapes a reading — without either having imported
/// the other.
/// The cases for one fixture section, refusing an empty list.
///
/// Every loop below iterates a slice pulled out of the fixture by key, with a
/// force-cast that turns a *renamed* key into a crash but an *empty* list into
/// a silent pass — the entire cross-language determinism claim switched off
/// with nothing to show for it. Go, Rust and JS grew the same guard.
private func cases(_ section: [String: Any], _ key: String,
                   file: StaticString = #filePath, line: UInt = #line) -> [[String: Any]] {
    guard let list = section[key] as? [[String: Any]] else {
        XCTFail("the fixture has no \(key) section; nothing was compared", file: file, line: line)
        return []
    }
    XCTAssertFalse(list.isEmpty, "fixture section \(key) is empty; nothing was compared",
                   file: file, line: line)
    return list
}

final class PhiloxConformanceTests: XCTestCase {
    func testKnownAnswerVectors() throws {
        let fixture = try Fixture.vectors()
        let philox = fixture["philox"] as! [String: Any]
        for kase in cases(philox, "kat") {
            let c = asInts(kase["counter"])!.map(UInt32.init)
            let k = asInts(kase["key"])!.map(UInt32.init)
            let want = asInts(kase["expected"])!.map(UInt32.init)
            let got = Philox.philox4x32_10(counter: (c[0], c[1], c[2], c[3]), key: (k[0], k[1]))
            XCTAssertEqual([got.0, got.1, got.2, got.3], want)
        }
    }

    func testUniformBits() throws {
        let fixture = try Fixture.vectors()
        let philox = fixture["philox"] as! [String: Any]
        for probe in cases(philox, "uniform_bits") {
            let seedHex = probe["seed"] as! String
            let seed = UInt64(seedHex.dropFirst(2), radix: 16)!
            let stream = UInt32((probe["stream"] as! NSNumber).intValue)
            let step0 = UInt32((probe["step0"] as! NSNumber).intValue)
            let nSteps = (probe["n_steps"] as! NSNumber).intValue
            let width = (probe["width"] as! NSNumber).intValue
            let u = Philox.uniforms(seed: seed, stream: stream, step0: step0,
                                    nSteps: nSteps, width: width)
            let bits = u.map { UInt64(($0 * 4_294_967_296.0 - 0.5).rounded()) }
            let want = (probe["bits"] as! [[NSNumber]]).flatMap { $0.map { UInt64(truncating: $0) } }
            XCTAssertEqual(bits, want, "uniform bits drifted for seed \(seedHex)")
        }
    }

    func testGumbelProbes() throws {
        let fixture = try Fixture.vectors()
        let philox = fixture["philox"] as! [String: Any]
        for probe in cases(philox, "gumbel") {
            let seed = UInt64((probe["seed"] as! NSNumber).uint64Value)
            let stream = UInt32((probe["stream"] as! NSNumber).intValue)
            let step = UInt32((probe["step"] as! NSNumber).intValue)
            let width = (probe["width"] as! NSNumber).intValue
            let rtol = (probe["rtol"] as! NSNumber).doubleValue
            let g = Philox.gumbelNoise(seed: seed, stream: stream, step0: step, nSteps: 1, width: width)
            let want = asDoubles(probe["values"])!
            for (a, b) in zip(g, want) {
                XCTAssertLessThanOrEqual(abs(a - b), rtol * max(abs(a), abs(b), 1.0),
                                         "gumbel value drifted beyond a libm ulp allowance")
            }
        }
    }
}

final class SamplerConformanceTests: XCTestCase {
    private func samplingConfig(_ raw: [String: Any]) -> SamplingConfig {
        var config = SamplingConfig()
        config.temperature = (raw["temperature"] as! NSNumber).doubleValue
        config.repetitionPenalty = (raw["repetition_penalty"] as! NSNumber).doubleValue
        config.minP = (raw["min_p"] as! NSNumber).doubleValue
        config.silenceTokenIds = asInts(raw["silence_token_ids"]) ?? []
        return config
    }

    private func logitsRows(_ kase: [String: Any]) -> [[Float]] {
        if let recipe = kase["logits_recipe"] as? [String: Any] {
            let seed = UInt64((recipe["seed"] as! NSNumber).uint64Value)
            let stream = UInt32((recipe["stream"] as! NSNumber).intValue)
            let scale = (recipe["scale"] as! NSNumber).doubleValue
            let offset = (recipe["offset"] as! NSNumber).doubleValue
            let vocab = (recipe["vocab"] as! NSNumber).intValue
            let steps = (recipe["steps"] as! NSNumber).intValue
            return (0..<steps).map { step in
                Philox.uniforms(seed: seed, stream: stream, step0: UInt32(step), nSteps: 1, width: vocab)
                    .map { Float($0 * scale + offset) }
            }
        }
        let literal = (kase["logits"] as! [[NSNumber]]).map { $0.map { Float(truncating: $0) } }
        let repeatCount = (kase["repeat_logits"] as? NSNumber)?.intValue ?? literal.count
        return literal.count == 1 ? Array(repeating: literal[0], count: repeatCount) : literal
    }

    func testTokenChoices() throws {
        let fixture = try Fixture.vectors()
        let sampler = fixture["sampler"] as! [String: Any]
        for kase in cases(sampler, "cases") {
            let name = kase["name"] as! String
            let config = samplingConfig(kase["config"] as! [String: Any])
            let seed = UInt64((kase["seed"] as! NSNumber).uint64Value)
            let rows = logitsRows(kase)
            let expected = asInts(kase["expected"])!
            let lr = LRSamplerV1(config: config, seed: seed)
            var seen = [Bool](repeating: false, count: rows[0].count)
            var got: [Int] = []
            for (step, row) in rows.enumerated() {
                let tok = lr.sample(logits: row, step: step, seen: seen)
                got.append(tok)
                seen[tok] = true
            }
            XCTAssertEqual(got, expected, "sampler case \(name) diverged")
        }
    }
}

final class FrontendConformanceTests: XCTestCase {
    func testTokenIds() throws {
        let fixture = try Fixture.vectors()
        let frontendSection = fixture["frontend"] as! [String: Any]
        let tokenizerURL = Fixture.conformanceDir
            .appendingPathComponent(frontendSection["tokenizer"] as! String)
        let frontend = try TextFrontend(tokenizerURL: tokenizerURL)
        for kase in cases(frontendSection, "cases") {
            let text = kase["text"] as! String
            let language = kase["language"] as! String
            let want = asInts(kase["ids"])!
            XCTAssertEqual(try frontend.encode(text, language: language), want,
                           "frontend diverged on: \(text)")
        }
    }
}

final class SeedConformanceTests: XCTestCase {
    func testDerivation() throws {
        let fixture = try Fixture.vectors()
        let seeds = fixture["seeds"] as! [String: Any]
        for probe in cases(seeds, "derivation") {
            let seed = UInt64((probe["seed"] as! NSNumber).uint64Value)
            let stream = UInt64((probe["stream"] as! NSNumber).uint64Value)
            let wantHex = probe["derived"] as! String
            let want = UInt64(wantHex.dropFirst(2), radix: 16)!
            XCTAssertEqual(Engine.derive(seed, stream), want)
        }
    }
}

final class AlgorithmConformanceTests: XCTestCase {
    func testFingerprintAndCanonicalJSON() throws {
        try Fixture.requireCheckpoint()
        let fixture = try Fixture.vectors()
        let section = fixture["algorithm"] as! [String: Any]
        let checkpoint = try Checkpoint(url: Fixture.checkpointURL)
        let algorithm = try checkpoint.algorithm()
        // canonical form first: a form mismatch names the drifted field
        XCTAssertEqual(algorithm.canonicalForm(), section["canonical_form"] as! String)
        XCTAssertEqual(algorithm.fingerprint(), section["fingerprint"] as! String)
    }

    func testEulerGrid() throws {
        try Fixture.requireCheckpoint()
        let fixture = try Fixture.vectors()
        let section = fixture["algorithm"] as! [String: Any]
        let checkpoint = try Checkpoint(url: Fixture.checkpointURL)
        let algorithm = try checkpoint.algorithm()
        let want = asDoubles(section["euler_grid"])!
        let rtol = (section["grid_rtol"] as! NSNumber).doubleValue
        let got = algorithm.timeGrid()
        XCTAssertEqual(got.count, want.count)
        for (a, b) in zip(got, want) {
            XCTAssertLessThanOrEqual(abs(a - b), rtol * max(abs(a), abs(b), 1.0))
        }
    }
}
