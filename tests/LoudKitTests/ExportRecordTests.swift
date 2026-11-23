import Foundation
import XCTest

@testable import LoudKit

final class ExportRecordTests: XCTestCase {
    func testRendererRecordMustMatchCheckpointAndEachOther() throws {
        let base = ["7cd75498ad4e7531", "b99ab84725b40e91", "f171bcc07df48436", "82c2c12d6e725542"]
        let turbo = ["e5303ba243087222", "e800f3bbb7eb6b4e", "06d8ccb48891899e", "62d674008ef832fd"]
        try checkRendererRecord(mode: "single", steps: 2, previous: base, other: turbo)
        try checkRendererRecord(mode: "fusion_mtp2", steps: 1, previous: turbo, other: base)
    }

    private func checkRendererRecord(mode: String, steps: Int, previous: [String], other: [String]) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("model.safetensors")
        var fields = ManifestFreezeTests.shipped011()
        fields["decode"] = ["mode": mode]
        fields["format_version"] = mode == "fusion_mtp2" ? 2 : 1
        fields["n_cfm_timesteps"] = steps
        let manifest = try JSONSerialization.data(withJSONObject: fields)
        try Safetensors.write([], metadata: ["manifest": String(decoding: manifest, as: UTF8.self)], to: url)
        let checkpoint = try Checkpoint(url: url)
        let algorithm = try checkpoint.algorithm()
        let record: [String: Any] = ["checkpoint_sha256": try LoudKit.fileSHA256(url),
                                     "algorithm_fingerprint": algorithm.fingerprint(),
                                     "euler_steps": algorithm.eulerSteps]
        var packages = Dictionary(uniqueKeysWithValues:
            ["flow_encoder", "flow_estimator", "vocoder"].map { ($0 + ".mlpackage", record) })
        func write() throws {
            try JSONSerialization.data(withJSONObject: ["packages": packages])
                .write(to: root.appendingPathComponent("export.json"))
        }
        try write()
        XCTAssertNoThrow(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
        for fingerprint in other {
            for name in packages.keys { packages[name]?["algorithm_fingerprint"] = fingerprint }
            try write()
            XCTAssertThrowsError(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
        }
        for fingerprint in previous {
            for name in packages.keys { packages[name]?["algorithm_fingerprint"] = fingerprint }
            try write()
            XCTAssertNoThrow(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
        }
        // Both records are compatible individually. Each member must still
        // come from a single export.
        packages["vocoder.mlpackage"]?["algorithm_fingerprint"] = previous[0]
        try write()
        XCTAssertThrowsError(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
        packages["vocoder.mlpackage"]?["algorithm_fingerprint"] = previous[1]
        packages["flow_estimator.mlpackage"]?["checkpoint_sha256"] = "another checkpoint"
        try write()
        XCTAssertThrowsError(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
        for name in packages.keys { packages[name]?["checkpoint_sha256"] = "another checkpoint" }
        try write()
        XCTAssertThrowsError(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
    }
}
