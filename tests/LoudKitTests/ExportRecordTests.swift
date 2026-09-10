import Foundation
import XCTest

@testable import LoudKit

final class ExportRecordTests: XCTestCase {
    func testRendererRecordMustMatchCheckpointAndEachOther() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("model.safetensors")
        try Safetensors.write([], metadata: ["manifest": #"{"format":"loudkit-checkpoint","format_version":1}"#], to: url)
        let checkpoint = try Checkpoint(url: url)
        let algorithm = AlgorithmConfig()
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
        packages["flow_estimator.mlpackage"]?["checkpoint_sha256"] = "another checkpoint"
        try write()
        XCTAssertThrowsError(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
        for name in packages.keys { packages[name]?["checkpoint_sha256"] = "another checkpoint" }
        try write()
        XCTAssertThrowsError(try checkpoint.verifyCoreMLExport(at: root, algorithm: algorithm))
    }
}
