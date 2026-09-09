import Foundation
import XCTest

@testable import LoudKit

final class PairDecoderTests: XCTestCase {
    private func generator(first: Int = 0, second: Int = 1, floor: Int = 0) throws -> TokenGenerator {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("tiny.safetensors")
        var entries: [Safetensors.Entry] = []
        func tensor(_ name: String, _ shape: [Int], value: Float = 0, values: [Float]? = nil) {
            let data = values ?? [Float](repeating: value, count: shape.reduce(1, *))
            entries.append(.init(name: "t3." + name, dtype: "F32", shape: shape,
                                 data: data.withUnsafeBytes { Data($0) }))
        }
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            tensor("tfmr.layers.0." + name + ".weight", [4, 4])
        }
        for name in ["tfmr.layers.0.input_layernorm", "tfmr.layers.0.post_attention_layernorm", "tfmr.norm"] {
            tensor(name + ".weight", [4], value: 1)
        }
        tensor("speech_emb.weight", [4, 4], value: 1)
        tensor("text_emb.weight", [256, 4])
        tensor("speech_pos_emb.emb.weight", [32, 4])
        tensor("text_pos_emb.emb.weight", [32, 4])
        tensor("speech_head.weight", [4, 4], values: (0..<16).map { $0 / 4 == first ? 25 : 0 })
        tensor("head2.weight", [4, 8], values: (0..<32).map { $0 / 8 == second ? 25 : 0 })
        tensor("fuse.0.weight", [4, 8]); tensor("fuse.0.bias", [4])
        tensor("fuse.2.weight", [4, 4]); tensor("fuse.2.bias", [4])
        tensor("cond_enc.spkr_enc.weight", [4, 256]); tensor("cond_enc.spkr_enc.bias", [4])
        tensor("cond_enc.emotion_adv_fc.weight", [4, 1])
        tensor("cond_enc.perceiver.pre_attention_query", [32, 4])
        tensor("cond_enc.perceiver.attn.norm.weight", [4], value: 1)
        tensor("cond_enc.perceiver.attn.norm.bias", [4])
        for name in ["to_q", "to_k", "to_v", "proj_out"] {
            tensor("cond_enc.perceiver.attn." + name + ".weight", [4, 4])
            tensor("cond_enc.perceiver.attn." + name + ".bias", [4])
        }
        let manifest: [String: Any] = ["format": "loudkit-checkpoint", "format_version": 2,
            "decode": ["mode": "fusion_mtp2"],
            "llama_config": ["hidden_size": 4, "intermediate_size": 4, "num_hidden_layers": 1,
                             "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 4]]
        let json = String(data: try JSONSerialization.data(withJSONObject: manifest), encoding: .utf8)!
        try Safetensors.write(entries, metadata: ["manifest": json], to: url)
        var algorithm = AlgorithmConfig()
        algorithm.decode = "fusion_mtp2"
        algorithm.speechVocabSize = 4
        algorithm.startSpeechToken = 2
        algorithm.stopSpeechToken = 3
        algorithm.sampling.minTokensFloor = floor
        algorithm.sampling.minTokensTextRatio = 0
        return try TokenGenerator(checkpoint: Checkpoint(url: url), config: algorithm)
    }

    private var voice: VoiceProfile {
        VoiceProfile(name: "tiny", speakerEmbedding: [Float](repeating: 0, count: 256),
                     flowEmbedding: [], promptTokens: [], promptMel: [], promptMelFrames: 0,
                     condPromptTokens: [0], language: "en")
    }

    func testBothHeadsAndOddCap() throws {
        for (first, second, cap, expected) in [(3, 1, 6, [3]), (0, 3, 6, [0, 3]),
                                               (0, 1, 3, [0, 1, 0]), (0, 1, 4, [0, 1, 0, 1])] {
            let model = try generator(first: first, second: second)
            let result = try model.generate(textTokens: [1], voice: voice,
                sampler: LRSamplerV1(config: SamplingConfig(), seed: 7), maxNewTokens: cap)
            XCTAssertEqual(result.rawTokens, expected)
            XCTAssertEqual(result.hitTokenCap, !expected.contains(3))
        }
    }

    func testFloorAndCancellation() throws {
        let model = try generator(first: 3, second: 3, floor: 2)
        let result = try model.generate(textTokens: [1], voice: voice,
            sampler: LRSamplerV1(config: SamplingConfig(), seed: 7), maxNewTokens: 6)
        XCTAssertEqual(result.rawTokens.count, 3)
        XCTAssertFalse(result.rawTokens.prefix(2).contains(3))
        XCTAssertEqual(result.rawTokens.last, 3)
        XCTAssertThrowsError(try model.generate(textTokens: [1], voice: voice,
            sampler: LRSamplerV1(config: SamplingConfig(), seed: 7), shouldCancel: { true })) { error in
            guard case LoudKitError.cancelled = error else { return XCTFail("\(error)") }
        }
    }
}
