/**
 * The packed checkpoint: manifest + the embedding tables the generator graphs
 * need on the host. A port of the `loudkit.checkpoint` reads, narrowed to what
 * a runtime-only backend touches.
 */

import { describeJson } from "./errors.js";
import { SafetensorsFile } from "./safetensors.js";
import { AlgorithmConfig, algorithmFromManifest, decodeFromManifest } from "./types.js";

/** Manifest versions and decoder loops implemented by this runtime. */
export const SUPPORTED_FORMAT_VERSIONS = [1, 2];
export const SUPPORTED_DECODE_MODES = ["single", "fusion_mtp2"];

export class Checkpoint {
  readonly manifest: Record<string, unknown>;
  private file: SafetensorsFile;

  private constructor(path: string, file: SafetensorsFile) {
    this.file = file;
    const meta = file.metadata ?? null;
    const manifestStr = meta ? (meta.manifest as string | undefined) : undefined;
    if (!manifestStr) {
      throw new Error(`${path}: no embedded manifest, so not a loudkit checkpoint`);
    }
    // Checked before the first key is read, as `checkpoint.py:read_manifest`
    // checks it. A manifest holding a JSON list, number or string reported "no
    // embedded manifest", which is not what is wrong with the file: there is
    // one, and it is not an object. One holding `null` reached a bare
    // TypeError from the first property read.
    const parsed: unknown = JSON.parse(manifestStr);
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error(
        `${path}: manifest is ${describeJson(parsed)}, expected a JSON object`
      );
    }
    const manifest = parsed as Record<string, unknown>;
    if (manifest.format !== "loudkit-checkpoint") {
      throw new Error(`${path}: no embedded manifest, so not a loudkit checkpoint`);
    }
    // `format_version` is checked, not only `format`. Python refuses a version
    // it does not read, and a port that accepts any version will happily load a
    // future checkpoint whose fields mean something else: the loader would
    // still "work", and the audio would be wrong for reasons no error names.
    //
    // A JSON number, truncated, and anything else refused naming the value.
    // `Number` read `null` as 0, and `true`, `"1"` and `[1]` all as 1, so
    // manifests `rust/src/checkpoint.rs` refuses opened here as version 1. The
    // sentence is the reference's; Rust folds the same set to version -1,
    // which reports a version the file does not carry.
    // Presence, not `??`: an absent key is -1 and refused for its version, and
    // an explicit `null` is a declared value that is not a version, which the
    // reference tells apart the same way.
    const declared = "format_version" in manifest ? manifest.format_version : -1;
    if (typeof declared !== "number" || !Number.isFinite(declared)) {
      throw new Error(
        `${path}: manifest['format_version'] is ${JSON.stringify(declared)}, ` +
          "expected a version number"
      );
    }
    const version = Math.trunc(declared);
    if (!SUPPORTED_FORMAT_VERSIONS.includes(version)) {
      throw new Error(
        `${path}: manifest format_version ${version}; this build reads ` +
          `${SUPPORTED_FORMAT_VERSIONS.join(", ")}`
      );
    }
    // And the decode mode, for the file that understates its version. Read
    // through the manifest reader rather than cast here: a cast asserts
    // `decode` is a block, so a list or a scalar answers `undefined` for
    // `.mode` and opens as the single loop, where `algorithmFromManifest`
    // refuses it by name. One door, one answer.
    const mode = decodeFromManifest(manifest);
    if (mode === "fusion_mtp2" && version < 2) throw new Error(`${path}: decode.mode fusion_mtp2 requires format_version 2`);
    if (!SUPPORTED_DECODE_MODES.includes(mode)) {
      throw new Error(
        `${path}: manifest declares decode.mode "${mode}"; this build decodes ` +
          `${SUPPORTED_DECODE_MODES.join(", ")}`
      );
    }
    this.manifest = manifest;
  }

  static open(path: string): Checkpoint {
    const file = new SafetensorsFile(path);
    const ckpt = new Checkpoint(path, file);
    return ckpt;
  }

  algorithm(): AlgorithmConfig {
    return algorithmFromManifest(this.manifest);
  }

  /** fp16 storage upcasts exactly; the exported graphs carry the same fp32
   * weights, so table and graph cannot drift. */
  generatorTables(): {
    textEmb: Float32Array;
    speechEmb: Float32Array;
    textPos: Float32Array;
    speechPos: Float32Array;
  } {
    return {
      textEmb: this.file.f32("t3.text_emb.weight"),
      speechEmb: this.file.f32("t3.speech_emb.weight"),
      textPos: this.file.f32("t3.text_pos_emb.emb.weight"),
      speechPos: this.file.f32("t3.speech_pos_emb.emb.weight"),
    };
  }

  fusionWeights(): { first: Float32Array; firstBias: Float32Array; second: Float32Array; secondBias: Float32Array } | undefined {
    if (this.algorithm().decode !== "fusion_mtp2") return undefined;
    const weights = {
      first: this.file.f32("t3.fuse.0.weight"),
      firstBias: this.file.f32("t3.fuse.0.bias"),
      second: this.file.f32("t3.fuse.2.weight"),
      secondBias: this.file.f32("t3.fuse.2.bias"),
    };
    if (weights.first.length !== 1024 * 2048 || weights.firstBias.length !== 1024 || weights.second.length !== 1024 * 1024 || weights.secondBias.length !== 1024) {
      throw new Error("fusion prefix tensors must have shapes [1024,2048], [1024], [1024,1024], [1024]");
    }
    return weights;
  }

  /** The 192->80 speaker affine the flow decoder conditions on. */
  speakerAffine(): { weight: Float32Array; bias: Float32Array } {
    return {
      weight: this.file.f32("s3gen.flow.spk_embed_affine_layer.weight"),
      bias: this.file.f32("s3gen.flow.spk_embed_affine_layer.bias"),
    };
  }
}
