/**
 * The packed checkpoint: manifest + the embedding tables the generator graphs
 * need on the host. A port of the `loudkit.checkpoint` reads, narrowed to what
 * a runtime-only backend touches.
 */

import { SafetensorsFile } from "./safetensors.js";
import { AlgorithmConfig, algorithmFromManifest } from "./types.js";

/** Manifest format versions this build understands, mirroring Python's. */
export const SUPPORTED_FORMAT_VERSIONS = [1];

export class Checkpoint {
  readonly manifest: Record<string, unknown>;
  private file: SafetensorsFile;

  private constructor(path: string, file: SafetensorsFile) {
    this.file = file;
    const meta = file.metadata ?? null;
    const manifestStr = meta ? (meta.manifest as string | undefined) : undefined;
    if (!manifestStr) {
      throw new Error(`${path}: no embedded manifest — not a loudkit checkpoint`);
    }
    const manifest = JSON.parse(manifestStr) as Record<string, unknown>;
    if (manifest.format !== "loudkit-checkpoint") {
      throw new Error(`${path}: no embedded manifest — not a loudkit checkpoint`);
    }
    // `format_version` is checked, not only `format`. Python refuses a version
    // it does not read, and a port that accepts any version will happily load a
    // future checkpoint whose fields mean something else — the loader would
    // still "work", and the audio would be wrong for reasons no error names.
    const version = Number(manifest.format_version ?? -1);
    if (!SUPPORTED_FORMAT_VERSIONS.includes(version)) {
      throw new Error(
        `${path}: manifest format_version ${version}; this build reads ` +
          `${SUPPORTED_FORMAT_VERSIONS.join(", ")}`
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

  /** The 192->80 speaker affine the flow decoder conditions on. */
  speakerAffine(): { weight: Float32Array; bias: Float32Array } {
    return {
      weight: this.file.f32("s3gen.flow.spk_embed_affine_layer.weight"),
      bias: this.file.f32("s3gen.flow.spk_embed_affine_layer.bias"),
    };
  }
}
