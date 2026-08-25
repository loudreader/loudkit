/**
 * Enrollment conformance against the enrollment fixture (needs the exported
 * enrollment graphs). The same reference clip must yield the fixture's prompt
 * tokens exactly and its embeddings to cosine > 0.9999.
 *
 * Skips when the assets are absent; set the env vars below to point at them.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert";

import { Enroller } from "../enroll.js";

const ONNX_DIR = process.env.LOUDKIT_ONNX_DIR;

/** The enrollment fixture, resolved by walking up to the repo root. */
function fixtureDir(): string {
  if (process.env.LOUDKIT_ENROLL_FIXTURE) return process.env.LOUDKIT_ENROLL_FIXTURE;
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "tests", "data", "enrollment");
    if (existsSync(join(candidate, "ref_audio.f32"))) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "tests", "data", "enrollment");
}

const available = Boolean(ONNX_DIR) && existsSync(join(fixtureDir(), "ref_audio.f32"));

if (!available && process.env.LOUDKIT_REQUIRE_ASSETS && process.env.LOUDKIT_REQUIRE_ASSETS !== "0") {
  throw new Error("LOUDKIT_REQUIRE_ASSETS is set but LOUDKIT_ONNX_DIR / the enrollment fixture are not present");
}

function readF32(path: string): Float32Array {
  const buf = readFileSync(path);
  const out = new Float32Array(buf.length / 4);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

function readI64(path: string): BigInt64Array {
  const buf = readFileSync(path);
  const out = new BigInt64Array(buf.length / 8);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let i = 0; i < out.length; i++) out[i] = dv.getBigInt64(i * 8, true);
  return out;
}

function cos(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / Math.sqrt(na * nb);
}

test("enrollment matches the fixture", { skip: !available }, async () => {
  const fx = fixtureDir();
  // CPU for the same reason the engine conformance test pins it: the prompt
  // tokens below are an exact-match assertion, so the provider has to be named
  // rather than chosen per machine.
  const enr = await Enroller.load(ONNX_DIR!, { onnxProvider: "cpu" });
  const audio = readF32(join(fx, "ref_audio.f32"));
  const res = await enr.enroll(audio, 24000);

  assert.deepEqual(res.promptTokens, readI64(join(fx, "prompt_tokens.i64")), "prompt tokens exact");
  assert.deepEqual(res.condPromptTokens, readI64(join(fx, "cond_prompt_tokens.i64")), "cond tokens exact");

  const cf = cos(res.flowEmbedding, readF32(join(fx, "flow_embedding.f32")));
  const cs = cos(res.speakerEmbedding, readF32(join(fx, "speaker_embedding.f32")));
  assert.ok(cf > 0.9999, `flow embedding cosine ${cf} <= 0.9999`);
  assert.ok(cs > 0.9999, `speaker embedding cosine ${cs} <= 0.9999`);
});
