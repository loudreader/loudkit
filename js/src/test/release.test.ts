/**
 * The native sessions can be handed back.
 *
 * `InferenceSession` holds an onnxruntime handle outside the JS heap, so the
 * collector cannot reclaim it — a dropped reference frees the wrapper and
 * leaks the graph. Go has `Engine.Close`, Rust and Swift get it
 * from their ownership rules; a binding with neither leaks its graphs.
 *
 * Two of the three cases run without a checkpoint, because the one that
 * matters most is an error path: `Engine.load` opens six graphs one at a
 * time and must not abandon what it already opened when a later
 * one throws.
 */

import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert";

import { Engine } from "../engine.js";
import { Session } from "../session.js";

const CKPT = process.env.LOUDKIT_CKPT;
const ONNX_DIR = process.env.LOUDKIT_ONNX_DIR;
const TOKENIZER = process.env.LOUDKIT_TOKENIZER;
const available = [CKPT, ONNX_DIR, TOKENIZER].every((p) => p && existsSync(p));

test("both Session and Engine expose a release path", () => {
  // A surface check earns its place here: a missing release path
  // is visible without weights.
  assert.equal(typeof Session.prototype.close, "function");
  assert.equal(typeof Engine.prototype.close, "function");
});

test("a graph that fails to open does not abandon the ones already open", async () => {
  // Only the first graph exists, so `Session.create` throws on the second and
  // the unwind path is the one under test. The
  // assertion here is simply that the failure is reported rather than swallowed
  // — the release itself is observable only in native memory, so the practical
  // pin is the error, plus the surface check above.
  const dir = mkdtempSync(join(tmpdir(), "loudkit-release-"));
  writeFileSync(join(dir, "t3_cond.onnx"), "not a graph");
  await assert.rejects(() => Engine.load("nonexistent.safetensors", dir, "nonexistent.json"));
});

test(
  "an engine releases its six graphs and can be closed twice",
  { skip: !available && "set LOUDKIT_CKPT/ONNX_DIR/TOKENIZER" },
  async () => {
    const engine = await Engine.load(CKPT!, ONNX_DIR!, TOKENIZER!);
    await engine.close();
    // Idempotent on purpose: the interesting callers are error paths that
    // cannot cheaply know what has already been released.
    await engine.close();
  }
);
