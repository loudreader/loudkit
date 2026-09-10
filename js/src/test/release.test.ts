/**
 * The native sessions can be handed back.
 *
 * `InferenceSession` holds an onnxruntime handle outside the JS heap, so the
 * collector cannot reclaim it: a dropped reference frees the wrapper and
 * leaks the graph. Go has `Engine.Close`, Rust and Swift get it
 * from their ownership rules; a binding with neither leaks its graphs.
 *
 * Two of the three cases run without a checkpoint, because the one that
 * matters most is an error path: `Engine.load` opens six graphs one at a
 * time and must not abandon what it already opened when a later
 * one throws.
 */

import { existsSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert";

import { Engine } from "../engine.js";
import { Session, openSessions } from "../session.js";
import { Checkpoint } from "../checkpoint.js";
import { refuseIfAssetsRequired } from "./assets.js";

const CKPT = process.env.LOUDKIT_CKPT;
const ONNX_DIR = process.env.LOUDKIT_ONNX_DIR;
const TOKENIZER = process.env.LOUDKIT_TOKENIZER;
const available = [CKPT, ONNX_DIR, TOKENIZER].every((p) => p && existsSync(p));
refuseIfAssetsRequired(available, "LOUDKIT_CKPT/ONNX_DIR/TOKENIZER are not all present");

test("both Session and Engine expose a release path", () => {
  // A surface check earns its place here: a missing release path
  // is visible without weights.
  assert.equal(typeof Session.prototype.close, "function");
  assert.equal(typeof Engine.prototype.close, "function");
});

test("a graph that fails to open closes the sessions already acquired", async (t) => {
  let opened = 0;
  let closed = 0;
  t.mock.method(Session, "create", () => {
    if (opened === 1) throw new Error("broken graph");
    opened++;
    return Promise.resolve({ close: () => { closed++; return Promise.resolve(); } });
  });
  await assert.rejects(openSessions([["first", "first"], ["second", "second"]], "cpu"), /broken graph/);
  assert.equal(opened, 1);
  assert.equal(closed, 1);
});

test("invalid tokenizer fails before any native sessions are acquired", async (t) => {
  let opened = 0;
  t.mock.method(Checkpoint, "open", () => ({
    algorithm: () => ({ decode: "single" }), fusionWeights: () => undefined,
  }));
  t.mock.method(Session, "create", () => { opened++; return Promise.resolve({}); });
  const dir = mkdtempSync(join(tmpdir(), "loudkit-load-"));
  try {
    const tokenizer = join(dir, "tokenizer.json");
    writeFileSync(tokenizer, "not json");
    await assert.rejects(Engine.loadPaths("mock", dir, tokenizer, { onnxProvider: "cpu" }), SyntaxError);
    assert.equal(opened, 0);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test(
  "an engine releases its six graphs and can be closed twice",
  { skip: !available && "set LOUDKIT_CKPT/ONNX_DIR/TOKENIZER" },
  async () => {
    const engine = await Engine.loadPaths(CKPT!, ONNX_DIR!, TOKENIZER!);
    await engine.close();
    // Idempotent on purpose: the interesting callers are error paths that
    // cannot cheaply know what has already been released.
    await engine.close();
  }
);
