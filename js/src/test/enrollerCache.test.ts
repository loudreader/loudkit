/**
 * The enrollment graphs open once per engine, however many callers ask at once.
 *
 * `Enroller.load` opens three native onnxruntime sessions. Reading the cached
 * field before the await lets two concurrent first calls each open a set, and
 * the second assignment drops the first with no `close()`: three graphs leaked
 * for the life of the process, invisible to the caller. No weights are needed
 * to show it, so this runs everywhere.
 */

import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { Engine } from "../engine.js";
import { Enroller, type Enrolled } from "../enroll.js";

/** What `Enroller.enroll` hands back, at the smallest size that is still one. */
const ENROLLED: Enrolled = {
  speakerEmbedding: new Float32Array(256),
  flowEmbedding: new Float32Array(192),
  promptTokens: new BigInt64Array(0),
  promptMel: new Float32Array(0),
  promptMelFrames: 0,
  condPromptTokens: new BigInt64Array(0),
};

/**
 * A release directory that `canEnroll` accepts, so `ensureCloning` returns
 * without a fetch. The files are never opened: `Enroller.load` is mocked.
 */
function releaseDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "loudkit-enroll-"));
  mkdirSync(join(dir, "onnx"), { recursive: true });
  for (const name of ["s3_tokenizer.onnx", "camp.onnx", "voice_encoder.onnx"]) {
    writeFileSync(join(dir, "onnx", name), "");
  }
  return dir;
}

/**
 * An engine with the four fields `enroll` reads, built without `load`.
 *
 * `Object.create` rather than a test-only constructor, the way
 * `speechPosition.test.ts` does it: widening the engine's surface to reach
 * private state would be a production change made for one assertion.
 */
function harness(dir: string): Engine {
  const engine = Object.create(Engine.prototype) as Engine;
  const stub = () => ({ close: () => Promise.resolve() });
  Object.assign(engine as unknown as Record<string, unknown>, {
    dir,
    onnxDir: join(dir, "onnx"),
    onnxProvider: "cpu",
    repo: undefined,
    // The six synthesis graphs `close` hands back; nothing here runs them.
    cond: stub(),
    prefill: stub(),
    step: stub(),
    encoder: stub(),
    estimator: stub(),
    vocoder: stub(),
  });
  return engine;
}

test("two concurrent first enrollments open one set of graphs", async (t) => {
  const dir = releaseDir();
  let opened = 0;
  let closed = 0;
  t.mock.method(Enroller, "load", async () => {
    opened += 1;
    // One turn of the loop between the read of the cache and its write, which
    // is the whole window the bug lived in.
    await Promise.resolve();
    return {
      enroll: () => Promise.resolve(ENROLLED),
      close: () => {
        closed += 1;
        return Promise.resolve();
      },
    } as unknown as Enroller;
  });

  try {
    const engine = harness(dir);
    const samples = new Float32Array(24_000);
    const [first, second] = await Promise.all([
      engine.enroll(samples, { name: "a", sampleRate: 24_000 }),
      engine.enroll(samples, { name: "b", sampleRate: 24_000 }),
    ]);
    assert.equal(opened, 1, "the three enrollment graphs were opened twice");
    assert.equal(first.name, "a");
    assert.equal(second.name, "b");

    // And the one set is handed back, once, by a close that is safe twice.
    await engine.close();
    await engine.close();
    assert.equal(closed, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a failed load is not cached, so the next call retries", async (t) => {
  const dir = releaseDir();
  let attempts = 0;
  t.mock.method(Enroller, "load", () => {
    attempts += 1;
    return attempts === 1
      ? Promise.reject(new Error("no such graph"))
      : Promise.resolve({
          enroll: () => Promise.resolve(ENROLLED),
          close: () => Promise.resolve(),
        } as unknown as Enroller);
  });

  try {
    const engine = harness(dir);
    const samples = new Float32Array(24_000);
    await assert.rejects(engine.enroll(samples, { sampleRate: 24_000 }), /no such graph/);
    const profile = await engine.enroll(samples, { name: "b", sampleRate: 24_000 });
    assert.equal(profile.name, "b");
    assert.equal(attempts, 2);
    await engine.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
