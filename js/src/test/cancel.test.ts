/**
 * The cancellation contract every port shares: cancelled at a decode step
 * inside the second chunk, `synthesize` throws `CancelledError` and returns
 * nothing, and `stream` yields the first chunk exactly as an uncancelled run
 * does and nothing after, without throwing. Needs the same assets as
 * engine.test.ts; skips without them.
 */

import { existsSync } from "node:fs";
import test from "node:test";
import assert from "node:assert";

import { CancelledError, Engine } from "../engine.js";
import type { ExecutionOptions } from "../execution.js";
import { algorithmFromManifest } from "../types.js";
import { loadVoice } from "../voice.js";
import { refuseIfAssetsRequired } from "./assets.js";
import { fileURLToPath } from "node:url";

const CKPT = process.env.LOUDKIT_CKPT;
const ONNX_DIR = process.env.LOUDKIT_ONNX_DIR;
const VOICE = process.env.LOUDKIT_VOICE;
const TOKENIZER = process.env.LOUDKIT_TOKENIZER;

const available = [CKPT, ONNX_DIR, VOICE, TOKENIZER].every((p) => p && existsSync(p));
refuseIfAssetsRequired(available, "LOUDKIT_CKPT/ONNX_DIR/VOICE/TOKENIZER are not all present");
const EXECUTION: ExecutionOptions = { onnxProvider: "cpu" };

// Comfortably past one window, so the second chunk exists to be cancelled in.
const TEXT =
  "The first sentence sets the scene and runs on for a while. " +
  "The second sentence follows it and is no shorter than the first one was. " +
  "The third sentence exists so that the splitter has somewhere to breathe. " +
  "The fourth sentence closes the passage without hurrying.";

/** A `shouldCancel` that counts its polls and fires past `at`. */
function cancelling(at: number): { shouldCancel: () => boolean; polls: () => number } {
  let polls = 0;
  return {
    shouldCancel: () => {
      polls += 1;
      return polls > at;
    },
    polls: () => polls,
  };
}

test(
  "cancel at a decode step: synthesize throws, stream ends after the chunks that finished",
  { skip: !available && "set LOUDKIT_CKPT/ONNX_DIR/VOICE/TOKENIZER" },
  async () => {
    const engine = await Engine.loadPaths(CKPT!, ONNX_DIR!, TOKENIZER!, EXECUTION);
    try {
    const voice = loadVoice(VOICE!);
    const options = { seed: 7, language: "en" };

    // Uncancelled first, counting the polls: the step to cancel at has to
    // land inside the second chunk's decode loop.
    const counting = cancelling(Number.MAX_SAFE_INTEGER);
    const pollsAtChunk: number[] = [];
    let first: number[] = [];
    for await (const chunk of engine.stream(TEXT, voice, { ...options, ...counting })) {
      pollsAtChunk.push(counting.polls());
      if (first.length === 0) first = chunk.tokens.slice();
    }
    assert.ok(pollsAtChunk.length >= 2, "the passage must split");
    const step = pollsAtChunk[0] + 5;
    assert.ok(step < pollsAtChunk[1], `step ${step} is not inside chunk 1`);

    // stream: the first chunk as it was, nothing after, no throw.
    const got: number[][] = [];
    for await (const chunk of engine.stream(TEXT, voice, { ...options, ...cancelling(step) })) {
      got.push(chunk.tokens.slice());
    }
    assert.equal(got.length, 1, "only the chunk before the cancel arrives");
    assert.deepEqual(got[0], first, "the chunk before the cancel is what an uncancelled run yielded");

    // synthesize: nothing returned, CancelledError.
    await assert.rejects(
      engine.synthesize(TEXT, voice, { ...options, ...cancelling(step) }),
      CancelledError
    );

    // synthesizeWindow, three steps in: the same signal.
    await assert.rejects(
      engine.synthesizeWindow("Hello from loudkit.", voice, { ...options, ...cancelling(3) }),
      CancelledError
    );
    } finally { await engine.close(); }
  }
);

for (const mode of ["synthesize", "synthesizeWindow", "stream"] as const) {
  test(`${mode} discards cancellation set inside the final vocoder`, async () => {
    // Substitute expensive stages, but exercise the public window/stream path.
    const engine = Object.create(Engine.prototype) as Engine;
    const voice = loadVoice(fileURLToPath(new URL("../../../tests/data/enrollment/profile.safetensors", import.meta.url)));
    let cancelled = false;
    Object.assign(engine, {
      config: algorithmFromManifest({}),
      frontend: { encode: () => [1, 2] },
      generateInspected: () => Promise.resolve({ tokens: [1, 2], inspection: {}, hitTokenCap: false, hitWindow: false }),
      decodeMel: () => Promise.resolve(new Float32Array(320)),
      vocode: () => { cancelled = true; return Promise.resolve(new Float32Array(1024)); },
    });
    const options = { shouldCancel: () => cancelled };
    if (mode === "stream") {
      const chunks = [];
      for await (const chunk of engine.stream("Hello world.", voice, options)) chunks.push(chunk);
      assert.deepEqual(chunks, []);
    } else {
      await assert.rejects(engine[mode]("Hello world.", voice, options), CancelledError);
    }
  });
}

test("public synthesis refuses fractional carry before generation", async () => {
  const engine = Object.create(Engine.prototype) as Engine;
  Object.assign(engine, { config: algorithmFromManifest({}) });
  const voice = loadVoice(fileURLToPath(new URL("../../../tests/data/enrollment/profile.safetensors", import.meta.url)));
  await assert.rejects(engine.synthesize("Hello.", voice, { previousTokens: [1.9] }), RangeError);
});
