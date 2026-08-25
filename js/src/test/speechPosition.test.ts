/**
 * The learned speech positional row a generated token is given.
 *
 * The prefill lays out `cond ‖ text ‖ BOS@0 ‖ prefix[i]@(i+1)`, so a prefix of
 * length P owns speech positions 1..P. The first generated token therefore sits
 * at P+1. Asking for `step + 1` hands it a row the prefill just wrote for a
 * carried token and never reaches P+1 or beyond, which only shows up when a
 * chunk carries a prefix — every multi-chunk synthesis, never a single window.
 * Python (backends/onnx_backend.py:353) and Swift
 * (LoudKit/TokenGenerator.swift:586) have always used `len(prefix) + step + 1`.
 *
 * The engine cannot run its graphs here, so this drives `generate` against
 * three recording stand-in sessions. That is the only seam that shows the bug:
 * the index is inside the decode loop, and the number it produces leaves the
 * engine only as a row of the step graph's `embeds` feed. The tables are rigged
 * so that row reads back as the position — `speechEmb` all zeros, `speechPos`
 * row r filled with r — which turns "which row did it ask for" into a value
 * assertion rather than a mock-argument one.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { Engine } from "../engine.js";
import { LRSamplerV1 } from "../sampler.js";
import { Session } from "../session.js";
import { algorithmFromManifest, type AlgorithmConfig, type VoiceProfile } from "../types.js";

const ROW = 1024;
/** Conditioning is a fixed 34 slots, and the framed text is `[255] + t + [0]`. */
const COND_ROWS = 34;
const TEXT_TOKENS = [7];
const TEXT_ROWS = TEXT_TOKENS.length + 2;
/** Repeats a token, so the repetition mask has something to have been seeded with. */
const PREFIX = [3, 5, 3, 7];

/** A small vocabulary: the fakes below allocate logits per position. */
const CONFIG: AlgorithmConfig = algorithmFromManifest({
  speech_vocab_size: 64,
  speech_tokens: { start: 40, stop: 41 },
});

/** What the stand-in step graph was handed on the first generated token. */
interface Recorded {
  prefillEmbeds: Float32Array;
  prefillLen: number;
  stepEmbeds: Float32Array;
  stepPosition: bigint;
  seenAtFirstStep: Uint8Array;
}

function fakeSession(run: (feeds: Record<string, { data: unknown }>) => Record<string, unknown>): Session {
  return {
    inNames: [],
    outNames: ["out"],
    run: (f: never) => Promise.resolve(run(f)),
  } as unknown as Session;
}

/**
 * An engine with rigged tables and recording sessions, built without `load`.
 *
 * `Object.create` rather than a test-only constructor: the fields below are the
 * whole of what `generate` touches, and widening the engine's surface to reach
 * them would be a production change made for one assertion.
 */
function harness(): { engine: Engine; recorded: Recorded } {
  const recorded = {} as Recorded;
  const vocab = CONFIG.speechVocabSize;

  const speechPos = new Float32Array(64 * ROW);
  for (let r = 0; r < 64; r++) speechPos.fill(r, r * ROW, (r + 1) * ROW);

  const kv = (prefix: string) => {
    const out: Record<string, unknown> = {};
    for (let i = 0; i < 16; i++) {
      out[`${prefix}_k_${i}`] = { data: new Float32Array(4 * 64) };
      out[`${prefix}_v_${i}`] = { data: new Float32Array(4 * 64) };
    }
    return out;
  };

  const engine = Object.create(Engine.prototype) as Engine;
  Object.assign(engine as unknown as Record<string, unknown>, {
    config: CONFIG,
    tables: {
      textEmb: new Float32Array(256 * ROW),
      textPos: new Float32Array(64 * ROW),
      speechEmb: new Float32Array(64 * ROW),
      speechPos,
    },
    cond: fakeSession(() => ({ out: { data: new Float32Array(COND_ROWS * ROW) } })),
    prefill: fakeSession((feeds) => {
      recorded.prefillEmbeds = feeds.embeds.data as Float32Array;
      recorded.prefillLen = recorded.prefillEmbeds.length / ROW;
      return { logits: { data: new Float32Array(recorded.prefillLen * vocab) }, ...kv("kv") };
    }),
    step: fakeSession((feeds) => {
      recorded.stepEmbeds = feeds.embeds.data as Float32Array;
      recorded.stepPosition = (feeds.position.data as BigInt64Array)[0];
      return { logits: { data: new Float32Array(vocab) }, ...kv("present") };
    }),
  });

  return { engine, recorded };
}

const VOICE: VoiceProfile = {
  name: "test",
  speakerEmbedding: new Float32Array(256),
  flowEmbedding: new Float32Array(192),
  promptTokens: new BigInt64Array(0),
  promptMel: new Float32Array(0),
  condPromptTokens: new BigInt64Array(0),
  sourceSampleRate: 24_000,
  language: "en",
};

/**
 * Returns a fixed token, and keeps the mask it was handed on step 0.
 *
 * A real sampler would answer from the zero logits above, which pins the test
 * to the sampling law rather than to the position index under test.
 */
function fixedSampler(recorded: Recorded, token: number): LRSamplerV1 {
  return {
    call: (_logits: Float32Array, step: number, seen: Uint8Array) => {
      if (step === 0) recorded.seenAtFirstStep = seen.slice();
      return token;
    },
  } as unknown as LRSamplerV1;
}

test("a generated token gets the row after the prefix, not row one", async () => {
  const { engine, recorded } = harness();
  // One token, so the loop makes exactly one step call and `step` is 0 there:
  // `step + 1` and `len(prefix) + step + 1` differ by the whole prefix.
  await engine.generate(TEXT_TOKENS, VOICE, fixedSampler(recorded, 9), 1, undefined, PREFIX);

  assert.equal(recorded.stepEmbeds.length, ROW);
  assert.equal(recorded.stepEmbeds[0], PREFIX.length + 1);
  assert.notEqual(recorded.stepEmbeds[0], 1);
});

test("the prefill is the authority: it writes the prefix at 1..P", async () => {
  const { engine, recorded } = harness();
  await engine.generate(TEXT_TOKENS, VOICE, fixedSampler(recorded, 9), 1, undefined, PREFIX);

  // Not a convention this test invents. Everything before the BOS row is zero
  // here, so each prefix row reads back as the position the prefill chose for
  // it, and the first free row is the one the step above must ask for.
  const bos = COND_ROWS + TEXT_ROWS;
  assert.equal(recorded.prefillLen, bos + 1 + PREFIX.length);
  assert.equal(recorded.prefillEmbeds[bos * ROW], 0);
  for (let i = 0; i < PREFIX.length; i++) {
    assert.equal(recorded.prefillEmbeds[(bos + 1 + i) * ROW], i + 1);
  }
});

test("the RoPE position stays contiguous with the prefill", async () => {
  const { engine, recorded } = harness();
  await engine.generate(TEXT_TOKENS, VOICE, fixedSampler(recorded, 9), 1, undefined, PREFIX);

  // The transformer's own positions were never wrong, and a fix that moved
  // them would be a second bug wearing the first one's fix.
  assert.equal(recorded.stepPosition, BigInt(recorded.prefillLen));
});

test("the repetition mask is seeded from the prefix", async () => {
  const { engine, recorded } = harness();
  await engine.generate(TEXT_TOKENS, VOICE, fixedSampler(recorded, 9), 1, undefined, PREFIX);

  // A token repeated across a join is as repeated as one within a chunk. Rust
  // and Go allocate this mask and go straight into the loop; JS seeds it at
  // engine.ts:464, and this is that claim checked rather than assumed.
  for (const t of PREFIX) assert.equal(recorded.seenAtFirstStep[t], 1);
  assert.equal(recorded.seenAtFirstStep[9], 0);
  assert.equal(recorded.seenAtFirstStep.reduce((n, v) => n + v, 0), new Set(PREFIX).size);
});

test("with no prefix the first generated token is row one", async () => {
  const { engine, recorded } = harness();
  await engine.generate(TEXT_TOKENS, VOICE, fixedSampler(recorded, 9), 1, undefined, []);

  // The single-window case, which is why the bug survived: here the two
  // expressions agree, and every fixture that pins one chunk passes either way.
  assert.equal(recorded.stepEmbeds[0], 1);
  assert.equal(recorded.seenAtFirstStep.reduce((n, v) => n + v, 0), 0);
});
