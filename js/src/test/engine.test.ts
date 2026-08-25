/**
 * End-to-end engine test against the conformance fixture (needs the
 * checkpoint + exported ONNX graphs + the reference voice).
 *
 * Skips when the assets are absent; set the env vars below to point at them.
 * This is the JS analogue of `tests/test_conformance.py::TestEndToEnd`.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert";

import { Engine } from "../engine.js";
import type { ExecutionOptions } from "../execution.js";
import { loadVoice } from "../voice.js";
import { LRSamplerV1 } from "../sampler.js";

const CKPT = process.env.LOUDKIT_CKPT;
const ONNX_DIR = process.env.LOUDKIT_ONNX_DIR;
const VOICE = process.env.LOUDKIT_VOICE;
const TOKENIZER = process.env.LOUDKIT_TOKENIZER;

const available = [CKPT, ONNX_DIR, VOICE, TOKENIZER].every((p) => p && existsSync(p));

// CPU, not `"auto"`. These cases assert exact fixture tokens, and `"auto"` is
// CoreML on an arm64 Mac and CUDA on a CUDA runner, which would make the same
// assertion mean a different thing on each machine. What a GPU provider does
// to these numbers is a measurement to record, not something to discover by
// watching this file go red.
const EXECUTION: ExecutionOptions = { onnxProvider: "cpu" };

if (!available && process.env.LOUDKIT_REQUIRE_ASSETS && process.env.LOUDKIT_REQUIRE_ASSETS !== "0") {
  // A skipped conformance test and a passing one look identical in a summary
  // line. On a runner that is supposed to have the assets, missing ones are a
  // broken environment — same switch, same meaning, as the Python suite's
  // requires() and the Go and Rust conformance tests.
  throw new Error(
    "LOUDKIT_REQUIRE_ASSETS is set but LOUDKIT_CKPT/ONNX_DIR/VOICE/TOKENIZER are not all present"
  );
}

/** The shared fixture, resolved by walking up to the repo root. */
function fixtureDir(): string {
  if (process.env.LOUDKIT_FIXTURE_DIR) return process.env.LOUDKIT_FIXTURE_DIR;
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "tests", "data", "conformance");
    if (existsSync(join(candidate, "vectors.json"))) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error("cannot locate tests/data/conformance: set LOUDKIT_FIXTURE_DIR");
}

/** The end-to-end case the tests below replay, by name. */
function endToEndCase(name: string): any {
  const vectors = JSON.parse(readFileSync(join(fixtureDir(), "vectors.json"), "utf8"));
  const found = vectors.end_to_end.find((c: any) => c.name === name);
  if (!found) throw new Error(`fixture has no end_to_end case ${name}`);
  return found;
}

/**
 * Pearson correlation, on the explicit condition that the two are the same
 * length. Correlating `Math.min(...)` samples scores a truncated render
 * perfectly against the prefix it did produce — and the length is the finding
 * in that case, not a detail to absorb.
 */

test("engine synthesises over the ONNX graphs", { skip: !available && "set LOUDKIT_CKPT/ONNX_DIR/VOICE/TOKENIZER" }, async () => {
  const engine = await Engine.load(CKPT!, ONNX_DIR!, TOKENIZER!, EXECUTION);
  const voice = loadVoice(VOICE!);
  const result = await engine.synthesize(
    "The quick brown fox jumps over the lazy dog.",
    voice,
    4242
  );
  assert.equal(result.tokens.length, 79);
  assert.ok(result.audio.length > 0);
  assert.ok(result.audio.every((v) => Number.isFinite(v)));
  // The truncation flag rides every result: Python declares every transport
  // must report hit_token_cap, because silent truncation presented as
  // complete audio reads as complete to an agent. A sentence ending at its
  // stop token, well under the cap, must report false.
  assert.equal(result.hitCap, false);
});

test("free-run tokens match the Python reference", { skip: !available }, async () => {
  // The token *values*, not their count. Asserting `length === 79` passes for
  // any 79 tokens whatsoever — including a completely different reading — and
  // the whole claim of this binding is that it samples the same stream as
  // Python, which only the values can show.
  const c = endToEndCase("s0");
  const engine = await Engine.load(CKPT!, ONNX_DIR!, TOKENIZER!, EXECUTION);
  const voice = loadVoice(VOICE!);
  const sampler = new LRSamplerV1(engine.config.sampling, c.seed);
  const raw = await engine.generate(
    engine.encode(c.text, c.language),
    voice,
    sampler
  );
  const stripped = raw.filter((t) => t < engine.config.startSpeechToken);
  assert.deepEqual(stripped, c.tokens);
});

test("render is bit-identical on repeat", { skip: !available }, async () => {
  const engine = await Engine.load(CKPT!, ONNX_DIR!, TOKENIZER!, EXECUTION);
  const voice = loadVoice(VOICE!);
  const tokens = [
    3943, 1272, 2264, 1083, 573, 2835, 4582, 4849, 2006, 1951, 2112, 2166,
    4357, 3863, 3731, 5843, 5357, 3584, 4315, 6426, 6537, 6453, 2845, 5027,
    602, 2060, 2032, 5211, 4590, 3543, 2031, 5325, 3156, 5999, 658, 2127,
    2059, 4755, 5645, 46, 2195, 2330, 3155, 6534, 801, 1848, 3456, 2243,
    3799, 5986, 2192, 2465, 5012, 5183, 5726, 2256, 5669, 1131, 1761, 159,
    4592, 5051, 2867, 728, 2184, 486, 162, 1539, 4299, 6486, 6405, 6405, 6405,
    6405, 6405, 6405, 6405, 6405, 6081,
  ];
  const seed1 = engine["deriveSeed"](4242, 1);
  const seed2 = engine["deriveSeed"](4242, 2);
  const a = await engine.decodeMel(tokens, voice, seed1);
  const b = await engine.decodeMel(tokens, voice, seed1);
  assert.deepEqual(Array.from(a), Array.from(b));
  // A "render" that stops at the mel tests two of the three stages. The
  // vocoder draws its own excitation noise from its own seeded sub-stream, so
  // it is the stage most able to differ between two runs and the one this test
  // was named for.
  const wa = await engine.vocode(a, seed2);
  const wb = await engine.vocode(b, seed2);
  assert.equal(wa.length, wb.length);
  assert.deepEqual(Array.from(wa), Array.from(wb));
});

test(
  "stream and synthesizeLong are one loop, not two",
  { skip: !available && "set LOUDKIT_CKPT/ONNX_DIR/VOICE/TOKENIZER" },
  async () => {
    // The whole-passage path is the streaming path with the chunks
    // concatenated. If they ever become two loops they will drift, and the
    // drift will be inaudible until a join lands somewhere different — so the
    // equality is asserted rather than assumed.
    const engine = await Engine.load(CKPT!, ONNX_DIR!, TOKENIZER!, EXECUTION);
    const voice = loadVoice(VOICE!);
    // Comfortably past one window. The budget is
    // floor(max_tokens * CHARS_PER_TOKEN) = 127 characters, so a passage that
    // merely feels long can still arrive as a single chunk and make this test
    // assert nothing — which is what the first draft of it did.
    const text =
      "The first sentence sets the scene and runs on for a while. " +
      "The second sentence follows it and is no shorter than the first one was. " +
      "The third sentence exists so that the splitter has somewhere to breathe. " +
      "The fourth sentence closes the passage without hurrying.";

    const whole = await engine.synthesizeLong(text, voice, 7, "en");

    const pieces: Float32Array[] = [];
    const tokens: number[] = [];
    for await (const chunk of engine.stream(text, voice, 7, "en")) {
      pieces.push(chunk.audio);
      tokens.push(...chunk.tokens);
      // The flag is per chunk here and ORed across chunks on the joined
      // result; a normal passage ends at stop tokens, so both are false.
      assert.equal(chunk.hitCap, false);
    }
    assert.equal(whole.hitCap, false);
    assert.ok(pieces.length > 1, "the passage must actually split");

    const total = pieces.reduce((n, a) => n + a.length, 0);
    assert.equal(total, whole.audio.length, "streamed and joined lengths differ");
    assert.deepEqual(tokens, whole.tokens, "streamed and joined tokens differ");

    let at = 0;
    for (const piece of pieces) {
      for (let i = 0; i < piece.length; i++) {
        assert.equal(piece[i], whole.audio[at + i], `sample ${at + i} differs`);
      }
      at += piece.length;
    }
  }
);
