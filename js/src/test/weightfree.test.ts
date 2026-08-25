/**
 * Weight-free conformance vectors: RNG, sampler, frontend, seeds.
 *
 * These run against `tests/data/conformance/vectors.json` from the loudkit
 * repo — the same fixture the Python and Swift suites verify. A drift in any
 * vector here is a broken port, not "close enough": the sampling law and the
 * tokenizer are what make free-running output identical across languages.
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { splitText } from "../chunking.js";
import { concatMelAlongTime } from "../engine.js";
import { PRODUCTION_CHUNKING } from "../types.js";
import assert from "node:assert";

import { uniforms, gumbelNoise, philox4x32, normalizeSeed } from "../rng.js";
import { LRSamplerV1 } from "../sampler.js";
import { GraphemeTextFrontend } from "../frontend.js";
import { algorithmFromManifest } from "../types.js";
import { canonicalForm, fingerprint } from "../fingerprint.js";

/**
 * Locate the shared conformance fixture, which lives in the loudkit repo root
 * under `tests/data/conformance/`. The path is resolved relative to this file
 * (walking up to the repo root) rather than the process CWD, so the documented
 * `cd js && npm test` works from anywhere. An explicit
 * `LOUDKIT_FIXTURE`/`LOUDKIT_TOKENIZER` still wins.
 */
function fixturePath(env: string | undefined, name: string): string {
  if (env) return env;
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "tests", "data", "conformance", name);
    if (existsSync(candidate)) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error(
    `cannot locate ${name}: set LOUDKIT_FIXTURE/LOUDKIT_TOKENIZER or run from the loudkit repo`
  );
}

const FIXTURE = fixturePath(process.env.LOUDKIT_FIXTURE, "vectors.json");
const SPEECHTEXT_FIXTURE = fixturePath(undefined, "speechtext.json");
const TOKENIZER = fixturePath(process.env.LOUDKIT_TOKENIZER, "tokenizer.json");

function vectors(): any {
  return JSON.parse(readFileSync(FIXTURE, "utf8"));
}

/**
 * The cases for one fixture section, refusing an empty list.
 *
 * Every loop below iterates a slice pulled out of the fixture by key. A
 * regeneration that renamed one — `philox` to `rng`, say — would leave the loop
 * comparing nothing and the test reporting a pass, switching the entire
 * cross-language determinism claim off silently.
 */
function casesOf(section: any, key: string): any[] {
  const list = section?.[key];
  assert.ok(Array.isArray(list), `the fixture has no ${key} section; nothing was compared`);
  assert.ok(list.length > 0, `fixture section ${key} is empty; nothing was compared`);
  return list;
}

test("philox known-answer vectors", () => {
  const d = casesOf(vectors().philox, "kat");
  for (const c of d) {
    const got = philox4x32(
      Number(c.counter[0]),
      Number(c.counter[1]),
      Number(c.counter[2]),
      Number(c.counter[3]),
      c.key[0],
      c.key[1]
    );
    assert.deepEqual(got, c.expected, `counter ${c.counter}`);
  }
});

test("uniform bits match the fixture exactly", () => {
  const d = casesOf(vectors().philox, "uniform_bits");
  for (const p of d) {
    const seed = BigInt(p.seed);
    const u = uniforms(seed, p.stream, p.step0, p.n_steps, p.width);
    const got = Array.from(u, (x) => BigInt(Math.round(x * 4294967296 - 0.5)));
    const want = p.bits.flat();
    assert.deepEqual(got, want, `seed ${p.seed}`);
  }
});

test("gumbel noise matches the fixture", () => {
  const d = casesOf(vectors().philox, "gumbel");
  for (const p of d) {
    const g = gumbelNoise(BigInt(p.seed), p.stream, p.step, 1, p.width);
    for (let i = 0; i < p.values.length; i++) {
      const rel = Math.abs((g[i] - p.values[i]) / p.values[i]);
      assert.ok(rel < 1e-12, `seed ${p.seed} idx ${i} rel ${rel}`);
    }
  }
});

test("sampler token choices match the fixture", () => {
  const d = casesOf(vectors().sampler, "cases");
  for (const c of d) {
    const cfg = c.config;
    const config = {
      temperature: cfg.temperature,
      repetitionPenalty: cfg.repetition_penalty,
      minP: cfg.min_p,
      maxNewTokens: cfg.max_new_tokens ?? 255,
      silenceTokenIds: cfg.silence_token_ids,
      minTokensFloor: 0,
      minTokensTextRatio: 0.0,
    };
    const sampler = new LRSamplerV1(config, c.seed);
    // build logit rows the way the fixture's recipe specifies
    let rows: number[][];
    if (c.logits_recipe) {
      const r = c.logits_recipe;
      rows = [];
      for (let step = 0; step < r.steps; step++) {
        const u = uniforms(BigInt(r.seed), r.stream, step, 1, r.vocab);
        const row: number[] = [];
        for (let i = 0; i < r.vocab; i++) row.push(u[i] * r.scale + r.offset);
        rows.push(row);
      }
    } else {
      const row = c.logits[0];
      rows = new Array(c.repeat_logits ?? c.logits.length).fill(row);
    }
    const seen = new Uint8Array(rows[0].length);
    const got: number[] = [];
    for (let step = 0; step < rows.length; step++) {
      const token = sampler.call(new Float32Array(rows[step]), step, seen);
      got.push(token);
      seen[token] = 1;
    }
    assert.deepEqual(got, c.expected, c.name);
  }
});

test("frontend token ids match the fixture", () => {
  const d = casesOf(vectors().frontend, "cases");
  const frontend = new GraphemeTextFrontend(TOKENIZER);
  for (const c of d) {
    const ids = frontend.encode(c.text, c.language);
    assert.deepEqual(ids, c.ids, c.text);
  }
});

test("seed derivation matches the fixture", () => {
  const PHI = 0x9e3779b97f4a7c15n;
  const PSI = 0xbf58476d1ce4e5b9n;
  const MASK = (1n << 64n) - 1n;
  const d = casesOf(vectors().seeds, "derivation");
  for (const p of d) {
    const derived = (BigInt(p.seed) * PHI + BigInt(p.stream) * PSI) & MASK;
    assert.equal(derived.toString(16), p.derived.slice(2), `seed ${p.seed} stream ${p.stream}`);
  }
});

test("production algorithm matches Python/Swift (prefix 6, recipe loudkit-1)", () => {
  // The prefix_tokens default changed 0 -> 6 and the artifact detectors landed
  // on top of it; Python and Swift call the result loudkit-1, and the JS port
  // reconstructs the same canonical form and must not drift. This pins the
  // values so a future change to the other backends fails here too.
  const algo = algorithmFromManifest({});
  assert.equal(algo.chunking.prefixTokens, 6);
  assert.equal(algo.recipeVersion, "loudkit-1");
  assert.equal(algo.chunking.maxTokens, 255);
  assert.equal(algo.postprocess.mode, "trim");
});

test("the one recipe is accepted and a foreign tag is refused by name", () => {
  // One recipe means one accepted value. Believing a foreign tag would
  // fingerprint it; defaulting it would claim this recipe for a checkpoint
  // that named another. All five ports refuse it identically.
  const algo = algorithmFromManifest({ recipe_version: "loudkit-1", chunking: {} });
  assert.equal(algo.recipeVersion, "loudkit-1");
  assert.throws(
    () => algorithmFromManifest({ recipe_version: "loudkit-9" }),
    /recipe_version "loudkit-9".*only recipe/
  );
  // Not even a string: refused, not defaulted. A manifest one port misreads
  // while another defaults is the divergence class this library exists to
  // prevent.
  assert.throws(() => algorithmFromManifest({ recipe_version: 9 }), /recipe_version 9/);
});

test("an unknown postprocess mode is refused", () => {
  assert.throws(
    () => algorithmFromManifest({ postprocess: { mode: "shave" } }),
    /unknown postprocess mode/
  );
});

test("seeds above 2^53 are refused rather than silently rounded", () => {
  // Every other binding takes a full 64-bit seed. A JS number is a double, so
  // a seed past MAX_SAFE_INTEGER rounds on the way in and addresses a
  // different Philox counter than Python/Go/Rust/Swift for the "same" seed —
  // breaking "same seed, same tokens" with no error.
  assert.throws(() => normalizeSeed(2 ** 60), /exceeds Number\.MAX_SAFE_INTEGER/);
  assert.throws(() => normalizeSeed(1.5), /must be an integer/);
  assert.throws(() => normalizeSeed(-1), /\[0, 2\^64\)/);
  assert.throws(() => normalizeSeed(1n << 64n), /\[0, 2\^64\)/);

  // A bigint carries the full range exactly, and small numbers are unchanged.
  assert.equal(normalizeSeed(1n << 60n), 1n << 60n);
  assert.equal(normalizeSeed(7), 7n);
});

test("an unknown or unimplemented guidance mode is refused", () => {
  // Python raises here and Swift throws; a port that casts the field and then
  // never reads it renders single-path audio for a checkpoint that asked
  // for something else.
  assert.throws(() => algorithmFromManifest({ guidance: "nonsense" }), /unknown guidance mode/);
  assert.throws(
    () => algorithmFromManifest({ guidance: "cfg_dual_path" }),
    /does not.*implement/s
  );
  assert.equal(algorithmFromManifest({ guidance: "single_path" }).guidance, "single_path");
});

test("the splitter cuts where the shared fixture says", () => {
  // Where the splits fall is audible —
  // a break at a full stop is inaudible, a break mid-clause is not — so a
  // different split is a different reading, not a formatting choice.
  const cases = JSON.parse(readFileSync(SPEECHTEXT_FIXTURE, "utf8")).chunking as Array<{
    config: string;
    max_tokens: number;
    prefix_tokens: number;
    split_on: string[];
    text: string;
    chunks: string[];
  }>;
  assert.ok(cases?.length, "the fixture carries no chunking cases");
  for (const c of cases) {
    const got = splitText(c.text, {
      enabled: true,
      maxTokens: c.max_tokens,
      prefixTokens: c.prefix_tokens,
      splitOn: c.split_on,
    });
    assert.deepEqual(got, c.chunks, `${c.config}: ${JSON.stringify(c.text.slice(0, 40))}`);
  }
});

test("long-form mel is concatenated along time, not appended end to end", () => {
  // A mel is row-major [80, frames]. Appending two flat buffers puts the
  // second chunk's bin 0 after the first chunk's bin 79, so every row but the
  // first is wrong. The audio is unaffected — each chunk is vocoded on its own
  // — but the mel is the diagnostic people reach for when two backends
  // disagree, and a mis-shaped one sends them looking in the wrong place.
  const bins = 80;
  // Two chunks whose values encode (bin, frame) so a wrong layout is visible.
  const make = (frames: number, offset: number) => {
    const m = new Float32Array(bins * frames);
    for (let b = 0; b < bins; b++) {
      for (let f = 0; f < frames; f++) m[b * frames + f] = b * 1000 + offset + f;
    }
    return m;
  };
  const joined = concatMelAlongTime([make(3, 0), make(2, 100)]);
  assert.equal(joined.length, bins * 5);
  for (let b = 0; b < bins; b++) {
    assert.deepEqual(
      Array.from(joined.subarray(b * 5, (b + 1) * 5)),
      [b * 1000 + 0, b * 1000 + 1, b * 1000 + 2, b * 1000 + 100, b * 1000 + 101],
      `row ${b} is not this bin's frames in order`
    );
  }
});

test("the manifest's chunking block is read, not assumed", () => {
  // A checkpoint that declares its own boundaries and a runtime that silently
  // uses different ones agree on recipe_version and disagree on the reading.
  const cfg = algorithmFromManifest({
    format: "loudkit-checkpoint",
    format_version: 1,
    chunking: { enabled: false, max_tokens: 99, prefix_tokens: 3, split_on: ["|"] },
  });
  assert.equal(cfg.chunking.enabled, false);
  assert.equal(cfg.chunking.maxTokens, 99);
  assert.equal(cfg.chunking.prefixTokens, 3);
  assert.deepEqual(cfg.chunking.splitOn, ["|"]);

  // A manifest that says nothing keeps the shipping recipe.
  const shipping = algorithmFromManifest({ format: "loudkit-checkpoint", format_version: 1 });
  assert.deepEqual(shipping.chunking, PRODUCTION_CHUNKING);
});

test("the manifest's chunking recipe is validated, not trusted", () => {
  // Python refuses these four; the ports were plain structs that read
  // max_tokens straight from the manifest. The zero-budget one is the reason
  // it matters: `splitText` cuts nothing and loops forever, which on a server
  // is a wedged request holding the single-flight engine (d8742aa).
  assert.throws(
    () => algorithmFromManifest({ chunking: { max_tokens: 0 } }),
    /max_tokens must be positive/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { max_tokens: 1, prefix_tokens: 0 } }),
    /no character budget/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { max_tokens: 20, prefix_tokens: 20 } }),
    /prefix_tokens must be in/
  );
  // A valid recipe still loads.
  assert.equal(
    algorithmFromManifest({ chunking: { max_tokens: 20, prefix_tokens: 6 } }).chunking.maxTokens,
    20
  );
});

test("the manifest's sampling values are validated, not trusted", () => {
  // Python's `SamplingConfig.__post_init__` refuses all four. A manifest one
  // port refuses and another accepts is two renders under one fingerprint —
  // and the failure modes are silent: temperature 0 divides by zero, min_p 1
  // empties the candidate set, a negative EOS floor lets a chunk stop on the
  // first token.
  assert.throws(
    () => algorithmFromManifest({ sampling_defaults: { temperature: 0 } }),
    /temperature out of range/
  );
  assert.throws(
    () => algorithmFromManifest({ sampling_defaults: { repetition_penalty: 0.9 } }),
    /repetition_penalty out of range/
  );
  assert.throws(
    () => algorithmFromManifest({ sampling_defaults: { min_p: 1 } }),
    /min_p out of range/
  );
  assert.throws(
    () => algorithmFromManifest({ eos_floor: { min_tokens_floor: -1 } }),
    /min_tokens_floor must be >= 0/
  );
  assert.throws(
    () => algorithmFromManifest({ eos_floor: { min_tokens_text_ratio: -0.5 } }),
    /min_tokens_text_ratio must be >= 0/
  );
  // Zero is a configuration, not a typo: it disables the floor.
  const off = algorithmFromManifest({
    eos_floor: { min_tokens_floor: 0, min_tokens_text_ratio: 0 },
  });
  assert.equal(off.sampling.minTokensFloor, 0);
  assert.equal(off.sampling.minTokensTextRatio, 0);
});

test("an explicit euler_grid is read, not thrown away", () => {
  // `timeGrid` honoured `eulerGrid`, so this port looked like the one with
  // explicit-grid support — while the manifest parser hard-coded `null` before
  // `timeGrid` ever saw it. An explicit grid exists precisely because "cosine"
  // is a formula two codebases can write two ways (config.py:296).
  assert.deepEqual(algorithmFromManifest({ euler_grid: [0, 0.25, 1] }).eulerGrid, [0, 0.25, 1]);
  assert.equal(algorithmFromManifest({}).eulerGrid, null);
  assert.equal(algorithmFromManifest({ euler_grid: null }).eulerGrid, null);
  assert.throws(() => algorithmFromManifest({ euler_grid: "cosine" }), /euler_grid/);
});

test("a manifest sequence field rejects a string", () => {
  // `str` is a sequence of characters in every one of these languages, so
  // `"123"` passed a "is it a list" check that was really a cast and was then
  // iterated. Python refuses it by name; so does this.
  assert.throws(
    () => algorithmFromManifest({ silence_token_ids: "123" }),
    /silence_token_ids/
  );
  assert.deepEqual(
    algorithmFromManifest({ silence_token_ids: [1, 2] }).sampling.silenceTokenIds,
    [1, 2]
  );
});

test("the algorithm fingerprint matches the shared fixture", () => {
  // Every other check in this file compares a behaviour somebody thought to
  // compare. This compares the whole configuration in one string, so a field
  // nobody wrote a test for still cannot drift — which is not hypothetical:
  // `eulerGrid` was hard-coded to null here while `timeGrid` honoured it, and
  // `tokenRateHz` was hard-coded too until this test asked for it.
  const algorithm = vectors().algorithm;
  assert.ok(algorithm, "the fixture has no algorithm section; nothing was compared");

  const cfg = algorithmFromManifest({
    recipe_version: "loudkit-1",
    guidance: "single_path",
    guidance_rate: 0.0,
    n_cfm_timesteps: 2,
    sample_rate: 24_000,
    token_rate_hz: 25.0,
    speech_vocab_size: 8194,
    speech_tokens: { start: 6561, stop: 6562 },
    window: {
      max_speech_tokens: 255,
      static_length: 255,
      pad_token_id: 4254,
      static_prompt_tokens: 238,
    },
    sampling_defaults: {
      temperature: 0.8,
      repetition_penalty: 1.2,
      min_p: 0.05,
      max_new_tokens: 255,
    },
    eos_floor: { min_tokens_floor: 10, min_tokens_text_ratio: 1.2 },
    silence_token_ids: [
      1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927, 3928, 3930, 4008,
      4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173, 4174, 4218, 4245, 4251, 4252, 4254,
      4255, 4260, 4282,
    ],
  });

  // The blob first: a mismatch there names the field that drifted, while a
  // mismatch in the hash alone says only that something did.
  assert.equal(canonicalForm(cfg), algorithm.canonical_form);
  assert.equal(fingerprint(cfg), algorithm.fingerprint);
});
