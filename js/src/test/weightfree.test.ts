/**
 * Weight-free conformance vectors: RNG, sampler, frontend, seeds.
 *
 * These run against `tests/data/conformance/vectors.json` from the loudkit
 * repo, the same fixture the Python and Swift suites verify. A drift in any
 * vector here is a broken port, not "close enough": the sampling law and the
 * tokenizer are what make free-running output identical across languages.
 */

import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { splitText } from "../chunking.js";
import { concatMelAlongTime, carryFrom } from "../engine.js";
import { PRODUCTION_CHUNKING } from "../types.js";
import {
  Checkpoint,
  SUPPORTED_DECODE_MODES,
  SUPPORTED_FORMAT_VERSIONS,
} from "../checkpoint.js";
import assert from "node:assert";

import { uniforms, gumbelNoise, philox4x32, normalizeSeed, deriveSeed } from "../rng.js";
import { LRSamplerV1 } from "../sampler.js";
import { GraphemeTextFrontend } from "../frontend.js";
import {
  PRODUCTION_EOS_FLOOR,
  PRODUCTION_EOS_TEXT_RATIO,
  algorithmFromManifest,
  productionWindow,
} from "../types.js";
import { requireStaticWindow, timeGrid } from "../windowing.js";
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
 * regeneration that renamed one (`philox` to `rng`, say) would leave the loop
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

test("the two characters the whitespace classes disagree about encode as Python", () => {
  // The tokenizer's word split drops whitespace and keeps everything else.
  // This port reads that class through a JavaScript regex and the reference
  // reads it through the Rust crate behind `tokenizers`, and the two classes
  // differ in exactly two characters, in opposite directions: U+FEFF is
  // whitespace only here, U+0085 NEL only there. So U+FEFF vanished where the
  // other four ports answer [UNK], and U+0085 answered an [UNK] the other four
  // drop. Every id below is Python's, read off
  // `loudkit.frontend.text.GraphemeTextFrontend` against this same tokenizer.
  const frontend = new GraphemeTextFrontend(TOKENIZER);
  const enc = (s: string) => frontend.encode(s, "en");
  assert.deepEqual(enc("ab"), [708, 109]);
  // U+FEFF is a symbol there, so it survives as an [UNK] between the letters,
  // and it splits "ab" into two words while it does so.
  assert.deepEqual(enc("a\uFEFFb"), [708, 14, 1, 15]);
  assert.deepEqual(enc("a\uFEFF\uFEFFb"), [708, 14, 1, 1, 15]);
  // U+0085 is whitespace there, so it leaves no token, only the split.
  assert.deepEqual(enc("a\u0085b"), [708, 14, 15]);
  // A run of symbols is one word, and the fix has to be a swap rather than two
  // independent rewrites, or a run mixing the pair would group the wrong way.
  assert.deepEqual(enc("a\uFEFF!b"), [708, 14, 1, 3, 15]);
  assert.deepEqual(enc("a\uFEFF\u0085b"), [708, 14, 1, 15]);
  assert.deepEqual(enc("a\u0085\uFEFFb"), [708, 14, 1, 15]);
});

test("seed derivation matches the fixture", () => {
  // The shipping function, not a second copy of its formula: this used to
  // declare PHI and PSI itself, so mutating the constants `Engine` actually
  // uses left it green.
  const d = casesOf(vectors().seeds, "derivation");
  for (const p of d) {
    const derived = deriveSeed(p.seed, p.stream);
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

test("a manifest scalar that is not a number is refused by key", () => {
  // `as number` on a JSON value is a claim, not a check: `"255"` reached the
  // window as a string and `2 * "255"` as concatenation, with nothing naming
  // the key. Python refuses each of these by name.
  assert.throws(
    () => algorithmFromManifest({ window: { max_speech_tokens: "255" } }),
    /manifest\['window'\]\['max_speech_tokens'\] should be a number, got "255"/
  );
  assert.throws(
    () => algorithmFromManifest({ token_rate_hz: "25" }),
    /manifest\['token_rate_hz'\] should be a number/
  );
  assert.throws(
    () => algorithmFromManifest({ sampling_defaults: { temperature: null } }),
    /manifest\['sampling_defaults'\]\['temperature'\] should be a number/
  );
  assert.throws(
    () => algorithmFromManifest({ speech_tokens: { start: "6561" } }),
    /manifest\['speech_tokens'\]\['start'\] should be a number/
  );
  // The window's three optional keys take null, which is the ragged window
  // said explicitly, and refuse anything else.
  const ragged = algorithmFromManifest({ window: { static_length: null } });
  assert.equal(ragged.window.staticLength, null);
  assert.throws(
    () => algorithmFromManifest({ window: { static_length: "4254" } }),
    /should be a number or null/
  );
});

test("the two postprocess range checks the reference carries are refused here too", () => {
  // Finite is not enough. A value the reference refuses and this port accepts
  // is one manifest rendering two ways under one fingerprint.
  assert.throws(
    () => algorithmFromManifest({ postprocess: { retry_max_attempts: 8 } }),
    /retry_max_attempts must be in \[0, 8\)/
  );
  assert.throws(
    () => algorithmFromManifest({ postprocess: { retry_max_attempts: -1 } }),
    /retry_max_attempts must be in \[0, 8\)/
  );
  assert.throws(
    () => algorithmFromManifest({ postprocess: { repetition_min_cycles: 1 } }),
    /repetition_min_cycles must be at least 2/
  );
  assert.equal(
    algorithmFromManifest({ postprocess: { retry_max_attempts: 7 } }).postprocess
      .retryMaxAttempts,
    7
  );
});

test("a chunking key this port does not implement is refused by name", () => {
  // Ignored, it would split the first chunk one way in Python and another way
  // here, under one fingerprint.
  assert.throws(
    () => algorithmFromManifest({ chunking: { first_chunk_max_tokens: 40 } }),
    /first_chunk_max_tokens.*not implemented/
  );
});

test("an unknown postprocess mode is refused", () => {
  assert.throws(
    () => algorithmFromManifest({ postprocess: { mode: "shave" } }),
    /unknown postprocess mode/
  );
});

test("an unknown repetition_resume is refused", () => {
  // A law this port does not implement must not fall back to a default: the
  // resolver would cut where the manifest said to condemn. "cut" names the
  // pre-amendment law and is read.
  assert.throws(
    () => algorithmFromManifest({ postprocess: { repetition_resume: "maybe" } }),
    /unknown repetition_resume/
  );
  const cfg = algorithmFromManifest({ postprocess: { repetition_resume: "cut" } });
  assert.equal(cfg.postprocess.repetitionResume, "cut");
});

test("an unknown repetition_silence is refused", () => {
  // A family this port does not implement must not fall back to a default:
  // the loop exemption would read one silence list under a manifest
  // declaring another. "sampling" names the pre-amendment law and is read.
  assert.throws(
    () => algorithmFromManifest({ postprocess: { repetition_silence: "both" } }),
    /unknown repetition_silence/
  );
  const cfg = algorithmFromManifest({ postprocess: { repetition_silence: "sampling" } });
  assert.equal(cfg.postprocess.repetitionSilence, "sampling");
});

test("render ids inside the postprocess block are refused", () => {
  // The censuses ride the manifest top level, beside silence_token_ids;
  // declaring them inside the block would give one value two homes in one
  // file, so it is refused by name, the same way Python's block reader
  // refuses it.
  for (const key of ["silence_render_ids", "quiet_render_ids"]) {
    assert.throws(
      () => algorithmFromManifest({ postprocess: { [key]: [1, 2] } }),
      new RegExp(key)
    );
  }
});

test("a postprocess constant out of range is refused", () => {
  // `_validate_ranges` runs in the reference's `__post_init__`, so a manifest
  // naming an impossible constant never becomes a config there. This reader
  // took every number it was handed: a negative slack widened the ceiling and
  // a zero run threshold called every row with one leading silence token a
  // stall. Every branch of that method, in its order, with its sentence.
  const cases: [Record<string, unknown>, RegExp][] = [
    [{ retry_max_attempts: 8 }, /retry_max_attempts must be in \[0, 8\): 8\./],
    [{ retry_max_attempts: -1 }, /retry_max_attempts must be in \[0, 8\)/],
    [{ repetition_min_cycles: 1 }, /repetition_min_cycles must be at least 2: 1/],
    [{ repetition_max_period: 0 }, /repetition_max_period must be positive: 0/],
    [{ stall_run_tokens: 0 }, /stall_run_tokens must be positive: 0/],
    [{ stall_run_tokens: -1 }, /stall_run_tokens must be positive: -1/],
    [
      { repetition_min_span: 2 },
      /repetition_min_span \(2\) must be at least repetition_min_cycles \(3\)/,
    ],
    [
      { ceiling_speech_per_text_token: 0 },
      /ceiling_speech_per_text_token must be positive: 0/,
    ],
    [
      { desperation_speech_per_text_token: 4 },
      /desperation_speech_per_text_token \(4\) must exceed ceiling_speech_per_text_token \(4\)/,
    ],
    [
      { desperation_min_keep_per_text_token: -0.1 },
      /desperation_min_keep_per_text_token must be >= 0: -0.1/,
    ],
    [
      { desperation_min_keep_per_text_token: 2.7 },
      /must not exceed desperation_band_ratio \(2.6\)/,
    ],
    [{ trailing_filler_threshold: 0 }, /trailing_filler_threshold must be in \(0, 1\]: 0/],
    [{ trailing_filler_threshold: 1.1 }, /trailing_filler_threshold must be in \(0, 1\]/],
    [{ filler_min_eos_probability: 1 }, /filler_min_eos_probability out of range: 1/],
    [{ filler_min_eos_probability: -0.1 }, /filler_min_eos_probability out of range/],
    [{ ceiling_slack_tokens: -1 }, /ceiling_slack_tokens must be >= 0: -1/],
    [{ dropout_min_tokens: -1 }, /dropout_min_tokens must be >= 0: -1/],
    [{ desperation_band_floor: -1 }, /desperation_band_floor must be >= 0: -1/],
    [{ echo_weak_max_tail: -1 }, /echo_weak_max_tail must be >= 0: -1/],
    [{ echo_strong_min_position_pct: 101 }, /echo_strong_min_position_pct is a percentage: 101/],
    [{ echo_weak_min_position_pct: -1 }, /echo_weak_min_position_pct is a percentage: -1/],
  ];
  for (const [postprocess, message] of cases) {
    assert.throws(() => algorithmFromManifest({ postprocess }), message, JSON.stringify(postprocess));
  }
});

test("the boundary values the reference admits still load", () => {
  // The rules are one-sided, and a validator that also refused the last legal
  // rung would be a second defect wearing the fix's clothes.
  const legal: Record<string, unknown>[] = [
    { retry_max_attempts: 7 }, // RETRY_LADDER_HEADROOM - 1
    { retry_max_attempts: 0 },
    { repetition_min_cycles: 2, repetition_min_span: 2 },
    { repetition_max_period: 1 },
    { stall_run_tokens: 1 },
    { trailing_filler_threshold: 1.0 },
    { filler_min_eos_probability: 0.0 },
    { desperation_min_keep_per_text_token: 2.6 }, // exactly desperation_band_ratio
    { ceiling_slack_tokens: 0 },
    { echo_strong_min_position_pct: 0 },
    { echo_weak_min_position_pct: 100 },
  ];
  for (const postprocess of legal) {
    algorithmFromManifest({ postprocess });
  }
});

test("the top-level censuses are read", () => {
  // With or without a postprocess block: a manifest with no detector
  // overrides still carries the properties of its weights.
  const bare = algorithmFromManifest({
    silence_render_ids: [7, 8],
    quiet_render_ids: [9],
  });
  assert.deepEqual(bare.postprocess.silenceRenderIds, [7, 8]);
  assert.deepEqual(bare.postprocess.quietRenderIds, [9]);

  const withBlock = algorithmFromManifest({
    silence_render_ids: [7],
    postprocess: { stall_run_tokens: 30 },
  });
  assert.equal(withBlock.postprocess.stallRunTokens, 30);
  assert.deepEqual(withBlock.postprocess.silenceRenderIds, [7]);
});

test("seeds above 2^53 are refused rather than silently rounded", () => {
  // Every other binding takes a full 64-bit seed. A JS number is a double, so
  // a seed past MAX_SAFE_INTEGER rounds on the way in and addresses a
  // different Philox counter than Python/Go/Rust/Swift for the "same" seed,
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
  // Where the splits fall is audible: a break at a full stop is inaudible and
  // a break mid-clause is not, so a different split is a different reading,
  // not a formatting choice.
  const cases = JSON.parse(readFileSync(SPEECHTEXT_FIXTURE, "utf8")).chunking as Array<{
    config: string;
    max_tokens: number;
    prefix_tokens: number;
    split_on: string[];
    abbreviations: string[];
    mid_sentence_period: "hold" | "break";
    text: string;
    chunks: string[];
  }>;
  assert.ok(cases?.length, "the fixture carries no chunking cases");
  for (const c of cases) {
    const got = splitText(c.text, {
      enabled: true,
      // Not a splitter input: `capResplit` decides what the engine does with a
      // window that overran, so the shared fixture does not vary it.
      capResplit: "word",
      maxTokens: c.max_tokens,
      prefixTokens: c.prefix_tokens,
      splitOn: c.split_on,
      abbreviations: c.abbreviations,
      midSentencePeriod: c.mid_sentence_period,
    });
    assert.deepEqual(got, c.chunks, `${c.config}: ${JSON.stringify(c.text.slice(0, 40))}`);
  }
});

test("a period that does not end a sentence does not end a chunk", () => {
  // "But Mr. Smith went home" used to break after the title and hand the
  // renderer a seven-character chunk: its own utterance, its own derived seed,
  // and a token ceiling proportional to seven characters. It is the only chunk
  // in a 9920-row rendered census that hit that ceiling. Surveyed over 1200
  // passages in ten languages, 59 of 3773 cuts landed on a period inside a
  // sentence; under this law, one.
  const title =
    "But Mr. Smith went home to the house on the hill where he had lived " +
    "for forty years without ever once complaining about any of it at all.";
  assert.notEqual(splitText(title, PRODUCTION_CHUNKING)[0], "But Mr.");

  // The old law, kept namable so a pack can say what it was measured under.
  const old = { ...PRODUCTION_CHUNKING, midSentencePeriod: "break" as const };
  assert.equal(splitText(title, old)[0], "But Mr.");

  // The half of the law that no list could do: `speech_text` folds a
  // mid-sentence ellipsis to a single period, and the next word is lower case.
  // It is the dominant cause in Polish, which has no abbreviation cuts at all.
  const ellipsis =
    "Grzeja sie i swieca. ciep\u0142em ktore pamietaja z lata i z kazdej " +
    "innej pory roku na swiecie, a potem gasna powoli i nikt juz nie pamieta.";
  assert.ok(!splitText(ellipsis, PRODUCTION_CHUNKING)[0].endsWith("swieca."));

  // Gated on the period. A comma is followed by a lower-case word almost every
  // time it is written, so a rule that did not gate would veto every comma.
  const commas =
    "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu, " +
    "nu xi omicron pi rho, sigma tau upsilon phi chi psi omega at the end.";
  assert.deepEqual(splitText(commas, PRODUCTION_CHUNKING), splitText(commas, old));

  // "NASA" ends in "A", and "A" is a listed initial; the guard on the character
  // in front of the match is what keeps this one breaking.
  const nasa =
    "The rocket that carried them up there was built by NASA. And the rest " +
    "of the afternoon went by without anybody saying much about it to anyone.";
  assert.ok(splitText(nasa, PRODUCTION_CHUNKING)[0].endsWith("by NASA."));

  // Every sentence end in the window is held, so the split falls through to the
  // latest comma. A comma break is heard; a chunk of "Mr." is heard worse.
  const norrell =
    "Mr. Norrell, who had been waiting in the hall for the better part of an " +
    "hour, said nothing at all to either of them about what he had seen there.";
  assert.ok(splitText(norrell, PRODUCTION_CHUNKING)[0].endsWith("hour,"));
});

test("long-form mel is concatenated along time, not appended end to end", () => {
  // A mel is row-major [80, frames]. Appending two flat buffers puts the
  // second chunk's bin 0 after the first chunk's bin 79, so every row but the
  // first is wrong. The audio is unaffected, since each chunk is vocoded on
  // its own, but the mel is the diagnostic people reach for when two backends
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
    chunking: {
      enabled: false,
      max_tokens: 99,
      prefix_tokens: 3,
      split_on: ["|"],
      abbreviations: ["Zz"],
      mid_sentence_period: "break",
    },
  });
  assert.equal(cfg.chunking.enabled, false);
  assert.equal(cfg.chunking.maxTokens, 99);
  assert.equal(cfg.chunking.prefixTokens, 3);
  assert.deepEqual(cfg.chunking.splitOn, ["|"]);
  assert.deepEqual(cfg.chunking.abbreviations, ["Zz"]);
  assert.equal(cfg.chunking.midSentencePeriod, "break");

  // An empty list is meaningful, being the old law spelled as data, so it
  // must not fall back to the shipping set the way an empty split_on does.
  const bare = algorithmFromManifest({
    format: "loudkit-checkpoint",
    format_version: 1,
    chunking: { abbreviations: [] },
  });
  assert.deepEqual(bare.chunking.abbreviations, []);

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
  // port refuses and another accepts is two renders under one fingerprint,
  // and the failure modes are silent: temperature 0 divides by zero, min_p 1
  // empties the candidate set, a negative EOS floor lets a chunk stop on the
  // first token.
  assert.throws(
    () => algorithmFromManifest({ sampling_defaults: { temperature: 0 } }),
    /temperature out of range/
  );
  assert.throws(
    () => algorithmFromManifest({ sampling_defaults: { repetition_penalty: 0.9 } }),
    /repetition_penalty below 1.0 rewards repetition/
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
  // explicit-grid support, while the manifest parser hard-coded `null` before
  // `timeGrid` ever saw it. An explicit grid exists precisely because "cosine"
  // is a formula two codebases can write two ways; `AlgorithmConfig.euler_grid`
  // in `config.py` is where the reference says so.
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
  // nobody wrote a test for still cannot drift, which is not hypothetical:
  // `eulerGrid` was hard-coded to null here while `timeGrid` honoured it, and
  // `tokenRateHz` was hard-coded too until this test asked for it.
  const algorithm = vectors().algorithm;
  assert.ok(algorithm, "the fixture has no algorithm section; nothing was compared");

  const cfg = algorithmFromManifest({
    edge_fade_seconds: 0.02,
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
    // The render-id censuses are properties of the weights and ride the
    // manifest top level, beside silence_token_ids, spelled out here like
    // that list, so the fingerprint pins them too.
    silence_render_ids: [4137, 4215, 4218, 4299, 6162, 6324, 6405, 6486],
    quiet_render_ids: [
      1458, 1461, 1488, 1701, 1704, 1707, 1716, 1731, 1785, 1788, 1869, 1947, 1950, 1951,
      1959, 1978, 2028, 2031, 2040, 2058, 2076, 2112, 2139, 3645, 3648, 3651, 3704, 3888,
      3894, 4188, 5838, 6081, 6183, 6537,
    ],
  });

  // The blob first: a mismatch there names the field that drifted, while a
  // mismatch in the hash alone says only that something did.
  assert.equal(canonicalForm(cfg), algorithm.canonical_form);
  assert.equal(fingerprint(cfg), algorithm.fingerprint);
});

test("the version and the decode mode are both gates", () => {
  assert.deepEqual(SUPPORTED_FORMAT_VERSIONS, [1, 2]);
  assert.deepEqual(SUPPORTED_DECODE_MODES, ["single", "fusion_mtp2"]);
});

/**
 * A safetensors file carrying nothing but a manifest.
 *
 * Small enough to write in a test and complete enough for `Checkpoint.open` to
 * reach every check it makes, which is the point: asserting on the two
 * constants alone left both guards deletable with the tests still green.
 */
function manifestOnly(manifest: string): string {
  const header = Buffer.from(
    JSON.stringify({ __metadata__: { manifest } }),
    "utf8",
  );
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  const path = join(mkdtempSync(join(tmpdir(), "loudkit-fmt-")), "ckpt.safetensors");
  writeFileSync(path, Buffer.concat([length, header]));
  return path;
}

test("the loader refuses what the constants declare", () => {
  const cases: Array<[string, string, string | null]> = [
    ["version 1 loads", `{"format":"loudkit-checkpoint","format_version":1}`, null],
    [
      "an explicit single is the same loop",
      `{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"single"}}`,
      null,
    ],
    [
      "version 2 loads",
      `{"format":"loudkit-checkpoint","format_version":2,"decode":{"mode":"fusion_mtp2"}}`,
      null,
    ],
    [
      "fusion requires version 2",
      `{"format":"loudkit-checkpoint","format_version":1,"decode":{"mode":"fusion_mtp2"}}`,
      "requires format_version 2",
    ],
    ["unknown version", `{"format":"loudkit-checkpoint","format_version":3}`, "format_version 3"],
    ["unknown loop", `{"format":"loudkit-checkpoint","format_version":2,"decode":{"mode":"future"}}`, "decode.mode"],
  ];
  for (const [name, manifest, wanted] of cases) {
    const path = manifestOnly(manifest);
    if (wanted === null) {
      assert.doesNotThrow(() => Checkpoint.open(path), name);
    } else {
      assert.throws(() => Checkpoint.open(path), new RegExp(wanted), name);
    }
  }
});


/** A safetensors file whose header is this JSON document, whatever its shape. */
function headerOnly(header: string): string {
  const body = Buffer.from(header, "utf8");
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(body.length));
  const path = join(mkdtempSync(join(tmpdir(), "loudkit-hdr-")), "ckpt.safetensors");
  writeFileSync(path, Buffer.concat([length, body]));
  return path;
}

test("an embedded manifest is an object, and its version is a number", () => {
  // A manifest holding a JSON list, number or string reported "no embedded
  // manifest", which is not what is wrong with the file. `checkpoint.py` and
  // `rust/src/checkpoint.rs` both refuse these; `Number` read `null` as 0 and
  // `true`, `"1"` and `[1]` all as 1, so four of them opened as version 1.
  for (const [body, wanted] of [
    ["[1,2,3]", /manifest is an array, expected a JSON object/],
    ["5", /manifest is a number, expected a JSON object/],
    ['"x"', /manifest is a string, expected a JSON object/],
    ["null", /manifest is null, expected a JSON object/],
  ] as const) {
    assert.throws(() => Checkpoint.open(manifestOnly(body)), wanted, body);
  }
  for (const declared of ['"1"', "true", "[1]", "null", '{"a":1}']) {
    assert.throws(
      () =>
        Checkpoint.open(
          manifestOnly(`{"format":"loudkit-checkpoint","format_version":${declared}}`)
        ),
      /format_version.*expected a version number/,
      declared
    );
  }
  // A whole number still opens. A fractional one is truncated here and refused
  // by the reference, which reads every count as a JSON number that has to be
  // whole; what this port should do about that is decided with the other
  // thirteen counts below, not here.
  assert.doesNotThrow(() =>
    Checkpoint.open(manifestOnly(`{"format":"loudkit-checkpoint","format_version":1.7}`))
  );
});

test("a safetensors header that is not an object is refused by name", () => {
  // `Object.entries(5)` is the empty list, so a header holding a bare number
  // opened as a valid file with zero tensors; `null` reached a bare TypeError
  // naming neither the file nor the problem. The reference, Go, Rust and Swift
  // each refuse all of these at open.
  for (const header of ["5", "true", "null", '"abc"', "[1,2,3]", "0"]) {
    assert.throws(
      () => Checkpoint.open(headerOnly(header)),
      /the safetensors header is not an object/,
      header
    );
  }
});

test("__metadata__ keeps only its string values", () => {
  // The map is string-to-string, which is what the reference's library hands
  // back and what Go, Rust and Swift each keep. A `manifest` holding an object
  // rather than the JSON text of one used to reach `JSON.parse` as an object
  // and throw a SyntaxError about a token.
  const path = headerOnly(JSON.stringify({ __metadata__: { manifest: { format: "x" } } }));
  assert.throws(() => Checkpoint.open(path), /no embedded manifest|not a loudkit checkpoint/);
  // A `__metadata__` that is not an object at all reads as no metadata, not as
  // a crash.
  const listed = headerOnly(JSON.stringify({ __metadata__: [1, 2, 3] }));
  assert.throws(() => Checkpoint.open(listed), /no embedded manifest|not a loudkit checkpoint/);
});

test("paired carry preserves complete pair boundaries", () => {
  assert.deepEqual(carryFrom([0, 1, 2, 3, 4, 5, 6], 3, 40, "fusion_mtp2"), [2, 3, 4, 5]);
  assert.deepEqual(carryFrom([0, 1, 2], 6, 40, "fusion_mtp2"), [0, 1]);
  assert.deepEqual(carryFrom([0, 1, 2], 0, 40, "fusion_mtp2"), []);
});

test("paired algorithm fingerprint matches Python", () => {
  const vector = JSON.parse(readFileSync(join(dirname(FIXTURE), "vectors_fusion_mtp2.json"), "utf8"));
  const algorithm = vector.algorithm;
  const canonical = JSON.parse(algorithm.canonical_form, (_key, value) =>
    typeof value === "string" && /^-?\d+(?:\.\d+)?$/.test(value) ? Number(value) : value
  ).algorithm;
  const manifest = {
    ...canonical,
    decode: { mode: "fusion_mtp2" },
    n_cfm_timesteps: canonical.euler_steps,
    sampling_defaults: canonical.sampling,
    speech_tokens: { start: canonical.start_speech_token, stop: canonical.stop_speech_token },
    silence_token_ids: canonical.sampling.silence_token_ids,
    eos_floor: canonical.sampling,
    silence_render_ids: canonical.postprocess.silence_render_ids,
    quiet_render_ids: canonical.postprocess.quiet_render_ids,
  };
  delete manifest.postprocess.silence_render_ids;
  delete manifest.postprocess.quiet_render_ids;
  assert.equal(fingerprint(algorithmFromManifest(manifest)), algorithm.fingerprint);
});

test("the 20 ms release ramp is fingerprinted and the legacy 5 ms ramp is omitted", () => {
  const base = algorithmFromManifest({ edge_fade_seconds: 0.02 });
  const explicit = algorithmFromManifest({ edge_fade_seconds: 0.02 });
  assert.equal(canonicalForm(explicit), canonicalForm(base));
  assert.ok(canonicalForm(base).includes('"edge_fade_seconds":"0.02"'));
  assert.ok(!canonicalForm(algorithmFromManifest({ edge_fade_seconds: 0.005 })).includes("edge_fade_seconds"));
  const other = algorithmFromManifest({ edge_fade_seconds: 0.008 });
  const want = canonicalForm(base).replace('"edge_fade_seconds":"0.02"', '"edge_fade_seconds":"0.008"');
  assert.equal(canonicalForm(other), want);
  assert.notEqual(fingerprint(other), fingerprint(base));
});

test("a token-id census is hashed in the order the manifest declared it", () => {
  // Every value below is Python's, from
  // `loudkit.manifest.algorithm_from(m).fingerprint()`. This port sorted the
  // three censuses before hashing, on the argument that the packer's order was
  // arbitrary; Python, Rust and Swift do not sort, so an unsorted census hashed
  // here to the number the reference answers for the sorted one, and the one
  // mechanism that exists to catch two engines computing different things
  // agreed with the misreading instead of reporting it.
  assert.equal(fingerprint(algorithmFromManifest({ silence_token_ids: [9, 1, 5] })), "0253d842841afe71");
  assert.equal(fingerprint(algorithmFromManifest({ silence_token_ids: [1, 5, 9] })), "29ce8cbf4b263a2a");
  assert.equal(fingerprint(algorithmFromManifest({ silence_render_ids: [9, 1, 5] })), "b1c9f0fbef48d5b8");
  assert.equal(fingerprint(algorithmFromManifest({ silence_render_ids: [1, 5, 9] })), "6a52663d1d5565d5");
  assert.equal(fingerprint(algorithmFromManifest({ quiet_render_ids: [9, 1, 5] })), "98d72152aadd6b64");
  assert.equal(fingerprint(algorithmFromManifest({ quiet_render_ids: [1, 5, 9] })), "002beda344c02360");
});

test("the shipped 0.1.1 manifest still fingerprints 7cd75498ad4e7531", () => {
  // The manifest `tools/amend_manifest.py` writes, spelled out here rather than
  // assembled from this port's constants: a manifest built from them would
  // agree with a reader that ignored every key and answered its defaults, which
  // is the failure this pins. `rust/tests/freeze.rs` and
  // `go/config/freeze_test.go` are the same test over the same bytes.
  //
  // All three censuses below are ascending, which is why removing the sort
  // above leaves this number where it was.
  const algo = algorithmFromManifest({
    format: "loudkit-checkpoint",
    format_version: 1,
    edge_fade_seconds: 0.02,
    guidance: "single_path",
    guidance_rate: 0.0,
    recipe_version: "loudkit-1",
    postprocess: {
      mode: "trim",
      ceiling_speech_per_text_token: 4.0,
      ceiling_slack_tokens: 40,
      trailing_filler_threshold: 0.7,
      trailing_silence_run_tokens: 12,
      filler_min_eos_probability: 0.05,
      filler_max_speech_after_run: 10,
      desperation_speech_per_text_token: 4.5,
      desperation_min_text_tokens: 10,
      ended_tail_silence_run: 6,
      ended_tail_blip_max: 2,
      ended_tail_word_max: 10,
      ended_tail_keep: 5,
      echo_strong_eos_probability: 0.1,
      echo_strong_max_tail: 30,
      echo_strong_min_position_pct: 68,
      echo_weak_eos_probability: 0.003,
      echo_weak_max_tail: 16,
      echo_weak_min_position_pct: 85,
    },
    window: {
      max_speech_tokens: 255,
      static_length: 255,
      pad_token_id: 4254,
      static_prompt_tokens: 238,
    },
    eos_floor: { min_tokens_floor: 10, min_tokens_text_ratio: 1.2 },
    chunking: {
      enabled: true,
      max_tokens: 255,
      prefix_tokens: 6,
      split_on: [". ", "! ", "? ", "; ", ", "],
      // prettier-ignore
      abbreviations: [
        "A", "B", "Cpn", "D", "Dr", "F", "H", "Hr", "I", "J", "K", "M",
        "Mr", "Mrs", "R", "S", "St", "T", "V", "Vors", "dr", "mrs",
        "prof", "św",
      ],
      mid_sentence_period: "hold",
    },
    sample_rate: 24_000,
    token_rate_hz: 25.0,
    speech_vocab_size: 8194,
    n_cfm_timesteps: 2,
    speech_tokens: { start: 6561, stop: 6562 },
    sampling_defaults: {
      temperature: 0.8,
      repetition_penalty: 1.2,
      min_p: 0.05,
      max_new_tokens: 255,
    },
    // prettier-ignore
    silence_token_ids: [
      1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927,
      3928, 3930, 4008, 4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173,
      4174, 4218, 4245, 4251, 4252, 4254, 4255, 4260, 4282,
    ],
    silence_render_ids: [4137, 4215, 4218, 4299, 6162, 6324, 6405, 6486],
    // prettier-ignore
    quiet_render_ids: [
      1458, 1461, 1488, 1701, 1704, 1707, 1716, 1731, 1785, 1788, 1869,
      1947, 1950, 1951, 1959, 1978, 2028, 2031, 2040, 2058, 2076, 2112,
      2139, 3645, 3648, 3651, 3704, 3888, 3894, 4188, 5838, 6081, 6183,
      6537,
    ],
  });
  assert.equal(fingerprint(algo), "7cd75498ad4e7531");
});

test("a manifest count is refused when it is not a whole number", () => {
  // A count with a fraction was truncated toward zero, which made the same
  // manifest fingerprint differently here *and* run a different algorithm:
  // `n_cfm_timesteps: 2.7` made `timeGrid` answer [0, 0.1645, 0.6039] where two
  // steps answer [0, 0.2929, 1.0], and the flow ODE stopped at t~0.60 with the
  // audio rendered from a half-integrated state.
  //
  // Truncating is not the fix, because 2.7 in a field that counts tokens was
  // written by a tool that made a mistake, and reading it as 2 turns that
  // mistake into a chunker that breathes in a different place under a
  // `recipe_version` saying the five implementations agree. Refused by name,
  // in the sentence `manifest._int` uses, which is what this reader already
  // said about the detector constants.
  for (const [written, sentence] of [
    [{ n_cfm_timesteps: 2.7 }, "manifest['n_cfm_timesteps'] must be a whole number, got 2.7"],
    [{ sample_rate: 24_000.7 }, "manifest['sample_rate'] must be a whole number, got 24000.7"],
    [{ speech_vocab_size: 8194.9 }, "manifest['speech_vocab_size'] must be a whole number, got 8194.9"],
    [{ speech_tokens: { start: 6561.5 } }, "manifest['speech_tokens']['start'] must be a whole number, got 6561.5"],
    [{ speech_tokens: { stop: 6562.5 } }, "manifest['speech_tokens']['stop'] must be a whole number, got 6562.5"],
    [{ sampling_defaults: { max_new_tokens: 255.9 } }, "manifest['sampling_defaults']['max_new_tokens'] must be a whole number, got 255.9"],
    [{ eos_floor: { min_tokens_floor: 10.9 } }, "manifest['eos_floor']['min_tokens_floor'] must be a whole number, got 10.9"],
    [{ window: { max_speech_tokens: 255.9 } }, "manifest['window']['max_speech_tokens'] must be a whole number, got 255.9"],
    [{ window: { static_length: 493.9 } }, "manifest['window']['static_length'] must be a whole number, got 493.9"],
    [{ window: { pad_token_id: 4254.9 } }, "manifest['window']['pad_token_id'] must be a whole number, got 4254.9"],
    [{ window: { static_prompt_tokens: 2.7 } }, "manifest['window']['static_prompt_tokens'] must be a whole number, got 2.7"],
    [{ chunking: { max_tokens: 255.9 } }, "manifest['chunking']['max_tokens'] must be a whole number, got 255.9"],
    [{ chunking: { prefix_tokens: 6.9 } }, "manifest['chunking']['prefix_tokens'] must be a whole number, got 6.9"],
    [{ silence_token_ids: [1, 2.7] }, "manifest['silence_token_ids'][1] must be a whole number, got 2.7"],
    [{ silence_render_ids: [1.7] }, "manifest['silence_render_ids'][0] must be a whole number, got 1.7"],
    [{ quiet_render_ids: [1.7] }, "manifest['quiet_render_ids'][0] must be a whole number, got 1.7"],
    [{ postprocess: { ended_tail_keep: 2.7 } }, "manifest['postprocess']['ended_tail_keep'] must be a whole number, got 2.7"],
    // The fraction is named before the sign, as the reference names only the
    // fraction: -2.7 is refused for what it is, not for being negative.
    [{ n_cfm_timesteps: -2.7 }, "manifest['n_cfm_timesteps'] must be a whole number, got -2.7"],
  ] as Array<[Record<string, unknown>, string]>) {
    assert.throws(
      () => algorithmFromManifest(written),
      (e: Error) => e.message === sentence,
      JSON.stringify(written)
    );
  }
  // And a count written as a float is the count. 255.0 is 255 in JSON and in
  // every reader here, so only the fraction is refused, and the fields that
  // hold a rate or a threshold keep theirs.
  assert.equal(
    fingerprint(algorithmFromManifest({ postprocess: { ended_tail_keep: 5.0 } })),
    fingerprint(algorithmFromManifest({}))
  );
  assert.equal(algorithmFromManifest({ n_cfm_timesteps: 2.0 }).eulerSteps, 2);
  assert.deepEqual(
    timeGrid(algorithmFromManifest({ n_cfm_timesteps: 2.0 })),
    timeGrid(algorithmFromManifest({ n_cfm_timesteps: 2 }))
  );
  const rates = algorithmFromManifest({
    token_rate_hz: 25.5,
    edge_fade_seconds: 0.03,
    sampling_defaults: { temperature: 0.75, repetition_penalty: 1.25, min_p: 0.055 },
    eos_floor: { min_tokens_text_ratio: 1.25 },
    postprocess: { pacing_tolerance: 0.17 },
  });
  assert.equal(rates.tokenRateHz, 25.5);
  assert.equal(rates.edgeFadeSeconds, 0.03);
  assert.equal(rates.sampling.temperature, 0.75);
  assert.equal(rates.sampling.minP, 0.055);
  assert.equal(rates.sampling.minTokensTextRatio, 1.25);
  assert.equal(rates.postprocess.pacingTolerance, 0.17);
});

test("a block the manifest names is read as a block or refused by name", () => {
  // `(manifest.X ?? {})` was six casts, and a cast is a claim rather than a
  // check: it read `null` as absent and a list as an object. Every refusal
  // below is one the reference makes and this port did not, and the shape of
  // the failure was always the same: the manifest declared a law, this port
  // took a default instead, and then recorded that default in the canonical
  // form, so the fingerprint agreed with the misreading.
  for (const key of ["chunking", "sampling_defaults", "eos_floor", "speech_tokens", "postprocess"]) {
    for (const bad of [null, [1, 2, 3], [], "x", 5, true]) {
      assert.throws(
        () => algorithmFromManifest({ [key]: bad }),
        new RegExp(`'${key}' must be an object`),
        `${key}: ${JSON.stringify(bad)}`
      );
    }
  }
  // `window: null` is the ragged window said out loud and `decode: null` the
  // single loop said out loud, so those two read null as a value.
  assert.equal(
    fingerprint(algorithmFromManifest({ window: null })),
    fingerprint(algorithmFromManifest({}))
  );
  assert.equal(algorithmFromManifest({ decode: null }).decode, "single");
  for (const bad of [[1, 2, 3], "x", 5, true]) {
    assert.throws(() => algorithmFromManifest({ window: bad }), /\['window'\] must be an object or null/);
    assert.throws(() => algorithmFromManifest({ decode: bad }), /\['decode'\] must be an object or null/);
  }
  // A census is a list or nothing; `null` is a declared value this port cannot
  // read, not an absence.
  for (const key of ["silence_token_ids", "silence_render_ids", "quiet_render_ids"]) {
    assert.throws(() => algorithmFromManifest({ [key]: null }), new RegExp(`'${key}' must be a list`));
    assert.throws(() => algorithmFromManifest({ [key]: "123" }), new RegExp(`'${key}' must be a list`));
  }
  // And a closed set is refused rather than defaulted, for the same reason.
  assert.throws(() => algorithmFromManifest({ guidance: null }), /unknown guidance mode/);
});

test("the chunking recipe is validated instead of falling back", () => {
  // The fallback ran before validation, so three refusals the reference makes
  // could not be reached from a manifest at all. Go and Rust describe fixing
  // this exact ordering.
  assert.throws(
    () => algorithmFromManifest({ chunking: { split_on: [] } }),
    /split_on cannot be empty/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { split_on: ". " } }),
    /split_on.*list of strings, got a string/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { abbreviations: "Dr" } }),
    /abbreviations.*list of strings, got a string/
  );
  // A present separator set is read, not merged with the shipping five.
  assert.deepEqual(algorithmFromManifest({ chunking: { split_on: ["|"] } }).chunking.splitOn, ["|"]);
  // An empty abbreviation list is meaningful, being the old law spelled as
  // data, so it must not fall back where an empty `split_on` is refused.
  assert.deepEqual(
    algorithmFromManifest({ chunking: { abbreviations: [] } }).chunking.abbreviations,
    []
  );
  // A closed set given the wrong type is refused rather than silently taking
  // the shipping recipe and breaking the text somewhere the manifest never
  // named. Refused by type, as `go/config.stringKey` and Rust's text reader
  // do, because `String(["word"])` is "word" in this language: rendering first
  // let a one-element list pass the membership check.
  assert.throws(
    () => algorithmFromManifest({ chunking: { cap_resplit: 5 } }),
    /cap_resplit.*must be a string/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { cap_resplit: ["word"] } }),
    /cap_resplit.*must be a string/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { mid_sentence_period: 5 } }),
    /mid_sentence_period.*must be a string/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { mid_sentence_period: ["hold"] } }),
    /mid_sentence_period.*must be a string/
  );
  // A string that is not in the set still gets the membership refusal.
  assert.throws(
    () => algorithmFromManifest({ chunking: { cap_resplit: "WORD" } }),
    /unknown cap_resplit/
  );
  assert.throws(
    () => algorithmFromManifest({ chunking: { mid_sentence_period: "hold-ish" } }),
    /unknown mid_sentence_period/
  );
});

test("chunking.enabled takes JSON true or false and nothing else", () => {
  // JavaScript calls 0 false and 1 true and JSON does not, so `"enabled": 0`
  // was written by a tool that meant `false` and emitted a number. Reading it
  // either way hides that mistake behind audio joined differently rather than
  // audio missing, and this is the one key whose misreading changes every join
  // in a passage at once. All five ports refuse it.
  for (const wrong of [0, 1, -1, 2.5, "", "no", [], [0], {}, { a: 1 }, null]) {
    assert.throws(
      () => algorithmFromManifest({ chunking: { enabled: wrong } }),
      /enabled.*must be JSON true or false/,
      JSON.stringify(wrong) ?? "null"
    );
  }
  for (const right of [true, false]) {
    assert.equal(
      algorithmFromManifest({ chunking: { enabled: right } }).chunking.enabled,
      right
    );
  }
  // An absent key is the shipping recipe's own value, and hashes as one.
  assert.equal(
    fingerprint(algorithmFromManifest({ chunking: { enabled: true } })),
    fingerprint(algorithmFromManifest({}))
  );
});

test("an explicit null edge fade is the legacy 5 ms, as in the other four ports", () => {
  // This port was the only one of the five that refused the explicit spelling.
  // A manifest that never names the key predates the field; one that names it
  // null says the same thing out loud, and both mean 5 ms.
  const fade = algorithmFromManifest({ edge_fade_seconds: null });
  assert.equal(fade.edgeFadeSeconds, 0.005);
  assert.equal(fingerprint(fade), fingerprint(algorithmFromManifest({})));
  assert.equal(fingerprint(fade), fingerprint(algorithmFromManifest({ edge_fade_seconds: 0.005 })));
});

test("a manifest that omits the optional keys fingerprints as the reference does", () => {
  // The value is `AlgorithmConfig.from_manifest({}).fingerprint()` in Python.
  // Filling the shipped window and EOS floor in for absent keys made the same
  // bytes describe a different algorithm in this port alone, which is the
  // divergence class the fingerprint exists to catch. Release manifests carry
  // the keys, so the shipped fingerprints never showed it.
  const algo = algorithmFromManifest({});
  assert.equal(fingerprint(algo), "2f7468a9a48fa2ab");
  assert.equal(algo.sampling.minTokensFloor, 0);
  assert.equal(algo.sampling.minTokensTextRatio, 0.0);
  assert.equal(algo.window.staticLength, null);
  assert.equal(algo.window.padTokenId, null);
  assert.equal(algo.window.staticPromptTokens, null);
  // `max_speech_tokens` is the one window key with a value when absent, as
  // `WindowConfig` declares it.
  assert.equal(algo.window.maxSpeechTokens, 255);
  // Named keys are still read, so the shipped manifest describes the shipped
  // window.
  const named = algorithmFromManifest({
    window: { max_speech_tokens: 255, static_length: 255, pad_token_id: 4254, static_prompt_tokens: 238 },
    eos_floor: { min_tokens_floor: 10, min_tokens_text_ratio: 1.2 },
  });
  assert.deepEqual(named.window, productionWindow());
  assert.equal(named.sampling.minTokensFloor, PRODUCTION_EOS_FLOOR);
  assert.equal(named.sampling.minTokensTextRatio, PRODUCTION_EOS_TEXT_RATIO);
});

test("a window the exported graphs cannot run is refused by name", () => {
  // Python refuses it in `onnx_backend._require_static_window`, before any
  // session is built. Without the same door here a ragged window reaches
  // onnxruntime as a tensor shape mismatch, which names a tensor rather than
  // the manifest key that caused it.
  assert.throws(
    () => requireStaticWindow(algorithmFromManifest({})),
    /static at query 255 \/ prompt 238/
  );
  requireStaticWindow(
    algorithmFromManifest({
      window: { max_speech_tokens: 255, static_length: 255, pad_token_id: 4254, static_prompt_tokens: 238 },
    })
  );
});
