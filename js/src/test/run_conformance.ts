/**
 * End-to-end conformance: the ONNX engine vs the shared fixture.
 *
 * Reads `vectors.json` + the voice and reference bins from `tests/data/
 * conformance` and asserts:
 *
 *  - free-running tokens are **exactly** the fixture's tokens for the same
 *    text, voice and seed (the sampler is counter-based and the graphs are
 *    fp32, so a token divergence is a port bug);
 *  - fixed-token renders land inside the fixture's mel and waveform
 *    correlation bands (the renderer is transparent at fp32);
 *  - a passage too long for one window produces the fixture's exact token
 *    stream in **every** chunk, prefix carried across the joins: the case the
 *    single-sentence ones above cannot reach, because with an empty prefix
 *    `prefix.length + step + 1` and `step + 1` are the same expression.
 *
 * This is the JS half of the same contract `pytest` and `swift test` verify.
 *
 * Usage (from the js dir):
 *   node dist/test/run_conformance.js --ckpt PATH --onnx DIR --voice PATH
 *     [--fixture tests/data/conformance]
 */

import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { splitText } from "../chunking.js";
import { Engine, carryFrom } from "../engine.js";
import type { ONNXProvider } from "../execution.js";
import { speechText } from "../speechText.js";
import type { VoiceProfile } from "../types.js";
import { loadVoice } from "../voice.js";
import { deriveSeed } from "../rng.js";
import { LRSamplerV1 } from "../sampler.js";

interface Case {
  name: string;
  text: string;
  language: string;
  seed: number;
  tokens: number[];
  mel: { file: string; shape: number[] };
  wav: { file: string; samples: number };
  gates: { mel_corr: number; wave_corr: number; wave_rms_db: number };
}

interface LongFormChunk {
  index: number;
  text: string;
  /** Hex, because a derived 64-bit seed does not survive a JSON double. */
  seed: string;
  prefix: number[];
  tokens: number[];
}

interface LongFormCase {
  public_tokens?: number[];
  name: string;
  text: string;
  language: string;
  seed: number;
  /** The passage after the speech funnel: what the splitter is given. */
  prepared: string;
  chunks: LongFormChunk[];
  tokens: number[];
}

interface LongFormSection {
  voice: string;
  prefix_tokens: number;
  chunk_stream_base: number;
  cases: LongFormCase[];
}

function parseArgs(): Record<string, string> {
  const out: Record<string, string> = {};
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i += 2) {
    out[argv[i].replace(/^--/, "")] = argv[i + 1];
  }
  return out;
}

/**
 * Pearson correlation, on the explicit condition that the two are the same
 * length.
 *
 * Correlating `Math.min(...)` samples scores a
 * truncated render perfectly against the prefix it did produce, and the length
 * *is* the finding in that case. This runner is a standalone script
 * (`npm run test:fixture`), so it is both the
 * least-run and the most likely to be quoted as evidence.
 */
function corr(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(
      `length mismatch: ${a.length} vs ${b.length}. Correlating a prefix would ` +
        "score a truncated render as a perfect one"
    );
  }
  const n = a.length;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i] - ma, y = b[i] - mb;
    num += x * y; da += x * x; db += y * y;
  }
  return num / Math.sqrt(da * db);
}

/**
 * The RMS ratio of a render to its reference, in dB.
 *
 * Correlation subtracts the mean and divides by the deviation, so it reports
 * 1.0 for a render at half volume, at twenty times volume, or with a DC
 * offset. Level is exactly what that normalisation discards, so it is the one
 * amplitude fact worth its own gate.
 */
function levelDB(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`length mismatch: ${a.length} vs ${b.length}`);
  }
  let sa = 0, sb = 0;
  for (let i = 0; i < a.length; i++) { sa += a[i] * a[i]; sb += b[i] * b[i]; }
  if (sa === 0) throw new Error("rendered silence");
  return 20 * Math.log10(Math.sqrt(sa / sb));
}

/**
 * The loudest sample, against the `[-1, 1]` a waveform is declared to occupy.
 * Everything downstream clips to that range, so a render outside it is audibly
 * wrong and needs no tolerance to say so.
 */
function peakOf(a: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < a.length; i++) peak = Math.max(peak, Math.abs(a[i]));
  return peak;
}

/**
 * Resolve the conformance fixture directory. Defaults to the loudkit repo
 * root's `tests/data/conformance` (located relative to this file, not the
 * process CWD) so the documented invocation works from anywhere. An explicit
 * `--fixture` wins.
 */
function resolveFixtureDir(explicit: string | undefined): string {
  if (explicit) return explicit;
  let dir = dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = join(dir, "tests", "data", "conformance");
    if (existsSync(join(candidate, "vectors.json"))) return candidate;
    const parent = dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error(
    "cannot locate the conformance fixture: pass --fixture or run from the loudkit repo"
  );
}

async function main(): Promise<void> {
  const args = parseArgs();
  // Environment variables as well as flags. `npm run test:all` passes no
  // arguments, so the README's and tutorial 07's documented invocation,
  // `LOUDKIT_CKPT=... LOUDKIT_ONNX_DIR=... LOUDKIT_VOICE=... npm run test:all`,
  // always exited 2: this script read only CLI flags and none of the
  // variables the surrounding prose prefixed it with. The same names the Go
  // and Rust conformance runners already use, so one export block drives all
  // three.
  const ckpt = args.ckpt ?? process.env.LOUDKIT_CKPT;
  const onnx = args.onnx ?? process.env.LOUDKIT_ONNX_DIR;
  const voicePath = args.voice ?? process.env.LOUDKIT_VOICE;
  const fixtureDir = resolveFixtureDir(args.fixture ?? process.env.LOUDKIT_FIXTURE_DIR);
  // CPU unless asked otherwise, and never `"auto"`. The token half of this
  // fixture is an exact-match gate shared with Python, Rust, Go and Swift, so
  // it has to name the provider it ran on rather than take the best one this
  // machine offers: on an arm64 Mac `"auto"` is CoreML, and the same script
  // would then be an exact-match gate against a different arithmetic without
  // saying so. `--provider` is how the divergence gets measured on purpose.
  const provider = (args.provider ?? process.env.LOUDKIT_ONNX_PROVIDER ?? "cpu") as ONNXProvider;
  if (!ckpt || !onnx || !voicePath) {
    console.error(
      "usage: node dist/test/run_conformance.js --ckpt PATH --onnx DIR --voice PATH " +
        "[--fixture DIR] [--provider auto|cpu|cuda|coreml|directml]\n" +
        "   or: LOUDKIT_CKPT=... LOUDKIT_ONNX_DIR=... LOUDKIT_VOICE=... npm run test:fixture\n" +
        "\nNeeds the checkpoint, the exported graphs and a voice profile; skipping is not " +
        "a pass, so this exits 2 rather than reporting success."
    );
    process.exit(2);
  }
  const engine = await Engine.loadPaths(ckpt, onnx, `${fixtureDir}/tokenizer.json`, {
    onnxProvider: provider,
  });
  const fixtureName = engine.config.decode === "fusion_mtp2" ? "vectors_fusion_mtp2.json" : "vectors.json";
  const vectors = JSON.parse(readFileSync(`${fixtureDir}/${fixtureName}`, "utf8"));
  const cases: Case[] = vectors.end_to_end ?? [];
  if (!cases.length) throw new Error("fixture has no end_to_end section");
  const voice = loadVoice(voicePath);

  // The provider is on the header line because every number below it is a
  // measurement, and a measurement that does not say what ran is not one.
  console.log(`provider: ${engine.onnxProvider}`);
  console.log(`engine: ${engine.config.recipeVersion} window ` +
    `${engine.config.window.staticPromptTokens}+${engine.config.window.staticLength} ` +
    `euler=${engine.config.eulerSteps} temp=${engine.config.sampling.temperature}`);

  let allPass = true;

  // ---- tokens: exact -------------------------------------------------
  for (const c of cases) {
    const sampler = new LRSamplerV1(engine.config.sampling, c.seed);
    const textIds = engine.encode(c.text, c.language);
    const raw = await engine.generate(textIds, voice, sampler);
    const stripped = raw.filter((t) => t < engine.config.startSpeechToken);
    // The fixture's tokens, whole. Slicing the *reference* to the window meant
    // an engine that stopped at 255 tokens for a 300-token reference printed
    // CONFORMANCE PASS, while this file's own header claims the tokens are "exactly
    // the fixture's".
    const want = c.tokens;
    const match = stripped.length === want.length && stripped.every((t, i) => t === want[i]);
    console.log(`tokens ${c.name}: ${match ? "PASS" : "FAIL"} ` +
      `(${stripped.length} vs ${want.length})`);
    if (!match) {
      allPass = false;
      if (stripped.length === want.length) {
        const diffs: number[] = [];
        for (let i = 0; i < stripped.length; i++) if (stripped[i] !== want[i]) diffs.push(i);
        console.log(`  diverged at ${diffs.slice(0, 10).join(",")}`);
      }
    }
    // The two public paths agree with the generator: chunk 0 draws the
    // caller's seed, so a text that fits one window renders the same tokens
    // through synthesizeWindow and through synthesize.
    const options = { seed: c.seed, language: c.language };
    const window = await engine.synthesizeWindow(c.text, voice, options);
    const whole = await engine.synthesize(c.text, voice, options);
    const same = (a: number[]) => a.length === want.length && a.every((t, i) => t === want[i]);
    const publicOk = same(window.tokens) && same(whole.tokens);
    console.log(`public ${c.name}: ${publicOk ? "PASS" : "FAIL"} ` +
      `(window ${window.tokens.length}, synthesize ${whole.tokens.length}, fixture ${want.length})`);
    if (!publicOk) allPass = false;
  }

  // ---- render: within the band ----------------------------------------
  for (const c of cases) {
    const seed = c.seed;
    const mel = await engine.decodeMel(c.tokens, voice, deriveSeed(seed, 1));
    const wav = await engine.vocode(mel, deriveSeed(seed, 2));

    const melRef = readFloat32File(`${fixtureDir}/${c.mel.file}`);
    const wavRef = readFloat32File(`${fixtureDir}/${c.wav.file}`);

    const melCorr = corr(mel, melRef);
    const waveCorr = corr(wav, wavRef);
    const level = levelDB(wav, wavRef);
    const peak = peakOf(wav);
    const melOk = melCorr >= c.gates.mel_corr;
    const waveOk = waveCorr >= c.gates.wave_corr;
    const levelOk = Math.abs(level) <= c.gates.wave_rms_db;
    const peakOk = peak <= 1;
    console.log(
      `render ${c.name}: mel ${melCorr.toFixed(6)} (gate ${c.gates.mel_corr}) ` +
      `${melOk ? "PASS" : "FAIL"} | wave ${waveCorr.toFixed(4)} (gate ${c.gates.wave_corr}) ` +
      `${waveOk ? "PASS" : "FAIL"} | level ${level >= 0 ? "+" : ""}${level.toFixed(4)} dB ` +
      `(gate ${c.gates.wave_rms_db}) ${levelOk ? "PASS" : "FAIL"} | ` +
      `peak ${peak.toFixed(4)} ${peakOk ? "PASS" : "FAIL"}`
    );
    if (!melOk || !waveOk || !levelOk || !peakOk) allPass = false;
  }

  // ---- long form: every chunk's tokens, exactly ------------------------
  if (!(await longForm(engine, voice, vectors.long_form))) allPass = false;

  console.log(allPass ? "CONFORMANCE PASS" : "CONFORMANCE FAIL");
  process.exit(allPass ? 0 : 1);
}

/**
 * A passage too long for one window, chunk by chunk.
 *
 * Everything above this is a single window with an empty prefix, and with an
 * empty prefix `prefix.length + step + 1` and `step + 1` are the same number
 * and a repetition mask seeded from the prefix is the empty one. Three ports
 * wrote the shorter form and this fixture passed throughout. A carried prefix
 * is what separates them.
 *
 * Every chunk is asserted on its own rather than on the concatenation: a
 * divergence inside chunk *k* shifts every token after it, so a whole-passage
 * comparison reports one enormous mismatch instead of naming the chunk and the
 * step.
 */
async function longForm(
  engine: Engine,
  voice: VoiceProfile,
  section: LongFormSection | undefined
): Promise<boolean> {
  if (!section) {
    console.error("fixture has no long_form section");
    return false;
  }
  if (engine.config.chunking.prefixTokens !== section.prefix_tokens) {
    console.error(
      `carry length: this port ${engine.config.chunking.prefixTokens}, ` +
        `fixture ${section.prefix_tokens}`
    );
    return false;
  }
  let ok = true;
  for (const c of section.cases) {
    // Funnel first, then split: the order the engine uses, and the order the
    // character budget assumes.
    const prepared = speechText(c.text, c.language);
    if (prepared !== c.prepared) {
      console.log(`long_form ${c.name}: FAIL (the speech funnel drifted)`);
      ok = false;
      continue;
    }
    const wantTexts = c.chunks.map((chunk) => chunk.text);
    const gotTexts = splitText(prepared, engine.config.chunking);
    if (gotTexts.length !== wantTexts.length || gotTexts.some((t, i) => t !== wantTexts[i])) {
      console.log(
        `long_form ${c.name}: FAIL (the split moved, so every chunk below ` +
          `would be asking about different text)`
      );
      ok = false;
      continue;
    }
    for (const chunk of c.chunks) {
      // The chain the streaming path walks: chunk k is conditioned on the tail
      // of chunk k-1. Spelled out in the fixture so a mismatch names the carry
      // rather than the tokens that followed from it.
      if (chunk.index > 0) {
        const previous = c.chunks[chunk.index - 1].tokens;
        const tail = carryFrom(previous, section.prefix_tokens, engine.config.startSpeechToken, engine.config.decode);
        if (chunk.prefix.some((t, i) => t !== tail[i])) {
          console.log(`long_form ${c.name} chunk ${chunk.index}: FAIL (carry)`);
          ok = false;
          continue;
        }
      }
      const sampler = new LRSamplerV1(engine.config.sampling, BigInt(chunk.seed));
      const ids = engine.encode(chunk.text, c.language);
      const raw = await engine.generate(ids, voice, sampler, undefined, undefined, chunk.prefix);
      const got = raw.filter((t) => t < engine.config.startSpeechToken);
      const match =
        got.length === chunk.tokens.length && got.every((t, i) => t === chunk.tokens[i]);
      console.log(
        `long_form ${c.name} chunk ${chunk.index}: ${match ? "PASS" : "FAIL"} ` +
          `(${got.length} vs ${chunk.tokens.length})`
      );
      if (!match) {
        ok = false;
        if (got.length === chunk.tokens.length) {
          const diffs: number[] = [];
          for (let i = 0; i < got.length; i++) if (got[i] !== chunk.tokens[i]) diffs.push(i);
          console.log(`  diverged at ${diffs.slice(0, 10).join(",")}`);
        }
      }
    }
    // The engine's own long-form path, not the generator driven with the
    // fixture's seeds: this holds the chunk seed law (chunk 0 the caller's
    // seed, chunk k derive(seed, 16 + k)) and the carry to the fixture.
    const whole = await engine.synthesize(c.text, voice, { seed: c.seed, language: c.language });
    const flat = c.public_tokens ?? c.tokens;
    const wholeOk = whole.tokens.length === flat.length && whole.tokens.every((t, i) => t === flat[i]);
    console.log(`long_form ${c.name} synthesize: ${wholeOk ? "PASS" : "FAIL"} ` +
      `(${whole.tokens.length} vs ${flat.length})`);
    if (!wholeOk) ok = false;
    const streamedTokens: number[] = [];
    const audio: Float32Array[] = [];
    for await (const chunk of engine.stream(c.text, voice, {seed:c.seed, language:c.language})) {
      streamedTokens.push(...chunk.tokens);
      audio.push(chunk.audio);
    }
    const repeated = Buffer.concat(audio.map(a => Buffer.from(a.buffer,a.byteOffset,a.byteLength)));
    const original = Buffer.from(whole.audio.buffer,whole.audio.byteOffset,whole.audio.byteLength);
    const repeatOk = audio.length === c.chunks.length && streamedTokens.length === whole.tokens.length && streamedTokens.every((t,i) => t === whole.tokens[i]) && repeated.equals(original);
    console.log(`long_form ${c.name} repeat stream: ${repeatOk ? "PASS" : "FAIL"} (${audio.length} chunks)`);
    if (!repeatOk) ok = false;

  }
  return ok;
}

function readFloat32File(path: string): Float32Array {
  const buf = readFileSync(path);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

main().catch((e) => {
  console.error("conformance error:", e);
  process.exit(1);
});
