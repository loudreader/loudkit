// Stage-by-stage probe of the JS decode loop, for cross-port comparison.
//
// The same stages in the same order as tools/decode_probe/probe.py, so a
// comparison stops at the first stage that disagrees instead of reporting
// noise from everything downstream of a divergence.
//
//   node tools/decode_probe/probe.mjs BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR
//
// Writes OUTDIR/js.json and OUTDIR/js.<stage>.bin (raw little-endian float32).
//
// `prefillEmbeds` and `condRow` are `private` in the TypeScript source, which
// is a compile-time claim only: this reaches them through the instance because
// the prefill row is the first stage a cross-port comparison has to check, and
// exporting it from the library to measure it would change the library.
import { Engine, loadVoice } from "../../js/dist/index.js";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const [bundle, voicePath, text, seedArg, language, outdir] = process.argv.slice(2);
if (!outdir) {
  console.error("usage: probe BUNDLE VOICE TEXT SEED LANGUAGE OUTDIR");
  process.exit(2);
}
const seed = BigInt(seedArg);

// Which module actually answered.
console.error(`js probe: ${fileURLToPath(import.meta.resolve("../../js/dist/index.js"))}`);

const f32bytes = (a) => Buffer.from(Float32Array.from(a).buffer);
const shaOf = (a) => createHash("sha256").update(f32bytes(a)).digest("hex");
const head = (a, n) => Array.from(a.slice(0, n));

mkdirSync(outdir, { recursive: true });
const engine = await Engine.load(bundle, { onnxProvider: "cpu" });
console.error(engine.describe ? engine.describe() : "(no describe)");
try {
  const voice = loadVoice(voicePath);
  const cfg = engine.config;
  const rec = {
    port: "js",
    loaded_from: fileURLToPath(import.meta.resolve("../../js/dist/index.js")),
    bundle,
    voice: voicePath,
    text,
    seed: Number(seed),
    language,
    decode: cfg.decode,
    sample_rate: cfg.sampleRate,
  };

  // 1. text tokens.
  const textTokens = engine.encode(text, language);
  rec.text_tokens = textTokens;

  // 2. the prefill row and the conditioning row.
  try {
    const cond = await engine.condRow(voice);
    writeFileSync(join(outdir, "js.cond.bin"), f32bytes(cond));
    rec.cond = { len: cond.length, sha: shaOf(cond), head: head(cond, 8) };
  } catch (e) {
    rec.cond = { error: String(e) };
  }
  try {
    const { embeds } = await engine.prefillEmbeds(textTokens, voice, []);
    writeFileSync(join(outdir, "js.prefill.bin"), f32bytes(embeds));
    rec.prefill = { len: embeds.length, sha: shaOf(embeds), head: head(embeds, 8) };
  } catch (e) {
    rec.prefill = { error: String(e) };
  }

  // 3. the token sequence.
  const { LRSamplerV1 } = await import("../../js/dist/sampler.js");
  const sampler = new LRSamplerV1(cfg.sampling, seed);
  const raw = await engine.generate(textTokens, voice, sampler, undefined, undefined, []);
  rec.speech_tokens_raw = raw;
  const limit = cfg.startSpeechToken;
  const stripped = raw.filter((t) => t < limit);
  rec.speech_tokens = stripped;

  // 4. the mel frames.
  const mel = await engine.decodeMel(stripped, voice, seed);
  writeFileSync(join(outdir, "js.mel.bin"), f32bytes(mel));
  rec.mel = { len: mel.length, sha: shaOf(mel), head: head(mel, 8) };

  // 5. the rendered samples.
  const audio = await engine.vocode(mel, seed);
  writeFileSync(join(outdir, "js.audio.bin"), f32bytes(audio));
  rec.audio = { len: audio.length, sha: shaOf(audio), head: head(audio, 8) };

  // 6. the long-form path.
  const out = await engine.synthesize(text, voice, { seed, language });
  writeFileSync(join(outdir, "js.longform.bin"), f32bytes(out.audio));
  rec.longform = {
    tokens: out.tokens,
    audio_len: out.audio.length,
    audio_sha: shaOf(out.audio),
    audio_head: head(out.audio, 8),
    n_chunks: out.chunks.length,
    hit_token_cap: out.hitTokenCap,
  };

  writeFileSync(join(outdir, "js.json"), JSON.stringify(rec, null, 1));
  console.error(`wrote ${join(outdir, "js.json")}`);
} finally {
  await engine.close();
}
