import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { loadVoice, saveVoice } from "../voice.js";
import { SafetensorsFile } from "../safetensors.js";
import { fileURLToPath } from "node:url";

/**
 * A voice profile written straight to bytes, so a shape or a header value can
 * be declared that `saveVoice` would never produce.
 *
 * The reader is the thing under test and it reads files other builds wrote, so
 * a fixture that can only be made by this port's writer cannot exercise it.
 */
function writeProfile(
  path: string,
  over: {
    mel?: number[];
    speaker?: number[];
    flow?: number[];
    tokens?: number[];
    cond?: number[];
    header?: Record<string, unknown>;
  } = {}
): string {
  const shapes = {
    speaker_embedding: over.speaker ?? [256],
    flow_embedding: over.flow ?? [192],
    prompt_tokens: over.tokens ?? [4],
    prompt_mel: over.mel ?? [80, 3],
    cond_prompt_tokens: over.cond ?? [4],
  };
  const width: Record<string, number> = { speaker_embedding: 4, flow_embedding: 4, prompt_mel: 4, prompt_tokens: 8, cond_prompt_tokens: 8 };
  const dtype: Record<string, string> = { speaker_embedding: "F32", flow_embedding: "F32", prompt_mel: "F32", prompt_tokens: "I64", cond_prompt_tokens: "I64" };
  const spec: Record<string, unknown> = {};
  const chunks: Buffer[] = [];
  let at = 0;
  for (const [name, shape] of Object.entries(shapes)) {
    const n = shape.reduce((a, b) => a * b, 1);
    const bytes = Buffer.alloc(n * width[name]);
    // Ones, not zeros: a zero speaker vector is refused for its norm, which is
    // a different refusal from the one under test.
    for (let i = 0; i < n; i++) {
      if (dtype[name] === "F32") bytes.writeFloatLE(1, i * 4);
      else bytes.writeBigInt64LE(1n, i * 8);
    }
    spec[name] = { dtype: dtype[name], shape, data_offsets: [at, at + bytes.length] };
    chunks.push(bytes);
    at += bytes.length;
  }
  spec.__metadata__ = {
    voice: JSON.stringify({
      format_version: 1,
      name: "probe",
      source_sample_rate: 24_000,
      language: "en",
      enrolment: "first-10s",
      ...(over.header ?? {}),
    }),
  };
  const head = Buffer.from(JSON.stringify(spec), "utf8");
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(head.length));
  writeFileSync(path, Buffer.concat([len, head, ...chunks]));
  return path;
}

test("a voice profile is read by its declared shape, not its element count", () => {
  // A tensor's shape says something the element count does not: `prompt_mel`
  // declared [2, 80] holds the same 160 floats as [80, 2] and means the
  // transpose of them, so the profile loaded here and the voice was read on
  // its side. A 256-float vector declared [16, 16] or [1, 256] is the same
  // shape of mistake one rank up. Every refusal below is one the reference
  // makes, measured through `loudkit.voice.VoiceProfile.load`.
  const dir = mkdtempSync(join(tmpdir(), "loudkit-voice-shape-"));
  try {
    const p = (n: string) => join(dir, `${n}.safetensors`);
    assert.doesNotThrow(() => loadVoice(writeProfile(p("ok"), { mel: [80, 2] })));
    // A profile with no mel frames at all is still (80, frames).
    assert.doesNotThrow(() => loadVoice(writeProfile(p("empty"), { mel: [80, 0] })));
    for (const mel of [[2, 80], [160], [1, 80, 2], [40, 4], [16, 10], [81, 3]]) {
      assert.throws(
        () => loadVoice(writeProfile(p("mel"), { mel })),
        /prompt_mel must be \(80, frames\)/,
        `mel ${JSON.stringify(mel)}`
      );
    }
    assert.throws(() => loadVoice(writeProfile(p("s1"), { speaker: [16, 16] })), /speaker_embedding must be 1-D/);
    assert.throws(() => loadVoice(writeProfile(p("s2"), { speaker: [1, 256] })), /speaker_embedding must be 1-D/);
    assert.throws(() => loadVoice(writeProfile(p("f1"), { flow: [2, 96] })), /flow_embedding must be 1-D/);
    assert.throws(() => loadVoice(writeProfile(p("t1"), { tokens: [2, 2] })), /prompt_tokens must be 1-D/);
    assert.throws(() => loadVoice(writeProfile(p("c1"), { cond: [1, 4] })), /cond_prompt_tokens must be 1-D/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("a voice profile names a law this build implements, or is refused", () => {
  // A profile naming a strategy this build does not implement was cut from its
  // recording a different way, so loading it speaks in a different voice under
  // the same name. Python and Go refuse it; this port loaded it.
  const dir = mkdtempSync(join(tmpdir(), "loudkit-voice-law-"));
  try {
    const p = (n: string) => join(dir, `${n}.safetensors`);
    for (const law of ["first-10s", "first-10s-pause"]) {
      assert.equal(loadVoice(writeProfile(p("ok"), { header: { enrolment: law } })).enrolment, law);
    }
    // Absent means the profile predates the field, and every one of those was
    // cut from the first ten seconds.
    assert.equal(loadVoice(writeProfile(p("absent"), { header: { enrolment: undefined } })).enrolment, "first-10s");
    for (const law of ["first-30s-nonsense", "", "First-10s", "first-10s ", "unknown"]) {
      assert.throws(
        () => loadVoice(writeProfile(p("bad"), { header: { enrolment: law } })),
        /enrolment strategy/,
        JSON.stringify(law)
      );
    }
    // A law that is not a string is refused for not being one, before the
    // membership check it could never pass.
    assert.throws(
      () => loadVoice(writeProfile(p("bad"), { header: { enrolment: 5 } })),
      /voice header 'enrolment' must be a string, got 5/
    );
    // And on the way out, because `VoiceProfile` here is an interface rather
    // than the frozen dataclass the reference validates on construction.
    const good = loadVoice(writeProfile(p("src")));
    assert.throws(
      () => saveVoice({ ...good, enrolment: "first-30s-nonsense" }, p("out")),
      /enrolment strategy/
    );
    // A rate is metadata, and every duration derived from a profile is samples
    // over this number.
    assert.throws(
      () => loadVoice(writeProfile(p("rate"), { header: { source_sample_rate: -48_000 } })),
      /source_sample_rate must be positive/
    );
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("a voice header that is not an object is refused by name", () => {
  // A header holding a JSON list, number or string answered `undefined` for
  // every key and reported "voice format version 0", which blames a version
  // the file does not carry; one holding `null` reached a bare TypeError
  // naming no file at all. The reference and Swift both refuse it by name, and
  // Swift's comment already claimed this port did.
  const dir = mkdtempSync(join(tmpdir(), "loudkit-voice-hdr-"));
  try {
    for (const [raw, phrase] of [
      ["null", "null"],
      ["[1,2,3]", "an array"],
      ["5", "a number"],
      ['"hello"', "a string"],
      ["true", "a boolean"],
    ] as const) {
      const path = join(dir, "v.safetensors");
      writeProfile(path);
      rewriteVoiceHeader(path, raw);
      assert.throws(
        () => loadVoice(path),
        new RegExp(`voice header is ${phrase.replace(/[[\]]/g, "\\$&")}, expected a JSON object`),
        raw
      );
    }
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test("a voice header value of the wrong JSON type is refused by name", () => {
  // `language: 5` was the language id "5" here, and language selects the text
  // funnel, so a voice whose header says 5 was read through whichever funnel
  // the coercion landed on. `_header_str` and `_header_int` refuse the five
  // fields the header carries, in these sentences.
  const dir = mkdtempSync(join(tmpdir(), "loudkit-voice-type-"));
  try {
    const path = join(dir, "v.safetensors");
    const refuses = (header: Record<string, unknown>, wanted: RegExp) =>
      assert.throws(() => loadVoice(writeProfile(path, { header })), wanted, JSON.stringify(header));

    for (const key of ["name", "language", "enrolment"]) {
      refuses({ [key]: 5 }, new RegExp(`v.safetensors: voice header '${key}' must be a string, got 5$`));
      refuses({ [key]: null }, new RegExp(`voice header '${key}' must be a string, got None$`));
      refuses({ [key]: true }, new RegExp(`voice header '${key}' must be a string, got True$`));
      refuses({ [key]: ["en"] }, new RegExp(`voice header '${key}' must be a string, got \\['en'\\]$`));
    }
    for (const key of ["format_version", "source_sample_rate"]) {
      refuses({ [key]: "1" }, new RegExp(`voice header '${key}' must be a number, got '1'$`));
      refuses({ [key]: true }, new RegExp(`voice header '${key}' must be a number, got True$`));
      // A fraction reached the algorithm as its truncation: 48000.7 divided
      // every duration a profile reports by a rate no reader would agree on,
      // and 1.9 was the version this build implements.
      refuses({ [key]: 1.9 }, new RegExp(`voice header '${key}' must be a whole number, got 1.9$`));
    }
    // An absent key still takes its default, which is the branch every profile
    // written before a field existed goes down.
    const bare = loadVoice(
      writeProfile(path, {
        header: { name: undefined, language: undefined, enrolment: undefined, source_sample_rate: undefined },
      })
    );
    assert.equal(bare.language, "en");
    assert.equal(bare.sourceSampleRate, 24_000);
    assert.equal(bare.enrolment, "first-10s");
    assert.equal(bare.name, "v");
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

/** Replace a written profile's `voice` header with this raw JSON document. */
function rewriteVoiceHeader(path: string, raw: string): void {
  const buf = readFileSync(path);
  const len = Number(buf.readBigUInt64LE(0));
  const spec = JSON.parse(buf.subarray(8, 8 + len).toString("utf8")) as Record<string, unknown>;
  (spec.__metadata__ as Record<string, unknown>).voice = raw;
  const head = Buffer.from(JSON.stringify(spec), "utf8");
  const out = Buffer.alloc(8);
  out.writeBigUInt64LE(BigInt(head.length));
  writeFileSync(path, Buffer.concat([out, head, buf.subarray(8 + len)]));
}

test("legacy Python voice defaults enrolment and pause-cut roundtrip preserves it", () => {
  const profile = loadVoice(fileURLToPath(new URL("../../../tests/data/enrollment/profile.safetensors", import.meta.url)));
  assert.equal(profile.enrolment, "first-10s");

  const dir = mkdtempSync(join(tmpdir(), "loudkit-voice-roundtrip-"));
  try {
    for (const strategy of ["first-10s", "first-10s-pause"]) {
      profile.enrolment = strategy;
      const path = join(dir, "voice.safetensors");
      saveVoice(profile, path);
      const out = loadVoice(path);
      assert.deepEqual(out, profile);
      assert.equal(JSON.parse(new SafetensorsFile(path).metadata!.voice as string).enrolment, strategy);
    }
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
