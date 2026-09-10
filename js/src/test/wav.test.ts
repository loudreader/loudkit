/**
 * The WAV codec, against the bytes Python writes.
 *
 * The header and the quantisation come from the shared fixtures, so the file
 * a JS user gets is the file the Python server returns for the same floats. A
 * round trip alone would pass for an encoder that wrote its own format.
 */

import assert from "node:assert";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { asSamples } from "../enroll.js";
import { decodeWav, encodeWav } from "../wav.js";
import { readWav, saveWav, withWav } from "../wavFile.js";

function bytes(...values: number[]): Uint8Array {
  return Uint8Array.from(values);
}

/** One of the shared fixtures, found by walking up to the repo root. */
function fixture(name: string): any {
  const env = process.env.LOUDKIT_FIXTURE_DIR;
  let dir = env ?? dirname(fileURLToPath(import.meta.url));
  for (;;) {
    const candidate = env ? join(dir, name) : join(dir, "tests", "data", "conformance", name);
    if (existsSync(candidate)) return JSON.parse(readFileSync(candidate, "utf8"));
    const parent = dirname(dir);
    if (parent === dir) throw new Error(`cannot locate ${name}: run from the loudkit repo`);
    dir = parent;
  }
}

/** A probe is a number, or the string "nan", which JSON cannot spell. */
function probe(x: number | string): number {
  return x === "nan" ? NaN : Number(x);
}

test("the WAV bytes are the fixture's, header and samples", () => {
  const cases = fixture("wav_header.json").cases;
  assert.ok(cases.length >= 3, "the fixture holds cases");
  for (const c of cases) {
    const wav = encodeWav(Float32Array.from(c.samples.map(probe)), c.sample_rate);
    assert.equal(Buffer.from(wav).toString("hex"), c.hex, `${c.samples.length} samples at ${c.sample_rate} Hz`);
  }
});

test("the quantisation is the shared rule: floor by 32768, clipped, NaN to 0", () => {
  const cases = fixture("wav_quantise.json").cases;
  assert.ok(cases.length >= 10, "the fixture holds probes");
  const wav = encodeWav(Float32Array.from(cases.map((c: any) => probe(c.x))), 8000);
  const view = new DataView(wav.buffer, 44);
  cases.forEach((c: any, i: number) => {
    assert.equal(view.getInt16(2 * i, true), c.pcm16, String(c.x));
  });
});

test("a non-integer or non-positive rate is refused", () => {
  assert.throws(() => encodeWav(new Float32Array(1), 0), /positive integer/);
  assert.throws(() => encodeWav(new Float32Array(1), 24000.5), /positive integer/);
});

test("the decoder reads back what the encoder wrote", () => {
  const audio = Float32Array.from([0, 0.5, -0.5, 1, -1]);
  const back = decodeWav(encodeWav(audio, 16000));
  assert.equal(back.sampleRate, 16000);
  assert.equal(back.audio.length, audio.length);
  for (let i = 0; i < audio.length; i++) {
    assert.ok(Math.abs(back.audio[i] - audio[i]) < 1e-4, `sample ${i}: ${back.audio[i]}`);
  }
});

test("32-bit float WAVs decode, and stereo folds down to mono", () => {
  // Hand-built: format 3, two channels, one frame of (1.0, 0.0).
  const head = bytes(
    0x52, 0x49, 0x46, 0x46, 0x2c, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45,
    0x66, 0x6d, 0x74, 0x20, 0x10, 0x00, 0x00, 0x00,
    0x03, 0x00, // IEEE float
    0x02, 0x00, // stereo
    0x40, 0x1f, 0x00, 0x00, // 8000 Hz
    0x00, 0xfa, 0x00, 0x00, 0x08, 0x00, 0x20, 0x00,
    0x64, 0x61, 0x74, 0x61, 0x08, 0x00, 0x00, 0x00
  );
  const body = new Uint8Array(8);
  new DataView(body.buffer).setFloat32(0, 1, true);
  const file = new Uint8Array([...head, ...body]);
  const wave = decodeWav(file);
  assert.equal(wave.sampleRate, 8000);
  assert.deepEqual([...wave.audio], [0.5]);
});

test("a chunk before the data is skipped rather than read as samples", () => {
  const plain = encodeWav(Float32Array.from([0.25]), 24000);
  // Splice a 4-byte LIST chunk in between "WAVE" and "fmt ".
  const list = bytes(0x4c, 0x49, 0x53, 0x54, 0x04, 0x00, 0x00, 0x00, 1, 2, 3, 4);
  const spliced = new Uint8Array([...plain.slice(0, 12), ...list, ...plain.slice(12)]);
  new DataView(spliced.buffer).setUint32(4, spliced.length - 8, true);
  assert.deepEqual([...decodeWav(spliced).audio], [...decodeWav(plain).audio]);
});

test("what is not a WAV, or not a format this reads, is named", () => {
  assert.throws(() => decodeWav(bytes(1, 2, 3)), /RIFF\/WAVE/);
  const wav = encodeWav(Float32Array.from([0]), 24000);
  const eightBit = Uint8Array.from(wav);
  new DataView(eightBit.buffer).setUint16(34, 8, true);
  assert.throws(() => decodeWav(eightBit), /16-bit PCM and 32-bit float/);
});

test("saveWav writes the fixture's bytes and readWav reads the floats back", () => {
  const dir = mkdtempSync(join(tmpdir(), "loudkit-wav-"));
  try {
    const c = fixture("wav_header.json").cases[0];
    const path = join(dir, "out.wav");
    saveWav(path, Float32Array.from(c.samples.map(probe)), c.sample_rate);
    assert.equal(readFileSync(path).toString("hex"), c.hex);
    const back = readWav(path);
    assert.equal(back.sampleRate, c.sample_rate);
    assert.equal(back.audio.length, c.samples.length);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("enrollment takes a WAV path, WAV bytes, or samples with their rate", () => {
  const dir = mkdtempSync(join(tmpdir(), "loudkit-wav-"));
  try {
    const path = join(dir, "ref.wav");
    saveWav(path, Float32Array.from([0.5, -0.5]), 16000);
    const bytes = readFileSync(path);

    assert.equal(asSamples(path).sampleRate, 16000);
    assert.equal(asSamples(bytes).sampleRate, 16000);
    assert.deepEqual([...asSamples(bytes).audio], [...asSamples(path).audio]);
    assert.equal(asSamples(Float32Array.from([0]), 8000).sampleRate, 8000);

    // A rate is required for raw samples and refused for a WAV: the file
    // already says what it is, and the two disagreeing is a mistake.
    assert.throws(() => asSamples(Float32Array.from([0])), /need their sample rate/);
    assert.throws(() => asSamples(bytes, 16000), /carries its own sample rate/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("withWav hangs the two exits off the render itself", () => {
  const dir = mkdtempSync(join(tmpdir(), "loudkit-wav-"));
  try {
    const c = fixture("wav_header.json").cases[0];
    const input = { audio: Float32Array.from(c.samples.map(probe)), sampleRate: c.sample_rate, tokens: [7] };
    const render = withWav(input);
    // The same object, decorated: `Object.assign` is the contract.
    assert.strictEqual(render, input);
    assert.deepEqual(render.tokens, [7]);
    assert.equal(Buffer.from(render.toWav()).toString("hex"), c.hex);
    const path = join(dir, "render.wav");
    render.saveWav(path);
    assert.equal(readFileSync(path).toString("hex"), c.hex);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
