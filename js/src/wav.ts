/**
 * WAV, both directions, with no imports.
 *
 * Pure by design: nothing here reaches for `node:fs`, so encoding a render for
 * `<audio>` or an upload is possible wherever the package exports this module.
 * Today the only export is `"."`, which loads onnxruntime-node, so the browser
 * path needs a subpath export before it is reachable. The file half lives in
 * `wavFile.ts`.
 */

/** Decoded audio: mono float32 in [-1, 1], at the rate the file declared. */
export interface Wave {
  audio: Float32Array;
  sampleRate: number;
}

const WAVE_PCM = 1;
const WAVE_FLOAT = 3;
const WAVE_EXTENSIBLE = 0xfffe;

/**
 * Writing is `floor(x * 32768)` clipped to int16; reading divides by 32768.
 *
 * That is `loudkit.synthesis._quantise`, which is what libsndfile does, and
 * what the Go and Rust writers do: the same render saved from any port is the
 * same file, byte for byte. 0.9 is 29491, -0.9 is -29492, 1.0 clips to 32767,
 * -1.0 is -32768, NaN is 0.
 */
const INT16_MIN = -32768;
const INT16_MAX = 32767;
const INT16_FULL_SCALE = 32768;

function fourcc(view: DataView, at: number): string {
  return String.fromCharCode(
    view.getUint8(at),
    view.getUint8(at + 1),
    view.getUint8(at + 2),
    view.getUint8(at + 3)
  );
}

function writeFourcc(view: DataView, at: number, text: string): void {
  for (let i = 0; i < 4; i++) view.setUint8(at + i, text.charCodeAt(i));
}

/**
 * `audio` as a mono 16-bit PCM WAV.
 *
 * 16-bit because that is what every player, every phone and every upload form
 * accepts without asking; the engine's float32 is one `result.audio` away for
 * anyone who wants it. Samples outside [-1, 1] are clamped rather than wrapped:
 * a loud render should sound loud, not inverted.
 */
export function encodeWav(audio: Float32Array, sampleRate: number): Uint8Array {
  if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
    throw new Error(`sample rate must be a positive integer, got ${sampleRate}`);
  }
  const dataBytes = audio.length * 2;
  const out = new Uint8Array(44 + dataBytes);
  const view = new DataView(out.buffer);

  writeFourcc(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeFourcc(view, 8, "WAVE");
  writeFourcc(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, WAVE_PCM, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeFourcc(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  for (let i = 0; i < audio.length; i++) {
    const scaled = Math.floor(audio[i] * INT16_FULL_SCALE);
    const clamped = Number.isNaN(scaled)
      ? 0
      : scaled < INT16_MIN ? INT16_MIN : scaled > INT16_MAX ? INT16_MAX : scaled;
    view.setInt16(44 + i * 2, clamped, true);
  }
  return out;
}

/**
 * A WAV back to mono float32.
 *
 * 16-bit PCM and 32-bit float only, which is what recorders and every export
 * dialog produce. Multi-channel input is averaged down to mono, because that
 * is what enrollment reads and the alternative is asking a caller to write the
 * downmix themselves.
 */
export function decodeWav(bytes: Uint8Array): Wave {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.byteLength < 12 || fourcc(view, 0) !== "RIFF" || fourcc(view, 8) !== "WAVE") {
    throw new Error("not a WAV file: no RIFF/WAVE header");
  }
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bits = 0;
  let dataAt = -1;
  let dataBytes = 0;

  // Chunk walk rather than a fixed 44-byte header: real files carry LIST,
  // fact and metadata chunks before the data, and a reader that assumes the
  // canonical layout reads those as samples.
  let at = 12;
  while (at + 8 <= bytes.byteLength) {
    const id = fourcc(view, at);
    const size = view.getUint32(at + 4, true);
    const body = at + 8;
    if (id === "fmt " && size >= 16) {
      format = view.getUint16(body, true);
      channels = view.getUint16(body + 2, true);
      sampleRate = view.getUint32(body + 4, true);
      bits = view.getUint16(body + 14, true);
      if (format === WAVE_EXTENSIBLE && size >= 26) {
        // The real format is the first two bytes of the extension's GUID.
        format = view.getUint16(body + 24, true);
      }
    } else if (id === "data") {
      dataAt = body;
      dataBytes = Math.min(size, bytes.byteLength - body);
    }
    at = body + size + (size % 2);
  }

  if (dataAt < 0) throw new Error("not a WAV file: no data chunk");
  if (channels < 1 || sampleRate < 1) throw new Error("WAV header declares no channels or rate");
  const isPcm16 = format === WAVE_PCM && bits === 16;
  const isFloat32 = format === WAVE_FLOAT && bits === 32;
  if (!isPcm16 && !isFloat32) {
    throw new Error(
      `WAV format ${format} at ${bits} bits: this reader takes 16-bit PCM and 32-bit float`
    );
  }

  const width = bits / 8;
  const frames = Math.floor(dataBytes / (width * channels));
  const audio = new Float32Array(frames);
  for (let f = 0; f < frames; f++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      const off = dataAt + (f * channels + c) * width;
      sum += isPcm16 ? view.getInt16(off, true) / INT16_FULL_SCALE : view.getFloat32(off, true);
    }
    audio[f] = sum / channels;
  }
  return { audio, sampleRate };
}
