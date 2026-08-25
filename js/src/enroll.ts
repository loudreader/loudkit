/**
 * Enrollment: reference audio to a voice profile — a bit-parity port of
 * `loudkit.models.enroll` over the exported enrollment ONNX graphs.
 *
 * The DSP (resampler, filterbanks, trim) is implemented here and held to the
 * enrollment fixture; the model stages run through s3_tokenizer.onnx,
 * camp.onnx and voice_encoder.onnx. The filter tables and windows are embedded
 * as the same float32 data every port loads (see dspData.ts).
 */

import { type ExecutionOptions, type ResolvedONNXProvider } from "./execution.js";
import { ort } from "./ort.js";
import { Session, openSessions } from "./session.js";
import { DSP_B64, decodeF32 } from "./dspData.js";

const MEL_SR = 24000;
const S3_SR = 16000;
const MAX_REF_SECONDS = 10;
const COND_SECONDS = 6;

const FRAME_400 = 400;
const HOP_160 = 160;
const KALDI_FFT = 512;
const MATCHA_NFFT = 1920;
const MATCHA_HOP = 480;

const PARTIAL_FRAMES = 160;
const PARTIAL_STEP = 77;

const tables = new Map<string, Float32Array>();
function table(name: string): Float32Array {
  let t = tables.get(name);
  if (!t) {
    t = decodeF32(DSP_B64[name]);
    tables.set(name, t);
  }
  return t;
}

// ------------------------------------------------------------------ resampler

function gcd(a: number, b: number): number {
  while (b !== 0) [a, b] = [b, a % b];
  return a;
}

function sincHannKernel(orig: number, new_: number): [Float32Array[], number] {
  const base = Math.min(orig, new_) * 0.99;
  const width = Math.ceil((6 * orig) / base);
  const kernel: Float32Array[] = [];
  for (let phase = 0; phase < new_; phase++) {
    const row = new Float32Array(2 * width + orig);
    for (let idx = 0; idx < 2 * width + orig; idx++) {
      let t = -phase / new_ + (idx - width) / orig;
      t *= base;
      t = Math.min(6, Math.max(-6, t));
      const w = Math.cos((t * Math.PI) / 6 / 2) ** 2;
      const tt = t * Math.PI;
      const sinc = tt === 0 ? 1 : Math.sin(tt) / tt;
      row[idx] = sinc * w * (base / orig);
    }
    kernel.push(row);
  }
  return [kernel, width];
}

/** The one portable Hann-windowed-sinc resampler (mirror of resample.py). */
export function resample(waveform: Float32Array, origFreq: number, newFreq: number): Float32Array {
  if (origFreq === newFreq) return waveform.slice();
  const g = gcd(origFreq, newFreq);
  const orig = origFreq / g;
  const new_ = newFreq / g;

  const [kernel, width] = sincHannKernel(orig, new_);
  const taps = kernel[0].length;

  const padded = new Float32Array(width + waveform.length + width + orig);
  padded.set(waveform, width);

  const nOut = Math.floor((padded.length - taps) / orig) + 1;
  const out = new Float32Array(nOut * new_);
  for (let i = 0; i < nOut; i++) {
    const base = i * orig;
    for (let phase = 0; phase < new_; phase++) {
      let acc = 0;
      for (let c = 0; c < taps; c++) acc += kernel[phase][c] * padded[base + c];
      out[i * new_ + phase] = acc;
    }
  }
  const target = Math.ceil((new_ * waveform.length) / orig);
  return out.slice(0, target);
}

// ------------------------------------------------------------------ filterbanks

const basisCache = new Map<number, [Float64Array[], Float64Array[]]>();
function basis(nfft: number): [Float64Array[], Float64Array[]] {
  let b = basisCache.get(nfft);
  if (b) return b;
  const bins = nfft / 2 + 1;
  const cos: Float64Array[] = [];
  const sin: Float64Array[] = [];
  for (let k = 0; k < bins; k++) {
    const c = new Float64Array(nfft);
    const s = new Float64Array(nfft);
    for (let n = 0; n < nfft; n++) {
      const a = (-2 * Math.PI * k * n) / nfft;
      c[n] = Math.cos(a);
      s[n] = Math.sin(a);
    }
    cos.push(c);
    sin.push(s);
  }
  b = [cos, sin];
  basisCache.set(nfft, b);
  return b;
}

function powerSpectrum(frame: Float64Array, nfft: number): Float64Array {
  const [cos, sin] = basis(nfft);
  const bins = nfft / 2 + 1;
  const out = new Float64Array(bins);
  for (let k = 0; k < bins; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < nfft; n++) {
      re += cos[k][n] * frame[n];
      im += sin[k][n] * frame[n];
    }
    out[k] = re * re + im * im;
  }
  return out;
}

function melMultiply(filters: Float32Array, rows: number, bins: number, spectra: Float64Array[], frames: number): Float32Array {
  const out = new Float32Array(rows * frames);
  for (let r = 0; r < rows; r++) {
    for (let f = 0; f < frames; f++) {
      let acc = 0;
      for (let b = 0; b < bins; b++) acc += filters[r * bins + b] * spectra[b][f];
      out[r * frames + f] = acc;
    }
  }
  return out;
}

function centredPowerSpectra(samples: Float64Array, window: Float32Array, dropLast: boolean): Float64Array[] {
  const nfft = window.length;
  const half = nfft / 2;
  const padded = new Float64Array(samples.length + nfft);
  for (let i = 0; i < half; i++) padded[i] = samples[half - i];
  padded.set(samples, half);
  for (let i = 0; i < half; i++) padded[half + samples.length + i] = samples[samples.length - 2 - i];

  let frames = Math.floor(samples.length / HOP_160) + 1;
  if (dropLast) frames -= 1;
  const bins = nfft / 2 + 1;
  const out: Float64Array[] = [];
  for (let k = 0; k < bins; k++) out.push(new Float64Array(frames));
  for (let f = 0; f < frames; f++) {
    const start = f * HOP_160;
    const frame = new Float64Array(nfft);
    for (let i = 0; i < nfft; i++) frame[i] = padded[start + i] * window[i];
    const sp = powerSpectrum(frame, nfft);
    for (let k = 0; k < bins; k++) out[k][f] = sp[k];
  }
  return out;
}

function tokenizerMel(samples: Float64Array): [Float32Array, number] {
  const s3hann = table("s3_hann400");
  const s3mel = table("s3_mel_filters");
  const spectra = centredPowerSpectra(samples, s3hann, true);
  const frames = spectra[0].length;
  const mel = melMultiply(s3mel, 128, 201, spectra, frames);
  let peak = -Infinity;
  for (const v of mel) {
    const x = Math.log10(Math.max(v, 1e-10));
    if (x > peak) peak = x;
  }
  const ceiling = peak - 8;
  for (let i = 0; i < mel.length; i++) {
    let v = Math.log10(Math.max(mel[i], 1e-10));
    if (v < ceiling) v = ceiling;
    mel[i] = (v + 4) * 0.25;
  }
  return [mel, frames];
}

function matchaMel(samples: Float64Array): Float32Array {
  const matchaHann = table("matcha_hann1920");
  const matchaMel = table("matcha_mel_filters");
  const pad = (MATCHA_NFFT - MATCHA_HOP) / 2;
  const padded = new Float64Array(samples.length + 2 * pad);
  for (let i = 0; i < pad; i++) padded[i] = samples[pad - i];
  padded.set(samples, pad);
  for (let i = 0; i < pad; i++) padded[pad + samples.length + i] = samples[samples.length - 2 - i];

  const frames = Math.floor((padded.length - MATCHA_NFFT) / MATCHA_HOP) + 1;
  const bins = MATCHA_NFFT / 2 + 1;
  const spectra: Float64Array[] = [];
  for (let k = 0; k < bins; k++) spectra.push(new Float64Array(frames));
  for (let f = 0; f < frames; f++) {
    const start = f * MATCHA_HOP;
    const frame = new Float64Array(MATCHA_NFFT);
    for (let i = 0; i < MATCHA_NFFT; i++) frame[i] = padded[start + i] * matchaHann[i];
    const sp = powerSpectrum(frame, MATCHA_NFFT);
    for (let k = 0; k < bins; k++) spectra[k][f] = Math.sqrt(sp[k] + 1e-9);
  }
  const mel = melMultiply(matchaMel, 80, bins, spectra, frames);
  for (let i = 0; i < mel.length; i++) mel[i] = Math.log(Math.max(mel[i], 1e-5));
  return mel;
}

function kaldiFbank(samples: Float64Array): Float32Array {
  const kaldiMel = table("kaldi_mel_filters");
  const kaldiPovey = table("kaldi_povey400");
  const frames = Math.floor((samples.length - FRAME_400) / HOP_160) + 1;
  const bins = KALDI_FFT / 2 + 1;
  const spectra: Float64Array[] = [];
  for (let k = 0; k < bins; k++) spectra.push(new Float64Array(frames));
  for (let f = 0; f < frames; f++) {
    const start = f * HOP_160;
    const frame = new Float64Array(KALDI_FFT);
    let mean = 0;
    for (let i = 0; i < FRAME_400; i++) {
      frame[i] = samples[start + i];
      mean += frame[i];
    }
    mean /= FRAME_400;
    for (let i = 0; i < FRAME_400; i++) frame[i] -= mean;
    const prev = frame[0];
    for (let i = FRAME_400 - 1; i >= 1; i--) frame[i] -= 0.97 * frame[i - 1];
    frame[0] -= 0.97 * prev;
    for (let i = 0; i < FRAME_400; i++) frame[i] *= kaldiPovey[i];
    const sp = powerSpectrum(frame, KALDI_FFT);
    for (let k = 0; k < bins; k++) spectra[k][f] = sp[k];
  }
  const mel = melMultiply(kaldiMel, 80, 256, spectra, frames);
  const epsilon = 1.1920928955078125e-7;
  for (let i = 0; i < mel.length; i++) mel[i] = Math.log(Math.max(mel[i], epsilon));
  for (let b = 0; b < 80; b++) {
    let m = 0;
    for (let f = 0; f < frames; f++) m += mel[b * frames + f];
    m /= frames;
    for (let f = 0; f < frames; f++) mel[b * frames + f] -= m;
  }
  const out = new Float32Array(mel.length);
  for (let f = 0; f < frames; f++) for (let b = 0; b < 80; b++) out[f * 80 + b] = mel[b * frames + f];
  return out;
}

function voiceEncoderMel(samples: Float64Array): [Float32Array, number] {
  const veHann = table("voiceenc_hann400");
  const veMel = table("voiceenc_mel_filters");
  const spectra = centredPowerSpectra(samples, veHann, false);
  const frames = spectra[0].length;
  const binMajor = melMultiply(veMel, 40, 201, spectra, frames);
  const out = new Float32Array(frames * 40);
  for (let f = 0; f < frames; f++) for (let b = 0; b < 40; b++) out[f * 40 + b] = binMajor[b * frames + f];
  return [out, frames];
}

function trim(samples: Float64Array): Float64Array {
  const FRAME = 2048;
  const HOP = 512;
  const half = FRAME / 2;
  const padded = new Float64Array(samples.length + FRAME);
  for (let i = 0; i < half; i++) padded[i] = samples[half - i];
  padded.set(samples, half);
  for (let i = 0; i < half; i++) padded[half + samples.length + i] = samples[samples.length - 2 - i];

  const nFrames = 1 + Math.floor(samples.length / HOP);
  const rms = new Float64Array(nFrames);
  let peak = 0;
  for (let f = 0; f < nFrames; f++) {
    const start = f * HOP;
    let sum = 0;
    for (let i = start; i < start + FRAME; i++) sum += padded[i] * padded[i];
    const r = Math.sqrt(sum / FRAME);
    rms[f] = r;
    if (r > peak) peak = r;
  }
  let first = -1;
  let last = -1;
  for (let f = 0; f < nFrames; f++) {
    if (rms[f] > 0.1 * peak) {
      if (first === -1) first = f;
      last = f;
    }
  }
  if (first === -1) return samples.slice();
  const start = first * HOP;
  const end = Math.min(last * HOP + HOP, samples.length);
  return end <= start ? samples.slice() : samples.slice(start, end);
}

// -------------------------------------------------------------------- enroller

export interface Enrolled {
  speakerEmbedding: Float32Array;
  flowEmbedding: Float32Array;
  promptTokens: BigInt64Array;
  promptMel: Float32Array;
  promptMelFrames: number;
  condPromptTokens: BigInt64Array;
}

export class Enroller {
  private tokenizer: Session;
  private camp: Session;
  private ve: Session;

  /**
   * The execution provider the three enrollment graphs were opened on — never
   * `"auto"`, always the one that ran.
   */
  readonly onnxProvider: ResolvedONNXProvider;

  private constructor(
    provider: ResolvedONNXProvider,
    tokenizer: Session,
    camp: Session,
    ve: Session
  ) {
    this.onnxProvider = provider;
    this.tokenizer = tokenizer;
    this.camp = camp;
    this.ve = ve;
  }

  /**
   * `execution.onnxProvider` selects the execution provider, same vocabulary
   * and same rules as `Engine.load`. Enrollment is the other ONNX path, and a
   * knob that reaches one of them and not the other is a knob nobody can
   * describe.
   */
  static async load(onnxDir: string, execution: ExecutionOptions = {}): Promise<Enroller> {
    const { provider, sessions } = await openSessions(
      [
        ["tokenizer", `${onnxDir}/s3_tokenizer.onnx`],
        ["camp", `${onnxDir}/camp.onnx`],
        ["ve", `${onnxDir}/voice_encoder.onnx`],
      ],
      execution.onnxProvider
    );
    return new Enroller(provider, sessions.tokenizer, sessions.camp, sessions.ve);
  }

  async enroll(audio: Float32Array, sampleRate: number): Promise<Enrolled> {
    // The guard the other four got and this one did not. A non-positive rate
    // reaches the resampler as a division by zero, which here surfaces as
    // `RangeError: offset is out of bounds` from a typed-array constructor —
    // an internal error, not a diagnosis. Same sentence as Go's.
    if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
      throw new Error(`sample rate must be positive, got ${sampleRate}`);
    }
    const wav = Float64Array.from(audio);
    const wav24Full = sampleRate === MEL_SR ? wav : Float64Array.from(resample(audio, sampleRate, MEL_SR));
    const maxSamples = MAX_REF_SECONDS * MEL_SR;
    const wav24 = wav24Full.slice(0, maxSamples);
    const wav16Flow = Float64Array.from(resample(Float32Array.from(wav24), MEL_SR, S3_SR));
    const wav16T3 = Float64Array.from(resample(Float32Array.from(wav24Full), MEL_SR, S3_SR));

    const promptMel = matchaMel(wav24);
    const promptMelFrames = promptMel.length / 80;

    const [tokMel] = tokenizerMel(wav16Flow);
    const tokens = await this.tokenize(tokMel);
    const nTok = Math.min(tokens.length, Math.floor(promptMelFrames / 2));
    const promptTokens = tokens.slice(0, nTok);
    const promptMelOut = promptMel.slice(0, 80 * 2 * nTok);

    const condSamples = Math.min(COND_SECONDS * S3_SR, wav16T3.length);
    const [condMel] = tokenizerMel(wav16T3.slice(0, condSamples));
    const condTokens = await this.tokenizeCapped(condMel, 150);

    const fbank = kaldiFbank(wav16Flow);
    const flowEmbedding = await this.camEmbedding(fbank);

    const speakerEmbedding = await this.speakerEmbedding(wav16T3);

    return {
      speakerEmbedding,
      flowEmbedding,
      promptTokens,
      promptMel: promptMelOut,
      promptMelFrames: 2 * nTok,
      condPromptTokens: condTokens,
    };
  }

  private async tokenize(mel: Float32Array): Promise<BigInt64Array> {
    const frames = mel.length / 128;
    const input = new ort.Tensor("float32", mel, [1, 128, frames]);
    const outputs = await this.tokenizer.run({ mel: input });
    return outputs.tokens.data as BigInt64Array;
  }

  private async tokenizeCapped(mel: Float32Array, cap: number): Promise<BigInt64Array> {
    if (mel.length / 128 > cap * 4) return this.tokenize(mel.slice(0, 128 * cap * 4));
    return this.tokenize(mel);
  }

  private async camEmbedding(fbank: Float32Array): Promise<Float32Array> {
    const frames = fbank.length / 80;
    const transposed = new Float32Array(fbank.length);
    for (let f = 0; f < frames; f++) for (let b = 0; b < 80; b++) transposed[b * frames + f] = fbank[f * 80 + b];
    const input = new ort.Tensor("float32", transposed, [1, 80, frames]);
    const outputs = await this.camp.run({ fbank: input });
    return outputs.out.data as Float32Array;
  }

  private async speakerEmbedding(wav16T3: Float64Array): Promise<Float32Array> {
    const trimmed = trim(wav16T3);
    const [mel, frames] = voiceEncoderMel(trimmed);

    let nWins = 0;
    let rem = 0;
    const span = mel.length / 40;
    if (span > PARTIAL_FRAMES - PARTIAL_STEP) {
      nWins = Math.floor((span - PARTIAL_FRAMES + PARTIAL_STEP) / PARTIAL_STEP);
      rem = (span - PARTIAL_FRAMES + PARTIAL_STEP) % PARTIAL_STEP;
    }
    if (nWins === 0 || (rem + (PARTIAL_FRAMES - PARTIAL_STEP)) / PARTIAL_FRAMES >= 0.8) nWins += 1;
    const target = PARTIAL_FRAMES + PARTIAL_STEP * (nWins - 1);
    let melOut = mel;
    if (target > frames) {
      const padded = new Float32Array(target * 40);
      padded.set(mel);
      melOut = padded;
    }

    const partials = new Float32Array(nWins * PARTIAL_FRAMES * 40);
    for (let i = 0; i < nWins; i++) {
      const start = i * PARTIAL_STEP * 40;
      partials.set(melOut.slice(start, start + PARTIAL_FRAMES * 40), i * PARTIAL_FRAMES * 40);
    }

    const input = new ort.Tensor("float32", partials, [nWins, PARTIAL_FRAMES, 40]);
    const outputs = await this.ve.run({ partials: input });
    const perPartial = outputs.out.data as Float32Array;

    const pooled = new Float32Array(256);
    for (let i = 0; i < nWins; i++) for (let d = 0; d < 256; d++) pooled[d] += perPartial[i * 256 + d];
    let norm = 0;
    for (const v of pooled) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < pooled.length; i++) pooled[i] /= norm;
    return pooled;
  }
}
