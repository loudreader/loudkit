/**
 * Minimal safetensors reader — enough to load the checkpoint's embedding
 * tables and a voice profile, with no dependency.
 *
 * Format: 8-byte little-endian header length, a JSON header naming each tensor
 * with its dtype, shape and byte offsets, then the raw tensors in order.
 */

import { readFileSync } from "node:fs";

export interface TensorSpec {
  dtype: "F32" | "F64" | "F16" | "I64" | "I32" | "I8" | "U8" | "BOOL";
  shape: number[];
  data: Uint8Array;
}

/**
 * Bytes per element, by dtype. Listed rather than inferred: an unknown dtype
 * must be refused at load, not discovered later by whichever accessor is asked
 * for it first.
 */
const BYTE_WIDTH: Record<string, number> = {
  F64: 8, I64: 8, U64: 8,
  F32: 4, I32: 4, U32: 4,
  F16: 2, BF16: 2, I16: 2, U16: 2,
  I8: 1, U8: 1, BOOL: 1,
};

export class SafetensorsFile {
  private tensors: Map<string, TensorSpec>;

  constructor(path: string) {
    const buf = readFileSync(path);
    const len = Number(readUint64(buf, 0));
    // Range-checked before slicing. `subarray` clamps silently, so a header
    // length past the end of the file fed JSON.parse whatever happened to be
    // there (often nothing, which throws a SyntaxError about the JSON rather
    // than about this file).
    if (8 + len > buf.length) {
      throw new Error(`${path}: header overruns file — truncated or corrupt`);
    }
    const header = JSON.parse(buf.subarray(8, 8 + len).toString("utf8"));
    const tensors = new Map<string, TensorSpec>();
    let metadata: Record<string, unknown> | null = null;
    for (const [name, spec] of Object.entries(header)) {
      if (name === "__metadata__") {
        metadata = spec as Record<string, unknown>;
        continue;
      }
      const s = spec as { dtype: string; shape: number[]; data_offsets: [number, number] };
      const [begin, end] = s.data_offsets;
      const base = 8 + len;
      // Range-checked before slicing. `subarray` clamps silently, so an
      // out-of-range tensor became a short buffer rather than an error, and the
      // reader below then produced fewer elements than the shape promised.
      if (!(begin >= 0 && begin <= end && base + end <= buf.length)) {
        throw new Error(
          `${path}: tensor ${name} spans ${begin}..${end} of a ` +
            `${buf.length - base}-byte payload — the file is truncated or the header is corrupt`
        );
      }
      // The shape must account for exactly the bytes claimed. Callers read
      // `shape` to size their work, so a header declaring [256] over four bytes
      // of payload is not a bad tensor but a reader computing with a length the
      // data does not have.
      const width = BYTE_WIDTH[s.dtype];
      if (width === undefined) {
        throw new Error(`${path}: tensor ${name} has unknown dtype ${s.dtype}`);
      }
      let elements = 1;
      for (const dim of s.shape) {
        if (!Number.isSafeInteger(dim) || dim < 0) {
          throw new Error(`${path}: tensor ${name} has a bad dimension in [${s.shape.join(", ")}]`);
        }
        elements *= dim;
        if (!Number.isSafeInteger(elements)) {
          throw new Error(`${path}: tensor ${name} shape [${s.shape.join(", ")}] overflows`);
        }
      }
      if (elements * width !== end - begin) {
        throw new Error(
          `${path}: tensor ${name} declares shape [${s.shape.join(", ")}] of ${s.dtype} ` +
            `(${elements * width} bytes) but occupies ${end - begin} bytes — ` +
            `the header does not describe the payload`
        );
      }
      tensors.set(name, {
        dtype: s.dtype as TensorSpec["dtype"],
        shape: s.shape,
        data: buf.subarray(base + begin, base + end),
      });
    }
    this.tensors = tensors;
    this.metadata = metadata;
  }

  /** The embedded `__metadata__` value, if any (e.g. a voice header). */
  metadata: Record<string, unknown> | null;

  has(name: string): boolean {
    return this.tensors.has(name);
  }

  /** Float32 tensor, read as a Float32Array of `shape`. */
  f32(name: string): Float32Array {
    const t = this.require(name);
    if (t.dtype === "F32") return copyFromBytes(t.data, Float32Array);
    if (t.dtype === "F16") return halfToFloat(t.data);
    throw new Error(`${name}: expected F32, got ${t.dtype}`);
  }

  /** Int64 tensor, read as a BigInt64Array of `shape`. */
  i64(name: string): BigInt64Array {
    const t = this.require(name);
    if (t.dtype !== "I64") throw new Error(`${name}: expected I64, got ${t.dtype}`);
    return copyFromBytes(t.data, BigInt64Array);
  }

  /** All tensor names. */
  keys(): string[] {
    return [...this.tensors.keys()];
  }

  private require(name: string): TensorSpec {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`no tensor named ${name}; have ${this.tensors.size} tensors`);
    return t;
  }
}

function readUint64(buf: Buffer, off: number): bigint {
  return buf.readBigUInt64LE(off);
}

/** Copy a byte slice into a fresh typed array, honouring any subarray offset. */
function copyFromBytes<T extends Float32Array | BigInt64Array | Uint16Array>(
  bytes: Uint8Array,
  ctor: new (arg: ArrayBufferLike) => T
): T {
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  return new ctor(buffer);
}

/** IEEE 754 half-precision -> float32, exact upcast (every fp16 value is
 * representable in fp32). This is how the checkpoint's fp16 embedding tables
 * become the fp32 tables the ONNX graphs were exported with. */
function halfToFloat(halfData: Uint8Array): Float32Array {
  const halves = copyFromBytes(halfData, Uint16Array);
  const out = new Float32Array(halves.length);
  for (let i = 0; i < halves.length; i++) {
    out[i] = halfToFloatOne(halves[i]);
  }
  return out;
}

function halfToFloatOne(h: number): number {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    // subnormal: value = frac * 2^-24
    return sign * frac * 2 ** -24;
  }
  if (exp === 31) {
    return frac === 0 ? sign * Infinity : NaN;
  }
  // normal: value = (1 + frac/1024) * 2^(exp-15)
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}
