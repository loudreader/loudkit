/**
 * Minimal safetensors reader: enough to load the checkpoint's embedding
 * tables and a voice profile, with no dependency.
 *
 * Format: 8-byte little-endian header length, a JSON header naming each tensor
 * with its dtype, shape and byte offsets, then the raw tensors in order.
 *
 * The header is read on its own and each tensor is read when it is asked for,
 * because the checkpoint is 747 MB and the five tables a runtime backend wants
 * out of it are about 50 MB. Reading the file whole is the difference between
 * working and not on a phone or a small container, and it is also what made
 * "what manifest does this file carry" a second copy of this parser elsewhere:
 * the reader could not answer that without reading 747 MB.
 */

import { closeSync, openSync, readSync, statSync, writeFileSync } from "node:fs";

export interface TensorSpec {
  dtype: "F32" | "F64" | "F16" | "I64" | "I32" | "I8" | "U8" | "BOOL";
  shape: number[];
  data: Uint8Array;
}

/** What the header says about one tensor, and where its bytes are. */
export interface TensorPlace {
  dtype: TensorSpec["dtype"];
  shape: number[];
  /** Absolute file offsets, the payload base already added. */
  begin: number;
  end: number;
}

/** A parsed header: every tensor's place, and the `__metadata__` value. */
export interface SafetensorsHeader {
  tensors: Map<string, TensorPlace>;
  metadata: Record<string, unknown> | null;
}

/**
 * The largest header this reader will read into memory.
 *
 * A safetensors header is JSON naming tensors; the checkpoint's is a few
 * hundred kilobytes. A length past this is a corrupt file or a different
 * format, and the point of the cap is that neither is read as an allocation.
 *
 * The reference's number, from `checkpoint.py`'s `100 << 20`. Go, Rust and
 * Swift read a header of any length the file can hold, so any cap here is
 * stricter than theirs; a cap that is stricter than the reference's as well
 * refuses a band of files all four of the others open.
 */
const MAX_HEADER_BYTES = 100 * 1024 * 1024;

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

/**
 * One tensor's header entry, checked rather than asserted.
 *
 * `spec as { dtype, shape, data_offsets }` was a claim about JSON somebody
 * else wrote. A header missing `data_offsets` destructured `undefined` and
 * threw `TypeError: undefined is not iterable`, naming neither the file nor
 * the tensor; a missing `shape` threw `s.shape is not iterable`; a spec that
 * was a number, a string, a list or null threw one of those two; and
 * `data_offsets: [0, 8, 9]` loaded, silently keeping the first two. Seven of
 * fifteen malformed specs surfaced as a bare TypeError and one as no error at
 * all. The reference's reader refuses all fifteen, and names the field.
 *
 * These files arrive by download, so the message is the whole of what a caller
 * has to work with: it must say which file and which tensor.
 */
function tensorSpec(
  path: string,
  name: string,
  spec: unknown
): { dtype: string; shape: number[]; data_offsets: [number, number] } {
  const at = `${path}: tensor ${name}`;
  const kind = (v: unknown): string =>
    v === null ? "null" : Array.isArray(v) ? "a list" : typeof v;
  if (typeof spec !== "object" || spec === null || Array.isArray(spec)) {
    throw new Error(`${at} is ${kind(spec)}, not a tensor entry`);
  }
  const s = spec as Record<string, unknown>;
  if (typeof s.dtype !== "string") {
    throw new Error(`${at} has no dtype: a tensor entry names dtype, shape and data_offsets`);
  }
  if (!Array.isArray(s.shape)) {
    throw new Error(`${at} has no shape: a tensor entry names dtype, shape and data_offsets`);
  }
  const offsets = s.data_offsets;
  if (!Array.isArray(offsets)) {
    throw new Error(
      `${at} declares data_offsets as ${kind(offsets)}: a tensor entry names ` +
        "where its bytes begin and end, as a pair"
    );
  }
  if (offsets.length !== 2) {
    throw new Error(
      `${at} declares ${offsets.length} data_offsets: a tensor entry names ` +
        "where its bytes begin and end, as a pair"
    );
  }
  if (typeof offsets[0] !== "number" || typeof offsets[1] !== "number") {
    throw new Error(`${at} has non-numeric data_offsets [${offsets.join(", ")}]`);
  }
  return {
    dtype: s.dtype,
    shape: s.shape as number[],
    data_offsets: [offsets[0], offsets[1]],
  };
}

export class SafetensorsFile {
  private path: string;
  private tensors: Map<string, TensorPlace>;

  constructor(path: string) {
    const { tensors, metadata } = SafetensorsFile.header(path);
    this.path = path;
    this.tensors = tensors;
    this.metadata = metadata;
  }

  /**
   * The header alone, without a byte of payload.
   *
   * Public because the hub asks a checkpoint what manifest it carries while
   * deciding whether to open it at all, and that question had grown its own
   * copy of this parser for want of an answer that did not read the file.
   */
  static header(path: string): SafetensorsHeader {
    const size = statSync(path).size;
    const fd = openSync(path, "r");
    let len: number;
    let header: Record<string, unknown>;
    try {
      const lengthBytes = Buffer.alloc(8);
      if (readSync(fd, lengthBytes, 0, 8, 0) !== 8) {
        throw new Error(`${path}: too short to hold a safetensors header length`);
      }
      len = Number(lengthBytes.readBigUInt64LE(0));
      if (len === 0 || len > MAX_HEADER_BYTES) {
        throw new Error(
          `${path}: header length ${len} is not a safetensors header ` +
            `(the limit is ${MAX_HEADER_BYTES} bytes)`
        );
      }
      // Range-checked before it is read, so a header length past the end of the
      // file is named here rather than reaching JSON.parse as whatever happened
      // to be there (often nothing, which throws a SyntaxError about the JSON
      // rather than about this file).
      if (8 + len > size) {
        throw new Error(`${path}: header overruns file, truncated or corrupt`);
      }
      const body = Buffer.alloc(len);
      readExactly(fd, body, 8, path, "header");
      // Checked to be an object, which `checkpoint.py`, `rust/src/safetensors.rs`,
      // `go/safetensors` and `swift/LoudKit/Safetensors.swift` each do in their
      // own words. `Object.entries(5)` is the empty list, so a header holding a
      // bare JSON number opened as a valid file with no tensors in it; a header
      // holding `null` reached a bare TypeError naming neither the file nor the
      // problem.
      const parsed: unknown = JSON.parse(body.toString("utf8"));
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error(`${path}: the safetensors header is not an object`);
      }
      header = parsed as Record<string, unknown>;
    } finally {
      closeSync(fd);
    }
    const base = 8 + len;
    const tensors = new Map<string, TensorPlace>();
    let metadata: Record<string, unknown> | null = null;
    for (const [name, spec] of Object.entries(header)) {
      if (name === "__metadata__") {
        // Narrowed, not cast, and only its string values are kept: the
        // safetensors metadata map is string-to-string, which is what the
        // reference's library hands back and what Go, Rust and Swift each
        // keep. A `__metadata__` that is not an object, or a `manifest` key
        // holding an object rather than the JSON text of one, used to reach
        // `JSON.parse` as an object and throw a SyntaxError about a token,
        // where the other four report that there is no embedded manifest.
        metadata = {};
        const block = spec;
        if (typeof block === "object" && block !== null && !Array.isArray(block)) {
          for (const [k, v] of Object.entries(block as Record<string, unknown>)) {
            if (typeof v === "string") metadata[k] = v;
          }
        }
        continue;
      }
      const s = tensorSpec(path, name, spec);
      const [begin, end] = s.data_offsets;
      // Range-checked before anything reads it, because a tensor running off the
      // end of the file would otherwise come back as a short read, and the
      // readers below would produce fewer elements than the shape promised.
      if (!(begin >= 0 && begin <= end && base + end <= size)) {
        throw new Error(
          `${path}: tensor ${name} spans ${begin}..${end} of a ` +
            `${size - base}-byte payload, so the file is truncated or the ` +
            `header is corrupt`
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
            `(${elements * width} bytes) but occupies ${end - begin} bytes: ` +
            `the header does not describe the payload`
        );
      }
      tensors.set(name, {
        dtype: s.dtype as TensorSpec["dtype"],
        shape: s.shape,
        begin: base + begin,
        end: base + end,
      });
    }
    return { tensors, metadata };
  }

  /** The embedded `__metadata__` value, if any (e.g. a voice header). */
  metadata: Record<string, unknown> | null;

  has(name: string): boolean {
    return this.tensors.has(name);
  }

  /** Float32 tensor, read as a Float32Array of `shape`. */
  f32(name: string): Float32Array {
    const t = this.require(name);
    if (t.dtype === "F32") return copyFromBytes(this.bytes(name), Float32Array);
    if (t.dtype === "F16") return halfToFloat(this.bytes(name));
    // Both accepted dtypes named, as `go/safetensors` and
    // `rust/src/safetensors.rs` name them: a refusal that lists a narrower set
    // than the reader takes sends a reader looking for a conversion that is
    // already here.
    throw new Error(`${name}: expected F32/F16, got ${t.dtype}`);
  }

  /** Int64 tensor, read as a BigInt64Array of `shape`. */
  i64(name: string): BigInt64Array {
    const t = this.require(name);
    if (t.dtype !== "I64") throw new Error(`${name}: expected I64, got ${t.dtype}`);
    return copyFromBytes(this.bytes(name), BigInt64Array);
  }

  /** All tensor names. */
  keys(): string[] {
    return [...this.tensors.keys()];
  }

  /**
   * The shape one tensor declares.
   *
   * The typed accessors return a flat array, which is the right shape for
   * everything the engine reads. It is not enough for a *reader*: a profile
   * declaring `prompt_mel` as [2, 80] holds the same 160 floats as one
   * declaring [80, 2] and means the transpose of it, so a caller checking only
   * the element count reads that file as a voice whose mel is turned on its
   * side. `loadVoice` is the caller that needs this.
   */
  shape(name: string): number[] {
    return [...this.require(name).shape];
  }

  /** One tensor's payload, read now and held by its caller alone. */
  private bytes(name: string): Uint8Array {
    const t = this.require(name);
    const out = Buffer.alloc(t.end - t.begin);
    const fd = openSync(this.path, "r");
    try {
      readExactly(fd, out, t.begin, this.path, `tensor ${name}`);
    } finally {
      closeSync(fd);
    }
    return out;
  }

  private require(name: string): TensorPlace {
    const t = this.tensors.get(name);
    if (!t) throw new Error(`no tensor named ${name}; have ${this.tensors.size} tensors`);
    return t;
  }
}

/**
 * Fill `into` from `position`, looping because one `readSync` is allowed to
 * return fewer bytes than it was asked for, and does on large reads.
 */
function readExactly(
  fd: number,
  into: Buffer,
  position: number,
  path: string,
  what: string
): void {
  let done = 0;
  while (done < into.length) {
    const n = readSync(fd, into, done, into.length - done, position + done);
    if (n === 0) {
      throw new Error(`${path}: ${what} ends after ${done} of ${into.length} bytes`);
    }
    done += n;
  }
}

/** One tensor to write: its name, dtype, shape and little-endian bytes. */
export interface TensorEntry {
  name: string;
  dtype: keyof typeof BYTE_WIDTH;
  shape: number[];
  data: Uint8Array;
}

// The order the safetensors library lays tensors out in: widest dtype first,
// then by name. Only the order the reader inverts; the format accepts any.
const DTYPE_RANK: Record<string, number> = {
  BOOL: 0, U8: 1, I8: 2, F8_E5M2: 3, F8_E4M3: 4, I16: 5, U16: 6,
  F16: 7, BF16: 8, I32: 9, U32: 10, F32: 11, F64: 12, I64: 13, U64: 14,
};

/**
 * Write tensors and metadata as a safetensors file: the 8-byte header length,
 * the JSON header padded to a multiple of eight with spaces, then the
 * payloads in header order. Owner-only permissions.
 */
export function writeSafetensors(
  path: string,
  entries: TensorEntry[],
  metadata: Record<string, string>
): void {
  const sorted = [...entries].sort(
    (a, b) => DTYPE_RANK[b.dtype] - DTYPE_RANK[a.dtype] || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0)
  );
  const fields: string[] = [];
  if (Object.keys(metadata).length > 0) fields.push(`"__metadata__":${JSON.stringify(metadata)}`);
  let offset = 0;
  for (const e of sorted) {
    const elements = e.shape.reduce((n, d) => n * d, 1);
    if (elements * BYTE_WIDTH[e.dtype] !== e.data.length) {
      throw new Error(
        `${e.name}: shape [${e.shape.join(", ")}] of ${e.dtype} is ${elements * BYTE_WIDTH[e.dtype]} bytes, data is ${e.data.length}`
      );
    }
    fields.push(
      `${JSON.stringify(e.name)}:{"dtype":"${e.dtype}","shape":${JSON.stringify(e.shape)},"data_offsets":[${offset},${offset + e.data.length}]}`
    );
    offset += e.data.length;
  }
  let header = `{${fields.join(",")}}`;
  while (header.length % 8 !== 0) header += " ";
  const headerBytes = Buffer.from(header, "utf8");
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(headerBytes.length));
  writeFileSync(path, Buffer.concat([length, headerBytes, ...sorted.map((e) => Buffer.from(e.data))]), {
    mode: 0o600,
  });
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
