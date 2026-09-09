/**
 * The tensor header, read rather than asserted.
 *
 * These files arrive by download, so a malformed one is the ordinary case, not
 * the exotic one, and the message a caller gets is the whole of what they have
 * to work with. `spec as { dtype, shape, data_offsets }` was a claim about JSON
 * somebody else wrote: seven of the fifteen malformed specs below surfaced as
 * `TypeError: undefined is not iterable`, naming neither the file nor the
 * tensor, and one loaded with no error at all. The reference's reader refuses
 * all fifteen and names the field it could not read.
 */

import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { SafetensorsFile } from "../safetensors.js";

/** A safetensors file whose header is exactly `header`, over 16 payload bytes. */
function withHeader(dir: string, name: string, header: unknown): string {
  const head = Buffer.from(JSON.stringify(header), "utf8");
  const len = Buffer.alloc(8);
  len.writeBigUInt64LE(BigInt(head.length));
  const path = join(dir, `${name}.safetensors`);
  writeFileSync(path, Buffer.concat([len, head, Buffer.alloc(16)]));
  return path;
}

test("a malformed tensor entry names the file and the tensor", () => {
  const dir = mkdtempSync(join(tmpdir(), "loudkit-st-"));
  try {
    const cases: Array<[string, unknown, RegExp]> = [
      ["no_offsets", { dtype: "F32", shape: [2] }, /tensor t declares data_offsets as undefined/],
      ["no_shape", { dtype: "F32", data_offsets: [0, 8] }, /tensor t has no shape/],
      ["no_dtype", { shape: [2], data_offsets: [0, 8] }, /tensor t has no dtype/],
      ["spec_number", 5, /tensor t is number, not a tensor entry/],
      ["spec_null", null, /tensor t is null, not a tensor entry/],
      ["spec_list", [1, 2], /tensor t is a list, not a tensor entry/],
      ["spec_string", "F32", /tensor t is string, not a tensor entry/],
      ["off_scalar", { dtype: "F32", shape: [2], data_offsets: 5 }, /declares data_offsets as number/],
      ["off_one", { dtype: "F32", shape: [2], data_offsets: [0] }, /declares 1 data_offsets/],
      // This one loaded, silently keeping the first two of three.
      ["off_three", { dtype: "F32", shape: [2], data_offsets: [0, 8, 9] }, /declares 3 data_offsets/],
      ["off_strings", { dtype: "F32", shape: [2], data_offsets: ["0", "8"] }, /non-numeric data_offsets/],
      ["shape_scalar", { dtype: "F32", shape: 2, data_offsets: [0, 8] }, /tensor t has no shape/],
      ["shape_string", { dtype: "F32", shape: ["2"], data_offsets: [0, 8] }, /bad dimension/],
      ["shape_null", { dtype: "F32", shape: [null], data_offsets: [0, 8] }, /bad dimension/],
      ["dtype_number", { dtype: 5, shape: [2], data_offsets: [0, 8] }, /tensor t has no dtype/],
    ];
    for (const [name, spec, wanted] of cases) {
      assert.throws(() => new SafetensorsFile(withHeader(dir, name, { t: spec })), wanted, name);
      // Whatever else it says, it says which file.
      assert.throws(
        () => new SafetensorsFile(withHeader(dir, name, { t: spec })),
        new RegExp(`${name}\\.safetensors`),
        `${name} names the file`
      );
    }
    // A well-formed entry still loads.
    const ok = withHeader(dir, "ok", { t: { dtype: "F32", shape: [2], data_offsets: [0, 8] } });
    assert.deepEqual(new SafetensorsFile(ok).keys(), ["t"]);
    assert.deepEqual(new SafetensorsFile(ok).shape("t"), [2]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
