import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import { checkExportRecord } from "../hub.js";

const GRAPHS = [
  "t3_cond.onnx",
  "t3_prefill.onnx",
  "t3_step.onnx",
  "flow_encoder.onnx",
  "flow_estimator.onnx",
  "vocoder.onnx",
];

const FINGERPRINT = "ae115ff943456910";

interface Fixture {
  dir: string;
  checkpoint: string;
  digest: string;
}

/** A scratch set with a stand-in for the checkpoint, whose digest a record names. */
function scratch(body: (fixture: Fixture) => Promise<void> | void): Promise<void> | void {
  const dir = mkdtempSync(join(tmpdir(), "loudkit-export-"));
  const checkpoint = join(dir, "ckpt.safetensors");
  writeFileSync(checkpoint, "ckpt");
  const digest = createHash("sha256").update("ckpt").digest("hex");
  const done = () => rmSync(dir, { recursive: true, force: true });
  try {
    const out = body({ dir, checkpoint, digest });
    return out instanceof Promise ? out.finally(done) : (done(), undefined);
  } catch (err) {
    done();
    throw err;
  }
}

/** The fields a record has to carry to match the fixture's checkpoint. */
function matching(digest: string): Record<string, unknown> {
  return {
    checkpoint_sha256: digest,
    algorithm_fingerprint: FINGERPRINT,
    euler_steps: 2,
  };
}

/** Write a record whose members all carry `entry`, and `odd` for the vocoder. */
function writeRecord(dir: string, entry: unknown, odd?: unknown): void {
  const graphs: Record<string, unknown> = {};
  for (const name of GRAPHS) graphs[name] = entry;
  if (odd !== undefined) graphs["vocoder.onnx"] = odd;
  writeFileSync(join(dir, "export.json"), JSON.stringify({ format: "loudkit-onnx-export", graphs }));
}

function check(fixture: Fixture, manifest: Record<string, unknown> = {}): Promise<void> {
  return checkExportRecord(fixture.dir, fixture.checkpoint, manifest, FINGERPRINT, 2, GRAPHS);
}

async function refusal(fixture: Fixture, manifest: Record<string, unknown> = {}): Promise<string> {
  try {
    await check(fixture, manifest);
  } catch (err) {
    return err instanceof Error ? err.message : String(err);
  }
  throw new Error("expected a refusal, got none");
}

test("a record naming this engine loads", async () => {
  await scratch(async (fixture) => {
    writeRecord(fixture.dir, matching(fixture.digest));
    await check(fixture);
  });
});

test("a set exported before the record still loads", async () => {
  await scratch(async (fixture) => {
    // The rule and its reason: refusing here would strand exactly the releases
    // that predate the record, which is the harder failure.
    const warnings: string[] = [];
    const listener = (warning: Error) => warnings.push(warning.name);
    process.on("warning", listener);
    try {
      await check(fixture);
      // `emitWarning` delivers on the next tick, so the listener has to outlive
      // the call that armed it.
      await new Promise((resolve) => setImmediate(resolve));
    } finally {
      process.off("warning", listener);
    }
    assert.deepEqual(warnings, ["LoudkitExportRecordWarning"]);
  });
});

test("a record naming another engine is refused", async () => {
  await scratch(async (fixture) => {
    writeRecord(fixture.dir, {
      ...matching(fixture.digest),
      algorithm_fingerprint: "5cfefec451bcedd1",
    });
    const message = await refusal(fixture);
    assert.match(message, /was exported from a different engine than the one loading it/);
    assert.match(message, /'algorithm_fingerprint': '5cfefec451bcedd1'/);
    assert.match(message, /'algorithm_fingerprint': 'ae115ff943456910'/);
  });
});

test("a record naming another checkpoint is refused", async () => {
  await scratch(async (fixture) => {
    writeRecord(fixture.dir, { ...matching(fixture.digest), checkpoint_sha256: "0".repeat(64) });
    assert.match(await refusal(fixture), /was exported from a different engine/);
  });
});

test("one stage re-exported on its own is a mixed set", async () => {
  await scratch(async (fixture) => {
    writeRecord(fixture.dir, matching(fixture.digest), {
      ...matching(fixture.digest),
      euler_steps: 7,
    });
    const message = await refusal(fixture);
    assert.match(message, /is a mixed graph set/);
    assert.match(message, /vocoder\.onnx: \{/);
  });
});

test("a member the record does not name is refused", async () => {
  await scratch(async (fixture) => {
    writeFileSync(
      join(fixture.dir, "export.json"),
      JSON.stringify({ format: "loudkit-onnx-export", graphs: {} })
    );
    assert.match(await refusal(fixture), /does not record t3_cond\.onnx/);
  });
});

test("a record that is not one names itself", async () => {
  await scratch(async (fixture) => {
    for (const [body, want] of [
      ['{"graphs": [1, 2]}', /is not a mapping of name to its export record/],
      ['{"graphs": {"t3_cond.onnx": {"euler_steps": [2]}}}', /is not a mapping/],
      ['{"format": "loudkit-onnx-export"}', /unreadable export record/],
      ["{", /unreadable export record/],
    ] as const) {
      writeFileSync(join(fixture.dir, "export.json"), body);
      assert.match(await refusal(fixture), want, body);
    }
  });
});

test("an absent field and a null one compare equal", async () => {
  await scratch(async (fixture) => {
    writeRecord(fixture.dir, matching(fixture.digest), {
      ...matching(fixture.digest),
      estimator_sha256: null,
    });
    await check(fixture);
  });
});

test("a renderer traced from another estimator is refused", async () => {
  await scratch(async (fixture) => {
    const manifest = {
      sources: { "flow.pt": { role: "estimator", sha256: "a".repeat(64) } },
    };
    writeRecord(fixture.dir, {
      ...matching(fixture.digest),
      estimator_sha256: "b".repeat(64),
    });
    assert.match(
      await refusal(fixture, manifest),
      /traced with an estimator the checkpoint was not packed from/
    );
    writeRecord(fixture.dir, {
      ...matching(fixture.digest),
      estimator_sha256: "a".repeat(64),
    });
    await check(fixture, manifest);
  });
});
