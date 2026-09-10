/**
 * Finding a release on disk.
 *
 * `Engine.load` derives the three paths it needs from the directory the
 * download wrote, and the interesting cases are the ones where a file is not
 * there: a half-fetched set has to fail here, naming the file, rather than six
 * graphs later inside onnxruntime.
 */

import assert from "node:assert";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { Engine } from "../engine.js";
import {
  CHECKPOINT_NAME,
  ENROLLMENT_NAME,
  TURBO_CHECKPOINT_NAME,
  VOICE_ENCODER_NAME,
  listVoices,
  resolveBundle,
  resolveCheckpoint,
  verifyReleaseInventory,
} from "../hub.js";

/** A safetensors file of no tensors whose header carries one manifest. */
function writeCheckpoint(path: string, manifest: Record<string, unknown>): void {
  const header = Buffer.from(
    JSON.stringify({ __metadata__: { manifest: JSON.stringify(manifest) } })
  );
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(path, Buffer.concat([length, header]));
}

const GRAPHS = [
  "t3_cond.onnx",
  "t3_prefill.onnx",
  "t3_step.onnx",
  "flow_encoder.onnx",
  "flow_estimator.onnx",
  "vocoder.onnx",
  "export.json",
];
const ENROLL_GRAPHS = ["s3_tokenizer.onnx", "camp.onnx", "voice_encoder.onnx"];

/** A release directory with nothing real in it: every file is one byte. */
function layout(options: { voices?: string[]; cloning?: boolean } = {}): string {
  const root = mkdtempSync(join(tmpdir(), "loudkit-bundle-"));
  mkdirSync(join(root, "onnx"));
  mkdirSync(join(root, "voices"));
  const names = [CHECKPOINT_NAME, "manifest.json", "tokenizer.json", "release.json"];
  for (const name of names) writeFileSync(join(root, name), "x");
  for (const name of GRAPHS) writeFileSync(join(root, "onnx", name), "x");
  if (options.cloning) {
    for (const name of ENROLL_GRAPHS) writeFileSync(join(root, "onnx", name), "x");
  }
  for (const name of options.voices ?? ["joe", "amy"]) {
    writeFileSync(join(root, "voices", `${name}.safetensors`), "x");
  }
  return root;
}

function withLayout(body: (root: string) => void, options?: Parameters<typeof layout>[0]): void {
  const root = layout(options);
  try {
    body(root);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("a complete layout resolves to three paths", () => {
  withLayout((root) => {
    const bundle = resolveBundle(root);
    assert.equal(bundle.checkpoint, join(root, CHECKPOINT_NAME));
    assert.equal(bundle.onnxDir, join(root, "onnx"));
    assert.equal(bundle.tokenizer, join(root, "tokenizer.json"));
  });
});

test("a missing graph is named, not discovered later", () => {
  withLayout((root) => {
    rmSync(join(root, "onnx", "vocoder.onnx"));
    assert.throws(() => resolveBundle(root), /onnx\/vocoder\.onnx/);
  });
});

test("what a load requires is the list Python requires, not a sample of it", () => {
  // Python's `verify_release_inventory` for the onnx backend, name for name.
  // Asserting the whole list rather than one absent graph is what keeps a
  // seventh name from creeping into the demand.
  const empty = mkdtempSync(join(tmpdir(), "loudkit-bundle-"));
  try {
    assert.throws(
      () => verifyReleaseInventory(empty, {}),
      (error: Error) =>
        error.message.includes(
          "missing: loudr-1.safetensors, onnx/flow_encoder.onnx, " +
            "onnx/flow_estimator.onnx, onnx/t3_cond.onnx, onnx/t3_prefill.onnx, " +
            "onnx/t3_step.onnx, onnx/vocoder.onnx, tokenizer.json."
        )
    );
  } finally {
    rmSync(empty, { recursive: true, force: true });
  }
});

test("manifest.json is the fetch's business, not the loader's", () => {
  // The engine reads the manifest the checkpoint embeds, so a hand-assembled
  // directory without the mirror loads here as it does in the other four.
  withLayout((root) => {
    rmSync(join(root, "manifest.json"));
    assert.doesNotThrow(() => resolveBundle(root));
    assert.throws(() => verifyReleaseInventory(root, { requireManifest: true }), /manifest\.json/);
  });
});

test("the export record is fetched with the graphs and not demanded of them", () => {
  // Python's rule and its reason: the runtime warns about a set with no
  // record rather than refusing it, so a set exported before the record
  // existed keeps loading. An inventory that demanded one would refuse exactly
  // those releases at the download, which is the harder failure.
  withLayout((root) => {
    rmSync(join(root, "onnx", "export.json"));
    assert.doesNotThrow(() => resolveBundle(root));
  });
});

test("a load does not demand voices, a fetch does", () => {
  withLayout(
    (root) => {
      assert.doesNotThrow(() => resolveBundle(root));
      assert.throws(
        () => verifyReleaseInventory(root, { requireVoices: true }),
        /voices\/\*\.safetensors/
      );
    },
    { voices: [] }
  );
});

test("cloning asks for the three enrollment graphs and nothing else", () => {
  withLayout((root) => {
    assert.doesNotThrow(() => verifyReleaseInventory(root, {}));
    assert.throws(() => verifyReleaseInventory(root, { cloning: true }), /onnx\/camp\.onnx/);
  });
});

test("the enrollment artefact and the voice encoder are not checkpoints", () => {
  withLayout((root) => {
    rmSync(join(root, CHECKPOINT_NAME));
    writeFileSync(join(root, ENROLLMENT_NAME), "x");
    writeFileSync(join(root, VOICE_ENCODER_NAME), "x");
    assert.throws(() => resolveCheckpoint(root), /no other checkpoint/);
  });
});

test("one checkpoint under another name still resolves", () => {
  withLayout((root) => {
    rmSync(join(root, CHECKPOINT_NAME));
    writeFileSync(join(root, "loudr-1-turbo.safetensors"), "x");
    assert.equal(resolveCheckpoint(root), join(root, "loudr-1-turbo.safetensors"));
  });
});

test("two unnamed checkpoints are an ambiguity that names both", () => {
  withLayout((root) => {
    rmSync(join(root, CHECKPOINT_NAME));
    writeFileSync(join(root, "a.safetensors"), "x");
    writeFileSync(join(root, "b.safetensors"), "x");
    assert.throws(() => resolveCheckpoint(root), /a\.safetensors, b\.safetensors/);
  });
});

test("a directory holding both models names them rather than guessing", () => {
  // They are different releases with different decode loops, so returning the
  // one whose name is checked first is returning a model nobody asked for.
  withLayout((root) => {
    writeFileSync(join(root, TURBO_CHECKPOINT_NAME), "x");
    assert.throws(() => resolveCheckpoint(root), /2 models here/);
    assert.throws(() => resolveCheckpoint(root), /no right one to pick/);
  });
});

test("a checkpoint that declares the enrollment role is refused by name", () => {
  // The manifest's claim outranks the file name: a directory carrying only
  // enrolment weights under the synthesis name is refused rather than opened.
  withLayout((root) => {
    writeCheckpoint(join(root, CHECKPOINT_NAME), { artifact_role: "enrollment" });
    assert.throws(() => resolveCheckpoint(root), /enrollment artefact/);
  });
});

test("the declared synthesis half wins over an unnamed neighbour", () => {
  withLayout((root) => {
    rmSync(join(root, CHECKPOINT_NAME));
    writeCheckpoint(join(root, "a.safetensors"), { artifact_role: "synthesis" });
    writeCheckpoint(join(root, "b.safetensors"), { artifact_role: "enrollment" });
    assert.equal(resolveCheckpoint(root), join(root, "a.safetensors"));
  });
});

test("a dot-prefixed file is not a checkpoint candidate", () => {
  // A release build cannot produce one, so the reference, Go, Rust and Swift
  // all skip it. Counting it turned a directory with one model and one editor
  // or sync leftover beside it into "name the one you mean".
  withLayout((root) => {
    rmSync(join(root, CHECKPOINT_NAME));
    writeFileSync(join(root, "model.safetensors"), "x");
    writeFileSync(join(root, ".model.safetensors"), "x");
    assert.equal(resolveCheckpoint(root), join(root, "model.safetensors"));
  });
});

test("Engine.load reads a directory through the same resolver", async () => {
  // One argument takes the bundle path; three keep the explicit form. What is
  // asserted is the routing, not the engine: a directory short of a graph has
  // to fail with that file's name rather than inside onnxruntime.
  const root = layout();
  try {
    rmSync(join(root, "onnx", "t3_step.onnx"));
    await assert.rejects(Engine.load(root), /onnx\/t3_step\.onnx/);
    await assert.rejects(Engine.load(join(root, "nowhere")), /no such directory/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("voices are listed by name, sorted, without their extension", () => {
  withLayout(
    (root) => {
      assert.deepEqual(listVoices(root), ["amy", "joe", "zoe"]);
      assert.deepEqual(listVoices(join(root, "nowhere")), []);
    },
    { voices: ["zoe", "joe", "amy"] }
  );
});
