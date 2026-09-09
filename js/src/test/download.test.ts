/**
 * The fetch plan, and the fetch, against a stub hub.
 *
 * `planDownload` is checked against the shared fixture Python writes; the
 * fetch against a local `http.createServer` that answers the two endpoints the
 * real hub does. Nothing here touches the network or downloads a model.
 */

import assert from "node:assert";
import { createHash } from "node:crypto";
import { createServer, type Server } from "node:http";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";

import { fileURLToPath } from "node:url";

import {
  cacheDir,
  cachePath,
  canEnroll,
  download,
  ensureCloning,
  isRepoId,
  matches,
  parseSums,
  planDownload,
  readReceipt,
  receiptHit,
  releasePatterns,
  resolveBundle,
  RECEIPT_NAME,
} from "../hub.js";

const REPO = "loudreader/loudr-1";

/** One of the shared fixtures, found by walking up to the repo root. */

function readFixture(name: string): any {
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

/**
 * A release as the hub would list it: everything `tools/build_release.py`
 * ships, in miniature, with a two-voice roster.
 */
function stubRelease(): Map<string, string> {
  const files = new Map<string, string>([
    ["loudr-1.safetensors", "checkpoint bytes"],
    ["manifest.json", '{"recipe_version": "loudkit-1"}'],
    ["tokenizer.json", '{"model": {}}'],
    ["release.json", '{"profile": "full-0.1", "verified": true}'],
    ["voices/joe.safetensors", "joe"],
    ["voices/amy.safetensors", "amy"],
    // The documents, the branding and the samples a release carries. None of
    // them is something this port opens, so none is fetched.
    ["README.md", "# the model card"],
    ["LICENSE", "Apache-2.0"],
    ["NOTICE", "attribution"],
    ["RESPONSIBLE_USE.md", "terms"],
    ["logo.png", "png bytes"],
    ["samples/joe.opus", "opus bytes"],
    [".gitattributes", "*.safetensors filter=lfs"],
    // Torch's two. A synthesis fetch must not move them.
    ["ve.safetensors", "voice encoder"],
    ["loudr-1-enrollment.safetensors", "enrollment tensors"],
    // Another backend's graphs.
    ["coreml/vocoder.mlpackage/Manifest.json", "{}"],
    ["coreml/vocoder.mlpackage/Data/com.apple.CoreML/model.mlmodel", "coreml bytes"],
    ["coreml/export.json", '{"checkpoint_sha256": "0"}'],
  ]);
  for (const name of [
    "t3_cond",
    "t3_prefill",
    "t3_step",
    "flow_encoder",
    "flow_estimator",
    "vocoder",
    "s3_tokenizer",
    "camp",
    "voice_encoder",
  ]) {
    files.set(`onnx/${name}.onnx`, `${name} graph`);
  }
  files.set("onnx/export.json", '{"checkpoint_sha256": "0"}');
  return files;
}

function sums(files: Map<string, string>, corrupt?: string): string {
  const lines: string[] = [];
  for (const [path, body] of files) {
    if (path === "SHA256SUMS") continue;
    const digest =
      path === corrupt ? "0".repeat(64) : createHash("sha256").update(body).digest("hex");
    lines.push(`${digest}  ${path}`);
  }
  return `${lines.join("\n")}\n`;
}

const COMMIT_A = "3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e";
const COMMIT_B = "b1d2f3a4c5e6b7d8f9a0c1e2b3d4f5a6c7e8b9d0";

interface Hub {
  endpoint: string;
  asked: string[];
  ranged: string[];
  /** What the revision route answers; a test moves it to move the revision. */
  commit: string;
  close: () => Promise<void>;
}

async function hub(files: Map<string, string>): Promise<Hub> {
  const asked: string[] = [];
  const ranged: string[] = [];
  const state = { commit: COMMIT_A };
  const server: Server = createServer((request, response) => {
    const url = request.url ?? "";
    asked.push(url);
    if (request.headers.range) ranged.push(`${url} ${request.headers.range}`);
    if (url.startsWith(`/api/models/`) && url.includes("/revision/")) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ sha: state.commit, siblings: [] }));
      return;
    }
    if (url.startsWith(`/api/models/`)) {
      const tree = [...files].map(([path, body]) => ({
        type: "file",
        path,
        size: Buffer.byteLength(body),
      }));
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify(tree));
      return;
    }
    const at = url.indexOf("/resolve/main/");
    const path = at >= 0 ? decodeURIComponent(url.slice(at + "/resolve/main/".length)) : "";
    const body = files.get(path);
    if (body === undefined) {
      response.writeHead(404).end();
      return;
    }
    const range = /^bytes=(\d+)-$/.exec(request.headers.range ?? "");
    if (range) {
      const from = Number(range[1]);
      response.writeHead(206, {
        "content-range": `bytes ${from}-${body.length - 1}/${body.length}`,
      });
      response.end(body.slice(from));
      return;
    }
    response.writeHead(200).end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no port");
  return {
    endpoint: `http://127.0.0.1:${address.port}`,
    asked,
    ranged,
    get commit() {
      return state.commit;
    },
    set commit(value: string) {
      state.commit = value;
    },
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

/** The files the stub served, by name, in the order they were asked for. */
function served(stub: Hub): string[] {
  return stub.asked
    .filter((u) => u.includes("/resolve/main/"))
    .map((u) => decodeURIComponent(u.slice(u.indexOf("/resolve/main/") + "/resolve/main/".length)));
}

function workdir(): string {
  return mkdtempSync(join(tmpdir(), "loudkit-download-"));
}

const quiet = { onProgress: () => {} };

test("the plan is the shared fixture's, pattern for pattern and file for file", () => {
  const plan = readFixture("release_plan.json");
  const cases = plan.cases.filter((c: any) => c.backend === "onnx");
  assert.equal(cases.length, 2, "the fixture holds two onnx cases");
  for (const c of cases) {
    const { allow, ignore } = releasePatterns(c.cloning);
    // The allow set as well as the ignore set. Only `ignore` and `wanted` were
    // compared, and the fixture's listing carries neither fusion graph, so the
    // set this port would fetch drifted from the one the other three fetch
    // with nothing to catch it: both loops are allowed, and a repo carrying
    // both was fetched here without `onnx/t3_step.onnx`.
    assert.deepEqual([...allow].sort(), [...c.allow].sort(), `cloning=${c.cloning} allow`);
    assert.deepEqual(ignore, c.ignore, `cloning=${c.cloning} ignore`);
    assert.deepEqual(planDownload(plan.listing, c.cloning), c.wanted, `cloning=${c.cloning}`);
  }
  // A listing carrying both decode loops brings back both, as the reference,
  // Go and Rust do; the mode is what `requireOnnxSet` and the engine read.
  const both = [...plan.listing, "onnx/t3_pair_step.onnx", "onnx/t3_head2.onnx"];
  const wanted = planDownload(both, false);
  for (const graph of ["onnx/t3_step.onnx", "onnx/t3_pair_step.onnx", "onnx/t3_head2.onnx"]) {
    assert.ok(wanted.includes(graph), graph);
  }
});

test("the glob is fnmatch: `*` crosses a slash, which is how voices arrive", () => {
  const cases = readFixture("glob.json").cases;
  assert.ok(cases.length >= 10, "the fixture holds probes");
  for (const c of cases) {
    assert.equal(matches(c.name, c.pattern), c.match, `${c.pattern} ~ ${c.name}`);
  }
});

test("a repo id is what the shared fixture says it is", () => {
  const cases = readFixture("repo_id.json").cases;
  assert.ok(cases.length >= 10, "the fixture holds probes");
  for (const c of cases) {
    assert.equal(isRepoId(c.ref), c.is_repo_id, c.ref);
  }
  // A path that exists is a path, however it is spelled.
  const dir = workdir();
  try {
    mkdirSync(join(dir, "org", "name"), { recursive: true });
    assert.equal(isRepoId(join(dir, "org", "name")), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a listing that keeps pointing at itself is refused, not followed", async () => {
  // The tree API pages by the thousand and a release is tens of files, so a
  // `Link: rel="next"` that never ends is a cycle rather than a large repo.
  // Uncapped, this call never returns.
  let pages = 0;
  let endpoint = "";
  const server: Server = createServer((request, response) => {
    const url = request.url ?? "";
    if (url.includes("/revision/")) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ sha: COMMIT_A, siblings: [] }));
      return;
    }
    pages += 1;
    response.writeHead(200, {
      "content-type": "application/json",
      link: `<${endpoint}${url}>; rel="next"`,
    });
    response.end("[]");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no port");
  endpoint = `http://127.0.0.1:${address.port}`;
  const dir = workdir();
  try {
    await assert.rejects(download(REPO, dir, { endpoint }), /the file listing does not end/);
    assert.ok(pages > 1 && pages <= 200, `followed ${pages} pages`);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("a fetch writes the plan and nothing else, the bookkeeping first", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    const lines: string[] = [];
    const got = await download(REPO, dir, { endpoint: stub.endpoint, onProgress: (m) => lines.push(m) });
    assert.equal(got, dir);
    assert.equal(readFileSync(join(dir, "loudr-1.safetensors"), "utf8"), "checkpoint bytes");
    assert.equal(readFileSync(join(dir, "onnx", "vocoder.onnx"), "utf8"), "vocoder graph");
    assert.equal(readFileSync(join(dir, "voices", "joe.safetensors"), "utf8"), "joe");
    for (const refused of ["ve.safetensors", "loudr-1-enrollment.safetensors", "README.md"]) {
      assert.throws(() => readFileSync(join(dir, refused)), /ENOENT/, refused);
    }
    // A repo that is not a release is refused before its weights move, so
    // the two files that say what it is come first.
    assert.deepEqual(served(stub).slice(0, 2), ["SHA256SUMS", "release.json"]);
    assert.ok(lines.some((l) => l.includes("verified")));
    assert.ok(lines.at(-1)?.includes(dir));
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a second fetch costs one revision lookup, no file and no hash", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    const after = stub.asked.length;
    const lines: string[] = [];
    await download(REPO, dir, { endpoint: stub.endpoint, onProgress: (m) => lines.push(m) });
    assert.equal(stub.asked.length - after, 1, stub.asked.slice(after).join(", "));
    assert.ok(stub.asked.at(-1)?.includes("/revision/"), "the one request resolves the revision");
    assert.ok(!lines.some((l) => l.includes("verified")), "a second fetch re-hashed the release");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("the receipt is the shared fixture, in shape and in rule", async () => {
  const fixture = readFixture("release_receipt.json");
  assert.equal(fixture.name, RECEIPT_NAME);
  assert.ok(fixture.cases.length >= 20, "the fixture holds cases");
  for (const c of fixture.cases) {
    const dir = workdir();
    try {
      if (c.receipt !== null) writeFileSync(join(dir, RECEIPT_NAME), JSON.stringify(c.receipt));
      if (c.sums !== null) writeFileSync(join(dir, "SHA256SUMS"), c.sums);
      for (const name of c.files) writeFileSync(join(dir, name), name);
      assert.equal((await readReceipt(dir, c.repo)) !== null, c.offline === "use", c.name);
      assert.equal(await receiptHit(dir, c.repo, c.commit), c.online === "hit", c.name);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
  // What this port writes has the fixture's keys, in the fixture's order.
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    const written = JSON.parse(readFileSync(join(dir, RECEIPT_NAME), "utf8"));
    assert.deepEqual(Object.keys(written), fixture.fields);
    const receipt = await readReceipt(dir, REPO);
    assert.ok(receipt);
    assert.equal(receipt.repo, REPO);
    assert.equal(receipt.revision, "main");
    assert.equal(receipt.commit, COMMIT_A);
    assert.equal(
      receipt.sha256sums,
      createHash("sha256").update(readFileSync(join(dir, "SHA256SUMS"))).digest("hex")
    );
    assert.match(receipt.fetched_at, /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a moved revision is fetched again, and what still matches is kept", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    const before = served(stub).length;
    stub.commit = COMMIT_B;
    const lines: string[] = [];
    await download(REPO, dir, { endpoint: stub.endpoint, onProgress: (m) => lines.push(m) });
    assert.deepEqual(served(stub).slice(before), ["SHA256SUMS"]);
    assert.ok(lines.some((l) => l.includes("verified")), "a moved revision was not re-verified");
    assert.equal((await readReceipt(dir, REPO))?.commit, COMMIT_B);

    // A file edited in place at its published length is caught on the next
    // miss, and it alone is fetched again.
    const graph = join(dir, "onnx", "vocoder.onnx");
    writeFileSync(graph, "vocoder hparg");
    const again = served(stub).length;
    stub.commit = COMMIT_A;
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    assert.deepEqual(served(stub).slice(again), ["SHA256SUMS", "onnx/vocoder.onnx"]);
    assert.equal(readFileSync(graph, "utf8"), "vocoder graph");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a listed file deleted under a receipt is fetched again; one edited in place is not", async () => {
  // A receipt vouches for every file the plan selects from SHA256SUMS, not
  // only the files the port cannot run without: a voice is one of many and
  // its absence is a miss. An edit in place rides on a matching receipt,
  // because a hit hashes nothing.
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    const amy = join(dir, "voices", "amy.safetensors");
    rmSync(amy);
    const before = served(stub).length;
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    assert.deepEqual(served(stub).slice(before), ["SHA256SUMS", "voices/amy.safetensors"]);
    assert.equal(readFileSync(amy, "utf8"), "amy");
    writeFileSync(amy, "yma");
    const again = served(stub).length;
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    assert.deepEqual(served(stub).slice(again), []);
    assert.equal(readFileSync(amy, "utf8"), "yma");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("offline with a receipt uses it and says so; without one there is nothing to use", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  const empty = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    // A port nobody listens on: the transport fails, the hub never answers.
    const lines: string[] = [];
    const got = await download(REPO, dir, {
      endpoint: "http://127.0.0.1:9",
      onProgress: (m) => lines.push(m),
    });
    assert.equal(got, dir);
    assert.ok(lines.some((l) => l.includes("cannot be reached") && l.includes(COMMIT_A)), lines.join("\n"));
    await assert.rejects(
      download(REPO, empty, { endpoint: "http://127.0.0.1:9", ...quiet }),
      /cannot be reached/
    );
    // A directory verified as one repo does not answer for another, and a
    // receipt no download wrote, one field forged, is no receipt.
    const receipt = await readReceipt(dir, REPO);
    assert.ok(receipt);
    const noCommit: Partial<typeof receipt> = { ...receipt };
    delete noCommit.commit;
    for (const forged of [
      { ...receipt, repo: "someone/other" },
      noCommit,
      { ...receipt, commit: "" },
      { ...receipt, sha256sums: "0".repeat(64) },
    ]) {
      writeFileSync(join(dir, RECEIPT_NAME), JSON.stringify(forged));
      await assert.rejects(
        download(REPO, dir, { endpoint: "http://127.0.0.1:9", ...quiet }),
        /cannot be reached/,
        JSON.stringify(forged)
      );
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(empty, { recursive: true, force: true });
    await stub.close();
  }
});

test("a stale receipt does not outlive a failed fetch", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    stub.commit = COMMIT_B;
    files.set("onnx/vocoder.onnx", "not the graph"); // signed as something else
    writeFileSync(join(dir, "onnx", "vocoder.onnx"), "xxx"); // and not kept either
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /failed the release checksum/
    );
    assert.ok(!existsSync(join(dir, RECEIPT_NAME)), "the receipt survived a failed fetch");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a half-written file is resumed rather than restarted", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    const target = join(dir, "loudr-1.safetensors");
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(`${target}.part`, "checkpoint");
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    assert.deepEqual(stub.ranged, ["/loudreader/loudr-1/resolve/main/loudr-1.safetensors bytes=10-"]);
    assert.equal(readFileSync(target, "utf8"), "checkpoint bytes");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a file that does not match SHA256SUMS is removed and reported", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files, "onnx/vocoder.onnx"));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /onnx\/vocoder\.onnx failed the release checksum/
    );
    assert.throws(() => readFileSync(join(dir, "onnx", "vocoder.onnx")), /ENOENT/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("an official release without SHA256SUMS is refused before any file moves", async () => {
  const stub = await hub(stubRelease());
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /Every loudreader release ships one/
    );
    assert.deepEqual(served(stub), []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a development bundle in an official repo is refused before its weights move", async () => {
  const files = stubRelease();
  files.set("release.json", '{"profile": "lenient", "verified": true}');
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /development bundle, not the release/
    );
    assert.deepEqual(served(stub), ["SHA256SUMS", "release.json"]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a bundle that never passed the builder's gate is refused", async () => {
  const files = stubRelease();
  files.set("release.json", '{"profile": "full-0.1"}');
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /does not record verified: true/
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("the turbo profile is a release too", async () => {
  // One gate for both models; what keeps turbo out of this port is the
  // graphs check, which this listing passes.
  const files = stubRelease();
  files.set("release.json", '{"profile": "turbo-0.1", "verified": true}');
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a release without synthesis graphs is refused before weights move", async () => {
  const files = new Map<string, string>([
    ["loudr-1-turbo.safetensors", "turbo bytes"],
    ["manifest.json", "{}"],
    ["tokenizer.json", "{}"],
    ["release.json", '{"profile": "turbo-0.1", "verified": true}'],
    ["voices/joe.safetensors", "joe"],
  ]);
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download("loudreader/loudr-1-turbo", dir, { endpoint: stub.endpoint, ...quiet }),
      /ships no onnx\/t3_cond.onnx/
    );
    assert.deepEqual(served(stub), []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a voice the manifest does not list is weights nothing vouches for", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  files.set("voices/mallory.safetensors", "not in the manifest");
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /does not list voices\/mallory\.safetensors/
    );
    assert.ok(!existsSync(join(dir, "voices", "mallory.safetensors")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a listing cannot name a file outside the directory", async () => {
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  files.set("../escaped.safetensors", "no");
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /nothing was written/
    );
    assert.deepEqual(served(stub), []);
    assert.ok(!existsSync(join(dirname(dir), "escaped.safetensors")));
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a mangled manifest is refused, not skipped", () => {
  const dir = workdir();
  try {
    const good = "a".repeat(64);
    const parse = (text: string) => {
      writeFileSync(join(dir, "SHA256SUMS"), text);
      return parseSums(join(dir, "SHA256SUMS"));
    };
    assert.deepEqual([...parse(`${good}  voices/joe.safetensors\n`)], [["voices/joe.safetensors", good]]);
    assert.throws(() => parse(`${good} voices/joe.safetensors\n`), /malformed/);
    assert.throws(() => parse(`abc  voices/joe.safetensors\n`), /malformed/);
    assert.throws(() => parse(`${good}  ../joe.safetensors\n`), /escapes/);
    assert.throws(() => parse(`${good}  /etc/passwd\n`), /absolute/);
    assert.throws(() => parse(`${good}  a\\b\n`), /backslash/);
    assert.throws(() => parse(`${good}  joe\n${good}  joe\n`), /duplicate/);
    assert.throws(() => parse("\n\n"), /no checksum entries/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a release short of a graph is refused before a byte moves", async () => {
  const files = stubRelease();
  files.delete("onnx/flow_estimator.onnx");
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /onnx\/flow_estimator\.onnx/
    );
    assert.deepEqual(served(stub), []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("something that is not `org/name` is refused before any request", async () => {
  await assert.rejects(
    download("unknown-model", join(tmpdir(), "never-written")),
    /not a Hugging Face repo id/
  );
});

test("enroll on a repo engine fetches the cloning set once; a directory engine is told how", async () => {
  // The three enrollment graphs land in the directory the engine was loaded
  // from, with the manifest that vouches for them, and the receipt widens to
  // the cloning set so the second enrollment fetches nothing.
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  const plain = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    assert.equal(canEnroll(dir), false);
    const before = served(stub).length;
    await ensureCloning(dir, REPO, { endpoint: stub.endpoint, ...quiet });
    assert.deepEqual(served(stub).slice(before).sort(), [
      "SHA256SUMS",
      "onnx/camp.onnx",
      "onnx/s3_tokenizer.onnx",
      "onnx/voice_encoder.onnx",
    ]);
    assert.equal(canEnroll(dir), true);
    assert.ok(await readReceipt(dir, REPO, true), "the receipt was not widened");
    const again = served(stub).length;
    await ensureCloning(dir, REPO, { endpoint: stub.endpoint, ...quiet });
    assert.deepEqual(served(stub).slice(again), []);

    await download(REPO, plain, { endpoint: stub.endpoint, ...quiet });
    const untouched = served(stub).length;
    await assert.rejects(ensureCloning(plain, undefined), /download\(repo, dir, \{ cloning: true \}\)/);
    assert.deepEqual(served(stub).slice(untouched), []);
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(plain, { recursive: true, force: true });
    await stub.close();
  }
});

/** A safetensors file of no tensors whose header carries one manifest. */
function writeCheckpoint(path: string, manifest: Record<string, unknown>): void {
  const header = Buffer.from(JSON.stringify({ __metadata__: { manifest: JSON.stringify(manifest) } }));
  const length = Buffer.alloc(8);
  length.writeBigUInt64LE(BigInt(header.length));
  writeFileSync(path, Buffer.concat([length, header]));
}

test("the manifest selects paired graph requirements regardless of filename", () => {
  const dir = workdir();
  try {
    writeCheckpoint(join(dir, "renamed.safetensors"), { decode: { mode: "fusion_mtp2" } });
    assert.throws(() => resolveBundle(dir), /t3_pair_step.onnx/);
    assert.throws(() => resolveBundle(dir), /t3_head2.onnx/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the cache path is the shared fixture", () => {
  const fixture = readFixture("cache_path.json");
  assert.equal(fixture.env, "LOUDKIT_CACHE");
  assert.ok(fixture.cases.length >= 4, "the fixture holds probes");
  const slashes = (p: string) => p.replace(/\\/g, "/");
  for (const c of fixture.cases) {
    assert.equal(slashes(cachePath(c.root, c.repo)), c.path, `${c.root} ${c.repo}`);
  }
  const saved = process.env.LOUDKIT_CACHE;
  try {
    for (const o of fixture.override) {
      process.env.LOUDKIT_CACHE = o.cache;
      assert.equal(slashes(cacheDir(o.repo)), o.path, o.repo);
    }
    delete process.env.LOUDKIT_CACHE;
    const dir = slashes(cacheDir("loudreader/loudr-1"));
    assert.ok(dir.endsWith("/loudkit/loudreader--loudr-1"), dir);
  } finally {
    if (saved === undefined) delete process.env.LOUDKIT_CACHE;
    else process.env.LOUDKIT_CACHE = saved;
  }
});

test("offline without the cloning set, the sentence names the graphs", async () => {
  // A receipt for the synthesis set answers a plain load when the hub is
  // away; an enrollment is told what to fetch, not what failed to connect.
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    await download(REPO, dir, { endpoint: stub.endpoint, ...quiet });
    const away = { endpoint: "http://127.0.0.1:9", ...quiet };
    assert.equal(await download(REPO, dir, away), dir);
    await assert.rejects(
      download(REPO, dir, { ...away, cloning: true }),
      /holds no enrollment graphs \(onnx\/s3_tokenizer\.onnx, onnx\/camp\.onnx, onnx\/voice_encoder\.onnx\)\. Connect once to fetch them\./
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a receipt that is not a record is nothing", async () => {
  // An empty file, garbage, and a file too large to be a receipt are no
  // receipt, and the last is not read.
  const dir = workdir();
  try {
    const path = join(dir, RECEIPT_NAME);
    const record =
      `{"repo": "someone/loudr-1", "revision": "main", "commit": "${COMMIT_A}", ` +
      `"sha256sums": null, "fetched_at": "2026-09-02T12:00:00Z"`;
    writeFileSync(path, `${record}}\n`);
    assert.ok(await readReceipt(dir, "someone/loudr-1"), "the record itself was refused");
    const bodies: Record<string, string> = {
      "an empty file": "",
      garbage: "\u0000\u00ff{".repeat(1 << 15),
      "a record padded past the limit": `${record}, "pad": "${"x".repeat(1 << 20)}"}`,
    };
    for (const [name, body] of Object.entries(bodies)) {
      writeFileSync(path, body);
      assert.equal(await readReceipt(dir, "someone/loudr-1"), null, name);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("nothing is claimed verified when the repo ships no SHA256SUMS", async () => {
  // The counter used to rise for every file in the plan while `verifyFetched`
  // returned from its first line, so a repo with no manifest reported
  // "verified 12 files against SHA256SUMS" having hashed none of them: the
  // sentence a caller reads to decide whether the bytes are the release's.
  const files = stubRelease();
  const stub = await hub(files);
  const dir = workdir();
  const lines: string[] = [];
  try {
    // Not an official repo: those are refused before a byte moves for want of
    // the manifest, which is the check this one has no equivalent of.
    await download("someone/loudr-1", dir, {
      endpoint: stub.endpoint,
      onProgress: (m) => lines.push(m),
    });
    assert.equal(
      lines.filter((l) => /^verified \d+ /.test(l)).length,
      0,
      `nothing was hashed, so nothing may count files verified: ${lines.join(" | ")}`
    );
    assert.ok(
      lines.some((l) => l.includes("ships no SHA256SUMS")),
      `the absence is said out loud: ${lines.join(" | ")}`
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a body shorter than the listing is refused rather than cached", async () => {
  // Without this the short file was renamed over the target and the receipt
  // written; the next run read the receipt and never looked again. SHA256SUMS
  // catches it for a release that ships one and lists the file, and a
  // stranger's repo ships neither.
  const files = stubRelease();
  const stub = await hub(files);
  const dir = workdir();
  try {
    // The listing is taken before the body shrinks, so the size the plan
    // carries is the full one and the body is not.
    const full = files.get("onnx/vocoder.onnx")!;
    let shrunk = false;
    await assert.rejects(
      download("someone/loudr-1", dir, {
        endpoint: stub.endpoint,
        onProgress: () => {
          if (shrunk) return;
          shrunk = true;
          files.set("onnx/vocoder.onnx", full.slice(0, 3));
        },
      }),
      /onnx\/vocoder\.onnx: the hub sent 3 bytes where the listing says/
    );
    assert.ok(!existsSync(join(dir, "onnx", "vocoder.onnx")), "the short body was not cached");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

/** A stub that answers 206 from the wrong offset, which is the whole point. */
async function hubAnsweringWrongRange(files: Map<string, string>): Promise<Hub> {
  const server = createServer((request, response) => {
    const url = request.url ?? "";
    if (url.includes("/revision/")) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(JSON.stringify({ sha: COMMIT_A, siblings: [] }));
      return;
    }
    if (url.startsWith("/api/models/")) {
      response.writeHead(200, { "content-type": "application/json" });
      response.end(
        JSON.stringify(
          [...files].map(([p, b]) => ({ type: "file", path: p, size: Buffer.byteLength(b) }))
        )
      );
      return;
    }
    const at = url.indexOf("/resolve/main/");
    const path = at >= 0 ? decodeURIComponent(url.slice(at + "/resolve/main/".length)) : "";
    const body = files.get(path);
    if (body === undefined) {
      response.writeHead(404).end();
      return;
    }
    if (request.headers.range) {
      // The whole file, announced as if it began at zero, which is not the
      // range that was asked for.
      response.writeHead(206, { "content-range": `bytes 0-${body.length - 1}/${body.length}` });
      response.end(body);
      return;
    }
    response.writeHead(200).end(body);
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") throw new Error("no port");
  return {
    endpoint: `http://127.0.0.1:${address.port}`,
    asked: [],
    ranged: [],
    commit: COMMIT_A,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}

test("a resume whose Content-Range does not match is discarded, not appended to", async () => {
  // The port asked for bytes from N and appended whatever came back. A server
  // that answered a different offset, or a file that changed on the hub since
  // the part file was written, produced a file corrupt byte for byte under a
  // name that says otherwise.
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hubAnsweringWrongRange(files);
  const dir = workdir();
  try {
    const target = join(dir, "loudr-1.safetensors");
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(`${target}.part`, "check");
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, ...quiet }),
      /asked the hub for bytes from 5 and it answered/
    );
    assert.ok(!existsSync(`${target}.part`), "the part file was discarded");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("an abort stops the download and is not mistaken for an unreachable hub", async () => {
  // A release is hundreds of megabytes over a link this module sets no
  // deadline on, and there was no way to say stop. An abort must also not fall
  // back to the receipt: the caller cancelled, they did not lose the network.
  const files = stubRelease();
  files.set("SHA256SUMS", sums(files));
  const stub = await hub(files);
  const dir = workdir();
  try {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      download(REPO, dir, { endpoint: stub.endpoint, signal: controller.signal, ...quiet }),
      (e: Error) => e.name === "AbortError" || /abort/i.test(e.message)
    );
    assert.ok(!existsSync(join(dir, RECEIPT_NAME)), "an aborted fetch writes no receipt");
  } finally {
    rmSync(dir, { recursive: true, force: true });
    await stub.close();
  }
});

test("a cache slug replaces every slash, as the other three ports do", () => {
  // `String.replace` with a string pattern replaces one occurrence; Go's
  // ReplaceAll, Rust's str::replace and Swift's replacingOccurrences replace
  // every one. `cacheDir` is exported, so a caller reaching it with a deeper
  // id cached where three ports would not look.
  assert.ok(cachePath("/root", "a/b/c").endsWith(join("loudkit", "a--b--c")), cachePath("/root", "a/b/c"));
  assert.ok(cachePath("/root", "org/name").endsWith(join("loudkit", "org--name")));
});
