/**
 * Getting a release: from the Hugging Face hub, or from a directory on disk.
 *
 * A port of the fetch plan and the inventory check in `loudkit/hub.py`, over
 * `fetch` and the Node standard library, so the first step of the JS
 * quickstart is not "install Python". Only the ONNX set is here, because it is
 * the only one this port can run.
 */

import { createHash } from "node:crypto";
import {
  createReadStream,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { open } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";

import { SafetensorsFile } from "./safetensors.js";
import { decodeFromManifest } from "./types.js";

/** The synthesis artefact's canonical name in a release. */
export const CHECKPOINT_NAME = "loudr-1.safetensors";

/** The turbo release's synthesis artefact, under its own canonical name. */
export const TURBO_CHECKPOINT_NAME = "loudr-1-turbo.safetensors";

/** Every canonical synthesis name, in the order a listing names them. */
const CHECKPOINT_NAMES = [CHECKPOINT_NAME, TURBO_CHECKPOINT_NAME];

/** The enrollment artefact. Torch reads it; this port enrols through graphs. */
export const ENROLLMENT_NAME = "loudr-1-enrollment.safetensors";

/** What `manifest["artifact_role"]` says about each half of a split release. */
const SYNTHESIS_ROLE = "synthesis";
const ENROLLMENT_ROLE = "enrollment";

/** The utterance voice encoder, at a release's root. Torch's, not this port's. */
export const VOICE_ENCODER_NAME = "ve.safetensors";

/** Where voices live inside a release. */
export const VOICE_DIR = "voices";

/** A voice file's extension. The same container as the checkpoint. */
export const VOICE_SUFFIX = ".safetensors";

/** What the exporter writes beside its graphs, naming the checkpoint they came from. */
const EXPORT_RECORD_NAME = "export.json";

/** The same record, addressed from the release root. */
const EXPORT_RECORD = `onnx/${EXPORT_RECORD_NAME}`;

/**
 * The six graphs this port runs, and the record saying they are one export.
 * The record is fetched with the graphs and required of no fetch:
 * `checkExportRecord` warns about a set that has none and refuses one whose
 * record names another engine, so demanding it at the download would refuse
 * exactly the releases exported before it existed.
 */
const ONNX_SYNTHESIS = [
  "onnx/t3_cond.onnx",
  "onnx/t3_prefill.onnx",
  "onnx/t3_step.onnx",
  "onnx/t3_pair_step.onnx",
  "onnx/t3_head2.onnx",
  "onnx/flow_encoder.onnx",
  "onnx/flow_estimator.onnx",
  "onnx/vocoder.onnx",
  EXPORT_RECORD,
];

/** The graphs the engine opens, without the record. */
function synthesisGraphs(mode: "single" | "fusion_mtp2"): string[] {
  return ONNX_SYNTHESIS.filter((g) => g !== EXPORT_RECORD && (
    mode === "fusion_mtp2" ? !g.endsWith("t3_step.onnx") : !g.endsWith("t3_pair_step.onnx") && !g.endsWith("t3_head2.onnx")
  ));
}

/** The three graphs `Enroller` runs. Fetched only when cloning was asked for. */
const ONNX_ENROLL = ["onnx/s3_tokenizer.onnx", "onnx/camp.onnx", "onnx/voice_encoder.onnx"];

/**
 * What every backend's set carries. `*.safetensors` is the checkpoint and,
 * nested, every voice: hub patterns are `fnmatch`, where `*` crosses `/`.
 */
const RELEASE_CORE = [
  "*.safetensors",
  "manifest.json",
  "tokenizer.json",
  "release.json",
  `${VOICE_DIR}/*`,
  "SHA256SUMS",
];

const SUMS = "SHA256SUMS";
const RELEASE_RECORD = "release.json";

const HUB = "https://huggingface.co";

/** The most pages of a file listing that are followed, as in Go and Swift. */
const MAX_LISTING_PAGES = 100;

/**
 * `org/name`, and nothing with a second separator in it.
 *
 * Letters and numbers, not `\w`: a repo id is a Hub name and Hub names are
 * Unicode, so `łódź/model` is one. `\w` is ASCII in this engine and in Go's,
 * and the reference, Rust and Swift all classify by character rather than by
 * byte, which made the same id a repo id in three ports and a missing file in
 * two.
 */
const REPO_ID = /^[\p{L}\p{N}_.-]+\/[\p{L}\p{N}_.-]+$/u;

/**
 * Whether `ref` names a Hugging Face repo rather than a path.
 *
 * Anything that exists on disk is a path, always; path-shaped strings are
 * paths even when they do not exist yet, so a mistyped filename gets "no such
 * file" rather than a request to a repo that will never exist.
 */
export function isRepoId(ref: string): boolean {
  if (ref === "" || existsSync(ref)) return false;
  if (ref.startsWith(".") || ref.startsWith("/") || ref.startsWith("~")) return false;
  if (ref.endsWith(VOICE_SUFFIX)) return false;
  return REPO_ID.test(ref);
}

/**
 * `{ allow, ignore }` for a selective ONNX fetch. The two ignored names are
 * the enrollment tensors and the voice encoder that only torch opens.
 *
 * Both decode loops are allowed, which is what `release.py:_onnx_allow`,
 * `go/hub.go` and `rust/src/hub.rs` each do and what the `onnx` cases in
 * `tests/data/conformance/release_plan.json` pin. Narrowing the set by the
 * loop a listing happens to advertise is an inventory question, not a fetch
 * one: a repo carrying both loops was fetched here without `t3_step.onnx`,
 * and the same narrowing then declared the cache complete over a directory
 * missing it. `synthesisGraphs` is where the loop does decide, at the two
 * places that open the graphs.
 */
export function releasePatterns(cloning = false): { allow: string[]; ignore: string[] } {
  return {
    allow: [...RELEASE_CORE, ...ONNX_SYNTHESIS, ...(cloning ? ONNX_ENROLL : [])],
    ignore: [VOICE_ENCODER_NAME, ENROLLMENT_NAME],
  };
}

/** `fnmatch`, where `*` crosses `/`. Only `*` and `?` are translated. */
export function matches(path: string, pattern: string): boolean {
  const source = pattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/[*?]/g, (c) => (c === "*" ? "[^]*" : "[^]"));
  return new RegExp(`^${source}$`).test(path);
}

/** The repo paths a fetch should bring back, sorted. */
export function planDownload(paths: readonly string[], cloning = false): string[] {
  const { allow, ignore } = releasePatterns(cloning);
  return paths
    .filter((p) => allow.some((a) => matches(p, a)) && !ignore.some((i) => matches(p, i)))
    .sort();
}

/** What a directory has to hold to count as a usable set. */
export interface InventoryOptions {
  /** The three enrollment graphs must be there too. Default false. */
  cloning?: boolean;
  /** A voiceless directory is a failure. True for a fetch, false for a load. */
  requireVoices?: boolean;
  /** `manifest.json` must be there. True for a fetch; a load reads the one the checkpoint embeds. */
  requireManifest?: boolean;
}

/**
 * The inventory `planDownload` promised must be on disk. `allow` is a
 * request, not a receipt: the hub answers with whatever subset the repo
 * holds.
 *
 * `tokenizer.json` is required unconditionally: this port's checkpoint reader
 * does not unpack assets, so the sibling is the only tokenizer it can open.
 */
export function verifyReleaseInventory(root: string, options: InventoryOptions = {}): void {
  const missing: string[] = [];
  let checkpoint: string | null = null;
  try {
    checkpoint = resolveCheckpoint(root);
  } catch {
    missing.push(CHECKPOINT_NAME);
  }
  let mode: "single" | "fusion_mtp2" = "single";
  if (checkpoint !== null) mode = decodeFromManifest(headerManifest(checkpoint) ?? {});
  const expected = ["tokenizer.json", ...synthesisGraphs(mode)];
  if (options.requireManifest) expected.push("manifest.json");
  if (options.cloning) expected.push(...ONNX_ENROLL);
  for (const rel of expected) {
    if (!isFile(join(root, rel))) missing.push(rel);
  }
  if (options.requireVoices && listVoices(root).length === 0) {
    missing.push(`${VOICE_DIR}/*${VOICE_SUFFIX}`);
  }
  if (missing.length > 0) {
    throw new Error(
      // The reference's sentence, which Go, Rust and Swift also say.
      `${root}: this fetch does not add up to a usable onnx set: ` +
        `missing: ${missing.sort().join(", ")}. The release does not carry these ` +
        "files, or the fetch was interrupted; retry, or pin a revision that ships them."
    );
  }
}

/** Whether the directory carries the three enrollment graphs. */
export function canEnroll(root: string): boolean {
  return ONNX_ENROLL.every((rel) => isFile(join(root, rel)));
}

function isFile(path: string): boolean {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

/** The voice names a release directory holds, sorted. */
export function listVoices(root: string): string[] {
  try {
    return readdirSync(join(root, VOICE_DIR))
      .filter((n) => n.endsWith(VOICE_SUFFIX))
      .map((n) => n.slice(0, -VOICE_SUFFIX.length))
      .sort();
  } catch {
    return [];
  }
}

/**
 * `manifest["artifact_role"]` for a checkpoint, or undefined for a file that
 * makes no claim.
 *
 * Undefined covers both a pre-split checkpoint, whose manifest predates the
 * field and which does carry every tensor, and a file this cannot read at all.
 * The field is only ever read to refuse, so a missing claim never promotes a
 * file to a role it did not ask for.
 */
function artifactRole(path: string): string | undefined {
  const role = headerManifest(path)?.artifact_role;
  return typeof role === "string" ? role : undefined;
}

/** Refuse a file whose manifest declares it to be the other artefact. */
function refuseRole(path: string, expected: string): void {
  const role = artifactRole(path);
  if (role !== undefined && role !== expected) {
    throw new Error(
      `${path}: this is a release's ${role} artefact, and the ${expected} ` +
        "artefact is what was asked for. Pass the release directory, or the " +
        "repo id, and let the resolver pick."
    );
  }
}

/**
 * The synthesis checkpoint inside a release directory, or a useful complaint.
 *
 * Three rules, in order: exactly one canonical name; otherwise the file that
 * declares the synthesis role; otherwise exactly one candidate, with anything
 * declaring the enrollment role set aside first. The voice encoder is not a
 * candidate, and voices are one directory down.
 */
export function resolveCheckpoint(root: string): string {
  const canonical = CHECKPOINT_NAMES.filter((n) => isFile(join(root, n)));
  if (canonical.length === 1) {
    const named = join(root, canonical[0]);
    refuseRole(named, SYNTHESIS_ROLE);
    return named;
  }
  // Two released models under their own names is not an ambiguity a resolver
  // may settle: they are different releases with different decode loops, so
  // picking either is picking a model the caller did not ask for.
  if (canonical.length > 1) {
    throw new Error(
      `${root}: ${canonical.length} models here (${canonical.join(", ")}): name ` +
        "the one you mean. They are different releases with different decode " +
        "loops, so there is no right one to pick."
    );
  }
  let entries: string[];
  try {
    entries = readdirSync(root);
  } catch {
    throw new Error(`${root}: not a directory`);
  }
  const found = entries
    .filter(
      (n) =>
        n.endsWith(VOICE_SUFFIX) &&
        n !== VOICE_ENCODER_NAME &&
        n !== ENROLLMENT_NAME &&
        // A dot-prefixed file is not something a release build can produce, so
        // it is not a candidate. The reference, Go, Rust and Swift all skip
        // them; counting one here turned a directory with one model and one
        // editor or sync leftover into "name the one you mean".
        !n.startsWith(".") &&
        isFile(join(root, n))
    )
    .sort();
  const roles = new Map(found.map((n) => [n, artifactRole(join(root, n))]));
  const declared = found.filter((n) => roles.get(n) === SYNTHESIS_ROLE);
  const candidates =
    declared.length > 0 ? declared : found.filter((n) => roles.get(n) !== ENROLLMENT_ROLE);
  if (candidates.length === 1) return join(root, candidates[0]);
  if (found.length === 0) {
    throw new Error(`${root}: no ${CHECKPOINT_NAME} and no other checkpoint beside it`);
  }
  if (candidates.length === 0) {
    throw new Error(
      `${root}: the only checkpoint here is a release's enrollment artefact ` +
        `(${found.join(", ")}). It carries the two modules a clone needs and ` +
        `nothing synthesis reads. Fetch the release's ${CHECKPOINT_NAME} beside it.`
    );
  }
  throw new Error(
    `${root}: ${candidates.join(", ")}, and no ${CHECKPOINT_NAME}, ` + "so name the one you mean"
  );
}

/** The three paths `Engine.load` needs, found inside a release directory. */
export interface Bundle {
  checkpoint: string;
  onnxDir: string;
  tokenizer: string;
}

/**
 * Read a release directory as a bundle, or say why it is not one. The
 * inventory check runs first so that a half-fetched directory fails here,
 * naming the missing graph, rather than inside onnxruntime.
 */
export function resolveBundle(root: string): Bundle {
  verifyReleaseInventory(root, {});
  return {
    checkpoint: resolveCheckpoint(root),
    onnxDir: join(root, "onnx"),
    tokenizer: join(root, "tokenizer.json"),
  };
}

/**
 * Where a fetch lands when the caller names no directory: `cachePath` under
 * the platform's user cache, or `$LOUDKIT_CACHE/<org>--<name>` when that
 * variable is set. The same layout in Go, Rust and Swift, pinned by
 * `tests/data/conformance/cache_path.json`, so a cache one port wrote is a
 * cache the other three read.
 */
export function cacheDir(repo: string): string {
  const override = env("LOUDKIT_CACHE");
  if (override) return join(override, cacheSlug(repo));
  return cachePath(userCacheRoot(), repo);
}

/** `<root>/loudkit/<org>--<name>`. */
export function cachePath(root: string, repo: string): string {
  return join(root, "loudkit", cacheSlug(repo));
}

function cacheSlug(repo: string): string {
  // Every slash, not the first: `String.replace` with a string pattern
  // replaces one occurrence, where Go's `strings.ReplaceAll`, Rust's
  // `str::replace` and Swift's `replacingOccurrences` all replace every one.
  // A repo id with two slashes therefore cached under a directory three ports
  // would not look in, and `cacheDir` is exported.
  return repo.replaceAll("/", "--");
}

/**
 * The platform's user cache: `~/Library/Caches` on macOS, `%LocalAppData%`
 * on Windows, `$XDG_CACHE_HOME` or `~/.cache` elsewhere.
 */
function userCacheRoot(): string {
  if (process.platform === "darwin") return join(homedir(), "Library", "Caches");
  if (process.platform === "win32") {
    return env("LOCALAPPDATA") ?? join(homedir(), "AppData", "Local");
  }
  return env("XDG_CACHE_HOME") ?? join(homedir(), ".cache");
}

/**
 * The manifest embedded in a checkpoint, from its header alone: the file is
 * 747 MB and this runs while deciding whether to open it at all.
 */
function headerManifest(path: string): Record<string, unknown> | undefined {
  try {
    const manifest = SafetensorsFile.header(path).metadata?.manifest;
    return typeof manifest === "string"
      ? (JSON.parse(manifest) as Record<string, unknown>)
      : undefined;
  } catch {
    // Undefined, not a throw: this runs while deciding what a directory holds,
    // and a file that is not a readable checkpoint is one this caller reports
    // in its own words rather than one it dies on.
    return undefined;
  }
}

/**
 * Make sure `dir` holds the three enrollment graphs. A release loaded by
 * repo id fetches them into its own directory, through the same
 * receipt-aware `download`, so the cache grows to the wider set once and
 * stays a hit afterwards. A directory of your own is told how to fetch them.
 */
export async function ensureCloning(
  dir: string,
  repo: string | undefined,
  options: DownloadOptions = {}
): Promise<void> {
  if (canEnroll(dir)) return;
  if (repo === undefined) {
    throw new Error(
      `${dir} carries no enrollment graphs. Fetch them with download(repo, dir, { cloning: true })`
    );
  }
  await download(repo, dir, { ...options, cloning: true });
}

/**
 * A release directory for `ref`, fetching it if `ref` is a repo id.
 *
 * Anything that exists on disk is a path, always. A repo id goes through
 * `download` into the cache, which holds the whole cache contract: a cached
 * copy whose receipt names today's commit is read as it is, and offline the
 * receipt stands in.
 */
export async function ensureBundle(ref: string, options: DownloadOptions = {}): Promise<string> {
  if (existsSync(ref)) return ref;
  ref = modelReference(ref);
  if (!isRepoId(ref)) {
    throw new Error(`${ref}: no such directory, and not an "org/name" repo id either`);
  }
  return download(ref, cacheDir(ref), options);
}

function modelReference(ref: string): string {
  return ref === "loudr-1" || ref === "loudr-1-turbo" ? `loudreader/${ref}` : ref;
}

/** What a verified download directory carries, in every port. */
export const RECEIPT_NAME = ".loudkit-release.json";

/**
 * The most a receipt file is read: five short fields. A file past it is not
 * a receipt, and is not read.
 */
const RECEIPT_LIMIT = 1 << 20;

/**
 * The receipt: the commit the asked revision resolved to when the files
 * were hashed. A later load whose revision still resolves to that commit,
 * over a receipt `readReceipt` accepts, is a hit and hashes no weight. The
 * shape and the rules are pinned by
 * `tests/data/conformance/release_receipt.json`.
 */
export interface Receipt {
  repo: string;
  /** As asked; `"main"` when the caller named none. */
  revision: string;
  commit: string;
  /** The digest of `SHA256SUMS`, or `null` for a release without one. */
  sha256sums: string | null;
  /** UTC, to the second. */
  fetched_at: string;
}

const COMMIT_HEX = /^[0-9a-f]{40}$/;

/**
 * The receipt under `dir` when it vouches for `repo`, else `null`: every
 * field present with its type (`null` is the wrong type for all but
 * `sha256sums`; keys it does not know are ignored), `repo` the one asked,
 * `commit` forty lowercase hex, `sha256sums` the digest of the `SHA256SUMS`
 * on disk (`null` for none), and every file it lists that the plan selects
 * present. No weight is hashed: the receipt is checked, not the release.
 * This is the only place the file is read.
 */
export async function readReceipt(
  dir: string,
  repo: string,
  cloning = false
): Promise<Receipt | null> {
  let record: unknown;
  try {
    const path = join(dir, RECEIPT_NAME);
    if (statSync(path).size > RECEIPT_LIMIT) return null;
    record = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return null;
  }
  if (record === null || typeof record !== "object" || Array.isArray(record)) return null;
  const r = record as Record<string, unknown>;
  const strings = ["repo", "revision", "commit", "fetched_at"].every(
    (key) => typeof r[key] === "string"
  );
  if (!strings || !(typeof r.sha256sums === "string" || r.sha256sums === null)) return null;
  const receipt: Receipt = {
    repo: r.repo as string,
    revision: r.revision as string,
    commit: r.commit as string,
    sha256sums: r.sha256sums,
    fetched_at: r.fetched_at as string,
  };
  if (receipt.repo !== repo || !COMMIT_HEX.test(receipt.commit)) return null;
  const sums = join(dir, SUMS);
  if (!isFile(sums)) return receipt.sha256sums === null ? receipt : null;
  if (receipt.sha256sums !== (await sha256(sums))) return null;
  try {
    const present = planDownload([...parseSums(sums).keys()], cloning).every((p) =>
      isFile(join(dir, p))
    );
    return present ? receipt : null;
  } catch {
    return null;
  }
}

/** Record that `dir` holds `repo` at `commit`, verified. */
async function writeReceipt(dir: string, repo: string, revision: string, commit: string) {
  const sums = join(dir, SUMS);
  const receipt: Receipt = {
    repo,
    revision,
    commit,
    sha256sums: isFile(sums) ? await sha256(sums) : null,
    fetched_at: new Date().toISOString().replace(/\.\d{3}Z$/, "Z"),
  };
  writeFileSync(join(dir, RECEIPT_NAME), `${JSON.stringify(receipt, null, 1)}\n`);
}

/**
 * Whether `dir` already holds `repo` at `commit`: a receipt `readReceipt`
 * accepts, naming that commit. An empty commit never hits: nothing was
 * resolved.
 */
export async function receiptHit(
  dir: string,
  repo: string,
  commit: string,
  cloning = false
): Promise<boolean> {
  return commit !== "" && (await readReceipt(dir, repo, cloning))?.commit === commit;
}

/**
 * The sha the hub resolves `revision` to today: one request, the same one
 * that says whether the repo and the revision exist at all. `null` when the
 * transport failed and the hub never answered.
 */
async function resolveCommit(
  repo: string,
  revision: string,
  endpoint: string,
  token: string | undefined,
  signal: AbortSignal | undefined
): Promise<string | null> {
  const url = `${endpoint}/api/models/${repo}/revision/${encodeURIComponent(revision)}`;
  let response: Response;
  try {
    response = await fetch(url, { headers: headers(token), signal });
  } catch (cause) {
    // An abort is a decision the caller made, not a hub that cannot be
    // reached: swallowing it here would answer a cancelled download with the
    // cached release, or with a sentence about the network that is not true.
    if (signal?.aborted) throw cause;
    return null;
  }
  if (!response.ok) {
    throw new Error(
      `${repo}: the hub answered ${response.status} resolving revision ${revision}. ` +
        "Check that the release is published and the repo id and revision are correct. Set HF_TOKEN if it is gated."
    );
  }
  const record = (await response.json()) as { sha?: unknown };
  if (typeof record.sha !== "string" || record.sha === "") {
    throw new Error(`${repo} at ${revision}: the hub named no commit for this revision`);
  }
  return record.sha;
}

/**
 * Whether the file already at `target` is the one the manifest lists under
 * `name`: present, listed, and hashing to the listed digest. Size decides
 * nothing; a file edited in place at its published length is caught here on
 * the next miss.
 */
async function keptByHash(target: string, name: string, sums: Map<string, string>) {
  const want = sums.get(name);
  return want !== undefined && isFile(target) && (await sha256(target)) === want;
}

/**
 * Whether `dir` holds the inventory the port runs on, by name. The receipt
 * vouches for what `SHA256SUMS` lists; this is the rest.
 */
function complete(dir: string, cloning: boolean | undefined): boolean {
  try {
    verifyReleaseInventory(dir, { cloning, requireVoices: true, requireManifest: true });
    return true;
  } catch {
    return false;
  }
}

/** How `download` should behave. Every field has a working default. */
export interface DownloadOptions {
  /** Also fetch the three enrollment graphs `Enroller` runs. Default false. */
  cloning?: boolean;
  /** Branch, tag or commit. Default `"main"`. */
  revision?: string;
  /** Hub base URL. Default `HF_ENDPOINT`, else `https://huggingface.co`. */
  endpoint?: string;
  /** Bearer token for a gated repo. Default `HF_TOKEN`. */
  token?: string;
  /** Where progress goes. Default: one line per file on stderr. */
  onProgress?: (message: string) => void;
  /**
   * Cancels the download, including a transfer already in flight.
   *
   * The form to reach for inside a server: a release is hundreds of megabytes
   * over a link this module sets no deadline on, so without one the fetch runs
   * to its own end whatever the caller decided. Aborting stops the transfer
   * and rejects with the signal's reason; the part file stays behind and the
   * next call resumes it.
   *
   * An abort is never mistaken for a hub that cannot be reached, so it does
   * not fall back to the receipt. Go names the same behaviour on
   * `DownloadContext`; this is the field rather than a second entry point
   * because `download`'s options are already a bag and `AbortSignal` is where
   * `fetch` itself takes cancellation.
   */
  signal?: AbortSignal;
}

interface TreeEntry {
  type: string;
  path: string;
  size?: number;
  /**
   * An LFS-backed file's real length. `size` is the pointer's for a repo whose
   * listing reports it that way, and every weight file in a release is LFS, so
   * a fetch checked its byte count against a few hundred bytes. `go/hub.go`
   * and `swift/LoudKit/Hub.swift` both prefer this field.
   */
  lfs?: { size?: number };
}

/**
 * The length the listing gives for a file, or -1 when it gives none.
 *
 * Type-checked, not `?? -1`: a listing answering `"size": "1234"` carried the
 * string through to the byte-count check, which then refused a complete
 * download saying it got 1234 bytes where the listing says 1234.
 */
function listedSize(entry: TreeEntry): number {
  const lfs = entry.lfs?.size;
  if (typeof lfs === "number" && lfs > 0) return lfs;
  return typeof entry.size === "number" ? entry.size : -1;
}

function env(name: string): string | undefined {
  const value = process.env[name];
  return value && value.length > 0 ? value : undefined;
}

function headers(token: string | undefined): Record<string, string> {
  return token ? { authorization: `Bearer ${token}` } : {};
}

async function listRepo(
  repo: string,
  revision: string,
  endpoint: string,
  token: string | undefined,
  signal: AbortSignal | undefined
): Promise<TreeEntry[]> {
  const out: TreeEntry[] = [];
  let url: string | null =
    `${endpoint}/api/models/${repo}/tree/${encodeURIComponent(revision)}?recursive=1`;
  let pages = 0;
  while (url) {
    // A release is tens of files and the tree API pages by the thousand, so a
    // listing past this is a `Link: rel="next"` cycle rather than a large
    // repository, and following it never returns.
    if (pages++ >= MAX_LISTING_PAGES) {
      throw new Error(`${repo}: the file listing does not end`);
    }
    const response: Response = await fetch(url, { headers: headers(token), signal });
    if (!response.ok) {
      throw new Error(
        `${repo}: the hub answered ${response.status} listing revision ${revision}. ` +
          "Check the repo id, and set HF_TOKEN if the repo is gated."
      );
    }
    out.push(...((await response.json()) as TreeEntry[]));
    // The listing pages at 1000 entries.
    const link = response.headers.get("link");
    const next = link ? /<([^>]+)>;\s*rel="next"/.exec(link) : null;
    url = next ? next[1] : null;
  }
  return out;
}

function human(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} kB`;
  return `${bytes} B`;
}

async function sha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk as Uint8Array);
  return hash.digest("hex");
}

/** The tool that writes the record, and what a refusal tells the reader to run. */
const EXPORT_TOOL = "tools/export_onnx.py";

/**
 * The fields that make one export's identity, in the order a message prints
 * them. The first three are the engine the graphs were traced against; the
 * fourth is the renderer inside it.
 */
const RECORD_KEYS = [
  "checkpoint_sha256",
  "algorithm_fingerprint",
  "euler_steps",
  "estimator_sha256",
] as const;

/**
 * One member's record, rendered for comparison. Each field holds the value the
 * way Python's `repr()` prints it, so a set that agrees here agrees there and a
 * refusal reads the same in both.
 */
type Identity = readonly string[];

/**
 * Refuse a graph set whose members did not come from one export of this
 * checkpoint.
 *
 * Graphs in one folder look like a set and need not be one: the exporters take
 * a stage list, so a run that names the renderer leaves the other graphs as
 * they were, and a mixed folder speaks one checkpoint's tokens through another
 * checkpoint's renderer. The record beside the set names, per member, the
 * checkpoint digest, the fingerprint, the step count and the estimator digest;
 * every member must agree with the others and with the checkpoint being loaded.
 *
 * An absent record is a warning rather than a refusal, because sets exported
 * before the record exist and stranding them buys nothing a sentence cannot
 * say.
 */
export async function checkExportRecord(
  assets: string,
  checkpointPath: string,
  manifest: Record<string, unknown>,
  fingerprint: string,
  eulerSteps: number,
  graphs: readonly string[]
): Promise<void> {
  const path = join(assets, EXPORT_RECORD_NAME);
  if (!existsSync(path)) {
    process.emitWarning(
      `${assets} carries no ${EXPORT_RECORD_NAME}, so nothing says its ${graphs.length} ` +
        "graphs came from one export of one checkpoint. Re-export with " +
        `${EXPORT_TOOL} to record it.`,
      "LoudkitExportRecordWarning"
    );
    return;
  }
  const seen = recordedGraphs(path);
  for (const name of graphs) {
    if (!(name in seen)) {
      throw new Error(
        `${path} does not record ${name}, so it came from some other run than the ` +
          "ones it does record. Re-export the set."
      );
    }
  }
  let agreed: Identity | undefined;
  for (const got of Object.values(seen)) {
    if (agreed !== undefined && !same(agreed, got)) {
      throw new Error(
        `${assets} is a mixed graph set: its members were exported from different ` +
          `inputs:\n${rows(seen)}\nRe-export every stage together.`
      );
    }
    agreed = got;
  }
  // Every graph asked for is recorded by now, so a set with graphs has an
  // identity; a caller asking for none has nothing to disagree with.
  if (agreed === undefined) return;
  const want = [quote(await sha256(checkpointPath)), quote(fingerprint), String(eulerSteps)];
  if (!same(agreed.slice(0, 3), want)) {
    throw new Error(
      `${assets} was exported from a different engine than the one loading it:\n` +
        `  graphs: ${fields(agreed.slice(0, 3))}\n` +
        `  checkpoint: ${fields(want)}\n` +
        `Re-export against ${basename(checkpointPath)}.`
    );
  }
  // A set traced from a swapped-in estimator agrees with itself and describes a
  // renderer the checkpoint does not; compared only when both sides record one.
  const packed = packedEstimator(manifest);
  if (packed !== undefined && agreed[3] !== "None" && agreed[3] !== quote(packed)) {
    throw new Error(
      `${assets} was traced with an estimator the checkpoint was not packed from:\n` +
        `  traced:  ${agreed[3].replaceAll("'", "")}\n  packed:  ${packed}\n` +
        `That is a different renderer than ${basename(checkpointPath)} describes, and ` +
        "its fingerprint does not cover the difference. Re-export without " +
        "--estimator-ckpt, or pack the estimator you traced."
    );
  }
}

/**
 * The record's graph block. An absent block is an unreadable record; a block
 * that is not a mapping of name to its fields is named as itself, because the
 * remedy differs.
 */
function recordedGraphs(path: string): Record<string, Identity> {
  const unreadable = (what: string) => new Error(`${path}: unreadable export record (${what})`);
  let root: unknown;
  try {
    root = JSON.parse(readFileSync(path, "utf8"));
  } catch (err) {
    throw unreadable(err instanceof Error ? err.message : String(err));
  }
  const block = isRecord(root) ? root["graphs"] : undefined;
  if (block === undefined) throw unreadable("no 'graphs' block");
  const malformed = new Error(
    `${path}: the 'graphs' block is not a mapping of name to its export record. ` +
      `Re-export the set with ${EXPORT_TOOL}.`
  );
  if (!isRecord(block)) throw malformed;
  const seen: Record<string, Identity> = {};
  for (const [name, entry] of Object.entries(block)) {
    const got = asIdentity(entry);
    if (got === undefined) throw malformed;
    seen[name] = got;
  }
  return seen;
}

/**
 * One member's entry, or undefined for a value shape the exporter never writes.
 * An absent field and a null one are the same answer, so a record written
 * before a field existed still compares.
 */
function asIdentity(entry: unknown): Identity | undefined {
  if (!isRecord(entry)) return undefined;
  const out: string[] = [];
  for (const key of RECORD_KEYS) {
    const value = entry[key];
    if (value === undefined || value === null) out.push("None");
    else if (typeof value === "string") out.push(quote(value));
    else if (typeof value === "number" && Number.isInteger(value)) out.push(String(value));
    else return undefined;
  }
  return out;
}

/** The sha256 of the estimator the checkpoint was packed from, if its sources say. */
function packedEstimator(manifest: Record<string, unknown>): string | undefined {
  const sources = manifest["sources"];
  if (!isRecord(sources)) return undefined;
  for (const entry of Object.values(sources)) {
    if (isRecord(entry) && entry["role"] === "estimator" && typeof entry["sha256"] === "string") {
      return entry["sha256"];
    }
  }
  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function same(a: Identity, b: Identity): boolean {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}

/** A value the way Python's `repr()` prints a string, so a refusal reads the same in both. */
function quote(value: string): string {
  return `'${value}'`;
}

/** The fields of one record, as the reference prints the mapping. */
function fields(got: Identity): string {
  return `{${got.map((value, i) => `${quote(RECORD_KEYS[i])}: ${value}`).join(", ")}}`;
}

/** Every member's record, one per line, for a set that disagrees with itself. */
function rows(seen: Record<string, Identity>): string {
  return Object.keys(seen)
    .sort()
    .map((name) => `  ${name}: ${fields(seen[name])}`)
    .join("\n");
}

const DRIVE_LETTER = /^[A-Za-z]:\//;

/** Why `name` cannot address bytes inside the release, or `null` when it can. */
export function rejectedName(name: string): string | null {
  if (name.includes("\\")) return "is not a POSIX path (it contains a backslash)";
  if (name.startsWith("/") || DRIVE_LETTER.test(name)) {
    return "is absolute, and a manifest name is relative to the release root";
  }
  const parts = name.split("/");
  if (parts.includes("..")) return "escapes the release root with '..'";
  if (parts.some((p) => p === "" || p === ".")) {
    return "is not normalised (an empty or '.' path component)";
  }
  return null;
}

/**
 * `SHA256SUMS` as `path -> digest`, refusing a line it cannot understand, a
 * name that escapes the release and a name listed twice.
 */
export function parseSums(path: string): Map<string, string> {
  const sums = new Map<string, string>();
  const lines = readFileSync(path, "utf8").split("\n");
  for (const [index, raw] of lines.entries()) {
    const line = raw.replace(/\r$/, "");
    if (line.trim() === "") continue;
    const match = /^([0-9a-f]{64}) {2}(\S.*)$/.exec(line);
    if (!match) {
      throw new Error(
        `${path}: malformed line ${index + 1}: ${JSON.stringify(line)}. This does not look ` +
          "like a loudkit release manifest; refusing to verify against it"
      );
    }
    const name = match[2];
    const why = rejectedName(name);
    if (why) {
      throw new Error(
        `${path}: line ${index + 1}: ${JSON.stringify(name)} ${why}; refusing to verify ` +
          "against a manifest that names files outside the release it describes"
      );
    }
    if (sums.has(name)) {
      throw new Error(
        `${path}: line ${index + 1}: duplicate entry for ${JSON.stringify(name)}. The ` +
          "manifest disagrees with itself about one file. Rebuild the release"
      );
    }
    sums.set(name, match[1]);
  }
  if (sums.size === 0) throw new Error(`${path}: no checksum entries`);
  return sums;
}

/** The org whose releases this library vouches for. Same name as Python's. */
const OFFICIAL_ORG = "loudreader";

/**
 * The profiles a builder stamps on a releasable bundle: `full-0.1` for
 * loudr-1 and `turbo-0.1` for loudr-1-turbo. Neither is written until the
 * builder's load-and-speak gate passes.
 */
const STRICT_PROFILES = ["full-0.1", "turbo-0.1"];

function isOfficial(repo: string): boolean {
  return repo.split("/")[0].toLowerCase() === OFFICIAL_ORG;
}

/**
 * An official release must say it is one. Checksums prove the bytes arrived
 * intact and say nothing about what the bytes are: a development bundle
 * carries a perfectly valid `SHA256SUMS`.
 */
async function requireReleasable(dir: string, where: string, sums: Map<string, string>) {
  const path = join(dir, RELEASE_RECORD);
  if (!isFile(path)) {
    throw new Error(
      `${where}: no release.json. Every ${OFFICIAL_ORG} release records its ` +
        "profile and its verified flag there, so this download cannot prove it is " +
        "a release. Pin a revision you trust."
    );
  }
  const expected = sums.get(RELEASE_RECORD);
  if (expected === undefined) {
    throw new Error(
      `${where}: ${SUMS} does not list release.json, so the record that would vouch ` +
        "for this release is vouched for by nothing. Pin a revision you trust."
    );
  }
  if ((await sha256(path)) !== expected) {
    throw new Error(`${where}: release.json failed the release checksum. Delete the directory and retry.`);
  }
  let record: unknown;
  try {
    record = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(
      `${where}: release.json is unreadable. Delete the directory and retry, ` +
        "or pin a revision you trust.",
      { cause: error }
    );
  }
  const claim = (record ?? {}) as Record<string, unknown>;
  if (typeof claim.profile !== "string" || !STRICT_PROFILES.includes(claim.profile)) {
    throw new Error(
      `${where}: release.json says profile ${JSON.stringify(claim.profile)}, and an ` +
        `${OFFICIAL_ORG} release is one of ${STRICT_PROFILES.join(", ")}. This is a ` +
        "development bundle, not the release. Fetch a published revision."
    );
  }
  if (claim.verified !== true) {
    throw new Error(
      `${where}: release.json does not record verified: true, so the bundle never ` +
        "passed the builder's load-and-speak gate. Fetch a published revision."
    );
  }
}

/**
 * Hash one file this run fetched against the release's `SHA256SUMS`. Three
 * rules, shared with `loudkit/hub.py`: no manifest is fine only for a
 * stranger's repo; a fetched file the manifest does not list is refused when
 * it is weights, refused under `loudreader/` whatever it is, and reported
 * otherwise; and a digest that does not match is refused, with the file
 * removed so the next run fetches it again.
 */
async function verifyFetched(
  dir: string,
  repo: string,
  name: string,
  sums: Map<string, string> | null,
  say: (m: string) => void
): Promise<boolean> {
  if (sums === null) return false;
  const target = join(dir, name);
  const expected = sums.get(name);
  if (expected === undefined) {
    if (name.endsWith(VOICE_SUFFIX)) {
      rmSync(target, { force: true });
      throw new Error(
        `${dir}: ${SUMS} does not list ${name}, which is weights loudkit would open ` +
          "with nothing vouching for them. Pin a revision you trust."
      );
    }
    if (isOfficial(repo)) {
      rmSync(target, { force: true });
      throw new Error(
        `${dir}: ${SUMS} does not list ${name}. A ${OFFICIAL_ORG} release checksums every ` +
          "file it ships, so this did not come from the release."
      );
    }
    say(`${dir}: ${name} is not covered by ${SUMS} and therefore not verified`);
    return false;
  }
  if ((await sha256(target)) !== expected) {
    rmSync(target, { force: true });
    throw new Error(
      `${dir}: ${name} failed the release checksum. The file has been removed; ` +
        "run the download again, or pin a revision you trust."
    );
  }
  return true;
}

/**
 * What the listing alone can say about a repo, before a byte moves: an
 * official repo must ship the two bookkeeping files, and a repo without the
 * onnx graphs cannot run here whatever else it carries.
 */
function refuseBeforeFetching(repo: string, revision: string, wanted: string[]): void {
  if (isOfficial(repo)) {
    for (const name of [SUMS, RELEASE_RECORD]) {
      if (!wanted.includes(name)) {
        throw new Error(
          `${repo} at ${revision}: no ${name}. Every ${OFFICIAL_ORG} release ships one, ` +
            "so this cannot be checked against anything and will not be fetched. Pin a " +
            "revision you trust."
        );
      }
    }
  }
  const mode = wanted.includes("onnx/t3_pair_step.onnx") ? "fusion_mtp2" : "single";
  for (const graph of synthesisGraphs(mode)) {
    if (wanted.includes(graph)) continue;
    throw new Error(
      `${repo} at ${revision} ships no ${graph}, which this port runs on. Use ` +
        "loudreader/loudr-1, or pin a revision that carries the graphs."
    );
  }
}

/** The two bookkeeping files ahead of everything else, the rest in order. */
function bookkeepingFirst(paths: string[]): string[] {
  const rank = (p: string) => (p === SUMS ? 0 : p === RELEASE_RECORD ? 1 : 2);
  return [...paths].sort((a, b) => rank(a) - rank(b));
}

/**
 * Where a `Content-Range` says the bytes in a 206 came from, or null when the
 * header is missing or does not parse.
 *
 * `bytes <first>-<last>/<total>`, and `*` for an unknown total. A resume that
 * does not read this is trusting a server to have answered the range it was
 * asked for.
 */
function contentRange(header: string | null): { first: number; total: number | null } | null {
  const m = header ? /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/.exec(header.trim()) : null;
  if (!m) return null;
  return { first: Number(m[1]), total: m[3] === "*" ? null : Number(m[3]) };
}

/**
 * Fetch one file, resuming a partial one.
 *
 * The download lands in `<name>.part` and is renamed on completion; a `.part`
 * already on disk is continued with a `Range` request, and a server that
 * answers 200 is honoured by starting over.
 *
 * `expected` is the length the repo listing already gave for this file, or -1
 * when the listing did not say. Three things turn on it, and none of them used
 * to happen here:
 *
 *  - A `.part` at or past that length has nothing left to ask for, so it is
 *    started over rather than resumed into a 416. Go says the same.
 *  - A 206 is checked against its `Content-Range`: a server that answered a
 *    different offset, or a file that changed on the hub since the `.part` was
 *    written, produces a file that is corrupt byte for byte and hashes to
 *    nothing anybody can name. The header says which, so it is read.
 *  - The bytes written are compared against it. Without that a short body was
 *    renamed over the target and cached, and the next run trusted the receipt.
 *    `SHA256SUMS` catches it for a release that ships one and lists the file;
 *    a stranger's repo ships neither, and `SHA256SUMS` cannot vouch for
 *    itself.
 *
 * The transport is wrapped so a broken connection names the file and the URL.
 * undici's bare `TypeError: terminated` says neither, where Go says
 * `onnx/flow_encoder.onnx: unexpected EOF`.
 */
async function fetchFile(
  url: string,
  target: string,
  name: string,
  token: string | undefined,
  report: (done: number) => void,
  expected: number,
  signal: AbortSignal | undefined
): Promise<void> {
  mkdirSync(dirname(target), { recursive: true });
  const part = `${target}.part`;
  const partial = isFile(part) ? statSync(part).size : 0;
  // Only a short `.part` resumes: one at or past the file's length has nothing
  // left to ask for, and the range request comes back 416.
  let have = expected >= 0 && partial >= expected ? 0 : partial;
  const request: Record<string, string> = headers(token);
  if (have > 0) request.range = `bytes=${have}-`;
  const response = await transport(
    () => fetch(url, { headers: request, signal }),
    name,
    url,
    signal
  );
  if (!response.ok || !response.body) {
    throw new Error(`${url}: the hub answered ${response.status}`);
  }
  const resumed = response.status === 206;
  if (resumed) {
    const range = contentRange(response.headers.get("content-range"));
    if (range === null || range.first !== have || (expected >= 0 && range.total !== null && range.total !== expected)) {
      // The part file goes, because it is no longer known what it holds.
      rmSync(part, { force: true });
      throw new Error(
        `${name}: asked the hub for bytes from ${have} and it answered ` +
          `${response.headers.get("content-range") ?? "no Content-Range"}. The ` +
          "partial download has been discarded; run it again."
      );
    }
  } else {
    have = 0;
  }
  let done = have;
  const handle = await open(part, resumed ? "a" : "w");
  try {
    const reader = response.body.getReader();
    for (;;) {
      const { done: finished, value } = await transport(
        () => reader.read(),
        name,
        url,
        signal
      );
      if (finished) break;
      await handle.write(value);
      done += value.length;
      report(done);
    }
  } finally {
    await handle.close();
  }
  if (expected >= 0 && done !== expected) {
    // A short body keeps its part file, because that is what a resume reads;
    // a long one is discarded, because nothing about it matches the listing.
    if (done > expected) rmSync(part, { force: true });
    throw new Error(
      `${name}: the hub sent ${done} bytes where the listing says ${expected}. ` +
        (done > expected
          ? "The partial download has been discarded; run it again."
          : "Run it again to resume.")
    );
  }
  renameSync(part, target);
}

/**
 * One transport step, with the file and the URL in whatever it throws.
 *
 * `fetch` and `reader.read()` reject with undici's `TypeError: terminated`
 * when a connection drops, which names neither, so a caller was left with two
 * words and no way to tell which of twelve files broke. An abort passes
 * through as itself: the caller asked for it and knows what they cancelled.
 */
async function transport<T>(
  step: () => Promise<T>,
  name: string,
  url: string,
  signal: AbortSignal | undefined
): Promise<T> {
  try {
    return await step();
  } catch (cause) {
    if (signal?.aborted) throw cause;
    const why = cause instanceof Error ? cause.message : String(cause);
    throw new Error(`${name}: ${why} (fetching ${url})`, { cause });
  }
}

/**
 * Fetch the ONNX release for `repo` into `dir`, and answer with `dir`.
 *
 * `dir` carries a receipt naming the commit it was verified against: a run
 * whose revision still resolves to that commit, over a directory whose
 * receipt `readReceipt` accepts and which still holds the set, fetches and
 * hashes no weight; a moved revision re-hashes what is here and fetches
 * only what changed; a run that cannot reach the hub uses the receipt and
 * says so. Every file that arrives is hashed against the release's own
 * `SHA256SUMS`, once, here;
 * the two bookkeeping files come first, so a repo that is not a release is
 * refused before its weights move.
 */
export async function download(
  repo: string,
  dir?: string,
  options: DownloadOptions = {}
): Promise<string> {
  repo = modelReference(repo);
  dir ??= cacheDir(repo);
  if (!isRepoId(repo)) {
    throw new Error(`${repo}: not a Hugging Face repo id; expected "org/name"`);
  }
  const revision = options.revision ?? "main";
  const endpoint = (options.endpoint ?? env("HF_ENDPOINT") ?? HUB).replace(/\/$/, "");
  const token = options.token ?? env("HF_TOKEN");
  const say = options.onProgress ?? ((message: string) => process.stderr.write(`${message}\n`));

  const commit = await resolveCommit(repo, revision, endpoint, token, options.signal);
  if (commit === null) {
    const receipt = await readReceipt(dir, repo, options.cloning);
    if (receipt && complete(dir, options.cloning)) {
      say(
        `the hub cannot be reached; using ${dir}, which holds ${receipt.repo} at ` +
          `${receipt.commit} (fetched ${receipt.fetched_at})`
      );
      return dir;
    }
    // A receipt for the synthesis set does not cover an enrollment: the
    // sentence names the graphs, not the transport.
    if (options.cloning && (await readReceipt(dir, repo, false)) && complete(dir, false)) {
      throw new Error(
        `${repo}: the hub at ${endpoint} cannot be reached, and ${dir} holds no enrollment ` +
          `graphs (${ONNX_ENROLL.join(", ")}). Connect once to fetch them.`
      );
    }
    throw new Error(
      `${repo}: the hub at ${endpoint} cannot be reached, and ${dir} holds no verified ` +
        "release to fall back on. Connect once to fetch it."
    );
  }
  if ((await receiptHit(dir, repo, commit, options.cloning)) && complete(dir, options.cloning)) {
    return dir;
  }
  // A stale receipt must not outlive the files it vouched for.
  rmSync(join(dir, RECEIPT_NAME), { force: true });

  const tree = await listRepo(repo, revision, endpoint, token, options.signal);
  const sizes = new Map(tree.filter((e) => e.type === "file").map((e) => [e.path, listedSize(e)]));
  for (const path of sizes.keys()) {
    const why = rejectedName(path);
    if (why) {
      throw new Error(`${repo}: the listing names a file that ${why}; nothing was written`);
    }
  }
  const wanted = planDownload([...sizes.keys()], options.cloning);
  if (wanted.length === 0) {
    throw new Error(`${repo} at ${revision} holds none of the files an onnx release needs`);
  }
  refuseBeforeFetching(repo, revision, wanted);

  mkdirSync(dir, { recursive: true });
  // SHA256SUMS cannot vouch for itself and always travels; a file already
  // here that hashes to its new entry is kept, and anything else is fetched.
  let sums: Map<string, string> | null = null;
  let hashed = 0;
  for (const [index, path] of bookkeepingFirst(wanted).entries()) {
    const target = join(dir, path);
    const size = sizes.get(path) ?? -1;
    const kept = path !== SUMS && sums !== null && (await keptByHash(target, path, sums));
    if (!kept) {
      rmSync(target, { force: true });
      const step = `[${index + 1}/${wanted.length}]`;
      let announced = -1;
      await fetchFile(
        `${endpoint}/${repo}/resolve/${encodeURIComponent(revision)}/${path}`,
        target,
        path,
        token,
        (done) => {
          // A line every 16 MB rather than every chunk.
          const step16 = Math.floor(done / (16 * 1024 * 1024));
          if (step16 === announced) return;
          announced = step16;
          if (size > 0 && done < size) say(`${step} ${path}  ${human(done)} / ${human(size)}`);
        },
        size,
        options.signal
      );
      say(`${step} ${path}  ${human(statSync(target).size)}`);
    }
    if (path === SUMS) {
      sums = parseSums(target);
      continue;
    }
    if (path === RELEASE_RECORD && isOfficial(repo)) {
      await requireReleasable(dir, `${repo} (${dir})`, sums ?? new Map<string, string>());
    }
    // Counted only when something was actually hashed. The counter used to
    // rise for every file in the plan, including the ones `verifyFetched`
    // returns from without touching: a repo with no `SHA256SUMS` at all, and a
    // file the manifest does not list. So a stranger's repo that ships no
    // manifest printed "verified 12 files against SHA256SUMS" having hashed
    // none of them. `kept` counts, because `keptByHash` hashed it.
    if (kept || (await verifyFetched(dir, repo, path, sums, say))) hashed++;
  }
  if (sums === null) {
    say(`${repo} ships no ${SUMS}; nothing was verified against one`);
  } else if (hashed > 0) {
    say(`verified ${hashed} of ${wanted.length} files against ${SUMS}`);
  }
  // Checked after the hashes: a set can be intact and still short, because
  // the plan fetches whatever subset the repo holds.
  verifyReleaseInventory(dir, {
    cloning: options.cloning,
    requireVoices: true,
    requireManifest: true,
  });
  await writeReceipt(dir, repo, revision, commit);
  say(`${repo} is ready in ${dir}`);
  return dir;
}
