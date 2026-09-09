//! Getting a release, and finding your way around it once you have one.
//!
//! [`download`] fetches the ONNX file set of a Hugging Face repo, and
//! [`Bundle`] names the three paths [`crate::engine::Engine`] needs inside it.
//! Together they are the reason this crate needs no Python to reach audio.
//!
//! The file set mirrors `python/loudkit/hub.py`: one repository serves every
//! backend, so what varies is the fetch, not the layout. The ONNX set is the
//! synthesis checkpoint, the tokenizer, the two manifests, all 28 voices
//! and the six exported graphs, and with `cloning` the three enrollment
//! graphs. The torch weights, the CoreML packages and the enrollment
//! checkpoint are hundreds of megabytes this port never opens.
//!
//! Anywhere this module takes a release it takes a repo id too, by the rule
//! `python/loudkit/hub.py` states: anything that exists on disk is a path,
//! always; `org/name` is a Hugging Face repo id; anything else is a path that
//! does not exist, and the error says so.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// The synthesis checkpoint's name in a release.
pub const CHECKPOINT_NAME: &str = "loudr-1.safetensors";
/// The fusion checkpoint's canonical release name.
const TURBO_NAME: &str = "loudr-1-turbo.safetensors";
/// The enrollment checkpoint's name. This port enrols through the exported
/// graphs and never reads it, so it is fetched by nobody here.
const ENROLLMENT_NAME: &str = "loudr-1-enrollment.safetensors";
/// The utterance voice encoder, which only the torch enroller reads.
const VOICE_ENCODER_NAME: &str = "ve.safetensors";
/// Where voices live inside a release.
pub const VOICE_DIR: &str = "voices";
/// A voice file's extension.
pub const VOICE_SUFFIX: &str = ".safetensors";

/// What every backend's fetch carries. `*.safetensors` is the checkpoint and,
/// nested, every voice; the two files above are carved back out by `IGNORE`.
const CORE: &[&str] = &[
    "*.safetensors",
    "manifest.json",
    "tokenizer.json",
    "release.json",
    "voices/*",
    "SHA256SUMS",
];

/// The six graphs a synthesis run opens.
pub const ONNX_SYNTHESIS: &[&str] = &[
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_step.onnx",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/vocoder.onnx",
];

const ONNX_FUSION: &[&str] = &[
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_pair_step.onnx",
    "onnx/t3_head2.onnx",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/vocoder.onnx",
];

fn synthesis_graphs(fused: bool) -> &'static [&'static str] {
    if fused {
        ONNX_FUSION
    } else {
        ONNX_SYNTHESIS
    }
}

/// The record `tools/export_onnx.py` writes beside the graphs, saying which
/// checkpoint one export produced them from.
///
/// Fetched with them and required of no fetch.
/// [`crate::export::check_export_record`] warns on an absent one and refuses a
/// present-and-wrong one, so demanding it here would refuse the releases
/// exported before it existed, and a release that has it and leaves it on the
/// Hub makes that check dead for everyone who fetched the ordinary way.
/// `python/loudkit/hub.py` carries it in `_ONNX_SYNTHESIS` and strikes it back
/// out of the inventory check, which is the same split as [`ONNX_SYNTHESIS`]
/// and `shortfall` here.
pub const ONNX_EXPORT_RECORD: &str = "onnx/export.json";

/// The three graphs [`crate::enroll::Enroller`] opens.
pub const ONNX_ENROLL: &[&str] = &[
    "onnx/s3_tokenizer.onnx",
    "onnx/camp.onnx",
    "onnx/voice_encoder.onnx",
];

/// Two names `*.safetensors` would otherwise drag in, neither of which this
/// port can read: torch's enrollment weights and the torch voice encoder.
const IGNORE: &[&str] = &[VOICE_ENCODER_NAME, ENROLLMENT_NAME];

// --------------------------------------------------------------- the plan

/// The relative paths a fetch should take from a repo's full listing.
///
/// The same thirteen patterns `release_patterns("onnx")` selects by in
/// `python/loudkit/hub.py`, entry for entry: the six core names, the six
/// graphs and [`ONNX_EXPORT_RECORD`].
///
/// Order is the listing's, so a caller reporting progress reports it in the
/// order the repo names its files.
#[must_use]
pub fn wanted(listing: &[String], cloning: bool) -> Vec<String> {
    let mut allow: Vec<&str> = CORE.to_vec();
    allow.extend_from_slice(ONNX_SYNTHESIS);
    allow.extend_from_slice(ONNX_FUSION);
    allow.push(ONNX_EXPORT_RECORD);
    if cloning {
        allow.extend_from_slice(ONNX_ENROLL);
    }
    listing
        .iter()
        .filter(|name| allow.iter().any(|p| glob(p, name)) && !IGNORE.iter().any(|p| glob(p, name)))
        .cloned()
        .collect()
}

/// `fnmatch` with `*` and `?` and nothing else, matching Python's, where `*`
/// crosses `/`, which is why `*.safetensors` brings the voices along.
fn glob(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    // Iterative backtracking rather than recursion: the patterns are untrusted
    // only in the sense that they are constants here, but a recursive matcher
    // on a long name is a stack the caller cannot bound.
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while j < n.len() {
        if i < p.len() && (p[i] == '?' || p[i] == n[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = i;
            mark = j;
            i += 1;
        } else if star != usize::MAX {
            i = star + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    p[i..].iter().all(|&c| c == '*')
}

/// Everything the fetch promised that is not on disk under `root`.
///
/// `allow_patterns` is a request, not a receipt: a repo missing its `onnx/`
/// graphs answers a pattern fetch with the same directory as one that has
/// them. The mirror of `verify_release_inventory` in `python/loudkit/hub.py`,
/// with the ONNX set it is the only one this port can run.
#[cfg(feature = "download")]
fn shortfall(root: &Path, cloning: bool) -> Vec<String> {
    let checkpoint = checkpoint_in(root).unwrap_or_else(|_| root.join(CHECKPOINT_NAME));
    let fused = header_manifest(&checkpoint).is_some_and(|m| m["decode"]["mode"] == "fusion_mtp2");
    let mut expected: Vec<String> = vec![
        checkpoint
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        "manifest.json".to_string(),
        "tokenizer.json".to_string(),
    ];
    expected.extend(synthesis_graphs(fused).iter().map(ToString::to_string));
    if cloning {
        expected.extend(ONNX_ENROLL.iter().map(ToString::to_string));
    }
    let mut missing: Vec<String> = expected
        .into_iter()
        .filter(|rel| !root.join(rel).is_file())
        .collect();
    if voice_names(root).is_empty() {
        missing.push(format!("{VOICE_DIR}/*{VOICE_SUFFIX}"));
    }
    missing
}

/// The voices under `root`, by bare name, sorted. Empty when there are none.
fn voice_names(root: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(root.join(VOICE_DIR)) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_file())
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            name.strip_suffix(VOICE_SUFFIX).map(ToString::to_string)
        })
        .filter(|name| !name.starts_with('.'))
        .collect();
    names.sort();
    names
}

// ---------------------------------------------------------- the resolver

/// Whether `name` is a Hugging Face repo id rather than a path.
///
/// `org/name`, one separator, and nothing that is secretly a path: anything
/// that exists on disk is a path, and so is anything path-shaped. The same
/// rule as `loudkit.hub.is_repo_id`, pinned to the shared fixture.
#[must_use]
pub fn is_repo_id(name: &str) -> bool {
    if name.is_empty() || Path::new(name).exists() {
        return false;
    }
    if name.starts_with(['.', '/', '~']) || name.ends_with(VOICE_SUFFIX) {
        return false;
    }
    let parts: Vec<&str> = name.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        })
}

/// Where a repo id fetched by name lands, so a second project on this machine
/// shares one copy of a 747 MB file rather than filling a working directory
/// with its own: [`cache_path`] under the platform's user cache, or
/// `$LOUDKIT_CACHE/<org>--<name>` when that variable is set. The same layout
/// in Go, JS and Swift, pinned by `tests/data/conformance/cache_path.json`,
/// so a cache one port wrote is a cache the other three read.
#[must_use]
pub fn cache_dir(repo: &str) -> PathBuf {
    if let Some(dir) = std::env::var_os("LOUDKIT_CACHE") {
        return PathBuf::from(dir).join(cache_slug(repo));
    }
    cache_path(&user_cache_root(), repo)
}

/// `<root>/loudkit/<org>--<name>`.
#[must_use]
pub fn cache_path(root: &Path, repo: &str) -> PathBuf {
    root.join("loudkit").join(cache_slug(repo))
}

fn cache_slug(repo: &str) -> String {
    repo.replace('/', "--")
}

/// The platform's user cache: `~/Library/Caches` on macOS, `%LocalAppData%`
/// on Windows, `$XDG_CACHE_HOME` or `~/.cache` elsewhere, and `.cache` beside
/// nothing at all when even `HOME` is unset, which is a container, where a
/// relative directory beats writing to `/`.
fn user_cache_root() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from);
    if cfg!(target_os = "macos") {
        if let Some(home) = home {
            return home.join("Library").join("Caches");
        }
    }
    if cfg!(target_os = "windows") {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(dir);
        }
    }
    if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir);
    }
    home.map_or_else(|| PathBuf::from(".cache"), |h| h.join(".cache"))
}

/// Whether `repo` names a turbo model: its last segment ends in `-turbo`.
///
/// Decided by name, before any Hub call, so a repo that is private or
/// unpublished today still gets the right sentence. This port runs the turbo
/// decoder itself, through `decode.mode = "fusion_mtp2"`, so the predicate no
/// longer marks a repository as out of reach and has no caller in the crate.
#[must_use]
pub fn is_turbo_repo(repo: &str) -> bool {
    repo.rsplit('/')
        .next()
        .is_some_and(|name| name.ends_with("-turbo"))
}

/// The manifest embedded in a checkpoint, from its header alone: the file is
/// 747 MB and this runs while deciding whether to open it at all.
fn header_manifest(path: &Path) -> Option<serde_json::Value> {
    use std::io::Read;

    let mut file = fs::File::open(path).ok()?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).ok()?;
    let size = usize::try_from(u64::from_le_bytes(length)).ok()?;
    if size == 0 || size > 64 << 20 {
        return None;
    }
    let mut body = vec![0u8; size];
    file.read_exact(&mut body).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&body).ok()?;
    serde_json::from_str(header["__metadata__"]["manifest"].as_str()?).ok()
}

/// A release directory for `dir_or_repo`, fetching it by name if need be.
///
/// The rule is `python/loudkit/hub.py`'s, word for word, because a caller who
/// learns it in one language should not have to learn it again in the next:
///
/// * anything that exists on disk is a path, always. A local directory wins,
///   so nothing reaches for the network because a directory happened to be
///   named like a repo.
/// * `org/name` is a Hugging Face repo id. It resolves under [`cache_dir`]
///   through [`download`], whose receipt makes the second load one cheap
///   call and no fetch.
/// * anything else is a path that does not exist, and the error says so.
///
/// # Errors
///
/// Returns an error for a name that is neither, for a repo id when the
/// `download` feature is off, and everything [`download`] returns.
pub fn resolve(dir_or_repo: impl AsRef<Path>) -> Result<PathBuf, String> {
    let given = dir_or_repo.as_ref();
    if given.is_dir() {
        return Ok(given.to_path_buf());
    }
    let name = given.to_string_lossy();
    if !is_repo_id(&name) {
        return Err(format!(
            "{name}: not a directory, and not a Hugging Face repo id either. \
             Pass a release directory, or an id such as \"loudreader/loudr-1\"."
        ));
    }
    #[cfg(not(feature = "download"))]
    {
        Err(format!(
            "{name} is a repo id, and this build has the `download` feature \
             off. Fetch the release some other way and pass its directory."
        ))
    }
    #[cfg(feature = "download")]
    {
        download(&name, cache_dir(&name))
    }
}

// ------------------------------------------------------------- the bundle

/// A downloaded release, with the paths an engine is built from.
///
/// The point of it: a caller passes one directory instead of three paths, and
/// the rules for which file is which live here rather than in every reader's
/// head.
pub struct Bundle {
    /// The release directory itself.
    pub root: PathBuf,
    /// The synthesis checkpoint.
    pub checkpoint: PathBuf,
    /// The directory holding the exported graphs.
    pub onnx_dir: PathBuf,
    /// The text tokenizer.
    pub tokenizer: PathBuf,
    /// The repo id the release was fetched by, so [`Bundle::fetch_cloning`]
    /// can fetch the rest of it into `root`. `None` for a directory of your
    /// own.
    pub repo: Option<String>,
}

impl Bundle {
    /// Find the checkpoint, the graphs and the tokenizer inside a release.
    ///
    /// `dir_or_repo` is a release directory, or a repo id such as
    /// `"loudreader/loudr-1"`, which is fetched into [`cache_dir`] once and
    /// read from there afterwards. See [`resolve`].
    ///
    /// The checkpoint is [`CHECKPOINT_NAME`] when it is there, and otherwise
    /// the single root-level `*.safetensors` that is neither the voice encoder
    /// nor the enrollment half. Voices are excluded by living in `voices/`,
    /// which is why the search is not recursive.
    ///
    /// Resolution is by name, not by reading each candidate's manifest as
    /// Python does: the manifest is metadata inside a 747 MB safetensors file,
    /// and opening every candidate to ask what it is would read the release
    /// twice before the engine reads it once.
    ///
    /// # Errors
    ///
    /// Returns an error naming what is absent, or naming the candidates when
    /// more than one file could be the checkpoint.
    pub fn open(dir_or_repo: impl AsRef<Path>) -> Result<Bundle, String> {
        let given = dir_or_repo.as_ref();
        let root = resolve(given)?;
        // `resolve` answers for a directory or a repo id and nothing else.
        let repo = (!given.is_dir()).then(|| given.to_string_lossy().into_owned());
        let checkpoint = checkpoint_in(&root)?;
        let onnx_dir = root.join("onnx");
        if !onnx_dir.is_dir() {
            return Err(format!(
                "{}: no onnx/ directory. This is a release fetched without the \
                 exported graphs; fetch it again with `loudkit::download`.",
                root.display()
            ));
        }
        let tokenizer = root.join("tokenizer.json");
        if !tokenizer.is_file() {
            return Err(format!(
                "{}: no tokenizer.json beside the checkpoint. The engine cannot \
                 be built without it.",
                root.display()
            ));
        }
        let fused =
            header_manifest(&checkpoint).is_some_and(|m| m["decode"]["mode"] == "fusion_mtp2");
        for graph in synthesis_graphs(fused) {
            if !root.join(graph).is_file() {
                return Err(format!("{}: missing {graph}", root.display()));
            }
        }
        Ok(Bundle {
            root,
            checkpoint,
            onnx_dir,
            tokenizer,
            repo,
        })
    }

    /// The voices this release ships, by bare name, sorted.
    #[must_use]
    pub fn voices(&self) -> Vec<String> {
        voice_names(&self.root)
    }

    /// The path of one voice, by bare name (`"joe"`) or filename.
    ///
    /// # Errors
    ///
    /// Returns an error listing the voices there are, when the name is not one
    /// of them. A name is a name: anything with a path separator in it is
    /// refused rather than joined, so a voice cannot address a file outside
    /// the release.
    pub fn voice_path(&self, name: &str) -> Result<PathBuf, String> {
        let file = if name.ends_with(VOICE_SUFFIX) {
            name.to_string()
        } else {
            format!("{name}{VOICE_SUFFIX}")
        };
        if file.starts_with('.') || file.contains('/') || file.contains('\\') {
            return Err(format!(
                "{name}: a voice is named, not addressed. Pass a bare name such \
                 as \"joe\", or load a path with `voice::load`."
            ));
        }
        let path = self.root.join(VOICE_DIR).join(&file);
        if path.is_file() {
            return Ok(path);
        }
        let have = self.voices();
        Err(format!(
            "{name}: no such voice in {}. This release ships: {}",
            self.root.join(VOICE_DIR).display(),
            if have.is_empty() {
                "none".to_string()
            } else {
                have.join(", ")
            }
        ))
    }
}

/// What `manifest["artifact_role"]` says about each half of a split release.
const SYNTHESIS_ROLE: &str = "synthesis";
const ENROLLMENT_ROLE: &str = "enrollment";

/// `manifest["artifact_role"]` for a checkpoint, or `None` for a file that
/// makes no claim.
///
/// `None` covers both a pre-split checkpoint, whose manifest predates the
/// field and which does carry every tensor, and a file this cannot read at
/// all. The field is only ever read to refuse, so a missing claim never
/// promotes a file to a role it did not ask for.
///
/// The header alone: this runs while deciding which of two files to open, and
/// a checkpoint is hundreds of megabytes.
fn artifact_role(path: &Path) -> Option<String> {
    use std::io::Read;

    let mut file = fs::File::open(path).ok()?;
    let mut length = [0u8; 8];
    file.read_exact(&mut length).ok()?;
    let size = usize::try_from(u64::from_le_bytes(length)).ok()?;
    // A header is JSON describing tensors; anything past this is not one, and
    // reading it would be the whole-file read this exists to avoid.
    if size > 8 * 1024 * 1024 {
        return None;
    }
    let mut header = vec![0u8; size];
    file.read_exact(&mut header).ok()?;
    let doc: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let manifest: serde_json::Value =
        serde_json::from_str(doc["__metadata__"]["manifest"].as_str()?).ok()?;
    manifest["artifact_role"].as_str().map(ToString::to_string)
}

/// Refuse a file whose manifest declares it to be the other artefact.
fn refuse_role(path: &Path, expected: &str) -> Result<(), String> {
    match artifact_role(path) {
        Some(role) if role != expected => Err(format!(
            "{}: this is a release's {role} artefact, and the {expected} artefact \
             is what was asked for. Pass the release directory, or the repo id, \
             and let the resolver pick.",
            path.display()
        )),
        _ => Ok(()),
    }
}

/// The synthesis checkpoint in `dir`, or a useful complaint.
///
/// Three rules, in order: exactly one canonical name; otherwise the file that
/// declares the synthesis role; otherwise exactly one candidate, with anything
/// declaring the enrollment role set aside first.
fn checkpoint_in(dir: &Path) -> Result<PathBuf, String> {
    let named = dir.join(CHECKPOINT_NAME);
    if named.is_file() && !dir.join(TURBO_NAME).is_file() {
        refuse_role(&named, SYNTHESIS_ROLE)?;
        return Ok(named);
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return Err(format!("{}: cannot be read", dir.display()));
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            name.ends_with(VOICE_SUFFIX)
                && !name.starts_with('.')
                && name != VOICE_ENCODER_NAME
                && name != ENROLLMENT_NAME
        })
        .collect();
    found.sort();
    let roles: Vec<Option<String>> = found.iter().map(|p| artifact_role(p)).collect();
    let declared: Vec<PathBuf> = found
        .iter()
        .zip(&roles)
        .filter(|(_, role)| role.as_deref() == Some(SYNTHESIS_ROLE))
        .map(|(path, _)| path.clone())
        .collect();
    let mut candidates: Vec<PathBuf> = if declared.is_empty() {
        found
            .iter()
            .zip(&roles)
            .filter(|(_, role)| role.as_deref() != Some(ENROLLMENT_ROLE))
            .map(|(path, _)| path.clone())
            .collect()
    } else {
        declared
    };
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }
    if found.is_empty() {
        return Err(format!(
            "{}: no {CHECKPOINT_NAME} here, and no other checkpoint beside it. A \
             loudkit release is a synthesis checkpoint, a tokenizer, an onnx/ \
             directory and a voices/ directory.",
            dir.display()
        ));
    }
    if candidates.is_empty() {
        return Err(format!(
            "{}: the only checkpoint here is a release's enrollment artefact. It \
             carries the two modules a clone needs and nothing synthesis reads. \
             Fetch the release's {CHECKPOINT_NAME} beside it.",
            dir.display()
        ));
    }
    // `Engine::load_paths`, because `Engine::load` is the call that reached
    // here: sending the caller back to it names the failure as its own remedy.
    // `load_paths` is the one that takes a named checkpoint.
    Err(format!(
        "{}: {} checkpoints ({}). Name the one you mean and build the engine \
         with `Engine::load_paths`.",
        dir.display(),
        candidates.len(),
        candidates
            .iter()
            .map(|p| p
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned())
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

// ------------------------------------------------------------ the receipt

/// What a verified download directory carries, in every port: the repo and
/// revision as asked, the commit the Hub resolved, the digest of `SHA256SUMS`
/// (`null` when the release ships none) and when it was fetched, in UTC.
pub const RECEIPT_NAME: &str = ".loudkit-release.json";

/// The most a receipt file is read: five short fields. A file past it is
/// not a receipt, and is not read.
const RECEIPT_LIMIT: u64 = 1 << 20;

/// The receipt's keys, in the order [`write_receipt`] writes them.
pub const RECEIPT_FIELDS: [&str; 5] = ["repo", "revision", "commit", "sha256sums", "fetched_at"];

/// A receipt [`read_receipt`] accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub repo: String,
    /// As asked; `main` when the caller named none.
    pub revision: String,
    pub commit: String,
    /// The digest of `SHA256SUMS`, or `None` for a release without one.
    pub sha256sums: Option<String>,
    /// UTC, to the second.
    pub fetched_at: String,
}

/// The receipt under `root` when it vouches for `repo`, else `None`: every
/// field present with its type (`null` is the wrong type for all but
/// `sha256sums`; keys it does not know are ignored), `repo` the one asked,
/// `commit` forty lowercase hex, `sha256sums` the digest of the `SHA256SUMS`
/// on disk (`None` for none), and every file it lists that the plan selects
/// present. No weight is hashed: the receipt is checked, not the release.
/// This is the only place the file is read.
#[must_use]
pub fn read_receipt(root: &Path, repo: &str, cloning: bool) -> Option<Receipt> {
    let path = root.join(RECEIPT_NAME);
    if fs::metadata(&path).ok()?.len() > RECEIPT_LIMIT {
        return None;
    }
    let text = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let record = value.as_object()?;
    let string = |key: &str| record.get(key)?.as_str().map(str::to_string);
    let sha256sums = match record.get("sha256sums")? {
        serde_json::Value::Null => None,
        serde_json::Value::String(digest) => Some(digest.clone()),
        _ => return None,
    };
    let receipt = Receipt {
        repo: string("repo")?,
        revision: string("revision")?,
        commit: string("commit")?,
        sha256sums,
        fetched_at: string("fetched_at")?,
    };
    let forty_hex = receipt.commit.len() == 40
        && receipt
            .commit
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if receipt.repo != repo || !forty_hex {
        return None;
    }
    let sums = root.join(SUMS_NAME);
    if !sums.is_file() {
        return receipt.sha256sums.is_none().then_some(receipt);
    }
    let digest = file_sha256(&sums).ok()?;
    if receipt.sha256sums.as_deref() != Some(digest.as_str()) {
        return None;
    }
    let listing = fs::read_to_string(&sums).ok()?;
    let listed = parse_sums(&listing, &sums.display().to_string()).ok()?;
    let names: Vec<String> = listed.into_keys().collect();
    wanted(&names, cloning)
        .iter()
        .all(|name| root.join(name).is_file())
        .then_some(receipt)
}

/// Record that `root` holds `repo` at `commit`, verified.
///
/// Laid out by hand rather than through a map, so the keys land in
/// [`RECEIPT_FIELDS`] order in every port's file.
///
/// # Errors
///
/// Returns an error when `SHA256SUMS` or the receipt cannot be written or read.
pub fn write_receipt(root: &Path, repo: &str, revision: &str, commit: &str) -> Result<(), String> {
    let sums = root.join(SUMS_NAME);
    let digest = if sums.is_file() {
        serde_json::Value::String(file_sha256(&sums)?)
    } else {
        serde_json::Value::Null
    };
    let values = [
        serde_json::Value::String(repo.to_string()),
        serde_json::Value::String(revision.to_string()),
        serde_json::Value::String(commit.to_string()),
        digest,
        serde_json::Value::String(utc_now()),
    ];
    let lines: Vec<String> = RECEIPT_FIELDS
        .iter()
        .zip(&values)
        .map(|(key, value)| format!(" \"{key}\": {value}"))
        .collect();
    let path = root.join(RECEIPT_NAME);
    fs::write(&path, format!("{{\n{}\n}}\n", lines.join(",\n")))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Whether `root` already holds `repo` at `commit`: a receipt
/// [`read_receipt`] accepts, naming that commit. An empty commit never hits:
/// nothing was resolved.
#[must_use]
pub fn receipt_hit(root: &Path, repo: &str, commit: &str, cloning: bool) -> bool {
    !commit.is_empty() && read_receipt(root, repo, cloning).is_some_and(|r| r.commit == commit)
}

/// Now, as `YYYY-MM-DDTHH:MM:SSZ`. The civil date is derived here because a
/// calendar crate for one line would be the receipt's only dependency.
fn utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (year, month, day) = civil_from_days(secs / 86_400);
    let clock = secs % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        clock / 3600,
        clock % 3600 / 60,
        clock % 60
    )
}

/// The proleptic Gregorian date `days` after 1970-01-01, by the era
/// arithmetic in Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

// ----------------------------------------------------------- the download

/// The org whose releases this crate vouches for: the only case where a
/// missing `SHA256SUMS` or `release.json` is a release that lost files rather
/// than a stranger's bare upload.
pub const OFFICIAL_ORG: &str = "loudreader";

/// The profiles a builder stamps on a releasable bundle: `full-0.1` for
/// loudr-1 and `turbo-0.1` for loudr-1-turbo. Neither is written until the
/// builder's load-and-speak gate passes.
pub const STRICT_PROFILES: &[&str] = &["full-0.1", "turbo-0.1"];

const SUMS_NAME: &str = "SHA256SUMS";
#[cfg(feature = "download")]
const RELEASE_RECORD: &str = "release.json";

/// Whether `repo` is one this project publishes.
#[must_use]
pub fn is_official(repo: &str) -> bool {
    repo.split_once('/')
        .is_some_and(|(org, _)| org.eq_ignore_ascii_case(OFFICIAL_ORG))
}

/// Whether the release under `root` carries the three enrollment graphs.
#[must_use]
pub fn can_enroll(root: &Path) -> bool {
    ONNX_ENROLL.iter().all(|g| root.join(g).is_file())
}

impl Bundle {
    /// Whether this release carries the three enrollment graphs, which only a
    /// fetch with `cloning` brings.
    #[must_use]
    pub fn can_enroll(&self) -> bool {
        can_enroll(&self.root)
    }

    /// Make sure the release holds the three enrollment graphs. A release
    /// opened by repo id fetches them into its own directory, through the
    /// same receipt-aware [`download_with`], so the cache grows to the wider
    /// set once and stays a hit afterwards.
    ///
    /// # Errors
    ///
    /// For a directory of your own without the graphs, naming the fetch that
    /// brings them; a build with the `download` feature off; and everything
    /// [`download_with`] returns.
    pub fn fetch_cloning(&self) -> Result<(), String> {
        #[cfg(feature = "download")]
        {
            self.fetch_cloning_with(&Options::default())
        }
        #[cfg(not(feature = "download"))]
        {
            if self.can_enroll() {
                return Ok(());
            }
            Err(self.no_cloning())
        }
    }

    /// [`Bundle::fetch_cloning`] with the revision or the mirror named.
    ///
    /// # Errors
    ///
    /// The same as [`Bundle::fetch_cloning`].
    #[cfg(feature = "download")]
    pub fn fetch_cloning_with(&self, options: &Options) -> Result<(), String> {
        if self.can_enroll() {
            return Ok(());
        }
        let Some(repo) = &self.repo else {
            return Err(self.no_cloning());
        };
        let options = Options {
            revision: options.revision.clone(),
            cloning: true,
            endpoint: options.endpoint.clone(),
        };
        download_with(repo, &self.root, &options).map(|_| ())
    }

    fn no_cloning(&self) -> String {
        format!(
            "{} carries no enrollment graphs. Fetch them with \
             `hub::download_with(repo, dir, &hub::Options {{ cloning: true, ..Default::default() }})`",
            self.root.display()
        )
    }
}

/// Why `name` cannot address bytes inside the release, or `None` when it can.
#[must_use]
pub fn rejected_name(name: &str) -> Option<&'static str> {
    if name.contains('\\') {
        return Some("is not a POSIX path (it contains a backslash)");
    }
    let drive = name.len() >= 3 && {
        let b = name.as_bytes();
        b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'/'
    };
    if name.starts_with('/') || drive {
        return Some("is absolute, and a manifest name is relative to the release root");
    }
    if name.split('/').any(|p| p == "..") {
        return Some("escapes the release root with '..'");
    }
    if name.is_empty() || name.split('/').any(|p| p.is_empty() || p == ".") {
        return Some("is not normalised (an empty or '.' path component)");
    }
    None
}

/// Parse a `SHA256SUMS`, refusing a line it cannot understand, a name that
/// escapes the release and a name listed twice.
///
/// # Errors
///
/// For every shape named above, with the line number.
pub fn parse_sums(text: &str, origin: &str) -> Result<HashMap<String, String>, String> {
    let mut out = HashMap::new();
    for (number, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let (digest, name) = line
            .split_once("  ")
            .filter(|(d, n)| {
                d.len() == 64
                    && d.chars()
                        .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
                    && !n.is_empty()
                    && !n.starts_with(' ')
            })
            .ok_or_else(|| {
                format!(
                    "{origin}: malformed line {}: {line:?}; this does not look like a \
                     loudkit release manifest, refusing to verify against it",
                    number + 1
                )
            })?;
        if let Some(why) = rejected_name(name) {
            return Err(format!(
                "{origin}: line {}: {name:?} {why}; refusing to verify against a \
                 manifest that names files outside the release it describes",
                number + 1
            ));
        }
        if out.insert(name.to_string(), digest.to_string()).is_some() {
            return Err(format!(
                "{origin}: line {}: duplicate entry for {name:?}; the manifest \
                 disagrees with itself about one file. Rebuild the release",
                number + 1
            ));
        }
    }
    if out.is_empty() {
        return Err(format!("{origin}: no checksum entries"));
    }
    Ok(out)
}

/// What to fetch, and from where.
#[cfg(feature = "download")]
pub struct Options {
    /// Branch, tag or commit. Pin it for anything reproducible: a moving
    /// `main` is a moving model.
    pub revision: String,
    /// Add the three enrollment graphs, for cloning a voice from a recording.
    pub cloning: bool,
    /// The Hugging Face endpoint. `HF_ENDPOINT` when set, for mirrors.
    pub endpoint: String,
}

#[cfg(feature = "download")]
impl Default for Options {
    fn default() -> Self {
        Options {
            revision: "main".to_string(),
            cloning: false,
            endpoint: std::env::var("HF_ENDPOINT")
                .unwrap_or_else(|_| "https://huggingface.co".to_string()),
        }
    }
}

/// A caller's switch for stopping a fetch that is already running.
///
/// A release is hundreds of megabytes over a link this module sets no deadline
/// on, so without one the fetch runs to its own end whatever the caller
/// decided: a server shutting down, a user closing the window, a request that
/// timed out three minutes ago. There was no way to say stop.
///
/// [`Clone`] hands the same switch to another thread, which is the only way it
/// is useful: the thread doing the fetch is inside [`download_cancellable`]
/// until it returns.
///
/// `go/hub.go`'s `DownloadContext` is the same seam under a `context.Context`;
/// this port has no ambient context type, so the switch is the argument.
///
/// ```no_run
/// let cancel = loudkit::hub::Cancel::new();
/// let switch = cancel.clone();
/// std::thread::spawn(move || switch.cancel());
/// let options = loudkit::hub::Options::default();
/// let _ = loudkit::hub::download_cancellable("loudreader/loudr-1", "loudr-1", &options, &cancel);
/// ```
#[cfg(feature = "download")]
#[derive(Clone, Default)]
pub struct Cancel(std::sync::Arc<std::sync::atomic::AtomicBool>);

#[cfg(feature = "download")]
impl Cancel {
    /// A switch nobody has flipped yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask the fetch to stop. Callable from any thread, and from more than
    /// one; flipping a switch that is already flipped does nothing.
    pub fn cancel(&self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the switch has been flipped.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// The refusal a cancelled fetch answers with, so every site says the same
    /// sentence and a caller can match on it.
    fn stop(&self, repo: &str) -> Result<(), String> {
        if self.is_cancelled() {
            return Err(format!("{repo}: the download was cancelled"));
        }
        Ok(())
    }
}

/// Fetch a release's ONNX file set into `dir`, and answer with `dir`.
///
/// One Hub call resolves the revision to a commit. A `dir` whose
/// [`RECEIPT_NAME`] passes [`read_receipt`] and names that commit, and
/// which still holds the set, is answered as it stands, with no fetch and no
/// weight hashed; a Hub that cannot be reached is answered by such a
/// receipt, and says so. Otherwise `SHA256SUMS` is fetched anew, a file
/// already on disk is kept when it matches it and refetched when it does not,
/// and every file that arrives is hashed against it. The two bookkeeping
/// files come first, so a repo that is not a release is refused before its
/// weights move. Progress goes to stderr.
///
/// ```no_run
/// let dir = loudkit::download("loudreader/loudr-1", "loudr-1")?;
/// let engine = loudkit::engine::Engine::load(&dir)?;
/// # Ok::<_, String>(())
/// ```
///
/// # Errors
///
/// Returns an error when the repo id is malformed, when the repo cannot be
/// reached or does not exist, when it is not a release this port can run, and
/// when what arrived does not add up to a usable set.
#[cfg(feature = "download")]
pub fn download(repo: &str, dir: impl AsRef<Path>) -> Result<PathBuf, String> {
    download_with(repo, dir, &Options::default())
}

/// [`download`] with a revision, the enrollment graphs, or a mirror.
///
/// # Errors
///
/// The same as [`download`].
#[cfg(feature = "download")]
pub fn download_with(
    repo: &str,
    dir: impl AsRef<Path>,
    options: &Options,
) -> Result<PathBuf, String> {
    download_cancellable(repo, dir, options, &Cancel::new())
}

/// [`download_with`] under a caller's [`Cancel`], which is the form to reach
/// for inside a server.
///
/// Cancelling stops the transfer between reads and answers "the download was
/// cancelled"; the part-file is removed, so the next call starts that file
/// again rather than resuming a short one. A cancelled fetch is never mistaken
/// for a hub that cannot be reached, so it does not fall back to the receipt:
/// a caller who stopped the fetch is told they stopped it.
///
/// Beside [`download_with`] rather than inside its signature: the switch is
/// what a server needs and a script never does, and every existing caller
/// keeps compiling.
///
/// # Errors
///
/// The same as [`download`], plus the cancellation.
#[cfg(feature = "download")]
pub fn download_cancellable(
    repo: &str,
    dir: impl AsRef<Path>,
    options: &Options,
    cancel: &Cancel,
) -> Result<PathBuf, String> {
    let root = dir.as_ref().to_path_buf();
    check_repo_id(repo)?;
    let base = options.endpoint.trim_end_matches('/');
    let revision = &options.revision;

    cancel.stop(repo)?;
    eprintln!("loudkit: resolving {repo} at {revision}");
    let complete = || shortfall(&root, options.cloning).is_empty();
    let (commit, listing) = match revision_info(base, repo, revision) {
        Ok(answer) => answer,
        Err(HubError::Unreachable(why)) => {
            // A caller who cancelled is told so, rather than being handed a
            // receipt for an older commit as if the network had failed.
            cancel.stop(repo)?;
            let receipt = read_receipt(&root, repo, options.cloning).filter(|_| complete());
            let Some(receipt) = receipt else {
                // A receipt for the synthesis set does not cover an
                // enrollment: the sentence names the graphs, not the transport.
                if options.cloning
                    && read_receipt(&root, repo, false).is_some()
                    && shortfall(&root, false).is_empty()
                {
                    return Err(format!(
                        "{repo}: the hub cannot be reached, and {} holds no enrollment \
                         graphs ({}). Connect once to fetch them.",
                        root.display(),
                        ONNX_ENROLL.join(", ")
                    ));
                }
                return Err(why);
            };
            eprintln!(
                "loudkit: the hub cannot be reached; using {}, which holds {} at {} \
                 (fetched {})",
                root.display(),
                receipt.repo,
                receipt.commit,
                receipt.fetched_at
            );
            return Ok(root);
        }
        Err(HubError::Answered(why)) => return Err(why),
    };
    if receipt_hit(&root, repo, &commit, options.cloning) && complete() {
        return Ok(root);
    }
    // A stale receipt must not outlive the files it vouched for.
    let _ = fs::remove_file(root.join(RECEIPT_NAME));

    for name in &listing {
        if let Some(why) = rejected_name(name) {
            return Err(format!(
                "{repo}: the listing names a file that {why}; nothing was written"
            ));
        }
    }
    let files = wanted(&listing, options.cloning);
    if files.is_empty() {
        return Err(format!(
            "{repo}: the repo holds none of the files a loudkit release is made \
             of. Is this a loudkit model?"
        ));
    }
    refuse_before_fetching(repo, revision, &files)?;

    fs::create_dir_all(&root).map_err(|e| format!("{}: {e}", root.display()))?;
    let official = is_official(repo);
    let mut sums: Option<HashMap<String, String>> = None;
    let mut hashed = 0usize;
    let total = files.len();
    for (i, name) in bookkeeping_first(&files).iter().enumerate() {
        cancel.stop(repo)?;
        let dest = root.join(name);
        // The manifest cannot vouch for itself, so it is always fetched anew;
        // anything else on disk stays only if the new manifest agrees with it.
        let kept = name != SUMS_NAME && dest.is_file() && listed_as(&dest, name, sums.as_ref())?;
        if kept {
            eprintln!("loudkit: [{}/{total}] {name} (have it)", i + 1);
        } else {
            if dest.exists() {
                fs::remove_file(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
            }
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            eprint!("loudkit: [{}/{total}] {name} ", i + 1);
            let bytes = fetch_to(
                &format!("{base}/{repo}/resolve/{revision}/{name}"),
                &dest,
                repo,
                cancel,
            )?;
            eprintln!("{}", human(bytes));
        }
        if name == SUMS_NAME {
            let text = fs::read_to_string(&dest).map_err(|e| format!("{}: {e}", dest.display()))?;
            sums = Some(parse_sums(&text, &dest.display().to_string())?);
            continue;
        }
        if name == RELEASE_RECORD && official {
            require_releasable(&root, repo, sums.as_ref().unwrap_or(&HashMap::new()))?;
        }
        // Counted only when a digest was actually compared, because the
        // count is printed as a claim about what was checked. `kept` means
        // `listed_as` hashed the file and matched it against the manifest;
        // `verify_fetched` says whether it did the same for a fetched one, and
        // it says no when the repo ships no manifest and when the manifest does
        // not list this file. An unconditional counter would say "verified N
        // files against SHA256SUMS" over a run in which nothing was hashed.
        if kept || verify_fetched(&root, repo, name, sums.as_ref())? {
            hashed += 1;
        }
    }
    if hashed > 0 {
        eprintln!("loudkit: verified {hashed} files against {SUMS_NAME}");
    }
    let missing = shortfall(&root, options.cloning);
    if !missing.is_empty() {
        return Err(format!(
            "{}: this fetch does not add up to a usable set, missing: {}. The \
             release does not carry these files, or the fetch was interrupted; \
             run it again, or pin a revision that ships them.",
            root.display(),
            missing.join(", ")
        ));
    }
    write_receipt(&root, repo, revision, &commit)?;
    eprintln!("loudkit: {repo} ready in {}", root.display());
    Ok(root)
}

/// Whether the file at `dest` is the one the manifest lists as `name`. No
/// manifest lists nothing, so without one every file is fetched again.
#[cfg(feature = "download")]
fn listed_as(
    dest: &Path,
    name: &str,
    sums: Option<&HashMap<String, String>>,
) -> Result<bool, String> {
    match sums.and_then(|s| s.get(name)) {
        Some(want) => Ok(&file_sha256(dest)? == want),
        None => Ok(false),
    }
}

/// What the listing alone can say about a repo, before a byte moves: an
/// official repo must ship the two bookkeeping files, and a repo without the
/// onnx graphs cannot run here whatever else it carries.
#[cfg(feature = "download")]
fn refuse_before_fetching(repo: &str, revision: &str, files: &[String]) -> Result<(), String> {
    let has = |name: &str| files.iter().any(|f| f == name);
    if is_official(repo) {
        for name in [SUMS_NAME, RELEASE_RECORD] {
            if !has(name) {
                return Err(format!(
                    "{repo} at {revision}: no {name}. Every {OFFICIAL_ORG} release ships \
                     one, so this cannot be checked against anything and will not be \
                     fetched. Pin a revision you trust."
                ));
            }
        }
    }
    let fused = has("onnx/t3_pair_step.onnx");
    for graph in synthesis_graphs(fused) {
        if has(graph) {
            continue;
        }
        return Err(format!(
            "{repo} at {revision} ships no {graph}, which this port runs on. Use \
             loudreader/loudr-1, or pin a revision that carries the graphs."
        ));
    }
    Ok(())
}

/// The two bookkeeping files ahead of everything else, the rest in order.
#[cfg(feature = "download")]
fn bookkeeping_first(files: &[String]) -> Vec<String> {
    let rank = |name: &str| match name {
        SUMS_NAME => 0,
        RELEASE_RECORD => 1,
        _ => 2,
    };
    let mut out = files.to_vec();
    out.sort_by_key(|name| rank(name));
    out
}

/// An official repo's `release.json` must name a release profile and record
/// `verified: true`, and be listed by the manifest that vouches for it.
#[cfg(feature = "download")]
fn require_releasable(
    root: &Path,
    repo: &str,
    sums: &HashMap<String, String>,
) -> Result<(), String> {
    let record = root.join(RELEASE_RECORD);
    if !record.is_file() {
        return Err(format!(
            "{repo} ({}): no release.json. Every {OFFICIAL_ORG} release records its \
             profile and its verified flag there, so this download cannot prove it \
             is a release. Pin a revision you trust.",
            root.display()
        ));
    }
    let Some(want) = sums.get(RELEASE_RECORD) else {
        return Err(format!(
            "{repo}: {SUMS_NAME} does not list release.json, so the record that \
             would vouch for this release is vouched for by nothing. Pin a revision \
             you trust."
        ));
    };
    if &file_sha256(&record)? != want {
        return Err(format!(
            "{}: release.json failed the release checksum. Delete the directory \
             and download again.",
            record.display()
        ));
    }
    let body = fs::read_to_string(&record).map_err(|e| format!("{}: {e}", record.display()))?;
    let claim: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
        format!(
            "{}: release.json is unreadable ({e}). Delete the directory and download again.",
            record.display()
        )
    })?;
    let profile = claim["profile"].as_str().unwrap_or("");
    if !STRICT_PROFILES.contains(&profile) {
        return Err(format!(
            "{}: release.json says profile {profile:?}, and a {OFFICIAL_ORG} release is \
             one of {}. This is a development bundle, not the release.",
            record.display(),
            STRICT_PROFILES.join(", ")
        ));
    }
    if claim["verified"] != serde_json::Value::Bool(true) {
        return Err(format!(
            "{}: release.json does not record verified: true, so the bundle never \
             passed the builder's load-and-speak gate.",
            record.display()
        ));
    }
    Ok(())
}

/// Hash one file this run fetched against the release's `SHA256SUMS`. No
/// manifest is fine only for a stranger's repo; a fetched file the manifest
/// does not list is refused when it is weights, refused under an official
/// repo whatever it is, and reported otherwise; a digest that does not match
/// is refused, with the file removed so the next run fetches it again.
///
/// Answers whether a digest was compared, which is not the same as whether the
/// call succeeded: the two paths that return without hashing are a repo that
/// ships no manifest and a stranger's file the manifest says nothing about.
/// The caller's count says "verified" out loud, so it has to be a count of
/// files that were.
#[cfg(feature = "download")]
fn verify_fetched(
    root: &Path,
    repo: &str,
    name: &str,
    sums: Option<&HashMap<String, String>>,
) -> Result<bool, String> {
    let Some(sums) = sums else {
        return Ok(false);
    };
    let target = root.join(name);
    let Some(want) = sums.get(name) else {
        if name.ends_with(VOICE_SUFFIX) {
            let _ = fs::remove_file(&target);
            return Err(format!(
                "{}: {SUMS_NAME} does not list {name}; these are weights loudkit would \
                 open with nothing vouching for them. Pin a revision you trust.",
                root.display()
            ));
        }
        if is_official(repo) {
            let _ = fs::remove_file(&target);
            return Err(format!(
                "{}: {SUMS_NAME} does not list {name}. A {OFFICIAL_ORG} release \
                 checksums every file it ships, so this did not come from the release.",
                root.display()
            ));
        }
        eprintln!(
            "loudkit: warning: {name} is not covered by {SUMS_NAME} and therefore not verified"
        );
        return Ok(false);
    };
    if &file_sha256(&target)? != want {
        let _ = fs::remove_file(&target);
        return Err(format!(
            "{}: {name} failed the release checksum. The file has been removed; run \
             the download again, or pin a revision you trust.",
            root.display()
        ));
    }
    Ok(true)
}

/// SHA-256 of a file, streamed a megabyte at a time.
pub(crate) fn file_sha256(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};

    let mut file = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(crate::hex(&hasher.finalize()))
}

/// `org/name`, and nothing that is secretly a path.
#[cfg(feature = "download")]
fn check_repo_id(repo: &str) -> Result<(), String> {
    if is_repo_id(repo) {
        Ok(())
    } else {
        Err(format!(
            "{repo}: not a Hugging Face repo id. Those look like \
             'loudreader/loudr-1'."
        ))
    }
}

/// Why there is no listing: the Hub could not be reached, which a receipt may
/// answer for, or it answered and the answer is the error.
#[cfg(feature = "download")]
enum HubError {
    Unreachable(String),
    Answered(String),
}

/// The commit `revision` names on the Hub today, and every file the repo
/// holds there. One call: the revision's record carries both.
#[cfg(feature = "download")]
fn revision_info(
    base: &str,
    repo: &str,
    revision: &str,
) -> Result<(String, Vec<String>), HubError> {
    let url = format!("{base}/api/models/{repo}/revision/{revision}");
    let body = ureq::get(&url)
        .call()
        .map_err(|e| repo_error(&e, repo, revision))?
        .into_body()
        .with_config()
        .limit(8 * 1024 * 1024)
        .read_to_vec()
        .map_err(|e| repo_error(&e, repo, revision))?;
    let doc: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|e| HubError::Answered(format!("{url}: not JSON: {e}")))?;
    let siblings = doc["siblings"]
        .as_array()
        .ok_or_else(|| HubError::Answered(format!("{url}: no file listing in the answer")))?;
    let commit = doc["sha"]
        .as_str()
        .filter(|sha| !sha.is_empty())
        .ok_or_else(|| {
            HubError::Answered(format!(
                "{repo} at {revision}: the hub named no commit for this revision"
            ))
        })?
        .to_string();
    let listing = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str().map(ToString::to_string))
        .collect();
    Ok((commit, listing))
}

/// How much of a file moves between two looks at the cancel switch.
///
/// `std::io::copy` picks its own buffer and runs to the end of the body, which
/// for a 747 MB weight file is the whole fetch in one call: a switch flipped a
/// second in is not read until it is over. 256 kB is small enough that a
/// cancel is honoured within a chunk on any link worth fetching over and large
/// enough that the loop costs nothing next to the transfer.
#[cfg(feature = "download")]
const FETCH_CHUNK: usize = 256 * 1024;

/// Stream one file to `dest`, and answer with its size.
///
/// Copied a chunk at a time rather than in one `std::io::copy` so `cancel` is
/// read while the bytes are moving, which is the only moment at which stopping
/// a release-sized fetch is worth anything.
#[cfg(feature = "download")]
fn fetch_to(url: &str, dest: &Path, repo: &str, cancel: &Cancel) -> Result<u64, String> {
    use std::io::{Read, Write};

    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut reader = response.into_body().into_reader();
    // Written beside the destination and renamed: an interrupted fetch must
    // not leave a short file the next run mistakes for a finished one.
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let mut out = fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?;
    let mut buffer = vec![0u8; FETCH_CHUNK];
    let mut written = 0u64;
    loop {
        if let Err(why) = cancel.stop(repo) {
            drop(out);
            let _ = fs::remove_file(&part);
            return Err(why);
        }
        let read = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) => {
                drop(out);
                let _ = fs::remove_file(&part);
                return Err(format!("{}: {e}", part.display()));
            }
        };
        if let Err(e) = out.write_all(&buffer[..read]) {
            drop(out);
            let _ = fs::remove_file(&part);
            return Err(format!("{}: {e}", part.display()));
        }
        written += read as u64;
    }
    drop(out);
    fs::rename(&part, dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    Ok(written)
}

/// Say what went wrong with the repo, not with HTTP. A status is the Hub's
/// answer; anything else is the Hub out of reach.
#[cfg(feature = "download")]
fn repo_error(e: &ureq::Error, repo: &str, revision: &str) -> HubError {
    match e {
        ureq::Error::StatusCode(404) => HubError::Answered(format!(
            "{repo} at {revision}: no such repo or revision on the hub"
        )),
        ureq::Error::StatusCode(401 | 403) => HubError::Answered(format!(
            "{repo}: this repo is private or gated. This crate fetches public \
             releases only; download it with the Hugging Face CLI and pass the \
             directory to `Engine::load`."
        )),
        ureq::Error::StatusCode(code) => HubError::Answered(format!("{repo}: HTTP {code}")),
        other => HubError::Unreachable(format!("{repo}: {other}")),
    }
}

/// A byte count a person can read.
#[cfg(feature = "download")]
fn human(bytes: u64) -> String {
    let b = bytes as f64;
    if bytes >= 1 << 20 {
        format!("{:.1} MB", b / (1 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} kB", b / (1 << 10) as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(v: &serde_json::Value) -> Vec<String> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect()
    }

    /// The plan is the shared fixture's, pattern for pattern and file for
    /// file, for both onnx cases.
    #[test]
    fn the_plan_is_the_shared_fixture() {
        let Some(plan) = crate::shared_fixture("release_plan.json") else {
            return;
        };
        let listing = strings(&plan["listing"]);
        let mut seen = 0;
        for case in plan["cases"].as_array().unwrap() {
            if case["backend"] != "onnx" {
                continue;
            }
            seen += 1;
            let cloning = case["cloning"].as_bool().unwrap();
            assert_eq!(IGNORE, strings(&case["ignore"]), "cloning={cloning} ignore");
            let mut got = wanted(&listing, cloning);
            got.sort();
            assert_eq!(got, strings(&case["wanted"]), "cloning={cloning} plan");
        }
        assert_eq!(seen, 2, "the fixture holds two onnx cases");
    }

    #[test]
    fn glob_is_fnmatch() {
        let Some(cases) = crate::shared_fixture("glob.json") else {
            return;
        };
        let cases = cases["cases"].as_array().unwrap();
        assert!(cases.len() >= 10, "the fixture holds probes");
        for c in cases {
            let (pattern, name) = (c["pattern"].as_str().unwrap(), c["name"].as_str().unwrap());
            assert_eq!(
                glob(pattern, name),
                c["match"].as_bool().unwrap(),
                "{pattern} ~ {name}"
            );
        }
    }

    #[test]
    fn a_repo_id_is_what_the_shared_fixture_says() {
        let Some(cases) = crate::shared_fixture("repo_id.json") else {
            return;
        };
        let cases = cases["cases"].as_array().unwrap();
        assert!(cases.len() >= 10, "the fixture holds probes");
        for c in cases {
            let name = c["ref"].as_str().unwrap();
            assert_eq!(
                is_repo_id(name),
                c["is_repo_id"].as_bool().unwrap(),
                "{name:?}"
            );
        }
        // A path that exists is a path, however it is spelled.
        assert!(!is_repo_id(env!("CARGO_MANIFEST_DIR")));
    }

    #[test]
    fn a_manifest_name_that_escapes_is_refused() {
        assert_eq!(rejected_name("onnx/t3_step.onnx"), None);
        assert_eq!(rejected_name("voices/joe.safetensors"), None);
        for bad in [
            "../evil",
            "a/../../evil",
            "a\\b",
            "/etc/passwd",
            "c:/x",
            "a//b",
            "./a",
            "",
        ] {
            assert!(rejected_name(bad).is_some(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn a_mangled_manifest_is_refused_not_skipped() {
        let good = "a".repeat(64);
        let parsed = parse_sums(&format!("{good}  voices/joe.safetensors\n"), "sums").unwrap();
        assert_eq!(parsed["voices/joe.safetensors"], good);
        for (text, why) in [
            (format!("{good} voices/joe.safetensors\n"), "malformed"),
            ("abc  voices/joe.safetensors\n".to_string(), "malformed"),
            (format!("{}  joe\n", "A".repeat(64)), "malformed"),
            (format!("{good}  ../joe.safetensors\n"), "escapes"),
            (format!("{good}  /etc/passwd\n"), "absolute"),
            (format!("{good}  joe\n{good}  joe\n"), "duplicate"),
            ("\n\n".to_string(), "no checksum entries"),
        ] {
            let err = parse_sums(&text, "sums").unwrap_err();
            assert!(err.contains(why), "{text:?}: {err}");
        }
    }

    /// Pinned against Python's `date.fromordinal` at the epoch, two leap days,
    /// the day after one, and the last day of a century year.
    #[test]
    fn the_civil_date_is_pythons() {
        for (days, ymd) in [
            (0, (1970, 1, 1)),
            (11_016, (2000, 2, 29)),
            (19_782, (2024, 2, 29)),
            (19_783, (2024, 3, 1)),
            (20_698, (2026, 9, 2)),
            (47_846, (2100, 12, 31)),
        ] {
            assert_eq!(civil_from_days(days), ymd, "{days} days");
        }
        let now = utc_now();
        assert_eq!(now.len(), 20, "{now}");
        assert!(now.starts_with("20") && now.ends_with('Z'), "{now}");
    }

    #[test]
    fn an_official_repo_is_one_under_the_org() {
        assert!(is_official("loudreader/loudr-1"));
        assert!(is_official("LoudReader/loudr-1-turbo"));
        assert!(!is_official("someone/loudr-1"));
        assert!(!is_official("loudreader"));
    }
}
