//! Bundle discovery against a directory laid out by hand, and the download
//! against a Hugging Face stub served from this process.
//!
//! Nothing here touches the network: the stub is a `TcpListener` on a loopback
//! port, and the release it serves is a few dozen bytes of nonsense with a
//! real `SHA256SUMS` over it.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "download")]
use std::collections::HashMap;
#[cfg(feature = "download")]
use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "download")]
use std::net::{TcpListener, TcpStream};
#[cfg(feature = "download")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "download")]
use std::thread;

use loudkit::hub::{self, Bundle};

// ---------------------------------------------------------------- scratch

/// A directory of our own, removed when the test ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Scratch {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("loudkit-hub-{}-{tag}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, rel: &str, body: &[u8]) {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, body).unwrap();
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const GRAPHS: [&str; 6] = [
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_step.onnx",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/vocoder.onnx",
];
#[cfg(feature = "download")]
const ENROLL_GRAPHS: [&str; 3] = [
    "onnx/s3_tokenizer.onnx",
    "onnx/camp.onnx",
    "onnx/voice_encoder.onnx",
];

/// The shape of an unpacked release, with a byte in each file.
fn lay_out_release(scratch: &Scratch) {
    scratch.write("loudr-1.safetensors", b"ckpt");
    scratch.write("manifest.json", b"{}");
    scratch.write("tokenizer.json", b"{}");
    scratch.write("release.json", b"{}");
    for graph in GRAPHS {
        scratch.write(graph, b"graph");
    }
    scratch.write("voices/joe.safetensors", b"voice");
    scratch.write("voices/kathleen.safetensors", b"voice");
}

// ------------------------------------------------------------- discovery

#[test]
fn bundle_finds_the_three_paths() {
    let scratch = Scratch::new("bundle");
    lay_out_release(&scratch);
    let bundle = Bundle::open(scratch.path()).unwrap();
    assert_eq!(
        bundle.checkpoint,
        scratch.path().join("loudr-1.safetensors")
    );
    assert_eq!(bundle.onnx_dir, scratch.path().join("onnx"));
    assert_eq!(bundle.tokenizer, scratch.path().join("tokenizer.json"));
    assert_eq!(bundle.voices(), vec!["joe", "kathleen"]);
    assert_eq!(
        bundle.voice_path("joe").unwrap(),
        scratch.path().join("voices/joe.safetensors")
    );
    // The filename spelling is the same voice.
    assert_eq!(
        bundle.voice_path("joe.safetensors").unwrap(),
        bundle.voice_path("joe").unwrap()
    );
    assert!(!bundle.can_enroll());
}

#[test]
fn bundle_ignores_the_two_files_that_are_not_the_checkpoint() {
    let scratch = Scratch::new("siblings");
    lay_out_release(&scratch);
    scratch.write("ve.safetensors", b"encoder");
    scratch.write("loudr-1-enrollment.safetensors", b"enrollment");
    let bundle = Bundle::open(scratch.path()).unwrap();
    assert_eq!(
        bundle.checkpoint,
        scratch.path().join("loudr-1.safetensors")
    );
}

#[test]
fn bundle_takes_a_renamed_checkpoint_when_it_is_the_only_one() {
    let scratch = Scratch::new("renamed");
    lay_out_release(&scratch);
    fs::rename(
        scratch.path().join("loudr-1.safetensors"),
        scratch.path().join("my-model.safetensors"),
    )
    .unwrap();
    let bundle = Bundle::open(scratch.path()).unwrap();
    assert_eq!(
        bundle.checkpoint,
        scratch.path().join("my-model.safetensors")
    );
}

/// The manifest's claim outranks the file name: a directory carrying only
/// enrolment weights under the synthesis name is refused rather than opened.
#[test]
fn the_canonical_name_does_not_outrank_the_manifests_claim() {
    let scratch = Scratch::new("declared-role");
    lay_out_release(&scratch);
    write_checkpoint(
        &scratch.path().join("loudr-1.safetensors"),
        r#"{"artifact_role":"enrollment"}"#,
    );
    let err = err_of(Bundle::open(scratch.path()));
    assert!(err.contains("enrollment artefact"), "{err}");
}

/// Two candidates and no canonical name: the half that declares the synthesis
/// role is the one, and the enrolment half is not a candidate at all.
#[test]
fn the_declared_synthesis_half_wins_over_an_unnamed_neighbour() {
    let scratch = Scratch::new("declared-pair");
    lay_out_release(&scratch);
    fs::remove_file(scratch.path().join("loudr-1.safetensors")).unwrap();
    write_checkpoint(
        &scratch.path().join("a.safetensors"),
        r#"{"artifact_role":"synthesis"}"#,
    );
    write_checkpoint(
        &scratch.path().join("b.safetensors"),
        r#"{"artifact_role":"enrollment"}"#,
    );
    let bundle = Bundle::open(scratch.path()).unwrap();
    assert_eq!(bundle.checkpoint, scratch.path().join("a.safetensors"));
}

/// Only the enrolment half is here, so nothing can speak, and the sentence
/// says which half is missing rather than "no checkpoint".
#[test]
fn the_enrollment_half_alone_cannot_speak() {
    let scratch = Scratch::new("enrollment-only");
    lay_out_release(&scratch);
    fs::remove_file(scratch.path().join("loudr-1.safetensors")).unwrap();
    write_checkpoint(
        &scratch.path().join("packed.safetensors"),
        r#"{"artifact_role":"enrollment"}"#,
    );
    let err = err_of(Bundle::open(scratch.path()));
    assert!(err.contains("enrollment artefact"), "{err}");
}

#[test]
fn bundle_names_what_is_wrong() {
    let scratch = Scratch::new("wrong");
    lay_out_release(&scratch);

    // Two candidates, and no canonical name to break the tie.
    fs::rename(
        scratch.path().join("loudr-1.safetensors"),
        scratch.path().join("a.safetensors"),
    )
    .unwrap();
    scratch.write("b.safetensors", b"other");
    let err = err_of(Bundle::open(scratch.path()));
    assert!(err.contains("a.safetensors, b.safetensors"), "{err}");
    fs::remove_file(scratch.path().join("b.safetensors")).unwrap();

    // No tokenizer beside the weights.
    fs::remove_file(scratch.path().join("tokenizer.json")).unwrap();
    let err = err_of(Bundle::open(scratch.path()));
    assert!(err.contains("tokenizer.json"), "{err}");

    // A voice that is not there lists the ones that are.
    lay_out_release(&scratch);
    let bundle = Bundle::open(scratch.path()).unwrap();
    let err = err_of(bundle.voice_path("nope"));
    assert!(err.contains("joe, kathleen"), "{err}");

    // A voice is a name, not a path out of the release.
    let err = err_of(bundle.voice_path("../../etc/passwd"));
    assert!(err.contains("named, not addressed"), "{err}");
}

#[test]
fn an_empty_directory_is_not_a_bundle() {
    let scratch = Scratch::new("empty");
    let err = err_of(Bundle::open(scratch.path()));
    assert!(err.contains("no loudr-1.safetensors here"), "{err}");
    let err = err_of(Bundle::open(scratch.path().join("nowhere")));
    assert!(err.contains("not a directory"), "{err}");
}

/// A directory that exists is a path, always, and never a repo id.
#[test]
fn a_directory_beats_a_repo_id_of_the_same_name() {
    let scratch = Scratch::new("shadow");
    let inner = scratch.path().join("loudreader/loudr-1");
    fs::create_dir_all(&inner).unwrap();
    assert!(hub::is_repo_id("loudreader/loudr-1"));
    assert_eq!(
        hub::resolve(scratch.path().join("loudreader/loudr-1")).unwrap(),
        inner
    );
    for bad in ["a/b/c", "loudreader", "../etc", "/no/such/place"] {
        assert!(!hub::is_repo_id(bad), "{bad}");
        let err = err_of(hub::resolve(bad));
        assert!(err.contains("not a directory"), "{bad}: {err}");
    }
}

/// A repo id resolves under the cache directory, in the layout the fixture
/// pins under `$LOUDKIT_CACHE`, and a release already there with a receipt
/// for today's commit is read after one Hub call and no fetch.
#[test]
#[cfg(feature = "download")]
fn a_repo_id_resolves_into_the_cache_and_then_stays_there() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("cache");
    let cached = scratch.path().join("loudreader--loudr-1");
    hub::download_with("loudreader/loudr-1", &cached, &hub_stub.options()).unwrap();
    let served = hub_stub.served();

    let restore = [
        ("LOUDKIT_CACHE", std::env::var_os("LOUDKIT_CACHE")),
        ("HF_ENDPOINT", std::env::var_os("HF_ENDPOINT")),
    ];
    std::env::set_var("LOUDKIT_CACHE", scratch.path());
    std::env::set_var("HF_ENDPOINT", &hub_stub.endpoint);
    let fixture = shared_fixture("cache_path.json");
    let overrides: Vec<(PathBuf, String)> = fixture["override"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            std::env::set_var("LOUDKIT_CACHE", o["cache"].as_str().unwrap());
            (
                hub::cache_dir(o["repo"].as_str().unwrap()),
                o["path"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    std::env::set_var("LOUDKIT_CACHE", scratch.path());
    let dir = hub::cache_dir("loudreader/loudr-1");
    let got = hub::resolve("loudreader/loudr-1");
    for (key, value) in restore {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
    assert!(!overrides.is_empty());
    for (got, want) in overrides {
        assert_eq!(got, PathBuf::from(want));
    }
    assert_eq!(dir, cached);
    assert_eq!(got.unwrap(), cached);
    assert_eq!(hub_stub.served(), served, "a cached release was refetched");
}

/// One of the shared fixtures Python writes.
fn shared_fixture(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/data/conformance")
        .join(name);
    serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap()
}

/// The cache layout is the fixture's: one path for a given root and repo.
#[test]
fn the_cache_path_is_the_shared_fixture() {
    let fixture = shared_fixture("cache_path.json");
    assert_eq!(fixture["env"], "LOUDKIT_CACHE");
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 4, "the fixture holds probes");
    for case in cases {
        let (root, repo) = (
            case["root"].as_str().unwrap(),
            case["repo"].as_str().unwrap(),
        );
        assert_eq!(
            hub::cache_path(Path::new(root), repo),
            PathBuf::from(case["path"].as_str().unwrap()),
            "{root} {repo}"
        );
    }
    let dir = hub::cache_dir("loudreader/loudr-1");
    assert_eq!(dir.file_name().unwrap(), "loudreader--loudr-1");
    if std::env::var_os("LOUDKIT_CACHE").is_none() {
        assert_eq!(dir.parent().unwrap().file_name().unwrap(), "loudkit");
    }
}

/// A safetensors file of no tensors whose header carries one manifest.
fn write_checkpoint(path: &Path, manifest: &str) {
    let header = format!(
        "{{\"__metadata__\":{{\"manifest\":{}}}}}",
        serde_json::Value::String(manifest.to_string())
    );
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    fs::write(path, bytes).unwrap();
}

#[test]
fn fusion_bundle_requires_its_pair_graphs() {
    let scratch = Scratch::new("fusion");
    lay_out_release(&scratch);
    fs::rename(
        scratch.path().join("loudr-1.safetensors"),
        scratch.path().join("loudr-1-turbo.safetensors"),
    )
    .unwrap();
    write_checkpoint(
        &scratch.path().join("loudr-1-turbo.safetensors"),
        r#"{"format_version":2,"decode":{"mode":"fusion_mtp2"}}"#,
    );
    let err = err_of(Bundle::open(scratch.path()));
    assert!(err.contains("t3_pair_step.onnx"), "{err}");
    scratch.write("onnx/t3_pair_step.onnx", b"graph");
    scratch.write("onnx/t3_head2.onnx", b"graph");
    fs::remove_file(scratch.path().join("onnx/t3_step.onnx")).unwrap();
    let bundle = Bundle::open(scratch.path()).unwrap();
    assert_eq!(
        bundle.checkpoint.file_name().unwrap(),
        "loudr-1-turbo.safetensors"
    );
}

/// A bundle opened by repo id fetches the three enrollment graphs into its
/// own directory, with the manifest that vouches for them, and the second
/// enrollment fetches nothing. A directory of your own is told how to fetch
/// them and moves no bytes.
#[test]
#[cfg(feature = "download")]
fn enroll_on_a_repo_bundle_fetches_the_cloning_set_once() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("enroll");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(hub_stub.served(), 14);
    let mut bundle = Bundle::open(&dir).unwrap();
    assert!(bundle.repo.is_none(), "a directory is not a repo");

    bundle.repo = Some("loudreader/loudr-1".to_string());
    bundle.fetch_cloning_with(&hub_stub.options()).unwrap();
    let mut fetched = hub_stub.names()[14..].to_vec();
    fetched.sort();
    let mut want: Vec<String> = ENROLL_GRAPHS.iter().map(ToString::to_string).collect();
    want.push("SHA256SUMS".to_string());
    want.sort();
    assert_eq!(fetched, want);
    assert!(bundle.can_enroll());
    assert!(hub::read_receipt(&dir, "loudreader/loudr-1", true).is_some());
    bundle.fetch_cloning_with(&hub_stub.options()).unwrap();
    assert_eq!(
        hub_stub.served(),
        18,
        "the second enrollment fetched something"
    );

    let plain = scratch.path().join("plain");
    hub::download_with("loudreader/loudr-1", &plain, &hub_stub.options()).unwrap();
    let before = hub_stub.served();
    let err = err_of(
        Bundle::open(&plain)
            .unwrap()
            .fetch_cloning_with(&hub_stub.options()),
    );
    assert!(err.contains("cloning: true"), "{err}");
    assert_eq!(
        hub_stub.served(),
        before,
        "a directory bundle fetched something"
    );
}

/// The receipt is the shared fixture's: the file name, the keys in its
/// order, and the predicate and the hit rule case for case, on disk.
#[test]
fn the_receipt_is_the_shared_fixture() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/data/conformance/release_receipt.json");
    let fixture: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(fixture["name"], hub::RECEIPT_NAME);
    let fields: Vec<&str> = fixture["fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f.as_str().unwrap())
        .collect();
    assert_eq!(fields, hub::RECEIPT_FIELDS);

    // Over the example's manifest, what this port writes is the example.
    let example = &fixture["example"];
    let scratch = Scratch::new("receipt");
    scratch.write("SHA256SUMS", fixture["sums"].as_str().unwrap().as_bytes());
    for name in fixture["files"].as_array().unwrap() {
        let name = name.as_str().unwrap();
        scratch.write(name, name.as_bytes());
    }
    hub::write_receipt(
        scratch.path(),
        example["repo"].as_str().unwrap(),
        example["revision"].as_str().unwrap(),
        example["commit"].as_str().unwrap(),
    )
    .unwrap();
    let text = fs::read_to_string(scratch.path().join(hub::RECEIPT_NAME)).unwrap();
    let positions: Vec<usize> = fields
        .iter()
        .map(|f| {
            text.find(&format!("\"{f}\":"))
                .unwrap_or_else(|| panic!("{f} is not in the receipt: {text}"))
        })
        .collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
    let receipt =
        hub::read_receipt(scratch.path(), example["repo"].as_str().unwrap(), false).unwrap();
    assert_eq!(receipt.repo, example["repo"]);
    assert_eq!(receipt.revision, example["revision"]);
    assert_eq!(receipt.commit, example["commit"]);
    assert_eq!(
        receipt.sha256sums.as_deref(),
        example["sha256sums"].as_str()
    );
    assert_eq!(receipt.fetched_at.len(), 20);
    assert!(hub::receipt_hit(
        scratch.path(),
        example["repo"].as_str().unwrap(),
        example["commit"].as_str().unwrap(),
        false
    ));

    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() >= 20, "the fixture holds probes");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let scratch = Scratch::new("case");
        if !case["receipt"].is_null() {
            scratch.write(hub::RECEIPT_NAME, case["receipt"].to_string().as_bytes());
        }
        if let Some(sums) = case["sums"].as_str() {
            scratch.write("SHA256SUMS", sums.as_bytes());
        }
        for file in case["files"].as_array().unwrap() {
            let file = file.as_str().unwrap();
            scratch.write(file, file.as_bytes());
        }
        let repo = case["repo"].as_str().unwrap();
        let valid = hub::read_receipt(scratch.path(), repo, false).is_some();
        assert_eq!(
            if valid { "use" } else { "error" },
            case["offline"],
            "{name}"
        );
        let hit = hub::receipt_hit(
            scratch.path(),
            repo,
            case["commit"].as_str().unwrap(),
            false,
        );
        assert_eq!(if hit { "hit" } else { "fetch" }, case["online"], "{name}");
    }
}

// ---------------------------------------------------------- the stub hub

/// The commit the stub's `main` resolves to until a test moves it.
#[cfg(feature = "download")]
const STUB_COMMIT: &str = "3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e";

/// A Hugging Face stub: the revision's commit, the file listing and the
/// bytes, with a record of what was asked for, so a test can tell a fetch
/// from a skip and see what moved before a refusal.
#[cfg(feature = "download")]
struct StubHub {
    endpoint: String,
    asked: Arc<Mutex<Vec<String>>>,
    commit: Arc<Mutex<String>>,
    trip: Trip,
}

/// A file name and the switch to flip when it is asked for, so a cancel can be
/// timed against a transfer instead of a clock.
#[cfg(feature = "download")]
type Trip = Arc<Mutex<Option<(String, hub::Cancel)>>>;

#[cfg(feature = "download")]
impl StubHub {
    fn start(files: HashMap<String, Vec<u8>>) -> StubHub {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let asked = Arc::new(Mutex::new(Vec::new()));
        let commit = Arc::new(Mutex::new(STUB_COMMIT.to_string()));
        let trip: Trip = Arc::new(Mutex::new(None));
        let record = Arc::clone(&asked);
        let sha = Arc::clone(&commit);
        let trigger = Arc::clone(&trip);
        let names: Vec<String> = {
            let mut n: Vec<String> = files.keys().cloned().collect();
            n.sort();
            n
        };
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                serve(stream, &files, &names, &sha, &record, &trigger);
            }
        });
        StubHub {
            endpoint,
            asked,
            commit,
            trip,
        }
    }

    /// Move `main` to another commit, as a push to the repo would.
    fn move_commit(&self, to: &str) {
        *self.commit.lock().unwrap() = to.to_string();
    }

    /// Flip `cancel` the moment `name` is asked for, before a byte of it is
    /// written: the caller is then guaranteed to be inside the transfer loop
    /// when the switch goes, with no sleep to make it flaky.
    fn cancel_on(&self, name: &str, cancel: &hub::Cancel) {
        *self.trip.lock().unwrap() = Some((name.to_string(), cancel.clone()));
    }

    fn options(&self) -> hub::Options {
        hub::Options {
            revision: "main".to_string(),
            cloning: false,
            endpoint: self.endpoint.clone(),
        }
    }

    /// How many files were served.
    fn served(&self) -> usize {
        self.asked.lock().unwrap().len()
    }

    /// Which files were served, in order.
    fn names(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }
}

#[cfg(feature = "download")]
fn serve(
    mut stream: TcpStream,
    files: &HashMap<String, Vec<u8>>,
    names: &[String],
    commit: &Mutex<String>,
    asked: &Mutex<Vec<String>>,
    trip: &Trip,
) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    loop {
        let mut header = String::new();
        match reader.read_line(&mut header) {
            Ok(0) => break,
            Ok(_) if header.trim().is_empty() => break,
            Ok(_) => {}
            Err(_) => return,
        }
    }
    let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let body: Option<Vec<u8>> = if path.starts_with("/api/models/") {
        let siblings: Vec<String> = names
            .iter()
            .map(|n| format!("{{\"rfilename\":\"{n}\"}}"))
            .collect();
        Some(
            format!(
                "{{\"sha\":\"{}\",\"siblings\":[{}]}}",
                commit.lock().unwrap(),
                siblings.join(",")
            )
            .into_bytes(),
        )
    } else if let Some(rest) = path.split("/resolve/main/").nth(1) {
        asked.lock().unwrap().push(rest.to_string());
        if let Some((when, cancel)) = trip.lock().unwrap().as_ref() {
            if when == rest {
                cancel.cancel();
            }
        }
        files.get(rest).cloned()
    } else {
        None
    };
    let head = match &body {
        Some(b) => format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            b.len()
        ),
        None => {
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
        }
    };
    let _ = stream.write_all(head.as_bytes());
    if let Some(b) = body {
        let _ = stream.write_all(&b);
    }
    let _ = stream.flush();
}

/// A release, signed. `SHA256SUMS` covers everything but itself.
#[cfg(feature = "download")]
fn sign(files: &mut HashMap<String, Vec<u8>>) {
    use sha2::{Digest, Sha256};

    files.remove("SHA256SUMS");
    let mut names: Vec<String> = files.keys().cloned().collect();
    names.sort();
    let sums: String = names
        .iter()
        .map(|n| {
            let digest: String = Sha256::digest(&files[n])
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            format!("{digest}  {n}\n")
        })
        .collect();
    files.insert("SHA256SUMS".into(), sums.into_bytes());
}

/// The release the stub serves, with a `SHA256SUMS` that describes it.
#[cfg(feature = "download")]
fn stub_release() -> HashMap<String, Vec<u8>> {
    let mut files: HashMap<String, Vec<u8>> = HashMap::new();
    files.insert("README.md".into(), b"not part of the plan".to_vec());
    files.insert("loudr-1.safetensors".into(), b"weights".to_vec());
    files.insert("loudr-1-enrollment.safetensors".into(), b"never".to_vec());
    files.insert("ve.safetensors".into(), b"never either".to_vec());
    files.insert("manifest.json".into(), b"{}".to_vec());
    files.insert(
        "release.json".into(),
        br#"{"profile": "full-0.1", "verified": true}"#.to_vec(),
    );
    files.insert("tokenizer.json".into(), b"{}".to_vec());
    files.insert(
        "coreml/vocoder.mlpackage/Manifest.json".into(),
        b"apple".to_vec(),
    );
    for graph in GRAPHS {
        files.insert(graph.to_string(), format!("{graph} bytes").into_bytes());
    }
    files.insert("onnx/export.json".to_string(), b"{}".to_vec());
    for graph in ENROLL_GRAPHS {
        files.insert(graph.to_string(), format!("{graph} bytes").into_bytes());
    }
    for voice in ["joe", "kathleen"] {
        files.insert(format!("voices/{voice}.safetensors"), b"voice".to_vec());
    }
    sign(&mut files);
    files
}

// ---------------------------------------------------------- the download

#[test]
#[cfg(feature = "download")]
fn download_fetches_the_plan_and_nothing_else_bookkeeping_first() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("download");
    let dir = scratch.path().join("loudr-1");

    let root = hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(root, dir);

    // 1 checkpoint + 2 manifests + tokenizer + SHA256SUMS + 6 graphs +
    // the export record + 2 voices
    assert_eq!(hub_stub.served(), 14);
    assert_eq!(&hub_stub.names()[..2], ["SHA256SUMS", "release.json"]);
    assert!(dir.join("onnx/export.json").is_file());
    for graph in GRAPHS {
        assert!(dir.join(graph).is_file(), "{graph} not fetched");
    }
    assert!(dir.join("voices/kathleen.safetensors").is_file());
    assert!(!dir.join("coreml").exists());
    assert!(!dir.join("ve.safetensors").exists());
    assert!(!dir.join("loudr-1-enrollment.safetensors").exists());
    assert!(!dir.join("README.md").exists());
    assert!(!dir.join("loudr-1.safetensors.part").exists());

    let bundle = Bundle::open(&dir).unwrap();
    assert_eq!(bundle.voices(), vec!["joe", "kathleen"]);

    // Idempotent: a second run is a receipt hit and asks for nothing.
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(hub_stub.served(), 14);
}

/// One download in a child process, for its stderr: the progress lines are
/// the only witness to what was hashed. Answers with whether it succeeded.
#[cfg(feature = "download")]
fn download_in_child(endpoint: &str, dir: &Path) -> (bool, String) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_download", "--nocapture"])
        .env("LOUDKIT_TEST_CHILD_ENDPOINT", endpoint)
        .env("LOUDKIT_TEST_CHILD_DIR", dir)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// `download_in_child` for a repo other than the official one.
#[cfg(feature = "download")]
fn download_in_child_from(endpoint: &str, dir: &Path, repo: &str) -> (bool, String) {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_download", "--nocapture"])
        .env("LOUDKIT_TEST_CHILD_ENDPOINT", endpoint)
        .env("LOUDKIT_TEST_CHILD_DIR", dir)
        .env("LOUDKIT_TEST_CHILD_REPO", repo)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The other half of `download_in_child`. Nothing happens unless a parent
/// process asked for it.
#[test]
#[cfg(feature = "download")]
fn child_download() {
    let (Ok(endpoint), Some(dir)) = (
        std::env::var("LOUDKIT_TEST_CHILD_ENDPOINT"),
        std::env::var_os("LOUDKIT_TEST_CHILD_DIR"),
    ) else {
        return;
    };
    let repo =
        std::env::var("LOUDKIT_TEST_CHILD_REPO").unwrap_or_else(|_| "loudreader/loudr-1".into());
    let options = hub::Options {
        revision: "main".to_string(),
        cloning: false,
        endpoint,
    };
    hub::download_with(&repo, dir, &options).unwrap();
}

/// The count in "verified N files against SHA256SUMS" is a count of files that
/// were.
///
/// The counter was raised once per file whatever happened, while the verifier
/// returns without hashing anything when the repo ships no manifest. A
/// stranger's repo with no `SHA256SUMS` therefore printed the sentence over a
/// run in which nothing had been hashed, which is the one claim a user has to
/// be able to believe.
#[test]
#[cfg(feature = "download")]
fn a_repo_without_a_manifest_claims_to_have_verified_nothing() {
    let mut files = stub_release();
    files.remove("SHA256SUMS");
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("noclaim");
    let dir = scratch.path().join("r");
    let (ok, log) = download_in_child_from(&hub_stub.endpoint, &dir, "someone/loudr-1");
    assert!(ok, "{log}");
    assert!(
        !log.contains("verified"),
        "nothing was hashed, so nothing may be claimed: {log}"
    );
    // The files did arrive; the run was real.
    assert!(dir.join("onnx/vocoder.onnx").is_file(), "{log}");

    // And with a manifest, the count is the files it covers: everything
    // fetched except the manifest, which cannot vouch for itself.
    let signed = StubHub::start(stub_release());
    let dir = scratch.path().join("signed");
    let (ok, log) = download_in_child_from(&signed.endpoint, &dir, "someone/loudr-1");
    assert!(ok, "{log}");
    assert!(
        log.contains("verified 13 files against SHA256SUMS"),
        "the count must be the files hashed: {log}"
    );
}

/// A caller can stop a fetch that is already moving bytes.
///
/// A release is hundreds of megabytes over a link this module sets no deadline
/// on, and there was no handle to stop one: a server shutting down had to wait
/// for the whole thing. The switch is read between reads, so the transfer ends
/// inside the file rather than at the end of it.
#[test]
#[cfg(feature = "download")]
fn a_fetch_in_flight_can_be_cancelled() {
    let mut files = stub_release();
    // Larger than one read, so the cancel lands inside the transfer loop
    // rather than between two files.
    files.insert("onnx/vocoder.onnx".into(), vec![b'x'; 3 * 1024 * 1024]);
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("cancel");
    let dir = scratch.path().join("r");

    let cancel = hub::Cancel::new();
    // Flipped by the stub the moment the big file is asked for, so the caller
    // is inside the transfer when it goes. No sleep, so no flake.
    hub_stub.cancel_on("onnx/vocoder.onnx", &cancel);
    let err = err_of(hub::download_cancellable(
        "loudreader/loudr-1",
        &dir,
        &hub_stub.options(),
        &cancel,
    ));
    assert!(err.contains("the download was cancelled"), "{err}");
    assert!(
        !dir.join("onnx/vocoder.onnx").exists(),
        "a cancelled file must not be left as if it had arrived"
    );
    assert!(
        !dir.join("onnx/vocoder.onnx.part").exists(),
        "the part file must not survive either"
    );
    // The run was under way: the bookkeeping files had already arrived.
    assert!(dir.join("SHA256SUMS").is_file());
    // And no receipt, so the next call fetches rather than trusting this one.
    assert!(!dir.join(hub::RECEIPT_NAME).exists());
}

/// A switch already flipped stops the call before it asks the hub anything,
/// and is never read as a hub that cannot be reached.
#[test]
#[cfg(feature = "download")]
fn a_cancelled_switch_is_not_an_unreachable_hub() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("precancel");
    let dir = scratch.path().join("r");
    // A receipt that would answer an unreachable hub, so the two paths are
    // told apart rather than coinciding.
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert!(dir.join(hub::RECEIPT_NAME).is_file());
    let served = hub_stub.served();

    let cancel = hub::Cancel::new();
    cancel.cancel();
    let err = err_of(hub::download_cancellable(
        "loudreader/loudr-1",
        &dir,
        &hub_stub.options(),
        &cancel,
    ));
    assert!(err.contains("the download was cancelled"), "{err}");
    assert_eq!(hub_stub.served(), served, "the hub was asked for something");

    // The plain entry points are unchanged: they carry a switch nobody holds.
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
}

/// A port nobody listens on, so a connection to it is refused at once.
#[cfg(feature = "download")]
fn unlistened_endpoint() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}")
}

/// A receipt for today's commit answers the download as it stands: no file
/// is fetched, none is hashed, and a short file is not noticed. Only a moved
/// commit brings the hashes back.
#[test]
#[cfg(feature = "download")]
fn a_receipt_for_todays_commit_fetches_and_hashes_nothing() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("hit");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(hub_stub.served(), 14);

    let (ok, log) = download_in_child(&hub_stub.endpoint, &dir);
    assert!(ok, "{log}");
    assert_eq!(hub_stub.served(), 14, "a hit fetched something");
    assert!(!log.contains("verified"), "a hit hashed something: {log}");

    // Size decides nothing: a truncated file rides on the receipt.
    fs::write(dir.join("onnx/vocoder.onnx"), b"onnx/voc").unwrap();
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(hub_stub.served(), 14);

    // Until the commit moves; then the hash catches it.
    hub_stub.move_commit("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(&hub_stub.names()[14..], ["SHA256SUMS", "onnx/vocoder.onnx"]);
    assert_eq!(
        fs::read(dir.join("onnx/vocoder.onnx")).unwrap(),
        b"onnx/vocoder.onnx bytes"
    );

    // A receipt over an incomplete set is not a hit either.
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(&hub_stub.names()[16..], ["SHA256SUMS", "tokenizer.json"]);

    // Incomplete means any file the plan selects from the manifest, not only
    // what the port cannot run without: a voice is one of many.
    fs::remove_file(dir.join("voices/kathleen.safetensors")).unwrap();
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(
        &hub_stub.names()[18..],
        ["SHA256SUMS", "voices/kathleen.safetensors"]
    );
}

/// One rule in five ports: a Hub that answers without a commit is an error,
/// never a fetch that writes no receipt.
#[test]
#[cfg(feature = "download")]
fn an_answer_naming_no_commit_is_refused() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("nosha");
    let dir = scratch.path().join("loudr-1");
    hub_stub.move_commit("");
    let why = err_of(hub::download_with(
        "loudreader/loudr-1",
        &dir,
        &hub_stub.options(),
    ));
    assert!(why.contains("named no commit"), "{why}");
    assert_eq!(hub_stub.served(), 0);
    assert!(!dir.join(hub::RECEIPT_NAME).exists());
}

/// When the revision moves, the manifest is fetched again and everything it
/// still lists is kept by hash; a file that no longer matches is fetched,
/// and only that one.
#[test]
#[cfg(feature = "download")]
fn a_moved_revision_refetches_the_manifest_and_keeps_what_it_lists() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("drift");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(hub_stub.served(), 14);
    let receipt = |dir: &Path| hub::read_receipt(dir, "loudreader/loudr-1", false).unwrap();
    assert_eq!(receipt(&dir).commit, STUB_COMMIT);

    let moved = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    hub_stub.move_commit(moved);
    let (ok, log) = download_in_child(&hub_stub.endpoint, &dir);
    assert!(ok, "{log}");
    assert_eq!(&hub_stub.names()[14..], ["SHA256SUMS"]);
    assert!(log.contains("verified 13 files"), "{log}");
    assert_eq!(receipt(&dir).commit, moved);

    // Same length, wrong bytes.
    fs::write(dir.join("onnx/t3_step.onnx"), b"onnx/t3_step.onnx BYTES").unwrap();
    hub_stub.move_commit("cccccccccccccccccccccccccccccccccccccccc");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert_eq!(&hub_stub.names()[15..], ["SHA256SUMS", "onnx/t3_step.onnx"]);
    assert_eq!(
        fs::read(dir.join("onnx/t3_step.onnx")).unwrap(),
        b"onnx/t3_step.onnx bytes"
    );
}

/// A Hub that cannot be reached is answered by the receipt over a complete
/// set, and says so. Without one, or over a short set, it is the error.
#[test]
#[cfg(feature = "download")]
fn an_unreachable_hub_is_answered_by_the_receipt() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("offline");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();

    let mut offline = hub_stub.options();
    offline.endpoint = unlistened_endpoint();
    let got = hub::download_with("loudreader/loudr-1", &dir, &offline).unwrap();
    assert_eq!(got, dir);
    assert_eq!(hub_stub.served(), 14);
    let (ok, log) = download_in_child(&offline.endpoint, &dir);
    assert!(ok, "{log}");
    assert!(log.contains("the hub cannot be reached"), "{log}");
    assert!(log.contains(STUB_COMMIT), "{log}");

    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    err_of(hub::download_with("loudreader/loudr-1", &dir, &offline));

    fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    fs::remove_file(dir.join(hub::RECEIPT_NAME)).unwrap();
    err_of(hub::download_with("loudreader/loudr-1", &dir, &offline));

    // A directory verified as one repo does not answer for another.
    hub::write_receipt(&dir, "loudreader/other", "main", STUB_COMMIT).unwrap();
    err_of(hub::download_with("loudreader/loudr-1", &dir, &offline));

    // Nor does a receipt no download wrote: one field forged is no receipt.
    hub::write_receipt(&dir, "loudreader/loudr-1", "main", STUB_COMMIT).unwrap();
    let good = fs::read_to_string(dir.join(hub::RECEIPT_NAME)).unwrap();
    for (what, forged) in [
        (
            "no commit",
            good.replacen(&format!(" \"commit\": \"{STUB_COMMIT}\",\n"), "", 1),
        ),
        ("an empty commit", good.replacen(STUB_COMMIT, "", 1)),
        (
            "the wrong digest",
            good.replacen("\"sha256sums\": \"", "\"sha256sums\": \"0", 1),
        ),
    ] {
        assert_ne!(forged, good, "{what}: nothing forged");
        fs::write(dir.join(hub::RECEIPT_NAME), &forged).unwrap();
        assert!(
            hub::download_with("loudreader/loudr-1", &dir, &offline).is_err(),
            "offline, a receipt with {what} answered"
        );
    }
    fs::write(dir.join(hub::RECEIPT_NAME), &good).unwrap();
    hub::download_with("loudreader/loudr-1", &dir, &offline).unwrap();
}

#[test]
#[cfg(feature = "download")]
fn cloning_adds_the_enrollment_graphs() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("cloning");
    let dir = scratch.path().join("loudr-1");
    let mut options = hub_stub.options();
    options.cloning = true;
    hub::download_with("loudreader/loudr-1", &dir, &options).unwrap();
    for graph in ENROLL_GRAPHS {
        assert!(dir.join(graph).is_file(), "{graph} not fetched");
    }
    assert!(!dir.join("ve.safetensors").exists());
    assert!(Bundle::open(&dir).unwrap().can_enroll());
}

#[test]
#[cfg(feature = "download")]
fn a_release_exported_before_the_record_still_works() {
    let mut files = stub_release();
    files.remove("onnx/export.json");
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("norecord");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    assert!(!dir.join("onnx/export.json").exists());
    assert!(Bundle::open(&dir).is_ok());
}

#[test]
#[cfg(feature = "download")]
fn a_release_without_graphs_is_refused_before_anything_moves() {
    let mut files = stub_release();
    for graph in GRAPHS {
        files.remove(graph);
    }
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("nographs");
    let dir = scratch.path().join("loudr-1");
    let err = err_of(hub::download_with(
        "someone/loudr-1",
        &dir,
        &hub_stub.options(),
    ));
    assert!(err.contains("onnx/t3_cond.onnx"), "{err}");
    assert_eq!(hub_stub.served(), 0, "bytes moved before the refusal");
}

#[test]
#[cfg(feature = "download")]
fn a_turbo_release_without_graphs_is_refused_before_download() {
    let mut files: HashMap<String, Vec<u8>> = HashMap::new();
    files.insert("loudr-1-turbo.safetensors".into(), b"turbo".to_vec());
    files.insert("manifest.json".into(), b"{}".to_vec());
    files.insert("tokenizer.json".into(), b"{}".to_vec());
    files.insert(
        "release.json".into(),
        br#"{"profile": "turbo-0.1", "verified": true}"#.to_vec(),
    );
    files.insert("voices/joe.safetensors".into(), b"voice".to_vec());
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("turbo");
    let err = err_of(hub::download_with(
        "loudreader/loudr-1-turbo",
        scratch.path().join("turbo"),
        &hub_stub.options(),
    ));
    assert!(err.contains("onnx/t3_cond.onnx"), "{err}");
    assert!(err.contains("loudreader/loudr-1"), "{err}");
    assert_eq!(hub_stub.served(), 0, "bytes moved before the refusal");
}

#[test]
#[cfg(feature = "download")]
fn a_corrupted_file_is_refused_and_removed() {
    let mut files = stub_release();
    // The manifest still describes the original bytes.
    files.insert("tokenizer.json".into(), b"tampered".to_vec());
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("corrupt");
    let dir = scratch.path().join("loudr-1");
    let err = err_of(hub::download_with(
        "loudreader/loudr-1",
        &dir,
        &hub_stub.options(),
    ));
    assert!(err.contains("tokenizer.json"), "{err}");
    assert!(err.contains("failed the release checksum"), "{err}");
    assert!(
        !dir.join("tokenizer.json").exists(),
        "the bad file was left behind"
    );
}

#[test]
#[cfg(feature = "download")]
fn an_official_release_without_a_manifest_is_refused_before_anything_moves() {
    let mut files = stub_release();
    files.remove("SHA256SUMS");
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("nosums");
    let err = err_of(hub::download_with(
        "loudreader/loudr-1",
        scratch.path().join("r"),
        &hub_stub.options(),
    ));
    assert!(err.contains("no SHA256SUMS"), "{err}");
    assert_eq!(hub_stub.served(), 0);
}

#[test]
#[cfg(feature = "download")]
fn a_third_party_release_without_a_manifest_is_allowed() {
    let mut files = stub_release();
    files.remove("SHA256SUMS");
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("stranger");
    hub::download_with(
        "someone/loudr-1",
        scratch.path().join("r"),
        &hub_stub.options(),
    )
    .unwrap();
}

#[test]
#[cfg(feature = "download")]
fn a_development_bundle_is_refused_before_its_weights_move() {
    let mut files = stub_release();
    files.insert(
        "release.json".into(),
        br#"{"profile": "lenient", "verified": true}"#.to_vec(),
    );
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("lenient");
    let err = err_of(hub::download_with(
        "loudreader/loudr-1",
        scratch.path().join("r"),
        &hub_stub.options(),
    ));
    assert!(err.contains("development bundle"), "{err}");
    assert_eq!(hub_stub.names(), ["SHA256SUMS", "release.json"]);
}

#[test]
#[cfg(feature = "download")]
fn an_ungated_build_is_refused() {
    let mut files = stub_release();
    files.insert(
        "release.json".into(),
        br#"{"profile": "full-0.1", "verified": false}"#.to_vec(),
    );
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("ungated");
    let err = err_of(hub::download_with(
        "loudreader/loudr-1",
        scratch.path().join("r"),
        &hub_stub.options(),
    ));
    assert!(err.contains("verified: true"), "{err}");
}

#[test]
#[cfg(feature = "download")]
fn the_turbo_profile_is_a_release_too() {
    // One gate for both models; what keeps turbo out of this port is the
    // graphs check, which this listing passes.
    let mut files = stub_release();
    files.insert(
        "release.json".into(),
        br#"{"profile": "turbo-0.1", "verified": true}"#.to_vec(),
    );
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("turboprofile");
    hub::download_with(
        "loudreader/loudr-1",
        scratch.path().join("r"),
        &hub_stub.options(),
    )
    .unwrap();
}

#[test]
#[cfg(feature = "download")]
fn unlisted_weights_are_refused() {
    let mut files = stub_release();
    // Added after signing, so the manifest says nothing about it.
    files.insert("voices/mallory.safetensors".into(), b"unvouched".to_vec());
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("unlisted");
    let dir = scratch.path().join("r");
    let err = err_of(hub::download_with(
        "someone/loudr-1",
        &dir,
        &hub_stub.options(),
    ));
    assert!(err.contains("voices/mallory.safetensors"), "{err}");
    assert!(err.contains("nothing vouching"), "{err}");
    assert!(!dir.join("voices/mallory.safetensors").exists());
}

#[test]
#[cfg(feature = "download")]
fn a_listing_cannot_name_a_file_outside_the_directory() {
    let mut files = stub_release();
    files.insert("../escaped.safetensors".into(), b"no".to_vec());
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("escape");
    let dir = scratch.path().join("r");
    let err = err_of(hub::download_with(
        "someone/loudr-1",
        &dir,
        &hub_stub.options(),
    ));
    assert!(err.contains("nothing was written"), "{err}");
    assert_eq!(hub_stub.served(), 0);
    assert!(!scratch.path().join("escaped.safetensors").exists());
}

#[test]
#[cfg(feature = "download")]
fn a_short_release_says_what_is_missing() {
    let mut files = stub_release();
    files.remove("voices/joe.safetensors");
    files.remove("voices/kathleen.safetensors");
    sign(&mut files);
    let hub_stub = StubHub::start(files);
    let scratch = Scratch::new("novoices");
    let dir = scratch.path().join("loudr-1");
    let err = err_of(hub::download_with(
        "loudreader/loudr-1",
        &dir,
        &hub_stub.options(),
    ));
    assert!(err.contains("voices/*.safetensors"), "{err}");
}

#[test]
#[cfg(feature = "download")]
fn a_repo_id_is_not_a_path() {
    let scratch = Scratch::new("repoid");
    for bad in ["../etc", "loudreader", "a/b/c", "/abs/path", "loudreader/"] {
        let err = err_of(hub::download(bad, scratch.path()));
        assert!(err.contains("not a Hugging Face repo id"), "{bad}: {err}");
    }
}

fn err_of<T>(result: Result<T, String>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

/// A receipt for the synthesis set answers a plain load when the Hub is
/// away, and an enrollment gets the sentence naming the three graphs, not
/// the transport error.
#[test]
#[cfg(feature = "download")]
fn offline_without_the_cloning_set_the_sentence_names_the_graphs() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("offline-cloning");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();
    let away = hub::Options {
        endpoint: unlistened_endpoint(),
        ..Default::default()
    };
    assert_eq!(
        hub::download_with("loudreader/loudr-1", &dir, &away).unwrap(),
        dir
    );
    let cloning = hub::Options {
        cloning: true,
        ..away
    };
    let err = err_of(hub::download_with("loudreader/loudr-1", &dir, &cloning));
    assert!(
        err.contains(
            "holds no enrollment graphs (onnx/s3_tokenizer.onnx, onnx/camp.onnx, \
             onnx/voice_encoder.onnx). Connect once to fetch them."
        ),
        "{err}"
    );
}

/// An empty file, garbage, and a file too large to be a receipt are no
/// receipt, and the last is not read.
#[test]
fn a_receipt_that_is_not_a_record_is_nothing() {
    let scratch = Scratch::new("not-a-record");
    let record = "{\"repo\": \"someone/loudr-1\", \"revision\": \"main\", \"commit\": \
                  \"3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e\", \"sha256sums\": null, \
                  \"fetched_at\": \"2026-09-02T12:00:00Z\"";
    scratch.write(hub::RECEIPT_NAME, format!("{record}}}\n").as_bytes());
    assert!(
        hub::read_receipt(scratch.path(), "someone/loudr-1", false).is_some(),
        "the record itself was refused"
    );
    let bodies: [(&str, Vec<u8>); 3] = [
        ("an empty file", Vec::new()),
        ("garbage", b"\x00\xff{".repeat(1 << 15)),
        (
            "a record padded past the limit",
            format!("{record}, \"pad\": \"{}\"}}", "x".repeat(1 << 20)).into_bytes(),
        ),
    ];
    for (name, body) in bodies {
        scratch.write(hub::RECEIPT_NAME, &body);
        assert!(
            hub::read_receipt(scratch.path(), "someone/loudr-1", false).is_none(),
            "{name} was accepted as a receipt"
        );
    }
}

/// Offline, a pin is answered by the bytes it names or by nothing.
///
/// A caller asking for one revision and getting another is the failure a pin
/// exists to prevent, so the receipt's own commit and ref are what the request
/// is held to. `main` is this port's unpinned request, because `Options` gives
/// `revision` that default and cannot tell it from one a caller wrote.
#[test]
#[cfg(feature = "download")]
fn offline_refuses_a_cache_that_is_not_the_pin() {
    let hub_stub = StubHub::start(stub_release());
    let scratch = Scratch::new("offline-pin");
    let dir = scratch.path().join("loudr-1");
    hub::download_with("loudreader/loudr-1", &dir, &hub_stub.options()).unwrap();

    let offline = |revision: &str| {
        let mut options = hub_stub.options();
        options.endpoint = unlistened_endpoint();
        options.revision = revision.to_string();
        options
    };

    // Forty hex names a commit, which the receipt records exactly.
    let why = err_of(hub::download_with(
        "loudreader/loudr-1",
        &dir,
        &offline(&"b".repeat(40)),
    ));
    assert!(why.contains("not a preference"), "{why}");
    assert_eq!(
        hub::download_with("loudreader/loudr-1", &dir, &offline(STUB_COMMIT)).unwrap(),
        dir
    );

    // A ref cannot be resolved with no hub, and the receipt records which one
    // this copy answers for.
    let why = err_of(hub::download_with(
        "loudreader/loudr-1",
        &dir,
        &offline("v0.2.0"),
    ));
    assert!(why.contains("cannot be resolved without it"), "{why}");

    // The default is not a pin: it takes whatever the cache holds.
    assert_eq!(
        hub::download_with("loudreader/loudr-1", &dir, &offline("main")).unwrap(),
        dir
    );
}
