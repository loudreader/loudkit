//! loudkit text-to-speech over ONNX Runtime. No torch, and no Python.
//!
//! ```no_run
//! use loudkit::engine::{Engine, Options};
//!
//! let mut engine = Engine::load("loudreader/loudr-1")?;
//! let joe = engine.voice("joe")?;
//! let out = engine.synthesize("Hello from loudkit.", &joe, &Options { seed: 7, ..Default::default() })?;
//! out.save_wav("hello.wav")?;
//! # Ok::<_, String>(())
//! ```
//!
//! `Engine::load` takes a release directory or a Hugging Face repo id. An id
//! is fetched into [`hub::cache_dir`] on the first run and read from there
//! afterwards, when the default `download` feature is on; [`download`] is the
//! same fetch into a directory you name. The onnxruntime shared library is
//! found by `Engine::load`; `LOUDKIT_ONNXRUNTIME_LIB` names one somewhere
//! unusual. `examples/hello.rs` is the code above in one file.

pub mod checkpoint;
pub mod chunking;
pub mod dates;
pub mod engine;
pub mod enroll;
pub mod error;
pub mod execution;
pub mod export;
pub mod fingerprint;
pub mod frontend;
pub(crate) mod grammar;
pub mod hub;
pub mod letters;
pub mod noise;
pub mod numbers;
pub mod postprocess;
pub mod respell;
pub mod rng;
pub mod safetensors;
pub mod sampler;
pub mod speechtext;
pub mod timestretch;
pub mod timing;
pub mod tokenizer;
pub(crate) mod unicode;
pub mod voice;
pub mod wav;
pub mod windowing;

/// The two names every caller needs, at the root the crate is named for.
///
/// A snippet that opens with `use loudkit::engine::{Engine, Options}` asks a
/// reader to know which module a type lives in before they can speak a
/// sentence. The modules stay public, so nothing that names them breaks.
pub use engine::{Engine, Options};

#[cfg(feature = "download")]
pub use hub::download;

/// One of the shared fixtures Python writes, or `None` when it is not there.
///
/// `Cargo.toml` ships `src/**` and not `tests/data/`, so a consumer running
/// `cargo test` on the published crate has no fixture to read, and a unit test
/// that panics on the missing file reports a failure that says nothing about
/// the crate. Returning `None` lets each test decline instead;
/// `LOUDKIT_REQUIRE_ASSETS=1` turns that back into a failure for the runs that
/// are supposed to have the files, which is the same switch the integration
/// tests read. One rule, one home.
///
/// # Panics
///
/// When the file is present and is not the JSON the fixture is supposed to be,
/// and when `LOUDKIT_REQUIRE_ASSETS` is set and the file is unreadable.
#[cfg(test)]
pub(crate) fn shared_fixture(name: &str) -> Option<serde_json::Value> {
    let dir = std::env::var("LOUDKIT_FIXTURE_DIR").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/data/conformance")
            .to_string_lossy()
            .into_owned()
    });
    let path = std::path::Path::new(&dir).join(name);
    match std::fs::read_to_string(&path) {
        Ok(raw) => {
            Some(serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        }
        Err(e) => {
            assert!(
                !std::env::var("LOUDKIT_REQUIRE_ASSETS").is_ok_and(|v| !v.is_empty() && v != "0"),
                "LOUDKIT_REQUIRE_ASSETS is set and {} is unreadable: {e}",
                path.display()
            );
            None
        }
    }
}

/// Bytes as lowercase hexadecimal, the crate's one spelling of it.
///
/// Whole, never truncated. Two callers want the first sixteen characters of a
/// SHA-256 and take them themselves, because a helper that truncated would put
/// the length of a fingerprint somewhere other than beside the fingerprint.
pub(crate) fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut acc, byte| {
        let _ = write!(acc, "{byte:02x}");
        acc
    })
}
