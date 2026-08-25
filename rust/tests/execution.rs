//! The execution-provider knob, over the public API.
//!
//! Everything here runs with no GPU, no onnxruntime and no assets: the
//! resolution rule takes the available set as an argument precisely so it can
//! be held to a fixture on a machine that has none of the hardware. What is
//! *not* covered is whether a CUDA, CoreML or DirectML session produces the
//! right numbers — that needs the hardware, and it is a measurement for the
//! convergence step rather than an assertion for this file.

use loudkit::engine::Engine;
use loudkit::execution::{Execution, ExecutionConfig, OnnxProvider};

fn config(provider: OnnxProvider) -> ExecutionConfig {
    ExecutionConfig {
        onnx_provider: provider,
    }
}

/// True when this build has to ask ONNX Runtime which providers exist.
///
/// A default build answers from `cfg!` alone and touches no shared library. A
/// build with any provider feature calls `GetAvailableProviders`, and `ort`
/// *panics* rather than erroring when ORT_DYLIB_PATH names nothing loadable —
/// so the two tests that reach that far step aside here. This is the
/// weight-free suite; the asset-backed job is where a real library exists.
fn probes_the_shared_library() -> bool {
    cfg!(any(
        feature = "cuda",
        feature = "coreml",
        feature = "directml"
    ))
}

#[test]
fn the_accepted_values_are_the_five_shared_spellings() {
    let want = [
        ("auto", OnnxProvider::Auto),
        ("cpu", OnnxProvider::Cpu),
        ("cuda", OnnxProvider::Cuda),
        ("coreml", OnnxProvider::CoreMl),
        ("directml", OnnxProvider::DirectMl),
    ];
    for (spelling, provider) in want {
        assert_eq!(OnnxProvider::parse(spelling), Ok(provider));
        assert_eq!(provider.as_str(), spelling);
    }
}

#[test]
fn a_misspelling_is_refused_rather_than_treated_as_auto() {
    for bad in ["", "gpu", "gpu ", "gPu", "gpu\n", "CPU", "metal", "mps"] {
        let err = OnnxProvider::parse(bad).unwrap_err();
        assert!(err.contains("unknown onnx provider"), "{bad:?}: {err}");
    }
}

#[test]
fn the_default_is_auto() {
    assert_eq!(ExecutionConfig::default().onnx_provider, OnnxProvider::Auto);
}

#[test]
fn auto_reports_the_provider_it_picked() {
    let picked = Execution::resolve(
        &config(OnnxProvider::Auto),
        &[OnnxProvider::Cpu, OnnxProvider::Cuda],
    )
    .unwrap();
    assert_eq!(picked.provider(), OnnxProvider::Cuda);
    assert!(picked.describe().contains("cuda"), "{}", picked.describe());
}

#[test]
fn auto_declines_coreml_even_where_the_build_offers_it() {
    // The EP is selectable by name and stays out of `auto`: it measured 0.62x
    // to 0.71x real time against the CPU provider's 1.22x to 1.47x and moved
    // the token stream. Naming it gets it; saying nothing does not.
    let picked = Execution::resolve(
        &config(OnnxProvider::Auto),
        &[OnnxProvider::Cpu, OnnxProvider::CoreMl],
    )
    .unwrap();
    assert_eq!(picked.provider(), OnnxProvider::Cpu);
}

#[test]
fn auto_prefers_cuda_over_everything_else() {
    let all = [
        OnnxProvider::Cpu,
        OnnxProvider::CoreMl,
        OnnxProvider::DirectMl,
        OnnxProvider::Cuda,
    ];
    let picked = Execution::resolve(&config(OnnxProvider::Auto), &all).unwrap();
    assert_eq!(picked.provider(), OnnxProvider::Cuda);
}

#[test]
fn an_unavailable_provider_errors_instead_of_falling_back_to_cpu() {
    for want in [
        OnnxProvider::Cuda,
        OnnxProvider::CoreMl,
        OnnxProvider::DirectMl,
    ] {
        let err = Execution::resolve(&config(want), &[OnnxProvider::Cpu]).unwrap_err();
        // The three things the message owes a reader: what was asked for, what
        // this build has, and what to do about the gap.
        assert!(err.contains(want.as_str()), "{err}");
        assert!(err.contains("this build offers cpu"), "{err}");
        assert!(err.contains("ORT_DYLIB_PATH"), "{err}");
    }
}

#[test]
fn the_chosen_provider_reaches_the_describe_line() {
    let explicit = Execution::resolve(&config(OnnxProvider::Cpu), &[OnnxProvider::Cpu]).unwrap();
    assert_eq!(explicit.describe(), "exec[onnx provider=cpu]");

    // The line says what ran, not what was asked for, so an automatic choice
    // and a named one that landed in the same place print the same thing. The
    // difference is kept on the value, where a caller can still read it.
    let automatic = Execution::resolve(&config(OnnxProvider::Auto), &[OnnxProvider::Cpu]).unwrap();
    assert_eq!(automatic.describe(), explicit.describe());
    assert_eq!(automatic.requested(), OnnxProvider::Auto);
}

#[test]
fn the_engine_refuses_an_uncompiled_provider_before_it_looks_for_assets() {
    // Only meaningful on a build without the feature, which is the default and
    // what CI runs. With `--features cuda` the provider may well be available
    // and the load would go on to fail on the missing checkpoint instead.
    if cfg!(feature = "cuda") || probes_the_shared_library() {
        eprintln!("SKIPPED (not a pass): built with a provider feature");
        return;
    }
    // `.err()` rather than `unwrap_err()`: an `Engine` holds six ONNX sessions
    // and is not `Debug`.
    let err = Engine::load_with(
        "/nonexistent/loudr-1.safetensors",
        "/nonexistent/onnx",
        "/nonexistent/tokenizer.json",
        &config(OnnxProvider::Cuda),
    )
    .err()
    .expect("a build without the cuda feature must refuse --provider cuda");
    assert!(err.contains("cuda"), "{err}");
    assert!(err.contains("--features cuda"), "{err}");
    // The provider is checked ahead of the checkpoint on purpose: a build that
    // cannot run CUDA at all should say so, not send the caller looking for a
    // file that was never the problem.
    assert!(!err.contains("nonexistent"), "{err}");
}

#[test]
fn cpu_is_always_reachable_so_auto_cannot_fail() {
    let picked = Execution::resolve(&config(OnnxProvider::Cpu), &[OnnxProvider::Cpu]).unwrap();
    assert_eq!(picked.provider(), OnnxProvider::Cpu);
    assert!(OnnxProvider::Cpu.is_compiled());
    if probes_the_shared_library() {
        eprintln!("SKIPPED (not a pass): built with a provider feature");
        return;
    }
    assert!(loudkit::execution::available_providers()
        .unwrap()
        .contains(&OnnxProvider::Cpu));
}
