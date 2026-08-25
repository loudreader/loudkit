//! The execution layer: how the exported graphs are run, never what is
//! computed. Nothing here reaches the fingerprint.
//!
//! Which provider runs the graphs is a knob, spelled the same in all four
//! ports: `onnx_provider`, over the same five values, with the same refusal.
//! Without it every ONNX path would be pinned to CPU, and a benchmark figure
//! measured on the torch path would stand over a `loudkit` user getting ~1.2x
//! real time with no way to say otherwise.
//!
//! Two gates decide whether a provider can run, and they fail for different
//! reasons, so the error names which one closed:
//!
//! * the **cargo feature**, because `ort` puts each provider's registration
//!   code behind its own feature — a feature left off is not a slower path, it
//!   is code that does not exist;
//! * the **libonnxruntime at `ORT_DYLIB_PATH`**, because `load-dynamic` means
//!   the provider set is a property of the library the user supplies at
//!   runtime, not of this crate.
//!
//! A GPU provider may change the numbers. It is not bit-parity with MLAS and
//! is not claimed to be — the conformance fixture is a CPU measurement, and a
//! provider that moves tokens is a measurement to record, not a tolerance to
//! widen.

// Only the CoreML builder needs a path: it is the one provider configured with a
// cache directory, so the import travels with `coreml_cache_dir` behind the same
// gate rather than sitting unused in the default build.
#[cfg(feature = "coreml")]
use std::path::PathBuf;

use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::SessionBuilder;
use ort::session::Session;

/// The three graphs CoreML is allowed to run.
///
/// An allowlist, not a denylist: a graph this crate opens later — the
/// enrollment graphs go through the same builder, and the voice encoder
/// decides what a cloned voice sounds like — stays on CPU until somebody
/// measures it on CoreML.
const RENDERER_GRAPHS: [&str; 3] = ["flow_encoder.onnx", "flow_estimator.onnx", "vocoder.onnx"];

/// Overrides where CoreML keeps its compiled models.
///
/// Gated with its sole caller: a build without the `coreml` feature cannot
/// register the provider at all, so a cache knob for it would be a setting that
/// nothing reads.
#[cfg(feature = "coreml")]
const COREML_CACHE_ENV: &str = "LOUDKIT_COREML_CACHE";

/// Where CoreML writes compiled models, and why it must be somewhere.
///
/// Compiling the renderer graphs takes about 146 s. With a cache directory
/// that is paid once per machine and later loads cost about 25 s; without one
/// it is paid on *every* session, which no interactive use can absorb. The
/// cache runs to roughly 1.6 GB.
#[cfg(feature = "coreml")]
fn coreml_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(COREML_CACHE_ENV) {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join("Library")
        .join("Caches")
        .join("loudkit")
        .join("coreml")
}

/// Which ONNX Runtime execution provider the sessions register.
///
/// The spellings are the shared contract: `ExecutionConfig.onnx_provider` in
/// Python, `ExecutionConfig.ONNXProvider` in Go and `ExecutionOptions.onnxProvider`
/// in TypeScript accept exactly these five strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnnxProvider {
    /// Pick the best provider this build and this machine actually offer.
    #[default]
    Auto,
    Cpu,
    Cuda,
    CoreMl,
    DirectMl,
}

/// What `auto` prefers, best first. Shared with the other three ports: a port
/// that reorders this picks different hardware for the same config.
///
/// CPU is last and always reachable, which is why `auto` cannot fail.
///
/// auto prefers a provider only where a measurement says it is faster. CoreML
/// is faster — the split placement in [`session_builder`] measures RTF
/// 1.35-1.70 on an M3 Pro against 0.85-1.02 for all-CPU — and is still not a
/// default, for a reason that is not speed: compiling the renderer graphs
/// costs about 146 s the first time on a machine and leaves 1.6 GB of cache
/// behind. A default may not spend either without being asked. DirectML has
/// never been run by this project. Both stay selectable by name; neither is a
/// default. CUDA leads until it is measured, and drops out the same way if it
/// loses.
const AUTO_ORDER: [OnnxProvider; 2] = [OnnxProvider::Cuda, OnnxProvider::Cpu];

impl OnnxProvider {
    /// The spelling a caller writes, which is also what `describe` prints.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
        }
    }

    /// Parse a caller's value.
    ///
    /// # Errors
    ///
    /// Returns an error naming the unknown value and listing all five. An
    /// unrecognised provider must not fall back to `auto`: a typo would then
    /// run on the CPU under the name of a GPU, which is the failure this whole
    /// module exists to prevent.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "coreml" => Ok(Self::CoreMl),
            "directml" => Ok(Self::DirectMl),
            other => Err(format!(
                "unknown onnx provider {other:?}; \
                 expected auto, cpu, cuda, coreml or directml"
            )),
        }
    }

    /// Whether this build compiled the provider's registration code in.
    #[must_use]
    pub const fn is_compiled(self) -> bool {
        match self {
            Self::Auto | Self::Cpu => true,
            Self::Cuda => cfg!(feature = "cuda"),
            Self::CoreMl => cfg!(feature = "coreml"),
            Self::DirectMl => cfg!(feature = "directml"),
        }
    }

    /// The cargo feature that compiles it in.
    const fn feature(self) -> &'static str {
        match self {
            Self::Auto | Self::Cpu => "",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
            Self::DirectMl => "directml",
        }
    }

    /// Which libonnxruntime carries it, for the half of the message a cargo
    /// feature cannot fix.
    const fn library_hint(self) -> &'static str {
        match self {
            Self::Auto | Self::Cpu => "any libonnxruntime",
            Self::Cuda => "the onnxruntime-gpu (CUDA) build",
            Self::CoreMl => "an Apple-platform onnxruntime built with CoreML",
            Self::DirectMl => "the Microsoft.ML.OnnxRuntime.DirectML build, on Windows",
        }
    }
}

/// How the graphs are run. Free to differ between machines; never changes what
/// is computed, which is why it is not part of [`crate::fingerprint`].
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ExecutionConfig {
    pub onnx_provider: OnnxProvider,
}

/// A resolved execution layer: what was asked for, and what was chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Execution {
    requested: OnnxProvider,
    provider: OnnxProvider,
}

impl Execution {
    /// Resolve a request against the providers this build and this machine
    /// offer.
    ///
    /// `available` is measured by [`available_providers`], which needs a loaded
    /// onnxruntime. The rule itself takes the list as an argument so it stays
    /// testable on a machine with no GPU and no shared library at all.
    ///
    /// # Errors
    ///
    /// An explicit provider that is not available is an error, never a quiet
    /// demotion to CPU. A silent fallback is worse than a refusal here: the
    /// benchmark row still says `cuda`, and the number under it is wrong.
    pub fn resolve(config: &ExecutionConfig, available: &[OnnxProvider]) -> Result<Self, String> {
        let requested = config.onnx_provider;
        if requested == OnnxProvider::Auto {
            let provider = AUTO_ORDER
                .into_iter()
                .find(|p| available.contains(p))
                .ok_or_else(|| {
                    format!(
                        "onnx_provider \"auto\" found no usable execution provider; \
                         the libonnxruntime at ORT_DYLIB_PATH offers none of {}",
                        list(&AUTO_ORDER)
                    )
                })?;
            return Ok(Self {
                requested,
                provider,
            });
        }
        if available.contains(&requested) {
            return Ok(Self {
                requested,
                provider: requested,
            });
        }
        Err(unavailable(requested, requested.is_compiled(), available))
    }

    /// The provider that actually runs.
    #[must_use]
    pub const fn provider(self) -> OnnxProvider {
        self.provider
    }

    /// What the caller asked for, which is `Auto` whenever the choice was made
    /// here. Not in [`Self::describe`]: the line carries the provider that ran,
    /// and the four ports agree on that much only if none of them decorates it.
    #[must_use]
    pub const fn requested(self) -> OnnxProvider {
        self.requested
    }

    /// The execution half of [`crate::engine::Engine::describe`].
    ///
    /// `onnx` is the placement, in the slot Python's `ExecutionConfig.describe`
    /// fills with the device, then the provider that ran. This port has one
    /// device and one precision, so the flags Python prints for those would say
    /// the same thing on every run and are left off.
    #[must_use]
    pub fn describe(self) -> String {
        format!("exec[onnx provider={}]", self.provider.as_str())
    }
}

/// `a, b, c`, for the error messages.
fn list(providers: &[OnnxProvider]) -> String {
    providers
        .iter()
        .map(|p| p.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// The refusal for an explicit provider this build cannot run.
///
/// Takes `compiled` rather than reading `want.is_compiled()`, so both halves of
/// the message are reachable from a test on a default build — the branch that
/// only fires when a GPU feature *is* on would otherwise never be exercised by
/// the CI this crate actually runs.
fn unavailable(want: OnnxProvider, compiled: bool, available: &[OnnxProvider]) -> String {
    let cause = if compiled {
        format!(
            "The `{}` cargo feature is on, so it is the libonnxruntime at ORT_DYLIB_PATH \
             that lacks it: point it at {}",
            want.feature(),
            want.library_hint()
        )
    } else {
        format!(
            "The `{feature}` cargo feature is off: rebuild with `--features {feature}` and \
             point ORT_DYLIB_PATH at {}",
            want.library_hint(),
            feature = want.feature()
        )
    };
    format!(
        "onnx_provider {:?} is not available; this build offers {}. {cause}.",
        want.as_str(),
        list(available)
    )
}

/// The concrete providers this build compiled in *and* the loaded onnxruntime
/// reports.
///
/// A default build answers from `cfg!` alone and touches no shared library. A
/// build with a provider feature calls `GetAvailableProviders`, which loads the
/// library at `ORT_DYLIB_PATH` — and `ort` panics rather than erroring when
/// that path names nothing loadable, the same panic the first
/// `Session::builder()` would raise anyway.
///
/// # Errors
///
/// Returns an error when ONNX Runtime's provider query fails. It does not fail
/// for a provider that is merely absent — that is the answer, not an error.
pub fn available_providers() -> Result<Vec<OnnxProvider>, String> {
    disable_ort_telemetry();
    // CPU is unconditional, the same way `ort::ep::CPU::is_available` is: MLAS
    // is part of every build of the library.
    #[allow(unused_mut)]
    let mut out = vec![OnnxProvider::Cpu];
    #[cfg(any(feature = "cuda", feature = "coreml", feature = "directml"))]
    {
        use ort::ep::ExecutionProvider as _;
        #[cfg(feature = "cuda")]
        if ort::ep::CUDA::default().is_available().map_err(ort_err)? {
            out.push(OnnxProvider::Cuda);
        }
        #[cfg(feature = "coreml")]
        if ort::ep::CoreML::default().is_available().map_err(ort_err)? {
            out.push(OnnxProvider::CoreMl);
        }
        #[cfg(feature = "directml")]
        if ort::ep::DirectML::default()
            .is_available()
            .map_err(ort_err)?
        {
            out.push(OnnxProvider::DirectMl);
        }
    }
    Ok(out)
}

/// A session builder with the resolved provider registered.
///
/// The dispatch is `error_on_failure`: `ort`'s default is to log a failed
/// registration and fall through to the CPU, which is exactly the silent
/// demotion this module refuses at the config layer. A provider that passed
/// [`Execution::resolve`] and then fails to attach is an error.
///
/// # Errors
///
/// Returns an error when the builder cannot be created or the provider cannot
/// be registered.
pub(crate) fn session_builder(
    provider: OnnxProvider,
    graph: &str,
) -> Result<SessionBuilder, String> {
    disable_ort_telemetry();
    Session::builder()
        .map_err(ort_err)?
        .with_execution_providers([dispatch(placement(provider, graph))?])
        .map_err(ort_err)
}

const ORT_DISABLE_TELEMETRY: &str = "ORT_DISABLE_TELEMETRY";

/// Suppress the process-lifetime telemetry enabled in official ONNX Runtime
/// builds before any `ort` call can initialize its global environment.
///
/// This crate deliberately does not call `ort::init`: a library must not take
/// ownership of its host's global logger, thread pool or execution-provider
/// configuration. The environment switch is ONNX Runtime's pre-initialization
/// privacy control and preserves that ownership boundary.
fn disable_ort_telemetry() {
    std::env::set_var(ORT_DISABLE_TELEMETRY, "1");
}

/// Which provider one graph actually runs on.
///
/// Every provider but CoreML is applied to all graphs alike. CoreML is a
/// *placement*: the renderer on CoreML, everything else on CPU. `t3_step` runs
/// once per speech token and CPU does it in 9.8 ms against CoreML's best
/// 17.6 ms; `t3_prefill` and `t3_step` also fail to compile under MLProgram
/// outright, and MLProgram is the only setting worth having. Keeping the
/// generator on CPU is what makes the token stream identical to a CPU run,
/// index for index. The waveform is not bit-identical, which is what the
/// identity contract already says about running the renderer elsewhere.
fn placement(provider: OnnxProvider, graph: &str) -> OnnxProvider {
    if provider == OnnxProvider::CoreMl && !RENDERER_GRAPHS.contains(&graph) {
        return OnnxProvider::Cpu;
    }
    provider
}

fn dispatch(provider: OnnxProvider) -> Result<ExecutionProviderDispatch, String> {
    match provider {
        // `resolve` turns Auto into a concrete provider before anything is
        // built; reaching here with it would mean a caller skipped that step.
        OnnxProvider::Auto => Err("execution provider was not resolved before use".to_string()),
        OnnxProvider::Cpu => Ok(ort::ep::CPU::default().build().error_on_failure()),
        #[cfg(feature = "cuda")]
        OnnxProvider::Cuda => Ok(ort::ep::CUDA::default().build().error_on_failure()),
        #[cfg(not(feature = "cuda"))]
        OnnxProvider::Cuda => Err(not_compiled(OnnxProvider::Cuda)),
        // MLProgram is not a tuning knob. Left at the default (NeuralNetwork)
        // the renderer shatters into hundreds of partitions -- flow_estimator
        // 342, flow_encoder 47, vocoder 51 against 2, 1 and 25 -- and it
        // changes the numbers: a NeuralNetwork vocoder sums 217.70 where CPU
        // sums 211.15, while MLProgram sums 211.149. The fast setting is also
        // the faithful one.
        #[cfg(feature = "coreml")]
        OnnxProvider::CoreMl => Ok(ort::ep::CoreML::default()
            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
            .with_model_cache_dir(coreml_cache_dir().display().to_string())
            .build()
            .error_on_failure()),
        #[cfg(not(feature = "coreml"))]
        OnnxProvider::CoreMl => Err(not_compiled(OnnxProvider::CoreMl)),
        #[cfg(feature = "directml")]
        OnnxProvider::DirectMl => Ok(ort::ep::DirectML::default().build().error_on_failure()),
        #[cfg(not(feature = "directml"))]
        OnnxProvider::DirectMl => Err(not_compiled(OnnxProvider::DirectMl)),
    }
}

#[cfg(not(all(feature = "cuda", feature = "coreml", feature = "directml")))]
fn not_compiled(provider: OnnxProvider) -> String {
    unavailable(provider, false, &[OnnxProvider::Cpu])
}

/// ort errors don't convert to String via `?`; this is the single conversion
/// point. Generic over the recovery type because `with_execution_providers`
/// hands back the builder it failed on.
fn ort_err<R>(e: ort::Error<R>) -> String {
    format!("{e}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_is_disabled_before_onnx_runtime_is_used() {
        std::env::set_var(ORT_DISABLE_TELEMETRY, "0");
        disable_ort_telemetry();
        assert_eq!(std::env::var(ORT_DISABLE_TELEMETRY).as_deref(), Ok("1"));
    }

    #[test]
    fn every_value_round_trips() {
        for p in [
            OnnxProvider::Auto,
            OnnxProvider::Cpu,
            OnnxProvider::Cuda,
            OnnxProvider::CoreMl,
            OnnxProvider::DirectMl,
        ] {
            assert_eq!(OnnxProvider::parse(p.as_str()), Ok(p));
        }
    }

    #[test]
    fn the_five_spellings_are_the_contract() {
        // Spelled out rather than derived: these strings are the cross-port
        // agreement, so a rename has to be a deliberate edit here too.
        assert_eq!(OnnxProvider::Auto.as_str(), "auto");
        assert_eq!(OnnxProvider::Cpu.as_str(), "cpu");
        assert_eq!(OnnxProvider::Cuda.as_str(), "cuda");
        assert_eq!(OnnxProvider::CoreMl.as_str(), "coreml");
        assert_eq!(OnnxProvider::DirectMl.as_str(), "directml");
    }

    #[test]
    fn an_unknown_value_is_refused_and_lists_the_alternatives() {
        let err = OnnxProvider::parse("CUDA").unwrap_err();
        assert!(err.contains("\"CUDA\""), "{err}");
        for want in ["auto", "cpu", "cuda", "coreml", "directml"] {
            assert!(err.contains(want), "{err}");
        }
    }

    #[test]
    fn the_default_is_auto() {
        assert_eq!(ExecutionConfig::default().onnx_provider, OnnxProvider::Auto);
    }

    fn resolve(provider: OnnxProvider, available: &[OnnxProvider]) -> Result<Execution, String> {
        Execution::resolve(
            &ExecutionConfig {
                onnx_provider: provider,
            },
            available,
        )
    }

    #[test]
    fn auto_takes_only_a_provider_a_measurement_backs() {
        let all = [
            OnnxProvider::Cpu,
            OnnxProvider::Cuda,
            OnnxProvider::CoreMl,
            OnnxProvider::DirectMl,
        ];
        let cases = [
            // CUDA leads while it is the one accelerator this project expects
            // to win. The measurement that decides it runs on the GPU box.
            (&all[..], OnnxProvider::Cuda),
            // CoreML and DirectML are available here and auto still refuses
            // them: CoreML measured slower than CPU and moved the tokens, and
            // DirectML has never been run. Asking for either by name works.
            (
                &[
                    OnnxProvider::Cpu,
                    OnnxProvider::CoreMl,
                    OnnxProvider::DirectMl,
                ][..],
                OnnxProvider::Cpu,
            ),
            (&[OnnxProvider::Cpu][..], OnnxProvider::Cpu),
        ];
        for (available, want) in cases {
            let got = resolve(OnnxProvider::Auto, available).unwrap();
            assert_eq!(got.provider(), want, "available: {available:?}");
        }
    }

    #[test]
    fn an_explicit_available_provider_is_taken_as_asked() {
        let got = resolve(
            OnnxProvider::CoreMl,
            &[OnnxProvider::Cpu, OnnxProvider::CoreMl],
        )
        .unwrap();
        assert_eq!(got.provider(), OnnxProvider::CoreMl);
        assert_eq!(got.describe(), "exec[onnx provider=coreml]");
    }

    #[test]
    fn an_explicit_missing_provider_is_an_error_not_a_fallback() {
        let err = resolve(OnnxProvider::Cuda, &[OnnxProvider::Cpu]).unwrap_err();
        assert!(err.contains("cuda"), "{err}");
        assert!(err.contains("this build offers cpu"), "{err}");
    }

    #[test]
    fn the_refusal_names_the_missing_cargo_feature() {
        let err = unavailable(OnnxProvider::Cuda, false, &[OnnxProvider::Cpu]);
        assert!(err.contains("--features cuda"), "{err}");
        assert!(err.contains("ORT_DYLIB_PATH"), "{err}");
        assert!(err.contains("onnxruntime-gpu"), "{err}");
    }

    #[test]
    fn the_refusal_blames_the_library_when_the_feature_is_on() {
        let err = unavailable(OnnxProvider::CoreMl, true, &[OnnxProvider::Cpu]);
        assert!(err.contains("`coreml` cargo feature"), "{err}");
        assert!(err.contains("ORT_DYLIB_PATH"), "{err}");
        assert!(!err.contains("--features"), "{err}");
    }

    #[test]
    fn every_gpu_provider_has_a_library_to_point_at() {
        for p in [
            OnnxProvider::Cuda,
            OnnxProvider::CoreMl,
            OnnxProvider::DirectMl,
        ] {
            assert!(!p.feature().is_empty(), "{p:?}");
            let err = unavailable(p, false, &[OnnxProvider::Cpu]);
            assert!(err.contains(p.library_hint()), "{err}");
        }
    }

    #[test]
    fn an_auto_choice_keeps_the_request_but_prints_the_answer() {
        let got = resolve(OnnxProvider::Auto, &[OnnxProvider::Cpu]).unwrap();
        assert_eq!(got.requested(), OnnxProvider::Auto);
        assert_eq!(got.describe(), "exec[onnx provider=cpu]");
    }

    #[test]
    fn coreml_runs_the_renderer_and_nothing_else() {
        // `t3_step` runs once per speech token: 9.8 ms on CPU against CoreML's
        // best 17.6 ms. Keeping the generator on CPU is also what makes the
        // token stream identical to a CPU run.
        for graph in RENDERER_GRAPHS {
            assert_eq!(placement(OnnxProvider::CoreMl, graph), OnnxProvider::CoreMl);
        }
        for graph in [
            "t3_cond.onnx",
            "t3_prefill.onnx",
            "t3_step.onnx",
            // The enrollment graphs share this builder. The voice encoder
            // decides what a cloned voice sounds like and has never been
            // measured on CoreML, so an allowlist keeps it on CPU.
            "s3_tokenizer.onnx",
            "camp.onnx",
            "voice_encoder.onnx",
            "added_later.onnx",
        ] {
            assert_eq!(placement(OnnxProvider::CoreMl, graph), OnnxProvider::Cpu);
        }
    }

    #[test]
    fn every_other_provider_takes_every_graph() {
        for provider in [
            OnnxProvider::Cpu,
            OnnxProvider::Cuda,
            OnnxProvider::DirectMl,
        ] {
            for graph in ["t3_step.onnx", "vocoder.onnx", "voice_encoder.onnx"] {
                assert_eq!(placement(provider, graph), provider);
            }
        }
    }

    #[cfg(feature = "coreml")]
    #[test]
    fn the_cache_directory_lands_under_loudkit() {
        let dir = coreml_cache_dir();
        assert!(dir.ends_with("loudkit/coreml"), "{}", dir.display());
    }

    #[test]
    fn cpu_is_always_compiled_and_the_rest_follow_their_features() {
        assert!(OnnxProvider::Cpu.is_compiled());
        assert_eq!(OnnxProvider::Cuda.is_compiled(), cfg!(feature = "cuda"));
        assert_eq!(OnnxProvider::CoreMl.is_compiled(), cfg!(feature = "coreml"));
        assert_eq!(
            OnnxProvider::DirectMl.is_compiled(),
            cfg!(feature = "directml")
        );
    }
}
