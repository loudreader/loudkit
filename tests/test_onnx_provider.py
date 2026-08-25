"""Which execution provider the ONNX backend runs on, and how it says so.

Every ONNX session was pinned to ``CPUExecutionProvider`` until
``ExecutionConfig.onnx_provider`` existed, so the published throughput figures
described the torch path while an ONNX user got roughly real time. Closing that
gap is only half the job: the other half is that the engine names the provider
it chose, and that an explicit provider this build cannot offer *fails* instead
of quietly measuring cpu under a cuda label.

None of this needs a GPU. onnxruntime's provider list is the one fact these
tests substitute, which is what lets a CUDA-only path be checked on a machine
with no NVIDIA hardware in it. What cannot be checked here — whether a GPU
provider changes the sampled tokens — is a measurement for the conformance
lane, not something this file may assume either way.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from loudkit.backends.onnx_backend import (
    AUTO_ORDER,
    COREML_CACHE_ENV,
    GENERATOR_GRAPHS,
    PROVIDER_NAMES,
    RENDERER_GRAPHS,
    _load_session,
    _ort_module,
    _session_providers,
    resolve_provider,
)
from loudkit.config import DEFAULT_ALGORITHM, ExecutionConfig, ONNXProvider

STATIC = replace(
    DEFAULT_ALGORITHM,
    window=replace(DEFAULT_ALGORITHM.window, static_length=255, static_prompt_tokens=238),
)
"""The framing the exported graphs are static at; the backend refuses any other,
and that refusal runs before the provider is resolved."""

CPU = PROVIDER_NAMES["cpu"]
CUDA = PROVIDER_NAMES["cuda"]
COREML = PROVIDER_NAMES["coreml"]
DML = PROVIDER_NAMES["directml"]


class _FakeInferenceSession:
    """Records the provider list it was handed; runs nothing."""

    calls: list[tuple[str, list[str]]] = []

    def __init__(self, path: str, options: Any, providers: list[str]) -> None:
        del options
        _FakeInferenceSession.calls.append((path, list(providers)))

    def get_inputs(self) -> list[Any]:
        return []

    def get_outputs(self) -> list[Any]:
        return []


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """An onnxruntime whose available providers this test decides.

    Substituted in ``sys.modules`` rather than patched onto the real module,
    because the real one is imported inside the functions under test and the
    point is to run the CUDA and DirectML branches on a machine that has
    neither.
    """

    def install(available: list[str]) -> type[_FakeInferenceSession]:
        _FakeInferenceSession.calls = []
        disable_calls: list[bool] = []
        module = SimpleNamespace(
            get_available_providers=lambda: list(available),
            disable_telemetry_events=lambda: disable_calls.append(True),
            SessionOptions=lambda: SimpleNamespace(
                intra_op_num_threads=0, use_deterministic_compute=False
            ),
            InferenceSession=_FakeInferenceSession,
            _disable_calls=disable_calls,
        )
        monkeypatch.setitem(sys.modules, "onnxruntime", module)
        return _FakeInferenceSession

    return install


def test_onnxruntime_is_disabled_before_loudkit_uses_it(
    fake_ort: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "0")
    fake_ort([CPU])
    module = _ort_module()
    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
    assert module._disable_calls == [True]


class TestAutoResolution:
    @pytest.mark.parametrize(
        ("available", "want"),
        [
            ([CUDA, COREML, DML, CPU], "cuda"),
            # auto refuses CoreML and DirectML even where they are offered:
            # CoreML measured slower than CPU and moved the tokens, DirectML
            # has never been run. Both stay reachable by name.
            ([COREML, DML, CPU], "cpu"),
            ([DML, CPU], "cpu"),
            ([CPU], "cpu"),
            # An unknown provider in the list is not a candidate; ORT ships
            # several (Azure, for one) that this backend has never run on.
            (["AzureExecutionProvider", CPU], "cpu"),
        ],
    )
    def test_auto_takes_the_best_the_build_offers(
        self, fake_ort: Any, available: list[str], want: str
    ) -> None:
        fake_ort(available)
        assert resolve_provider("auto") == want

    def test_preference_order_is_the_documented_one(self) -> None:
        """The four ports resolve ``auto`` against this order. It is a contract,
        not a local preference, so it is asserted rather than inferred."""
        assert AUTO_ORDER == ("cuda", "cpu")

    def test_auto_fails_when_nothing_known_is_offered(self, fake_ort: Any) -> None:
        fake_ort(["AzureExecutionProvider"])
        with pytest.raises(RuntimeError, match="no provider loudkit knows"):
            resolve_provider("auto")


class TestExplicitProvider:
    def test_available_provider_is_taken(self, fake_ort: Any) -> None:
        fake_ort([CUDA, CPU])
        assert resolve_provider("cuda") == "cuda"

    def test_missing_provider_is_an_error_not_a_fallback(self, fake_ort: Any) -> None:
        """The failure this field exists to prevent: a run that says cuda,
        measures cpu, and publishes the number. Raising is the whole behaviour;
        the message says so too, because the caller's next move is to ask for
        ``auto`` if a fallback is what they wanted."""
        fake_ort([CPU])
        with pytest.raises(ValueError, match="never falls"):
            resolve_provider("cuda")

    @pytest.mark.parametrize(
        ("provider", "ort_name", "hint"),
        [
            ("cuda", CUDA, "onnxruntime-gpu"),
            ("coreml", COREML, "macOS"),
            ("directml", DML, "onnxruntime-directml"),
        ],
    )
    def test_the_error_names_the_ask_the_build_and_the_cure(
        self, fake_ort: Any, provider: str, ort_name: str, hint: str
    ) -> None:
        fake_ort([CPU, "AzureExecutionProvider"])
        with pytest.raises(ValueError) as excinfo:
            resolve_provider(provider)  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert provider in message  # what was asked for
        assert ort_name in message  # what onnxruntime calls it
        assert CPU in message  # what this build offers, verbatim
        assert "AzureExecutionProvider" in message
        assert hint in message  # how to get the missing one


ANY_GRAPH = "t3_step.onnx"


class TestSessionProviderList:
    def test_cpu_runs_alone(self, fake_ort: Any) -> None:
        assert _session_providers("cpu", ANY_GRAPH) == [CPU]

    @pytest.mark.parametrize("provider", ["cuda", "directml"])
    @pytest.mark.parametrize("graph", sorted(GENERATOR_GRAPHS | RENDERER_GRAPHS))
    def test_an_accelerator_keeps_cpu_behind_it(
        self, provider: ONNXProvider, graph: str
    ) -> None:
        """The tail is ORT's per-*operator* placement fallback, not a provider
        fallback: availability was settled in resolve_provider, and a graph
        holding one unplaceable op fails to load at all without it.

        Every provider but CoreML is applied to all six graphs alike.
        """
        assert _session_providers(provider, graph) == [PROVIDER_NAMES[provider], CPU]


class TestCoreMLIsAPlacementNotAProvider:
    """CoreML is asked for once and lands on three of the six graphs.

    ``t3_step`` runs once per speech token, and CPU does it in 9.8 ms against
    CoreML's best 17.6 ms; ``t3_prefill`` and ``t3_step`` also fail to compile
    under MLProgram at all. Keeping the generator on CPU is what makes the
    token stream identical to a CPU run.
    """

    @pytest.mark.parametrize("graph", sorted(GENERATOR_GRAPHS))
    def test_the_generator_stays_on_cpu(self, graph: str) -> None:
        assert _session_providers("coreml", graph) == [CPU]

    @pytest.mark.parametrize("graph", ["voice_encoder.onnx", "camp.onnx", "new.onnx"])
    def test_an_unlisted_graph_stays_on_cpu(self, graph: str) -> None:
        """Allowlist, not denylist. A graph nobody has measured on CoreML —
        the enrollment graphs, or one added later — must not inherit an
        accelerator by default. The voice encoder is the case that bites:
        it decides what a cloned voice sounds like."""
        assert _session_providers("coreml", graph) == [CPU]

    @pytest.mark.parametrize("graph", sorted(RENDERER_GRAPHS))
    def test_the_renderer_goes_to_coreml(self, graph: str) -> None:
        placement = _session_providers("coreml", graph)
        assert placement[-1] == CPU
        name, options = placement[0]
        assert name == PROVIDER_NAMES["coreml"]
        assert options["ModelFormat"] == "MLProgram"

    @pytest.mark.parametrize("graph", sorted(RENDERER_GRAPHS))
    def test_mlprogram_is_never_left_to_its_default(self, graph: str) -> None:
        """The default (NeuralNetwork) is not merely slower: it changes the
        numbers. A NeuralNetwork vocoder summed 217.70 where CPU summed 211.15
        and MLProgram summed 211.149."""
        _, options = _session_providers("coreml", graph)[0]
        assert options.get("ModelFormat") == "MLProgram"

    @pytest.mark.parametrize("graph", sorted(RENDERER_GRAPHS))
    def test_a_cache_directory_is_always_named(
        self, graph: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Compiling costs ~130 s. Without a cache that is paid every session,
        so the directory is not optional."""
        monkeypatch.setenv(COREML_CACHE_ENV, str(tmp_path / "cache"))
        _, options = _session_providers("coreml", graph)[0]
        assert options["ModelCacheDirectory"] == str(tmp_path / "cache")

    def test_the_cache_directory_has_a_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(COREML_CACHE_ENV, raising=False)
        _, options = _session_providers("coreml", "vocoder.onnx")[0]
        assert options["ModelCacheDirectory"].endswith("loudkit/coreml")

    def test_the_two_graph_sets_are_disjoint_and_complete(self) -> None:
        assert not (GENERATOR_GRAPHS & RENDERER_GRAPHS)
        assert len(GENERATOR_GRAPHS) == 3
        assert len(RENDERER_GRAPHS) == 3


class TestSessionsUseTheChosenProvider:
    def test_resolved_provider_reaches_onnxruntime(self, fake_ort: Any) -> None:
        recorder = fake_ort([CUDA, CPU])
        _load_session(
            Path("/nonexistent"),
            "t3_step.onnx",
            ExecutionConfig(device="onnx", onnx_provider="cuda"),
        )
        assert recorder.calls[-1][1] == [CUDA, CPU]

    def test_auto_reaches_onnxruntime_resolved(self, fake_ort: Any) -> None:
        """A component built directly — the exporter does this — carries an
        unresolved ``auto`` and must still land on a concrete provider."""
        recorder = fake_ort([CUDA, CPU])
        _load_session(Path("/nonexistent"), "t3_step.onnx", ExecutionConfig(device="onnx"))
        assert recorder.calls[-1][1] == [CUDA, CPU]

    def test_an_unavailable_provider_never_opens_a_session(self, fake_ort: Any) -> None:
        recorder = fake_ort([CPU])
        with pytest.raises(ValueError):
            _load_session(
                Path("/nonexistent"),
                "t3_step.onnx",
                ExecutionConfig(device="onnx", onnx_provider="directml"),
            )
        assert recorder.calls == []


class _StubComponent:
    """Enough of a component for the engine's fingerprint check."""

    def __init__(self, config: Any, *args: Any, execution: ExecutionConfig, **kw: Any) -> None:
        del args, kw
        self.config = config
        self.execution = execution

    def attach_speaker_affine(self, weight: Any, bias: Any) -> None:
        del weight, bias


class _StubCheckpoint:
    file_digest = "0" * 64

    def tensors(self, prefix: str) -> dict[str, Any]:
        del prefix
        return {
            "weight": np.zeros((80, 80), dtype=np.float32),
            "bias": np.zeros(80, dtype=np.float32),
        }

    def resolve_asset(self, name: str, *, manifest_key: str) -> Path:
        del name, manifest_key
        return Path("/nonexistent/tokenizer.json")


@pytest.fixture
def stub_backend(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """``build_onnx_engine`` with the graph loading taken out.

    What is left is the part under test: the provider is resolved once, written
    back into the config the Engine carries, and handed to every component.
    """
    from loudkit.backends import onnx_backend

    monkeypatch.setattr(onnx_backend, "_assets_dir", lambda _ckpt: Path("/nonexistent"))
    monkeypatch.setattr(onnx_backend, "GraphemeTextFrontend", lambda _path: object())
    for name in ("ONNXTokenGenerator", "ONNXMelDecoder", "ONNXVocoder"):
        monkeypatch.setattr(onnx_backend, name, _StubComponent)

    def build(execution: ExecutionConfig) -> Any:
        return onnx_backend.build_onnx_engine(
            _StubCheckpoint(),  # type: ignore[arg-type]
            execution,
            STATIC,
        )

    return build


class TestEngineReportsTheProvider:
    def test_describe_names_the_provider_auto_chose(
        self, fake_ort: Any, stub_backend: Any
    ) -> None:
        """``auto`` is a request, not a record. The engine has to carry the
        answer, or a benchmark row taken on a GPU box is indistinguishable from
        one taken on a laptop."""
        fake_ort([CUDA, CPU])
        engine = stub_backend(ExecutionConfig(device="onnx"))
        assert engine.execution.onnx_provider == "cuda"
        assert "provider=cuda" in engine.describe()

    def test_the_components_are_built_with_the_resolved_provider(
        self, fake_ort: Any, stub_backend: Any
    ) -> None:
        """Not just the engine's copy: a component that resolved ``auto`` for
        itself could open its sessions on a different provider from the one
        describe() reports."""
        fake_ort([CUDA, CPU])
        engine = stub_backend(ExecutionConfig(device="onnx"))
        for component in (engine.token_generator, engine.mel_decoder, engine.vocoder):
            assert component.execution.onnx_provider == "cuda"

    def test_an_unavailable_provider_stops_the_build(
        self, fake_ort: Any, stub_backend: Any
    ) -> None:
        fake_ort([CPU])
        with pytest.raises(ValueError, match="CUDAExecutionProvider"):
            stub_backend(ExecutionConfig(device="onnx", onnx_provider="cuda"))
