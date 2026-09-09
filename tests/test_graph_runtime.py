"""The graph host loop and enrollment DSP without model weights."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from loudkit.backends.graph_enroll import _fbank, _prompt_mel, _token_mel
from loudkit.backends.onnx_backend import ONNXTokenGenerator
from loudkit.config import AlgorithmConfig
from loudkit.manifest import decode_from


def test_graph_modules_import_without_torch() -> None:
    code = """
import sys
class RefuseTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchaudio'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RefuseTorch())
import loudkit
from loudkit.backends import coreml_backend, onnx_backend, graph_enroll
assert graph_enroll._table('s3_mel_filters',128).shape == (128,201)
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
@dataclass
class Profile:
    language: str = "en"
    enrolment: str = "first-10s"
graph_enroll.build_graph_enroller = lambda *a, **kw: SimpleNamespace(
    enroll=lambda *a, **kw: Profile())
voice = loudkit.enroll(
    np.ones(96000, dtype=np.float32) * .1, 'mock', device='coreml',
    language='pl', end_in_silence=True)
assert voice.language == 'pl'
assert voice.enrolment == 'first-10s-pause'
assert 'torch' not in sys.modules

"""
    subprocess.run([sys.executable, "-c", code], check=True, env=os.environ)


@pytest.mark.parametrize("name", ["tokenizer_mel", "matcha_mel", "kaldi_fbank"])
def test_enrollment_dsp_matches_reference(name: str) -> None:
    root = Path(__file__).parent / "data/enrollment"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    wav = np.fromfile(root / "wav16_flow.f32", dtype="<f4")
    actual = {
        "tokenizer_mel": lambda: _token_mel(wav),
        "kaldi_fbank": lambda: _fbank(wav).T,
        "matcha_mel": lambda: _prompt_mel(
            np.fromfile(root / "ref_audio.f32", dtype="<f4")[:240000]
        ),
    }[name]()
    ref = np.fromfile(root / (name + ".f32"), dtype="<f4").reshape(
        manifest["files"][name + ".f32"]["shape"]
    )
    assert actual.shape == ref.shape
    assert np.max(np.abs(actual - ref)) < 0.001
    assert np.corrcoef(actual.ravel(), ref.ravel())[0, 1] > 0.999999


class Session:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[Any] = []

    def run_positional(self, values: Any) -> Any:
        self.calls.append(values)
        return self.result


@pytest.mark.parametrize("tokens", [[11, 12, 13], [11, 8193], [8193], [11, 12, 13, 14]])
def test_pair_loop_counts_tokens_and_stops_either_head(tokens: list[int]) -> None:
    gen = ONNXTokenGenerator.__new__(ONNXTokenGenerator)
    gen.config = AlgorithmConfig(decode_mode=decode_from({"mode": "fusion_mtp2"}))
    logits = np.zeros(gen.config.speech_vocab_size, dtype=np.float32)
    hidden = np.zeros((1, 2), dtype=np.float32)
    gen._head2 = Session([logits[None].copy()])
    gen._step = Session([logits[None].copy(), hidden])
    calls = []

    def sample(row: Any, *, step: int, seen: Any) -> int:
        calls.append((step, bool(np.isneginf(row[8193])), seen.copy()))
        return tokens[step]

    out = gen._generate_pairs(
        logits,
        hidden,
        [],
        np.zeros(len(logits), dtype=bool),
        len(tokens),
        2,
        8193,
        4,
        42,
        sample,
        None,
    )
    assert out == tokens
    assert [call[0] for call in calls] == list(range(len(tokens)))
    assert all(call[1] for call in calls[:2])
    if len(tokens) > 2:
        pair, speech_position, position = gen._step.calls[0]
        assert pair.tolist() == [[11, 12]]
        assert speech_position.tolist() == [3]
        assert position.tolist() == [42]
        assert calls[2][2][11]
        assert calls[2][2][12]


def test_pair_cancellation_before_sample() -> None:
    gen = ONNXTokenGenerator.__new__(ONNXTokenGenerator)
    gen._head2 = Session([])
    assert gen._generate_pairs(None, None, [], None, 4, 0, 8193, 0, 0, None, lambda: True) == []


def test_the_graph_enroller_signs_the_enroller_contract() -> None:
    """`GraphEnroller.enroll` must keep the annotations `VoiceEnroller` declares.

    Nothing at runtime reads them, and an `Any` agrees with every protocol
    without stating anything, so `mypy` cannot hold the class to the contract on
    its own. Comparing the two hint maps here is what does.
    """
    from typing import get_type_hints

    from loudkit.backends.graph_enroll import GraphEnroller
    from loudkit.contracts import VoiceEnroller

    assert issubclass(GraphEnroller, VoiceEnroller)
    assert get_type_hints(GraphEnroller.enroll) == get_type_hints(VoiceEnroller.enroll)


def test_the_graph_enrollers_helpers_are_typed() -> None:
    """`Any` on a helper's input and output is that helper opting out of the gate
    every other module in the package runs under. None of these may."""
    from typing import get_type_hints

    from loudkit.backends import graph_enroll

    for name in ("_table", "_frames", "_spectra", "_token_mel", "_prompt_mel", "_fbank"):
        hints = get_type_hints(getattr(graph_enroll, name))
        assert hints, name
        assert Any not in hints.values(), f"{name} is annotated Any"


@pytest.mark.parametrize("device", ["onnx", "coreml"])
def test_graph_enrollment_matches_portable_profile(device: str) -> None:
    from loudkit.backends.graph_enroll import GraphEnroller

    from .assets import asset, skip_or_fail

    directory = asset("checkpoint").parent / device
    suffix = ".onnx" if device == "onnx" else ".mlpackage"
    if not (directory / ("voice_encoder" + suffix)).exists():
        skip_or_fail(f"enrollment graphs absent: {directory}")
    root = Path(__file__).parent / "data/enrollment"
    profile = GraphEnroller(directory, device).enroll(
        np.fromfile(root / "ref_audio.f32", dtype="<f4"), 24000
    )
    for field in ("prompt_tokens", "cond_prompt_tokens"):
        assert np.array_equal(
            getattr(profile, field), np.fromfile(root / (field + ".i64"), dtype="<i8")
        )
    for field in ("speaker_embedding", "flow_embedding"):
        assert (
            np.corrcoef(
                getattr(profile, field), np.fromfile(root / (field + ".f32"), dtype="<f4")
            )[0, 1]
            > 0.99999
        )


def test_coreml_dynamic_pins_share_bounded_storage() -> None:
    from loudkit.backends.coreml_backend import _PinnedInputs

    class Model:
        def predict(self, data: Any) -> dict:
            assert data["kv"].flags.c_contiguous
            return {}

    pinned = _PinnedInputs(Model())
    for length in range(1, 129):
        pinned.predict({"kv": np.full((1, 2, length, 4), length, dtype=np.float32)})
    bases = {id(value.base): value.base for value in pinned._views.values()}
    assert sum(base.nbytes for base in bases.values()) < 4 * 128 * 2 * 4 * 4
    assert np.all(pinned._buffers["kv"] == 128)
    for length in range(1, 129):
        pinned.predict({"kv": np.zeros((1, 2, length, 4), dtype=np.float32)})
    headers = len(pinned._views)
    for length in range(1, 129):
        pinned.predict({"kv": np.zeros((1, 2, length, 4), dtype=np.float32)})
    assert len(pinned._views) == headers


def test_coreml_renderer_packages_keep_their_measured_placement() -> None:
    """Where each package runs is a measured decision, not a default.

    The estimator is the fp16 pipeline the Neural Engine runs; the encoder and
    the vocoder are fp32 on the CPU. A unit changed by accident moves both the
    speed and the numerics of a render, silently, so the mapping is a diff in a
    test rather than three strings at call sites.
    """
    from loudkit.backends.coreml_backend import _COMPUTE_UNITS

    assert _COMPUTE_UNITS == {
        "flow_encoder.mlpackage": "CPU_ONLY",
        "flow_estimator.mlpackage": "CPU_AND_NE",
        "vocoder.mlpackage": "CPU_ONLY",
    }


def test_coreml_refuses_unexported_precision_before_loading() -> None:
    from loudkit.backends.coreml_backend import build_coreml_engine
    from loudkit.config import ExecutionConfig

    with pytest.raises(ValueError, match="native fp32 or PyTorch fp16"):
        build_coreml_engine(
            None, ExecutionConfig(precision={"token_generator": "bf16"}), AlgorithmConfig()
        )


def test_unknown_decode_refuses_before_loading_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from loudkit.config import ExecutionConfig

    def forbidden(*args: Any) -> None:
        raise AssertionError("session loaded before decode validation")

    monkeypatch.setattr(ONNXTokenGenerator, "_load_session", staticmethod(forbidden))
    with pytest.raises(ValueError, match="unsupported decode"):
        ONNXTokenGenerator(
            SimpleNamespace(decode="unknown"),
            None,
            Path("/absent"),
            execution=ExecutionConfig(),
        )


def test_voice_trim_uses_zero_padding_at_clip_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    from loudkit.backends import graph_enroll

    audio = np.zeros(32000, dtype=np.float32)
    audio[:3000] = 0.11
    audio[8000:12000] = 1
    spectra = graph_enroll._spectra
    trimmed = []

    def capture(value: Any, *args: Any) -> Any:
        trimmed.append(value.copy())
        return spectra(value, *args)

    monkeypatch.setattr(graph_enroll, "_spectra", capture)
    graph_enroll._partials(audio)
    # librosa.effects.trim's constant-padded RMS selects these hop boundaries.
    assert np.array_equal(trimmed[0], audio[1024:13312])


@pytest.mark.parametrize(("mode", "prefix"), [("single", ""), ("fusion_mtp2", "model ")])
def test_parity_labels_preserve_the_base_report(
    monkeypatch: pytest.MonkeyPatch, mode: str, prefix: str
) -> None:
    from tools import parity_table

    from loudkit import checkpoint

    labels = []
    monkeypatch.setattr(
        sys, "argv", ["parity_table.py", "--checkpoint", "model.safetensors", "--out", "-"]
    )
    monkeypatch.setattr(checkpoint, "read_manifest", lambda _path: {"decode": {"mode": mode}})
    monkeypatch.setattr(parity_table, "measure_weight_free", lambda _report: None)
    monkeypatch.setattr(parity_table, "measure_against_reference", lambda *_args: None)
    monkeypatch.setattr(
        parity_table,
        "measure_backend",
        lambda _report, _path, _device, label: labels.append(label),
    )
    monkeypatch.setattr(parity_table, "render", lambda *_args: "ok")
    monkeypatch.setattr(parity_table, "_environment", lambda: "")
    parity_table.main()
    assert labels == [prefix + "ONNX", prefix + "CoreML"]


@pytest.mark.parametrize("explicit", [False, True])
def test_coreml_default_load_uses_exported_precision(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    from types import SimpleNamespace

    import loudkit
    from loudkit import backends, hub
    from loudkit.backends.coreml_backend import build_coreml_engine
    from loudkit.config import ExecutionConfig

    checkpoint = SimpleNamespace(manifest={}, dtype_map={"t3": "float16"})
    monkeypatch.setattr(hub, "resolve_checkpoint", lambda *_a, **_kw: "mock.safetensors")
    monkeypatch.setattr(backends.Checkpoint, "open", lambda _path: checkpoint)
    monkeypatch.setattr(backends, "require_backend", lambda _device: None)
    captured = []

    def builder(ckpt, execution, algorithm):
        if explicit:
            return build_coreml_engine(ckpt, execution, algorithm)
        captured.append(execution.precision_map())
        return "engine"

    monkeypatch.setitem(backends._REGISTRY, "coreml", builder)
    execution = ExecutionConfig(precision={"token_generator": "bf16"}) if explicit else None
    if explicit:
        with pytest.raises(ValueError, match="native fp32 or PyTorch fp16"):
            loudkit.load(
                "mock", device="coreml", execution=execution, algorithm=AlgorithmConfig()
            )
    else:
        assert loudkit.load("mock", device="coreml", algorithm=AlgorithmConfig()) == "engine"
        assert captured == [
            {
                "token_generator": "fp32",
                "mel_decoder.estimator": "fp16",
                "mel_decoder.encoder": "fp32",
                "vocoder": "fp32",
            }
        ]


@pytest.mark.parametrize("partial", [False, True, "recorded"])
def test_legacy_coreml_fallback_only_for_renderer_only_release(tmp_path, monkeypatch, partial):
    from types import SimpleNamespace

    from loudkit import backends
    from loudkit.backends import coreml_backend as coreml
    from loudkit.backends import torch_backend
    from loudkit.config import ExecutionConfig

    assets = tmp_path / "coreml"
    assets.mkdir()
    for name in (coreml.ENCODER_PACKAGE, coreml.ESTIMATOR_PACKAGE, coreml.HIFT_PACKAGE):
        (assets / name).mkdir()
    if partial == "recorded":
        (assets / "export.json").write_text(json.dumps({"packages": {"t3_cond.mlpackage": {}}}))
    elif partial:
        (assets / "t3_cond.mlpackage").mkdir()
    monkeypatch.delenv(coreml.ASSETS_ENV, raising=False)
    ckpt = SimpleNamespace(path=tmp_path / "model.safetensors")

    class LegacySelectedError(Exception):
        pass

    def legacy(*args):
        raise LegacySelectedError

    monkeypatch.setattr(torch_backend, "build_torch_frontend_and_generator", legacy)
    monkeypatch.setattr(backends, "check_export_record", lambda *_a, **_kw: None)
    expected = FileNotFoundError if partial else LegacySelectedError
    with pytest.raises(expected):
        coreml.build_coreml_engine(ckpt, ExecutionConfig(device="coreml"), AlgorithmConfig())


@pytest.mark.parametrize("generator_device", ["cpu", None])
def test_explicit_cpu_placement_selects_torch_with_new_coreml_packages(
    tmp_path, monkeypatch, generator_device
):
    from types import SimpleNamespace

    from loudkit import backends
    from loudkit.backends import coreml_backend as coreml
    from loudkit.backends import torch_backend
    from loudkit.config import ExecutionConfig

    assets = tmp_path / "coreml"
    assets.mkdir()
    for name in (
        coreml.ENCODER_PACKAGE,
        coreml.ESTIMATOR_PACKAGE,
        coreml.HIFT_PACKAGE,
        "t3_cond.mlpackage",
        "t3_prefill.mlpackage",
        "t3_step.mlpackage",
    ):
        (assets / name).mkdir()
    monkeypatch.delenv(coreml.ASSETS_ENV, raising=False)

    class SelectedTorchError(Exception):
        pass

    def selected(_ckpt, execution, _algorithm):
        assert execution.precision_map()["token_generator"] == "fp16"
        raise SelectedTorchError

    monkeypatch.setattr(torch_backend, "build_torch_frontend_and_generator", selected)
    monkeypatch.setattr(backends, "check_export_record", lambda *_a, **_kw: None)
    expected = SelectedTorchError if generator_device == "cpu" else ValueError
    with pytest.raises(expected):
        coreml.build_coreml_engine(
            SimpleNamespace(path=tmp_path / "model.safetensors"),
            ExecutionConfig(
                generator_device=generator_device, precision={"token_generator": "fp16"}
            ),
            AlgorithmConfig(),
        )
