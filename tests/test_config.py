"""The execution provider as a configuration value: accepted, refused, reported.

The provider is the first execution knob whose *name* is a cross-language
contract — Python, Rust, Go and TypeScript all take the same five words for the
same concept, so a port that quietly accepts a sixth, or spells one of the five
differently, is the defect this file guards. Nothing here loads onnxruntime;
availability is the backend's question (see ``test_onnx_provider.py``) and this
one only asks whether the value survives validation, merging and ``describe()``
intact.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from loudkit.config import (
    _UNSET,
    ONNX_PROVIDERS,
    AlgorithmConfig,
    ChunkConfig,
    ExecutionConfig,
)
from loudkit.manifest import decode_from, edge_fade_from


class TestONNXProviderValue:
    def test_default_is_auto(self) -> None:
        """The default has to pick the best available provider, because the
        published figures described the torch path while every ONNX session was
        pinned to CPU. Defaulting to cpu would leave that gap open."""
        assert ExecutionConfig().onnx_provider is None
        assert ExecutionConfig().resolved().onnx_provider == "auto"

    def test_accepted_values_are_exactly_the_five(self) -> None:
        assert ONNX_PROVIDERS == ("auto", "cpu", "cuda", "coreml", "directml")

    @pytest.mark.parametrize("provider", ONNX_PROVIDERS)
    def test_every_accepted_value_constructs(self, provider: str) -> None:
        assert ExecutionConfig(onnx_provider=provider).onnx_provider == provider

    @pytest.mark.parametrize("provider", ["CUDA", "metal", "gpu", "CoreML", "", "cpu "])
    def test_unknown_value_is_refused_at_construction(self, provider: str) -> None:
        """The Literal binds the type checker only, and this value arrives from
        CLI flags and JSON bodies. An unvalidated typo reaches the backend as a
        lookup on an onnxruntime symbol, or misses a comparison and runs on cpu
        under a config that says otherwise."""
        with pytest.raises(ValueError, match="unknown onnx_provider"):
            ExecutionConfig(onnx_provider=provider)

    def test_the_error_lists_what_would_have_worked(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ExecutionConfig(onnx_provider="gpu")
        message = str(excinfo.value)
        for name in ONNX_PROVIDERS:
            assert name in message


class TestONNXProviderInDescribe:
    def test_onnx_device_always_names_the_provider(self) -> None:
        """A benchmark row and a bug report both have to say which provider
        ran, and describe() is where both of them read it from."""
        assert "provider=auto" in ExecutionConfig(device="onnx").describe()
        assert "provider=cpu" in ExecutionConfig(device="onnx", onnx_provider="cpu").describe()

    def test_torch_runs_do_not_carry_a_provider(self) -> None:
        """Every other backend ignores the field; printing it there would put a
        provider name on a line where nothing read one."""
        assert "provider=" not in ExecutionConfig(device="cpu").describe()
        assert "provider=" not in ExecutionConfig(device="cuda").describe()

    def test_an_explicit_provider_shows_even_off_the_onnx_device(self) -> None:
        """A caller who named a provider is owed the answer wherever the config
        travels — a full ExecutionConfig can be built before the device that
        will run it is settled."""
        assert "provider=cuda" in ExecutionConfig(device="cpu", onnx_provider="cuda").describe()


class TestONNXProviderOverrides:
    def test_override_wins_over_the_backend_default(self) -> None:
        merged = ExecutionConfig(onnx_provider="cpu").resolved(
            ExecutionConfig(device="onnx", onnx_provider="cuda")
        )
        assert merged.onnx_provider == "cpu"

    def test_unset_override_inherits(self) -> None:
        merged = ExecutionConfig(num_threads=2).resolved(
            ExecutionConfig(device="onnx", onnx_provider="coreml")
        )
        assert merged.onnx_provider == "coreml"

    def test_naming_auto_explicitly_is_not_the_same_as_leaving_it_unset(self) -> None:
        """Asking for auto against a cuda-pinned default means re-resolve, not keep cuda."""
        pinned = ExecutionConfig(device="onnx", onnx_provider="cuda")
        assert ExecutionConfig(onnx_provider="auto").resolved(pinned).onnx_provider == "auto"
        assert ExecutionConfig().resolved(pinned).onnx_provider == "cuda"

    def test_a_bad_override_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown onnx_provider"):
            ExecutionConfig(onnx_provider="metal")


class TestEveryClosedFieldIsChecked:
    """The provider was the only field validated; the rest are Literals too.

    A `Literal` binds the type checker and nothing else, and these values
    arrive from CLI flags and JSON bodies. Measured before this class existed:
    `attention='bogus'`, `device='gpu'`, `num_threads=-4` and
    `precision={'token_generator': 'int8'}` all constructed, and the last one
    surfaced three layers down as `KeyError('int8')`, naming neither the field
    nor the config.
    """

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"device": "gpu"}, "unknown device"),
            ({"generator_device": "nope"}, "unknown generator_device"),
            ({"renderer_device": "xpu"}, "unknown renderer_device"),
            ({"attention": "bogus"}, "unknown attention"),
            ({"precision": {"token_generator": "int8"}}, "unknown precision"),
            ({"num_threads": -4}, "num_threads must be positive"),
            ({"num_threads": 0}, "num_threads must be positive"),
        ],
    )
    def test_a_bad_value_is_refused_at_construction(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            ExecutionConfig(**kwargs)

    @pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "onnx", "coreml", "cuda:1"])
    def test_an_ordinal_names_the_same_backend(self, device: str) -> None:
        """The stem is the field's value; which ordinals exist is a question
        about the machine, and `loudkit._require_device` asks it later."""
        assert ExecutionConfig(device=device).device == device

    def test_an_unknown_precision_module_still_passes(self) -> None:
        """Only the dtype is closed. A backend may carry stages this package
        does not name, and a key no consumer reads changes nothing."""
        assert ExecutionConfig(precision={"a_later_stage": "fp16"}).precision is not None

    def test_the_error_names_the_field_and_lists_the_alternatives(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ExecutionConfig(attention="bogus")
        message = str(excinfo.value)
        assert "attention" in message
        for name in ("auto", "eager", "sdpa"):
            assert name in message

    def test_resolving_a_valid_config_still_works(self) -> None:
        """`resolved()` rebuilds the dataclass, so `__post_init__` runs again
        on every merge; a check that refused its own output would break loading."""
        merged = ExecutionConfig(num_threads=4).resolved()
        assert merged.num_threads == 4
        assert merged.resolved().device == "cpu"


class TestTheOrdinalIsReadOrRefused:
    """The half of a device string `ExecutionConfig` deliberately does not check.

    `ExecutionConfig` validates the stem, because which ordinals exist is a
    question about the machine. `loudkit._require_device` is where that
    question is asked, before any download, and it used to answer it for a
    string it could not read: a non-numeric ordinal was taken as 0, so
    `cuda:abc` walked through the one check that exists to keep it away from
    torch. Nothing here imports torch: the refusal is a string test and runs
    first, which is the point.
    """

    @pytest.mark.parametrize("device", ["cuda:abc", "cuda:", "cuda:1x", "mps:abc"])
    def test_a_non_numeric_ordinal_is_refused(self, device: str) -> None:
        import loudkit

        with pytest.raises(ValueError, match="is not a device ordinal"):
            loudkit._require_device(device)

    @pytest.mark.parametrize("device", ["cpu", "onnx", "coreml", "onnx:whatever"])
    def test_devices_torch_does_not_place_are_left_alone(self, device: str) -> None:
        """Only cuda and mps are torch placements; the graph backends carry
        their own provider strings and are not this function's business."""
        import loudkit

        assert loudkit._require_device(device) is None


class TestTheUnsetSentinelIsInTheAnnotations:
    """Three fields hold a sentinel until their accessor is read, and the
    annotations say so.

    A field annotated `DecodeMode` that answers `False` to
    `== "single"` is an annotation a caller type-checks against and is
    misled by; a type checker reading it stayed silent because the sentinel
    was `Any`. The union is the truth, and the narrow accessors beside each
    field are how the effective value is read.
    """

    @pytest.mark.parametrize(
        ("owner", "field", "admits"),
        [
            (AlgorithmConfig, "decode_mode", "DecodeMode"),
            (AlgorithmConfig, "edge_fade_seconds", "float"),
            (ChunkConfig, "first_chunk_max_tokens", "int"),
        ],
    )
    def test_the_field_admits_the_sentinel(self, owner: type, field: str, admits: str) -> None:
        annotation = owner.__annotations__[field]
        assert "_UnsetType" in annotation, annotation
        assert admits in annotation, annotation

    def test_the_default_really_is_the_sentinel(self) -> None:
        assert isinstance(AlgorithmConfig().decode_mode, type(_UNSET))
        assert isinstance(AlgorithmConfig().edge_fade_seconds, type(_UNSET))
        assert isinstance(ChunkConfig().first_chunk_max_tokens, type(_UNSET))

    def test_the_accessors_are_narrow(self) -> None:
        """What a caller should read instead, and what the annotations there
        promise: a mode, a number, and a count or nothing."""
        algorithm = AlgorithmConfig()
        assert algorithm.decode == "single"
        assert algorithm.edge_fade * 2 == pytest.approx(0.04)
        assert ChunkConfig().resolved_first_chunk_max_tokens() is None

    def test_the_manifest_readers_return_the_same_union(self) -> None:
        """The two readers that produce these values returned `Any`, which is
        assignable to the old untrue annotations and to anything else."""
        hints = get_type_hints(decode_from)
        assert "_UnsetType" in str(hints["return"])
        hints = get_type_hints(edge_fade_from)
        assert "_UnsetType" in str(hints["return"])
