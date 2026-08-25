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

import pytest

from loudkit.config import ONNX_PROVIDERS, ExecutionConfig, ExecutionOverrides


class TestONNXProviderValue:
    def test_default_is_auto(self) -> None:
        """The default has to pick the best available provider, because the
        published figures described the torch path while every ONNX session was
        pinned to CPU. Defaulting to cpu would leave that gap open."""
        assert ExecutionConfig().onnx_provider == "auto"

    def test_accepted_values_are_exactly_the_five(self) -> None:
        assert ONNX_PROVIDERS == ("auto", "cpu", "cuda", "coreml", "directml")

    @pytest.mark.parametrize("provider", ONNX_PROVIDERS)
    def test_every_accepted_value_constructs(self, provider: str) -> None:
        assert ExecutionConfig(onnx_provider=provider).onnx_provider == provider  # type: ignore[arg-type]

    @pytest.mark.parametrize("provider", ["CUDA", "metal", "gpu", "CoreML", "", "cpu "])
    def test_unknown_value_is_refused_at_construction(self, provider: str) -> None:
        """The Literal binds the type checker only, and this value arrives from
        CLI flags and JSON bodies. An unvalidated typo reaches the backend as a
        lookup on an onnxruntime symbol, or misses a comparison and runs on cpu
        under a config that says otherwise."""
        with pytest.raises(ValueError, match="unknown onnx_provider"):
            ExecutionConfig(onnx_provider=provider)  # type: ignore[arg-type]

    def test_the_error_lists_what_would_have_worked(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            ExecutionConfig(onnx_provider="gpu")  # type: ignore[arg-type]
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
        merged = ExecutionOverrides(onnx_provider="cpu").applied_to(
            ExecutionConfig(device="onnx", onnx_provider="cuda")
        )
        assert merged.onnx_provider == "cpu"

    def test_unset_override_inherits(self) -> None:
        merged = ExecutionOverrides(num_threads=2).applied_to(
            ExecutionConfig(device="onnx", onnx_provider="coreml")
        )
        assert merged.onnx_provider == "coreml"

    def test_naming_auto_explicitly_is_not_the_same_as_leaving_it_unset(self) -> None:
        """The whole reason ExecutionOverrides exists: "unset" and "set to the
        value that happens to be the default" are different requests. Asking
        for auto against a cuda-pinned default means *re-resolve*, not *keep
        cuda*."""
        pinned = ExecutionConfig(device="onnx", onnx_provider="cuda")
        named = ExecutionOverrides(onnx_provider="auto").applied_to(pinned)
        assert named.onnx_provider == "auto"
        assert ExecutionOverrides().applied_to(pinned).onnx_provider == "cuda"

    def test_a_bad_override_is_refused_when_it_is_applied(self) -> None:
        with pytest.raises(ValueError, match="unknown onnx_provider"):
            ExecutionOverrides(onnx_provider="metal").applied_to(  # type: ignore[arg-type]
                ExecutionConfig(device="onnx")
            )

    def test_describe_names_only_what_was_set(self) -> None:
        assert "onnx_provider='cuda'" in ExecutionOverrides(onnx_provider="cuda").describe()
        assert "onnx_provider" not in ExecutionOverrides(num_threads=1).describe()
