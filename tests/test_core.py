"""Tests for the parts that must be right before any model is loaded.

Nothing here needs weights, a GPU, or a network. If these pass, the algorithm
layer is sound and any remaining disagreement between backends is in the
execution layer, which is where it is allowed to be.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loudkit.config import (
    AlgorithmConfig,
    ChunkConfig,
    ExecutionConfig,
    ExecutionOverrides,
    SamplingConfig,
    WindowConfig,
)
from loudkit.rng import KAT_VECTORS, gumbel_noise, philox_4x32_10, selftest, uniforms
from loudkit.sampler import LRSamplerV1

REPO = Path(__file__).resolve().parent.parent


class TestPhilox:
    """The RNG is the foundation of cross-backend agreement, so it is checked
    against a published standard rather than against itself."""

    @pytest.mark.parametrize(("ctr", "key", "want"), KAT_VECTORS)
    def test_known_answer_vectors(
        self, ctr: tuple[int, ...], key: tuple[int, int], want: tuple[int, ...]
    ) -> None:
        got = philox_4x32_10(*(np.array([c], dtype=np.uint64) for c in ctr), key[0], key[1])
        assert tuple(int(g[0]) for g in got) == want

    def test_selftest_passes(self) -> None:
        selftest()

    def test_uniforms_are_open_interval(self) -> None:
        """Zero or one would make the Gumbel transform produce an infinity."""
        u = uniforms(seed=1, stream=0, step0=0, n_steps=8, width=512)
        assert u.shape == (8, 512)
        assert (u > 0.0).all()
        assert (u < 1.0).all()

    def test_uniforms_are_addressed_not_streamed(self) -> None:
        """A step's numbers must not depend on the block it was drawn in.

        This is the property that lets one backend generate a block ahead and
        another generate one at a time, and still agree.
        """
        alone = uniforms(seed=42, stream=0, step0=300, n_steps=1, width=64)
        in_block = uniforms(seed=42, stream=0, step0=256, n_steps=256, width=64)[300 - 256]
        np.testing.assert_array_equal(alone[0], in_block)

    def test_streams_are_independent(self) -> None:
        """Sampling, the flow prior and the vocoder must never collide."""
        a = uniforms(seed=5, stream=0, step0=0, n_steps=4, width=64)
        b = uniforms(seed=5, stream=1, step0=0, n_steps=4, width=64)
        assert not np.allclose(a, b)

    def test_uniforms_look_uniform(self) -> None:
        u = uniforms(seed=9, stream=0, step0=0, n_steps=64, width=1024).ravel()
        counts, _ = np.histogram(u, bins=16, range=(0.0, 1.0))
        expected = u.size / 16
        assert np.abs(counts - expected).max() < 0.15 * expected

    def test_gumbel_is_finite(self) -> None:
        g = gumbel_noise(seed=3, stream=0, step0=0, n_steps=16, width=256)
        assert np.isfinite(g).all()


class TestSampler:
    def _cfg(self, **kw: object) -> SamplingConfig:
        return SamplingConfig(**kw)  # type: ignore[arg-type]

    def test_picks_the_dominant_token(self) -> None:
        s = LRSamplerV1(self._cfg(), seed=1)
        logits = np.zeros(64, dtype=np.float32)
        logits[7] = 50.0
        assert s(logits, step=0, seen=np.zeros(64, bool)) == 7

    def test_is_reproducible(self) -> None:
        logits = np.random.default_rng(0).normal(size=256).astype(np.float32) * 3
        seen = np.zeros(256, bool)
        a = [LRSamplerV1(self._cfg(), seed=11)(logits, step=i, seen=seen) for i in range(32)]
        b = [LRSamplerV1(self._cfg(), seed=11)(logits, step=i, seen=seen) for i in range(32)]
        assert a == b

    def test_call_order_does_not_matter(self) -> None:
        """Statelessness, tested rather than asserted: drawing step 5 before
        step 0 must not change either result."""
        logits = np.random.default_rng(1).normal(size=128).astype(np.float32) * 2
        seen = np.zeros(128, bool)
        fwd = LRSamplerV1(self._cfg(), seed=3)
        forward = [fwd(logits, step=i, seen=seen) for i in range(16)]
        rev = LRSamplerV1(self._cfg(), seed=3)
        backward = [rev(logits, step=i, seen=seen) for i in reversed(range(16))][::-1]
        assert forward == backward

    def test_block_boundary_is_invisible(self) -> None:
        logits = np.random.default_rng(2).normal(size=96).astype(np.float32)
        seen = np.zeros(96, bool)
        small = LRSamplerV1(self._cfg(), seed=4, block=8)
        big = LRSamplerV1(self._cfg(), seed=4, block=512)
        assert [small(logits, step=i, seen=seen) for i in range(40)] == [
            big(logits, step=i, seen=seen) for i in range(40)
        ]

    def test_min_p_truncates_the_tail(self) -> None:
        """The logit-space threshold must select exactly what the probability
        form would, which is the whole justification for using it."""
        rng = np.random.default_rng(5)
        logits = (rng.normal(size=512) * 4).astype(np.float32)
        cfg = self._cfg(min_p=0.05, repetition_penalty=1.0)
        s = logits.astype(np.float64) / cfg.temperature
        by_logit = s >= s.max() + np.log(cfg.min_p)
        p = np.exp(s - s.max())
        p /= p.sum()
        by_prob = p >= cfg.min_p * p.max()
        np.testing.assert_array_equal(by_logit, by_prob)

    @staticmethod
    def _rate(sampler: LRSamplerV1, logits: np.ndarray, seen: np.ndarray, token: int) -> float:
        """How often ``token`` is chosen over many steps.

        Rates, not single draws: the penalty shifts a distribution, it does not
        forbid an outcome. With two tokens at logit 10 and a penalty of 1.2, the
        penalised one still wins about one time in nine, so asserting on one
        seed tests the seed rather than the sampler.
        """
        n = 3000
        hits = sum(sampler(logits, step=i, seen=seen) == token for i in range(n))
        return hits / n

    def test_repetition_penalty_applies_to_seen_tokens(self) -> None:
        logits = np.zeros(32, dtype=np.float32)
        logits[[3, 4]] = 10.0
        seen = np.zeros(32, bool)
        seen[3] = True

        s = LRSamplerV1(self._cfg(), seed=11, block=4096)
        penalised = self._rate(s, logits, seen, 3)
        s = LRSamplerV1(self._cfg(), seed=11, block=4096)
        clean = self._rate(s, logits, seen, 4)
        assert penalised < 0.25 < clean, (
            f"a penalised token should lose to its unpenalised twin: "
            f"{penalised:.3f} vs {clean:.3f}"
        )

    def test_silence_tokens_are_exempt(self) -> None:
        """A reader pauses repeatedly; penalising silence removes pauses. This
        was measured at pause ratio 0.112 -> 0.085 on a pause-heavy sentence."""
        logits = np.zeros(32, dtype=np.float32)
        logits[[3, 4]] = 10.0
        seen = np.zeros(32, bool)
        seen[[3, 4]] = True

        exempt = LRSamplerV1(self._cfg(silence_token_ids=(3,)), seed=5, block=4096)
        rate_exempt = self._rate(exempt, logits, seen, 3)
        plain = LRSamplerV1(self._cfg(), seed=5, block=4096)
        rate_plain = self._rate(plain, logits, seen, 3)

        # Exempt: token 3 keeps its logit while 4 is divided down, so it should
        # dominate. Not exempt: both are penalised equally, so it should be even.
        assert rate_exempt > 0.7, f"exempt silence token chosen only {rate_exempt:.3f}"
        assert 0.35 < rate_plain < 0.65, f"unexempt tokens should tie, got {rate_plain:.3f}"

    def test_distribution_matches_the_reference_law(self) -> None:
        """Same law, different stream: the counts must agree within noise."""
        rng = np.random.default_rng(7)
        logits = (rng.normal(size=48) * 2).astype(np.float32)
        cfg = self._cfg(repetition_penalty=1.0)
        seen = np.zeros(48, bool)

        n = 40_000
        s = LRSamplerV1(cfg, seed=123, block=4096)
        got = np.bincount([s(logits, step=i, seen=seen) for i in range(n)], minlength=48) / n

        z = logits.astype(np.float64) / cfg.temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        p = np.where(p < cfg.min_p * p.max(), 0.0, p)
        p /= p.sum()

        tv = np.abs(got - p).sum() / 2
        assert tv < 0.02, f"total variation {tv:.4f} — the law drifted, not just the stream"


class TestAlgorithmConfig:
    def test_default_is_single_path(self) -> None:
        """The shipping default must not be the mode that applies guidance to a
        guidance-distilled estimator."""
        assert AlgorithmConfig().guidance == "single_path"
        assert AlgorithmConfig().guidance_rate == 0.0

    def test_single_path_rejects_a_guidance_rate(self) -> None:
        with pytest.raises(ValueError, match="single_path"):
            AlgorithmConfig(guidance="single_path", guidance_rate=0.7)

    def test_dual_path_rejects_a_zero_rate(self) -> None:
        with pytest.raises(ValueError, match="twice the work"):
            AlgorithmConfig(guidance="cfg_dual_path", guidance_rate=0.0)

    def test_fingerprint_is_stable_and_sensitive(self) -> None:
        a = AlgorithmConfig()
        assert a.fingerprint() == AlgorithmConfig().fingerprint()
        assert a.fingerprint() != a.with_(euler_steps=3).fingerprint()
        assert (
            a.fingerprint() != a.with_(sampling=SamplingConfig(temperature=0.9)).fingerprint()
        )

    def test_fingerprint_ignores_nothing_that_matters(self) -> None:
        """Guidance mode is the value whose silent divergence cost a day."""
        a = AlgorithmConfig()
        b = AlgorithmConfig(guidance="cfg_dual_path", guidance_rate=0.7)
        assert a.fingerprint() != b.fingerprint()

    def test_euler_grid_is_validated(self) -> None:
        with pytest.raises(ValueError, match="points"):
            AlgorithmConfig(euler_steps=2, euler_grid=(0.0, 1.0))
        with pytest.raises(ValueError, match="increasing"):
            AlgorithmConfig(euler_steps=2, euler_grid=(0.0, 0.9, 0.5))
        with pytest.raises(ValueError, match="0.0 to 1.0"):
            AlgorithmConfig(euler_steps=2, euler_grid=(0.1, 0.5, 1.0))
        AlgorithmConfig(euler_steps=2, euler_grid=(0.0, 0.5, 1.0))

    def test_describe_names_the_mode(self) -> None:
        assert "single_path" in AlgorithmConfig().describe()
        assert (
            "cfg@0.7" in AlgorithmConfig(guidance="cfg_dual_path", guidance_rate=0.7).describe()
        )

    def test_sampling_validation(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            SamplingConfig(temperature=0.0)
        with pytest.raises(ValueError, match="rewards repetition"):
            SamplingConfig(repetition_penalty=0.9)
        with pytest.raises(ValueError, match="min_p"):
            SamplingConfig(min_p=1.0)

    def test_window_validation(self) -> None:
        with pytest.raises(ValueError, match="shorter"):
            WindowConfig(max_speech_tokens=255, static_length=128)


class TestExecutionConfig:
    def test_mps_resolves_to_eager_attention(self) -> None:
        """The fused path aborts the process on MPS with no Python traceback,
        so ``auto`` must never choose it there."""
        assert ExecutionConfig(device="mps").resolved_attention() == "eager"
        assert ExecutionConfig(device="cuda").resolved_attention() == "sdpa"
        assert ExecutionConfig(device="cpu").resolved_attention() == "sdpa"

    def test_explicit_attention_is_respected(self) -> None:
        assert ExecutionConfig(device="mps", attention="sdpa").resolved_attention() == "sdpa"

    def test_pre_ampere_cuda_falls_back_to_eager(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """SDPA lowers to flash-attention, which does not exist before Ampere
        (compute 6.x Pascal, 7.x Volta/Turing). On such a GPU the fused path
        raises mid-decode with a traceback naming none of this code; ``auto``
        must choose eager instead of letting the caller crash."""
        import torch

        def cap(*args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return (6, 1)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", cap)
        assert ExecutionConfig(device="cuda").resolved_attention() == "eager"

    def test_ampere_cuda_keeps_sdpa(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import torch

        def cap(*args, **kwargs):  # type: ignore[no-untyped-def]
            del args, kwargs
            return (8, 6)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", cap)
        assert ExecutionConfig(device="cuda").resolved_attention() == "sdpa"

    def test_describe_mentions_precision(self) -> None:
        d = ExecutionConfig(device="cuda", precision={"vocoder": "fp32"}).describe()
        assert "cuda" in d
        assert "vocoder=fp32" in d

    def test_graph_flags_surface_in_describe(self) -> None:
        """cuda_graphs/compile_model are execution tuning knobs; describe() must
        name them so a recorded configuration is the running one."""
        assert "graphs" in ExecutionConfig(device="cuda", cuda_graphs=True).describe()
        assert "compiled" in ExecutionConfig(device="cuda", compile_model=True).describe()
        assert "graphs" not in ExecutionConfig(device="cuda").describe()
        assert "compiled" not in ExecutionConfig(device="cuda").describe()

    @staticmethod
    def _shipping_defaults() -> ExecutionConfig:
        """What a CUDA build resolves to from the manifest: fp16 where measured safe."""
        return ExecutionConfig(
            device="cuda",
            precision={
                "token_generator": "fp16",
                "mel_decoder.estimator": "fp16",
                "mel_decoder.encoder": "fp32",
                "vocoder": "fp32",
            },
        )

    def test_partial_override_preserves_defaults(self) -> None:
        """A caller who names one execution field must not silently reset the
        others to their dataclass defaults — in particular the manifest's fp16
        dtype map, which is what the benchmarks were measured in."""
        from loudkit.backends import _resolve_execution

        merged = _resolve_execution(
            self._shipping_defaults(), ExecutionOverrides(cuda_graphs=True)
        )
        assert merged.cuda_graphs is True
        assert merged.device == "cuda"
        assert merged.precision["token_generator"] == "fp16", (
            "a partial override must inherit the manifest's fp16 map"
        )

    def test_override_equal_to_the_dataclass_default_still_applies(self) -> None:
        """ "Unset" and "set to the default value" are different requests.

        The old merge compared each field against ``ExecutionConfig()`` and
        treated equality as "not specified". So a conformance run that asked
        for an all-fp32 map — which *is* the dataclass default — silently got
        the manifest's fp16 generator and reported fp32, and ``device="cpu"``
        over a CUDA default was ignored for the same reason. Both are the
        single most damaging failure this library can have: a measurement that
        names a configuration it did not run.
        """
        from loudkit.backends import _resolve_execution

        defaults = self._shipping_defaults()
        all_fp32 = {
            "token_generator": "fp32",
            "mel_decoder.estimator": "fp32",
            "mel_decoder.encoder": "fp32",
            "vocoder": "fp32",
        }
        # Every value below equals ExecutionConfig()'s default for its field.
        merged = _resolve_execution(
            defaults,
            ExecutionOverrides(device="cpu", precision=all_fp32, deterministic=True),
        )
        assert merged.device == "cpu"
        assert dict(merged.precision) == all_fp32
        assert merged.deterministic is True

    def test_precision_override_merges_per_module(self) -> None:
        """Naming one module changes that module and no other.

        Replacing the whole map instead would make ``{"vocoder": "fp32"}`` mean
        "and reset the generator to whatever a bare dict lacks", which is how a
        partial dict silently drops a dtype.
        """
        from loudkit.backends import _resolve_execution

        merged = _resolve_execution(
            self._shipping_defaults(), ExecutionOverrides(precision={"vocoder": "fp16"})
        )
        assert merged.precision["vocoder"] == "fp16"
        assert merged.precision["token_generator"] == "fp16"
        assert merged.precision["mel_decoder.encoder"] == "fp32"

    def test_a_full_execution_config_is_taken_as_complete(self) -> None:
        """``ExecutionConfig`` is the configuration; ``ExecutionOverrides`` is a patch.

        Passing the former means "run exactly this", which is what a
        conformance fixture replaying a recorded execution needs. Nothing is
        inherited, so nothing can be inherited by surprise.
        """
        from loudkit.backends import _resolve_execution

        exact = ExecutionConfig(device="cpu")
        assert _resolve_execution(self._shipping_defaults(), exact) is exact

    def test_unset_fields_are_left_alone(self) -> None:
        """An empty override changes nothing at all."""
        from loudkit.backends import _resolve_execution

        defaults = self._shipping_defaults()
        assert _resolve_execution(defaults, ExecutionOverrides()) == defaults
        assert _resolve_execution(defaults, None) == defaults

    def test_static_cache_warns(self) -> None:
        """The static-cache path (cuda_graphs / compile_model) must warn: it is
        the identity contract's ``equivalent`` class, not bit-exact, and the
        user deserves to know before output silently differs from eager."""
        from loudkit.backends import _warn_if_static_cache

        with pytest.warns(RuntimeWarning, match="static KV cache"):
            _warn_if_static_cache(ExecutionConfig(cuda_graphs=True))
        with pytest.warns(RuntimeWarning, match="static KV cache"):
            _warn_if_static_cache(ExecutionConfig(compile_model=True))
        # Default path: silent.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_static_cache(ExecutionConfig(device="cpu"))


class TestFingerprintCoversTheRecipe:
    """Regressions for the four channels a review found could still diverge
    between backends without the fingerprint noticing."""

    def test_recipe_version_is_fingerprinted(self) -> None:
        """The sampling law and framing recipe are code, not settings. Two
        builds agreeing on every field can still compute different things if
        one shipped a new recipe, which is the founding defect by another
        route."""
        a = AlgorithmConfig()
        assert a.fingerprint() != a.with_(recipe_version="loudkit-9").fingerprint()

    def test_describe_names_the_recipe(self) -> None:
        assert "loudkit-1" in AlgorithmConfig().describe()

    def test_manifest_carries_guidance(self) -> None:
        """A teacher checkpoint loading silently as single_path is the founding
        defect with its arrow reversed, and just as invisible."""
        cfg = AlgorithmConfig.from_manifest({"guidance": "cfg_dual_path", "guidance_rate": 0.7})
        assert cfg.guidance == "cfg_dual_path"
        assert cfg.guidance_rate == 0.7
        assert AlgorithmConfig.from_manifest({}).guidance == "single_path"

    def test_manifest_rejects_an_unknown_guidance_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown guidance mode"):
            AlgorithmConfig.from_manifest({"guidance": "sorta_guided"})

    def test_manifest_accepts_only_the_one_recipe(self) -> None:
        """One recipe means one accepted value, and the error names the tag.

        Believing a foreign tag would fingerprint it; defaulting it would claim
        this recipe for a checkpoint that named another. All five ports refuse
        it identically.
        """
        assert (
            AlgorithmConfig.from_manifest({"recipe_version": "loudkit-1"}).recipe_version
            == "loudkit-1"
        )
        assert AlgorithmConfig.from_manifest({}).recipe_version == "loudkit-1"
        with pytest.raises(ValueError, match=r"recipe_version 'loudkit-9'.*only recipe"):
            AlgorithmConfig.from_manifest({"recipe_version": "loudkit-9"})
        # Not even a string: refused, not defaulted. A manifest one port
        # misreads while another defaults is the divergence this library
        # exists to prevent.
        with pytest.raises(ValueError, match="recipe_version '9'"):
            AlgorithmConfig.from_manifest({"recipe_version": 9})

    def test_manifest_carries_chunking(self) -> None:
        """Chunking decides where the reader breathes; it is not a default.

        The parser used to ignore the block entirely, so a checkpoint could
        declare its own boundaries and prefix carry and the runtime would build
        `ChunkConfig()` regardless — while `prefix_tokens` is hashed into the
        fingerprint, so both sides reported agreement they did not have.
        """
        cfg = AlgorithmConfig.from_manifest(
            {
                "chunking": {
                    "enabled": False,
                    "max_tokens": 99,
                    "prefix_tokens": 3,
                    "split_on": ["|"],
                }
            }
        )
        assert cfg.chunking.enabled is False
        assert cfg.chunking.max_tokens == 99
        assert cfg.chunking.prefix_tokens == 3
        assert cfg.chunking.split_on == ("|",)
        # And a manifest that says nothing keeps the shipping recipe.
        assert AlgorithmConfig.from_manifest({}).chunking == ChunkConfig()

    def test_manifest_carries_euler_grid_and_token_rate(self) -> None:
        cfg = AlgorithmConfig.from_manifest(
            {"n_cfm_timesteps": 2, "euler_grid": [0.0, 0.5, 1.0], "token_rate_hz": 50.0}
        )
        assert cfg.euler_grid == (0.0, 0.5, 1.0)
        assert cfg.token_rate_hz == 50.0
        assert AlgorithmConfig.from_manifest({}).euler_grid is None

    def test_a_present_but_malformed_block_fails_the_load(self) -> None:
        """Absent and present-but-wrong are different requests.

        `manifest.get(key) or default` treated them alike, so a truncated or
        hand-edited pack — `window: []`, `sampling_defaults: {}` — loaded
        silently as the defaults. That is a checkpoint running an algorithm
        nobody chose, under a fingerprint asserting it was chosen, which is
        precisely the class of defect this library exists to make impossible.
        """
        for manifest, expect in [
            ({"window": []}, "window"),
            ({"sampling_defaults": []}, "sampling_defaults"),
            ({"chunking": "yes"}, "chunking"),
            ({"eos_floor": 10}, "eos_floor"),
            ({"euler_grid": "cosine"}, "euler_grid"),
            # A string *is* a Sequence, so these two passed the type check and
            # were then iterated character by character: `"123"` became three
            # arbitrary tokens exempted from the repetition penalty and the
            # min_p floor, and `". "` became a breathing recipe that breaks at
            # the full stop inside "Version 3.14" and again at every space.
            # Both loaded without a word, under a fingerprint that faithfully
            # recorded the wrong recipe.
            ({"silence_token_ids": "123"}, "silence_token_ids"),
            ({"chunking": {"split_on": ". "}}, "split_on"),
        ]:
            with pytest.raises(ValueError, match=expect):
                AlgorithmConfig.from_manifest(manifest)

    def test_window_null_means_ragged_and_is_not_an_error(self) -> None:
        """The one non-mapping the key accepts, because it means something."""
        assert AlgorithmConfig.from_manifest({"window": None}).window == WindowConfig()


class TestTF32IsDeclared:
    """PyTorch ships cudnn TF32 on and matmul TF32 off, so "fp32" inherited from
    the defaults is neither fp32 nor bit-reproducible against it. It cost a
    contaminated baseline once; here it is a field with a default and a place in
    describe()."""

    def test_off_by_default(self) -> None:
        assert ExecutionConfig().allow_tf32 is False

    def test_appears_in_describe_either_way(self) -> None:
        assert "tf32=off" in ExecutionConfig().describe()
        assert "tf32=on" in ExecutionConfig(allow_tf32=True).describe()

    def test_is_execution_not_algorithm(self) -> None:
        """It changes numerics but not the reading, and it is a property of the
        hardware — so it belongs in the layer that is allowed to differ."""
        from dataclasses import fields

        assert "allow_tf32" in {f.name for f in fields(ExecutionConfig)}
        assert "allow_tf32" not in {f.name for f in fields(AlgorithmConfig)}


class TestPerStageDevice:
    """The two stages want different hardware, and the README says so. Before
    this existed the claim was documentation of an unimplemented feature —
    `device` was a single value, so on Apple silicon everything went to the GPU,
    which is the slower arrangement for the generator."""

    def test_defaults_to_the_single_device(self) -> None:
        e = ExecutionConfig(device="cuda")
        assert e.resolved_generator_device() == "cuda"
        assert e.resolved_renderer_device() == "cuda"

    def test_split_is_expressible(self) -> None:
        e = ExecutionConfig(device="mps", generator_device="cpu")
        assert e.resolved_generator_device() == "cpu"
        assert e.resolved_renderer_device() == "mps"

    def test_describe_shows_the_split(self) -> None:
        assert (
            "gen=cpu/render=mps"
            in ExecutionConfig(device="mps", generator_device="cpu").describe()
        )
        assert "gen=" not in ExecutionConfig(device="cpu").describe()

    def test_attention_follows_the_generator(self) -> None:
        """The generator owns the attention, and MPS is where the fused path
        aborts the process. A split that put the generator on the CPU should not
        inherit the GPU's workaround."""
        assert (
            ExecutionConfig(device="mps", generator_device="cpu").resolved_attention() == "sdpa"
        )
        assert (
            ExecutionConfig(device="cpu", generator_device="mps").resolved_attention()
            == "eager"
        )

    def test_placement_is_execution_not_algorithm(self) -> None:
        from dataclasses import fields

        names = {f.name for f in fields(ExecutionConfig)}
        assert {"generator_device", "renderer_device"} <= names


class TestTheAssetGate:
    """The switch that decides whether a green run means anything.

    ``LOUDKIT_REQUIRE_ASSETS`` was read in exactly one place — ``requires()``,
    which covers four named large assets. Every other reason a weighted test
    declined to run was an unconditional skip: ``importorskip("onnxruntime")``,
    ``pytest.skip("graphs missing")``. That is how the ONNX backend came to run
    in no CI job at all while ``tests/test_onnx.py``'s own docstring said the
    switch turned its skips into failures.

    These pin the two helpers that closed it, because a gate nothing tests is
    the same shape of problem as a suite nothing runs.
    """

    def test_a_missing_module_is_a_skip_by_default(self, monkeypatch) -> None:
        from . import assets

        monkeypatch.setattr(assets, "REQUIRE_ASSETS", False)
        # `Skipped` is a BaseException, so a bare `pytest.raises(Exception)`
        # lets it through and reports *this* test as skipped — which is the
        # same "looks like a pass" failure the helper exists to prevent.
        with pytest.raises(pytest.skip.Exception, match="not installed"):
            assets.needs_module("a_module_that_does_not_exist")

    def test_the_switch_turns_that_skip_into_a_failure(self, monkeypatch) -> None:
        from . import assets

        monkeypatch.setattr(assets, "REQUIRE_ASSETS", True)
        with pytest.raises(AssertionError, match="LOUDKIT_REQUIRE_ASSETS is set but"):
            assets.needs_module("a_module_that_does_not_exist")

    def test_an_installed_module_is_returned(self) -> None:
        from . import assets

        assert assets.needs_module("json").dumps({"a": 1}) == '{"a": 1}'


class TestPackedAssets:
    """The tokenizer and the lexicon, carried inside the checkpoint.

    A release used to be a 1.2 GB weights file plus a 68 KB `tokenizer.json`
    beside it, plus 6.3 MB of respelling lexicon compiled into each of five
    ports. The weights are content-addressed and immutable; the text files were
    neither, and that gap produced a tokenizer bound only by a digest the
    shipping manifest does not carry, and three ports whose funnels had
    silently diverged.

    Packed as `uint8` tensors under `assets.` rather than in a new container
    format, because every port already has a safetensors reader.
    """

    def _write(self, tmp_path, assets: dict[str, bytes]):
        """A minimal checkpoint carrying `assets`, and nothing else real."""
        import json as _json

        import numpy as _np
        from safetensors.numpy import save_file

        from loudkit.checkpoint import ASSET_PREFIX

        tensors = {"t3.dummy": _np.zeros(2, _np.float32)}
        for name, payload in assets.items():
            tensors[f"{ASSET_PREFIX}{name}"] = _np.frombuffer(payload, dtype=_np.uint8)
        manifest = {"format": "loudkit-checkpoint", "format_version": 1}
        path = tmp_path / "packed.safetensors"
        save_file(tensors, str(path), metadata={"manifest": _json.dumps(manifest)})
        return path

    def test_a_packed_asset_round_trips(self, tmp_path) -> None:
        from loudkit.checkpoint import Checkpoint

        payload = "słowo → word\n".encode()
        ckpt = Checkpoint.open(self._write(tmp_path, {"pl_en_respell.json": payload}))
        assert ckpt.asset("pl_en_respell.json") == payload
        assert ckpt.asset("tokenizer.json") is None, "an absent asset is None, not an error"

    def test_assets_are_not_mistaken_for_weights(self, tmp_path) -> None:
        """`tensors(prefix)` takes everything under a prefix. The asset
        namespace is chosen so a byte blob can never be handed to a module as
        if it were a weight."""
        from loudkit.checkpoint import Checkpoint

        ckpt = Checkpoint.open(self._write(tmp_path, {"tokenizer.json": b"{}"}))
        assert set(ckpt.tensors("t3.")) == {"dummy"}

    def test_the_packed_copy_wins_over_a_sibling(self, tmp_path) -> None:
        """A packed checkpoint is self-contained: it must not read a file that
        happens to sit beside it, because that file is exactly what packing
        exists to stop mattering."""
        from loudkit.checkpoint import Checkpoint

        path = self._write(tmp_path, {"tokenizer.json": b"packed"})
        (tmp_path / "tokenizer.json").write_bytes(b"sibling")
        ckpt = Checkpoint.open(path)
        packed = ckpt.resolve_asset("tokenizer.json", manifest_key="tokenizer_sha256")
        assert packed == b"packed"

    def test_an_unpacked_checkpoint_still_uses_the_verified_sibling(self, tmp_path) -> None:
        """Packs predating the convention keep working, and keep being checked
        when their manifest says what to expect."""
        import hashlib

        from loudkit.checkpoint import Checkpoint

        path = self._write(tmp_path, {})
        (tmp_path / "tokenizer.json").write_bytes(b"sibling")
        ckpt = Checkpoint.open(path)
        got = ckpt.resolve_asset("tokenizer.json", manifest_key="tokenizer_sha256")
        assert got == b"sibling"

        # And the digest is still enforced when the manifest records one.
        ckpt.manifest["tokenizer_sha256"] = hashlib.sha256(b"a different file").hexdigest()
        with pytest.raises(ValueError, match="does not belong to this checkpoint"):
            ckpt.resolve_asset("tokenizer.json", manifest_key="tokenizer_sha256")

    def test_packing_is_idempotent_and_refuses_to_overwrite(self, tmp_path) -> None:
        """Re-packing replaces the assets rather than accumulating them, and
        the input is never rewritten — every measurement in this repository is
        stated against a specific checkpoint, and rewriting one in place
        changes what past results mean."""
        import sys

        sys.path.insert(0, str(REPO / "tools"))
        import pack_assets

        source = self._write(tmp_path, {"tokenizer.json": b"old"})
        lexicon = tmp_path / "pl_en_respell.json"
        lexicon.write_bytes(b'{"respell":{}}')
        (tmp_path / "tokenizer.json").write_bytes(b"new")

        with pytest.raises(ValueError, match="refusing to overwrite"):
            pack_assets.pack(source, source)

        out = tmp_path / "packed2.safetensors"
        monkeyed = dict(pack_assets.ASSETS)
        monkeyed["pl_en_respell.json"] = (lexicon, "pl_en_respell_sha256")
        original, pack_assets.ASSETS = pack_assets.ASSETS, monkeyed
        try:
            pack_assets.pack(source, out)
            again = tmp_path / "packed3.safetensors"
            pack_assets.pack(out, again)
        finally:
            pack_assets.ASSETS = original

        from loudkit.checkpoint import ASSET_PREFIX, Checkpoint

        final = Checkpoint.open(again)
        assert final.asset("tokenizer.json") == b"new", "the asset was refreshed, not stacked"
        names = [k for k in final.keys() if k.startswith(ASSET_PREFIX)]  # noqa: SIM118
        assert sorted(names) == [
            f"{ASSET_PREFIX}pl_en_respell.json",
            f"{ASSET_PREFIX}tokenizer.json",
        ], names
        assert final.manifest["packed_assets"] == ["pl_en_respell.json", "tokenizer.json"]
