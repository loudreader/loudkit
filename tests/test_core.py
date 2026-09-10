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
    SamplingConfig,
    WindowConfig,
)
from loudkit.rng import KAT_VECTORS, gumbel_noise, philox_4x32_10, selftest, uniforms
from loudkit.sampler import LRSamplerV1

from .conftest import tool

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
        return SamplingConfig(**kw)

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

    def test_silence_is_penalised_like_everything_else(self) -> None:
        """The interior-stall fix. Silence ids were exempt from the penalty;
        together with the min_p exemption that made a silence run absorbing
        (zero escapes in 1,031 instrumented trap steps, a >1 s hole in 33.0%
        of long-form paragraphs). The penalty now covers every seen token, so
        a listed silence id ties its unlisted twin instead of dominating it."""
        logits = np.zeros(32, dtype=np.float32)
        logits[[3, 4]] = 10.0
        seen = np.zeros(32, bool)
        seen[[3, 4]] = True

        listed = LRSamplerV1(self._cfg(silence_token_ids=(3,)), seed=5, block=4096)
        rate_listed = self._rate(listed, logits, seen, 3)
        plain = LRSamplerV1(self._cfg(), seed=5, block=4096)
        rate_plain = self._rate(plain, logits, seen, 3)

        assert 0.35 < rate_listed < 0.65, (
            f"a listed silence token must be penalised like its twin, got {rate_listed:.3f}"
        )
        assert 0.35 < rate_plain < 0.65, f"unlisted tokens should tie, got {rate_plain:.3f}"

    def test_silence_keeps_the_min_p_exemption(self) -> None:
        """The one exemption silence keeps: a pause token is the only way to
        pause, and dropping this one was measured catastrophic (median
        long-form gap 2.46 s -> 4.64 s)."""
        logits = np.full(32, -30.0, dtype=np.float32)
        logits[5] = 10.0
        logits[[3, 4]] = 8.0  # both fall below min_p against token 5
        seen = np.zeros(32, bool)

        sampler = LRSamplerV1(self._cfg(min_p=0.5, silence_token_ids=(3,)), seed=9, block=4096)
        rate_silence = self._rate(sampler, logits, seen, 3)
        sampler = LRSamplerV1(self._cfg(min_p=0.5, silence_token_ids=(3,)), seed=9, block=4096)
        rate_speech_twin = self._rate(sampler, logits, seen, 4)

        # The listed id survives the cutoff and is still drawn now and then;
        # its unlisted twin at the same logit is removed outright.
        assert rate_silence > 0.01, (
            f"the silence id should survive min_p, drawn {rate_silence:.4f}"
        )
        assert rate_speech_twin == 0.0, (
            f"an unlisted id below min_p must never be drawn, got {rate_speech_twin:.4f}"
        )

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


class TestRaggedVocoderDeclaration:
    """The ragged vocoder changes shipped bytes, so what it promises is pinned.

    None of this needs a checkpoint: it is about the declaration, which is the
    part a reader of `COMPATIBILITY.md` relies on and the part that was silently
    wrong twice — once in the CLI, which forced it off for anyone who asked for
    CUDA graphs, and once in the backend, which enabled it on the CPU reference
    path.
    """

    def test_an_explicit_single_hashes_like_an_absent_block(self) -> None:
        """Equal audible decisions must give equal fingerprints, or comparing
        two of them means nothing."""
        from loudkit.manifest import decode_from

        base = AlgorithmConfig()
        explicit = AlgorithmConfig(decode_mode=decode_from({"mode": "single"}))
        assert explicit.fingerprint() == base.fingerprint()
        fused = AlgorithmConfig(decode_mode=decode_from({"mode": "fusion_mtp2"}))
        assert fused.fingerprint() != base.fingerprint()

    def test_no_flag_names_a_knob_its_caller_did_not(self) -> None:
        """`--cuda-graphs` alone must not name `vocoder_ragged`, or `compile_model`:
        an explicit False over a default of True is how every benchmark in one
        campaign measured the padded vocoder while reporting the ragged one."""
        import argparse
        from dataclasses import fields

        bench = tool("bench")
        every = {"cuda_graphs", "compile_model", "vocoder_ragged"}
        for flag, expected in (
            ("cuda_graphs", "cuda_graphs"),
            ("compile", "compile_model"),
            ("vocoder_ragged", "vocoder_ragged"),
        ):
            args = argparse.Namespace(
                cuda_graphs=False,
                compile=False,
                vocoder_ragged=None,
                device="cuda",
                provider=None,
            )
            setattr(args, flag, True)
            execution = bench.execution_for(args)
            named = {
                f.name
                for f in fields(execution)
                if f.name in every and getattr(execution, f.name) is not None
            }
            assert named == {expected}, f"--{flag} also named {named - {expected}}"

    def test_the_padded_vocoder_has_a_spelling(self) -> None:
        """`--no-vocoder-ragged` is the only way to measure the padded vocoder,
        and it reaches the reproduce line."""
        import argparse

        bench = tool("bench")
        ap = argparse.ArgumentParser()
        bench.add_engine_flags(ap)
        ap.add_argument("--texts", nargs="*", default=None)
        common = ["--checkpoint", "c", "--voice", "v"]

        off = ap.parse_args([*common, "--no-vocoder-ragged"])
        assert bench.execution_for(off).vocoder_ragged is False
        assert "--no-vocoder-ragged" in bench.command_line(off)

        on = ap.parse_args([*common, "--vocoder-ragged"])
        assert bench.execution_for(on).vocoder_ragged is True
        assert "--vocoder-ragged" in bench.command_line(on)

        bare = ap.parse_args(common)
        assert bench.execution_for(bare).vocoder_ragged is None
        assert "vocoder-ragged" not in bench.command_line(bare)


class TestReportedExecutionIsRunExecution:
    """`describe()` is the answer to "which engine actually ran".

    It exists because the defect that shaped this library survived an entire
    optimisation campaign for want of exactly this line, and `loudkit bench`
    records it verbatim as the run's `execution` field. A flag it reports that
    the backend declined to build is the same failure wearing the fix's
    clothes.
    """

    def test_ragged_is_claimed_only_where_it_runs(self) -> None:
        from loudkit.config import ExecutionConfig

        # CUDA is the only renderer the torch backend builds a ragged vocoder
        # for, and the graph backends have no TorchVocoder at all.
        assert ExecutionConfig(device="cuda").resolved_vocoder_ragged() is True
        for device in ("cpu", "mps", "onnx", "coreml"):
            execution = ExecutionConfig(device=device)
            assert execution.resolved().vocoder_ragged is True, device
            assert execution.resolved_vocoder_ragged() is False, device
            assert "ragged-vocoder" not in execution.describe(), device

    def test_a_split_engine_is_judged_on_its_renderer(self) -> None:
        """`--device mps` puts the generator on the CPU and the renderer on the
        GPU; the vocoder is a renderer stage, so the renderer decides."""
        from loudkit.config import ExecutionConfig

        split = ExecutionConfig(device="cpu", renderer_device="cuda")
        assert split.resolved_vocoder_ragged() is True
        assert (
            ExecutionConfig(device="cuda", renderer_device="cpu").resolved_vocoder_ragged()
            is False
        )

    def test_turning_it_off_is_reported(self) -> None:
        from loudkit.config import ExecutionConfig

        off = ExecutionConfig(device="cuda", vocoder_ragged=False)
        assert off.resolved_vocoder_ragged() is False
        assert "ragged-vocoder" not in off.describe()

    def test_the_decode_mode_is_named(self) -> None:
        """Two engines running different decode loops printed the same line.

        A version-1 loop run over fusion weights speaks fluent nonsense rather
        than failing, which is the failure this project treats as worse than a
        crash — so the mode belongs on the line a bug report pastes.
        """
        from loudkit.manifest import decode_from

        single = AlgorithmConfig()
        fused = AlgorithmConfig(decode_mode=decode_from({"mode": "fusion_mtp2"}))
        assert "decode=fusion_mtp2" in fused.describe()
        # Absent for the default, like the manifest block: every 0.1.0 log line
        # is the one without it.
        assert "decode=" not in single.describe()


class TestDeviceNoise:
    """The device draw and the NumPy draw are the same draw.

    Philox is integer arithmetic, so the counter-to-bits half is exact by
    construction and is checked against the published known-answer vectors
    below. The Box-Muller half is not bound that tightly — `log` and `cos` are
    where two libms are free to differ in the last bit — so the agreement is
    measured rather than argued. It has held exactly on CPU and CUDA; if a
    future toolchain breaks that, this says so instead of the waveform
    changing quietly.
    """

    def test_the_philox_core_matches_the_published_vectors(self) -> None:
        torch = pytest.importorskip("torch")
        from loudkit.models.noise import _philox_torch
        from loudkit.rng import KAT_VECTORS

        for ctr, key, want in KAT_VECTORS:
            lanes = [torch.tensor([v], dtype=torch.int64) for v in ctr]
            got = _philox_torch(lanes[0], lanes[1], lanes[2], lanes[3], key[0], key[1])
            assert tuple(int(x[0]) for x in got) == want

    @pytest.mark.parametrize(("rows", "cols"), [(9, 4096), (80, 510), (3, 7)])
    def test_the_field_matches_numpy(self, rows: int, cols: int) -> None:
        pytest.importorskip("torch")
        from loudkit.models.noise import gaussian_field, gaussian_field_torch

        want = gaussian_field(7, 4, rows, cols)
        got = np.asarray(gaussian_field_torch(7, 4, rows, cols, "cpu"))
        np.testing.assert_array_equal(got, want)


class TestDeviceSampler:
    """The device form of the law must choose what the host form chooses.

    Run on CPU torch, which is where a difference in the *arithmetic* shows up;
    a difference in the *hardware* is a separate question and the CUDA-graph
    decode's own gate. The point of these is that a second implementation of a
    sampling law is a second thing that can drift, so the drift is measured
    rather than assumed absent.
    """

    def _pair(self, **kw: object):
        cfg = SamplingConfig(
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            silence_token_ids=tuple(range(90, 96)),
            **kw,
        )
        host = LRSamplerV1(cfg, seed=5, stop_token=90, eos_floor=4)
        mirror = LRSamplerV1(cfg, seed=5, stop_token=90, eos_floor=4)
        return host, mirror, mirror.on_device("cpu")

    def test_chooses_the_same_tokens(self) -> None:
        torch = pytest.importorskip("torch")
        width = 128
        host, _mirror, dev = self._pair()
        seen = np.zeros(width, bool)
        dev.prepare(width, seen, floor=4, step=0)
        rng = np.random.default_rng(3)
        for step in range(120):
            logits = (rng.normal(size=width) * 4.0).astype(np.float32)
            if step < 4:
                logits[90] = -np.inf  # the floor, as the decode loop applies it
            want = host(logits, step=step, seen=seen)
            seen[want] = True
            dev.advance_to(step)
            got = int(dev.select(torch.from_numpy(logits).reshape(1, -1)).item())
            assert got == want, f"step {step}: device chose {got}, host chose {want}"

    def test_the_eos_observation_agrees_to_the_last_bits(self) -> None:
        """Not bit-identical, and this is the one place that is true.

        The observation divides by a sum over the survivors, taken left to
        right on the host and as a parallel reduction here. The step it picks
        must still be the same step, and the probability must agree to within
        floating-point noise rather than merely being close.
        """
        torch = pytest.importorskip("torch")
        width = 128
        host, mirror, dev = self._pair()
        seen = np.zeros(width, bool)
        dev.prepare(width, seen, floor=4, step=0)
        rng = np.random.default_rng(11)
        for step in range(120):
            logits = (rng.normal(size=width) * 4.0).astype(np.float32)
            if step < 4:
                logits[90] = -np.inf
            seen[host(logits, step=step, seen=seen)] = True
            dev.advance_to(step)
            dev.select(torch.from_numpy(logits).reshape(1, -1))
        dev.flush_peak()
        host_at, host_prob = host.eos_peak
        dev_at, dev_prob = mirror.eos_peak
        assert host_at == dev_at
        assert abs(host_prob - dev_prob) <= 1e-12 * max(host_prob, 1e-300)

    def test_the_floor_holds_even_with_no_eos_observation(self) -> None:
        """`Engine` builds the sampler without a stop token when the postprocess
        rules are off, and that must not disarm the EOS floor.

        The floor and the observation are different things that happen to name
        the same id: the floor is applied by every host loop regardless, while
        the observation is what postprocess reads. Tying the mask to the
        optional one let the stop token win from the second token onward on a
        build with `postprocess.mode="off"` — a short utterance, on the one path
        where the floor is all that holds the decode open.
        """
        torch = pytest.importorskip("torch")
        width = 32
        cfg = SamplingConfig(temperature=0.8, repetition_penalty=1.0, min_p=0.0)
        host = LRSamplerV1(cfg, seed=5, stop_token=None, eos_floor=6)
        dev = host.on_device("cpu")
        seen = np.zeros(width, bool)
        dev.prepare(width, seen, floor=6, step=0, stop_token=3)

        logits = np.full(width, -8.0, dtype=np.float32)
        logits[3] = 40.0  # the stop token dominates everything
        for step in range(6):
            dev.advance_to(step)
            got = int(dev.select(torch.from_numpy(logits).reshape(1, -1)).item())
            assert got != 3, f"step {step} chose the stop token below the floor"
        dev.advance_to(6)
        assert int(dev.select(torch.from_numpy(logits).reshape(1, -1)).item()) == 3

    def test_a_block_that_would_split_a_pair_is_refused(self) -> None:
        """A fused pair draws twice from one replay, and the second draw's step
        is advanced on the device — so both must be in the resident block. Out
        of range there is a bad read inside a captured graph, which surfaces
        later as an error naming nothing."""
        pytest.importorskip("torch")
        cfg = SamplingConfig(silence_token_ids=())
        dev = LRSamplerV1(cfg, seed=1, block=8).on_device("cpu")
        dev.prepare(16, np.zeros(16, bool), floor=0, step=0)
        dev.advance_to(6, draws=2)  # 6 and 7 are both inside the first block
        with pytest.raises(ValueError, match="cannot hold draws"):
            dev.advance_to(7, draws=2)  # 8 is not


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

    def test_a_renderer_on_mps_is_enough_to_choose_eager(self) -> None:
        """The rule is about the device that runs attention, and both halves do.

        The renderer's flow estimator calls the same fused path, so the split
        that puts the generator on the CPU and the renderer on MPS asked only
        the generator, answered `sdpa`, and handed Metal the kernel this rule
        exists to keep away from it. It renders on one macOS and fails to
        compile the shader on another, which is how CI found it and a laptop
        did not.
        """
        split = ExecutionConfig(generator_device="cpu", renderer_device="mps")
        assert split.resolved_attention() == "eager"
        assert (
            ExecutionConfig(generator_device="mps", renderer_device="cpu").resolved_attention()
            == "eager"
        )

    def test_explicit_attention_is_respected(self) -> None:
        assert ExecutionConfig(device="mps", attention="sdpa").resolved_attention() == "sdpa"

    def test_pre_ampere_cuda_falls_back_to_eager(self, monkeypatch) -> None:
        """SDPA lowers to flash-attention, which does not exist before Ampere
        (compute 6.x Pascal, 7.x Volta/Turing). On such a GPU the fused path
        raises mid-decode with a traceback naming none of this code; ``auto``
        must choose eager instead of letting the caller crash."""
        import torch

        def cap(*args, **kwargs):
            del args, kwargs
            return (6, 1)

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", cap)
        assert ExecutionConfig(device="cuda").resolved_attention() == "eager"

    def test_ampere_cuda_keeps_sdpa(self, monkeypatch) -> None:
        import torch

        def cap(*args, **kwargs):
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
        """Naming one execution field must not reset the others: the manifest's fp16
        map is what the benchmarks were measured in."""
        merged = ExecutionConfig(cuda_graphs=True).resolved(self._shipping_defaults())
        assert merged.cuda_graphs is True
        assert merged.device == "cuda"
        assert merged.precision is not None
        assert merged.precision["token_generator"] == "fp16", (
            "a partial override must inherit the manifest's fp16 map"
        )

    def test_override_equal_to_the_fallback_still_applies(self) -> None:
        """Unset and "set to the value that happens to be the fallback" are different
        requests: an explicit all-fp32 map over an fp16 default must win."""
        defaults = self._shipping_defaults()
        all_fp32 = {
            "token_generator": "fp32",
            "mel_decoder.estimator": "fp32",
            "mel_decoder.encoder": "fp32",
            "vocoder": "fp32",
        }
        merged = ExecutionConfig(device="cpu", precision=all_fp32, deterministic=True).resolved(
            defaults
        )
        assert merged.device == "cpu"
        assert dict(merged.precision or {}) == all_fp32
        assert merged.deterministic is True

    def test_precision_override_merges_per_module(self) -> None:
        """Naming one module changes that module and no other."""
        merged = ExecutionConfig(precision={"vocoder": "fp16"}).resolved(
            self._shipping_defaults()
        )
        assert merged.precision is not None
        assert merged.precision["vocoder"] == "fp16"
        assert merged.precision["token_generator"] == "fp16"
        assert merged.precision["mel_decoder.encoder"] == "fp32"

    def test_unset_fields_are_left_alone(self) -> None:
        """An empty config changes nothing at all, and every field comes back filled."""
        defaults = self._shipping_defaults().resolved()
        assert ExecutionConfig().resolved(defaults) == defaults
        assert defaults.compile_model is False
        assert defaults.deterministic is True

    def test_build_engine_resolves_against_the_manifest(self, tmp_path) -> None:
        """What `build_engine` does with the caller's config: fill it from the
        checkpoint's defaults, so `execution=ExecutionConfig(cuda_graphs=True)` is
        the shipping engine plus graphs and nothing else."""
        from loudkit.backends import _default_execution

        class _Ckpt:
            dtype_map = {"t3": "float16", "s3gen.flow.decoder.estimator": "float16"}

        defaults = _default_execution(_Ckpt(), "cuda", decode="single")
        merged = ExecutionConfig(cuda_graphs=True).resolved(defaults)
        assert merged.cuda_graphs is True
        assert merged.precision is not None
        assert merged.precision["token_generator"] == "fp16"
        assert merged.vocoder_ragged is True

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
        one shipped a new recipe, so the recipe has to be hashed with the
        rest."""
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

    def test_manifest_rejects_the_chunking_words_outside_their_two_sets(self) -> None:
        """The two chunking laws are closed sets, and the refusal names the key.

        The refusal belongs to the manifest reader, not to `ChunkConfig`: a
        word outside the set is a claim the file makes, and the message has to
        say which key made it.
        """
        with pytest.raises(ValueError, match="unknown chunking.cap_resplit 'halve'"):
            AlgorithmConfig.from_manifest({"chunking": {"cap_resplit": "halve"}})
        with pytest.raises(ValueError, match="unknown chunking.mid_sentence_period 'maybe'"):
            AlgorithmConfig.from_manifest({"chunking": {"mid_sentence_period": "maybe"}})

    def test_manifest_keeps_the_words_it_accepts(self) -> None:
        cfg = AlgorithmConfig.from_manifest(
            {"chunking": {"cap_resplit": "off", "mid_sentence_period": "break"}}
        )
        assert cfg.chunking.cap_resplit == "off"
        assert cfg.chunking.mid_sentence_period == "break"

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
        assert ExecutionConfig().resolved().allow_tf32 is False

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

    def test_attention_follows_whichever_half_runs_it(self) -> None:
        """Both halves run attention, so either one on MPS chooses eager.

        This asserted the opposite, on the premise that the generator owns the
        attention and a split with the generator on the CPU should not inherit
        the GPU's workaround. The renderer's flow estimator calls the same
        fused path (`models/flow.py`), so that split handed Metal the kernel
        the rule exists to keep away from it. It renders on one macOS and, on
        the hosted runner, fails to compile the shader and raises out of
        `scaled_dot_product_attention`.
        """
        assert (
            ExecutionConfig(device="mps", generator_device="cpu").resolved_attention()
            == "eager"
        )
        assert (
            ExecutionConfig(device="cpu", generator_device="mps").resolved_attention()
            == "eager"
        )
        assert ExecutionConfig(device="cpu").resolved_attention() == "sdpa"

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
        pack_assets = tool("pack_assets")

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


class TestEdgeFadeIdentity:
    """The 20 ms default is hashed; legacy manifests retain their 5 ms identity."""

    def test_new_default_is_hashed_and_legacy_manifest_retains_its_identity(self) -> None:
        from loudkit.config import EDGE_FADE_SECONDS
        from loudkit.manifest import edge_fade_from

        base = AlgorithmConfig()
        explicit = AlgorithmConfig(edge_fade_seconds=edge_fade_from(EDGE_FADE_SECONDS))
        assert explicit.fingerprint() == base.fingerprint()
        assert '"edge_fade_seconds":"0.02"' in base.canonical_form()
        legacy = AlgorithmConfig(edge_fade_seconds=edge_fade_from(None))
        assert legacy.edge_fade == 0.005
        assert "edge_fade_seconds" not in legacy.canonical_form()
        assert legacy.fingerprint() != base.fingerprint()
        assert base.edge_fade == EDGE_FADE_SECONDS

    def test_another_ramp_moves_the_fingerprint_where_the_ports_expect_it(self) -> None:
        base = AlgorithmConfig()
        other = AlgorithmConfig(edge_fade_seconds=0.008)
        assert other.fingerprint() != base.fingerprint()
        assert other.edge_fade == 0.008
        # The exact bytes every port must emit, at the position sorted keys give.
        want = base.canonical_form().replace(
            '"edge_fade_seconds":"0.02"', '"edge_fade_seconds":"0.008"', 1
        )
        assert other.canonical_form() == want
        assert "fade=0.008s" in other.describe()
        assert "fade=" not in base.describe()

    def test_the_ramp_is_bounded(self) -> None:
        """At least a millisecond: a port whose zero value means unset can then
        never mistake a real ramp for an absent one."""
        for bad in (0.5, 0.0):
            with pytest.raises(ValueError, match="edge_fade_seconds"):
                AlgorithmConfig(edge_fade_seconds=bad)
