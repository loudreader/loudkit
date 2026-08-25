"""Tests for the model layer that need no weights.

Everything here checks algorithm-layer behaviour — window framing, the Euler
grid, the EOS floor, noise addressing, the precision refusals — which is
exactly the layer whose silent divergence has cost this project real time.
If these pass, a parity failure (test_parity.py) is arithmetic, not recipe.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from loudkit.config import AlgorithmConfig, SamplingConfig, WindowConfig
from loudkit.models.generator import eos_floor
from loudkit.models.noise import gaussian_field, symmetric_uniforms
from loudkit.voice import VoiceProfile

from .assets import asset as _asset

CHATTERBOX_TOKENIZER = Path(str(_asset("tokenizer")))

PRODUCTION_WINDOW = WindowConfig(
    max_speech_tokens=255, static_length=255, pad_token_id=4254, static_prompt_tokens=238
)


def _voice(prompt_tokens: int = 250, prompt_frames: int = 500) -> VoiceProfile:
    rng = np.random.default_rng(0)
    return VoiceProfile(
        name="test",
        speaker_embedding=rng.normal(size=256).astype(np.float32),
        flow_embedding=rng.normal(size=192).astype(np.float32),
        prompt_tokens=rng.integers(0, 6561, size=prompt_tokens).astype(np.int64),
        prompt_mel=rng.normal(size=(80, prompt_frames)).astype(np.float32),
        cond_prompt_tokens=rng.integers(0, 6561, size=150).astype(np.int64),
    )


class TestWindowFraming:
    """The recipe that was the entire measured ANE-vs-torch mel deviation."""

    def _cfg(self, window: WindowConfig) -> AlgorithmConfig:
        return AlgorithmConfig(window=window)

    def test_static_recipe_shapes(self) -> None:
        from loudkit.models.flow import frame_windows

        voice = _voice()
        row, cond, prompt_frames, n = frame_windows(
            self._cfg(PRODUCTION_WINDOW), [1, 2, 3], voice
        )
        assert row.shape == (1, 238 + 255)
        assert cond.shape == (1, 80, 2 * (238 + 255))
        assert prompt_frames == 476
        assert n == 3

    def test_prompt_truncates_and_query_pads_with_silence(self) -> None:
        from loudkit.models.flow import frame_windows

        voice = _voice(prompt_tokens=250)
        row, cond, _, _ = frame_windows(self._cfg(PRODUCTION_WINDOW), [1, 2, 3], voice)
        # prompt: first 238 of the 250 enrolled tokens, no padding
        np.testing.assert_array_equal(row[0, :238], voice.prompt_tokens[:238])
        # query: 3 real tokens then the silence unit, never token 0 — an
        # ordinary speech unit there bleeds +3 dB HF into the tail
        assert list(row[0, 238:241]) == [1, 2, 3]
        assert (row[0, 241:] == 4254).all()
        # the mel condition holds exactly the prompt window and zeros after
        assert (cond[0, :, :476] == voice.prompt_mel[:, :476]).all()
        assert (cond[0, :, 476:] == 0).all()

    def test_short_prompt_pads_with_silence(self) -> None:
        from loudkit.models.flow import frame_windows

        voice = _voice(prompt_tokens=100, prompt_frames=200)
        row, cond, _, _ = frame_windows(self._cfg(PRODUCTION_WINDOW), [7], voice)
        assert (row[0, 100:238] == 4254).all()
        assert (cond[0, :, 200:476] == 0).all()

    def test_ragged_mode_keeps_natural_lengths(self) -> None:
        from loudkit.models.flow import frame_windows

        voice = _voice(prompt_tokens=250, prompt_frames=500)
        row, cond, prompt_frames, n = frame_windows(
            self._cfg(WindowConfig()), list(range(40)), voice
        )
        assert row.shape == (1, 250 + 40)
        assert prompt_frames == 500
        assert n == 40

    def test_over_window_is_refused_rather_than_trimmed(self) -> None:
        """The end of a passage must not vanish while the audio still sounds fine.

        This used to assert the truncation — 400 tokens in, 255 out, no error —
        which is silent data loss noticed only by a listener who knows the text.
        Rust, Go, JS and Swift all refuse it in this same function, each with a
        comment saying so, and `Engine` refuses it one layer up; Python's
        low-level path was the last one that still cut. A caller reaching
        `MelDecoder.decode` directly is exactly the caller with no other layer
        to catch it.
        """
        from loudkit.errors import WindowOverflowError
        from loudkit.models.flow import frame_windows

        with pytest.raises(WindowOverflowError):
            frame_windows(self._cfg(PRODUCTION_WINDOW), list(range(400)), _voice())

        # And the window's own capacity still frames cleanly.
        _, _, _, n = frame_windows(self._cfg(PRODUCTION_WINDOW), list(range(255)), _voice())
        assert n == 255

    def test_pad_token_falls_back_to_silence_list_then_refuses(self) -> None:
        from loudkit.models.flow import pad_token_id

        no_pad = PRODUCTION_WINDOW.__class__(
            max_speech_tokens=255, static_length=255, static_prompt_tokens=238
        )
        cfg = AlgorithmConfig(
            window=no_pad, sampling=SamplingConfig(silence_token_ids=(1731, 4254))
        )
        assert pad_token_id(cfg) == 1731
        with pytest.raises(ValueError, match="pad token"):
            pad_token_id(AlgorithmConfig(window=no_pad))


class TestTimeGrid:
    def test_cosine_grid_matches_the_shipped_formula(self) -> None:
        """t_i = 1 − cos(i/K·π/2): what the students were distilled against
        and what the Swift engine computes. Not the linear grid the upstream
        meanflow branch integrates — that was a torch-side deviation."""
        from loudkit.models.flow import time_grid

        grid = time_grid(AlgorithmConfig(euler_steps=2))
        assert grid[0] == 0.0
        assert grid[1] == pytest.approx(1.0 - math.cos(math.pi / 4), abs=1e-12)
        assert grid[2] == pytest.approx(1.0, abs=1e-12)

    def test_explicit_grid_wins(self) -> None:
        from loudkit.models.flow import time_grid

        cfg = AlgorithmConfig(euler_steps=2, euler_grid=(0.0, 0.25, 1.0))
        assert time_grid(cfg) == [0.0, 0.25, 1.0]


class TestEOSFloor:
    def test_matches_the_shipped_integer_arithmetic(self) -> None:
        """The Swift runner computes ``max(10, textIds * 6 / 5)`` in integers;
        the float form must never round differently on any real length."""
        cfg = AlgorithmConfig(
            sampling=SamplingConfig(min_tokens_floor=10, min_tokens_text_ratio=1.2)
        )
        for n_text in range(0, 600):
            assert eos_floor(n_text, cfg) == max(10, n_text * 6 // 5), n_text

    def test_disabled_by_default(self) -> None:
        assert eos_floor(100, AlgorithmConfig()) == 0


class TestNoise:
    """Render randomness is data: addressed, independent, and clean at Nyquist."""

    def test_gaussian_field_is_addressed(self) -> None:
        a = gaussian_field(7, 0, 4, 64)
        b = gaussian_field(7, 0, 4, 64)
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, gaussian_field(8, 0, 4, 64))
        assert not np.allclose(a, gaussian_field(7, 2, 4, 64))

    def test_gaussian_field_moments(self) -> None:
        z = gaussian_field(3, 0, 80, 2048).ravel()
        assert abs(z.mean()) < 0.01
        assert abs(z.std() - 1.0) < 0.01
        assert np.isfinite(z).all()

    def test_no_nyquist_structure(self) -> None:
        """The cached-spare Box–Muller variant puts a period-2 artefact exactly
        on Nyquist (+5.3 dB measured); fresh pairs must not."""
        z = gaussian_field(11, 0, 1, 1 << 16)[0].astype(np.float64)
        power = np.abs(np.fft.rfft(z)) ** 2
        nyquist = power[-1]
        mean_power = power[1:-1].mean()
        assert nyquist < 6.0 * mean_power  # generous; the defect was ~3.4x

    def test_symmetric_uniforms_bounds(self) -> None:
        u = symmetric_uniforms(5, 0, 4096, math.pi)
        assert (np.abs(u) < math.pi).all()
        assert abs(u.mean()) < 0.1


class TestPrecisionRefusals:
    def test_vocoder_refuses_fp16(self) -> None:
        """A cumulative phase accumulator at ~1400 cycles cannot live in fp16;
        the failure is an audible Nyquist tone, so the refusal is loud."""
        import torch

        from loudkit.models.vocoder import TorchVocoder

        voc = TorchVocoder(AlgorithmConfig())
        with pytest.raises(TypeError, match="Nyquist"):
            voc.half()
        with pytest.raises(TypeError, match="Nyquist"):
            voc.to(torch.float16)

    def test_backend_rejects_fp16_flow_encoder(self) -> None:
        from loudkit.backends.torch_backend import _check_precision
        from loudkit.config import ExecutionConfig

        bad = ExecutionConfig(precision={"mel_decoder.encoder": "fp16"})
        with pytest.raises(ValueError, match="mel corr 0.619"):
            _check_precision(bad)

    def test_a_swapped_tokenizer_is_refused_when_the_manifest_names_one(self, tmp_path) -> None:
        """The one artefact mismatch no fingerprint can see.

        The tokenizer is a separate file resolved by name from the checkpoint's
        directory. Swapping it for another valid one changes the text ids, the
        speech, and possibly where EOS lands — while
        ``AlgorithmConfig.fingerprint()`` does not move, because a tokenizer is
        not part of the algorithm config and ``TextFrontend`` carries no config
        for ``_assert_one_algorithm`` to compare. Two different readings, one
        reported identity.
        """
        from loudkit.checkpoint import Checkpoint, file_sha256

        tok = tmp_path / "tokenizer.json"
        tok.write_text('{"vocab": {}}', encoding="utf-8")
        digest = file_sha256(tok)

        def ckpt(manifest: dict) -> Checkpoint:
            return Checkpoint(path=tmp_path / "ckpt.safetensors", manifest=manifest)

        # Matching digest: resolves normally.
        assert (
            ckpt({"tokenizer_sha256": digest}).verified_sibling(
                "tokenizer.json", manifest_key="tokenizer_sha256"
            )
            == tok
        )

        # Swapped file, same name: refused, and the message says why.
        tok.write_text('{"vocab": {"a": 1}}', encoding="utf-8")
        with pytest.raises(ValueError, match="does not belong to this checkpoint"):
            ckpt({"tokenizer_sha256": digest}).verified_sibling(
                "tokenizer.json", manifest_key="tokenizer_sha256"
            )

        # A manifest that records no digest cannot have its expectation
        # checked; packs predating the field must still load.
        assert (
            ckpt({}).verified_sibling("tokenizer.json", manifest_key="tokenizer_sha256") == tok
        )

        # But a manifest that *does* record one and finds the file missing is a
        # broken release, not an optional extra.
        tok.unlink()
        with pytest.raises(FileNotFoundError, match="part of this release"):
            ckpt({"tokenizer_sha256": digest}).verified_sibling(
                "tokenizer.json", manifest_key="tokenizer_sha256"
            )

    def test_a_component_without_a_config_is_refused(self) -> None:
        """Skipping the check is the hole the check exists to close.

        ``_assert_one_algorithm`` used to `continue` past a component exposing
        no ``config``. Such a component can compute anything at all while every
        fingerprint the engine reports still agrees — which is the founding
        defect with an extra step.
        """
        from loudkit.config import AlgorithmConfig
        from loudkit.engine import Engine

        algo = AlgorithmConfig()

        class _NoConfigVocoder:
            def synthesize(self, mel, voice, *, seed):  # pragma: no cover - never called
                raise AssertionError("unreachable")

        class _Ok:
            def __init__(self) -> None:
                self.config = algo

            def __getattr__(self, name):  # pragma: no cover - never called
                raise AssertionError("unreachable")

        with pytest.raises(ValueError, match="exposes no `config`"):
            Engine(
                frontend=object(),  # type: ignore[arg-type]
                token_generator=_Ok(),  # type: ignore[arg-type]
                mel_decoder=_Ok(),  # type: ignore[arg-type]
                vocoder=_NoConfigVocoder(),  # type: ignore[arg-type]
                algorithm=algo,
            )

    def test_onnx_refuses_a_guidance_mode_it_does_not_implement(self, tmp_path) -> None:
        """A backend may not accept an algorithm it silently does not run.

        The ONNX decode loop calls the estimator once per step and never forms
        ``(1+w)·v_cond − w·v_uncond``. It used to accept a ``cfg_dual_path``
        algorithm anyway and compute single-path — plausible audio, wrong
        maths, and a *matching* fingerprint, because the component carries the
        very config it is disobeying. ``_assert_one_algorithm`` compares
        components to each other, so it cannot catch a component lying about
        itself; only the component can refuse.

        Asserted before any asset is touched: the error must name guidance, not
        a missing graph file, or a user with no exports gets the wrong
        diagnosis for a real algorithm mismatch.
        """
        from loudkit.backends.onnx_backend import ONNXMelDecoder
        from loudkit.config import AlgorithmConfig, ExecutionConfig

        dual = AlgorithmConfig().with_(guidance="cfg_dual_path", guidance_rate=0.7)
        with pytest.raises(ValueError, match="single_path"):
            ONNXMelDecoder(dual, tmp_path, execution=ExecutionConfig(device="onnx"))

        # CoreML refuses the same mode; the two backends must not disagree
        # about which algorithms they can honour.
        coreml = pytest.importorskip("loudkit.backends.coreml_backend")
        with pytest.raises(ValueError, match="cfg_dual_path"):
            coreml.CoreMLMelDecoder(dual, encoder=None, estimator=None)


@pytest.mark.skipif(not CHATTERBOX_TOKENIZER.exists(), reason="tokenizer asset not present")
class TestTextFrontend:
    def _frontend(self):  # type: ignore[no-untyped-def]
        from loudkit.frontend.text import GraphemeTextFrontend

        return GraphemeTextFrontend(CHATTERBOX_TOKENIZER)

    def test_deterministic(self) -> None:
        fe = self._frontend()
        a = fe.encode("The quick brown fox.", "en")
        b = fe.encode("The quick brown fox.", "en")
        np.testing.assert_array_equal(a, b)
        assert a.dtype == np.int64

    def test_language_tag_changes_ids(self) -> None:
        fe = self._frontend()
        assert fe.encode("dom", "en").tolist() != fe.encode("dom", "pl").tolist()

    def test_model_based_languages_are_refused(self) -> None:
        """Still a `NotImplementedError` — the base is kept so nothing that
        caught the builtin breaks — but a named one, so the HTTP server can
        tell this apart from a backend method nobody finished."""
        from loudkit.errors import UnsupportedLanguageError

        fe = self._frontend()
        with pytest.raises(NotImplementedError, match="zh") as exc:
            fe.encode("你好", "zh")
        assert isinstance(exc.value, UnsupportedLanguageError)
        assert exc.value.language == "zh"
        # The specific reason survives the move to an allowlist: these five are
        # refused for a knowable cause, not merely for being off the roster.
        assert "model-based" in str(exc.value)

    def test_the_roster_is_an_allowlist_not_a_blacklist(self) -> None:
        """A tag the tokenizer knows is not a language the kit can speak.

        The vocabulary carries tags for 31 languages; the text layer is written
        for twelve. While this was a blacklist of zh/ja/he/ko/ru the other 26
        went straight through — `encode(text, "bg")` NFKD-mangled Cyrillic into
        ids the model reads as sounds it never learned, with no error and
        plausible-sounding audio. Once `UnsupportedLanguageError` started
        advertising `.supported`, it advertised those 26 as well: a client
        refused for `zh` would read the list and retry into the same trap.
        """
        from loudkit.errors import UnsupportedLanguageError
        from loudkit.frontend.numbers import supported_languages

        fe = self._frontend()
        with pytest.raises(UnsupportedLanguageError) as exc:
            fe.encode("Добър ден", "bg")
        assert exc.value.language == "bg"
        # Equality, not membership: one roster, one authority. `supported` used
        # to be derived from the tokenizer vocabulary, which is a different set
        # and included the non-ISO tag "ea".
        assert exc.value.supported == supported_languages()
        assert len(exc.value.supported) == 12

    def test_every_language_on_the_roster_encodes(self) -> None:
        """The other half of the allowlist: it must not refuse what it ships."""
        from loudkit.frontend.numbers import supported_languages

        fe = self._frontend()
        for lang in supported_languages():
            assert fe.encode("jeden dwa trzy", lang).size > 0, lang


class TestGeneratorContract:
    def test_attention_mode_is_validated(self) -> None:
        from loudkit.models.generator import TorchTokenGenerator

        with pytest.raises(ValueError, match="attention"):
            TorchTokenGenerator(
                AlgorithmConfig(),
                {
                    "hidden_size": 64,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "intermediate_size": 128,
                    "head_dim": 32,
                },
                attention="flash",
            )

    def test_construction_is_deterministic_and_finite(self) -> None:
        """``torch.manual_seed(s)`` must actually pin a build.

        ``_Perceiver.pre_attention_query`` was the one parameter in the file
        with no initialiser: ``nn.Parameter(torch.empty(...))`` reads whatever
        was in the recycled allocation. The checkpoint overwrites it, so
        nothing noticed at runtime — but two seeded constructions differed, and
        on a heap that had recently held NaNs the model's logits came out NaN.
        That surfaced as ``TestStaticCacheDecode`` failing about one run in
        three with ``drift = nan`` while passing when run alone.

        The heap is dirtied deliberately: an uninitialised read is invisible on
        a clean allocator and obvious on a used one, which is why this has to
        be provoked rather than waited for.
        """
        from loudkit.models.generator import TorchTokenGenerator

        llama = {
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 128,
            "head_dim": 32,
        }
        junk = [torch.full((1, 32, 64), float("nan")) for _ in range(64)]
        del junk

        torch.manual_seed(0)
        a = TorchTokenGenerator(AlgorithmConfig(), llama, attention="eager")
        torch.manual_seed(0)
        b = TorchTokenGenerator(AlgorithmConfig(), llama, attention="eager")

        for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
            assert torch.isfinite(pa).all(), f"{name} is not finite before any weights load"
            assert torch.equal(pa, pb), f"{name} differs between two seeded constructions"


class TestStaticCacheDecode:
    """The ``cuda_graphs`` path, checked on the CPU where it runs the same
    static-cache math eagerly. The identity contract's ``equivalent`` class:
    the padded attention reduction changes the reduction order, which on CUDA
    can switch the cuBLAS kernel at large widths and drift logits enough to
    flip a sampled token on long sequences. Two properties are load-bearing
    and both are gates:

    * **Determinism.** Same seed, same build must give the same tokens —
      every time, even when the path diverges from the dynamic one.
    * **Short-sequence identity.** On short sequences (and CPU, where the
      matmul is order-independent) the static path is token-identical to the
      dynamic one. This is the regression guard: it catches a *bug* in the
      static path, distinct from the sanctioned equivalent-class drift.
    """

    def _generators(self, cuda_graphs: bool = False, config: AlgorithmConfig | None = None):
        from loudkit.models.generator import TorchTokenGenerator

        llama = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 128,
            "head_dim": 32,
        }
        cfg = config or AlgorithmConfig()
        g = TorchTokenGenerator(cfg, llama, attention="eager", cuda_graphs=cuda_graphs)
        g.eval()
        for p in g.parameters():
            p.requires_grad_(False)
        return g, cfg

    def _voice(self) -> VoiceProfile:
        rng = np.random.default_rng(0)
        return VoiceProfile(
            name="test",
            speaker_embedding=rng.normal(size=256).astype(np.float32),
            flow_embedding=rng.normal(size=192).astype(np.float32),
            prompt_tokens=rng.integers(0, 6561, size=150).astype(np.int64),
            prompt_mel=rng.normal(size=(80, 100)).astype(np.float32),
            cond_prompt_tokens=rng.integers(0, 6561, size=40).astype(np.int64),
        )

    def test_free_run_tokens_identical_to_dynamic(self) -> None:
        from dataclasses import replace

        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        # A long cap and a high floor force a real decode run: the stop token
        # is unmaskable until the floor, so the loop must sample 20+ tokens
        # rather than stopping on the first step.
        base = AlgorithmConfig()
        cfg = replace(
            base,
            sampling=replace(base.sampling, max_new_tokens=32, min_tokens_floor=20),
        )
        g_dyn, cfg = self._generators(cuda_graphs=False, config=cfg)
        g_static, _ = self._generators(cuda_graphs=True, config=cfg)
        g_static.load_state_dict(g_dyn.state_dict())

        voice = self._voice()
        text = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)

        with torch.inference_mode():
            dyn = g_dyn.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=7))
            stat = g_static.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=7))
        assert len(dyn) >= 20, f"decode stopped too early: {dyn}"
        assert dyn == stat, f"static cache diverged at a sampled token: {dyn} vs {stat}"

    def test_static_path_is_deterministic(self) -> None:
        from dataclasses import replace

        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        base = AlgorithmConfig()
        cfg = replace(base, sampling=replace(base.sampling, min_tokens_floor=20))
        g, cfg = self._generators(cuda_graphs=True, config=cfg)
        voice = self._voice()
        text = np.array([10, 20, 30, 40], dtype=np.int64)

        with torch.inference_mode():
            a = g.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=7))
            b = g.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=7))
        assert a == b
        assert len(a) >= 20, f"decode stopped too early: {a}"

    def test_logits_within_equivalent_band(self) -> None:
        """The static path must stay inside the contract's ``equivalent`` band:
        logit drift ~1e-6 (padded reduction), never a re-baseline surprise."""
        torch.manual_seed(0)
        g_dyn, _ = self._generators(cuda_graphs=False)
        g_static, _ = self._generators(cuda_graphs=True)
        g_static.load_state_dict(g_dyn.state_dict())
        voice = self._voice()
        text = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
        forced = [100, 200, 300, 400, 500, 600, 700, 800]

        with torch.inference_mode():
            ref = g_dyn.teacher_forced_logits(text, voice, forced)
            # Static path: step the cache by hand over the same forced tokens.
            prefill_len = g_static._prefill_embeds(text, voice).shape[1]
            k_bufs = torch.zeros(2, 1, 1, prefill_len + 10, 32)
            v_bufs = torch.zeros_like(k_bufs)
            embeds = g_static._prefill_embeds(text, voice)
            hidden, cache = g_static.tfmr(
                embeds, torch.arange(prefill_len), None, attention="eager"
            )
            for i, (k, v) in enumerate(cache):
                k_bufs[i, 0, :, :prefill_len, :].copy_(k[0])
                v_bufs[i, 0, :, :prefill_len, :].copy_(v[0])
            grid = (
                torch.zeros(1 * 32, dtype=torch.long),
                torch.arange(1, dtype=torch.long).repeat_interleave(32).contiguous(),
                torch.arange(32, dtype=torch.long).repeat(1).contiguous(),
            )
            token_buf = torch.zeros(1, 1, dtype=torch.long)
            emb_pos_buf = torch.zeros(1, dtype=torch.long)
            rope_pos_buf = torch.zeros(1, dtype=torch.long)
            logits_buf = torch.zeros(1, g_static.SPEECH_VOCAB, dtype=torch.float32)
            got = []
            for step_idx, tok in enumerate(forced):
                token_buf.fill_(tok)
                emb_pos_buf.fill_(step_idx + 1)
                rope_pos_buf.fill_(prefill_len + step_idx)
                emb = g_static.speech_emb(token_buf) + g_static.speech_pos_emb.at_buf(
                    emb_pos_buf
                )
                hidden = g_static.tfmr.forward_static(
                    emb.to(g_static._dtype), rope_pos_buf, k_bufs, v_bufs, grid
                )
                logits_buf.copy_(g_static.speech_head(hidden[:, -1]).float())
                got.append(logits_buf.numpy()[0].copy())
            got = np.stack(got)
        drift = float(np.abs(got - ref[1:]).max())
        assert drift < 1e-4, f"static cache drifted {drift} — outside equivalent band"
        assert np.argmax(got, 1).tolist() == np.argmax(ref[1:], 1).tolist(), (
            "static cache changed a top-1 decision"
        )


class TestGlobalTorchFlags:
    """`pin_determinism` mutates process-global torch state, so two engines in
    one process cannot both be running what they report."""

    def test_a_contradictory_second_pin_is_reported(self, monkeypatch) -> None:
        """The first engine keeps describing flags it no longer runs under.

        loudkit cannot make two contradictory engines both correct in one
        process — but a recorded configuration that is not the running one is
        precisely the defect this library exists to end, so it must at least be
        impossible to reach silently.
        """
        import warnings

        from loudkit.backends import torch_backend
        from loudkit.config import ExecutionConfig

        monkeypatch.setattr(torch_backend, "_PINNED", None)

        # Repinning the *same* flags is not a conflict and must stay quiet.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch_backend.pin_determinism(ExecutionConfig(deterministic=True))
            torch_backend.pin_determinism(ExecutionConfig(deterministic=True))

        with pytest.warns(RuntimeWarning, match="re-pins those process-global flags"):
            torch_backend.pin_determinism(ExecutionConfig(deterministic=False))

    def test_determinism_is_pinned_symmetrically(self, monkeypatch) -> None:
        """Turning determinism off must actually turn cudnn's flag off.

        It used to be set only in the `deterministic` branch, so a later
        non-deterministic engine inherited `cudnn.deterministic=True` from an
        earlier one and ran ~5% slower than its own config claims.
        """
        import torch

        from loudkit.backends import torch_backend
        from loudkit.config import ExecutionConfig

        monkeypatch.setattr(torch_backend, "_PINNED", None)
        torch_backend.pin_determinism(ExecutionConfig(deterministic=True))
        assert torch.backends.cudnn.deterministic is True

        monkeypatch.setattr(torch_backend, "_PINNED", None)
        torch_backend.pin_determinism(ExecutionConfig(deterministic=False))
        assert torch.backends.cudnn.deterministic is False
