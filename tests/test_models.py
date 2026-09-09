"""Tests for the model layer that need no weights.

Everything here checks algorithm-layer behaviour — window framing, the Euler
grid, the EOS floor, noise addressing, the precision refusals — which is
exactly the layer whose silent divergence has cost this project real time.
If these pass, a parity failure (test_parity.py) is arithmetic, not recipe.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from loudkit.config import AlgorithmConfig, SamplingConfig, WindowConfig
from loudkit.models.generator import eos_floor
from loudkit.models.noise import gaussian_field, symmetric_uniforms
from loudkit.voice import VoiceProfile

from .assets import asset as _asset
from .assets import requires

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

    def test_the_recipe_is_named_and_still_a_tuple(self) -> None:
        """`frame_windows` crosses into three backends, which unpack it
        positionally, and `decode_batch` used to read it by index. Naming the
        fields may not cost the positional unpack: both spellings have to
        address the same four values in the same order."""
        from loudkit.models.windowing import FramedWindow, frame_windows

        framed = frame_windows(self._cfg(PRODUCTION_WINDOW), [1, 2, 3], _voice())
        assert isinstance(framed, FramedWindow)
        assert isinstance(framed, tuple)

        row, cond, prompt_frames, n = framed
        assert framed[0] is row is framed.row
        assert framed[1] is cond is framed.cond
        assert framed[2] == prompt_frames == framed.prompt_frames
        assert framed[3] == n == framed.n
        assert len(framed) == 4


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

    def test_the_ragged_vocoder_agrees_with_the_padded_one(self) -> None:
        """The 1e-6 claim, measured rather than repeated.

        `ExecutionConfig.vocoder_ragged` is on by default and it changes shipped
        bytes: instead of padding every mel out to twice the window, it pads by
        `VOCODER_RIGHT_CONTEXT` and rounds to a length bucket. The docstring
        says the difference is 1e-6 — fp32 noise from cuDNN picking different
        algorithms for different widths, 120 dB down — and nothing anywhere
        held it to that.

        Two properties, and the second is the one a rollback needs: the ragged
        output agrees with the padded one, and turning the flag off reproduces
        the padded bytes exactly. This runs on CPU with untrained weights, so
        it measures the *padding arithmetic* rather than cuDNN's choices, and
        the agreement it pins is exact rather than the shipped 1e-6.

        The width asserts are what a right-context or bucketing change trips,
        and the 1e-5 comparison is not. Measured on this CPU, the deviation is
        0.0 at right context 0, 1, 4, 16 and 32 alike: the conv stack is
        bit-exact over the trimmed region whatever the pad width, so no
        constant you can move here makes that number rise. The comparison
        catches padding arithmetic that misaligns or truncates. The widths
        catch everything else.

        `PRODUCTION_WINDOW`, not `AlgorithmConfig()`, and the padded lengths are
        asserted apart before the outputs are compared. A window with no
        `static_length` pads to the mel either way, so both objects would take
        one path, the ragged branch would never run and every assertion below
        would hold for a vocoder that had no ragged mode at all.
        """
        import torch

        from loudkit.models.vocoder import VOCODER_RIGHT_CONTEXT, TorchVocoder

        torch.manual_seed(0)
        cfg = AlgorithmConfig(window=PRODUCTION_WINDOW)
        padded = TorchVocoder(cfg, ragged=False).eval()
        ragged = TorchVocoder(cfg, ragged=True).eval()
        ragged.load_state_dict(padded.state_dict())
        assert VOCODER_RIGHT_CONTEXT > 0, "no right context is the 1e-1 case, not the 1e-6 one"

        rng = np.random.default_rng(0)
        # A short chunk, which is the whole point: a two-second mel against a
        # window sized for ten is where the padding is nearly all of the work.
        # The last one is the boundary where the bucket reaches the window and
        # the two paths legitimately meet.
        full = 2 * cfg.window.max_speech_tokens
        for n_frames, want_ragged in ((40, 128), (137, 192), (full, full)):
            mel = rng.normal(size=(80, n_frames)).astype(np.float32)
            # The frame counts the two objects actually run, so this test
            # cannot pass by both of them taking the same path.
            assert padded._pad_to_window(mel)[1] == full, n_frames
            assert ragged._pad_to_window(mel)[1] == want_ragged, n_frames

            a = padded.synthesize(mel, _voice(), seed=3)
            b = ragged.synthesize(mel, _voice(), seed=3)
            assert a.shape == b.shape, n_frames
            worst = float(np.abs(a - b).max())
            assert worst < 1e-5, f"{n_frames} frames: ragged differs by {worst:.2e}"

        # And off is off: the same object with the flag cleared is byte-for-byte
        # the pre-0.1.1 renderer, which is what "turn it off when you need byte
        # agreement with another build" has to mean.
        mel = rng.normal(size=(80, 61)).astype(np.float32)
        again = TorchVocoder(cfg, ragged=False).eval()
        again.load_state_dict(padded.state_dict())
        np.testing.assert_array_equal(
            padded.synthesize(mel, _voice(), seed=5),
            again.synthesize(mel, _voice(), seed=5),
        )

    def test_a_mel_past_the_window_is_cut_to_the_window(self) -> None:
        """The frames past the static window are dropped, in every renderer.

        The graph takes a fixed number of frames, so a longer mel loses its
        tail and the audio is shorter than the frame count would suggest. Go,
        Rust, JS and Swift all cut it the same way, and the ONNX vocoder here
        does too, so the torch reference has to agree rather than refuse: a
        refusal in one renderer out of five is the divergence, not the fix.

        `frame_windows` refuses an over-long *token* run one layer up, which is
        why this is reachable only by calling the stage directly.
        """
        import torch

        from loudkit.models.vocoder import TorchVocoder

        torch.manual_seed(0)
        cfg = AlgorithmConfig(window=PRODUCTION_WINDOW)
        voc = TorchVocoder(cfg, ragged=False).eval()
        full = 2 * cfg.window.max_speech_tokens
        rng = np.random.default_rng(1)

        long_mel = rng.normal(size=(80, full + 97)).astype(np.float32)
        wav = voc.synthesize(long_mel, _voice(), seed=7)
        # The count the caller can rely on: what was rendered, not what was
        # handed in. Reading it off the input mel is how the truncation goes
        # unnoticed.
        assert wav.shape == (full * 480,)
        assert voc._pad_to_window(long_mel)[1] == full

        # And the tail really is dropped rather than folded in somewhere: the
        # same mel cut by hand renders the same samples.
        np.testing.assert_array_equal(wav, voc.synthesize(long_mel[:, :full], _voice(), seed=7))

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
                frontend=object(),
                token_generator=_Ok(),
                mel_decoder=_Ok(),
                vocoder=_NoConfigVocoder(),
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
    def _frontend(self):
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

    def test_the_decoder_refuses_an_unknown_kernel_under_dash_oh(self) -> None:
        """`LlamaDecoder.forward` guarded its attention mode with an `assert`,
        which `python -O` strips, and the line after it would then look up a
        kernel the stack does not have. The module argues for raising two
        hundred lines above, over a value from the same source."""
        import torch

        from loudkit.models.generator import LlamaDecoder

        decoder = LlamaDecoder(
            {
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 128,
                "head_dim": 32,
            }
        )
        embeds = torch.zeros(1, 3, 64)
        positions = torch.arange(3)[None]
        with pytest.raises(ValueError, match="attention must be one of"):
            decoder.forward(embeds, positions, None, attention="flash")

    def test_the_device_sampler_probe_asks_what_getattr_asked(self) -> None:
        """The graph decode used to find the device form with
        `getattr(sampler, "on_device", None)`. A runtime-checkable Protocol
        replaced it, and the two have to answer alike: the shipped sampler
        offers the method, an injected sampler without it keeps the host loop.
        """
        from loudkit.config import SamplingConfig
        from loudkit.models.generator import _OnDevice
        from loudkit.sampler import LRSamplerV1

        shipped = LRSamplerV1(SamplingConfig(), seed=1)
        assert isinstance(shipped, _OnDevice)
        assert getattr(shipped, "on_device", None) is not None

        class HostOnly:
            def sample(self, logits: object, **kwargs: object) -> int:
                return 0

        injected = HostOnly()
        assert not isinstance(injected, _OnDevice)
        assert getattr(injected, "on_device", None) is None

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

    def test_a_row_view_is_the_lookup_it_replaced(self) -> None:
        """The decode reads embedding rows as views to skip a host-to-device
        copy per step. A view and an index lookup must be the same numbers."""
        from dataclasses import replace

        from loudkit.models.generator import TorchTokenGenerator

        cfg = replace(AlgorithmConfig(), decode_mode="fusion_mtp2")
        torch.manual_seed(0)
        g = TorchTokenGenerator(
            cfg,
            {
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "intermediate_size": 128,
                "head_dim": 32,
            },
            attention="eager",
        )
        hidden = torch.randn(1, 64)
        with torch.inference_mode():
            for first, second, position in ((0, 1, 0), (17, 4096, 3), (8193, 6561, 2049)):
                index = torch.tensor([[first, second]])
                pair = g.speech_emb(index)
                assert torch.equal(g._speech_row(first), pair[0, :1])
                assert torch.equal(g._speech_row(second), pair[0, 1:])
                assert torch.equal(
                    g.speech_pos_emb.at(position),
                    g.speech_pos_emb.emb(torch.tensor([[position]])),
                )
                e_a = pair[:, :1]
                both = torch.cat((e_a, pair[:, 1:]), dim=-1)
                assert torch.equal(
                    g._pair_slot_embed(first, second, position),
                    0.5 * (e_a + pair[:, 1:])
                    + g.fuse(both)
                    + g.speech_pos_emb.emb(torch.tensor([[position]])),
                )
                assert np.array_equal(
                    g._pair_second_logits(hidden, first),
                    g.head2(torch.cat((hidden, pair[0, :1]), dim=-1)).float().numpy(),
                )
            # A row off either end of a table is still an error, not an empty
            # tensor that broadcasts away into silence and not the last row.
            for bad in (g.SPEECH_VOCAB, -1):
                with pytest.raises(IndexError):
                    g._speech_row(bad)
            for bad in (g.MAX_SPEECH_POSITIONS, -1):
                with pytest.raises(IndexError):
                    g.speech_pos_emb.at(bad)

    @pytest.mark.parametrize("decode_mode", ["single", "fusion_mtp2"])
    @pytest.mark.parametrize("cuda_graphs", [False, True])
    def test_a_decode_does_not_write_into_the_weights(
        self, decode_mode: str, cuda_graphs: bool
    ) -> None:
        """The decode hands out views of the embedding tables, and a view aliases
        the parameter: an in-place op on one would edit the model itself, silently,
        and only on whichever loop took that path."""
        from dataclasses import replace

        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        base = AlgorithmConfig()
        cfg = replace(
            base,
            decode_mode=decode_mode,
            sampling=replace(base.sampling, max_new_tokens=32, min_tokens_floor=20),
        )
        harness = TestStaticCacheDecode()
        g, cfg = harness._generators(cuda_graphs=cuda_graphs, config=cfg)
        voice = harness._voice()

        before = {name: p.clone() for name, p in g.named_parameters()}
        with torch.inference_mode():
            tokens = g.generate(
                np.array([10, 20, 30, 40], dtype=np.int64),
                voice,
                sampler=LRSamplerV1(cfg.sampling, seed=7),
            )
        assert len(tokens) >= 20, f"decode stopped too early to prove anything: {tokens}"
        for name, p in g.named_parameters():
            assert torch.equal(p, before[name]), f"the decode wrote into {name}"


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

    @pytest.mark.parametrize("decode_mode", ["single", "fusion_mtp2"])
    @pytest.mark.parametrize("slack", [2, 33])
    def test_the_step_ignores_everything_past_its_position(
        self, decode_mode: str, slack: int
    ) -> None:
        """The stack builds one padding mask per step. It is the only thing
        keeping the KV buffer's unwritten tail out of the answer, so poisoning
        that tail must not move a single logit."""
        from dataclasses import replace

        torch.manual_seed(0)
        g, _ = self._generators(
            cuda_graphs=True, config=replace(AlgorithmConfig(), decode_mode=decode_mode)
        )
        voice = self._voice()
        text = np.array([10, 20, 30, 40], dtype=np.int64)

        with torch.inference_mode():
            embeds = g._prefill_embeds(text, voice)
            prefill_len = embeds.shape[1]
            hidden, cache = g.tfmr(embeds, torch.arange(prefill_len), None, attention="eager")
            k_bufs, v_bufs, grid = g._static_kv(prefill_len + slack)
            for i, (k, v) in enumerate(cache):
                k_bufs[i, 0, :, :prefill_len, :].copy_(k[0])
                v_bufs[i, 0, :, :prefill_len, :].copy_(v[0])
            pos = torch.tensor([prefill_len])
            emb = g._speech_token_embed(123, 1).to(g._dtype)
            clean = g.tfmr.forward_static(emb, pos, k_bufs, v_bufs, grid).clone()
            k_bufs[:, :, :, prefill_len + 1 :, :].normal_(0.0, 10.0)
            v_bufs[:, :, :, prefill_len + 1 :, :].normal_(0.0, 10.0)
            poisoned = g.tfmr.forward_static(emb, pos, k_bufs, v_bufs, grid)
        assert torch.equal(clean, poisoned), "the buffer's tail reached the answer"

    @pytest.mark.parametrize("decode_mode", ["single", "fusion_mtp2"])
    def test_every_layer_gets_the_mask_its_own_step_would_have_built(
        self, decode_mode: str
    ) -> None:
        """One mask, built by the stack, handed to every layer.

        It has to be what a layer building its own would have got: the padding
        of *this* step over *this* buffer, in the dtype the layer computes in.
        And it has to be the mask the layer actually uses, or hoisting it is
        decoration and the tail test above is passing for another reason.
        """
        from dataclasses import replace

        from loudkit.models import generator as gen_mod

        torch.manual_seed(0)
        cfg = replace(AlgorithmConfig(), decode_mode=decode_mode)
        g, _ = self._generators(cuda_graphs=True, config=cfg)
        real = gen_mod._Attention.forward_static
        seen: list[torch.Tensor] = []

        def spy(self, x, cos, sin, k_buf, v_buf, grid, pos, mask):
            seen.append(mask)
            return real(self, x, cos, sin, k_buf, v_buf, grid, pos, mask)

        layer = g.tfmr.layers[0]
        for buf_len in (8, 64, 137):
            for step in (0, 1, buf_len // 2, buf_len - 1):
                k_bufs, v_bufs, grid = g._static_kv(buf_len)
                k_bufs.normal_()
                v_bufs.normal_()
                emb = torch.randn(1, 1, 64, dtype=g._dtype)
                pos = torch.tensor([step])
                seen.clear()
                gen_mod._Attention.forward_static = spy
                try:
                    with torch.inference_mode():
                        g.tfmr.forward_static(emb, pos, k_bufs.clone(), v_bufs.clone(), grid)
                finally:
                    gen_mod._Attention.forward_static = real

                assert len(seen) == g.tfmr.n_layers
                want = torch.where(
                    torch.arange(buf_len) > pos,
                    torch.full((), float("-inf"), dtype=g._dtype),
                    torch.zeros((), dtype=g._dtype),
                )
                for mask in seen:
                    assert mask.dtype == emb.dtype, "the mask is not the layer's dtype"
                    assert torch.equal(mask, want), f"wrong mask at len={buf_len} step={step}"
                    assert mask is seen[0], "the layers were handed different masks"

                # The layer must use what it is handed. Give it a mask that
                # hides nothing and the unwritten tail has to change the answer,
                # or hoisting the mask was decoration.
                with torch.inference_mode():
                    right = layer.forward_static(
                        emb,
                        *g.tfmr._rope(pos, g._dtype),
                        k_bufs.clone()[0],
                        v_bufs.clone()[0],
                        grid,
                        pos,
                        want,
                    ).clone()
                    none = layer.forward_static(
                        emb,
                        *g.tfmr._rope(pos, g._dtype),
                        k_bufs.clone()[0],
                        v_bufs.clone()[0],
                        grid,
                        pos,
                        torch.zeros_like(want),
                    )
                if step < buf_len - 1:
                    assert not torch.equal(right, none), "the layer ignores the mask"
                else:
                    assert torch.equal(right, none), "the last slot masks nothing"


class TestFusionDecode:
    """``decode_mode="fusion_mtp2"``: two tokens per forward, one KV slot.

    The gate that matters is agreement between the two implementations of the
    same arithmetic — the decode loop, which samples a pair at a time, and
    ``teacher_forced_logits``, which reads the whole sequence in one causal
    forward. They share no code, so if the pairing, the position clock or the
    second head's input were wrong in one of them, they would disagree.
    """

    def _config(self, **sampling: object) -> AlgorithmConfig:
        from dataclasses import replace

        base = AlgorithmConfig()
        return replace(
            base,
            decode_mode="fusion_mtp2",
            sampling=replace(base.sampling, **sampling),
        )

    def _generator(self, cfg: AlgorithmConfig, *, cuda_graphs: bool = False):
        harness = TestStaticCacheDecode()
        g, _ = harness._generators(cuda_graphs=cuda_graphs, config=cfg)
        return g

    def _voice(self) -> VoiceProfile:
        return TestStaticCacheDecode()._voice()

    def test_the_fusion_weights_exist_only_in_fusion_mode(self) -> None:
        single = self._generator(AlgorithmConfig())
        fused = self._generator(self._config())
        assert not hasattr(single, "head2")
        assert hasattr(fused, "head2")
        assert hasattr(fused, "fuse")
        # A mode/weights disagreement has to be an error, not a silent
        # half-loaded model: this is the whole reason the modules are
        # conditional rather than always built.
        with pytest.raises(RuntimeError):
            single.load_state_dict(fused.state_dict())
        with pytest.raises(RuntimeError):
            fused.load_state_dict(single.state_dict())

    def test_decode_loop_agrees_with_one_causal_forward(self) -> None:
        """Every token the loop sampled is the argmax of the row teacher
        forcing puts in front of it — the two paths reading one model."""
        torch.manual_seed(0)
        cfg = self._config(max_new_tokens=24, min_tokens_floor=16)
        g = self._generator(cfg)
        voice = self._voice()
        text = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)

        with torch.inference_mode():
            tokens = list(g.generate(text, voice, sampler=_Greedy()))
            rows = g.teacher_forced_logits(text, voice, tokens)
        assert len(tokens) >= 16, f"decode stopped too early: {tokens}"
        assert rows.shape[0] == len(tokens) + 1
        assert np.argmax(rows[:-1], 1).tolist() == tokens

    def test_static_path_matches_the_eager_one(self) -> None:
        torch.manual_seed(0)
        cfg = self._config(max_new_tokens=24, min_tokens_floor=16)
        eager = self._generator(cfg)
        static = self._generator(cfg, cuda_graphs=True)
        static.load_state_dict(eager.state_dict())
        voice = self._voice()
        text = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)

        from loudkit.sampler import LRSamplerV1

        with torch.inference_mode():
            a = eager.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=7))
            b = static.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=7))
        assert list(a) == list(b), f"static fused cache diverged: {a} vs {b}"

    def test_one_forward_per_pair(self) -> None:
        """The speed claim, asserted rather than assumed: N tokens cost N/2
        transformer calls, not N."""
        torch.manual_seed(0)
        cfg = self._config(max_new_tokens=24, min_tokens_floor=16)
        g = self._generator(cfg)
        voice = self._voice()
        text = np.array([10, 20, 30, 40], dtype=np.int64)

        calls = 0
        inner = g.tfmr.forward

        def counted(*a: object, **kw: object) -> object:
            nonlocal calls
            calls += 1
            return inner(*a, **kw)

        g.tfmr.forward = counted
        with torch.inference_mode():
            tokens = list(g.generate(text, voice, sampler=_Greedy()))
        # One prefill, then one forward per completed pair.
        assert calls == 1 + len(tokens) // 2, f"{calls} forwards for {len(tokens)} tokens"

    @pytest.mark.parametrize(
        ("floor", "cap"),
        [(16, 24), (0, 24), (8, 9), (16, 40)],
        ids=["ordinary", "stops-at-once", "odd-cap", "long"],
    )
    def test_the_on_device_loop_is_the_same_loop(self, floor: int, cap: int) -> None:
        """The third decode path, which nothing else in the suite reaches.

        `generate` picks between three loops. The eager one and the static one
        that samples on the host are both exercised above; the third — the one
        that draws both tokens inside the captured graph — is chosen only when
        `_device_sampler` finds CUDA, so on every machine this suite runs on it
        silently falls back to the second. That left the loop with the most
        moving parts untested anywhere: the opening pair drawn on the host, the
        `prepare` that has to happen *after* capture because the warm-up
        replays advanced the step and the positions, the pair written back into
        `pair_buf` for the next replay to consume, and the cap and stop checks
        applied to two integers that have already been drawn.

        `DeviceSamplerV1` runs on CPU torch — `TestDeviceSampler` uses it that
        way — so the only thing standing between this loop and this test was
        the CUDA gate. Stubbing that gate is the whole trick.

        The parameters are the boundaries: a floor that unmasks mid-pair, a
        floor of zero so the stop token can win at step one, an odd cap that
        must stop a pair half-way, and a run long enough to cross several.
        """
        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        cfg = self._config(max_new_tokens=cap, min_tokens_floor=floor)
        eager = self._generator(cfg)
        host = self._generator(cfg, cuda_graphs=True)
        device = self._generator(cfg, cuda_graphs=True)
        host.load_state_dict(eager.state_dict())
        device.load_state_dict(eager.state_dict())
        # The one gate: `_device_sampler` returns None off CUDA, and returning
        # the CPU form is exactly what it would return on a card.
        device._device_sampler = lambda sampler: sampler.on_device("cpu")

        voice = self._voice()
        text = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
        stop = cfg.stop_speech_token
        runs, peaks = [], []
        with torch.inference_mode():
            for g in (eager, host, device):
                s = LRSamplerV1(cfg.sampling, seed=7, stop_token=stop, eos_floor=floor)
                runs.append(list(g.generate(text, voice, sampler=s)))
                peaks.append(s.eos_peak)
        assert runs[0] == runs[1] == runs[2], (
            f"three loops, three answers: eager={runs[0]} host={runs[1]} device={runs[2]}"
        )
        assert len(runs[0]) <= cap, "the cap counts tokens, and a pair can carry it past"
        # The peak is audible — postprocess compares it against thresholds — so
        # agreeing on tokens is not agreeing. The device path's divisor is a
        # parallel sum where the host's is ordered, so the probability is
        # compared with a tolerance and the *step* exactly.
        assert peaks[0][0] == peaks[1][0] == peaks[2][0], f"eos_peak step: {peaks}"
        assert peaks[2][1] == pytest.approx(peaks[1][1], rel=1e-9), f"eos_peak prob: {peaks}"
        for at, _ in peaks:
            assert at < len(runs[0]), (
                f"eos_peak_at {at} is past the {len(runs[0])} tokens returned; "
                f"the postprocess guards `eos_peak_at < len(tokens)` stop holding"
            )


class TestTheStepsTheFiveDecodeLoopsShare:
    """The steps the loops call instead of each spelling them out.

    One implementation is what keeps the five from drifting apart, and it is
    also the one thing that could move all five at once. The tests above ask
    whether the loops agree with each other, and they would go on agreeing if
    a shared step were wrong. So each shared step is checked against the
    arithmetic it replaced, written out here.
    """

    def test_the_fused_slot_is_the_mean_plus_what_fuse_reads(self) -> None:
        """`_fuse_pair`, against the expression the five sites carried."""
        fusion = TestFusionDecode()
        g = fusion._generator(fusion._config())
        torch.manual_seed(3)
        hidden = g.speech_emb.embedding_dim
        e_a, e_b = torch.randn(1, 4, hidden), torch.randn(1, 4, hidden)

        with torch.inference_mode():
            got = g._fuse_pair(e_a, e_b)
            want = 0.5 * (e_a + e_b) + g.fuse(torch.cat((e_a, e_b), dim=-1))
        assert torch.equal(got, want), "the fused slot is not what the five sites built"

    def test_a_draw_masks_below_the_floor_and_marks_only_what_survives(self) -> None:
        """`_draw`: the floor closes over the stop token, the sampler is asked
        at the step `out` has reached, and a stop token is never marked seen."""
        from loudkit.models.generator import TorchTokenGenerator

        stop, floor = 3, 2
        seen = np.zeros(8, dtype=bool)
        steps: list[int] = []
        # The stop token is the argmax, so it wins whenever it is reachable.
        logits = np.array([0.0, 1.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        def sampler(logits: np.ndarray, *, step: int, seen: np.ndarray) -> int:
            steps.append(step)
            return int(np.argmax(logits))

        out: list[int] = []
        drawn = [
            TorchTokenGenerator._draw(
                logits.copy(), out, sampler, floor=floor, stop=stop, seen=seen
            )
            for _ in range(3)
        ]

        assert drawn == [1, 1, stop], (
            "below the floor the stop token has to be out of reach, and at the "
            f"floor it has to be reachable again: {drawn}"
        )
        assert out == [1, 1, stop], "every draw lands on `out`, stop token included"
        assert seen[1], "a token that did not end the sequence is marked seen"
        assert not seen[stop], "a stop token ends the sequence and is never marked seen"
        assert steps == [0, 1, 2], "the sampler is asked at the step `out` has reached"

    def test_the_bucket_rounds_up_to_a_shared_graph_length(self) -> None:
        """`_bucket`, which decides whether two chunks share a captured graph."""
        from loudkit.models.generator import _GRAPH_BUCKET, _bucket

        assert _bucket(1) == _GRAPH_BUCKET, "a short decode still takes a whole bucket"
        assert _bucket(_GRAPH_BUCKET) == _GRAPH_BUCKET, "an exact fit must not round up"
        assert _bucket(_GRAPH_BUCKET + 1) == 2 * _GRAPH_BUCKET, (
            "one slot over the bucket takes the next one whole"
        )
        assert _bucket(0) == 0, "the rounding carries no minimum of its own"

    def test_every_decode_loop_opens_on_the_one_parameter_object(self) -> None:
        """`_Decode`: the ten a loop is handed rather than chooses.

        Asked structurally, because the loops would agree on tokens either way:
        a loop handed nine of the ten and its own tenth is still a loop, and it
        decodes fluently until the day its `floor` and its `seen` come from
        different utterances. Ten names spelled out in five signatures is five
        places for that to start, and a sixth loop taking anything but the plan
        is the drift this catches.
        """
        import dataclasses
        import inspect

        from loudkit.models.generator import TorchTokenGenerator, _Decode

        shared = {f.name for f in dataclasses.fields(_Decode)}
        # Not a count: an eleventh shared field is what the plan is for. These
        # six are the ones no loop can decode without.
        assert {"cap", "floor", "stop", "seen", "sampler", "logits"} <= shared, (
            f"the plan stopped carrying what every loop reads: {sorted(shared)}"
        )

        for name in (
            "_static_decode",
            "_generate_fused",
            "_generate_static",
            "_generate_static_fused",
            "_generate_static_fused_ondevice",
        ):
            params = inspect.signature(getattr(TorchTokenGenerator, name)).parameters
            assert "plan" in params, f"{name} does not take the shared plan"
            # `from __future__ import annotations`, so the annotation is its text.
            assert params["plan"].annotation == "_Decode", (
                f"{name} takes a `plan` that is not the plan"
            )
            assert not shared & set(params), (
                f"{name} re-declares what the plan already carries: {shared & set(params)}"
            )

    def test_the_cancellation_poll_asks_once_and_only_when_asked_for(self) -> None:
        """`_Decode.cancelled`, against the expression the six sites carried.

        Two properties, and a loop depends on both: barge-in is polled once per
        step, so a poll that ran twice would double whatever the caller does to
        answer it, and a caller who passed no callback is never cancelled.
        """
        from loudkit.models.generator import _Decode

        polls: list[int] = []

        def plan(should_cancel):
            return _Decode(
                cap=8,
                floor=0,
                stop=6562,
                seen=np.zeros(8194, dtype=bool),
                prefill_len=4,
                cache=[],
                prefix_len=0,
                # Never asked for a token: `cancelled` is the whole subject here.
                sampler=lambda *_a, **_k: 0,
                logits=np.zeros(8194, dtype=np.float32),
                should_cancel=should_cancel,
            )

        def fires_on_the_second() -> bool:
            polls.append(len(polls))
            return len(polls) > 1

        cut = plan(fires_on_the_second)
        assert not cut.cancelled(), "the first poll answered no and was not believed"
        assert cut.cancelled(), "the second poll answered yes and was not believed"
        assert polls == [0, 1], f"one poll per call, and this made {len(polls)} in two"

        never = plan(None)
        assert not never.cancelled(), "a caller who passed nothing was cancelled"


class _ScriptedHead(torch.nn.Module):
    """A head whose logits are written down instead of computed.

    The pair boundary cases — the first head stopping the sequence, the second
    stopping it, a cap closing between the two — do not arise on random weights
    often enough to test by running and hoping. Scripting the heads makes each
    one a single call away.
    """

    def __init__(self, rows: list[np.ndarray], in_features: int) -> None:
        super().__init__()
        self.rows = [torch.from_numpy(r) for r in rows]
        self.in_features = in_features
        self.calls = 0
        # `TorchTokenGenerator._device` and `._dtype` read
        # `speech_head.weight`, so a stand-in needs one even though the logits
        # do not come from it.
        self.weight = torch.nn.Parameter(torch.zeros(1, in_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row = self.rows[min(self.calls, len(self.rows) - 1)]
        self.calls += 1
        return row.unsqueeze(0).to(x.device)


class TestTheDiscardedHalfOfAPairLeavesNothingBehind:
    """A fused pair draws twice per replay; the host keeps one or both.

    The device path has no choice about the second draw — a captured graph has
    one shape — so when the first token ends the sequence, or the cap closes
    between the two, a token is drawn that is not in the output. Until the
    `valid` argument existed it still marked `seen` and still moved the EOS
    peak, and the peak is audible: `loudkit.postprocess` compares it against
    thresholds and guards on `eos_peak_at < len(tokens)`, which a ghost peak
    breaks by construction.

    The host and eager loops cannot have this defect — they never draw the
    second token when the first stopped — so this is the one place where the
    three loops are *not* structurally the same, and it needs its own test.
    """

    VOCAB = 8194

    def _generator(self, first_rows, second_rows, *, floor: int, cap: int):
        from dataclasses import replace

        base = AlgorithmConfig()
        cfg = replace(
            base,
            decode_mode="fusion_mtp2",
            sampling=replace(base.sampling, max_new_tokens=cap, min_tokens_floor=floor),
        )
        g, _ = TestStaticCacheDecode()._generators(cuda_graphs=True, config=cfg)
        hidden = g.speech_head.in_features
        g.speech_head = _ScriptedHead(first_rows, hidden)
        g.head2 = _ScriptedHead(second_rows, 2 * hidden)
        g._device_sampler = lambda sampler: sampler.on_device("cpu")
        return g, cfg

    @staticmethod
    def _row(peak: int, value: float = 30.0) -> np.ndarray:
        row = np.full(TestTheDiscardedHalfOfAPairLeavesNothingBehind.VOCAB, -20.0, np.float32)
        row[peak] = value
        return row

    def test_a_stop_from_the_first_head_discards_the_second_draw(self) -> None:
        """The pair whose first token ends the sequence.

        The first head's stop wins on a margin, so its own peak is high but not
        1.0; the discarded draw's stop dominates, so with the gate removed it
        wins the running maximum and lands at an index the sequence does not
        have. A dominant stop in *both* would not test anything — the peaks tie
        and `prob > peak_prob` keeps the first.
        """
        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        stop = AlgorithmConfig().stop_speech_token
        margin = self._row(stop, 0.0)
        # A token the script never emits, so the repetition penalty leaves it
        # alone: penalised, -2.0 falls below the min_p threshold and the stop
        # is the only survivor again, which puts the peak back at 1.0.
        margin[13] = -2.0
        g, cfg = self._generator(
            [self._row(11), margin],
            [self._row(12), self._row(stop, 60.0)],
            floor=0,
            cap=32,
        )
        sampler = LRSamplerV1(cfg.sampling, seed=3, stop_token=stop, eos_floor=0)
        voice = TestStaticCacheDecode()._voice()
        with torch.inference_mode():
            tokens = list(
                g.generate(np.array([10, 20], dtype=np.int64), voice, sampler=sampler)
            )

        assert tokens[-1] == stop, f"the case needs the first head to stop it: {tokens}"
        assert len(tokens) == 3, (
            f"the case needs the stop to arrive on a pair's first: {tokens}"
        )
        at, prob = sampler.eos_peak
        assert at < len(tokens), (
            f"eos_peak_at {at} is past the {len(tokens)} tokens returned: the second "
            f"draw of the stopping pair was discarded and still moved the peak"
        )
        assert prob < 1.0, (
            "the peak is the discarded draw's dominant stop wearing a valid index"
        )

    def test_a_cap_closing_mid_pair_discards_the_second_draw(self) -> None:
        """The odd cap: the last pair contributes one token, not two.

        No stop anywhere in the script, so the run reaches the cap; the head
        that fills the discarded half emits a dominant stop, which is the only
        thing that could move the peak and must not.
        """
        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        stop = AlgorithmConfig().stop_speech_token
        plain, loud = self._row(11), self._row(stop, 60.0)
        # cap 7: the opening pair, two full replays, then a replay whose first
        # token closes the cap and whose second is drawn and thrown away. That
        # discarded draw is the fourth call to the second head.
        g, cfg = self._generator(
            [plain, plain, plain, plain],
            [self._row(12), self._row(12), self._row(12), loud],
            floor=0,
            cap=7,
        )
        sampler = LRSamplerV1(cfg.sampling, seed=3, stop_token=stop, eos_floor=0)
        voice = TestStaticCacheDecode()._voice()
        with torch.inference_mode():
            tokens = list(
                g.generate(np.array([10, 20], dtype=np.int64), voice, sampler=sampler)
            )

        assert len(tokens) == 7, f"the case needs an odd cap to be reached: {len(tokens)}"
        assert g.head2.calls == 4, (
            f"the case needs the discarded draw to happen: {g.head2.calls}"
        )
        at, _prob = sampler.eos_peak
        assert at < len(tokens), (
            f"eos_peak_at {at} is past the {len(tokens)} tokens returned: the eighth "
            f"token was drawn to fill the pair, discarded, and moved the peak"
        )

    def test_cancellation_is_honoured_before_the_opening_pair(self) -> None:
        """All three loops poll before the first draw, or one of them does not.

        The eager and host-static loops check `should_cancel` at the top of
        their loop. The device loop drew its opening pair on the host *before*
        any check, so an interrupt that had already fired still paid for two
        tokens and returned them. The engine discards a cancelled window, so
        nothing was ever heard — but the cancellation boundary is a contract,
        and barge-in is the feature it exists for.
        """
        from dataclasses import replace

        from loudkit.sampler import LRSamplerV1

        base = AlgorithmConfig()
        cfg = replace(
            base,
            decode_mode="fusion_mtp2",
            sampling=replace(base.sampling, max_new_tokens=24, min_tokens_floor=16),
        )
        text = np.array([10, 20, 30], dtype=np.int64)
        voice = TestStaticCacheDecode()._voice()

        out = {}
        for name, graphs, ondevice in (
            ("eager", False, False),
            ("host", True, False),
            ("device", True, True),
        ):
            torch.manual_seed(0)
            g, _ = TestStaticCacheDecode()._generators(cuda_graphs=graphs, config=cfg)
            if ondevice:
                g._device_sampler = lambda sampler: sampler.on_device("cpu")
            with torch.inference_mode():
                out[name] = list(
                    g.generate(
                        text,
                        voice,
                        sampler=LRSamplerV1(cfg.sampling, seed=7),
                        should_cancel=lambda: True,
                    )
                )
        assert out == {"eager": [], "host": [], "device": []}, (
            f"an interrupt that fired before the first draw was not honoured: {out}"
        )

    def test_the_sampler_suppresses_both_pieces_of_state(self) -> None:
        """The mechanism, one layer down and without a model.

        Two properties, and both matter: an invalid draw must not move the peak
        and must not mark `seen`, because `seen` feeds the repetition penalty of
        every later step.
        """
        from loudkit.config import SamplingConfig
        from loudkit.sampler import LRSamplerV1

        width, stop = 32, 3
        cfg = SamplingConfig(
            temperature=0.8, repetition_penalty=1.0, min_p=0.0, silence_token_ids=()
        )
        host = LRSamplerV1(cfg, seed=5, stop_token=stop, eos_floor=0)
        dev = host.on_device("cpu")
        dev.prepare(width, np.zeros(width, bool), floor=0, step=1, stop_token=stop)

        quiet = np.full(width, -8.0, np.float32)
        quiet[7] = 6.0
        loud = np.full(width, -8.0, np.float32)
        loud[stop] = 40.0
        row = lambda a: torch.from_numpy(a).reshape(1, -1)  # noqa: E731

        dev.advance_to(1, draws=2)
        dev.select(row(quiet))
        kept = (int(dev._peak_at), float(dev._peak_prob))
        dev.select(row(loud), valid=torch.zeros((), dtype=torch.bool))
        assert (int(dev._peak_at), float(dev._peak_prob)) == kept, (
            "a discarded draw moved the peak"
        )
        assert not bool(dev.seen[stop].item()), "a discarded draw marked `seen`"

        # And a valid one still does both, so the gate is a gate and not an off switch.
        dev.advance_to(3, draws=2)
        dev.select(row(loud), valid=torch.ones((), dtype=torch.bool))
        assert bool(dev.seen[stop].item())
        assert (int(dev._peak_at), float(dev._peak_prob)) != kept


class TestOneGeneratorDecodesOneThingAtATime:
    """Keeping graphs across calls made `generate` non-re-entrant.

    Before 0.1.1 the decode buffers were per-call. Reusing a captured graph
    between chunks moved them onto the generator — the KV cache, the position
    scalars, the pair register and the device sampler's noise block are all
    addresses the graph holds — so two concurrent calls on one engine write
    into each other's slots.

    Nothing crashes. Both callers get fluent speech and neither gets their
    text, which is the failure this project treats as worse than a crash, so
    the second caller is refused rather than served something plausible.
    """

    def _generator(self, *, cuda_graphs: bool):
        harness = TestStaticCacheDecode()
        g, cfg = harness._generators(cuda_graphs=cuda_graphs)
        return g, cfg, harness._voice()

    def test_a_second_static_decode_is_refused_rather_than_served(self) -> None:
        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        g, cfg, voice = self._generator(cuda_graphs=True)
        text = np.array([10, 20, 30], dtype=np.int64)

        # Re-entered from inside the decode, which is the shape the real defect
        # has: a second request arriving while the first is between steps.
        reentered: list[object] = []

        class Reentrant:
            def __call__(self, logits, *, step, seen):
                if step == 2 and not reentered:
                    try:
                        g.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=1))
                        reentered.append("served")
                    except RuntimeError as exc:
                        reentered.append(str(exc))
                return int(np.argmax(logits))

        with torch.inference_mode():
            g.generate(text, voice, sampler=Reentrant())

        assert reentered, "the case never re-entered; the decode stopped too early"
        assert reentered[0] != "served", (
            "a second concurrent decode was served: both callers now share one "
            "KV cache and neither gets their own text"
        )
        assert "already decoding" in reentered[0]

    def test_the_guard_is_released_and_the_eager_path_is_untouched(self) -> None:
        """Two properties in one: the lock is not leaked across calls, and the
        eager path allocates per call so it is genuinely re-entrant."""
        from loudkit.sampler import LRSamplerV1

        torch.manual_seed(0)
        g, cfg, voice = self._generator(cuda_graphs=True)
        text = np.array([10, 20, 30], dtype=np.int64)
        with torch.inference_mode():
            for _ in range(3):
                assert g.generate(text, voice, sampler=LRSamplerV1(cfg.sampling, seed=2))

        torch.manual_seed(0)
        eager, cfg, voice = self._generator(cuda_graphs=False)
        nested: list[list[int]] = []

        class Nested:
            def __call__(self, logits, *, step, seen):
                if step == 2 and not nested:
                    nested.append(
                        list(
                            eager.generate(
                                text, voice, sampler=LRSamplerV1(cfg.sampling, seed=1)
                            )
                        )
                    )
                return int(np.argmax(logits))

        with torch.inference_mode():
            eager.generate(text, voice, sampler=Nested())
        assert nested, "the case never re-entered; the decode stopped too early"
        assert nested[0], "the eager path must stay re-entrant"


class _Greedy:
    """The sampler reduced to argmax: no RNG, so the decode loop and the
    teacher-forced forward can be compared token for token."""

    def __call__(
        self,
        logits: np.ndarray,
        *,
        step: int,
        seen: np.ndarray,
    ) -> int:
        return int(np.argmax(logits))


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

    def test_a_second_thread_count_is_reported_too(self, monkeypatch) -> None:
        """`num_threads` is process-global like the other two, and was not

        tracked. `torch.set_num_threads` changes the whole interpreter, so a
        second engine asking for a different count re-pins the first one's
        threads while the first keeps printing its own. The omission was
        invisible because thread count moves wall time rather than audio: a
        benchmark row simply measured a count other than the one it recorded.
        """
        import warnings

        from loudkit.backends import torch_backend
        from loudkit.config import ExecutionConfig

        monkeypatch.setattr(torch_backend, "_PINNED", None)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            torch_backend.pin_determinism(ExecutionConfig(num_threads=2))
            torch_backend.pin_determinism(ExecutionConfig(num_threads=2))

        with pytest.warns(RuntimeWarning, match="num_threads"):
            torch_backend.pin_determinism(ExecutionConfig(num_threads=4))

    def test_a_named_thread_count_is_in_the_line_a_benchmark_records(self) -> None:
        """A number is a decision the run made; `None` is a machine property."""
        from loudkit.config import ExecutionConfig

        assert "threads=" not in ExecutionConfig().describe()
        assert "threads=3" in ExecutionConfig(num_threads=3).describe()

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


class TestTheFusionLoopAgainstRealWeights:
    """One golden, from the packed turbo checkpoint.

    Everything else holding this loop runs on a two-layer model with random
    weights. That compares the three decode paths against each other and
    against one causal forward, which is worth having — but a regression in the
    *pairing* against real weights is caught by it only if it also changes the
    toy's answer, and `docs/design/two-token-decode.md` says so rather than
    leaving it to be discovered.

    This is the missing half: a fixed text, voice and seed through the shipping
    entry point, against the checkpoint that actually carries `head2` and
    `fuse`. It pins the token stream, which is the layer a mis-paired slot
    changes first, alongside the shared conformance fixture.

    Skipped where the checkpoint is absent, unless `LOUDKIT_REQUIRE_ASSETS`
    makes missing release assets a failure.
    """

    TEXT = "The pair boundary is where a two token decode goes wrong."
    SEED = 1234
    FINGERPRINT = json.loads(
        (
            Path(__file__).resolve().parent / "data/conformance/vectors_fusion_mtp2.json"
        ).read_text(encoding="utf-8")
    )["algorithm"]["fingerprint"]
    TOKENS = 93
    DIGEST = "9aa9ecc2fbecba3c"
    VOICE = Path(__file__).resolve().parent / "data/reference/testvoice.voice.safetensors"

    @staticmethod
    def _release_directory(tmp_path: Path) -> Path:
        """The shape a user gets, `./loudr-1-turbo/`, assembled from symlinks:
        the point is which file a resolver returns, and copying 1.2 GB to find
        that out is not worth a second of anyone's disk. The turbo manifest
        records `tokenizer_sha256`, so `Checkpoint.verified_sibling` requires
        the tokenizer beside it and checks it."""
        from loudkit.hub import TURBO_CHECKPOINT_NAME

        bundle = tmp_path / "loudr-1-turbo"
        bundle.mkdir()
        (bundle / TURBO_CHECKPOINT_NAME).symlink_to(_asset("turbo_checkpoint"))
        (bundle / "tokenizer.json").symlink_to(_asset("tokenizer"))
        return bundle

    @requires("turbo_checkpoint")
    def test_the_shipping_path_reproduces_its_golden(self) -> None:
        import hashlib

        import loudkit

        engine = loudkit.load(str(_asset("turbo_checkpoint")), device="cpu")
        assert engine.algorithm.decode == "fusion_mtp2"
        assert engine.algorithm.euler_steps == 1
        assert engine.algorithm.fingerprint() == self.FINGERPRINT, (
            "the algorithm moved; re-take the golden below and say why in the "
            "commit, because this is an audible change"
        )

        voice = VoiceProfile.load(
            Path(__file__).resolve().parent / "data/reference/testvoice.voice.safetensors"
        )
        result = engine.synthesize(self.TEXT, voice, seed=self.SEED)
        tokens = np.asarray(result.tokens, dtype=np.int64)
        digest = hashlib.sha256(tokens.tobytes()).hexdigest()[:16]
        assert (len(tokens), digest) == (self.TOKENS, self.DIGEST), (
            f"the fusion decode produced {len(tokens)} tokens digesting to "
            f"{digest}, against {self.TOKENS}/{self.DIGEST}. A mis-paired slot, "
            f"a position clock ticking per token, or a carry off a pair "
            f"boundary all land here first."
        )

    @requires("turbo_checkpoint", "tokenizer")
    def test_a_release_directory_loads_by_its_directory(self, tmp_path) -> None:
        """The shape a user actually gets: `lk.load("./loudr-1-turbo")`.

        The golden above names the checkpoint file, which is the developer's
        path. A downloaded or unpacked release is a directory, and resolving
        one goes through `hub._only_checkpoint_in` and the canonical turbo
        name rather than through "the file you pointed at". Same text, voice
        and seed, so a difference here is the resolver picking a different
        file, not the decode changing.

        Assembled with symlinks: the point is which file the resolver returns,
        and copying 1.2 GB to find that out is not worth a second of anyone's
        disk.
        """
        import hashlib

        import loudkit

        engine = loudkit.load(str(self._release_directory(tmp_path)), device="cpu")
        assert engine.algorithm.decode == "fusion_mtp2"
        voice = VoiceProfile.load(self.VOICE)
        tokens = np.asarray(
            engine.synthesize(self.TEXT, voice, seed=self.SEED).tokens, dtype=np.int64
        )
        digest = hashlib.sha256(tokens.tobytes()).hexdigest()[:16]
        assert (len(tokens), digest) == (self.TOKENS, self.DIGEST)

    @requires("turbo_checkpoint", "tokenizer")
    def test_the_cli_speaks_from_a_release_directory(self, tmp_path, capsys) -> None:
        """`loudkit speak --checkpoint ./loudr-1-turbo`, then `loudkit verify`.

        The golden above is the library's; this is the command a user types.
        The WAV's own manifest says which algorithm and seed rendered it, so
        the round trip through `verify` is what pins that the command spoke
        with the turbo loop and the seed it was given.
        """
        from loudkit.cli import main

        out = tmp_path / "hello.wav"
        argv = [
            "speak",
            "--checkpoint",
            str(self._release_directory(tmp_path)),
            "--voice",
            str(self.VOICE),
            "--device",
            "cpu",
            "--seed",
            str(self.SEED),
            "-o",
            str(out),
            "--verbose",
            self.TEXT,
        ]
        assert main(argv) == 0
        err = capsys.readouterr().err
        assert f"algo[{self.FINGERPRINT}]" in err, err
        assert "decode=fusion_mtp2" in err, err
        assert str(out) in err

        assert main(["verify", str(out)]) == 0
        report = capsys.readouterr().out
        assert f"algorithm_fingerprint: {self.FINGERPRINT}" in report, report
        assert f"seed: {self.SEED}" in report, report
        assert "provenance verified" in report, report

    @requires("turbo_checkpoint", "tokenizer", "voice_encoder")
    def test_enroll_on_a_release_directory(self, tmp_path) -> None:
        """`lk.enroll(clip, "./loudr-1-turbo")`: the profile is the reference's.

        The release carries the synthesis half and its matching enrollment half.
        The resulting profile is held to the shared enrollment fixture.
        """
        from .assets import needs_module

        needs_module("torchaudio")
        import loudkit

        bundle = self._release_directory(tmp_path)
        (bundle / "ve.safetensors").symlink_to(_asset("voice_encoder"))
        (bundle / "loudr-1-enrollment.safetensors").symlink_to(
            _asset("turbo_checkpoint").parent / "loudr-1-enrollment.safetensors"
        )
        enrollment = Path(__file__).resolve().parent / "data/enrollment"
        clip = np.fromfile(enrollment / "ref_audio.f32", dtype="<f4")

        mine = loudkit.enroll(clip, str(bundle), name="en_reader1")
        want = VoiceProfile.load(enrollment / "profile.safetensors")
        np.testing.assert_array_equal(mine.prompt_tokens, want.prompt_tokens)
        np.testing.assert_array_equal(mine.cond_prompt_tokens, want.cond_prompt_tokens)

        def cos(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        assert cos(mine.flow_embedding, want.flow_embedding) > 0.9999
        assert cos(mine.speaker_embedding, want.speaker_embedding) > 0.9999

    @requires("turbo_checkpoint")
    def test_a_graph_backend_requires_its_exported_assets(self, tmp_path) -> None:
        """Against the real manifest, not a fabricated one.

        The unit tests in `test_checkpoint.py` write their own `decode` block,
        so they prove the door works and not that the shipped checkpoint walks
        into it. This reads the block `tools/pack_turbo.py` actually wrote.
        """
        from loudkit.hub import TURBO_CHECKPOINT_NAME, verify_release_inventory

        bundle = tmp_path / "loudr-1-turbo"
        bundle.mkdir()
        (bundle / TURBO_CHECKPOINT_NAME).symlink_to(_asset("turbo_checkpoint"))
        for backend in ("onnx", "coreml"):
            with pytest.raises(FileNotFoundError, match="missing:"):
                verify_release_inventory(bundle, backend)

    @requires("turbo_checkpoint")
    def test_it_is_deterministic(self) -> None:
        """I-2 on the path that keeps buffers per generator."""
        import loudkit

        engine = loudkit.load(str(_asset("turbo_checkpoint")), device="cpu")
        voice = VoiceProfile.load(
            Path(__file__).resolve().parent / "data/reference/testvoice.voice.safetensors"
        )
        first = list(engine.synthesize(self.TEXT, voice, seed=self.SEED).tokens)
        second = list(engine.synthesize(self.TEXT, voice, seed=self.SEED).tokens)
        assert first == second


class TestTheBatchedRendererIsTheSameRenderer:
    """`decode_batch` and `synthesize_batch` are prototypes beside the contract:
    nothing in the engine calls them, and `research/bench_render.py` says
    nothing should.

    What they are worth rests on the addressing. The flow prior and the
    vocoder's excitation are drawn per utterance from that utterance's seed, and
    the speaker embedding is per utterance too, so a row that borrows its
    neighbour's noise or its neighbour's voice renders someone else's audio
    under its own seed. That is what these tests hold.

    Byte equality is held where it is true and nowhere else. A batch of one is
    byte-equal, always. **Above one row it is not, and no setting makes it so on
    every device, so this is not a drop-in for anything that has to reproduce a
    render.** Measured at production shapes: the CPU mel moves 1.6e-2 max and
    2.9e-3 rms, which through the vocoder is a waveform correlating 0.99871 with
    its own single call, -25.9 dB error to signal. The MPS mel and the CPU
    vocoder do not move at all; the MPS vocoder moves 1.5e-6.

    The cause is the intra-op reduction order, not the addressing: two identical
    rows in one batched CPU call disagree by the same 1.6e-2, and
    `torch.set_num_threads(1)` removes it. The same mechanism already moves the
    *single* call by 4.7e-2 between a one-thread and a five-thread process, so
    the batch widens an existing gap rather than opening a new one. See
    ``docs/design/models-notes.md``.
    """

    WINDOW = WindowConfig(
        max_speech_tokens=32, static_length=32, pad_token_id=4254, static_prompt_tokens=16
    )
    """Static, because that is what makes every row the same width. Small,
    because the property has nothing to do with the size."""

    SEEDS = (11, 22, 33)

    def _config(self) -> AlgorithmConfig:
        """`WINDOW` with the two token budgets that have to agree with it."""
        from loudkit.config import ChunkConfig

        return AlgorithmConfig(
            window=self.WINDOW,
            chunking=ChunkConfig(max_tokens=self.WINDOW.max_speech_tokens),
            sampling=SamplingConfig(max_new_tokens=self.WINDOW.max_speech_tokens),
        )

    @staticmethod
    def _two_voices() -> tuple[VoiceProfile, VoiceProfile]:
        """Two genuinely different voices: the speaker embedding is per row too."""
        import dataclasses

        one = _voice(prompt_tokens=16, prompt_frames=32)
        return one, dataclasses.replace(one, flow_embedding=-one.flow_embedding)

    @staticmethod
    def _row_carries_its_own_draw(
        single: list[NDArray[np.float32]],
        batched: list[NDArray[np.float32]],
        wrong: list[NDArray[np.float32]],
    ) -> None:
        """Each batched row is its own single call, not its neighbour's.

        Stated as a ratio rather than a tolerance: the agreement with the row's
        own draw has to be orders below the disagreement with the neighbour's,
        so the check keeps its meaning on a device that rounds differently.
        """
        for i, (mine, batch, theirs) in enumerate(zip(single, batched, wrong, strict=True)):
            assert batch.shape == mine.shape, i
            agree = float(np.abs(batch - mine).max())
            differ = float(np.abs(mine - theirs).max())
            assert agree < differ / 100.0, (
                f"row {i} sits {agree:.2e} from its own draw and {differ:.2e} from "
                f"its neighbour's: the batch is not addressing rows by seed"
            )

    def test_the_batched_vocoder_gives_every_row_its_own_seed(self) -> None:
        from loudkit.models.vocoder import TorchVocoder

        torch.manual_seed(0)
        voc = TorchVocoder(self._config()).eval()
        one, two = self._two_voices()
        voices = [one, two, one]
        rng = np.random.default_rng(1)
        mels = [rng.normal(size=(80, n)).astype(np.float32) for n in (5, 31, 64)]
        shuffled = [self.SEEDS[1], self.SEEDS[2], self.SEEDS[0]]

        single = [
            voc.synthesize(m, v, seed=s)
            for m, v, s in zip(mels, voices, self.SEEDS, strict=True)
        ]
        wrong = [
            voc.synthesize(m, v, seed=s) for m, v, s in zip(mels, voices, shuffled, strict=True)
        ]
        batched = voc.synthesize_batch(mels, voices, seeds=list(self.SEEDS))
        self._row_carries_its_own_draw(single, batched, wrong)

        # One row is the whole single-utterance path, byte for byte.
        np.testing.assert_array_equal(
            single[0], voc.synthesize_batch(mels[:1], voices[:1], seeds=[self.SEEDS[0]])[0]
        )

    def test_the_batched_mel_decoder_gives_every_row_its_own_seed_and_voice(self) -> None:
        from loudkit.models.flow import TorchMelDecoder

        torch.manual_seed(0)
        dec = TorchMelDecoder(self._config()).eval()
        one, two = self._two_voices()
        voices = [one, two, one]
        tokens: list[list[int]] = [[3, 1, 4, 1, 5], list(range(9, 41)), [2, 7]]
        shuffled = [self.SEEDS[1], self.SEEDS[2], self.SEEDS[0]]

        single = [
            dec.decode(t, v, seed=s) for t, v, s in zip(tokens, voices, self.SEEDS, strict=True)
        ]
        # Wrong seed and wrong voice at once: both are addressed per row.
        wrong = [
            dec.decode(t, v, seed=s)
            for t, v, s in zip(tokens, [two, one, two], shuffled, strict=True)
        ]
        batched = dec.decode_batch(tokens, voices, seeds=list(self.SEEDS))
        self._row_carries_its_own_draw(single, batched, wrong)

        np.testing.assert_array_equal(
            single[0], dec.decode_batch(tokens[:1], voices[:1], seeds=[self.SEEDS[0]])[0]
        )

        with pytest.raises(ValueError, match="one voice and one seed"):
            dec.decode_batch(tokens, voices, seeds=[1])

    def test_the_batch_is_the_same_render_within_rounding(self) -> None:
        """Rounding, not a different renderer.

        Above one row the batch is not byte-equal, and no condition makes it so
        on every shape: measured, a batched CPU mel lands up to 1.6e-2 from its
        own single call at production shapes, while the MPS mel and the CPU
        vocoder agree exactly there. Two *identical* rows inside one batch also
        disagree with each other, and pinning `torch.set_num_threads(1)` removes
        the CPU difference at those shapes, which together say the cause is the
        intra-op reduction order rather than the addressing.

        So the bound held here is correlation, which is the shape of the claim
        that survives a different machine. A row that took the wrong noise or
        the wrong voice lands far below it: drift-checked, rolling the seeds by
        one row brings the vocoder to 0.9984 and the mel to 4.9 absolute, both
        red here. See the two tests above, which pin that separately and by a
        margin.

        Correlation on the *mel*, which is what this compares. The waveform the
        production mel renders to correlates lower, 0.99871, because the vocoder
        is where a small mel difference becomes an audible-scale one. That
        number is in ``docs/design/models-notes.md`` rather than here: it needs
        real weights and a production window, and this test has neither.
        """
        from loudkit.models.flow import TorchMelDecoder
        from loudkit.models.vocoder import TorchVocoder

        torch.manual_seed(0)
        cfg = self._config()
        one, two = self._two_voices()
        voices = [one, two, one]
        seeds = list(self.SEEDS)
        tokens: list[list[int]] = [[3, 1, 4, 1, 5], list(range(9, 41)), [2, 7]]

        dec = TorchMelDecoder(cfg).eval()
        mels = [dec.decode(t, v, seed=s) for t, v, s in zip(tokens, voices, seeds, strict=True)]
        for i, batch in enumerate(dec.decode_batch(tokens, voices, seeds=seeds)):
            corr = float(np.corrcoef(mels[i].ravel(), batch.ravel())[0, 1])
            assert corr > 0.99999, f"mel row {i} correlates {corr:.7f} with its own call"

        voc = TorchVocoder(cfg).eval()
        wav = [
            voc.synthesize(m, v, seed=s) for m, v, s in zip(mels, voices, seeds, strict=True)
        ]
        for i, batch in enumerate(voc.synthesize_batch(mels, voices, seeds=seeds)):
            corr = float(np.corrcoef(wav[i], batch)[0, 1])
            assert corr > 0.99999, f"waveform row {i} correlates {corr:.7f} with its own call"

    def test_the_batch_holds_under_dual_path_guidance(self) -> None:
        """Guidance doubles the prior inside the estimator, so the timestep has
        to be one per row of the doubled tensor rather than the pair a single
        utterance needed. No shipped checkpoint asks for this mode; the code
        carries it, so it is held."""
        from loudkit.models.flow import TorchMelDecoder

        torch.manual_seed(0)
        cfg = self._config().with_(guidance="cfg_dual_path", guidance_rate=0.7)
        dec = TorchMelDecoder(cfg).eval()
        voices = list(self._two_voices())
        tokens: list[list[int]] = [[3, 1, 4, 1, 5], list(range(9, 41))]
        seeds = list(self.SEEDS[:2])

        single = [
            dec.decode(t, v, seed=s) for t, v, s in zip(tokens, voices, seeds, strict=True)
        ]
        for i, batch in enumerate(dec.decode_batch(tokens, voices, seeds=seeds)):
            worst = float(np.abs(batch - single[i]).max())
            assert worst < 1e-5, f"row {i} moved {worst:.2e}"

    def test_the_batch_refuses_rows_of_different_widths(self) -> None:
        """Not a padding problem that a mask would solve. The conformer attends
        over the whole row unmasked and the vocoder's iSTFT tail is not
        padding-invariant, so a short row padded out to its neighbour comes back
        changed. Refused, so a caller groups by length rather than finding out.
        """
        from loudkit.models.vocoder import TorchVocoder

        # Ragged: the padded length is a function of the mel, which is exactly
        # what makes two mels disagree about it.
        voc = TorchVocoder(AlgorithmConfig(window=PRODUCTION_WINDOW), ragged=True).eval()
        one, _ = self._two_voices()
        rng = np.random.default_rng(1)
        mels = [rng.normal(size=(80, n)).astype(np.float32) for n in (5, 200)]
        with pytest.raises(ValueError, match=r"one padded length.*\[64, 256\]"):
            voc.synthesize_batch(mels, [one, one], seeds=[1, 2])

        with pytest.raises(ValueError, match="one voice and one seed"):
            voc.synthesize_batch(mels, [one], seeds=[1, 2])
