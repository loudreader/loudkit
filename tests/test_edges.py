"""Rendered windows meet at zero; enrollment silence fits inside the prompt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import loudkit
from loudkit import END_SILENCE_SECONDS, with_prompt_ending_in_silence
from loudkit.config import EDGE_FADE_SECONDS
from loudkit.window import _fade_edges

SR = 24_000


class TestFadeEdges:
    def test_both_edges_start_and_end_at_zero(self) -> None:
        audio = np.full(SR, 0.25, dtype=np.float32)
        out = _fade_edges(audio, sample_rate=SR, seconds=EDGE_FADE_SECONDS)
        assert out[0] == 0.0
        assert out[-1] == 0.0
        assert out.dtype == audio.dtype

    def test_the_middle_is_untouched(self) -> None:
        rng = np.random.default_rng(7)
        audio = rng.standard_normal(SR).astype(np.float32)
        out = _fade_edges(audio, sample_rate=SR, seconds=EDGE_FADE_SECONDS)
        n = int(EDGE_FADE_SECONDS * SR)
        np.testing.assert_array_equal(out[n:-n], audio[n:-n])

    def test_the_ramp_is_monotonic_and_symmetric(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        out = _fade_edges(audio, sample_rate=SR, seconds=EDGE_FADE_SECONDS)
        n = int(EDGE_FADE_SECONDS * SR)
        head, tail = out[:n], out[-n:]
        assert np.all(np.diff(head) >= 0)
        np.testing.assert_allclose(head, tail[::-1], atol=1e-7)

    def test_a_window_shorter_than_two_ramps_is_left_alone(self) -> None:
        audio = np.ones(10, dtype=np.float32)
        out = _fade_edges(audio, sample_rate=SR, seconds=EDGE_FADE_SECONDS)
        np.testing.assert_array_equal(out, audio)

    def test_the_ramp_length_comes_from_the_algorithm(self) -> None:
        """A checkpoint may name another ramp; the renderer must obey it, or the
        fingerprint would move for a change the audio never made."""
        audio = np.ones(SR, dtype=np.float32)
        out = _fade_edges(audio, sample_rate=SR, seconds=0.008)
        n = int(0.008 * SR)
        assert out[0] == 0.0
        assert out[n - 1] == 1.0
        assert out[n - 2] < 1.0

    def test_the_input_is_not_modified(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        _fade_edges(audio, sample_rate=SR, seconds=EDGE_FADE_SECONDS)
        assert audio[0] == 1.0

    def test_every_pinned_ramp_is_this_function_s_own_output(self) -> None:
        """The four ports carry these bits as tables; here is where they come from.

        numpy takes the cosine in float32 off a float32 ``linspace``. A port
        that evaluates it in double and narrows lands within two units in the
        last place, so the lengths a release actually applies are pinned rather
        than recomputed abroad, and this is the assertion that says the pins are
        the renderer's own output and not a second spelling of it.
        """
        fixture = json.loads(
            (Path(__file__).resolve().parent / "data/conformance/edge_fade.json").read_text(
                encoding="utf-8"
            )
        )
        seconds = [ramp["seconds"] for ramp in fixture["ramps"]]
        assert EDGE_FADE_SECONDS in seconds, "the shipped ramp is pinned by no table"
        assert 0.005 in seconds, "the historical ramp stays pinned"
        # The 5 ms ramp is repeated at the top level, where a reader written
        # against the older fixture already looks for it.
        assert fixture["samples"] == fixture["ramps"][0]["samples"]
        assert fixture["bits"] == fixture["ramps"][0]["bits"]
        for ramp in fixture["ramps"]:
            n = ramp["samples"]
            assert n == int(ramp["seconds"] * fixture["sample_rate"])
            out = _fade_edges(
                np.ones(4 * n, dtype=np.float32),
                sample_rate=fixture["sample_rate"],
                seconds=ramp["seconds"],
            )
            assert out[:n].view(np.uint32).tolist() == ramp["bits"], ramp["seconds"]
            assert out[-n:].view(np.uint32).tolist()[::-1] == ramp["bits"], ramp["seconds"]


class TestPromptEndingInSilence:
    def test_the_cut_lives_with_the_other_reference_audio_dsp(self) -> None:
        """It moved out of the package root, and `loudkit` re-exports it.

        Both doors are asserted because the re-export is the compatible half of
        that move: an import from `loudkit` is what every caller and this file
        already write, and dropping one of the six names would be silent.
        """
        from loudkit.models import enrollment_audio

        cut = enrollment_audio.with_prompt_ending_in_silence
        assert with_prompt_ending_in_silence is cut
        for name in (
            "PROMPT_SECONDS",
            "PAUSE_FLOOR_DBFS",
            "PAUSE_MIN_SECONDS",
            "CLIP_MIN_SECONDS",
            "END_SILENCE_SECONDS",
        ):
            assert getattr(loudkit, name) == getattr(enrollment_audio, name), name
        # The frame the pause detector measures over, unnamed until the move.
        assert enrollment_audio.PAUSE_FRAME_SECONDS == 0.005

    def _clip(self, seconds: float, pause_at: float | None) -> np.ndarray:
        rng = np.random.default_rng(3)
        clip = (0.3 * rng.standard_normal(int(seconds * SR))).astype(np.float32)
        if pause_at is not None:
            a = int(pause_at * SR)
            clip[a : a + int(0.1 * SR)] = 0.0
        return clip

    def test_it_cuts_at_the_last_pause_before_the_limit_and_pads(self) -> None:
        clip = self._clip(12.0, pause_at=7.5)
        clip[4 * SR : int(4.1 * SR)] = 0.0
        clip[int(11.0 * SR) : int(11.1 * SR)] = 0.0  # a pause past the limit does not count
        out = with_prompt_ending_in_silence(clip, SR)
        assert len(out) == int(7.5 * SR) + int(END_SILENCE_SECONDS * SR)
        assert not out[int(7.5 * SR) :].any()
        np.testing.assert_array_equal(out[: int(7.5 * SR)], clip[: int(7.5 * SR)])

    def test_a_clip_without_a_pause_only_gets_the_silence(self) -> None:
        clip = self._clip(8.0, pause_at=None)
        out = with_prompt_ending_in_silence(clip, SR)
        assert len(out) == len(clip) + int(END_SILENCE_SECONDS * SR)
        np.testing.assert_array_equal(out[: len(clip)], clip)

    def test_a_pause_too_early_is_not_taken(self) -> None:
        clip = self._clip(8.0, pause_at=1.0)  # cutting there would leave 1 s of voice
        out = with_prompt_ending_in_silence(clip, SR)
        assert len(out) == len(clip) + int(END_SILENCE_SECONDS * SR)

    def test_dtype_is_kept(self) -> None:
        clip = self._clip(5.0, pause_at=4.0)
        assert with_prompt_ending_in_silence(clip, SR).dtype == np.float32

    @pytest.mark.parametrize("sample_rate", [16_000, 24_000, 44_100])
    @pytest.mark.parametrize("pause_at", [None, 1.0])
    def test_long_clip_keeps_silence_inside_prompt(
        self, sample_rate: int, pause_at: float | None
    ) -> None:
        clip = np.full(12 * sample_rate, 0.3, dtype=np.float32)
        if pause_at is not None:
            start = int(pause_at * sample_rate)
            clip[start : start + int(0.1 * sample_rate)] = 0
        out = with_prompt_ending_in_silence(clip, sample_rate)
        tail = int(END_SILENCE_SECONDS * sample_rate)
        assert len(out) == 10 * sample_rate
        assert not out[-tail:].any()
        np.testing.assert_array_equal(out[:-tail], clip[: len(out) - tail])

    @pytest.mark.parametrize("sample_rate", [16_000, 24_000, 44_100])
    def test_late_pause_is_kept_instead_of_cutting_an_earlier_word(
        self, sample_rate: int
    ) -> None:
        clip = np.full(12 * sample_rate, 0.3, dtype=np.float32)
        pause = int(9.8 * sample_rate)
        clip[pause : int(9.9 * sample_rate)] = 0
        out = with_prompt_ending_in_silence(clip, sample_rate)
        np.testing.assert_array_equal(out[:pause], clip[:pause])
        assert not out[pause:].any()
        cut = len(out) - int(END_SILENCE_SECONDS * sample_rate)
        assert pause <= cut < int(9.9 * sample_rate)
        np.testing.assert_array_equal(out[:cut], clip[:cut])
        prompt = out[: 10 * sample_rate]
        assert not prompt[-int(0.1 * sample_rate) :].any()
