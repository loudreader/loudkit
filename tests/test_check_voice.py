"""The stall gate's statistic, pinned on fixtures rather than on a render.

The probe in tools/check_voice.py renders; these tests do not. What must not
drift is the arithmetic between a token stream and a verdict: which ids count
as silence, how runs are counted across chunk boundaries, where the band-B
share lands, and which side of each calibrated line a known shape falls on.
A render can move with the checkpoint; the classifier must only move when
someone means it to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import tool

REPO = Path(__file__).resolve().parent.parent


cv = tool("check_voice")

SPEECH = [3377, 1200, 880, 2500]  # ordinary speech ids, none in either band


def _stream(*pieces: list[int]) -> list[int]:
    out: list[int] = []
    for piece in pieces:
        out.extend(piece)
    return out


class TestBandStats:
    def test_the_two_bands_are_the_split_fsq_cell(self) -> None:
        assert frozenset((4137, 4215, 4218, 4299)) == cv.BAND_A
        assert frozenset((6162, 6324, 6405, 6486)) == cv.BAND_B
        # +2187 is one step of digit 7: each listed A id maps onto a B id.
        assert {4137 + 2187, 4218 + 2187, 4299 + 2187} <= cv.BAND_B

    def test_share_counts_tokens_not_runs(self) -> None:
        stats = cv.band_stats([_stream(SPEECH, [4218, 4218, 4218, 6405], SPEECH)])
        assert stats["band_a"] == 3
        assert stats["band_b"] == 1
        assert stats["band_b_share"] == pytest.approx(0.25)
        assert stats["pause_runs"] == 1
        assert stats["run_max"] == 4

    def test_runs_do_not_bridge_chunks(self) -> None:
        # 3 trailing + 3 leading silence in adjacent chunks is two runs of 3,
        # not one of 6: the probe judges streams chunk by chunk, as the
        # engine emits them.
        stats = cv.band_stats(
            [
                _stream(SPEECH, [4137] * 3),
                _stream([4137] * 3, SPEECH),
            ]
        )
        assert stats["pause_runs"] == 2
        assert stats["run_max"] == 3

    def test_no_silence_yields_none_share(self) -> None:
        stats = cv.band_stats([list(SPEECH)])
        assert stats["silence_tokens"] == 0
        assert stats["band_b_share"] is None


class TestClassify:
    def _healthy(self) -> list[list[int]]:
        # A colette-shaped probe: pauses spread over both bands, runs of
        # healthy length. 30 runs x 4 tokens alternating A/B = share 0.5.
        return [_stream(SPEECH, [4218, 6405, 4137, 6486], SPEECH) for _ in range(30)]

    def _prone(self) -> list[list[int]]:
        # A soren-shaped probe: the same pause count concentrated on band A
        # with one free-run past the 100-token stall line (a 4 s hole).
        chunks = [_stream(SPEECH, [4218, 4218, 4137, 4218], SPEECH) for _ in range(29)]
        chunks.append(_stream(SPEECH, [4218] * 150, SPEECH))
        return chunks

    def test_spread_pauses_pass(self) -> None:
        verdict, _ = cv.classify(cv.band_stats(self._healthy()))
        assert verdict == "pass"

    def test_concentration_and_a_free_run_fail(self) -> None:
        stats = cv.band_stats(self._prone())
        verdict, reasons = cv.classify(stats)
        assert verdict == "fail"
        # Both signatures are named: the share and the run.
        assert any("band-B share" in r for r in reasons)
        assert any("run" in r for r in reasons)

    def test_a_majority_silent_cap_hit_fails_on_its_own(self) -> None:
        verdict, reasons = cv.classify(cv.band_stats(self._healthy()), stall_caps=1)
        assert verdict == "fail"
        assert any("cap" in r for r in reasons)

    def test_a_healthy_long_run_is_not_the_stall_line(self) -> None:
        # freja produced a 29-token run and is clean; the line is 100, and
        # 39 was the clean-roster maximum. Both sides of the calibration.
        chunks = self._healthy()
        chunks.append(_stream(SPEECH, [4218, 6405] * 20, SPEECH))  # run of 40
        stats = cv.band_stats(chunks)
        assert stats["run_max"] == 40
        verdict, _ = cv.classify(stats)
        assert verdict == "pass"

    def test_a_thin_probe_warns_instead_of_passing(self) -> None:
        # One clean pause is not evidence of health.
        stats = cv.band_stats([_stream(SPEECH, [6405, 4218], SPEECH)])
        verdict, reasons = cv.classify(stats)
        assert verdict == "warn"
        assert any("thin" in r for r in reasons)

    def test_the_warn_band_sits_between_the_calibrated_lines(self) -> None:
        # Share 0.15: above every patched voice, below every clean one.
        chunks = []
        for index in range(40):
            band = 6405 if index < 6 else 4218
            chunks.append(_stream(SPEECH, [band, band], SPEECH))
        stats = cv.band_stats(chunks)
        assert cv.FAIL_BELOW < stats["band_b_share"] < cv.WARN_BELOW
        verdict, _ = cv.classify(stats)
        assert verdict == "warn"


class TestParser:
    def test_defaults(self) -> None:
        args = cv._parser().parse_args(["mine.safetensors"])
        assert args.passages == 4
        assert args.seed == 1234
        assert args.checkpoint == "loudreader/loudr-1"
        assert args.reading_set.name == "reading-en.json"

    def test_flags_land_where_they_say(self) -> None:
        args = cv._parser().parse_args(
            ["soren", "--checkpoint", "dist/loudr-1", "--passages", "8", "--seed", "7"]
        )
        assert (args.profile_path.name, args.passages, args.seed) == ("soren", 8, 7)
