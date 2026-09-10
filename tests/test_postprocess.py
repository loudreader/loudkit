"""The postprocess layer, against the shared conformance fixture.

Every case in ``tests/data/conformance/postprocess.json`` is a regression from
the shipped reader or a named device trace, and every port runs the same file.
This module is the Python side of that; Go, Rust, TypeScript and Swift have
theirs, and a rule that drifts in one language fails in one language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from loudkit.config import AlgorithmConfig, SamplingConfig
from loudkit.postprocess import (
    PostprocessConfig,
    ceiling_for,
    desperation_cut,
    ended_tail_trim,
    inspect,
    is_stalled,
    is_trailing_filler,
    pacing_outliers,
    repetition_cut,
    terminal_echo_cut,
)
from loudkit.sampler import LRSamplerV1

FIXTURE = Path(__file__).parent / "data" / "conformance" / "postprocess.json"


@pytest.fixture(scope="module")
def fx() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def build(shape: list[list[Any]]) -> list[int]:
    """The fixture's token-shape builder, spelled out in the fixture header."""
    out: list[int] = []
    for segment in shape:
        kind, count = segment[0], segment[1]
        if kind == "speech":
            out.extend(20 + i % 60 for i in range(count))
        elif kind == "quiet":
            out.extend(i % 8 for i in range(count))
        elif kind == "sil":
            # True digital silence in the stall section's two-class scheme.
            out.extend(i % 4 for i in range(count))
        elif kind == "breath":
            # The contextually-quiet family: extends a dead-air run without
            # counting toward its gate.
            out.extend(4 + i % 4 for i in range(count))
        elif kind == "cycle":
            # `count` is the period here and segment[2] the repeat count.
            cycle = [20 + i % 60 for i in range(count)]
            out.extend(cycle * segment[2])
        elif kind == "cycle_mixed":
            # Second half silence: the word-then-pause stutter.
            half = count // 2
            cycle = [20 + i for i in range(count - half)] + [i % 8 for i in range(half)]
            out.extend(cycle * segment[2])
        elif kind == "dead":
            # True silence only the render census knows (the 6405 class):
            # outside the fixture's sampler list, inside its silence census.
            out.extend([12] * count)
        elif kind == "sigh":
            # Breath only the quiet census knows.
            out.extend([13] * count)
        elif kind == "cycle_dead":
            # The stutter with its pause on a census-only id.
            half = count // 2
            cycle = [20 + i for i in range(count - half)] + [12] * half
            out.extend(cycle * segment[2])
        else:  # pragma: no cover - a typo in the fixture, not a code path
            raise ValueError(f"unknown segment kind {kind!r}")
    return out


def config_from(fx: dict[str, Any], **overrides: Any) -> PostprocessConfig:
    return PostprocessConfig(**{**fx["config"], **overrides})


class TestCeiling:
    def test_cases(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        for case in fx["ceiling"]:
            got = ceiling_for(case["text_tokens"], config=cfg, window=case["window"])
            assert got == case["expect"], f"{case['name']}: {case['why']}"


class TestLanguageGuard:
    """The ceiling was settled on English traces; nine languages ship.

    Speech tokens per *text* token is a property of the orthography, so a
    constant tuned on one language is an assumption everywhere else — and the
    expensive direction of that assumption is a guard that truncates correct
    speech in a language nobody measured.

    Measured with one voice held constant across nine language tags (the
    voice-to-voice spread on a single sentence is larger than the
    language-to-language spread, so mixing narrators would confound exactly the
    effect being tested).
    """

    def test_the_ceiling_arithmetic_holds_per_language(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        cases = fx["language_guard"]["cases"]
        assert cases, "the fixture has no language_guard cases; nothing was compared"
        for case in cases:
            got = ceiling_for(case["text_tokens"], config=cfg, window=case["window"])
            assert got == case["expect"], f"{case['name']}: {case['why']}"

    def test_no_ordinary_sentence_is_truncated_in_any_language(
        self, fx: dict[str, Any]
    ) -> None:
        cfg = config_from(fx)
        stopped = []
        for case in fx["language_guard"]["cases"]:
            ceiling = ceiling_for(case["text_tokens"], config=cfg, window=case["window"])
            hit = case["measured_speech_tokens"] >= ceiling
            assert hit == case["expect_stopped_by_ceiling"], (
                f"{case['name']} changed side of the ceiling: {case['why']}"
            )
            if hit:
                stopped.append(case["name"])
        # One row is expected to be here and it is not a false positive: a
        # Spanish three-word phrase whose decoder never emitted a stop token.
        # The guard caught a runaway; it did not cut a legitimate read.
        assert stopped == ["es_short"], (
            "the set of rows the ceiling stops changed — a new entry is a "
            f"language being truncated by an English-tuned constant: {stopped}"
        )


class TestRepetition:
    """The loop the tail rules cannot see, because it happens mid-row.

    Every other rule here reads the end of the chunk. A stuck decoder repeats
    inside it, and the literature puts that failure first or second in every
    ranking of what goes wrong with autoregressive speech models — so a layer
    that claims to catch generation artifacts and has no rule for it has a hole
    where its most-cited failure should be.
    """

    def test_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        cases = fx["repetition"]
        assert cases, "the fixture has no repetition cases; nothing was compared"
        for case in cases:
            got = repetition_cut(
                build(case["shape"]), silence=fx["silence_token_ids"], config=cfg
            )
            assert got == case["expect"], f"{case['name']}: {case['why']}"

    def test_the_negatives_outnumber_nothing(self, fx: dict[str, Any]) -> None:
        # A mid-sequence cut is the most destructive thing this layer can do, so
        # the cases that must NOT fire carry more weight than the ones that must.
        negatives = [c for c in fx["repetition"] if c["expect"] is None]
        assert len(negatives) >= 4, "too few negative cases to trust a mid-row cut"


class TestStall:
    """The decoder trapped in silence — the failure no tail rule can see.

    Every mute chunk and every mid-row hole in the interior-stall study
    shipped as ``clean``, because all six other rules anchor on the tail. The
    stall rule condemns instead of cutting: the failure is a hole, and the fix
    is the retry ladder. Detection is two-class (a true-silence gate, a
    quiet-family continuation), which the fixture pins because single-set
    counting was measured broken.
    """

    @staticmethod
    def _config(fx: dict[str, Any], case: dict[str, Any]) -> PostprocessConfig:
        if not case["render_ids"]:
            # The fallback arm: a checkpoint packed before the censuses.
            return config_from(fx)
        section = fx["stall"]
        return config_from(
            fx,
            silence_render_ids=tuple(section["silence_render_ids"]),
            quiet_render_ids=tuple(section["quiet_render_ids"]),
        )

    def test_the_rule_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        cases = fx["stall"]["cases"]
        assert cases, "the fixture has no stall cases; nothing was compared"
        for case in cases:
            got = is_stalled(
                build(case["shape"]),
                hit_ceiling=case["hit_ceiling"],
                silence=fx["silence_token_ids"],
                config=self._config(fx, case),
            )
            assert got == case["stalled"], f"{case['name']}: {case['why']}"

    def test_the_resolver_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        # The wiring is part of the contract: after repetition, before every
        # tail rescue, condemned like dropout.
        for case in fx["stall"]["cases"]:
            got = inspect(
                build(case["shape"]),
                text_token_count=case["text_tokens"],
                min_tokens=case["min_tokens"],
                eos_peak_at=case["eos_peak_at"],
                eos_peak_prob=case["eos_peak_prob"],
                ended=case["ended"],
                is_terminal=case["is_terminal"],
                hit_ceiling=case["hit_ceiling"],
                silence=fx["silence_token_ids"],
                config=self._config(fx, case),
            )
            want = case["expect"]
            why = f"{case['name']}: {case['why']}"
            assert got.keep == want["keep"], why
            assert got.reason == want["reason"], why
            assert got.suspect == want["suspect"], why

    def test_it_never_cuts(self, fx: dict[str, Any]) -> None:
        section = fx["stall"]
        cfg = config_from(
            fx,
            silence_render_ids=tuple(section["silence_render_ids"]),
            quiet_render_ids=tuple(section["quiet_render_ids"]),
        )
        row = build([["speech", 30], ["sil", 30], ["speech", 30]])
        got = inspect(
            row,
            text_token_count=40,
            min_tokens=48,
            eos_peak_at=-1,
            eos_peak_prob=0.0,
            ended=True,
            is_terminal=True,
            hit_ceiling=False,
            silence=fx["silence_token_ids"],
            config=cfg,
        )
        assert got.reason == "stall"
        assert got.keep == len(row), (
            "a stalled row must be handed back whole; the hole is mid-row and "
            "no cut can remove it"
        )
        assert got.suspect, "the caller has to be told, since nothing was changed"

    def test_repetition_outranks_it(self, fx: dict[str, Any]) -> None:
        # A row that both loops and stalls answers to the loop: an exactly
        # repeated cycle pins where the failure began. Here the decoder
        # resumed after the region, so the loop condemns rather than cuts —
        # but it still outranks the stall's condemnation, and the verdict
        # names the anchor that was found.
        section = fx["stall"]
        cfg = config_from(
            fx,
            silence_render_ids=tuple(section["silence_render_ids"]),
            quiet_render_ids=tuple(section["quiet_render_ids"]),
        )
        row = build([["cycle", 4, 8], ["sil", 30], ["speech", 20]])
        got = inspect(
            row,
            text_token_count=40,
            min_tokens=48,
            eos_peak_at=-1,
            eos_peak_prob=0.0,
            ended=True,
            is_terminal=True,
            hit_ceiling=False,
            silence=fx["silence_token_ids"],
            config=cfg,
        )
        assert got.reason == "repetition", "the exact anchor outranks the condemnation"

    def test_zero_run_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="stall_run_tokens"):
            PostprocessConfig(stall_run_tokens=0)

    def test_every_count_is_covered_by_the_non_negative_rule(self) -> None:
        """The two names the `>= 0` loop omitted, and why they are in it now.

        A negative `desperation_band_floor` closes the band it exists to open
        and a negative `dropout_min_tokens` makes no row short enough to be a
        dropout: both are typos in a manifest, refused like every other count.
        Rust holds its counts as `usize` and cannot represent either, so while
        these were unnamed here that port refused two manifests the reference
        accepted.
        """
        for name in ("desperation_band_floor", "dropout_min_tokens"):
            with pytest.raises(ValueError, match=f"{name} must be >= 0: -1"):
                PostprocessConfig(**{name: -1})
        # Zero is a configuration, not a typo, and still loads.
        PostprocessConfig(desperation_band_floor=0, dropout_min_tokens=0)


class TestStarvedRescue:
    """A cap-hit desperation cut that keeps less than any full read.

    The one row that survived the stall fix: soren/da0028 chunk 4 burned 132
    tokens to the ceiling and the seam cut kept 36 — 1.44 s in which 33 of
    the 36 kept tokens render near-silent through ids outside both manifest
    censuses, invisible to every set-membership rule. The keep's *length* is
    the only evidence there is: a keep under
    ``desperation_min_keep_per_text_token`` per text token cannot hold a full
    read, so the verdict is condemned into the retry ladder. The cut stands
    as the keep — an exhausted ladder ships the trim, flagged, rather than
    the untrimmed babble.
    """

    def _inspect(self, fx: dict[str, Any], case: dict[str, Any], **overrides: Any):
        return inspect(
            build(case["shape"]),
            text_token_count=case["text_tokens"],
            min_tokens=case["min_tokens"],
            eos_peak_at=case["eos_peak_at"],
            eos_peak_prob=case["eos_peak_prob"],
            ended=case["ended"],
            is_terminal=case["is_terminal"],
            hit_ceiling=case["hit_ceiling"],
            silence=fx["silence_token_ids"],
            config=config_from(fx, **overrides),
        )

    def test_the_resolver_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        cases = fx["starved_rescue"]["cases"]
        assert cases, "the fixture has no starved_rescue cases; nothing was compared"
        for case in cases:
            got = self._inspect(fx, case)
            want = case["expect"]
            why = f"{case['name']}: {case['why']}"
            assert got.keep == want["keep"], why
            assert got.reason == want["reason"], why
            assert got.suspect == want["suspect"], why

    def test_the_floor_is_exclusive(self, fx: dict[str, Any]) -> None:
        # keep == floor ships: `<`, not `<=`, so the pinned law has no
        # ambiguity at the boundary for a port to resolve differently.
        # text 20 puts the floor at exactly 34.0.
        case = dict(fx["starved_rescue"]["cases"][0])
        case["shape"] = [["speech", 34], ["sil", 12], ["speech", 90]]
        case["text_tokens"] = 20
        case["min_tokens"] = 24
        got = self._inspect(fx, case)
        assert got.reason == "desperation"
        assert got.keep == 34
        assert not got.suspect, "a keep exactly at the floor is not under it"

    def test_zero_disables_the_trigger(self, fx: dict[str, Any]) -> None:
        case = fx["starved_rescue"]["cases"][0]
        got = self._inspect(fx, case, desperation_min_keep_per_text_token=0.0)
        assert got.reason == "desperation"
        assert not got.suspect

    def test_a_negative_floor_is_refused(self) -> None:
        with pytest.raises(ValueError, match="desperation_min_keep_per_text_token"):
            PostprocessConfig(desperation_min_keep_per_text_token=-0.1)

    def test_a_floor_above_the_band_is_refused(self) -> None:
        # Above `desperation_band_ratio` the rule would condemn cuts landing
        # exactly where the band admits them.
        with pytest.raises(ValueError, match="desperation_band_ratio"):
            PostprocessConfig(desperation_min_keep_per_text_token=2.7)

    def test_the_condemned_verdict_keeps_the_cut(self, fx: dict[str, Any]) -> None:
        # Unlike `stall`, which hands the row back whole (a hole has no
        # anchor), the starved verdict carries the trim: when every retry is
        # also condemned the engine ships `keep` — today's audio — rather
        # than the full cap-hit babble.
        case = fx["starved_rescue"]["cases"][0]
        got = self._inspect(fx, case)
        assert got.suspect
        assert got.keep < len(build(case["shape"]))


class TestRepetitionSilence:
    """The loop exemption keys on acoustic silence, not the sampler list.

    The specimen: kathleen/en0023 seed 1234 parked a mid-chunk pause on ids
    6486 (x7) then 6405 (x24) — both render true silence, both in the
    manifest's ``silence_render_ids``, neither in the sampler's list the
    exemption used to read. The period-1 run fired as a loop and the cut
    deleted the pause plus two whole sentences of correctly-read speech
    behind it, verdict ``repetition``, not suspect, audibly fluent. Six of
    the checkpoint's eight truly-silent ids sit outside the sampler list, so
    the exemption was blind on most real pauses. The law now unions the
    sampler list with both render censuses for this one rule; the tail rules
    keep the list they were calibrated against.
    """

    @staticmethod
    def _config(fx: dict[str, Any], **overrides: Any) -> PostprocessConfig:
        section = fx["repetition_silence"]
        return config_from(
            fx,
            silence_render_ids=tuple(section["silence_render_ids"]),
            quiet_render_ids=tuple(section["quiet_render_ids"]),
            **overrides,
        )

    def test_the_rule_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        cases = fx["repetition_silence"]["cases"]
        assert cases, "the fixture has no repetition_silence cases; nothing was compared"
        for case in cases:
            got = repetition_cut(
                build(case["shape"]),
                silence=fx["silence_token_ids"],
                config=self._config(fx),
            )
            assert got == case["loop"], f"{case['name']}: {case['why']}"

    def test_the_resolver_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        # The cascade is part of the contract: a declined loop falls through
        # to `stall`, which condemns the specimen's pause into the retry
        # ladder instead of shipping the cut.
        for case in fx["repetition_silence"]["cases"]:
            got = inspect(
                build(case["shape"]),
                text_token_count=case["text_tokens"],
                min_tokens=case["min_tokens"],
                eos_peak_at=case["eos_peak_at"],
                eos_peak_prob=case["eos_peak_prob"],
                ended=case["ended"],
                is_terminal=case["is_terminal"],
                hit_ceiling=case["hit_ceiling"],
                silence=fx["silence_token_ids"],
                config=self._config(fx),
            )
            want = case["expect"]
            why = f"{case['name']}: {case['why']}"
            assert got.keep == want["keep"], why
            assert got.reason == want["reason"], why
            assert got.suspect == want["suspect"], why

    def test_sampling_names_the_old_law(self, fx: dict[str, Any]) -> None:
        # The pre-amendment behaviour stays nameable — a checkpoint measured
        # under it can declare what it measured — and this is what it did:
        # cut at the pause and delete everything behind it.
        case = fx["repetition_silence"]["cases"][0]
        got = repetition_cut(
            build(case["shape"]),
            silence=fx["silence_token_ids"],
            config=self._config(fx, repetition_silence="sampling"),
        )
        assert got == 31, "the old law cut one token past the pause's start"

    def test_without_censuses_the_union_is_the_sampler_list(self, fx: dict[str, Any]) -> None:
        # A checkpoint packed before the censuses changes nothing: nothing on
        # such a build knows id 12 is silent, so the run still reads as a
        # loop there, under either field value.
        case = fx["repetition_silence"]["cases"][0]
        got = repetition_cut(
            build(case["shape"]), silence=fx["silence_token_ids"], config=config_from(fx)
        )
        assert got == 31

    def test_an_unknown_family_is_refused(self) -> None:
        with pytest.raises(ValueError, match="repetition_silence"):
            PostprocessConfig(repetition_silence="both")


class TestRepetitionResume:
    """A loop the decoder resumed from is condemned, never cut.

    The census fix (``repetition_silence``) needs a manifest that names the
    silent ids, and the published pack has none: on it the en0023 pause fired
    again as a period-1 loop and the cut kept 52 of 206 tokens, deleting two
    sentences of correctly-read speech, verdict ``repetition``, no retry.
    The guard here needs no silence knowledge at all: a genuine lock-up runs
    its cycle to the end of the row (a ceiling truncates at most one
    incomplete copy, ``period - 1`` tokens), so a qualifying loop followed by
    a full period or more of other content is a decoder that resumed — and a
    decoder that resumed was never locked. Such a row is handed back whole,
    ``suspect``, into the retry ladder. The section configures no censuses on
    purpose: it is the arm ``repetition_silence`` cannot reach.
    """

    def test_the_rule_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        # The bare rule still reports the loop; the law lives in the resolver.
        cases = fx["repetition_resume"]["cases"]
        assert cases, "the fixture has no repetition_resume cases; nothing was compared"
        for case in cases:
            got = repetition_cut(
                build(case["shape"]), silence=fx["silence_token_ids"], config=config_from(fx)
            )
            assert got == case["loop"], f"{case['name']}: {case['why']}"

    def test_the_resolver_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        for case in fx["repetition_resume"]["cases"]:
            got = inspect(
                build(case["shape"]),
                text_token_count=case["text_tokens"],
                min_tokens=case["min_tokens"],
                eos_peak_at=case["eos_peak_at"],
                eos_peak_prob=case["eos_peak_prob"],
                ended=case["ended"],
                is_terminal=case["is_terminal"],
                hit_ceiling=case["hit_ceiling"],
                silence=fx["silence_token_ids"],
                config=config_from(fx),
            )
            want = case["expect"]
            why = f"{case['name']}: {case['why']}"
            assert got.keep == want["keep"], why
            assert got.reason == want["reason"], why
            assert got.suspect == want["suspect"], why

    def test_the_condemned_row_is_handed_back_whole(self, fx: dict[str, Any]) -> None:
        # Unlike a starved desperation cut, there is no trim worth keeping as
        # a fallback: the trim is the defect, so an exhausted retry ladder
        # ships the full row rather than the cut.
        case = fx["repetition_resume"]["cases"][0]
        tokens = build(case["shape"])
        got = inspect(
            tokens,
            text_token_count=case["text_tokens"],
            min_tokens=case["min_tokens"],
            eos_peak_at=case["eos_peak_at"],
            eos_peak_prob=case["eos_peak_prob"],
            ended=case["ended"],
            is_terminal=case["is_terminal"],
            hit_ceiling=case["hit_ceiling"],
            silence=fx["silence_token_ids"],
            config=config_from(fx),
        )
        assert got.keep == len(tokens), (
            "a resumed loop must be handed back whole; the cut would delete "
            "what the decoder came back to say"
        )
        assert got.suspect, "the caller has to be told, since nothing was changed"

    def test_cut_names_the_old_law(self, fx: dict[str, Any]) -> None:
        # The pre-amendment behaviour stays nameable — a checkpoint measured
        # under it can declare what it measured — and this is what it did:
        # cut at the pause and delete everything behind it.
        case = fx["repetition_resume"]["cases"][0]
        got = inspect(
            build(case["shape"]),
            text_token_count=case["text_tokens"],
            min_tokens=case["min_tokens"],
            eos_peak_at=case["eos_peak_at"],
            eos_peak_prob=case["eos_peak_prob"],
            ended=case["ended"],
            is_terminal=case["is_terminal"],
            hit_ceiling=case["hit_ceiling"],
            silence=fx["silence_token_ids"],
            config=config_from(fx, repetition_resume="cut"),
        )
        assert got.reason == "repetition"
        assert got.keep == case["loop"], "the old law shipped the specimen's cut"
        assert not got.suspect

    def test_the_census_arm_is_untouched(self, fx: dict[str, Any]) -> None:
        # With the censuses configured the specimen's pause is exempt from the
        # loop rule entirely and `stall` condemns it — the repetition_silence
        # contract, byte for byte, guard or no guard.
        section = fx["repetition_silence"]
        case = section["cases"][0]
        got = inspect(
            build(case["shape"]),
            text_token_count=case["text_tokens"],
            min_tokens=case["min_tokens"],
            eos_peak_at=case["eos_peak_at"],
            eos_peak_prob=case["eos_peak_prob"],
            ended=case["ended"],
            is_terminal=case["is_terminal"],
            hit_ceiling=case["hit_ceiling"],
            silence=fx["silence_token_ids"],
            config=config_from(
                fx,
                silence_render_ids=tuple(section["silence_render_ids"]),
                quiet_render_ids=tuple(section["quiet_render_ids"]),
            ),
        )
        assert got.reason == "stall"
        assert got.keep == case["expect"]["keep"]

    def test_an_unknown_law_is_refused(self) -> None:
        with pytest.raises(ValueError, match="repetition_resume"):
            PostprocessConfig(repetition_resume="maybe")


class TestTrailingFiller:
    def test_cases(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        sil = fx["silence_token_ids"]
        for case in fx["trailing_filler"]:
            tokens = build(case["shape"])
            got = is_trailing_filler(tokens, case["from"], silence=sil, config=cfg)
            assert got == case["expect"], f"{case['name']}: {case['why']}"


class TestDesperation:
    def test_cases(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        sil = fx["silence_token_ids"]
        for case in fx["desperation"]:
            got = desperation_cut(
                build(case["shape"]),
                text_token_count=case["text_tokens"],
                min_tokens=case["min_tokens"],
                eos_peak_at=case["eos_peak_at"],
                silence=sil,
                config=cfg,
                peak_allowed=case["peak_allowed"],
            )
            assert got == case["expect"], f"{case['name']}: {case['why']}"


class TestEndedTail:
    def test_cases(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        sil = fx["silence_token_ids"]
        for case in fx["ended_tail"]:
            got = ended_tail_trim(
                build(case["shape"]),
                silence=sil,
                config=cfg,
                is_terminal=case["is_terminal"],
            )
            assert got == case["expect"], f"{case['name']}: {case['why']}"


class TestTerminalEcho:
    def test_cases(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        for case in fx["terminal_echo"]:
            got = terminal_echo_cut(
                token_count=case["token_count"],
                eos_peak_at=case["eos_peak_at"],
                eos_peak_prob=case["eos_peak_prob"],
                min_tokens=case["min_tokens"],
                is_terminal=case["is_terminal"],
                hit_ceiling=case["hit_ceiling"],
                config=cfg,
            )
            assert got == case["expect"], f"{case['name']}: {case['why']}"


class TestResolve:
    """The precedence, which is the part a caller cannot get right by itself."""

    def test_cases(self, fx: dict[str, Any]) -> None:
        sil = fx["silence_token_ids"]
        for case in fx["resolve"]:
            cfg = config_from(fx, **({"mode": case["mode"]} if "mode" in case else {}))
            tokens = build(case["shape"])
            got = inspect(
                tokens,
                text_token_count=case["text_tokens"],
                min_tokens=case["min_tokens"],
                eos_peak_at=case["eos_peak_at"],
                eos_peak_prob=case["eos_peak_prob"],
                ended=case["ended"],
                is_terminal=case["is_terminal"],
                hit_ceiling=case["hit_ceiling"],
                silence=sil,
                config=cfg,
            )
            want = case["expect"]
            why = f"{case['name']}: {case['why']}"
            assert got.keep == want["keep"], why
            assert got.reason == want["reason"], why
            assert got.suspect == want["suspect"], why


class TestConfigRefusesNonsense:
    def test_desperation_must_exceed_the_ceiling(self) -> None:
        # Below the ceiling, "certainly broken" would fire on every row the
        # ceiling stopped correctly.
        with pytest.raises(ValueError, match="must exceed"):
            PostprocessConfig(
                ceiling_speech_per_text_token=4.5, desperation_speech_per_text_token=4.0
            )

    def test_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="unknown postprocess mode"):
            PostprocessConfig(mode="quiet")


class TestSamplerObservesTheEosPeak:
    """The peak is read off the sampler, so it must not disturb the sampler."""

    @staticmethod
    def _logits(vocab: int, stop: int, stop_score: float) -> np.ndarray:
        logits = np.zeros(vocab, dtype=np.float32)
        logits[5] = 4.0
        logits[stop] = stop_score
        return logits

    def test_observation_does_not_change_the_draw(self) -> None:
        cfg = SamplingConfig(silence_token_ids=(0, 1))
        vocab, stop = 32, 31
        rows = [self._logits(vocab, stop, s) for s in (-2.0, 1.0, 3.0, 0.5)]

        plain = LRSamplerV1(cfg, seed=7)
        watched = LRSamplerV1(cfg, seed=7, stop_token=stop, eos_floor=0)
        seen_a = np.zeros(vocab, dtype=bool)
        seen_b = np.zeros(vocab, dtype=bool)
        for step, row in enumerate(rows):
            a = plain(row.copy(), step=step, seen=seen_a)
            b = watched(row.copy(), step=step, seen=seen_b)
            assert a == b, "watching the stop token changed which token was drawn"
            seen_a[a] = True
            seen_b[b] = True

    def test_peak_is_the_highest_and_its_step(self) -> None:
        cfg = SamplingConfig(min_p=0.0, silence_token_ids=())
        vocab, stop = 32, 31
        sampler = LRSamplerV1(cfg, seed=7, stop_token=stop, eos_floor=0)
        seen = np.zeros(vocab, dtype=bool)
        # step 0 is at the floor and is not recorded; step 2 is the highest of
        # the rest.
        for step, score in enumerate((9.0, 1.0, 3.0, 0.5)):
            sampler(self._logits(vocab, stop, score), step=step, seen=seen)
        at, prob = sampler.eos_peak
        assert at == 2, "the step of the highest stop probability"
        assert 0.0 < prob < 1.0

    def test_floor_is_exclusive(self) -> None:
        cfg = SamplingConfig(min_p=0.0, silence_token_ids=())
        vocab, stop = 32, 31
        sampler = LRSamplerV1(cfg, seed=7, stop_token=stop, eos_floor=2)
        seen = np.zeros(vocab, dtype=bool)
        # Every step offers the same stop score, so the recorded one is simply
        # the first that qualifies: steps 0..2 are at or below the floor.
        for step in range(4):
            sampler(self._logits(vocab, stop, 5.0), step=step, seen=seen)
        at, _ = sampler.eos_peak
        assert at == 3, "recorded from the step after the floor, as the reader does"

    def test_masked_stop_reports_zero(self) -> None:
        # Below the floor the generator masks the stop token to -inf. The
        # observation must then describe the model, not the mask.
        cfg = SamplingConfig(min_p=0.0, silence_token_ids=())
        vocab, stop = 32, 31
        sampler = LRSamplerV1(cfg, seed=7, stop_token=stop, eos_floor=0)
        seen = np.zeros(vocab, dtype=bool)
        logits = self._logits(vocab, stop, 0.0)
        logits[stop] = -np.inf
        sampler(logits, step=1, seen=seen)
        assert sampler.eos_peak == (-1, 0.0)

    def test_without_a_stop_token_nothing_is_observed(self) -> None:
        sampler = LRSamplerV1(SamplingConfig(), seed=7)
        seen = np.zeros(32, dtype=bool)
        sampler(self._logits(32, 31, 5.0), step=1, seen=seen)
        assert sampler.eos_peak == (-1, 0.0)

    def test_against_the_shared_fixture(self) -> None:
        # The value is hand-written in five languages and it is *audible*: two
        # detector rules compare it against a threshold, so a port that computes
        # it differently cuts a chunk somewhere else.
        from loudkit.rng import uniforms

        section = json.loads((FIXTURE.parent / "vectors.json").read_text(encoding="utf-8"))[
            "eos_peak"
        ]
        assert section["cases"], "the fixture has no eos_peak cases; nothing was compared"
        for case in section["cases"]:
            cfg = SamplingConfig(
                temperature=case["config"]["temperature"],
                repetition_penalty=case["config"]["repetition_penalty"],
                min_p=case["config"]["min_p"],
                silence_token_ids=tuple(case["config"]["silence_token_ids"]),
            )
            r = case["logits_recipe"]
            sampler = LRSamplerV1(
                cfg,
                seed=case["seed"],
                stop_token=case["stop_token"],
                eos_floor=case["eos_floor"],
            )
            seen = np.zeros(r["vocab"], dtype=bool)
            for step in range(r["steps"]):
                u = uniforms(r["seed"], r["stream"], step, 1, r["vocab"])[0]
                row = (u * r["scale"] + r["offset"]).astype(np.float32)
                seen[sampler(row, step=step, seen=seen)] = True
            at, prob = sampler.eos_peak
            assert at == case["expected_at"], case["name"]
            assert abs(prob - case["expected_prob"]) <= section["prob_rtol"] * abs(
                case["expected_prob"]
            ), case["name"]


class TestRecipeAndManifest:
    def test_a_manifest_that_says_nothing_gets_the_shipping_defaults(self) -> None:
        # There is one recipe, so a manifest that does not mention the
        # detectors is not an older pack. It is a manifest that left a shipping
        # default unstated, and the default is what ships.
        cfg = AlgorithmConfig.from_manifest({"recipe_version": "loudkit-1"})
        assert cfg.postprocess.mode == "trim"

    def test_an_explicit_block_wins_over_the_recipe(self) -> None:
        cfg = AlgorithmConfig.from_manifest(
            {"recipe_version": "loudkit-1", "postprocess": {"mode": "report"}}
        )
        assert cfg.postprocess.mode == "report"

    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown postprocess mode"):
            AlgorithmConfig.from_manifest({"postprocess": {"mode": "shave"}})

    def test_postprocess_is_hashed(self) -> None:
        # The whole reason the constants are config: a port that quietly uses a
        # different number must not agree on the fingerprint.
        base = AlgorithmConfig()
        moved = base.with_(postprocess=PostprocessConfig(trailing_silence_run_tokens=13))
        assert base.fingerprint() != moved.fingerprint()
        assert "postprocess" in base.canonical_form()

    def test_render_censuses_are_read_from_the_top_level(self) -> None:
        # Properties of the weights live beside silence_token_ids, where the
        # next backend cannot re-guess them.
        cfg = AlgorithmConfig.from_manifest(
            {
                "recipe_version": "loudkit-1",
                "silence_render_ids": [4137, 4218],
                "quiet_render_ids": [1458, 1461],
            }
        )
        assert cfg.postprocess.silence_render_ids == (4137, 4218)
        assert cfg.postprocess.quiet_render_ids == (1458, 1461)

    def test_absent_censuses_default_to_empty(self) -> None:
        # The fallback contract: an old pack loads, and the stall rule keys
        # its run trigger to silence_token_ids.
        cfg = AlgorithmConfig.from_manifest({"recipe_version": "loudkit-1"})
        assert cfg.postprocess.silence_render_ids == ()
        assert cfg.postprocess.quiet_render_ids == ()

    def test_a_census_in_the_postprocess_block_is_refused(self) -> None:
        # One value, one home: the censuses are top-level manifest fields.
        with pytest.raises(ValueError, match="top level"):
            AlgorithmConfig.from_manifest({"postprocess": {"silence_render_ids": [1, 2]}})

    def test_the_censuses_are_hashed(self) -> None:
        # The stall detector reads them, so they are audible values: two
        # engines disagreeing on the census must not share a fingerprint.
        base = AlgorithmConfig()
        moved = base.with_(postprocess=PostprocessConfig(silence_render_ids=(4137,)))
        assert base.fingerprint() != moved.fingerprint()


class TestDropout:
    """Early truncation — the failure a listener cannot hear.

    Every other rule here says the end of the row is wrong. This one says the
    row is incomplete, which is why it reports rather than cuts: there is
    nothing to remove, and the missing content cannot be recovered.
    """

    def test_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        cases = fx["dropout"]["cases"]
        assert cases, "the fixture has no dropout cases; nothing was compared"
        for case in cases:
            got = inspect(
                list(range(20, 20 + case["tokens"])),
                text_token_count=case["text_tokens"],
                min_tokens=10,
                eos_peak_at=-1,
                eos_peak_prob=0.0,
                ended=True,
                is_terminal=True,
                hit_ceiling=False,
                silence=set(fx["silence_token_ids"]),
                config=cfg,
            )
            fired = got.reason == "dropout"
            assert fired == case["expect"], f"{case['name']}: {case['why']}"

    def test_it_never_cuts(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        got = inspect(
            list(range(20, 28)),
            text_token_count=40,
            min_tokens=10,
            eos_peak_at=-1,
            eos_peak_prob=0.0,
            ended=True,
            is_terminal=True,
            hit_ceiling=False,
            silence=set(fx["silence_token_ids"]),
            config=cfg,
        )
        assert got.reason == "dropout"
        assert got.keep == 8, (
            "a truncated row must be handed back whole; there is nothing to trim"
        )
        assert got.suspect, "the caller has to be told, since nothing was changed"


class TestPacing:
    """Long-form drift, report-only, in the same integer-derived domain."""

    def test_matches_the_fixture(self, fx: dict[str, Any]) -> None:
        cfg = config_from(fx)
        cases = fx["pacing"]["cases"]
        assert cases, "the fixture has no pacing cases; nothing was compared"
        for case in cases:
            got = pacing_outliers(case["ratios"], config=cfg)
            assert got == case["expect"], f"{case['name']}: {case['why']}"


class TestTwoSeamIsNotTrailingFiller:
    """Substantial speech between two qualifying seams disarms the rule.

    [seam][80 real tokens][seam][one word] used to read as trailing filler of
    the FIRST seam — exactly the mislabeled-language row the rule exists to
    protect, cut at 80 real tokens.
    """

    def test_substantial_speech_between_seams_disarms(self) -> None:
        from loudkit.postprocess import is_trailing_filler

        cfg = PostprocessConfig()
        silence = {2, 3, 4}
        tokens = (
            list(silence) * (cfg.trailing_silence_run_tokens // len(silence))
            + [10] * 80
            + list(silence) * (cfg.trailing_silence_run_tokens // len(silence))
            + [11]
        )
        # Peak lands mid-sentence (index 40 of the 80-token block): the old
        # semantics read the tail as trailing filler of the FIRST seam and cut
        # all 80 real tokens. Substantial speech since the last qualifying run
        # disarms the rule.
        assert not is_trailing_filler(tokens, 60, silence=silence, config=cfg)

    def test_single_seam_still_cuts_the_filler(self) -> None:
        from loudkit.postprocess import is_trailing_filler

        cfg = PostprocessConfig()
        silence = {2, 3, 4}
        tokens = (
            [10] * 20 + list(silence) * (cfg.trailing_silence_run_tokens // len(silence)) + [11]
        )
        # The peak sits on the last real token; everything after it is the
        # proposed discard.
        assert is_trailing_filler(tokens, 19, silence=silence, config=cfg)
