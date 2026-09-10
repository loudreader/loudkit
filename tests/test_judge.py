"""The pairwise judge's arithmetic, pinned.

Everything here is the part of Tier 2.5 that runs without a network: which
system a verdict names, how a truncated reply is recovered, how the interval is
resampled, and which paragraphs the reading set will accept. The judgment
itself belongs to a model and cannot be tested; the bookkeeping around it
decides whether a published number means what the caption says, and that can.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import tool

RESEARCH = Path(__file__).resolve().parent.parent / "research"


judge = tool("judge_pairwise")
report = tool("judge_report")
reading = tool("build_reading_set")


class TestVerdictExtraction:
    def test_plain_object(self) -> None:
        assert judge.extract_json('{"prosody": 1, "correctness": 2, "note": "x"}') == {
            "prosody": 1,
            "correctness": 2,
            "note": "x",
        }

    def test_fenced_object(self) -> None:
        verdict = judge.extract_json('```json\n{"prosody": 0, "correctness": 1}\n```')
        assert verdict == {"prosody": 0, "correctness": 1}

    def test_truncated_reply_keeps_the_verdict(self) -> None:
        # A thinking judge can spend its budget before closing the note. The
        # numbers are the verdict; the note is decoration.
        verdict = judge.extract_json('```json\n{"prosody": 2, "correctness": 0, "note": "Both')
        assert judge.valid_verdict(verdict)
        assert verdict["prosody"] == 2
        assert verdict["correctness"] == 0

    def test_prose_without_a_verdict_is_rejected(self) -> None:
        assert judge.extract_json("I could not decide between them.") is None

    def test_out_of_range_choice_is_invalid(self) -> None:
        assert not judge.valid_verdict({"prosody": 3, "correctness": 1})
        assert not judge.valid_verdict({"prosody": 1})
        assert not judge.valid_verdict(None)


class TestScoring:
    def test_a_win_tie_and_loss(self) -> None:
        assert report.score("loudkit", "loudkit") == 1.0
        assert report.score("tie", "loudkit") == 0.5
        assert report.score("kokoro", "loudkit") == 0.0

    def test_parity_scores_fifty(self) -> None:
        clusters = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        flat = [v for c in clusters for v in c]
        assert sum(flat) / len(flat) == pytest.approx(0.5)


class TestBootstrap:
    def test_unanimous_input_has_no_spread(self) -> None:
        low, high = report.bootstrap([[1.0], [1.0], [1.0]])
        assert low == high == 1.0

    def test_interval_brackets_the_mean(self) -> None:
        clusters = [[1.0, 1.0]] * 30 + [[0.0, 0.0]] * 10
        low, high = report.bootstrap(clusters)
        assert low < 0.75 < high

    def test_deterministic_for_one_input(self) -> None:
        clusters = [[1.0, 0.0], [1.0, 1.0], [0.0, 0.5]] * 10
        assert report.bootstrap(clusters) == report.bootstrap(clusters)

    def test_clustering_widens_the_interval(self) -> None:
        # Both orders of a passage agreeing is one observation, not two.
        # Treating them as two reports a narrower interval than the data earns,
        # which is exactly the mistake this harness must not make.
        clustered = [[1.0, 1.0]] * 20 + [[0.0, 0.0]] * 20
        split = [[value] for cluster in clustered for value in cluster]
        c_low, c_high = report.bootstrap(clustered)
        s_low, s_high = report.bootstrap(split)
        assert (c_high - c_low) > (s_high - s_low)


class TestSummary:
    def _records(self, winners: list[tuple[str, str, str]]) -> dict[str, list[dict]]:
        by_passage: dict[str, list[dict]] = {}
        for index, (order, prosody, correctness) in enumerate(winners):
            passage = f"p{index // 2:03d}"
            by_passage.setdefault(passage, []).append(
                {
                    "id": passage,
                    "order": order,
                    "first": "loudkit" if order == "ab" else "rival",
                    "prosody_winner": prosody,
                    "correctness_winner": correctness,
                }
            )
        return by_passage

    def test_a_clean_sweep_scores_one(self) -> None:
        summary = report.summarise(
            self._records([("ab", "loudkit", "loudkit"), ("ba", "loudkit", "loudkit")]),
            "loudkit",
        )
        assert summary["prosody"]["score"] == 1.0
        assert summary["prosody"]["order_consistency"] == 1.0
        assert summary["prosody"]["tie_rate"] == 0.0

    def test_orders_disagreeing_is_visible(self) -> None:
        summary = report.summarise(
            self._records([("ab", "loudkit", "tie"), ("ba", "rival", "tie")]),
            "loudkit",
        )
        assert summary["prosody"]["score"] == 0.5
        assert summary["prosody"]["order_consistency"] == 0.0
        assert summary["correctness"]["tie_rate"] == 1.0

    def test_a_passage_judged_once_is_not_an_agreement(self) -> None:
        """A lost call must not read as confidence.

        A passage judged in one order has one winner by arithmetic, and
        counting it says the two orders agreed when only one of them ran. Five
        balanced passages disagreeing every time, beside five whose second call
        failed, reported 0.5 where the truth over the balanced ones is 0.0, in
        the number the report tells a reader to trust the other three by.
        """
        by_passage = self._records([("ab", "loudkit", "loudkit"), ("ba", "rival", "rival")] * 5)
        for index in range(5, 10):
            by_passage[f"p{index:03d}"] = [
                {
                    "id": f"p{index:03d}",
                    "order": "ab",
                    "first": "loudkit",
                    "prosody_winner": "loudkit",
                    "correctness_winner": "loudkit",
                }
            ]
        summary = report.summarise(by_passage, "loudkit")
        assert summary["passages"] == 10
        assert summary["order_balanced_passages"] == 5
        assert summary["prosody"]["order_consistency"] == 0.0

    def test_position_bias_catches_a_first_slot_favourite(self) -> None:
        # Whoever went first won every time: the judge is scoring the slot.
        summary = report.summarise(
            self._records([("ab", "loudkit", "loudkit"), ("ba", "rival", "rival")]),
            "loudkit",
        )
        assert summary["position_bias"] == 1.0
        assert summary["prosody"]["score"] == 0.5

    def test_a_missing_order_is_counted(self) -> None:
        summary = report.summarise(self._records([("ab", "loudkit", "loudkit")]), "loudkit")
        assert summary["passages"] == 1
        assert summary["order_balanced_passages"] == 0


class TestReadingSetFilters:
    LONG = (
        "The road wound upward between the hedges until it reached the crest, "
        "and from there the whole valley lay open beneath a sky that had not "
        "decided whether to clear or to close again, while the wind came across "
        "the stubble in long slow waves that no one was there to watch."
    )

    def test_accepts_plain_prose(self) -> None:
        assert reading.acceptable(self.LONG)

    def test_rejects_dialogue(self) -> None:
        assert not reading.acceptable(self.LONG.replace("The road", '"The road'))

    def test_rejects_digits(self) -> None:
        assert not reading.acceptable(self.LONG.replace("the crest", "the 3rd crest"))

    def test_rejects_short_and_long(self) -> None:
        assert not reading.acceptable("Too short.")
        assert not reading.acceptable(self.LONG * 3)

    def test_rejects_an_unfinished_paragraph(self) -> None:
        assert not reading.acceptable(self.LONG.rstrip("."))

    def test_crlf_source_parses_like_lf(self) -> None:
        # The mirror serves CRLF and a cached read normalises it away. Before
        # this was handled, a fresh fetch found zero paragraphs and a cached
        # one found hundreds, from the same bytes.
        text = "*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\r\nOne.\r\n\r\nTwo.\r\n"
        assert list(reading.paragraphs(text)) == ["One.", "Two."]
