"""The CLDR differential: 1300 spellouts we did not write.

Our own fixture is a hundred hand-written cases. This corpus is Unicode's —
maintained for decades, by people who were not looking at our code — and it
found eleven real defects in the first run: an English "minus" shipped in nine
languages, scale words glued solid in four, Spanish hundreds that did not
inflect, and a Polish trailing *jeden* that inflected when it must not.

Four row classes are excluded, each *declared* rather than silent:

- **unmapped rulesets** (ordinals, oblique cases, years) — no implementation
  claims them; they are listed in the fixture's ``not_covered``.
- **past-scale values** — our grammars refuse past their largest scale on
  purpose; the refusal is asserted, not skipped.
- **disputed rows** — places where CLDR contradicts itself. Danish writes
  ``tusinde``/``million`` in its common ruleset and ``tusind``/``en million``
  in its neuter one, so the whole Danish >=1000 zone is contested and our
  grammar keeps the hand-written reference forms until a native adjudicates.
  Each row carries its reason; the count is pinned so a regenerated corpus
  cannot quietly move the line.
- **pending languages** — none today. fi/no/sv activated with their grammars;
  the mechanism stays for the next language.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from loudkit.frontend.numbers import NumberGrammarError, cardinal, supported_languages

FIXTURE = Path(__file__).parent / "data" / "conformance" / "numbers_cldr.json"

# Grows only when a grammar lands and its cases activate; shrinks only when a
# dispute is resolved. Either move is a review event, which is the point.
EXPECTED_DISPUTED = 75
PENDING_LANGUAGES = frozenset()


@pytest.fixture(scope="module")
def fx() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_every_mapped_case_agrees(fx: dict[str, Any]) -> None:
    checked = disputed = past_scale = 0
    failures: list[str] = []
    for language, cases in fx["cases"].items():
        if language in PENDING_LANGUAGES:
            continue
        for case in cases:
            if case.get("disputed"):
                disputed += 1
                continue
            try:
                got = cardinal(case["value"], language, gender=case["gender"])
            except NumberGrammarError:
                past_scale += 1
                continue
            checked += 1
            if got != case["expect"]:
                failures.append(
                    f"{language} {case['value']} g={case['gender']}: "
                    f"ours {got!r} != cldr {case['expect']!r}"
                )
    assert checked > 900, f"only {checked} cases ran; the corpus went missing"
    assert not failures, "\n".join(failures[:20])
    assert disputed == EXPECTED_DISPUTED, (
        f"{disputed} disputed rows, expected {EXPECTED_DISPUTED} — resolving or "
        "adding a dispute is a review event, not a drive-by"
    )


def test_pending_languages_are_actually_pending(fx: dict[str, Any]) -> None:
    # The day a Nordic grammar lands, this fails, and the fix is to move the
    # language out of PENDING_LANGUAGES — activating its 242 waiting cases.
    for language in PENDING_LANGUAGES:
        assert language in fx["cases"], f"{language} lost its imported corpus"
        assert language not in supported_languages(), (
            f"{language} gained a grammar; activate its CLDR cases by removing "
            "it from PENDING_LANGUAGES"
        )


def test_past_scale_rows_actually_refuse(fx: dict[str, Any]) -> None:
    # "Past our scale" must mean a loud refusal, not a silent wrong answer.
    for language, cases in fx["cases"].items():
        if language in PENDING_LANGUAGES:
            continue
        for case in cases:
            if case["value"] >= 10**15 and not case.get("disputed"):
                with pytest.raises(NumberGrammarError):
                    cardinal(case["value"], language, gender=case["gender"])
