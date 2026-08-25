"""Negative controls for the conformance fixture: a gate that cannot fail.

Every conformance test in this repository, and in the four ports, walks a list
out of ``vectors.json``:

    for case in fixture["sampler"]["cases"]:
        ...

An empty list makes that loop pass without comparing anything, and a pass is
what a reader takes as proof. The end-to-end test is more explicit about it --
``cases = fixture.get("end_to_end"); if not cases: return`` -- which turns a
truncated fixture into a green build.

Nothing here checks behaviour. These are the checks that keep the other checks
honest: the fixture has the sections it claims, each section has at least the
cases it had when it was written, and the comparison the suites perform does
notice a changed value.

The counts are lower bounds. Adding a case is normal and must not fail; losing
one is the accident this guards.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).resolve().parent / "data/conformance/vectors.json"

# Section -> the number of cases present when this control was written.
EXPECTED_AT_LEAST = {
    ("philox", "kat"): 3,
    ("philox", "uniform_bits"): 4,
    ("philox", "gumbel"): 2,
    ("sampler", "cases"): 3,
    ("eos_peak", "cases"): 3,
    ("frontend", "cases"): 5,
    ("end_to_end",): 2,
    ("long_form", "cases"): 1,
}


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _at(data: dict, path: tuple[str, ...]) -> object:
    for key in path:
        data = data[key]  # type: ignore[assignment]
    return data


@pytest.mark.parametrize(("path", "least"), sorted(EXPECTED_AT_LEAST.items()))
def test_every_section_still_carries_its_cases(
    fixture: dict, path: tuple[str, ...], least: int
) -> None:
    section = _at(fixture, path)
    assert isinstance(section, list), f"{'.'.join(path)} is not a list of cases"
    assert len(section) >= least, (
        f"{'.'.join(path)} has {len(section)} cases, down from {least}. "
        "A section that loses cases makes every suite that walks it quieter "
        "without making any of them fail."
    )


def test_the_algorithm_section_pins_a_fingerprint(fixture: dict) -> None:
    algo = fixture["algorithm"]
    assert algo["fingerprint"], "the fingerprint is empty, so every port's check is vacuous"
    assert algo["canonical_form"], "the canonical form is empty"


def test_the_seed_streams_are_named(fixture: dict) -> None:
    streams = fixture["seeds"]["streams"]
    assert isinstance(streams, dict), "the seed streams are not a mapping"
    assert streams, "no seed streams are pinned"


def test_a_changed_fixture_value_is_visible_to_the_comparison(fixture: dict) -> None:
    """The control proper.

    Philox is chosen because it needs no checkpoint and no graphs, so this runs
    in every job rather than only the asset-backed one. The real suite asserts
    that the computed counter equals ``case["expected"]``. Here the expectation
    is perturbed by one bit and the same assertion must now fail. If it does
    not, the suite is comparing something other than what it claims to.
    """
    import numpy as np

    from loudkit.rng import philox_4x32_10

    case = copy.deepcopy(fixture["philox"]["kat"][0])
    got = [
        int(g[0])
        for g in philox_4x32_10(
            *(np.array([c], dtype=np.uint64) for c in case["counter"]),
            case["key"][0],
            case["key"][1],
        )
    ]

    assert got == list(case["expected"]), "the positive direction must hold first"

    case["expected"][0] ^= 1
    assert got != list(case["expected"]), (
        "a one-bit change in the pinned value did not change the comparison, "
        "so the comparison is not reading the pinned value"
    )


def test_the_fixture_declares_the_version_the_ports_check(fixture: dict) -> None:
    assert fixture["version"], "an empty version lets a stale fixture pass as current"
    assert fixture["generated_by"], "nothing records what produced these vectors"
