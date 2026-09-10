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
POSTPROCESS = Path(__file__).resolve().parent / "data/conformance/postprocess.json"

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
    # Both of these were walked by the ports and floored by nothing. `resplit`
    # is the whole `cap_resplit` law — the one whose own port comments record
    # that it broke three ways while it was being written — and it holds a
    # single case, so emptying it left all five suites green.
    ("seeds", "derivation"): 6,
    ("resplit", "cases"): 1,
    # The Euler grid the flow ODE integrates on. Unfloored until the control
    # above learned to recurse, and emptying it left the weight-free suites of
    # Python, Rust and Go green: only two tests read it and both are
    # checkpoint-gated, so the grid was pinned nowhere a fork would notice.
    ("algorithm", "euler_grid"): 3,
}

# The second shared fixture. It had no control at all: six of its sections are
# bare lists walked with an unguarded `for` in Python, Rust, Go and JS (Swift
# is the exception and guards every one), so emptying them reported 50 passes
# in Python — including every test of the precedence resolver, which is the
# layer that file exists for.
POSTPROCESS_AT_LEAST = {
    ("ceiling",): 3,
    ("trailing_filler",): 6,
    ("desperation",): 7,
    ("ended_tail",): 8,
    ("terminal_echo",): 5,
    ("repetition",): 9,
    ("resolve",): 8,
    ("silence_token_ids",): 8,
    ("dropout", "cases"): 6,
    ("pacing", "cases"): 5,
    ("stall", "cases"): 8,
    # The render censuses the two rules read. Not "cases", so the shallow
    # control never saw them; emptying any of the four turns five Python tests
    # red, which is what makes them worth a floor rather than a shrug.
    ("stall", "silence_render_ids"): 4,
    ("stall", "quiet_render_ids"): 4,
    ("repetition_silence", "silence_render_ids"): 5,
    ("repetition_silence", "quiet_render_ids"): 5,
    ("starved_rescue", "cases"): 4,
    ("repetition_silence", "cases"): 5,
    ("repetition_resume", "cases"): 6,
    ("language_guard", "cases"): 27,
}


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def postprocess_fixture() -> dict:
    return json.loads(POSTPROCESS.read_text(encoding="utf-8"))


def _at(data: dict, path: tuple[str, ...]) -> object:
    for key in path:
        data = data[key]
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


@pytest.mark.parametrize(("path", "least"), sorted(POSTPROCESS_AT_LEAST.items()))
def test_every_postprocess_section_still_carries_its_cases(
    postprocess_fixture: dict, path: tuple[str, ...], least: int
) -> None:
    section = _at(postprocess_fixture, path)
    assert isinstance(section, list), f"{'.'.join(path)} is not a list of cases"
    assert len(section) >= least, (
        f"postprocess.json {'.'.join(path)} has {len(section)} cases, down from "
        f"{least}. Four of the five ports walk this with an unguarded `for`."
    )


@pytest.mark.parametrize(
    ("name", "path", "floors"),
    [
        ("vectors.json", FIXTURE, EXPECTED_AT_LEAST),
        ("postprocess.json", POSTPROCESS, POSTPROCESS_AT_LEAST),
    ],
)
def test_no_section_carries_cases_without_a_floor(
    name: str, path: Path, floors: dict[tuple[str, ...], int]
) -> None:
    """The control inverted, which is the half that was missing.

    A table of floors only guards the sections somebody remembered to list.
    `resplit` landed in `vectors.json` without an entry and nothing noticed;
    the plan for the next release adds a `fusion` section, which would land in
    the same hole. So walk the fixture instead and refuse a list of cases that
    no floor covers — a new section is then a red build until it is floored,
    which is one line and the right kind of friction.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    covered = set(floors)
    unfloored = []

    def walk(node: object, path_so_far: tuple[str, ...]) -> None:
        """Every list of cases, at any depth.

        This used to look one level down and one key deep — a top-level list,
        or a `<section>.cases`. Five of this file's own eleven floors are
        neither shape (`philox.kat`, `philox.uniform_bits`, `philox.gumbel`,
        `seeds.derivation`), so the control could not see the very sections it
        was written to cover, and the `fusion` block it names as the next
        arrival lands in exactly that hole. `algorithm.euler_grid` was
        unfloored the whole time for the same reason: emptying it left the
        weight-free suites of Python, Rust and Go green.
        """
        if isinstance(node, list):
            looks_like_cases = (
                bool(node)
                and isinstance(node[0], (dict, list, int, float))
                and bool(path_so_far)
            )
            if looks_like_cases and path_so_far not in covered:
                unfloored.append(".".join(path_so_far))
            return
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, (*path_so_far, key))

    walk(data, ())
    assert not unfloored, (
        f"{name} carries case lists with no floor in this file: {sorted(unfloored)}. "
        f"Add each to the table above with the count it has today."
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
