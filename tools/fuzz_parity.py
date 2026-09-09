"""Random text through five funnels, compared byte for byte.

Every parity break found in four rounds of review was found *outside* the shared
fixture: the Swift combining-mark divergence, the Go/Rust glued-group
divergence, the half-expansion family. The fixture holds exactly the cases
somebody thought of, and passed 105 of them while two of those breaks were live.

This is the part that finds what nobody thought of. It generates text from the
shapes that have actually broken -- digits against letters, grouping spaces,
decimal points, exponents, currency marks, combining characters -- writes a
fixture whose expectations are Python's output, and runs the four ports' own
conformance tests against it. A divergence is a port disagreeing with Python on
a string no human chose.

Deterministic by seed, so a failure is reproducible: the seed is printed on
every run and a failing case is written out ready to paste into
`tools/make_speechtext_fixture.py`.

    python tools/fuzz_parity.py --cases 400 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

# Derived, not written out, for the reason `tests/test_funnel_properties.py`
# gives: this is the one differential gate in CI, and a language missing from a
# hand-kept list gets no fuzz coverage under a green build.
from loudkit.frontend.numbers import supported_languages  # noqa: E402

LANGUAGES = supported_languages()

# Fragments, weighted towards what has broken. A uniform character soup mostly
# generates text with no numbers in it, which exercises the one pass that has
# never diverged.
ATOMS = (
    # digits and the separators that decide what they mean
    "0",
    "1",
    "5",
    "12",
    "200",
    "000",
    "1000",
    "2024",
    "3.14",
    "0.49",
    "2,50",
    "1 000",
    "200 000",
    "1 234 567",
    "14:30",
    "14.30",
    "24:00",
    "12.03.2026",
    "1e6",
    "1e-3",
    "1e+3",
    "2.5E+1",
    "v1.2.3",
    "192.168.0.1",
    # letters that glue to them
    "x",
    "abc",
    "iOS",
    "R2",
    "CIA",
    "zł",
    "kg",
    # marks that change meaning
    "$",
    "€",
    "¢",
    "£",
    "%",
    "-",
    "+",
    "−",
    "(",
    ")",
    "[12]",
    # ...and the marks that used to carry their meaning no further than the
    # funnel. Each of these is a rule whose guard is the whole difficulty: an
    # operator against markup, a magnitude against a unit, a fraction against a
    # date, a numeral against an initialism. The generator's joiners glue them
    # to each other, which is where a guard that reads one character too few
    # shows itself.
    ">",
    "<",
    "<=",
    ">=",
    "!=",
    "==",
    "<p>",
    "</p>",
    '<div class="x">',
    "<!-- note -->",
    "IV",
    "II",
    "XIV",
    "CD",
    "XL",
    "MIX",
    "10:30:45",
    "10:00:45",
    "3:45:00pm",
    "2026-03-04T10:00",
    "2026-03-04",
    "$2.5M",
    "$5bn",
    "$20k",
    "$5 million",
    "$5kg",
    "24/7",
    "1/2",
    "¾",
    "3 April",
    "12 marzo",
    # The day-first date and the telephone number, whose bounds are the same
    # rule the ISO form's are: the joiners glue a letter to either end.
    "25/03/2026",
    "+12345678",
    "+48 123 456 789",
    # scripts and combining characters
    "٣٫١٤",
    "١٢٣",
    "a̬",
    "é",
    # numerals this layer has no reading for: `No` and `Nl`. They are here
    # because they were not, and the family they hid, a superscript against a
    # digit leaving a bare digit in the output, was found by hand rather than
    # by this file. Every one of them glues to a digit through a "" joiner,
    # which is the shape that broke.
    "²",
    "½",
    "③",
    "Ⅳ",
    # ...and ideographic numerals, which are `Lo`. The guard on the fix above:
    # a pass keyed on numeric type rather than on the general category would
    # delete these, and deleting a language's numerals is the failure being
    # fixed rather than a way to fix it.
    "一二三",
    "十",
    # Letter classes the five ports answered differently: a circled letter is
    # `So` and an Other_Alphabetic mark is `Mn`, and Rust and Swift read both as
    # letters where the other three did not. Astral, because Foundation's
    # `CharacterSet` is wrong above the BMP and nothing here reached that far.
    "ⓐ",
    "aͅ",
    "𗀀",
    # Whitespace the five disagreed about: NEL is not `\s` in ECMAScript, and VT
    # was missing from Go's hand-expanded class.
    "",
    "",
    # Digits of other scripts, which used to pass the whole funnel untouched.
    "৩",
    "１",
    "๓",
    # A date glued to a letter, where three boundary spellings disagreed.
    "12.03.2026",
    # ordinary words, so the funnel sees text and not only symbols
    "the",
    "and",
    "koszt",
    "Preis",
    "es",
    "de",
)
JOINERS = (" ", " ", " ", "", ".", ", ", " - ")


def _sentence(rng: random.Random) -> str:
    parts = [rng.choice(ATOMS) for _ in range(rng.randint(1, 7))]
    out = "".join(p + rng.choice(JOINERS) for p in parts).strip()
    return out + rng.choice([".", "!", "?", ""])


def _generate(count: int, seed: int) -> list[dict[str, str]]:
    from loudkit.frontend.speechtext import speech_text

    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        text = _sentence(rng)
        if not text:
            continue
        language = rng.choice(LANGUAGES)
        cases.append(
            {
                "text": text,
                "language": language,
                "expected": speech_text(text, language),
                "why": "generated by tools/fuzz_parity.py",
            }
        )
    return cases


PORTS = {
    "go": [
        "go",
        "test",
        "-count=1",
        "-run",
        "TestFunnelAgainstTheSharedFixture",
        "./speechtext/",
    ],
    "rust": ["cargo", "test", "--quiet", "funnel_matches_the_shared_fixture"],
    # Only the funnel test: `npm test` runs the weight-free vectors too, and
    # those are not what this compares.
    "js": ["node", "--test", "dist/test/speechtext.test.js"],
    "swift": ["swift", "test", "--filter", "SpeechFunnelTests"],
}
CWDS = {"go": REPO / "go", "rust": REPO / "rust", "js": REPO / "js", "swift": REPO}


# Go reports one line per mismatch; the other harnesses print their total.
COUNTERS = {
    "go": r"speechtext_test\.go:\d+:",
    "js": r"actual: (\d+)",
    "rust": r"shared fixture in (\d+)/",
    "swift": r"shared fixture in (\d+)/",
}


def _port_list(value: str) -> list[str]:
    """``--ports`` as names this tool can run, refused by argparse if not.

    An unknown name reached the loop and came out as a ``KeyError`` traceback
    on ``PORTS[port]``, after the cases had been generated and the fixture
    copied. A typed argument refuses it on the command line instead.
    """
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise argparse.ArgumentTypeError("name at least one port")
    unknown = sorted(set(names) - set(PORTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown port(s) {', '.join(unknown)}; choose from {', '.join(sorted(PORTS))}"
        )
    return names


def divergence_count(port: str, output: str) -> int | None:
    """The harness's mismatch count, or None when it reported no such count.

    A failed build or missing runtime is not evidence of a text divergence.
    Print the full harness output so failures outside the comparison can be diagnosed.
    """
    hits = re.findall(COUNTERS[port], output, re.MULTILINE)
    if not hits:
        return None
    return int(hits[0]) if hits[0].isdigit() else len(hits)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ports",
        default="go,rust,js,swift",
        type=_port_list,
        help="comma-separated subset of " + ",".join(sorted(PORTS)),
    )
    args = parser.parse_args()

    print(f"seed {args.seed}, {args.cases} cases")
    cases = _generate(args.cases, args.seed)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        source = REPO / "tests" / "data" / "conformance"
        # The whole directory, unchanged, and then one file replaced. The
        # harnesses read more than the funnel fixture, a partial copy makes
        # them fail on a missing file, which reads exactly like a divergence and
        # is not one.
        for item in source.iterdir():
            if item.is_file():
                (out / item.name).write_bytes(item.read_bytes())
        (out / "speechtext.json").write_text(
            json.dumps(
                {
                    "why": "generated; do not commit",
                    "cases": cases,
                    "disputed": [],
                    "divergent": {"why": "not used by the fuzzer"},
                },
                ensure_ascii=False,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

        # Two names, two meanings, the same way CI spells them: _DIR is the
        # directory the funnel suites read, LOUDKIT_FIXTURE is the vectors.json
        # the weight-free suites read. Pointing both at the directory handed a
        # directory to every reader that opens it as a file.
        env = {
            **os.environ,
            "LOUDKIT_FIXTURE": str(out / "vectors.json"),
            "LOUDKIT_FIXTURE_DIR": str(out),
        }
        failed = []
        for port in args.ports:
            result = subprocess.run(
                PORTS[port],
                cwd=CWDS[port],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            ok = result.returncode == 0
            if ok:
                print(f"  {port:6} ok")
            else:
                n = divergence_count(port, result.stdout + result.stderr)
                if n is None:
                    print(f"  {port:6} FAILED  (no comparison count; see log)")
                else:
                    print(f"  {port:6} DIVERGED  ({n} mismatch{'es' if n != 1 else ''})")
                failed.append(port)
            if not ok:
                combined = (result.stdout + result.stderr).strip()
                # `out` is temporary and disappears on return. Print the full
                # output instead of pointing at a log that will no longer exist.
                print(combined)

    if failed:
        print(
            f"\n{', '.join(failed)} failed. Reproduce with "
            f"--seed {args.seed} --cases {args.cases}. For a text mismatch, add the case "
            "to tools/make_speechtext_fixture.py so it is checked forever."
        )
        return 1
    print("\nall ports agree with Python on every generated case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
