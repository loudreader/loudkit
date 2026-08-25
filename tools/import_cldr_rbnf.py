"""Import CLDR's RBNF test data as a conformance corpus for loudkit.frontend.numbers.

Why this exists
---------------

Our own numbers fixture is 100 hand-written cases. CLDR ships **thousands** of
expected spellouts for exactly our languages, maintained by Unicode under a
permissive licence — a differential-testing corpus two orders of magnitude
larger than ours, written by people who were not looking at our code. Where we
disagree with CLDR, one of us is wrong, and finding out which is the point.

What it imports, and what it deliberately does not
--------------------------------------------------

Only the rulesets that map onto what :func:`loudkit.frontend.numbers.cardinal` claims to
do today. A test corpus that includes rulesets we have no implementation for
would report thousands of "failures" that are really absences, and a suite
where most failures are expected is a suite nobody reads. The unmapped rulesets
(ordinals, oblique cases, year forms) are *counted* and written into the
fixture header as declared non-coverage, so the gap is stated rather than
hidden.

The mapping is per language because the ruleset names encode each language's
own grammar: Polish `-masculine` is our citation form, German `-neuter` is
theirs; Norwegian's citation cardinal is `-masculine`.

Source files land in tests/data/cldr/ verbatim (with the Unicode copyright
header intact); the mapped cases land in tests/data/conformance/numbers_cldr.json.

Run: python tools/import_cldr_rbnf.py <dir-with-ssv-files>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ruleset -> our gender argument (None = citation form). Only rulesets listed
# here are asserted; everything else is counted as declared non-coverage.
MAPPINGS: dict[str, dict[str, str | None]] = {
    # The citation ruleset differs per language because the names encode each
    # language's own grammar. Where CLDR's gendered cardinal apocopates (Spanish
    # "ciento un", Italian "ventun") and ours does not, the citation is
    # `%spellout-numbering`, which CLDR defines as the form used for reading a
    # number out on its own — exactly what `cardinal` returns.
    "en": {"%spellout-cardinal-verbose": None},  # ours says "one hundred and one"
    "pl": {
        "%spellout-numbering": None,
        "%spellout-cardinal-feminine": "f",
        "%spellout-cardinal-neuter": "n",
        "%spellout-cardinal-masculine-personal": "virile",
    },
    "de": {"%spellout-numbering": None},  # standalone "eins"; oblique cases unmapped
    "es": {"%spellout-numbering": None, "%spellout-cardinal-feminine": "f"},
    "fr": {"%spellout-numbering": None},
    "it": {"%spellout-numbering": None},
    "pt_PT": {"%spellout-numbering": None, "%spellout-cardinal-feminine": "f"},
    "nl": {"%spellout-cardinal": None},
    "da": {"%spellout-cardinal-common": None, "%spellout-cardinal-neuter": "neuter"},
    # No grammars yet — imported now so the corpus is waiting when they land.
    "fi": {"%spellout-cardinal": None},
    "no": {"%spellout-cardinal-masculine": None, "%spellout-cardinal-neuter": "neuter"},
    "sv": {"%spellout-cardinal-neuter": None, "%spellout-cardinal-reale": "common"},
}

# European Portuguese: our grammar says "dezasseis", which is pt_PT. Plain
# "pt" in CLDR is Brazilian and would fail on exactly the forms that make the
# variant a decision rather than a default.
TARGET_OVERRIDES = {"pt_PT": "pt"}

# Rows where CLDR contradicts *itself*, so asserting them would enshrine one
# side of a contradiction. Each entry names the internal conflict; the test
# skips these and asserts the count so a regenerated corpus cannot quietly
# grow the list.
DISPUTED: dict[tuple[str, str], str] = {
    ("da", "scale-words"): (
        "CLDR's common ruleset writes 'tusinde' and a bare 'million'; its "
        "neuter ruleset writes 'tusind' and 'en million' for the same values. "
        "Both cannot be right. Pending native adjudication."
    ),
    ("pl", "trailing-one-after-miliard"): (
        "CLDR agrees the trailing 1 of 'tysiąc jeden' and 'milion jeden' does "
        "not inflect, then inflects it in 'miliard jedna'. The uninflected "
        "form follows CLDR's own majority rule."
    ),
    ("fi", "thousand-boundary"): (
        "CLDR joins the thousands group solid to its remainder "
        "(kaksituhattayksi); Kotus and Kielikello prescribe the space "
        "(kaksituhatta yksi), with Korpela's handbook agreeing. The grammar "
        "follows the national authority over CLDR."
    ),
    ("pl", "virile-scale-multipliers"): (
        "CLDR's 999999 keeps scale multipliers plain in the virile series, "
        "then inflects them at 1999999999. The plain form follows CLDR's own "
        "majority rule."
    ),
}


def disputed_reason(lang: str, ruleset: str, value: int) -> str | None:
    if lang == "da" and value >= 1000:
        return DISPUTED[("da", "scale-words")]
    if lang == "fi" and value >= 1000 and value % 1000 and (value // 1000) % 1000:
        return DISPUTED[("fi", "thousand-boundary")]
    if lang == "pl" and ruleset == "%spellout-cardinal-feminine" and value == 1_000_000_001:
        return DISPUTED[("pl", "trailing-one-after-miliard")]
    if (
        lang == "pl"
        and ruleset == "%spellout-cardinal-masculine-personal"
        and value == 1_999_999_999
    ):
        return DISPUTED[("pl", "virile-scale-multipliers")]
    return None


def parse_ssv(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(";", 3)
        if len(parts) == 4:
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def main(source: Path) -> None:
    out: dict[str, list[dict[str, object]]] = {}
    skipped: Counter[str] = Counter()
    unmapped: dict[str, list[str]] = {}

    for ssv in sorted(source.glob("*.ssv")):
        lang = ssv.stem
        if lang not in MAPPINGS:
            continue
        mapping = MAPPINGS[lang]
        cases: list[dict[str, object]] = []
        seen_rulesets: set[str] = set()
        for kind, ruleset, number, expected in parse_ssv(ssv):
            if kind != "spell":
                skipped[f"{lang}:{kind}"] += 1
                continue
            seen_rulesets.add(ruleset)
            if ruleset not in mapping:
                skipped[f"{lang}:{ruleset}"] += 1
                continue
            # Integers only: CLDR writes the number column with "." as its
            # decimal mark regardless of locale, and its fraction *readings*
            # follow conventions ("five tenths") we do not claim to implement.
            if any(ch in number for ch in ".,eE") or not number.lstrip("-").isdigit():
                skipped[f"{lang}:non-integer"] += 1
                continue
            value = int(number)
            # Our grammars refuse past their largest scale on purpose; CLDR
            # spells out up to 10^17. Those rows are non-coverage, not failures.
            reason = disputed_reason(lang, ruleset, value)
            cases.append(
                {
                    "value": value,
                    "gender": mapping[ruleset],
                    "ruleset": ruleset,
                    **({"disputed": reason} if reason else {}),
                    # CLDR marks compound-internal joints with a soft hyphen
                    # (U+00AD): "en\u00adog\u00adtyve". Real orthography has
                    # no hyphen there, and the funnel strips the character as
                    # invisible anyway.
                    "expect": expected.replace("\u00ad", ""),
                }
            )
        target = TARGET_OVERRIDES.get(lang, lang)
        out.setdefault(target, []).extend(cases)
        unmapped[lang] = sorted(r for r in seen_rulesets if r not in mapping)

    fixture = {
        "version": 1,
        "source": "Unicode CLDR common/testData/rbnf (https://github.com/unicode-org/cldr)",
        "license": "Unicode License v3 — https://www.unicode.org/license.txt",
        "about": [
            "Differential corpus: CLDR's expected spellouts for the rulesets that",
            "map onto loudkit.frontend.numbers.cardinal. Written by Unicode, not by us —",
            "where we disagree, one of us is wrong, which is the point.",
            "",
            "Rulesets with no mapping (ordinals, oblique cases, year forms) are",
            "declared non-coverage below, not silent omissions. Values past a",
            "grammar's largest scale are expected to raise and are skipped by the",
            "test with a counted reason.",
        ],
        "not_covered": unmapped,
        "cases": out,
    }
    dest = REPO / "tests" / "data" / "conformance" / "numbers_cldr.json"
    dest.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(len(v) for v in out.values())
    print(f"wrote {dest.name}: {total} cases across {len(out)} languages")
    for lang_key, cases_list in sorted(out.items()):
        print(f"  {lang_key}: {len(cases_list)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
