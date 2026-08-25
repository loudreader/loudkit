"""Fetch the NST pronunciation lexicons from Språkbanken (all CC0).

Three lexicons, one licence, and the single most actionable external resource
the per-language research found: the Swedish one carries the accent-1/accent-2
labels no open TTS toolchain uses (espeak-ng has no word-accent machinery at
all), the Danish one marks stød — the contrast the standard phonemizer
collapses entirely (hun/hund come out identical) — and the Norwegian one
carries tonelag. 237k–927k manually checked entries each.

They are downloads, not vendored files: 20–100 MB each, CC0 so nothing blocks
redistribution, but a lexicon in the repo would be a copy that rots while the
source is maintained. This script fetches into ``lexicons/`` (gitignored) and
prints the SHA-256 of what arrived, so a use of the data can pin what it used.

    python tools/fetch_nst_lexicons.py sv da no

The file only appears at its final name once it is complete and its digest
checks out, so an interrupted or hijacked download cannot be mistaken for a
lexicon by whatever consumes ``lexicons/`` next.

Formats: semicolon-separated, SAMPA transcriptions (not IPA — budget a mapping
before consuming), compound decomposition marked with '+', accent marked with
'"' (accent 1) and '""' (accent 2) per the NST convention docs. The Danish
stød marker is '?' after the vowel. Retroflexion is applied within words and
deliberately NOT across compound boundaries — a documented divergence from
Lindqvist's phonetics description, so pick a side per use and say which.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Seconds without a byte before the read is abandoned. Without one, a stalled
# server holds the socket open forever and a 100 MB fetch has no failure mode
# except the operator noticing.
TIMEOUT_S = 60.0

# The smallest of the three lexicons is ~20 MB. An HTTP error page, a captive
# portal splash or a truncated transfer is kilobytes, and arrives with a 200.
MIN_BYTES = 5_000_000

# Språkbanken resource ids, verified 2026-08: sbr-22 (sv), sbr-23 (no), sbr-26
# (da). Språkbanken publishes no checksums for these tarballs, so the pins are
# recorded from a fetch: run this script, then paste the printed digest here.
# ``None`` means unpinned — the download is size-checked but not verified, and
# the script says so on every run.
SOURCES: dict[str, tuple[str, str | None]] = {
    "sv": ("https://www.nb.no/sbfil/leksikalske_databaser/leksikon/sv.leksikon.tar.gz", None),
    "no": ("https://www.nb.no/sbfil/leksikalske_databaser/leksikon/no.leksikon.tar.gz", None),
    "da": ("https://www.nb.no/sbfil/leksikalske_databaser/leksikon/da.leksikon.tar.gz", None),
}


def fetch(language: str, dest: Path) -> None:
    url, pinned = SOURCES[language]
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / f"nst_{language}.tar.gz"
    part = target.with_name(target.name + ".part")
    print(f"{language}: {url}")

    sha = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(url, timeout=TIMEOUT_S) as response,  # noqa: S310 - fixed https hosts
            part.open("wb") as out,
        ):
            while chunk := response.read(1 << 20):
                sha.update(chunk)
                out.write(chunk)
                size += len(chunk)
        digest = sha.hexdigest()
        if size < MIN_BYTES:
            raise SystemExit(
                f"{language}: {size:,} bytes is too small for a lexicon "
                f"(minimum {MIN_BYTES:,}) — an error page, not the tarball"
            )
        if pinned is not None and digest != pinned:
            raise SystemExit(f"{language}: sha256 {digest} does not match the pin {pinned}")
        os.replace(part, target)
    finally:
        part.unlink(missing_ok=True)

    print(f"  -> {target} ({size:,} bytes)")
    print(f"  sha256 {digest}")
    if pinned is None:
        print(f"  unpinned — paste that digest into SOURCES[{language!r}] to pin it")


if __name__ == "__main__":
    languages = sys.argv[1:] or list(SOURCES)
    for lang in languages:
        fetch(lang, REPO / "lexicons")
