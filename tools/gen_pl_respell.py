"""Generate the Polish respelling lexicon from CMUdict.

An English word embedded in Polish text should be WRITTEN the way a Polish
reader SAYS it — "download" → "dałnloud" — because the engine is grapheme
based and reads Polish text with Polish letter-to-sound rules. The curated
list in LexicalRespelling.swift topped out around a hundred words and the
long tail sounded exactly like the problem it was meant to fix; this script
makes the long tail: ARPAbet phonemes → Polish orthography, over all of
CMUdict.

The mapping is deliberately the POLISH-ACCENT rendition, not IPA fidelity:
TH → t, DH → d, NG → ng, W → ł, R stays a Polish r. That is how the words
sound in a Polish sentence, and it matches the hand-curated forms the ear
test approved ("fidbek", "łikend", "dedlajn").

  python tools/gen_pl_respell.py   # -> swift/LoudKitText/Resources/pl_en_respell.json
"""

import hashlib
import io
import json
import os
import pathlib
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "tools", "cmudict.dict")
PL = os.path.join(ROOT, "tools", "pl_50k.txt")
# Every copy, not just Swift's. This wrote one file and `tools/pack_assets.py`
# read a different one (`python/loudkit/models/data/`), so the generated lexicon and
# the packed lexicon were two artefacts with one generator between them —
# regenerating updated Swift and left the other four to be copied by hand. They
# are byte-identical today; nothing was keeping them that way.
OUTPUTS = [
    os.path.join(ROOT, "swift", "LoudKitText", "Resources", "pl_en_respell.json"),
    os.path.join(ROOT, "python", "loudkit", "models", "data", "pl_en_respell.json"),
    os.path.join(ROOT, "go", "speechtext", "pl_en_respell.json"),
    os.path.join(ROOT, "rust", "src", "pl_en_respell.json"),
    os.path.join(ROOT, "js", "data", "pl_en_respell.json"),
]
OUT = OUTPUTS[0]

# ARPAbet -> Polish orthography, Polish-accent flavour.
PHONES = {
    "AA": "a",
    "AE": "e",
    "AH": "a",
    "AO": "o",
    "AW": "ał",
    "AY": "aj",
    "B": "b",
    "CH": "cz",
    "D": "d",
    "DH": "d",
    "EH": "e",
    "ER": "er",
    "EY": "ej",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "y",
    "IY": "i",
    "JH": "dż",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ng",
    "OW": "oł",
    "OY": "oj",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "sz",
    "T": "t",
    "TH": "t",
    "UH": "u",
    "UW": "u",
    "V": "w",
    "W": "ł",
    "Y": "j",
    "Z": "z",
    "ZH": "ż",
}

# Unstressed AH (schwa) reads better as "e" in some endings, but chasing that
# needs per-word care; "a" is the safe Polish-accent default everywhere.


def respell(phones):
    out = []
    for ph in phones:
        stress = ph[-1].isdigit()
        base = ph[:-1] if stress else ph
        mapped = PHONES.get(base)
        if mapped is None:
            return None
        out.append(mapped)
    s = "".join(out)
    # Polish orthography cleanups: double letters collapse (except after a
    # syllable break we cannot see — collapse is the safe default), and a
    # trailing "y" after a vowel becomes "j" ("plej", not "pley").
    s = re.sub(r"(.)\1", r"\1", s)
    # [ts] is exactly the Polish letter "c": "notes" -> "nołc", "sports" ->
    # "sporc". The cluster "ts" written out reads as two separate sounds with
    # a seam the model audibly trips on.
    return s.replace("ts", "c")


def _write_all(payload: str) -> str:
    """Write the lexicon to every copy and return the digest they share.

    One generator, five files. This wrote Swift's copy alone while
    `tools/pack_assets.py` read Python's, so regenerating updated one of the two
    artefacts and left the other four to be copied by hand — the drift the
    fingerprint's grammar digest exists to catch, arriving through the file the
    digest does not cover.
    """
    for path in OUTPUTS:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(payload)
    digests = {hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in OUTPUTS}
    if len(digests) != 1:
        raise SystemExit(f"copies disagree after writing: {sorted(digests)}")
    return digests.pop()


def main():
    # Everything this prints is Polish, and the summary below reports the
    # respellings themselves — "ł", "ę", "ż". Python picks the console's
    # encoding for stdout, which on Windows is cp1252 and cannot represent any
    # of them: the generator wrote all five copies of the lexicon correctly and
    # then died on the diagnostic line describing them, exit 1, output already
    # on disk. The work is done in UTF-8; say so about the report too. Guarded:
    # under a capture (pytest, a pipe wrapper) stdout is not a TextIOWrapper and
    # has no reconfigure.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    # The false-positive gate, baked at generation time: any word that is
    # ALSO a common Polish form ("system", "problem", "to", "ten") must stay
    # Polish — respelling it would mangle native text, which is a worse
    # failure than missing an anglicism. OpenSubtitles top-50k carries the
    # inflected forms a lemma list would miss, which is exactly what we need.
    polish = set()
    with open(PL, encoding="utf-8") as f:
        for line in f:
            w = line.split()[0].strip().lower()
            if w:
                polish.add(w)
    dropped = 0
    lex = {}
    all_words = set()
    respell_all = {}
    with open(SRC, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";;;"):
                continue
            head, *phones = line.split()
            # cmudict alternates: "word(2)" — first pronunciation wins.
            if "(" in head:
                continue
            # apostrophes and dots stay out of the lexicon; the runtime
            # matcher handles inflection endings itself.
            word = head.lower()
            if not re.fullmatch(r"[a-z][a-z'\-\.]*", word):
                continue
            if "'" in word or "." in word:
                continue
            r = respell(phones)
            if r is None:
                continue
            all_words.add(word)
            # Rescue from the gate: the Polish list is subtitles and carries
            # English words verbatim ("weekend", "thought"). Orthography that
            # Polish never uses natively marks a word as English even when
            # the list has it — q/v/x, and digraphs th/ck/gh/ph/ee/oo/sh.
            english_marked = re.search(r"[qvx]|th|ck|gh|ph|ee|oo|sh", word)
            respell_all[word] = r
            if word in polish and not english_marked:
                dropped += 1
                continue
            # A respelling identical to the word would be a wasted lookup.
            if r != word:
                lex[word] = r
    # Two payloads: "respell" is the gated lexicon (safe to transliterate);
    # "words" is EVERY cmudict word — the span detector's question is "is
    # this an English word at all", and the gate must not answer it: "brown"
    # and "dog" leak into the Polish frequency list via subtitles and were
    # breaking spans they belonged inside.
    buf = io.StringIO()
    # respellAll: EVERY word's transliteration, gate ignored — used only
    # INSIDE detected English spans, where "brown" must become "brałn"
    # even though alone it stays Polish-gated. Spans read word-by-word in
    # Polish orthography: the inline [en] tag experiment lost the ear test
    # decisively ("brzmi jak totalne gówno").
    # "polish": the frequency set itself — the runtime stem-walk needs to
    # know a FULL word is Polish before it tries English stems on it
    # ("temperatura" matched temperature+a and came out "tempraczera").
    json.dump(
        {
            "respell": lex,
            "words": sorted(all_words),
            "respellAll": respell_all,
            "polish": sorted(polish),
        },
        buf,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = _write_all(buf.getvalue())
    print(
        f"{len(lex)} entries ({dropped} dropped as common Polish) -> "
        f"{len(OUTPUTS)} copies, sha256 {digest[:16]}"
    )
    for probe in (
        "download",
        "feedback",
        "weekend",
        "deadline",
        "workflow",
        "science",
        "thought",
        "juice",
        "queue",
        "john",
    ):
        print(f"  {probe} -> {lex.get(probe)}")


if __name__ == "__main__":
    main()
