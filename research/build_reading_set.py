"""Freeze the English reading set used by the pairwise judge.

Draws 250-500 character passages from public-domain Project Gutenberg texts,
one book at a time so that no single author's cadence dominates, and writes a
frozen JSON with a content hash. The set is built once and committed; this tool
exists so the build is reproducible and auditable, not so it runs every time.

Passages are prose paragraphs, not sentences, because the thing being measured
is reading: phrasing across clause boundaries, breath placement, and whether a
system still knows where it is four sentences in. A benchmark of isolated
sentences measures none of that.

Dialogue-heavy and verse paragraphs are dropped. They are legitimate reading
material but they reward a different skill, and mixing them in makes a single
preference number harder to interpret rather than more representative.

**Contamination.** Gutenberg prose is in the training data of most open TTS
systems, loudkit included, and LibriVox recordings of these same books are in
many. This set therefore measures reading quality on familiar material. It does
not measure generalisation to unseen text, and no claim built on it should say
otherwise.

Usage::

    python research/build_reading_set.py --out tests/data/judge/reading-en.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from collections.abc import Iterator
from pathlib import Path

MIRROR = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# Public-domain English prose, spread across period, register and author, so a
# preference score is not a verdict on one writer's sentence length.
BOOKS: tuple[tuple[int, str], ...] = (
    (1342, "Austen, Pride and Prejudice"),
    (11, "Carroll, Alice's Adventures in Wonderland"),
    (98, "Dickens, A Tale of Two Cities"),
    (1661, "Doyle, The Adventures of Sherlock Holmes"),
    (2701, "Melville, Moby Dick"),
    (84, "Shelley, Frankenstein"),
    (74, "Twain, The Adventures of Tom Sawyer"),
    (145, "Eliot, Middlemarch"),
    (219, "Conrad, Heart of Darkness"),
    (4300, "Joyce, Ulysses"),
    (5200, "Kafka, Metamorphosis (Wyllie tr.)"),
    (1400, "Dickens, Great Expectations"),
    (76, "Twain, Huckleberry Finn"),
    (2600, "Tolstoy, War and Peace (Maude tr.)"),
    (120, "Stevenson, Treasure Island"),
    (205, "Thoreau, Walden"),
    (174, "Wilde, The Picture of Dorian Gray"),
    (2814, "Joyce, Dubliners"),
    (1260, "Bronte, Jane Eyre"),
    (768, "Bronte, Wuthering Heights"),
    (35, "Wells, The Time Machine"),
    (36, "Wells, The War of the Worlds"),
)

MIN_CHARS, MAX_CHARS = 250, 500

# A Gutenberg text is wrapped in a licence header and footer; the marks below
# are stable across the mirror's plain-text editions.
START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)
END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.S)


def fetch(book_id: int, cache: Path) -> str:
    target = cache / f"pg{book_id}.txt"
    if target.is_file():
        return target.read_text(encoding="utf-8", errors="replace")
    url = MIRROR.format(id=book_id)
    print(f"  fetching {url}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8", errors="replace")
    cache.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text


def body(text: str) -> str:
    # The mirror serves CRLF. Reading a cached copy back through Python's
    # universal-newline translation hides that, so a fresh fetch and a cached
    # one would otherwise parse differently and yield different sets.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = START.search(text)
    end = END.search(text)
    return text[start.end() if start else 0 : end.start() if end else len(text)]


def paragraphs(text: str) -> Iterator[str]:
    for raw in re.split(r"\n[ \t]*\n", body(text)):
        collapsed = re.sub(r"\s+", " ", raw).strip()
        if collapsed:
            yield collapsed


def acceptable(paragraph: str) -> bool:
    if not MIN_CHARS <= len(paragraph) <= MAX_CHARS:
        return False
    # Prose only: quotation marks signal dialogue, and an all-caps or heavily
    # punctuated line is usually a heading, a chapter mark or verse.
    if any(mark in paragraph for mark in ('"', "“", "”", "_", "[", "]", "*")):
        return False
    if not paragraph[0].isupper() or paragraph[-1] not in ".?!":
        return False
    letters = [c for c in paragraph if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) > 0.15:
        return False
    # Digits and abbreviations belong in a text-normalisation probe, not in a
    # reading-prosody set where they would confound the correctness dimension.
    if re.search(r"\d", paragraph):
        return False
    words = paragraph.split()
    return len(words) >= 40 and max(len(w) for w in words) <= 20


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--cache", type=Path, default=Path("out/gutenberg"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    per_book = -(-args.count // len(BOOKS))
    chosen: list[dict] = []
    leftover: list[dict] = []

    # An even first pass, then a fill from what is left over. A dialogue-heavy
    # book yields few eligible paragraphs, and without the fill its shortfall
    # would silently shrink the set rather than being made up elsewhere.
    for book_id, title in BOOKS:
        try:
            candidates = [p for p in paragraphs(fetch(book_id, args.cache)) if acceptable(p)]
        except Exception as exc:  # one unreachable mirror is not fatal
            print(f"  ! {title}: {exc}", file=sys.stderr, flush=True)
            continue
        rng.shuffle(candidates)
        rows = [{"source": title, "gutenberg_id": book_id, "text": t} for t in candidates]
        chosen.extend(rows[:per_book])
        # Cap the fill pool per book so a 1,200-paragraph Tolstoy cannot take
        # over the remainder.
        leftover.extend(rows[per_book : per_book * 2])
        print(
            f"  {title}: {len(candidates)} eligible, took {min(len(rows), per_book)}",
            flush=True,
        )

    rng.shuffle(leftover)
    chosen.extend(leftover[: max(0, args.count - len(chosen))])
    if len(chosen) < args.count:
        raise SystemExit(f"only {len(chosen)} passages available, wanted {args.count}")

    rng.shuffle(chosen)
    chosen = chosen[: args.count]
    for index, item in enumerate(chosen):
        item["id"] = f"en{index:04d}"

    payload = {
        "version": 1,
        "about": (
            "English reading passages for the order-balanced pairwise judge. "
            "Public-domain Project Gutenberg prose, 250-500 characters, no "
            "dialogue, no digits. Present in the training data of most open TTS "
            "systems; this set measures reading quality on familiar material, "
            "not generalisation."
        ),
        "language": "en",
        "seed": args.seed,
        "count": len(chosen),
        "passages": [
            {
                "id": i["id"],
                "source": i["source"],
                "gutenberg_id": i["gutenberg_id"],
                "text": i["text"],
            }
            for i in chosen
        ],
    }
    body_bytes = json.dumps(payload["passages"], ensure_ascii=False).encode()
    payload["sha256"] = hashlib.sha256(body_bytes).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    lengths = [len(p["text"]) for p in payload["passages"]]
    print(
        f"wrote {args.out}: {len(lengths)} passages, "
        f"{min(lengths)}-{max(lengths)} chars, sha256 {payload['sha256'][:16]}"
    )
    return 0


REPO_DEFAULT = Path(__file__).resolve().parent.parent / "tests/data/judge/reading-en.json"

if __name__ == "__main__":
    sys.exit(main())
