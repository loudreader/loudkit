"""A listening page: every passage, every system, side by side with the verdict.

The scores say who won. This says why, and it is the only output in the Tier 2.5
chain that lets a person overrule the judge. Open it, read the passage, play the
renders in order, and decide whether the model heard what you hear. A number
nobody has listened behind is a number nobody should publish.

Rows are ordered worst first: passages lost in both presentation orders come at
the top, because those are the ones with something to fix. The judge's note sits
next to the audio that earned it.

Audio is referenced, not embedded, so the page stays small and stays honest
about where its evidence lives. Keep it next to the ``audio/`` tree it points
at.

Usage::

    python research/judge_contact_sheet.py \\
        --results out/judge/loudkit-vs-kokoro.jsonl out/judge/loudkit-vs-pockettts.jsonl \\
        --out out/judge/listen.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
READING_SET = REPO / "tests/data/judge/reading-en.json"
DIMENSIONS = ("prosody", "correctness")

STYLE = """
:root { color-scheme: light dark; --line: #d8d8d8; --dim: #6b6b6b;
        --loss: #b3261e; --win: #1f6f43; --tie: #6b6b6b; --bg: #fff; --fg: #111;
        --card: #fafafa; }
@media (prefers-color-scheme: dark) {
  :root { --line: #333; --dim: #9a9a9a; --loss: #f2b8b5; --win: #7ddba4;
          --bg: #121212; --fg: #e8e8e8; --card: #1c1c1c; }
}
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 2rem 1.25rem 6rem; max-width: 62rem; background: var(--bg);
       color: var(--fg); font: 15px/1.55 system-ui, -apple-system, sans-serif; }
h1 { font-size: 1.35rem; margin: 0 0 .3rem; }
.sub { color: var(--dim); margin: 0 0 1.5rem; }
table.summary { border-collapse: collapse; margin: 0 0 2.5rem; font-size: .9rem; }
table.summary th, table.summary td { padding: .35rem .8rem .35rem 0; text-align: left; }
table.summary th { border-bottom: 1px solid var(--line); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.passage { border-top: 1px solid var(--line); padding: 1.4rem 0; }
.pid { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .8rem;
       color: var(--dim); }
.text { margin: .5rem 0 1rem; }
.row { display: grid; grid-template-columns: 9rem minmax(15rem, 1fr) minmax(0, 1.2fr);
       gap: .75rem; align-items: center; padding: .3rem 0; }
.name { font-weight: 600; font-size: .9rem; }
.name .voice { display: block; font-weight: 400; font-size: .78rem; color: var(--dim); }
audio { width: 100%; height: 34px; }
.note { font-size: .82rem; color: var(--dim); }
.verdict { font-size: .78rem; font-weight: 600; letter-spacing: .02em; }
.loss { color: var(--loss); } .win { color: var(--win); } .tie { color: var(--tie); }
.flag { display: inline-block; margin-left: .5rem; padding: .05rem .4rem; border-radius: 3px;
        background: var(--card); border: 1px solid var(--line); font-size: .72rem;
        color: var(--dim); }
"""


def load_results(paths: list[Path], system_a: str) -> dict[str, dict[str, list[dict]]]:
    """{comparator: {passage id: [record, ...]}}

    The comparator is resolved once per file, from every name the file
    mentions. Deciding it per record fails on a record that is all ties and
    played loudkit first: such a record names nobody else at all.
    """
    grouped: dict[str, dict[str, list[dict]]] = {}
    for path in paths:
        records = [json.loads(raw) for raw in path.read_text().splitlines() if raw.strip()]
        names: set[str] = set()
        for record in records:
            names.add(record["first"])
            names.update(record[f"{d}_winner"] for d in DIMENSIONS)
        names -= {"tie", system_a}
        if len(names) != 1:
            raise SystemExit(f"{path}: expected one comparator, found {sorted(names)}")
        by_passage = grouped.setdefault(names.pop(), {})
        for record in records:
            by_passage.setdefault(record["id"], []).append(record)
    return grouped


def outcome(records: list[dict], system_a: str, dimension: str) -> tuple[str, str]:
    """A both-orders verdict, and the CSS class that colours it."""
    winners = {record[f"{dimension}_winner"] for record in records}
    if len(records) < 2 or len(winners) > 1:
        return "split", "tie"
    winner = winners.pop()
    if winner == "tie":
        return "tie", "tie"
    return ("won", "win") if winner == system_a else ("lost", "loss")


def rank(records_by_comparator: dict[str, list[dict]], system_a: str) -> tuple[int, int]:
    """Worst first: most both-order prosody losses, then correctness losses."""
    lost = sum(
        1
        for records in records_by_comparator.values()
        if outcome(records, system_a, "prosody")[0] == "lost"
    )
    lost_correctness = sum(
        1
        for records in records_by_comparator.values()
        if outcome(records, system_a, "correctness")[0] == "lost"
    )
    return (-lost, -lost_correctness)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--reading-set", type=Path, default=READING_SET)
    parser.add_argument("--audio-root", type=Path, default=Path("out/judge/audio"))
    parser.add_argument("--system-a", default="loudkit")
    parser.add_argument("--out", type=Path, default=Path("out/judge/listen.html"))
    args = parser.parse_args()

    passages = {p["id"]: p for p in json.loads(args.reading_set.read_text())["passages"]}
    grouped = load_results(args.results, args.system_a)
    comparators = sorted(grouped)

    def voice_of(system: str) -> str:
        meta = args.audio_root / system / "render.json"
        if not meta.is_file():
            return ""
        payload = json.loads(meta.read_text())
        return str(payload.get("voice", ""))

    # Only passages judged against every comparator, so each row compares like
    # with like rather than quietly dropping a column.
    ids = sorted(set.intersection(*(set(grouped[c]) for c in comparators)))
    ids.sort(key=lambda i: rank({c: grouped[c][i] for c in comparators}, args.system_a))

    audio_root = args.audio_root.resolve()
    out_dir = args.out.resolve().parent

    def src(system: str, passage_id: str) -> str:
        target = audio_root / system / f"{passage_id}.wav"
        try:
            return str(target.relative_to(out_dir))
        except ValueError:
            return target.as_uri()

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(args.system_a)} listening sheet</title>",
        f"<style>{STYLE}</style>",
        f"<h1>{html.escape(args.system_a)} against {html.escape(', '.join(comparators))}</h1>",
        f"<p class='sub'>{len(ids)} passages, worst first. "
        "A verdict is shown only where both presentation orders agreed; "
        "<em>split</em> means they did not, which is its own finding.</p>",
    ]

    parts.append(
        "<table class='summary'><tr><th>system</th><th>voice</th>"
        "<th>lost prosody</th><th>lost correctness</th></tr>"
    )
    for comparator in comparators:
        lost_p = sum(
            1
            for i in ids
            if outcome(grouped[comparator][i], args.system_a, "prosody")[0] == "lost"
        )
        lost_c = sum(
            1
            for i in ids
            if outcome(grouped[comparator][i], args.system_a, "correctness")[0] == "lost"
        )
        parts.append(
            f"<tr><td>{html.escape(comparator)}</td>"
            f"<td>{html.escape(voice_of(comparator))}</td>"
            f"<td class='num'>{lost_p}/{len(ids)}</td>"
            f"<td class='num'>{lost_c}/{len(ids)}</td></tr>"
        )
    parts.append("</table>")

    for passage_id in ids:
        passage = passages[passage_id]
        parts.append("<div class='passage'>")
        parts.append(
            f"<div class='pid'>{html.escape(passage_id)} · "
            f"{html.escape(passage['source'])} · {len(passage['text'])} chars</div>"
        )
        parts.append(f"<div class='text'>{html.escape(passage['text'])}</div>")
        parts.append(
            f"<div class='row'><div class='name'>{html.escape(args.system_a)}"
            f"<span class='voice'>{html.escape(voice_of(args.system_a))}</span></div>"
            f"<div><audio controls preload='none' "
            f"src='{html.escape(src(args.system_a, passage_id))}'></audio></div>"
            f"<div class='note'></div></div>"
        )
        for comparator in comparators:
            records = grouped[comparator][passage_id]
            verdicts = []
            for dimension in DIMENSIONS:
                label, css = outcome(records, args.system_a, dimension)
                verdicts.append(f"<span class='verdict {css}'>{dimension[:4]}: {label}</span>")
            notes = " / ".join(
                html.escape(record.get("note", "")) for record in records if record.get("note")
            )
            parts.append(
                f"<div class='row'><div class='name'>{html.escape(comparator)}"
                f"<span class='voice'>{html.escape(voice_of(comparator))}</span></div>"
                f"<div><audio controls preload='none' "
                f"src='{html.escape(src(comparator, passage_id))}'></audio></div>"
                f"<div class='note'>{' '.join(verdicts)}<br>{notes}</div></div>"
            )
        parts.append("</div>")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts) + "\n")
    print(f"wrote {args.out}: {len(ids)} passages against {', '.join(comparators)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
