"""Aggregate pairwise judgments into a table and a dot-and-whisker chart.

Reads one or more JSONL files written by ``research/judge_pairwise.py``, one per
comparator, and reports for each: the preference score for loudkit, a 95%
confidence interval, the tie rate, and two numbers that say how much to trust
the first three.

**Order consistency** is the share of passages where both presentation orders
named the same winner. A high score with low consistency is a coin flip
wearing a rosette.

**Position bias** is the mean score of whichever recording was played first,
across every call and both systems. 0.5 means the slot did not matter. A value
far from 0.5 means the judge has a favourite slot, and the order-balanced
design is the only reason the headline number survives it.

The interval is a passage-cluster bootstrap: passages are resampled with
replacement and both of a passage's judgments travel together, because they are
not independent observations. Resampling individual calls would report an
interval roughly a third too narrow.

Usage::

    python research/judge_report.py out/judge/*.jsonl --svg docs/assets/judge.svg
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BOOTSTRAP_DRAWS = 10000
DIMENSIONS = ("prosody", "correctness")


def load(path: Path, system_a: str) -> tuple[str, dict[str, list[dict]]]:
    """Group a comparator's judgments by passage, and name the comparator."""
    by_passage: dict[str, list[dict]] = {}
    others: set[str] = set()
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        by_passage.setdefault(record["id"], []).append(record)
        for dimension in DIMENSIONS:
            winner = record[f"{dimension}_winner"]
            if winner not in ("tie", system_a):
                others.add(winner)
        others.add(record["first"])
    others.discard(system_a)
    if len(others) != 1:
        raise SystemExit(f"{path}: expected exactly one comparator, found {sorted(others)}")
    return others.pop(), by_passage


def score(winner: str, system_a: str) -> float:
    if winner == "tie":
        return 0.5
    return 1.0 if winner == system_a else 0.0


def bootstrap(per_passage: list[list[float]], seed: int = 1234) -> tuple[float, float]:
    """95% interval over passage clusters. Deterministic for a given input."""
    if not per_passage:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    count = len(per_passage)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        pooled = [value for _ in range(count) for value in per_passage[rng.randrange(count)]]
        draws.append(sum(pooled) / len(pooled))
    draws.sort()
    return (draws[int(0.025 * BOOTSTRAP_DRAWS)], draws[int(0.975 * BOOTSTRAP_DRAWS)])


def summarise(by_passage: dict[str, list[dict]], system_a: str) -> dict:
    calls = [record for records in by_passage.values() for record in records]
    both_orders = sum(
        1 for records in by_passage.values() if len({r["order"] for r in records}) == 2
    )

    result: dict = {
        "passages": len(by_passage),
        "judgments": len(calls),
        "order_balanced_passages": both_orders,
        # 0.5 means the slot did not matter; the distance from 0.5 is the bias
        # the order-balanced design is cancelling.
        "position_bias": (
            sum(score(r[f"{d}_winner"], r["first"]) for r in calls for d in DIMENSIONS)
            / (len(calls) * len(DIMENSIONS))
            if calls
            else float("nan")
        ),
    }

    for dimension in DIMENSIONS:
        key = f"{dimension}_winner"
        clusters = [
            [score(record[key], system_a) for record in records]
            for records in by_passage.values()
        ]
        flat = [value for cluster in clusters for value in cluster]
        low, high = bootstrap(clusters)
        # Over the order-balanced passages only, and both halves of that
        # matter. A passage judged in one order has one winner by arithmetic,
        # not by agreement, so counting it says the orders agreed when only one
        # of them ran. A run that lost calls to the API would then report its
        # own losses as confidence, in the number this report tells a reader to
        # trust the other three by. `losses` already answers to `len == 2`.
        balanced = [
            records
            for records in by_passage.values()
            if len({record["order"] for record in records}) == 2
        ]
        agreed = sum(1 for records in balanced if len({record[key] for record in records}) == 1)
        result[dimension] = {
            "score": sum(flat) / len(flat) if flat else float("nan"),
            "ci": [low, high],
            "tie_rate": sum(1 for record in calls if record[key] == "tie") / len(calls)
            if calls
            else float("nan"),
            "order_consistency": agreed / len(balanced) if balanced else float("nan"),
        }
    return result


# --- chart -------------------------------------------------------------------

WIDTH, ROW, PAD_LEFT, PAD_RIGHT, PAD_TOP = 940, 78, 210, 90, 74
COLOURS = {"prosody": "#2f6f4f", "correctness": "#8a5a1f"}


def svg(results: list[tuple[str, dict]], system_a: str, low: float, high: float) -> str:
    """A dot-and-whisker chart, one row per comparator and dimension.

    Hand-written SVG rather than a plotting dependency: the chart is six
    numbers per row, and a docs image should not require a toolchain to
    regenerate.
    """
    rows = len(results) * 2
    height = PAD_TOP + rows * ROW + 56
    plot = WIDTH - PAD_LEFT - PAD_RIGHT

    def x(value: float) -> float:
        return PAD_LEFT + (value - low) / (high - low) * plot

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" font-family="system-ui, -apple-system, sans-serif">',
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
        f'<text x="{PAD_LEFT}" y="30" font-size="17" font-weight="600" fill="#111">'
        f"{system_a} preference, order-balanced pairwise judge</text>",
        f'<text x="{PAD_LEFT}" y="52" font-size="13" fill="#555">'
        f"50% is parity. Whiskers are 95% passage-cluster bootstrap intervals. "
        f"Tie rate in grey.</text>",
    ]

    for tick in range(int(low * 100), int(high * 100) + 1, 10):
        position = x(tick / 100)
        parity = tick == 50
        out.append(
            f'<line x1="{position:.1f}" y1="{PAD_TOP - 10}" x2="{position:.1f}" '
            f'y2="{PAD_TOP + rows * ROW - 20}" stroke="{"#999" if parity else "#e6e6e6"}" '
            f'stroke-width="{2 if parity else 1}"{"" if parity else ""}/>'
        )
        out.append(
            f'<text x="{position:.1f}" y="{PAD_TOP + rows * ROW + 4}" font-size="12" '
            f'fill="#666" text-anchor="middle">{tick}%</text>'
        )

    y = PAD_TOP + 16
    for comparator, summary in results:
        for dimension in DIMENSIONS:
            entry = summary[dimension]
            centre, (ci_low, ci_high) = entry["score"], entry["ci"]
            colour = COLOURS[dimension]
            out.append(
                f'<text x="{PAD_LEFT - 14}" y="{y + 5}" font-size="13" fill="#222" '
                f'text-anchor="end">{comparator} · {dimension}</text>'
            )
            out.append(
                f'<line x1="{x(ci_low):.1f}" y1="{y}" x2="{x(ci_high):.1f}" y2="{y}" '
                f'stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>'
            )
            for edge in (ci_low, ci_high):
                out.append(
                    f'<line x1="{x(edge):.1f}" y1="{y - 6}" x2="{x(edge):.1f}" y2="{y + 6}" '
                    f'stroke="{colour}" stroke-width="2.5"/>'
                )
            out.append(f'<circle cx="{x(centre):.1f}" cy="{y}" r="6.5" fill="{colour}"/>')
            out.append(
                f'<text x="{x(centre):.1f}" y="{y - 14}" font-size="13" font-weight="600" '
                f'fill="{colour}" text-anchor="middle">{centre * 100:.1f}%</text>'
            )
            out.append(
                f'<text x="{x(centre):.1f}" y="{y + 26}" font-size="11" fill="#888" '
                f'text-anchor="middle">tie {entry["tie_rate"] * 100:.0f}%</text>'
            )
            y += ROW
    out.append("</svg>")
    return "\n".join(out)


# --- main --------------------------------------------------------------------


def losses(by_passage: dict[str, list[dict]], system_a: str, dimension: str) -> list[dict]:
    """Passages the comparator won in *both* orders, with the judge's reason.

    Both orders, not one: a single-order loss is as likely to be the slot as
    the reading. What survives order-balancing is the actionable list, and it
    is the only output here that says what to go and fix.
    """
    key = f"{dimension}_winner"
    found = []
    for passage, records in sorted(by_passage.items()):
        winners = {record[key] for record in records}
        if len(records) == 2 and winners and system_a not in winners and "tie" not in winners:
            found.append(
                {
                    "id": passage,
                    "against": winners.pop(),
                    "notes": [record.get("note", "") for record in records],
                }
            )
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="JSONL files from judge_pairwise")
    parser.add_argument("--system-a", default="loudkit")
    parser.add_argument("--svg", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--losses",
        action="store_true",
        help="list the passages lost in both orders, with the judge's reason",
    )
    args = parser.parse_args()

    summaries, grouped = [], {}
    for path in args.results:
        comparator, by_passage = load(path, args.system_a)
        grouped[comparator] = by_passage
        summaries.append((comparator, summarise(by_passage, args.system_a)))
    summaries.sort(key=lambda row: row[1]["prosody"]["score"], reverse=True)

    print(
        f"{'comparator':<24} {'dimension':<12} {'score':>7} {'95% CI':>16} "
        f"{'tie':>6} {'order-agree':>12}"
    )
    for comparator, summary in summaries:
        for dimension in DIMENSIONS:
            entry = summary[dimension]
            interval = f"{entry['ci'][0] * 100:.1f}-{entry['ci'][1] * 100:.1f}"
            print(
                f"{comparator:<24} {dimension:<12} {entry['score'] * 100:>6.1f}% "
                f"{interval:>16} {entry['tie_rate'] * 100:>5.0f}% "
                f"{entry['order_consistency'] * 100:>11.0f}%"
            )
    print()
    for comparator, summary in summaries:
        note = (
            ""
            if summary["order_balanced_passages"] == summary["passages"]
            else (
                f"  WARNING {summary['passages'] - summary['order_balanced_passages']} "
                f"passages have only one order"
            )
        )
        print(
            f"{comparator:<24} {summary['passages']} passages, "
            f"{summary['judgments']} judgments, "
            f"position bias {summary['position_bias'] * 100:.1f}%{note}"
        )

    if args.losses:
        for comparator, _ in summaries:
            for dimension in DIMENSIONS:
                lost = losses(grouped[comparator], args.system_a, dimension)
                print(
                    f"\n{args.system_a} lost {dimension} to {comparator} in both orders: "
                    f"{len(lost)}/{len(grouped[comparator])} passages"
                )
                for entry in lost:
                    print(f"  {entry['id']}")
                    for note in entry["notes"]:
                        print(f"      {note}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"system_a": args.system_a, "comparators": dict(summaries)}, indent=2)
            + "\n"
        )
        print(f"\nwrote {args.json}")

    if args.svg:
        bounds = [
            value
            for _, summary in summaries
            for dimension in DIMENSIONS
            for value in summary[dimension]["ci"]
        ] + [0.5]
        low = min(0.4, (min(bounds) * 100 // 10) / 10)
        high = max(0.6, -(-max(bounds) * 100 // 10) / 10)
        args.svg.parent.mkdir(parents=True, exist_ok=True)
        args.svg.write_text(svg(summaries, args.system_a, low, high) + "\n")
        print(f"wrote {args.svg}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
