"""Tier 2.5: order-balanced pairwise LLM-as-judge over an OpenAI-compatible API.

Plays a judge model two recordings of the *same* passage, one from loudkit and
one from a comparator, and asks which reads it better. Every passage is judged
twice, once in each presentation order, because a listener model asked "1 or 2"
does not answer symmetrically: measured position bias in this harness is
reported alongside every result so a reader can see how much of a win is the
system and how much is the slot it sat in.

Two dimensions, scored separately because they dissociate. **Prosody** is
phrasing, emphasis and rhythm. **Correctness** is whether the words of the
passage came out, word for word, without omission, repetition or substitution.
A system can read fluently and drop a clause; the two numbers catch that, one
number hides it.

Scoring follows the usual pairwise convention: a loudkit preference is 1, a tie
0.5, a comparator preference 0, and the reported figure is the mean over every
order-balanced judgment. 50% is parity, not a passing grade.

What this tier cannot see, so nobody reads it further than it goes: a judge
model is not a listening panel, its notion of "better prosody" is its own and
unvalidated against human raters, and it hears voice identity even when the
rubric tells it not to. When the comparator uses a fixed provider voice and
loudkit uses a cloned reference, that asymmetry is in the number. Say so
wherever the number is published.

Audio is loudness-matched before it is sent. An untouched pair leaks a level
difference into a preference score, and level is the single easiest thing to
prefer for the wrong reason.

Credentials come from the environment (``OPENROUTER_API_KEY``) or from
opencode's stored auth, the same path the other OpenRouter tools here use. The
token is never printed and never written to the output.

Usage::

    # one row per passage: id, text, and one wav per system
    python research/judge_pairwise.py --manifest out/judge/manifest.jsonl \\
        --system-a loudkit --system-b kokoro \\
        --out out/judge/loudkit-vs-kokoro.jsonl

    # resume an interrupted run by pointing at the same --out
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeGuard

REPO = Path(__file__).resolve().parent.parent
AUTH_PATH = Path.home() / ".local/share/opencode/auth.json"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"

# Independent normalisation of each waveform to this average level, so the
# louder system cannot win on loudness alone.
TARGET_DBFS = -20.0

SYSTEM_PROMPT = """\
You are grading two text-to-speech recordings of the same written passage.

You will be given the passage text, then Recording 1, then Recording 2. Judge \
on exactly two dimensions, independently:

PROSODY - phrasing, emphasis, rhythm, sentence melody, and how naturally the \
passage is read aloud. Ignore voice identity, timbre, age, accent and \
recording character. You are not judging who has the nicer voice; you are \
judging who reads the passage better.

CORRECTNESS - whether the spoken words match the written passage word for \
word. Penalise omitted words, repeated words, substituted words, invented \
words, mangled numbers or abbreviations, and audio that stops before the \
passage ends or continues past it. Pronunciation preference is not an error; a \
wrong word is.

Answer with a JSON object and nothing else:

{"prosody": 1 | 2 | 0, "correctness": 1 | 2 | 0, "note": "at most 12 words"}

Use 1 or 2 for the better recording on that dimension, and 0 for a genuine \
tie. Do not use 0 to avoid deciding. Judge the two dimensions separately: the \
same recording need not win both."""

USER_TEMPLATE = """\
Passage text:

{text}

Recording 1 follows, then Recording 2."""


# --- credentials -------------------------------------------------------------


def load_token() -> str:
    """Environment first, then opencode's stored auth. Never logged."""
    token = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if token:
        return token
    try:
        entry = json.loads(AUTH_PATH.read_text()).get("openrouter", {})
    except (OSError, ValueError):
        return ""
    return (entry.get("key") or entry.get("access_token") or "").strip()


# --- audio preparation -------------------------------------------------------


def _rms_dbfs(samples) -> float:
    import numpy as np

    rms = float(np.sqrt(np.mean(np.square(samples, dtype="float64")))) if samples.size else 0.0
    return -120.0 if rms <= 0 else 20.0 * float(np.log10(rms))


def prepare_audio(path: Path, fmt: str, bitrate: str) -> tuple[str, float]:
    """Loudness-match one clip and return (base64 payload, duration seconds).

    Normalisation happens on the samples, before any lossy encode, so both
    systems are levelled by the same rule regardless of what they wrote out.
    """
    import numpy as np
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    gain = 10.0 ** ((TARGET_DBFS - _rms_dbfs(samples)) / 20.0)
    samples = np.clip(samples * gain, -1.0, 1.0)
    duration = len(samples) / float(rate)

    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    wav_bytes = buffer.getvalue()

    if fmt == "wav":
        return base64.b64encode(wav_bytes).decode(), duration

    # mp3 keeps a 400-passage run inside a sane upload budget. It is applied
    # identically to both systems, so it cannot favour either one.
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=True,
    ).stdout
    return base64.b64encode(encoded).decode(), duration


# --- judge call --------------------------------------------------------------


def build_messages(text: str, first_b64: str, second_b64: str, fmt: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEMPLATE.format(text=text.strip())},
                {"type": "text", "text": "Recording 1:"},
                {"type": "input_audio", "input_audio": {"data": first_b64, "format": fmt}},
                {"type": "text", "text": "Recording 2:"},
                {"type": "input_audio", "input_audio": {"data": second_b64, "format": fmt}},
            ],
        },
    ]


def extract_json(text: str) -> dict | None:
    """Pull the verdict out of a reply that may be fenced or cut short.

    A thinking model can spend its whole completion budget before finishing the
    note, leaving valid numbers inside an unterminated object. The two fields
    that carry the verdict are recovered directly in that case; the note is
    decoration.
    """
    body = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M)
    match = re.search(r"\{.*\}", body, re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass
    fields = {
        name: int(found.group(1))
        for name in ("prosody", "correctness")
        if (found := re.search(rf'"{name}"\s*:\s*([012])', body))
    }
    if len(fields) == 2:
        note = re.search(r'"note"\s*:\s*"([^"]*)', body)
        return {**fields, "note": note.group(1) if note else ""}
    return None


def valid_verdict(verdict: dict | None) -> TypeGuard[dict]:
    """Whether the parse produced a verdict, stated as a narrowing.

    A plain `bool` left every caller reading `verdict["prosody"]` off a
    `dict | None`: the check ran, and the checker still had to assume the
    `None`. `TypeGuard` is the same predicate saying what it proves. (tools-07)
    """
    return (
        isinstance(verdict, dict)
        and verdict.get("prosody") in (0, 1, 2)
        and verdict.get("correctness") in (0, 1, 2)
    )


def chat(token: str, messages: list[dict], model: str, *, retries: int = 6) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": messages,
        # A thinking judge spends most of this before it writes the verdict;
        # 800 truncated roughly three replies in four.
        "max_tokens": 3000,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                ENDPOINT,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/loudreader/loudkit",
                    "X-Title": "loudkit pairwise judge",
                },
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                data = json.loads(response.read())
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            if content:
                return content, data.get("usage", {}) or {}
            last = RuntimeError(f"empty completion ({data['choices'][0].get('finish_reason')})")
        except Exception as exc:  # 429 and 5xx are routine here
            last = exc
        if attempt < retries - 1:
            time.sleep(min(120.0, 5.0 * (2**attempt)) + random.uniform(0, 3))
    raise RuntimeError(f"judge call failed after {retries} attempts: {last}")


def judge_once(
    token: str,
    item: dict,
    order: str,
    *,
    model: str,
    fmt: str,
    bitrate: str,
    system_a: str,
    system_b: str,
) -> dict:
    """One judgment in one presentation order.

    ``order`` is "ab" when system A is played first. The verdict comes back as
    a recording number, and is mapped to a system name here so that nothing
    downstream has to remember which slot was which.
    """
    first_path, second_path = (
        (item["a"], item["b"]) if order == "ab" else (item["b"], item["a"])
    )
    first_b64, first_seconds = prepare_audio(Path(first_path), fmt, bitrate)
    second_b64, second_seconds = prepare_audio(Path(second_path), fmt, bitrate)

    content, usage = chat(
        token, build_messages(item["text"], first_b64, second_b64, fmt), model
    )
    verdict = extract_json(content)
    if not valid_verdict(verdict):
        raise RuntimeError(f"unparsable verdict for {item['id']} [{order}]: {content[:200]!r}")

    slots = [system_a, system_b] if order == "ab" else [system_b, system_a]

    def winner(choice: int) -> str:
        return "tie" if choice == 0 else slots[choice - 1]

    return {
        "id": item["id"],
        "order": order,
        "first": slots[0],
        "model": model,
        "prosody_winner": winner(verdict["prosody"]),
        "correctness_winner": winner(verdict["correctness"]),
        "note": str(verdict.get("note", ""))[:400],
        "seconds": {slots[0]: round(first_seconds, 2), slots[1]: round(second_seconds, 2)},
        "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
    }


# --- run ---------------------------------------------------------------------


def load_manifest(path: Path) -> list[dict]:
    items: list[dict] = []
    for raw in path.read_text().splitlines():
        if raw.strip():
            items.append(json.loads(raw))
    missing = [
        f"{item['id']}:{side}"
        for item in items
        for side in ("a", "b")
        if not Path(item[side]).is_file()
    ]
    if missing:
        raise SystemExit(f"manifest references missing audio: {', '.join(missing[:8])}")
    return items


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    done: set[tuple[str, str]] = set()
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        done.add((record["id"], record["order"]))
    return done


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL with id, text, a and b (wav paths) per passage",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSONL of judgments, appended")
    parser.add_argument("--system-a", default="loudkit")
    parser.add_argument("--system-b", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--audio-format", choices=("mp3", "wav"), default="mp3")
    parser.add_argument("--bitrate", default="128k", help="mp3 bitrate when --audio-format mp3")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="judge only the first N passages")
    args = parser.parse_args()

    token = load_token()
    if not token:
        raise SystemExit(
            "no OpenRouter credentials: set OPENROUTER_API_KEY, or sign in with opencode"
        )

    items = load_manifest(args.manifest)
    if args.limit:
        items = items[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_keys(args.out)
    jobs = [
        (item, order)
        for item in items
        for order in ("ab", "ba")
        if (item["id"], order) not in done
    ]
    print(
        f"{len(items)} passages, {len(jobs)} judgments to run "
        f"({len(done)} already in {args.out})",
        flush=True,
    )
    if not jobs:
        return 0

    failures = 0
    spent = 0.0

    def run(job: tuple[dict, str]) -> dict | None:
        item, order = job
        try:
            return judge_once(
                token,
                item,
                order,
                model=args.model,
                fmt=args.audio_format,
                bitrate=args.bitrate,
                system_a=args.system_a,
                system_b=args.system_b,
            )
        except Exception as exc:  # one bad passage must not end the run
            print(f"  ! {item['id']} [{order}]: {exc}", file=sys.stderr, flush=True)
            return None

    with args.out.open("a") as handle, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for index, record in enumerate(pool.map(run, jobs), start=1):
            if record is None:
                failures += 1
                continue
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            spent += float(record["usage"].get("cost") or 0.0)
            if index % 25 == 0 or index == len(jobs):
                print(f"  {index}/{len(jobs)}", flush=True)

    per_call = spent / max(1, len(jobs) - failures)
    print(
        f"wrote {args.out}: {len(jobs) - failures} judgments, ${spent:.2f} "
        f"(${per_call:.4f} each)"
        + (f", {failures} failed - rerun the same command to retry" if failures else ""),
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
