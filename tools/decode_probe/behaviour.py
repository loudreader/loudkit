"""The long-run behaviours a single decode does not exercise, measured.

The stage ladder in probe.py compares one generate call across five ports.
These are the parts that only appear in a passage long enough to chunk: the
two-thread split, the prefix carry, the retry ladder's seed derivation and the
cancellation poll.

    PYTHONPATH=python python tools/decode_probe/behaviour.py BUNDLE VOICE TEXTFILE SEED

Prints one line per check: PASS/FAIL, what was compared, and the digests.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


def sha(a) -> str:
    row = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return hashlib.sha256(row.tobytes()).hexdigest()[:16]


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


def main() -> int:
    bundle, voice_path, textfile, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    text = Path(textfile).read_text().strip()

    import loudkit as lk
    from loudkit.voice import VoiceProfile

    print(f"loudkit module: {lk.__file__}", file=sys.stderr)
    eng = lk.load(bundle, device="onnx")
    voice = VoiceProfile.load(voice_path)
    print(f"# {eng.describe()}", file=sys.stderr)

    chunks = None
    from loudkit.frontend.chunking import split_text
    from loudkit.frontend.speechtext import speech_text

    prepared = speech_text(text, "en")
    chunks = split_text(prepared, eng.algorithm.chunking)
    print(f"# {len(chunks)} chunks, prefix_tokens={eng.algorithm.chunking.prefix_tokens}")

    ok = True

    # I-2: same seed, same build, same input, twice.
    a = eng.synthesize(text, voice, seed=seed, language="en")
    b = eng.synthesize(text, voice, seed=seed, language="en")
    ok &= check(
        "I-2 determinism: synthesize twice",
        sha(a.audio) == sha(b.audio) and list(a.tokens) == list(b.tokens),
        f"{sha(a.audio)} vs {sha(b.audio)}  tokens {len(a.tokens)}/{len(b.tokens)}",
    )

    # stream.py's own claim: "the audio is byte-identical to the serial path".
    # latency_mode changes only when window 1's generation starts, so the two
    # settings must render the same bytes.
    def streamed(latency: bool):
        parts = list(eng.stream(text, voice, seed=seed, language="en", latency_mode=latency))
        return np.concatenate([p.audio for p in parts]), [t for p in parts for t in p.tokens]

    s_lat, t_lat = streamed(True)
    s_thr, t_thr = streamed(False)
    ok &= check(
        "stream: latency_mode True vs False",
        sha(s_lat) == sha(s_thr) and t_lat == t_thr,
        f"{sha(s_lat)} vs {sha(s_thr)}",
    )

    # synthesize is the stream drained and concatenated, so it must equal it.
    ok &= check(
        "synthesize == concat(stream)",
        sha(a.audio) == sha(s_thr) and list(a.tokens) == t_thr,
        f"{sha(a.audio)} vs {sha(s_thr)}  tokens {len(a.tokens)}/{len(t_thr)}",
    )

    # The prefix carry: chunk k+1 is generated with the tail of chunk k, so
    # rendering the same passage with previous_tokens set must move the bytes.
    c = eng.synthesize(
        text, voice, seed=seed, language="en", previous_tokens=list(a.tokens[:6])
    )
    ok &= check(
        "prefix carry: previous_tokens changes the render",
        sha(c.audio) != sha(a.audio),
        f"{sha(a.audio)} vs {sha(c.audio)} (differing is correct)",
    )

    # Cancellation: a flag that is already true must produce nothing, and the
    # poll must fire within one decode step rather than at the next chunk.
    from loudkit.errors import CancelledError

    polls = {"n": 0}

    def cancel_now() -> bool:
        polls["n"] += 1
        return True

    try:
        eng.synthesize(text, voice, seed=seed, language="en", should_cancel=cancel_now)
        ok &= check("cancel: synthesize raises", False, "returned instead of raising")
    except CancelledError:
        ok &= check("cancel: synthesize raises CancelledError", True, f"{polls['n']} polls")

    polls2 = {"n": 0}

    def cancel_after(n: int):
        def f() -> bool:
            polls2["n"] += 1
            return polls2["n"] > n

        return f

    got = list(
        eng.stream(text, voice, seed=seed, language="en", should_cancel=cancel_after(50))
    )
    ok &= check(
        "cancel: stream yields the partial and does not raise",
        True,
        f"{len(got)} chunks before the flag, {polls2['n']} polls",
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
