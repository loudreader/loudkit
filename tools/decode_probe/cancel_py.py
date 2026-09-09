"""What each port does when the cancel flag is edge-triggered, in Python.

`python/loudkit/models/generator.py` and `backends/onnx_backend.py` `break` out
of the decode and return the partial row; Go, Rust, JS and Swift discard it and
return an error. Python's one caller, `window.py:173`, re-polls the flag at
`window.py:182` and raises if it is still set, which hides the difference for a
level-triggered flag.

This measures the other case: a flag that is true once and false afterwards,
which is what an edge-triggered "cancel the current utterance" toggle is.

    PYTHONPATH=python python tools/decode_probe/cancel_py.py BUNDLE VOICE TEXT
"""

from __future__ import annotations

import sys

import numpy as np


def main() -> int:
    bundle, voice_path, text = sys.argv[1], sys.argv[2], sys.argv[3]

    import loudkit as lk
    from loudkit.errors import CancelledError
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    print(f"loudkit module: {lk.__file__}", file=sys.stderr)
    eng = lk.load(bundle, device="onnx")
    voice = VoiceProfile.load(voice_path)
    cfg = eng.algorithm
    text_tokens = [int(t) for t in eng.frontend.encode(text, language="en")]

    # Baseline: no cancellation.
    s = LRSamplerV1(cfg.sampling, seed=7)
    full = eng.token_generator.generate(
        np.asarray(text_tokens, dtype=np.int64), voice, sampler=s
    )
    print(f"baseline generate: {len(full)} tokens")

    # 1. The low-level loop with a level-triggered flag (true from step 20 on).
    state = {"n": 0}

    def level() -> bool:
        state["n"] += 1
        return state["n"] > 20

    s = LRSamplerV1(cfg.sampling, seed=7)
    got = eng.token_generator.generate(
        np.asarray(text_tokens, dtype=np.int64), voice, sampler=s, should_cancel=level
    )
    print(f"generate, level-triggered cancel: returned {len(got)} tokens (no exception)")

    # 2. The low-level loop with an edge-triggered flag (true exactly once).
    fired = {"done": False, "n": 0}

    def edge() -> bool:
        fired["n"] += 1
        if fired["n"] == 21 and not fired["done"]:
            fired["done"] = True
            return True
        return False

    s = LRSamplerV1(cfg.sampling, seed=7)
    got2 = eng.token_generator.generate(
        np.asarray(text_tokens, dtype=np.int64), voice, sampler=s, should_cancel=edge
    )
    print(f"generate, edge-triggered cancel: returned {len(got2)} tokens (no exception)")

    # 3. The public API with the same edge-triggered flag. This is the one that
    #    matters: `window.py:182` re-polls, and a flag that has gone false again
    #    lets the truncated row through into the render.
    fired2 = {"done": False, "n": 0}

    def edge2() -> bool:
        fired2["n"] += 1
        if fired2["n"] == 21 and not fired2["done"]:
            fired2["done"] = True
            return True
        return False

    try:
        res = eng.synthesize(text, voice, seed=7, language="en", should_cancel=edge2)
        print(
            f"synthesize, edge-triggered cancel: RETURNED {len(res.tokens)} tokens, "
            f"{len(res.audio)} samples, {len(res.audio) / cfg.sample_rate:.2f}s "
            f"(baseline is {len(full)} tokens) -- no exception raised"
        )
    except CancelledError as exc:
        print(f"synthesize, edge-triggered cancel: raised CancelledError ({exc})")

    # 4. The public API with a level-triggered flag, for the contrast.
    state2 = {"n": 0}

    def level2() -> bool:
        state2["n"] += 1
        return state2["n"] > 20

    try:
        res = eng.synthesize(text, voice, seed=7, language="en", should_cancel=level2)
        print(f"synthesize, level-triggered cancel: RETURNED {len(res.tokens)} tokens")
    except CancelledError:
        print("synthesize, level-triggered cancel: raised CancelledError")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
