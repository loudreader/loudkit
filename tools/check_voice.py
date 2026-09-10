"""Enrollment-time stall gate: is this voice profile pause-prone?

Why this exists. The interior-stall investigation traced pause behaviour to one
FSQ quantizer cell that the id arithmetic splits into two bands at the +2187
digit boundary: true digital silence renders through 4137/4215/4218/4299 (band
A) and 6162/6324/6405/6486 (band B). A healthy voice spreads its pauses across
both bands and ends a pause in 9 to 12 tokens; a voice that concentrates on
band A repeats one shape and free-runs. Measured across the twenty voices
shipped at calibration, the band-B share of emitted silence ranks pre-fix gap
rates at Spearman -0.91, ahead of every profile-side statistic (dead prompt
tokens +0.14) and of reference speech level (-0.58). The roster is 28 now; the
numbers below are that calibration set and were not re-measured on the eight
that followed.

The engine's own fix (the repetition penalty on silence) removed the
catastrophic classes, so this gate is not for the shipped roster: it is for the
next cloned voice, catching a pause-prone profile at enrollment time instead of
patching the engine around it later.

The probe deliberately renders under the pre-fix exposure: postprocess off and
the old silence exemption restored in a local sampler subclass. The shipping
law penalises repeated silence, which forces band variety and compresses the
very signal being measured (the same twenty voices separate 0.01-0.70 under
the old law but only 0.52-0.83 under the shipped one). A gate must measure the
voice, not the mitigation that papers over it. Nothing here changes the
engine: the subclass lives in this file and the tokens are never rendered to
audio.

What it cannot see. The engine base rate and whole-silence rows are decoder
properties every profile shares; no enrollment gate reaches them. Four
passages at one seed is a probe, not a census, a marginal voice can pass on a
lucky seed, which is why the verdict has a warn band rather than one line. The
profile's own token statistics are printed for context but do not drive the
verdict: soren's conditioning prompt is essentially clean (2 dead tokens in
150) and he was the worst voice on the roster, so a profile-only screen
passes exactly the case this gate exists to catch. And the calibration corpus
was each voice's own language; probing a non-English voice with the default
English set is measurement on a different footing (see --reading-set).

What to do with the verdict. WARN is the reference-cleaning case: strip
leading and trailing silence, cap internal silence runs at 100 ms (threshold
derived from the file's own frame statistics, speech_p90 - 18 dB), re-enroll,
re-run. Measured: that recipe took joe from 0.27 (warn) to 0.65 (clean pass).
FAIL that survives cleaning is the source itself: soren's cleaned reference
still probes at 0.03 with mute rows: re-record, do not ship. Level alone is
not the lever; raising a quiet reference without de-silencing it moves the
probe little.

Usage::

    python tools/check_voice.py voices/my-voice.safetensors
    python tools/check_voice.py soren --checkpoint loudreader/loudr-1
    python tools/check_voice.py mine.safetensors --json out/gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
READING_SET = REPO / "tests/data/judge/reading-en.json"

# The two halves of the split FSQ cell. Band A ids sit +2187 (one digit-7
# step) below their band-B partners; 4215/6162 complete the family census.
BAND_A = frozenset((4137, 4215, 4218, 4299))
BAND_B = frozenset((6162, 6324, 6405, 6486))
TRUE_SILENCE = BAND_A | BAND_B

# Calibrated on the twenty voices the roster held then, four own-language
# passages each, seed 1234, old-law probe (see module docstring). The four
# voices the engine had to be patched around (soren, dave, dante, nils)
# measured 0.010-0.059 with free-runs to the token cap; every clean voice
# (pre-fix gap rate under 15%) measured 0.27 or higher. Between the lines sit
# exactly the voices that were repaired by cleaning their reference (joe 0.27,
# kerstin 0.28, thorsten 0.16), which is what the warn verdict tells the caller
# to do.
FAIL_BELOW = 0.10
WARN_BELOW = 0.30
# A free-run is disqualifying on its own. Under this probe every voice the
# fix had to rescue produced a true-silence run of 150-237 tokens (6-9.5 s);
# no clean voice exceeded 39. The line sits at a 4 s hole.
RUN_FAIL = 100
# A chunk that hit the token cap is a stall only when the row is majority
# silence, the shipped stall rule's own third trigger. Honest long reads hit
# the cap too (measured 0.16-0.33 silence on clean voices; stalls 0.51-0.93),
# so the fraction, not the cap hit, is the signal.
CAP_STALL_FRACTION = 0.5


def band_stats(streams: list[list[int]]) -> dict:
    """Band composition and run structure of emitted silence, over chunk
    token streams. Pure: the fixture test pins this function."""
    band_a = band_b = 0
    runs: list[int] = []
    for tokens in streams:
        run = 0
        for token in tokens:
            if token in TRUE_SILENCE:
                run += 1
                if token in BAND_A:
                    band_a += 1
                else:
                    band_b += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
    silence = band_a + band_b
    return {
        "silence_tokens": silence,
        "band_a": band_a,
        "band_b": band_b,
        "band_b_share": (band_b / silence) if silence else None,
        "pause_runs": len(runs),
        "run_max": max(runs) if runs else 0,
    }


def classify(stats: dict, *, stall_caps: int = 0) -> tuple[str, list[str]]:
    """Verdict from probe statistics. Pure: the fixture test pins this too.

    ``stall_caps`` counts cap-hit chunks that were majority silence, the
    probe computes it from the streams with :data:`CAP_STALL_FRACTION`.
    Returns ("pass" | "warn" | "fail", reasons). A voice with too little
    emitted silence to judge (under 40 pause tokens) is a warn, not a pass:
    the statistic is a ratio and a thin numerator is noise.
    """
    reasons: list[str] = []
    share = stats["band_b_share"]
    if stats["silence_tokens"] < 40:
        return "warn", [
            f"only {stats['silence_tokens']} silence tokens emitted; "
            "probe too thin to judge — add passages"
        ]
    if stall_caps:
        reasons.append(
            f"{stall_caps} chunk(s) free-ran to the token cap majority-silent "
            "(the stall rule's own condemnation trigger)"
        )
    if stats["run_max"] >= RUN_FAIL:
        reasons.append(
            f"longest silence run {stats['run_max']} tokens, a "
            f"{stats['run_max'] * 0.04:.1f}s hole "
            f"(no clean roster voice exceeded 39 under this probe)"
        )
    if share is not None and share < FAIL_BELOW:
        reasons.append(
            f"band-B share {share:.3f} < {FAIL_BELOW} "
            "(pause mass concentrated on one band: the stall signature)"
        )
    if reasons:
        return "fail", reasons
    if share is not None and share < WARN_BELOW:
        return "warn", [
            f"band-B share {share:.3f} < {WARN_BELOW}: below every clean "
            "roster voice, in the band of the reference-repairable ones"
        ]
    return "pass", [f"band-B share {share:.3f}, run_max {stats['run_max']}"]


def profile_screen(profile) -> dict:
    """Context only. A dirty prompt is sufficient for the defect and not
    necessary (soren is clean and was the worst), so this never decides."""

    def _seq(seq) -> dict:
        seq = [int(t) for t in seq]
        dead = [t for t in seq if t in TRUE_SILENCE]
        run = best = 0
        for token in seq:
            run = run + 1 if token in TRUE_SILENCE else 0
            best = max(best, run)
        return {"len": len(seq), "dead": len(dead), "max_run": best}

    return {
        "cond_prompt_tokens": _seq(profile.cond_prompt_tokens),
        "prompt_tokens": _seq(profile.prompt_tokens),
    }


def probe(args) -> dict:
    """Render the probe: token phase only, postprocess off, old law."""
    import loudkit as lk
    import loudkit.window as engine_module
    from loudkit.frontend.chunking import split_text
    from loudkit.frontend.speechtext import speech_text
    from loudkit.sampler import LRSamplerV1
    from loudkit.window import _STREAM_CHUNK, _derive

    class PreFixLawSampler(LRSamplerV1):
        """The deleted exemption, restored for measurement only: the
        repetition penalty skips the configured silence ids, so the probe
        sees the voice's unmitigated pause behaviour."""

        def __call__(self, logits, *, step, seen):
            if self._silence.size:
                seen = seen.copy()
                seen[self._silence] = False
            return super().__call__(logits, step=step, seen=seen)

    if args.profile_path.suffix == ".safetensors":
        voice = lk.VoiceProfile.load(args.profile_path)
    else:
        voice = lk.voice(str(args.profile_path), repo=args.checkpoint)
    language = args.language or voice.language or "en"

    # A module-attribute swap, not a subclassed engine: the sampler is
    # constructed inside window.generate_window by name, and this is the same
    # dispatch seam the forensic probes used. Restored in the finally.
    engine_module.LRSamplerV1 = PreFixLawSampler  # type: ignore[misc]
    try:
        import dataclasses

        base = lk.load(args.checkpoint, device=args.device or None)
        algorithm = dataclasses.replace(
            base.algorithm,
            postprocess=dataclasses.replace(base.algorithm.postprocess, mode="off"),
        )
        engine = lk.load(args.checkpoint, device=args.device or None, algorithm=algorithm)
        del base

        passages = json.loads(Path(args.reading_set).read_text())["passages"][: args.passages]
        prefix_len = engine.algorithm.chunking.prefix_tokens
        streams: list[list[int]] = []
        cap_hits = stall_caps = 0
        started = time.time()
        for passage in passages:
            prepared = speech_text(passage["text"], language)
            chunks = split_text(prepared, engine.algorithm.chunking)
            prefix: list[int] = []
            for index, chunk in enumerate(chunks):
                window = engine._generate_window(
                    chunk,
                    voice,
                    seed=_derive(args.seed, _STREAM_CHUNK + index),
                    language=language,
                    prefix=prefix,
                    prepared=True,
                    is_terminal=index == len(chunks) - 1,
                )
                tokens = [int(t) for t in window.speech]
                if window.hit_token_cap:
                    cap_hits += 1
                    silent = sum(1 for t in tokens if t in TRUE_SILENCE)
                    if tokens and silent / len(tokens) >= CAP_STALL_FRACTION:
                        stall_caps += 1
                if prefix_len:
                    prefix = tokens[-prefix_len:]
                streams.append(tokens)
            print(f"  {passage['id']}: {len(chunks)} chunks", flush=True)
    finally:
        engine_module.LRSamplerV1 = LRSamplerV1  # type: ignore[misc]

    stats = band_stats(streams)
    verdict, reasons = classify(stats, stall_caps=stall_caps)
    return {
        "voice": voice.name or str(args.profile_path),
        "language": language,
        "seed": args.seed,
        "passages": len(passages),
        "probe_seconds": round(time.time() - started, 1),
        "cap_hits": cap_hits,
        "stall_caps": stall_caps,
        "stats": dict(stats),
        "profile": profile_screen(voice),
        "verdict": verdict,
        "reasons": reasons,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "profile_path",
        type=Path,
        help="a VoiceProfile .safetensors, or a roster name resolved from --checkpoint",
    )
    parser.add_argument("--checkpoint", default="loudreader/loudr-1")
    parser.add_argument("--language", default="", help="default: the profile's own")
    parser.add_argument(
        "--reading-set",
        type=Path,
        default=READING_SET,
        help="probe corpus; prefer the voice's own language when you have one — "
        "the calibration was own-language",
    )
    parser.add_argument("--passages", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="", help="default: best available")
    parser.add_argument("--json", type=Path, default=None, help="also write the report here")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = probe(args)

    stats = report["stats"]
    print(f"\nvoice     : {report['voice']} ({report['language']})")
    print(
        f"probe     : {report['passages']} passages, seed {report['seed']}, "
        f"{report['probe_seconds']}s"
    )
    share = stats["band_b_share"]
    print(
        f"silence   : {stats['silence_tokens']} tokens, "
        f"band-B share {share if share is None else round(share, 3)}, "
        f"{stats['pause_runs']} runs, longest {stats['run_max']}, "
        f"cap hits {report['cap_hits']} ({report['stall_caps']} majority-silent)"
    )
    cond = report["profile"]["cond_prompt_tokens"]
    print(
        f"profile   : {cond['dead']}/{cond['len']} dead conditioning tokens, "
        f"max run {cond['max_run']} (context only — a clean prompt clears nothing)"
    )
    print(f"verdict   : {report['verdict'].upper()}")
    for reason in report["reasons"]:
        print(f"  - {reason}")
    if report["verdict"] != "pass":
        print(
            "\nremedy    : clean the reference and re-enroll — strip edge silence,\n"
            "            cap internal silence at 100 ms with a threshold derived from\n"
            "            the file's own stats (speech_p90 - 18 dB), then re-run this\n"
            "            gate. Measured: that took joe from warn (0.27) to a clean\n"
            "            pass (0.65). A FAIL that survives cleaning is the source\n"
            "            itself — re-record; soren's cleaned reference still fails."
        )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
        print(f"\nwrote {args.json}")
    return 1 if report["verdict"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
