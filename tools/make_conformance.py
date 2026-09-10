"""Generate the cross-language conformance fixture.

One directory (``tests/data/conformance``) that the Python suite and the Swift
package's ``swift test`` both read, covering the layers where two independent
implementations of the same law could drift:

  philox       the three Random123 KAT vectors, raw uniform *bits* for fixed
               counters (exact, integer), and gumbel probe values (float64,
               compared to 1e-12, two libms may differ in the last ulp)
  sampler      LR-SAMPLER-v1 token choices: small hand-set logits, a
               silence-exemption case, and a full-vocab case whose logits are
               *derived from Philox bits* so neither language has to ship an
               8194-float table to agree on what the input was
  frontend     text -> token ids, en and pl, punctuation / diacritics /
               whitespace traps; the tokenizer JSON is copied in so this layer
               is checkable with no weights at all
  algorithm    the production fingerprint and the exact canonical JSON blob it
               hashes, a fingerprint mismatch with a matching blob means the
               hash is wrong, a blob mismatch names the drifted field
  end_to_end   text + voice + seed -> speech tokens (exact), mel and waveform
               (banded), rendered by the coreml backend with the token
               generator declared fp32, the precision the Swift generator
               runs, because "same precision, same tokens" is the contract
               (engine.synthesize docstring) and fp16-vs-fp32 token identity
               on these sentences is an observation, not a promise
  long_form    a passage that does *not* fit one window: the funnel, the split,
               the per-chunk seed, the carried prefix and the exact token
               stream of EVERY chunk
  eos_peak     the stop-token observation the postprocess detectors compare
               against a threshold, pinned because it is hand-written in five
               languages and a port that computes it differently cuts a chunk
               somewhere else
  seeds        the derived 64-bit seed for a (seed, stream) pair, and the
               stream numbers the flow and the vocoder draw from
  resplit      a chunk that had to be split again: the same chain constants
               ``long_form`` carries, plus the stream the second half draws
               from; the wiring around ``split_in_half`` was held by nothing

``--fixtures-only`` writes a second, weight-free set beside these: the seven
shared fixtures (fetch plan, ``fnmatch`` semantics, repo-id recognition, the
release receipt, the cache path, PCM16 quantisation and the WAV header) that
replace hand-transcribed tests in the four downloaders and WAV writers. They
know nothing about weights; see ``shared_fixtures``.

``long_form`` is a separate section rather than another ``end_to_end`` entry
because every port loops over that array expecting one text, one token stream
and one pair of reference renders. What it covers is the half of the engine the
single-sentence cases cannot reach: with an empty prefix ``len(prefix) + step +
1`` and ``step + 1`` are the same expression and an unseeded repetition mask is
the seeded one, so three ports indexed the speech positional table by ``step +
1`` and two of them started the mask blind, and the fixture passed throughout.
A carried prefix separates the two expressions; it is asserted per chunk rather
than on the concatenation so a diverged chunk names itself instead of shifting
everything after it.

Tokens here are the generator's own, without the postprocess trim the shipping
path applies to the terminal chunk, exactly like ``end_to_end``, and for the
same reason: the contract being pinned is "this text, this voice, this seed,
this prefix -> these tokens", which is the layer the ports reimplement.
`public_tokens` separately records the shipping pipeline after terminal trimming.

Additional decode modes write a separate model fixture referencing vectors.json
for shared weight-free cases. --render-device chooses the reference renderer.

Numbers that do not fit a JSON double exactly (derived 64-bit seeds) are
stored as hex strings. Mel and waveform go beside the JSON as raw little-
endian float32 (shapes in the JSON) so Swift needs no npy parser.

Usage (regenerates everything; run when the engine legitimately changes):
  .venv/bin/python tools/make_conformance.py \
      --checkpoint /path/to/loudr-1.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.config import Device, Precision, SamplingConfig
from loudkit.models.flow import time_grid
from loudkit.rng import KAT_VECTORS, uniforms
from loudkit.sampler import LRSamplerV1
from loudkit.window import (
    _STREAM_CHUNK,
    _STREAM_RESPLIT,
    _derive,
    carry_pair_aligned,
)

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "conformance"

E2E_SENTENCES = [
    ("s0", "The quick brown fox jumps over the lazy dog.", "en", 4242),
    ("s2", "Wait — was that a knock at the door, or only the wind?", "en", 7),
]

E2E_RESPLIT = [
    (
        # One passage under a 56-token window, which the engine can actually
        # run: `max_speech_tokens`, `sampling.max_new_tokens` and
        # `chunking.max_tokens` all move together, because `AlgorithmConfig`
        # refuses a chunk budget larger than the window. Thirteen chunks, four
        # of which overrun and are repaired, nine untouched.
        #
        # An earlier version declared only a token ceiling and left the window
        # at 255. The engine gates the re-split on `len(speech) >=
        # window.max_speech_tokens`, so that fixture pinned a path the engine
        # could not take -- the test reimplemented the orchestration and agreed
        # with itself. What weights cannot reach here is the no-recursion
        # branch: a valid config gives a chunk at most `window * 0.5`
        # characters, so a half is a quarter of the window and cannot overrun.
        # That branch is covered by `tests/test_engine.py` with a fake
        # generator, where the cap is controllable.
        #
        # The ceiling is declared rather than left to the voice's pace. Pace is
        # data and the reference voice's is nobody's contract; what five
        # implementations must agree on is the law -- which index each window
        # carries, which stream the second half draws from, and where the carry
        # comes off. Forcing the cap makes that reachable without a slow voice.
        "rs0",
        "This is a longer passage, written to exercise more than a single window of "
        "speech tokens. It should run through several chunks and a couple of joins, "
        "so the streaming path and the long-form path are both measured, not just the "
        "shortest sentence that is fastest to type.",
        "en",
        1234,
        56,
    ),
]

E2E_LONG_FORM = [
    (
        "lf0",
        # The passage the divergence was found on (`loudkit.bench.DEFAULT_TEXTS[2]`
        # at the time of writing), spelled out rather than imported: a benchmark
        # constant is free to change and a conformance expectation is not.
        # 265 characters against a 127-character budget, so it splits three ways
        # under the shipping recipe and the last two chunks carry a prefix.
        "This is a longer passage, written to exercise more than a single window of "
        "speech tokens. It should run through several chunks and a couple of joins, "
        "so the streaming path and the long-form path are both measured, not just the "
        "shortest sentence that is fastest to type.",
        "en",
        1234,
    ),
]

CONFORMANCE_EXECUTION: dict[str, Precision] = {
    "token_generator": "fp32",
    "mel_decoder.estimator": "fp16",
    "mel_decoder.encoder": "fp32",
    "vocoder": "fp32",
}


def philox_section() -> dict:
    kat = [
        {"counter": list(ctr), "key": list(key), "expected": list(want)}
        for ctr, key, want in KAT_VECTORS
    ]
    bit_probes = []
    for seed, stream, step0, n_steps, width in [
        (0, 0, 0, 2, 8),
        (7, 0, 0, 1, 12),
        (0xDEADBEEF, 2, 300, 2, 6),
        (2**63 + 11, 1, 0, 1, 8),  # exercises the high seed word
    ]:
        u = uniforms(seed, stream, step0, n_steps, width)
        bits = np.round(u * 4294967296.0 - 0.5).astype(np.uint64)
        bit_probes.append(
            {
                "seed": hex(seed),
                "stream": stream,
                "step0": step0,
                "n_steps": n_steps,
                "width": width,
                "bits": [[int(b) for b in row] for row in bits],
            }
        )
    from loudkit.rng import gumbel_noise

    gumbel = []
    for seed, stream, step, width in [(0, 0, 0, 8), (4242, 0, 17, 8)]:
        g = gumbel_noise(seed, stream, step, 1, width)[0]
        gumbel.append(
            {
                "seed": seed,
                "stream": stream,
                "step": step,
                "width": width,
                "values": [float(v) for v in g],
                "rtol": 1e-12,
            }
        )
    return {"kat": kat, "uniform_bits": bit_probes, "gumbel": gumbel}


def _run_sampler(cfg: SamplingConfig, seed: int, logits_rows: list[np.ndarray]) -> list[int]:
    sampler = LRSamplerV1(cfg, seed=seed)
    vocab = logits_rows[0].shape[0]
    seen = np.zeros(vocab, dtype=bool)
    out = []
    for step, row in enumerate(logits_rows):
        tok = sampler(row, step=step, seen=seen)
        out.append(tok)
        seen[tok] = True
    return out


def sampler_section() -> dict:
    cases = []

    # 1. small vocab, literal logits, default law
    cfg = SamplingConfig(temperature=0.8, repetition_penalty=1.2, min_p=0.05)
    logits = [
        np.array(
            [
                0.1,
                2.5,
                -1.0,
                3.0,
                0.0,
                1.5,
                -2.0,
                2.9,
                0.5,
                1.0,
                -0.5,
                2.0,
                0.25,
                1.75,
                -1.5,
                2.75,
            ],
            dtype=np.float32,
        )
    ] * 10
    cases.append(
        {
            "name": "tiny_default",
            "config": {
                "temperature": 0.8,
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "silence_token_ids": [],
            },
            "seed": 7,
            "logits": [[float(v) for v in row] for row in logits[:1]],
            "repeat_logits": 10,
            "expected": _run_sampler(cfg, 7, logits),
        }
    )

    # 2. silence exemption: ids 3 and 5 exempt from min_p (and, like every
    # other token since the interior-stall fix, penalised on repetition)
    cfg2 = SamplingConfig(
        temperature=0.8, repetition_penalty=1.2, min_p=0.05, silence_token_ids=(3, 5)
    )
    cases.append(
        {
            "name": "silence_exempt",
            "config": {
                "temperature": 0.8,
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "silence_token_ids": [3, 5],
            },
            "seed": 21,
            "logits": [[float(v) for v in row] for row in logits[:1]],
            "repeat_logits": 10,
            "expected": _run_sampler(cfg2, 21, logits),
        }
    )

    # 3. full vocabulary, logits derived from Philox bits: both languages
    # regenerate the identical float32 input without a table in the fixture
    cfg3 = SamplingConfig(
        temperature=0.8, repetition_penalty=1.2, min_p=0.05, silence_token_ids=(1731, 4254)
    )
    vocab, steps = 8194, 24
    rows = []
    for step in range(steps):
        u = uniforms(777, 9, step, 1, vocab)[0]
        rows.append((u * 20.0 - 10.0).astype(np.float32))
    cases.append(
        {
            "name": "full_vocab_philox_logits",
            "config": {
                "temperature": 0.8,
                "repetition_penalty": 1.2,
                "min_p": 0.05,
                "silence_token_ids": [1731, 4254],
            },
            "seed": 4242,
            "logits_recipe": {
                "seed": 777,
                "stream": 9,
                "scale": 20.0,
                "offset": -10.0,
                "vocab": vocab,
                "steps": steps,
            },
            "expected": _run_sampler(cfg3, 4242, rows),
        }
    )
    return {"cases": cases}


def eos_peak_section() -> dict:
    """The stop-token observation the postprocess layer reads.

    Pinned across languages because it is hand-written in five of them and it
    is *audible*: two of the detector rules compare it against a threshold, so a
    port that computes it differently cuts a chunk somewhere else. The quantity
    has two subtleties either of which a reimplementation gets wrong silently,
    the numerator is the stop token's weight taken **before** the ``min_p``
    cutoff, and the peak is recorded only **past** the floor.

    Logits come from the same Philox recipe the sampler cases use, so a port
    reproduces the inputs rather than carrying a megabyte of floats.
    """
    from loudkit.rng import uniforms

    cases = []
    stop = 6562
    for name, floor, silence in (
        ("floor_10", 10, (1731, 4254)),
        ("floor_0_no_silence", 0, ()),
        ("floor_20", 20, (1731, 4254)),
    ):
        cfg = SamplingConfig(
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            silence_token_ids=silence,
        )
        vocab, steps = 8194, 32
        sampler = LRSamplerV1(cfg, seed=4242, stop_token=stop, eos_floor=floor)
        seen: NDArray[np.bool_] = np.zeros(vocab, dtype=bool)
        for step in range(steps):
            u = uniforms(777, 9, step, 1, vocab)[0]
            row = (u * 20.0 - 10.0).astype(np.float32)
            tok = sampler(row, step=step, seen=seen)
            seen[tok] = True
        at, prob = sampler.eos_peak
        cases.append(
            {
                "name": name,
                "config": {
                    "temperature": 0.8,
                    "repetition_penalty": 1.2,
                    "min_p": 0.05,
                    "silence_token_ids": list(silence),
                },
                "seed": 4242,
                "stop_token": stop,
                "eos_floor": floor,
                "logits_recipe": {
                    "seed": 777,
                    "stream": 9,
                    "scale": 20.0,
                    "offset": -10.0,
                    "vocab": vocab,
                    "steps": steps,
                },
                "expected_at": at,
                "expected_prob": prob,
            }
        )
    # A relative tolerance rather than equality: the value is a sum of 8194
    # exponentials, and a port is free to accumulate it in a different order.
    # Tight enough that a wrong *formula* (numerator after the cutoff, floor off
    # by one) fails by orders of magnitude.
    return {"cases": cases, "prob_rtol": 1e-9}


def frontend_section(tokenizer_path: Path) -> dict:
    from loudkit.frontend.text import GraphemeTextFrontend

    shutil.copyfile(tokenizer_path, OUT / "tokenizer.json")
    frontend = GraphemeTextFrontend(OUT / "tokenizer.json")
    texts = [
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("Wait — was that a knock at the door, or only the wind?", "en"),
        ("Hello,   world!\tTabs and  double  spaces.", "en"),
        ("Zażółć gęślą jaźń — pchnąć w tę łódź jeża.", "pl"),
        ("Ël naïve façade: 3.14, 100%!", "en"),
    ]
    return {
        "tokenizer": "tokenizer.json",
        "cases": [
            {"text": t, "language": lang, "ids": [int(i) for i in frontend.encode(t, lang)]}
            for t, lang in texts
        ],
    }


def seed_section() -> dict:
    probes = []
    for seed, stream in [(0, 1), (0, 2), (4242, 1), (4242, 2), (99, 1), (99, 2)]:
        probes.append({"seed": seed, "stream": stream, "derived": hex(_derive(seed, stream))})
    return {"derivation": probes, "streams": {"flow": 1, "vocoder": 2}}


def _tokens_phase(checkpoint: str, out_json: str) -> None:
    """Child-process phase: free-run the fp32 token generator on the torch cpu
    backend and write the stripped speech tokens.

    The child isolates the CPU token reference from native renderer sessions.
    """
    import loudkit
    from loudkit.config import ExecutionConfig
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    # The fixture is the CPU-to-CPU floor every port is held to, so the
    # provider is pinned rather than resolved. `auto` picks CoreML on this
    # laptop, and CoreML moves the token stream (docs/benchmarks.md).
    execution = ExecutionConfig(
        device="cpu", precision=CONFORMANCE_EXECUTION, onnx_provider="cpu"
    )
    engine = loudkit.load(checkpoint, device="cpu", execution=execution)
    voice = VoiceProfile.load(
        Path(__file__).resolve().parent.parent
        / "tests/data/reference/testvoice.voice.safetensors"
    )
    out: dict = {}
    for name, text, lang, seed in E2E_SENTENCES:
        text_tokens = engine.frontend.encode(text, lang)
        sampler = LRSamplerV1(engine.algorithm.sampling, seed=seed)
        raw = list(engine.token_generator.generate(text_tokens, voice, sampler=sampler))
        assert engine.algorithm.stop_speech_token in raw, f"{name} hit the token cap"
        out[name] = {"raw": raw}
    out["__long_form__"] = _long_form_chains(engine, voice)
    out["__resplit__"] = _resplit_chains(engine, voice)
    # `newline="\n"`: five implementations read these bytes.
    with open(out_json, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f)


def _long_form_chains(engine, voice) -> list[dict]:
    """Walk the shipping chunk chain for each long-form case and record it.

    The chain, in the order the engine walks it: run the speech funnel on the
    whole passage *before* splitting (Polish respelling changes the length, so
    a budget computed after it would be a budget for text nobody speaks), split
    at the chunking recipe, then seed chunk 0 with the caller's seed itself and
    every later chunk from ``_STREAM_CHUNK + index``, carrying the last
    ``prefix_tokens`` speech tokens of the chunk before it. Chunk 0's raw seed
    is what makes this section and ``end_to_end`` one law rather than two: a
    text that fits one window is chunk 0 alone.

    Three properties are asserted rather than hoped for, because a case that
    loses any of them stops covering what it was added for:

    * **more than one chunk**, with an empty prefix the two indexing bugs this
      section exists to catch are unobservable;
    * **every chunk ends on the stop token**, a chunk that ran into the cap
      pins a truncation, and the cap is a different fact from the token stream;
    * **at least one carried token outside the silence manifest**, kept from
      the era when the sampler exempted silence ids from the repetition
      penalty: a tail made only of them made a seeded mask and a blind one
      pick the same token, which is precisely how the unseeded-mask defect
      survived the existing fixture. The penalty covers silence now, but a
      mixed carry still discriminates a port that seeds its mask from the
      prefix from one that starts blind, so the property stays asserted.
    """
    from loudkit.frontend.chunking import split_text
    from loudkit.frontend.speechtext import speech_text

    algo = engine.algorithm
    prefix_tokens = algo.chunking.prefix_tokens
    silence = set(algo.sampling.silence_token_ids)
    chains = []
    for name, text, lang, seed in E2E_LONG_FORM:
        prepared = speech_text(text, lang)
        texts = split_text(prepared, algo.chunking)
        assert len(texts) > 1, f"{name} fits one window: it cannot carry a prefix"
        carry: list[int] = []
        chunks: list[dict] = []
        # Accumulated as we go rather than flattened out of `chunks`: the chunk
        # records are heterogeneous dicts and picking the token lists back out
        # of them is a cast, not a read.
        flat: list[int] = []
        for index, chunk_text in enumerate(texts):
            chunk_seed = seed if index == 0 else _derive(seed, _STREAM_CHUNK + index)
            text_tokens = engine.frontend.encode(chunk_text, lang)
            sampler = LRSamplerV1(algo.sampling, seed=chunk_seed)
            raw = list(
                engine.token_generator.generate(
                    text_tokens, voice, sampler=sampler, prefix=carry
                )
            )
            assert algo.stop_speech_token in raw, f"{name} chunk {index} hit the token cap"
            speech = [int(t) for t in raw if int(t) < algo.start_speech_token]
            chunks.append(
                {
                    "index": index,
                    "text": chunk_text,
                    "seed": hex(chunk_seed),
                    "prefix": list(carry),
                    "tokens": speech,
                }
            )
            flat.extend(speech)
            carry = list(carry_pair_aligned(algo, speech, prefix_tokens))
            if index + 1 < len(texts):
                assert any(t not in silence for t in carry), (
                    f"{name} chunk {index} carries only silence ids {carry}: a "
                    "mixed carry is what makes a seeded repetition mask "
                    "distinguishable from a blind one in every port"
                )
        chains.append(
            {
                "name": name,
                "text": text,
                "language": lang,
                "seed": seed,
                "prepared": prepared,
                "chunks": chunks,
                "tokens": flat,
                "public_tokens": [
                    int(t)
                    for t in engine.synthesize(text, voice, seed=seed, language=lang).tokens
                ],
            }
        )
    return chains


def _resplit_chains(engine, voice) -> list[dict]:
    """Walk the `cap_resplit` chain and record every window it produces.

    `long_form` deliberately asserts that no chunk hits the cap, because a
    truncation is a different fact from a token stream. This section pins the
    opposite case, and it is the only thing that holds the four ports to the
    *law* rather than to the helper: `split_in_half` is covered by unit tests in
    five languages, while the engine wiring around it -- the work queue, the
    index both halves keep, the stream the second half draws from, the window
    the carry comes off -- was covered by nothing. Two defects in that wiring
    were found by reading it, not by running it.

    Mirrors `Engine._windows_for_chunk` step for step. A port reproduces this
    array by running its own re-split, so a divergence names the window it
    happened on instead of shifting every seed after it.
    """
    from dataclasses import replace

    from loudkit.frontend.chunking import split_in_half, split_text
    from loudkit.frontend.speechtext import speech_text

    def run(
        chunk_text: str,
        window_seed: int,
        prefix: list[int],
        *,
        algo,
        lang: str,
        window: int,
    ) -> tuple[list[int], bool]:
        """One window, generated exactly as the engine generates it.

        Defined outside the loop and handed `algo`, `lang` and `window`
        explicitly rather than closing over them: a nested function that
        captures a loop variable reads the value the variable has when it is
        *called*, and this generator writes a fixture five ports are held to.
        The call happens to be synchronous today, so the capture was harmless
        today; making the binding explicit costs nothing and removes a way for
        a later edit to move somebody else's fixture.
        """
        ids = engine.frontend.encode(chunk_text, lang)
        sampler = LRSamplerV1(algo.sampling, seed=window_seed)
        raw = list(
            engine.token_generator.generate(
                ids, voice, sampler=sampler, prefix=prefix, max_new_tokens=window
            )
        )
        capped = algo.stop_speech_token not in raw
        return [int(t) for t in raw if int(t) < algo.start_speech_token], capped

    base = engine.algorithm
    chains = []
    for name, text, lang, seed, window in E2E_RESPLIT:
        # One knob, three fields: the config refuses a chunk budget larger than
        # the window, and the sampler's own cap has to move with it or the
        # window is never what stops a row.
        algo = replace(
            base,
            window=replace(base.window, max_speech_tokens=window, static_length=window),
            sampling=replace(base.sampling, max_new_tokens=window),
            chunking=replace(base.chunking, max_tokens=window),
        )
        prefix_tokens = algo.chunking.prefix_tokens
        prepared = speech_text(text, lang)
        texts = split_text(prepared, algo.chunking)
        carry: list[int] = []
        windows: list[dict] = []
        here = {"algo": algo, "lang": lang, "window": window}

        for index, chunk_text in enumerate(texts):
            chunk_seed = seed if index == 0 else _derive(seed, _STREAM_CHUNK + index)
            speech, capped = run(chunk_text, chunk_seed, list(carry), **here)
            # The *window* has to be what stopped it, not the length-proportional
            # ceiling, exactly as every engine gates it. A fixture that pinned a
            # broader rule than the engines run would hold the ports to a law
            # nobody executes, which is worse than no fixture at all.
            hit_the_window = len(speech) >= algo.window.max_speech_tokens
            halves = split_in_half(chunk_text) if capped and hit_the_window else None
            if halves is None:
                windows.append(
                    {
                        "index": index,
                        "text": chunk_text,
                        "seed": hex(chunk_seed),
                        "prefix": list(carry),
                        "tokens": speech,
                        "split": False,
                        "hit_cap": capped,
                    }
                )
                carry = list(carry_pair_aligned(algo, speech, prefix_tokens))
                continue
            first_text, second_text = halves
            # Both halves keep `index`: a repair must not move a later chunk's
            # seed, or a passage that was fine before stops sounding the same.
            first, first_capped = run(first_text, chunk_seed, list(carry), **here)
            windows.append(
                {
                    "index": index,
                    "text": first_text,
                    "seed": hex(chunk_seed),
                    "prefix": list(carry),
                    "tokens": first,
                    "split": True,
                    "hit_cap": first_capped,
                }
            )
            second_prefix = (
                list(carry_pair_aligned(algo, first, prefix_tokens))
                if prefix_tokens > 0
                else list(carry)
            )
            second_seed = _derive(chunk_seed, _STREAM_RESPLIT)
            second, second_capped = run(second_text, second_seed, list(second_prefix), **here)
            windows.append(
                {
                    "index": index,
                    "text": second_text,
                    "seed": hex(second_seed),
                    "prefix": list(second_prefix),
                    "tokens": second,
                    "split": True,
                    "hit_cap": second_capped,
                }
            )
            carry = list(carry_pair_aligned(algo, second, prefix_tokens))

        assert len(windows) > len(texts), (
            f"{name} produced no re-split: the case stops covering what it was added for"
        )
        # No assertion that a half still overruns: a valid config gives a chunk
        # at most `window * CHARS_PER_TOKEN` characters, so a half is a quarter
        # of the window and cannot reach it. That branch lives in the fake-based
        # engine tests, where the cap is controllable.
        assert any(not w["split"] for w in windows), (
            f"{name} splits every chunk: nothing pins that an untouched chunk stays untouched"
        )
        chains.append(
            {
                "name": name,
                "text": text,
                "language": lang,
                "seed": seed,
                "window": window,
                "prepared": prepared,
                "windows": windows,
            }
        )
    return chains


def _render_end_to_end(
    ckpt_path: Path, algo, raw_tokens: dict, device: Device = "coreml"
) -> list[dict]:
    """Render each single-sentence case on CoreML and write its mel and wave.

    The tokens are the child phase's; this only turns them into audio, which is
    the half that needs the CoreML packages and the reason ``--skip-render``
    exists.
    """
    import loudkit
    from loudkit.config import ExecutionConfig
    from loudkit.voice import VoiceProfile

    execution = ExecutionConfig(device=device, precision=CONFORMANCE_EXECUTION)
    engine = loudkit.load(str(ckpt_path), device=device, execution=execution)
    assert algo.fingerprint() == engine.algorithm.fingerprint()
    voice = VoiceProfile.load(
        Path(__file__).resolve().parent.parent
        / "tests/data/reference/testvoice.voice.safetensors"
    )
    e2e = []
    for name, text, lang, seed in E2E_SENTENCES:
        # The child phase records the generator's raw row, stop marker and all;
        # the public render entry refuses control tokens (b3074fb), and the
        # fixture's contract is the stripped stream anyway.
        speech = [int(t) for t in raw_tokens[name]["raw"] if int(t) < algo.start_speech_token]
        result = engine.synthesize_tokens(speech, voice, seed=seed)
        suffix = "" if algo.decode == "single" else f"{algo.decode}_"
        mel_file, wav_file = f"e2e_{suffix}{name}_mel.bin", f"e2e_{suffix}{name}_wav.bin"
        result.mel.astype("<f4").tofile(OUT / mel_file)
        result.audio.astype("<f4").tofile(OUT / wav_file)
        e2e.append(
            {
                "name": name,
                "text": text,
                "language": lang,
                "seed": seed,
                "voice": "../reference/testvoice.voice.safetensors",
                "backend": device,
                "execution": CONFORMANCE_EXECUTION,
                "tokens": [int(t) for t in result.tokens],
                "mel": {"file": mel_file, "shape": list(result.mel.shape)},
                "wav": {"file": wav_file, "samples": int(len(result.audio))},
                "gates": {
                    "mel_corr": 0.999,
                    "wave_corr": 0.999 if algo.decode == "fusion_mtp2" else 0.95,
                    # Correlation is scale- and offset-free, so it cannot see a
                    # port rendering at half volume. Level gets its own gate:
                    # the RMS ratio against this reference, in dB. Measured
                    # spread between rendering paths is 0.007 dB, so 0.1 dB is
                    # fourteen times the observed divergence and still 1.2% of
                    # amplitude, sixty times tighter than a half-volume render.
                    "wave_rms_db": 0.1,
                },
            }
        )
    return e2e


SHARED_LISTING = [
    ".gitattributes",
    "LICENSE",
    "README.md",
    "SHA256SUMS",
    "coreml/camp.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "coreml/camp.mlpackage/Manifest.json",
    "coreml/export.json",
    "coreml/flow_encoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "coreml/flow_encoder.mlpackage/Manifest.json",
    "coreml/flow_estimator.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
    "coreml/flow_estimator.mlpackage/Manifest.json",
    "coreml/s3_tokenizer.mlpackage/Manifest.json",
    "coreml/vocoder.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    "coreml/vocoder.mlpackage/Manifest.json",
    "coreml/voice_encoder.mlpackage/Manifest.json",
    "logo.png",
    "loudr-1-enrollment.safetensors",
    "loudr-1.safetensors",
    "manifest.json",
    "onnx/camp.onnx",
    "onnx/export.json",
    "onnx/flow_encoder.onnx",
    "onnx/flow_estimator.onnx",
    "onnx/s3_tokenizer.onnx",
    "onnx/t3_cond.onnx",
    "onnx/t3_prefill.onnx",
    "onnx/t3_step.onnx",
    "onnx/vocoder.onnx",
    "onnx/voice_encoder.onnx",
    "release.json",
    "samples/joe.opus",
    "tokenizer.json",
    "torch/pytorch_model.bin",
    "ve.safetensors",
    "voices/en/nested.safetensors",
    "voices/gosia.safetensors",
    "voices/joe.safetensors",
]
"""One repo listing, wide enough to catch a plan that fetches a neighbouring
backend's set or a stranger's file."""

GLOB_PROBES = [
    ("*.safetensors", "voices/joe.safetensors"),
    ("*.safetensors", "loudr-1.safetensors"),
    ("*.safetensors", "loudr-1.safetensors.part"),
    ("*.safetensors", "voices/en/nested.safetensors"),
    ("voices/*", "voices/joe.safetensors"),
    ("voices/*", "voice.json"),
    ("onnx/t3_step.onnx", "onnx/t3_step.onnx"),
    ("onnx/t3_step.onnx", "onnx/t3_step.onnx.bak"),
    ("manifest.json", "coreml/x.mlpackage/Manifest.json"),
    ("coreml/vocoder.mlpackage/*", "coreml/vocoder.mlpackage/Data/x"),
    ("*", "anything/at/all"),
    ("*.onnx", "onnx/t3.onnx.tmp"),
    ("onnx/t3_?tep.onnx", "onnx/t3_step.onnx"),
    ("a?c", "abc"),
    ("a?c", "ac"),
    ("ve.safetensors", "ve.safetensors.bak"),
]

REPO_ID_PROBES = [
    "loudreader/loudr-1",
    "loudreader/loudr-1-turbo",
    "someone/model.v2",
    "org/name-with_dots.v2",
    # A Hub name is Unicode, so a repo id is classified by character and not
    # by byte. Both halves are probed: an ASCII-only word class accepts
    # neither, which is what the fixture is here to catch.
    "łódź/model",
    "org/日本語",
    "loudr-1",
    "loudreader",
    "./model.safetensors",
    "./loudr-1",
    "/no/such/place",
    "~/loudr-1",
    "a/b/c",
    "loudr-1.safetensors",
    "org/x.safetensors",
    "org/",
    "/org",
    "",
]
"""None of these may exist on disk where the generator or a test runs: a
path that exists is a path, however it is spelled."""

RECEIPT_REPO = "loudreader/loudr-1"
RECEIPT_COMMIT = "3f2c1e0d9b8a7f6e5d4c3b2a1f0e9d8c7b6a5f4e"
RECEIPT_FILES = ["manifest.json", "tokenizer.json"]
"""The two files every backend's plan selects, so one manifest serves five
suites."""
RECEIPT_SUMS = "".join(
    f"{hashlib.sha256(name.encode()).hexdigest()}  {name}\n" for name in RECEIPT_FILES
)
RECEIPT_ESCAPING_SUMS = RECEIPT_SUMS + f"{'0' * 64}  ../escape\n"
RECEIPT_REWRITTEN_SUMS = RECEIPT_SUMS.replace(
    hashlib.sha256(b"tokenizer.json").hexdigest(), "0" * 64
)
RECEIPT_EXAMPLE = {
    "repo": RECEIPT_REPO,
    "revision": "main",
    "commit": RECEIPT_COMMIT,
    "sha256sums": hashlib.sha256(RECEIPT_SUMS.encode()).hexdigest(),
    "fetched_at": "2026-09-02T12:00:00Z",
}
"""One receipt as every port writes it, over :data:`RECEIPT_SUMS`. The cases
below vary the receipt, the manifest or the files one thing at a time."""

CACHE_ROOTS = ["/home/ada/.cache", "/Users/ada/Library/Caches"]
"""A user cache root per platform the four ports run their tests on: Linux's
``$XDG_CACHE_HOME`` (``~/.cache`` unset) and macOS's ``~/Library/Caches``.
Windows is ``%LocalAppData%`` by the same rule."""
CACHE_REPOS = ["loudreader/loudr-1", "loudreader/loudr-1-turbo", "someone/model.v2"]
CACHE_ENV = "LOUDKIT_CACHE"
"""The one variable that moves the ports' cache. Python keeps the Hugging Face
cache, which has its own."""


def port_cache_path(root: str, repo: str) -> str:
    """Where a port keeps a release fetched by repo id: ``<root>/loudkit/<org>--<name>``,
    one layout for Go, Rust, JS and Swift, so a cache one port wrote is a
    cache the other three read. ``$LOUDKIT_CACHE`` replaces ``<root>/loudkit``."""
    return f"{root}/loudkit/{repo.replace('/', '--')}"


def _receipt_probe(
    name: str,
    receipt: Any,
    *,
    sums: str | None = RECEIPT_SUMS,
    files: list[str] = RECEIPT_FILES,
    repo: str = RECEIPT_REPO,
    commit: str = RECEIPT_COMMIT,
) -> dict[str, Any]:
    return {
        "name": name,
        "receipt": receipt,
        "sums": sums,
        "files": files,
        "repo": repo,
        "commit": commit,
    }


RECEIPT_PROBES: list[dict[str, Any]] = [
    _receipt_probe("the commit matches", RECEIPT_EXAMPLE),
    _receipt_probe("the revision moved", RECEIPT_EXAMPLE, commit="b" * 40),
    _receipt_probe("no receipt", None),
    _receipt_probe("a receipt for another repo", RECEIPT_EXAMPLE, repo="someone/loudr-1"),
    _receipt_probe(
        "a receipt with no commit",
        {k: v for k, v in RECEIPT_EXAMPLE.items() if k != "commit"},
    ),
    _receipt_probe("an empty commit", {**RECEIPT_EXAMPLE, "commit": ""}),
    _receipt_probe("a commit that is not forty hex", {**RECEIPT_EXAMPLE, "commit": "main"}),
    _receipt_probe("a field of the wrong type", {**RECEIPT_EXAMPLE, "revision": 1}),
    # A file that parses but is not an object.
    _receipt_probe("a receipt that is a JSON array", []),
    _receipt_probe("a receipt that is a number", 3),
    _receipt_probe("a receipt that is a string", "main"),
    # null is a wrong type too, in every field: a reader that decodes into a
    # plain string swallows it as "" and would pass the receipt.
    *(
        _receipt_probe(f"a null {field}", {**RECEIPT_EXAMPLE, field: None})
        for field in RECEIPT_EXAMPLE
    ),
    _receipt_probe(
        "a digest that is not the manifest's", {**RECEIPT_EXAMPLE, "sha256sums": "0" * 64}
    ),
    _receipt_probe(
        "a receipt claiming no manifest over a release that ships one",
        {**RECEIPT_EXAMPLE, "sha256sums": None},
    ),
    _receipt_probe("a listed file absent", RECEIPT_EXAMPLE, files=["manifest.json"]),
    _receipt_probe("nothing resolved", RECEIPT_EXAMPLE, commit=""),
    _receipt_probe("a field a reader does not know", {**RECEIPT_EXAMPLE, "note": "x"}),
    _receipt_probe(
        "the revision asked for is not the rule", {**RECEIPT_EXAMPLE, "revision": "v0.1.1"}
    ),
    _receipt_probe(
        "no sums in the release", {**RECEIPT_EXAMPLE, "sha256sums": None}, sums=None, files=[]
    ),
    # The reviewer's cases: one rule in five for what the type check leaves open.
    _receipt_probe(
        "a commit in uppercase hex", {**RECEIPT_EXAMPLE, "commit": RECEIPT_COMMIT.upper()}
    ),
    _receipt_probe(
        "a digest in uppercase hex",
        {**RECEIPT_EXAMPLE, "sha256sums": RECEIPT_EXAMPLE["sha256sums"].upper()},
    ),
    _receipt_probe(
        "a fetched_at that is not a timestamp", {**RECEIPT_EXAMPLE, "fetched_at": "yesterday"}
    ),
    _receipt_probe(
        "a manifest naming a file outside the release",
        {
            **RECEIPT_EXAMPLE,
            "sha256sums": hashlib.sha256(RECEIPT_ESCAPING_SUMS.encode()).hexdigest(),
        },
        sums=RECEIPT_ESCAPING_SUMS,
    ),
    # Trust on first use: a manifest rewritten with the receipt's digest
    # recomputed passes the predicate. Only the online commit check catches it.
    _receipt_probe(
        "a rewritten manifest under a recomputed digest",
        {
            **RECEIPT_EXAMPLE,
            "sha256sums": hashlib.sha256(RECEIPT_REWRITTEN_SUMS.encode()).hexdigest(),
        },
        sums=RECEIPT_REWRITTEN_SUMS,
    ),
]


def _receipt_case(probe: dict[str, Any]) -> dict[str, Any]:
    """One probe laid out on disk and judged by the reference: what a load
    does with it offline (``use`` or ``error``) and online (``hit`` or
    ``fetch``)."""
    from loudkit import hub

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        if probe["receipt"] is not None:
            (root / hub.RECEIPT_NAME).write_text(
                json.dumps(probe["receipt"]), encoding="utf-8", newline="\n"
            )
        if probe["sums"] is not None:
            (root / "SHA256SUMS").write_text(probe["sums"], encoding="utf-8", newline="\n")
        for name in probe["files"]:
            (root / name).write_text(name, encoding="utf-8", newline="\n")
        valid = hub.read_receipt(root, probe["repo"]) is not None
        hit = hub.receipt_hit(root, probe["repo"], probe["commit"])
    return {
        **probe,
        "offline": "use" if valid else "error",
        "online": "hit" if hit else "fetch",
    }


QUANTISE_PROBES: list[float | str] = [0, 0.5, -0.5, 0.9, -0.9, 1, -1, 2, -2, "nan"]
"""The ten frames every port's WAV writer is pinned on. NaN is a string
because JSON has no spelling for it."""

WAV_HEADER_PROBES: list[tuple[list[float | str], int]] = [
    ([0, 1], 24000),
    ([0, 0.5, -0.5, 1, -1], 24000),
    ([0, 1.0, -1.0, 0.5, -0.5, -0.00001], 24000),
    ([0, 0], 16000),
    ([], 24000),
    (QUANTISE_PROBES, 8000),
]


def _probe_value(x: object) -> float:
    return float("nan") if x == "nan" else float(x)  # type: ignore[arg-type]


EDGE_FADE_RATE = 24_000
"""The synthesis rate the edge-fade tables are cut for."""

EDGE_FADE_LENGTHS: list[float] = [0.005, 0.02]
"""Every ramp a port carries as a table: the historical 5 ms and the shipped 20 ms.

Both are pinned because neither can be recomputed abroad. numpy takes the cosine
in float32 off a float32 ``linspace``; a port that evaluates it in double and
narrows lands within two units in the last place, which is the identity
contract's `equivalent` class, not its bit-exact one. A length the ports do not
ship a table for is left to that computed fallback on purpose.

Spelled out rather than read from ``AlgorithmConfig``: a manifest that already
shipped keeps applying its own ramp, so a new default is added here beside the
old ones, never in place of them.
"""


def _edge_fade_ramp(seconds: float) -> list[int]:
    """One ramp's float32 bits, read off the reference renderer's own output.

    Faded from ones, so the head is the ramp itself and the fixture is what
    ``loudkit.window._fade_edges`` applies rather than a second spelling of it.
    """
    from loudkit.window import _fade_edges

    n = int(seconds * EDGE_FADE_RATE)
    ones = np.ones(4 * n, dtype=np.float32)
    faded = _fade_edges(ones, sample_rate=EDGE_FADE_RATE, seconds=seconds)
    return [int(b) for b in faded[:n].view(np.uint32)]


def shared_fixtures(out: Path) -> list[Path]:
    """The eight fixtures that replace the hand-transcribed port tests.

    Fetch plan, PCM16 quantisation, WAV header bytes, ``fnmatch`` semantics,
    repo-id recognition, the cache receipt, the cache path and the edge-fade
    ramps: each is one rule the four downloaders, WAV writers and renderers
    reimplement, read here off the reference and written as JSON that the
    suites read. No weights are needed.
    """
    import fnmatch
    import warnings

    from loudkit import hub
    from loudkit.synthesis import _encode, _quantise

    stamp = {"generated_by": "tools/make_conformance.py --fixtures-only"}
    plan_cases = []
    for backend in ("torch", "onnx", "coreml"):
        for cloning in (False, True):
            allow, ignore = hub.release_patterns(backend, cloning=cloning)
            wanted = sorted(
                p
                for p in SHARED_LISTING
                if any(fnmatch.fnmatch(p, a) for a in allow)
                and not any(fnmatch.fnmatch(p, i) for i in ignore)
            )
            plan_cases.append(
                {
                    "backend": backend,
                    "cloning": cloning,
                    "allow": list(allow),
                    "ignore": list(ignore),
                    "wanted": wanted,
                }
            )
    ramps: list[dict[str, Any]] = [
        {"seconds": s, "samples": int(s * EDGE_FADE_RATE), "bits": _edge_fade_ramp(s)}
        for s in EDGE_FADE_LENGTHS
    ]
    files = {
        "release_plan.json": {**stamp, "listing": SHARED_LISTING, "cases": plan_cases},
        "glob.json": {
            **stamp,
            "cases": [
                {"pattern": p, "name": n, "match": fnmatch.fnmatch(n, p)}
                for p, n in GLOB_PROBES
            ],
        },
        "repo_id.json": {
            **stamp,
            "cases": [{"ref": r, "is_repo_id": hub.is_repo_id(r)} for r in REPO_ID_PROBES],
        },
        "release_receipt.json": {
            **stamp,
            "name": hub.RECEIPT_NAME,
            "fields": list(hub.RECEIPT_FIELDS),
            "example": RECEIPT_EXAMPLE,
            "sums": RECEIPT_SUMS,
            "files": RECEIPT_FILES,
            "cases": [_receipt_case(probe) for probe in RECEIPT_PROBES],
        },
        "cache_path.json": {
            **stamp,
            "env": CACHE_ENV,
            "cases": [
                {"root": root, "repo": repo, "path": port_cache_path(root, repo)}
                for root in CACHE_ROOTS
                for repo in CACHE_REPOS
            ],
            "override": [
                {
                    "cache": "/data/loudkit",
                    "repo": repo,
                    "path": f"/data/loudkit/{repo.replace('/', '--')}",
                }
                for repo in CACHE_REPOS
            ],
        },
        "edge_fade.json": {
            **stamp,
            "sample_rate": EDGE_FADE_RATE,
            # `samples` and `bits` repeat the first ramp where a reader written
            # against the 5 ms-only fixture already looks for them.
            "samples": ramps[0]["samples"],
            "bits": ramps[0]["bits"],
            "ramps": ramps,
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # NaN's cast is the probe
        probes = np.array([_probe_value(x) for x in QUANTISE_PROBES], dtype=np.float32)
        frames: NDArray[Any] = _quantise(probes)
        files["wav_quantise.json"] = {
            **stamp,
            "cases": [
                {"x": x, "pcm16": int(f)} for x, f in zip(QUANTISE_PROBES, frames, strict=True)
            ],
        }
        header_cases = []
        for samples, rate in WAV_HEADER_PROBES:
            values = np.array([_probe_value(x) for x in samples], dtype=np.float32)
            wav, _ = _encode(_quantise(values), rate, "wav")
            header_cases.append({"samples": samples, "sample_rate": rate, "hex": wav.hex()})
        files["wav_header.json"] = {**stamp, "cases": header_cases}
    written = []
    for name, body in files.items():
        path = out / name
        path.write_text(
            json.dumps(body, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    return written


def _arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint")
    ap.add_argument(
        "--fixtures-only",
        action="store_true",
        help="write the eight weight-free shared fixtures and nothing else",
    )
    ap.add_argument("--out", default=None, help="directory for --fixtures-only")
    ap.add_argument(
        "--skip-e2e", action="store_true", help="regenerate only the weight-free sections"
    )
    ap.add_argument(
        "--skip-render",
        action="store_true",
        help=(
            "regenerate the token sections but carry end_to_end (and its mel/wav "
            "bins) over unchanged — for a machine without usable CoreML packages"
        ),
    )
    ap.add_argument("--phase", choices=["all", "tokens"], default="all")
    ap.add_argument("--tokens-out", default=None)
    ap.add_argument("--render-device", choices=["cpu", "onnx", "coreml"], default="coreml")
    return ap.parse_args()


def main() -> None:
    args = _arguments()
    if args.fixtures_only:
        out = Path(args.out) if args.out else OUT
        out.mkdir(parents=True, exist_ok=True)
        for path in shared_fixtures(out):
            print(f"wrote {path}")
        return
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --fixtures-only is given")

    if args.phase == "tokens":
        _tokens_phase(args.checkpoint, args.tokens_out)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)

    fixture: dict = {
        "version": 1,
        "generated_by": "tools/make_conformance.py",
        "philox": philox_section(),
        "sampler": sampler_section(),
        "eos_peak": eos_peak_section(),
        "frontend": frontend_section(ckpt_path.parent / "tokenizer.json"),
        "seeds": seed_section(),
    }

    # algorithm identity + the euler grid both languages must compute
    from loudkit.backends import production_algorithm
    from loudkit.checkpoint import Checkpoint

    ckpt = Checkpoint.open(str(ckpt_path))
    algo = production_algorithm(ckpt)
    vectors_name = "vectors.json" if algo.decode == "single" else f"vectors_{algo.decode}.json"
    fixture["algorithm"] = {
        "fingerprint": algo.fingerprint(),
        # the exact string hashed (AlgorithmConfig.canonical_form): floats as
        # repr strings, sorted keys, {"schema", "algorithm"} envelope, a
        # specification a Swift port can implement, not an accident of
        # json.dumps
        "canonical_form": algo.canonical_form(),
        "euler_grid": [float(t) for t in time_grid(algo)],
        "grid_rtol": 1e-15,
    }

    def carried(section: str) -> object | None:
        """The section as the fixture on disk already holds it, or ``None``."""
        existing = OUT / vectors_name
        if not existing.exists():
            return None
        return json.loads(existing.read_text(encoding="utf-8")).get(section)

    if not args.skip_e2e:
        import subprocess
        import tempfile

        # tokens in a child process (torch decode + coremltools segfault when
        # they share one, see _tokens_phase), renders in this one
        with tempfile.NamedTemporaryFile(suffix=".json") as tf:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--checkpoint",
                    args.checkpoint,
                    "--phase",
                    "tokens",
                    "--tokens-out",
                    tf.name,
                ],
                check=True,
            )
            raw_tokens = json.loads(Path(tf.name).read_text(encoding="utf-8"))

        if args.skip_render:
            # The token layer regenerates here, the rendered one does not: the
            # long-form section needs the token generator only, while
            # `end_to_end` needs the CoreML packages, which are an optional
            # download and are not always intact. Carrying the rendered section
            # over verbatim keeps a partial machine from writing a fixture that
            # claims a render it never ran.
            fixture["end_to_end"] = carried("end_to_end") or []
        else:
            fixture["end_to_end"] = _render_end_to_end(
                ckpt_path, algo, raw_tokens, args.render_device
            )
        fixture["long_form"] = {
            "voice": "../reference/testvoice.voice.safetensors",
            "execution": CONFORMANCE_EXECUTION,
            # Restated beside the cases so a port reads the chain's two
            # constants from the fixture rather than from a comment: how many
            # tokens carry across a join, and which seed stream the chunk seeds
            # come from. Both are already pinned elsewhere, `prefix_tokens` in
            # the fingerprinted algorithm, the stream base in every engine, and
            # a port whose values differ diverges here rather than in a listener's
            # ear. The base applies from chunk 1 up; chunk 0 draws the caller's
            # seed, which each recorded `seed` states outright.
            "prefix_tokens": algo.chunking.prefix_tokens,
            "chunk_stream_base": _STREAM_CHUNK,
            "cases": raw_tokens["__long_form__"],
        }
        fixture["resplit"] = {
            "voice": "../reference/testvoice.voice.safetensors",
            "execution": CONFORMANCE_EXECUTION,
            # The same two chain constants `long_form` restates, plus the stream
            # the second half of a re-split draws from. A port whose value
            # differs diverges here rather than in a listener's ear, which is
            # the whole reason this section exists: the wiring around
            # `split_in_half` was held by nothing.
            "prefix_tokens": algo.chunking.prefix_tokens,
            "chunk_stream_base": _STREAM_CHUNK,
            "resplit_stream": _STREAM_RESPLIT,
            "cases": raw_tokens["__resplit__"],
        }
    else:
        for section in ("end_to_end", "long_form", "resplit"):
            previous = carried(section)
            if previous is not None:
                fixture[section] = previous

    if algo.decode != "single":
        fixture = {
            key: value
            for key, value in fixture.items()
            if key
            in {"version", "generated_by", "algorithm", "end_to_end", "long_form", "resplit"}
        }
        fixture["shared"] = "vectors.json"

    # `newline="\n"`: five implementations read these bytes.
    with open(OUT / vectors_name, "w", encoding="utf-8", newline="\n") as f:
        # `ensure_ascii=False`, matching the fixture already on disk: the file is
        # UTF-8, every reader parses it as UTF-8, and escaping the em dash and
        # the Polish diacritics turns "did the frontend cases change?" into a
        # diff nobody can read.
        json.dump(fixture, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT / vectors_name}")
    for name in sorted(p.name for p in OUT.iterdir()):
        print(" ", name)


if __name__ == "__main__":
    main()
