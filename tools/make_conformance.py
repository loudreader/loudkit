"""Generate the cross-language conformance fixture.

One directory (``tests/data/conformance``) that the Python suite and the Swift
package's ``swift test`` both read, covering the layers where two independent
implementations of the same law could drift:

  philox       the three Random123 KAT vectors, raw uniform *bits* for fixed
               counters (exact, integer), and gumbel probe values (float64,
               compared to 1e-12 — two libms may differ in the last ulp)
  sampler      LR-SAMPLER-v1 token choices: small hand-set logits, a
               silence-exemption case, and a full-vocab case whose logits are
               *derived from Philox bits* so neither language has to ship an
               8194-float table to agree on what the input was
  frontend     text -> token ids, en and pl, punctuation / diacritics /
               whitespace traps; the tokenizer JSON is copied in so this layer
               is checkable with no weights at all
  algorithm    the production fingerprint and the exact canonical JSON blob it
               hashes — a fingerprint mismatch with a matching blob means the
               hash is wrong, a blob mismatch names the drifted field
  end_to_end   text + voice + seed -> speech tokens (exact), mel and waveform
               (banded), rendered by the coreml backend with the token
               generator declared fp32 — the precision the Swift generator
               runs, because "same precision, same tokens" is the contract
               (engine.synthesize docstring) and fp16-vs-fp32 token identity
               on these sentences is an observation, not a promise
  long_form    a passage that does *not* fit one window: the funnel, the split,
               the per-chunk seed, the carried prefix and the exact token
               stream of EVERY chunk

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
path applies to the terminal chunk — exactly like ``end_to_end``, and for the
same reason: the contract being pinned is "this text, this voice, this seed,
this prefix -> these tokens", which is the layer the ports reimplement.

Numbers that do not fit a JSON double exactly (derived 64-bit seeds) are
stored as hex strings. Mel and waveform go beside the JSON as raw little-
endian float32 (shapes in the JSON) so Swift needs no npy parser.

Usage (regenerates everything; run when the engine legitimately changes):
  .venv/bin/python tools/make_conformance.py \
      --checkpoint /path/to/loudr-1.safetensors
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from loudkit.config import Precision, SamplingConfig  # noqa: E402
from loudkit.engine import _STREAM_CHUNK, _derive  # noqa: E402
from loudkit.models.flow import time_grid  # noqa: E402
from loudkit.rng import KAT_VECTORS, uniforms  # noqa: E402
from loudkit.sampler import LRSamplerV1  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tests" / "data" / "conformance"

E2E_SENTENCES = [
    ("s0", "The quick brown fox jumps over the lazy dog.", "en", 4242),
    ("s2", "Wait — was that a knock at the door, or only the wind?", "en", 7),
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

    # 2. silence exemption: ids 3 and 5 exempt from penalty and min_p
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
    has two subtleties either of which a reimplementation gets wrong silently —
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

    A separate process because torch's decode loop and coremltools segfault
    when they share one (the same instability the sample wall documented for
    the T3 export harness). The composition is still the conformance engine's:
    the coreml backend runs its token generator on torch/CPU anyway, and the
    value boundary between stages is exactly what makes the two-process split
    equivalent to one call.
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
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)


def _long_form_chains(engine, voice) -> list[dict]:  # type: ignore[no-untyped-def]
    """Walk the shipping chunk chain for each long-form case and record it.

    The chain, in the order the engine walks it: run the speech funnel on the
    whole passage *before* splitting (Polish respelling changes the length, so
    a budget computed after it would be a budget for text nobody speaks), split
    at the chunking recipe, then per chunk derive the seed from
    ``_STREAM_CHUNK + index`` and carry the last ``prefix_tokens`` speech tokens
    of the chunk before it.

    Three properties are asserted rather than hoped for, because a case that
    loses any of them stops covering what it was added for:

    * **more than one chunk** — with an empty prefix the two indexing bugs this
      section exists to catch are unobservable;
    * **every chunk ends on the stop token** — a chunk that ran into the cap
      pins a truncation, and the cap is a different fact from the token stream;
    * **at least one carried token outside the silence manifest** — the sampler
      exempts silence ids from the repetition penalty, so a tail made only of
      them makes a seeded mask and a blind one pick the same token, which is
      precisely how the unseeded-mask defect survived the existing fixture.
    """
    from loudkit.frontend.chunking import split_text
    from loudkit.frontend.polish import speech_text

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
            chunk_seed = _derive(seed, _STREAM_CHUNK + index)
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
            carry = speech[-prefix_tokens:] if prefix_tokens > 0 else []
            if index + 1 < len(texts):
                assert any(t not in silence for t in carry), (
                    f"{name} chunk {index} carries only silence ids {carry}: the "
                    "repetition penalty exempts those, so the next chunk samples "
                    "identically with a seeded mask and a blind one"
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
            }
        )
    return chains


def _render_end_to_end(ckpt_path: Path, algo, raw_tokens: dict) -> list[dict]:  # type: ignore[no-untyped-def]
    """Render each single-sentence case on CoreML and write its mel and wave.

    The tokens are the child phase's; this only turns them into audio, which is
    the half that needs the CoreML packages and the reason ``--skip-render``
    exists.
    """
    import loudkit
    from loudkit.config import ExecutionConfig
    from loudkit.voice import VoiceProfile

    execution = ExecutionConfig(device="coreml", precision=CONFORMANCE_EXECUTION)
    engine = loudkit.load(str(ckpt_path), device="coreml", execution=execution)
    assert algo.fingerprint() == engine.algorithm.fingerprint()
    voice = VoiceProfile.load(
        Path(__file__).resolve().parent.parent
        / "tests/data/reference/testvoice.voice.safetensors"
    )
    e2e = []
    for name, text, lang, seed in E2E_SENTENCES:
        result = engine.synthesize_tokens(raw_tokens[name]["raw"], voice, seed=seed)
        mel_file, wav_file = f"e2e_{name}_mel.bin", f"e2e_{name}_wav.bin"
        result.mel.astype("<f4").tofile(OUT / mel_file)
        result.audio.astype("<f4").tofile(OUT / wav_file)
        e2e.append(
            {
                "name": name,
                "text": text,
                "language": lang,
                "seed": seed,
                "voice": "../reference/testvoice.voice.safetensors",
                "backend": "coreml",
                "execution": CONFORMANCE_EXECUTION,
                "tokens": [int(t) for t in result.tokens],
                "mel": {"file": mel_file, "shape": list(result.mel.shape)},
                "wav": {"file": wav_file, "samples": int(len(result.audio))},
                "gates": {"mel_corr": 0.999, "wave_corr": 0.95},
            }
        )
    return e2e


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
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
    args = ap.parse_args()

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
    fixture["algorithm"] = {
        "fingerprint": algo.fingerprint(),
        # the exact string hashed (AlgorithmConfig.canonical_form): floats as
        # repr strings, sorted keys, {"schema", "algorithm"} envelope — a
        # specification a Swift port can implement, not an accident of
        # json.dumps
        "canonical_form": algo.canonical_form(),
        "euler_grid": [float(t) for t in time_grid(algo)],
        "grid_rtol": 1e-15,
    }

    def carried(section: str) -> object | None:
        """The section as the fixture on disk already holds it, or ``None``."""
        existing = OUT / "vectors.json"
        if not existing.exists():
            return None
        return json.loads(existing.read_text(encoding="utf-8")).get(section)

    if not args.skip_e2e:
        import subprocess
        import tempfile

        # tokens in a child process (torch decode + coremltools segfault when
        # they share one — see _tokens_phase), renders in this one
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
            fixture["end_to_end"] = _render_end_to_end(ckpt_path, algo, raw_tokens)
        fixture["long_form"] = {
            "voice": "../reference/testvoice.voice.safetensors",
            "execution": CONFORMANCE_EXECUTION,
            # Restated beside the cases so a port reads the chain's two
            # constants from the fixture rather than from a comment: how many
            # tokens carry across a join, and which seed stream the chunk seeds
            # come from. Both are already pinned elsewhere — `prefix_tokens` in
            # the fingerprinted algorithm, the stream base in every engine — and
            # a port whose values differ diverges here rather than in a listener's
            # ear.
            "prefix_tokens": algo.chunking.prefix_tokens,
            "chunk_stream_base": _STREAM_CHUNK,
            "cases": raw_tokens["__long_form__"],
        }
    else:
        for section in ("end_to_end", "long_form"):
            previous = carried(section)
            if previous is not None:
                fixture[section] = previous

    with open(OUT / "vectors.json", "w", encoding="utf-8") as f:
        # `ensure_ascii=False`, matching the fixture already on disk: the file is
        # UTF-8, every reader parses it as UTF-8, and escaping the em dash and
        # the Polish diacritics turns "did the frontend cases change?" into a
        # diff nobody can read.
        json.dump(fixture, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT / 'vectors.json'}")
    for name in sorted(p.name for p in OUT.iterdir()):
        print(" ", name)


if __name__ == "__main__":
    main()
