"""The cross-language conformance vectors, checked against this implementation.

``tests/data/conformance`` is read by two test suites: this one, and the Swift
package's ``swift test``. The fixture is the contract — if Python drifts from
it, this file fails; if Swift drifts from it, the Swift tests fail; if both
pass, the two implementations agree without ever having met.

The weight-free sections (Philox, sampler, frontend, seeds) run everywhere.
The algorithm-identity and end-to-end sections need the synthesis checkpoint and
the exported CoreML packages, and skip with a named reason without them
(``LOUDKIT_REQUIRE_ASSETS=1`` turns that into a failure, as everywhere else).

Order matters inside this module and is load-bearing: the token-generation
test runs the torch decode loop and must execute before anything imports
coremltools — the two segfault when torch decodes after CoreML loads in the
same process (the same instability that kept the T3 export un-validated from
Python). pytest runs tests in definition order, so the generator test is
defined before the renderer test and both are in one class to keep it that
way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from .assets import asset, needs_module, requires, skip_or_fail
from .conftest import assert_amplitude

FIXTURE_DIR = Path(__file__).parent / "data" / "conformance"
CKPT = asset("checkpoint")


@pytest.fixture(scope="module")
def fixture() -> dict:
    with open(FIXTURE_DIR / "vectors.json", encoding="utf-8") as f:
        return json.load(f)


class TestPhilox:
    def test_known_answer_vectors(self, fixture: dict) -> None:
        from loudkit.rng import philox_4x32_10

        for case in fixture["philox"]["kat"]:
            c0, c1, c2, c3 = (np.array([c], dtype=np.uint64) for c in case["counter"])
            got = philox_4x32_10(c0, c1, c2, c3, case["key"][0], case["key"][1])
            assert [int(g[0]) for g in got] == case["expected"]

    def test_uniform_bits(self, fixture: dict) -> None:
        from loudkit.rng import uniforms

        for probe in fixture["philox"]["uniform_bits"]:
            u = uniforms(
                int(probe["seed"], 16),
                probe["stream"],
                probe["step0"],
                probe["n_steps"],
                probe["width"],
            )
            bits = np.round(u * 4294967296.0 - 0.5).astype(np.uint64)
            assert bits.tolist() == probe["bits"], probe

    def test_gumbel_probes(self, fixture: dict) -> None:
        from loudkit.rng import gumbel_noise

        for probe in fixture["philox"]["gumbel"]:
            g = gumbel_noise(probe["seed"], probe["stream"], probe["step"], 1, probe["width"])[
                0
            ]
            # exact for this implementation — the rtol in the fixture is the
            # allowance for another language's libm, not for ours
            assert [float(v) for v in g] == probe["values"]


class TestSampler:
    @staticmethod
    def _logits_rows(case: dict) -> list[np.ndarray]:
        if "logits_recipe" in case:
            from loudkit.rng import uniforms

            r = case["logits_recipe"]
            return [
                (
                    uniforms(r["seed"], r["stream"], step, 1, r["vocab"])[0] * r["scale"]
                    + r["offset"]
                ).astype(np.float32)
                for step in range(r["steps"])
            ]
        row = np.asarray(case["logits"][0], dtype=np.float32)
        return [row] * case.get("repeat_logits", len(case["logits"]))

    def test_token_choices(self, fixture: dict) -> None:
        from loudkit.config import SamplingConfig
        from loudkit.sampler import LRSamplerV1

        for case in fixture["sampler"]["cases"]:
            c = case["config"]
            cfg = SamplingConfig(
                temperature=c["temperature"],
                repetition_penalty=c["repetition_penalty"],
                min_p=c["min_p"],
                silence_token_ids=tuple(c["silence_token_ids"]),
            )
            sampler = LRSamplerV1(cfg, seed=case["seed"])
            rows = self._logits_rows(case)
            seen = np.zeros(rows[0].shape[0], dtype=bool)
            got = []
            for step, row in enumerate(rows):
                tok = sampler(row, step=step, seen=seen)
                got.append(tok)
                seen[tok] = True
            assert got == case["expected"], case["name"]


class TestFrontend:
    def test_token_ids(self, fixture: dict) -> None:
        from loudkit.frontend.text import GraphemeTextFrontend

        frontend = GraphemeTextFrontend(FIXTURE_DIR / fixture["frontend"]["tokenizer"])
        for case in fixture["frontend"]["cases"]:
            ids = frontend.encode(case["text"], case["language"])
            assert ids.tolist() == case["ids"], case["text"]


class TestSeeds:
    def test_derivation(self, fixture: dict) -> None:
        from loudkit.window import _derive

        for probe in fixture["seeds"]["derivation"]:
            assert hex(_derive(probe["seed"], probe["stream"])) == probe["derived"]

    def test_the_stream_bases_are_the_ones_the_fixture_publishes(self, fixture: dict) -> None:
        """Two constants the four ports read weight-free and Python did not.

        `chunk_stream_base` and `resplit_stream` are in the fixture precisely so
        five implementations derive the same per-chunk and per-half seeds. Rust,
        Go, JS and Swift all read them; Python's own tests took the seed
        straight out of the fixture row instead of deriving it, and
        `tests/test_engine.py` imports `_STREAM_CHUNK` to *build* the
        expectation it should be pinning. Verified by mutation: `_STREAM_CHUNK
        = 17` and seeding the second half from the chunk seed instead of
        `_derive(chunk_seed, _STREAM_RESPLIT)` both survive the whole
        weight-free suite. They are caught once assets are present, which is
        the asset-backed job on `main` and same-repo PRs only.
        """
        from loudkit.window import _STREAM_CHUNK, _STREAM_RESPLIT

        assert fixture["long_form"]["chunk_stream_base"] == _STREAM_CHUNK
        assert fixture["resplit"]["resplit_stream"] == _STREAM_RESPLIT


def test_the_production_values_hash_to_the_fixture(fixture: dict) -> None:
    """The fingerprint, from the values alone, with no checkpoint.

    Go, Rust and JS spell the production algorithm out and compare it to the
    fixture weight-free; Python read it from the checkpoint, so in the
    weight-free job a change to a Python default that moved the fingerprint
    was caught only by a port's pin. The three lists are the checkpoint's
    censuses; everything else is a default this package ships.
    """
    from loudkit.backends import PRODUCTION_WINDOW
    from loudkit.config import AlgorithmConfig, SamplingConfig
    from loudkit.postprocess import PostprocessConfig

    silence_render_ids = (4137, 4215, 4218, 4299, 6162, 6324, 6405, 6486)
    quiet_render_ids = (
        1458, 1461, 1488, 1701, 1704, 1707, 1716, 1731, 1785, 1788, 1869, 1947,
        1950, 1951, 1959, 1978, 2028, 2031, 2040, 2058, 2076, 2112, 2139, 3645,
        3648, 3651, 3704, 3888, 3894, 4188, 5838, 6081, 6183, 6537,
    )  # fmt: skip
    silence_token_ids = (
        1731, 1821, 1822, 1824, 1975, 2058, 2068, 3190, 3377, 3918, 3927, 3928,
        3930, 4008, 4009, 4011, 4012, 4137, 4146, 4161, 4171, 4173, 4174, 4218,
        4245, 4251, 4252, 4254, 4255, 4260, 4282,
    )  # fmt: skip
    algo = AlgorithmConfig(
        recipe_version="loudkit-1",
        guidance="single_path",
        guidance_rate=0.0,
        euler_steps=2,
        euler_grid=None,
        sample_rate=24_000,
        token_rate_hz=25.0,
        speech_vocab_size=8194,
        start_speech_token=6561,
        stop_speech_token=6562,
        window=PRODUCTION_WINDOW,
        postprocess=PostprocessConfig(
            silence_render_ids=silence_render_ids, quiet_render_ids=quiet_render_ids
        ),
        sampling=SamplingConfig(
            temperature=0.8,
            repetition_penalty=1.2,
            min_p=0.05,
            max_new_tokens=255,
            min_tokens_floor=10,
            min_tokens_text_ratio=1.2,
            silence_token_ids=silence_token_ids,
        ),
    )
    assert algo.canonical_form() == fixture["algorithm"]["canonical_form"]
    assert algo.fingerprint() == fixture["algorithm"]["fingerprint"]


@pytest.mark.parametrize(
    ("what", "render"),
    [
        ("-6 dB", lambda w: w * 0.5),
        ("-12 dB", lambda w: w * 0.25),
        ("+0.2 DC", lambda w: w + 0.2),
        ("20x gain", lambda w: w * 20.0),
    ],
)
def test_the_amplitude_gate_sees_what_the_correlation_cannot(what, render, fixture) -> None:
    """Weight-free, on the committed reference: four renders no port should ship.

    Each of them correlates 1.000000 with the reference, because correlation is
    invariant to scale and to offset. A port rendering everything at half
    volume, or clipping, passes a correlation band and nothing else in the
    end-to-end stage would say so. The amplitude gate is what does.
    """
    case = fixture["end_to_end"][0]
    reference = np.fromfile(FIXTURE_DIR / case["wav"]["file"], dtype="<f4")
    audio = render(reference.astype(np.float64))
    assert np.corrcoef(audio, reference)[0, 1] == pytest.approx(1.0, abs=1e-12), what
    with pytest.raises(AssertionError):
        assert_amplitude(what, case["gates"]["wave_rms_db"], audio, reference)


@requires("checkpoint")
class TestAlgorithmIdentity:
    def test_fingerprint_and_blob(self, fixture: dict) -> None:
        from loudkit.backends import production_algorithm
        from loudkit.checkpoint import Checkpoint

        algo = production_algorithm(Checkpoint.open(str(CKPT)))
        # canonical form first: a form mismatch names the drifted field, a
        # fingerprint mismatch alone names nothing
        assert algo.canonical_form() == fixture["algorithm"]["canonical_form"]
        assert algo.fingerprint() == fixture["algorithm"]["fingerprint"]

    def test_euler_grid(self, fixture: dict) -> None:
        from loudkit.backends import production_algorithm
        from loudkit.checkpoint import Checkpoint
        from loudkit.models.flow import time_grid

        algo = production_algorithm(Checkpoint.open(str(CKPT)))
        got = time_grid(algo)
        want = fixture["algorithm"]["euler_grid"]
        assert len(got) == len(want)
        np.testing.assert_allclose(got, want, rtol=fixture["algorithm"]["grid_rtol"])


@pytest.mark.slow
@requires("checkpoint")
class TestEndToEnd:
    """Tokens exactly, renders within the band. Generator first — see the
    module docstring for why the order is not a style choice."""

    def test_tokens_from_seed(self, fixture: dict) -> None:
        import loudkit
        from loudkit.config import ExecutionConfig
        from loudkit.sampler import LRSamplerV1
        from loudkit.voice import VoiceProfile

        cases = fixture.get("end_to_end")
        if not cases:
            skip_or_fail("fixture has no end_to_end section")
        execution = ExecutionConfig(device="cpu", precision=cases[0]["execution"])
        engine = loudkit.load(str(CKPT), device="cpu", execution=execution)
        voice = VoiceProfile.load(FIXTURE_DIR / cases[0]["voice"])
        for case in cases:
            text_tokens = engine.frontend.encode(case["text"], case["language"])
            sampler = LRSamplerV1(engine.algorithm.sampling, seed=case["seed"])
            raw = list(engine.token_generator.generate(text_tokens, voice, sampler=sampler))
            stripped = [t for t in raw if t < engine.algorithm.start_speech_token]
            assert stripped[: engine.algorithm.window.max_speech_tokens] == case["tokens"], (
                case["name"]
            )

    def test_long_form_chunk_tokens(self, fixture: dict) -> None:
        """A passage too long for one window, chunk by chunk.

        Everything above this line is one window with an empty prefix, where
        ``len(prefix) + step + 1`` and ``step + 1`` are the same expression and
        a repetition mask seeded from the prefix is the empty one. Three ports
        wrote the shorter form and the fixture passed for months. A carried
        prefix is what separates them.

        Each chunk is asserted on its own rather than on the concatenation, so
        a port that diverges inside chunk *k* is told which chunk and at which
        step, instead of seeing every token after the divergence shift.
        """
        import loudkit
        from loudkit.config import ExecutionConfig
        from loudkit.frontend.chunking import split_text
        from loudkit.frontend.speechtext import speech_text
        from loudkit.sampler import LRSamplerV1
        from loudkit.voice import VoiceProfile

        section = fixture.get("long_form")
        if not section:
            skip_or_fail("fixture has no long_form section")
        execution = ExecutionConfig(device="cpu", precision=section["execution"])
        engine = loudkit.load(str(CKPT), device="cpu", execution=execution)
        voice = VoiceProfile.load(FIXTURE_DIR / section["voice"])
        algo = engine.algorithm
        assert algo.chunking.prefix_tokens == section["prefix_tokens"]

        for case in section["cases"]:
            name, language = case["name"], case["language"]
            # The funnel runs on the whole passage before the split, which is
            # the order the engine uses and the order the budget assumes.
            prepared = speech_text(case["text"], language)
            assert prepared == case["prepared"], f"{name}: the speech funnel drifted"
            chunks = case["chunks"]
            assert len(chunks) > 1, f"{name} is a single window and proves nothing"
            assert split_text(prepared, algo.chunking) == [c["text"] for c in chunks], (
                f"{name}: the split moved, so every chunk below is asking about different text"
            )
            for chunk in chunks:
                index = chunk["index"]
                # The chain the streaming path walks: chunk *k* is conditioned
                # on the tail of chunk *k-1*, and the fixture spells that tail
                # out so a mismatch names the carry rather than the tokens.
                if index > 0:
                    tail = chunks[index - 1]["tokens"][-section["prefix_tokens"] :]
                    assert chunk["prefix"] == tail, f"{name} chunk {index}: carry"
                text_tokens = engine.frontend.encode(chunk["text"], language)
                sampler = LRSamplerV1(algo.sampling, seed=int(chunk["seed"], 16))
                raw = list(
                    engine.token_generator.generate(
                        text_tokens, voice, sampler=sampler, prefix=chunk["prefix"]
                    )
                )
                got = [int(t) for t in raw if int(t) < algo.start_speech_token]
                assert got == chunk["tokens"], f"{name} chunk {index}"

            # The loop above reads each chunk's seed and carry from the
            # fixture, so it cannot notice the engine deriving either one
            # differently. The public paths walk their own chunk loop: they
            # are what the four ports hold against the same rows.
            streamed = list(
                engine.stream(case["text"], voice, seed=case["seed"], language=language)
            )
            assert [list(r.tokens) for r in streamed] == [c["tokens"] for c in chunks], (
                f"{name}: stream"
            )
            joined = engine.synthesize(
                case["text"], voice, seed=case["seed"], language=language
            )
            assert list(joined.tokens) == case["tokens"], f"{name}: synthesize"

    def test_resplit_windows(self, fixture: dict) -> None:
        """A chunk the window could not hold, and the two it becomes.

        This drives `engine.stream` rather than reimplementing the chain. An
        earlier version walked the token generator itself and agreed with its
        own arithmetic: it injected a token ceiling the engine never passes,
        and after the trigger was gated on `window.max_speech_tokens` the
        fixture pinned a path the engine could not take at all. Three mutations
        to `_windows_for_chunk` survived the whole suite.

        The case moves the window rather than the ceiling, because
        `AlgorithmConfig` refuses a chunk budget larger than the window and the
        sampler's cap has to move with it or the window is never what stops a
        row. What weights cannot reach here is a half that still overruns: a
        chunk is at most `window * CHARS_PER_TOKEN` characters, so a half is a
        quarter of the window. `tests/test_engine.py` covers that with a fake.
        """
        from dataclasses import replace

        import loudkit
        from loudkit.config import ExecutionConfig
        from loudkit.frontend.chunking import split_text
        from loudkit.frontend.speechtext import speech_text
        from loudkit.voice import VoiceProfile

        section = fixture.get("resplit")
        if not section:
            skip_or_fail("fixture has no resplit section")
        execution = ExecutionConfig(device="cpu", precision=section["execution"])
        base = loudkit.load(str(CKPT), device="cpu", execution=execution)
        voice = VoiceProfile.load(FIXTURE_DIR / section["voice"])
        assert base.algorithm.chunking.prefix_tokens == section["prefix_tokens"]
        assert base.algorithm.chunking.cap_resplit == "word"

        for case in section["cases"]:
            name, language, window = case["name"], case["language"], case["window"]
            a = base.algorithm
            algo = a.with_(
                window=replace(a.window, max_speech_tokens=window, static_length=window),
                sampling=replace(a.sampling, max_new_tokens=window),
                chunking=replace(a.chunking, max_tokens=window),
            )
            engine = loudkit.load(str(CKPT), device="cpu", execution=execution, algorithm=algo)
            prepared = speech_text(case["text"], language)
            assert prepared == case["prepared"], f"{name}: the speech funnel drifted"
            chunks = split_text(prepared, algo.chunking)
            want = case["windows"]
            assert len(want) > len(chunks), f"{name} recorded no re-split"

            got = list(engine.stream(case["text"], voice, seed=case["seed"], language=language))
            assert len(got) == len(want), (
                f"{name}: the engine produced {len(got)} windows, the fixture has "
                f"{len(want)} — a split that did not happen, or one that happened twice"
            )
            for i, (result, w) in enumerate(zip(got, want, strict=True)):
                where = f"{name} window {i} (chunk index {w['index']})"
                text = " ".join(span.text for span in result.chunks)
                assert text == w["text"], f"{where}: text"
                assert list(result.tokens) == w["tokens"], f"{where}: tokens"
                assert result.hit_token_cap == w["hit_cap"], f"{where}: cap flag"
            # Both halves of a repair carry the ORIGINAL chunk's index, so a
            # later chunk's seed cannot move. Nothing else in the suite says so.
            indices = [w["index"] for w in want]
            assert indices == sorted(indices), f"{name}: indices are not monotonic"
            assert max(indices) + 1 == len(chunks), (
                f"{name}: {max(indices) + 1} distinct indices for {len(chunks)} chunks"
            )

    def test_render_band(self, fixture: dict) -> None:
        needs_module("coremltools")
        import loudkit
        from loudkit.backends.coreml_backend import _assets_dir
        from loudkit.checkpoint import Checkpoint
        from loudkit.config import ExecutionConfig
        from loudkit.voice import VoiceProfile

        cases = fixture.get("end_to_end")
        if not cases:
            skip_or_fail("fixture has no end_to_end section")
        try:
            _assets_dir(Checkpoint.open(str(CKPT)))
        except FileNotFoundError as e:
            skip_or_fail(str(e))
        execution = ExecutionConfig(device="coreml", precision=cases[0]["execution"])
        engine = loudkit.load(str(CKPT), device="coreml", execution=execution)
        voice = VoiceProfile.load(FIXTURE_DIR / cases[0]["voice"])
        for case in cases:
            result = engine.synthesize_tokens(case["tokens"], voice, seed=case["seed"])
            mel_ref = np.fromfile(FIXTURE_DIR / case["mel"]["file"], dtype="<f4").reshape(
                case["mel"]["shape"]
            )
            wav_ref = np.fromfile(FIXTURE_DIR / case["wav"]["file"], dtype="<f4")
            assert result.mel.shape == mel_ref.shape
            # The length is asserted before the correlation, not discarded by
            # it: correlating `min(len(a), len(b))` samples makes a truncated
            # render score perfectly against the prefix it did produce, and
            # the missing tail — the end of the passage — is the finding. The
            # ports were fixed for exactly this; the Python side still had it.
            assert len(result.audio) == len(wav_ref), (
                f"{case['name']}: rendered {len(result.audio)} samples against a "
                f"{len(wav_ref)}-sample reference; correlating the shorter of the "
                "two would score a truncated render as a perfect one"
            )
            mel_corr = np.corrcoef(result.mel.ravel(), mel_ref.ravel())[0, 1]
            wave_corr = np.corrcoef(result.audio, wav_ref)[0, 1]
            assert mel_corr >= case["gates"]["mel_corr"], f"{case['name']} mel {mel_corr:.6f}"
            assert wave_corr >= case["gates"]["wave_corr"], (
                f"{case['name']} wave {wave_corr:.4f}"
            )
            assert_amplitude(case["name"], case["gates"]["wave_rms_db"], result.audio, wav_ref)

    def test_render_band_onnx(self, fixture: dict) -> None:
        """The ONNX renderer must land inside the same fixture band the CoreML
        renderer is gated on — the fixture is the contract, and a second
        backend does not get a second, looser bar."""
        needs_module("onnxruntime")
        import loudkit
        from loudkit.backends.onnx_backend import _assets_dir
        from loudkit.checkpoint import Checkpoint
        from loudkit.config import ExecutionConfig
        from loudkit.voice import VoiceProfile

        cases = fixture.get("end_to_end")
        if not cases:
            skip_or_fail("fixture has no end_to_end section")
        try:
            _assets_dir(Checkpoint.open(str(CKPT)))
        except FileNotFoundError as e:
            skip_or_fail(str(e))
        execution = ExecutionConfig(
            device="onnx",
            precision={
                "token_generator": "fp32",
                "mel_decoder.estimator": "fp32",
                "mel_decoder.encoder": "fp32",
                "vocoder": "fp32",
            },
        )
        engine = loudkit.load(str(CKPT), device="onnx", execution=execution)
        voice = VoiceProfile.load(FIXTURE_DIR / cases[0]["voice"])
        for case in cases:
            result = engine.synthesize_tokens(case["tokens"], voice, seed=case["seed"])
            mel_ref = np.fromfile(FIXTURE_DIR / case["mel"]["file"], dtype="<f4").reshape(
                case["mel"]["shape"]
            )
            wav_ref = np.fromfile(FIXTURE_DIR / case["wav"]["file"], dtype="<f4")
            assert result.mel.shape == mel_ref.shape
            # The length is asserted before the correlation, not discarded by
            # it: correlating `min(len(a), len(b))` samples makes a truncated
            # render score perfectly against the prefix it did produce, and
            # the missing tail — the end of the passage — is the finding. The
            # ports were fixed for exactly this; the Python side still had it.
            assert len(result.audio) == len(wav_ref), (
                f"{case['name']}: rendered {len(result.audio)} samples against a "
                f"{len(wav_ref)}-sample reference; correlating the shorter of the "
                "two would score a truncated render as a perfect one"
            )
            mel_corr = np.corrcoef(result.mel.ravel(), mel_ref.ravel())[0, 1]
            wave_corr = np.corrcoef(result.audio, wav_ref)[0, 1]
            assert mel_corr >= case["gates"]["mel_corr"], f"{case['name']} mel {mel_corr:.6f}"
            assert wave_corr >= case["gates"]["wave_corr"], (
                f"{case['name']} wave {wave_corr:.4f}"
            )
            assert_amplitude(case["name"], case["gates"]["wave_rms_db"], result.audio, wav_ref)


@requires("turbo_checkpoint")
@pytest.mark.parametrize("device", ["onnx", "coreml"])
def test_fused_graph_conformance(device: Literal["onnx", "coreml"]) -> None:
    """Both graph hosts consume the CPU reference's own model recipe."""
    checkpoint = asset("turbo_checkpoint")
    needs_module("onnxruntime" if device == "onnx" else "coremltools")
    import loudkit
    from loudkit.config import ExecutionConfig
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    rows = json.loads((FIXTURE_DIR / "vectors_fusion_mtp2.json").read_text(encoding="utf-8"))
    engine = loudkit.load(
        str(checkpoint),
        device=device,
        execution=ExecutionConfig(device=device, num_threads=1, onnx_provider="cpu"),
    )
    assert engine.algorithm.fingerprint() == rows["algorithm"]["fingerprint"]
    for case in rows["end_to_end"]:
        voice = VoiceProfile.load(FIXTURE_DIR / case["voice"])
        text = engine.frontend.encode(case["text"], case["language"])
        tokens = engine.token_generator.generate(
            text, voice, sampler=LRSamplerV1(engine.algorithm.sampling, seed=case["seed"])
        )
        assert [int(t) for t in tokens if t < engine.algorithm.start_speech_token] == case[
            "tokens"
        ]
        result = engine.synthesize_tokens(case["tokens"], voice, seed=case["seed"])
        mel = np.fromfile(FIXTURE_DIR / case["mel"]["file"], dtype="<f4").reshape(
            case["mel"]["shape"]
        )
        wave = np.fromfile(FIXTURE_DIR / case["wav"]["file"], dtype="<f4")
        assert result.mel.shape == mel.shape
        assert result.audio.shape == wave.shape
        assert np.corrcoef(result.mel.ravel(), mel.ravel())[0, 1] >= case["gates"]["mel_corr"]
        assert np.corrcoef(result.audio, wave)[0, 1] >= case["gates"]["wave_corr"]
        assert_amplitude(case["name"], case["gates"]["wave_rms_db"], result.audio, wave)

    chain = rows["long_form"]
    voice = VoiceProfile.load(FIXTURE_DIR / chain["voice"])
    for case in chain["cases"]:
        for chunk in case["chunks"]:
            text = engine.frontend.encode(chunk["text"], case["language"])
            tokens = engine.token_generator.generate(
                text,
                voice,
                sampler=LRSamplerV1(engine.algorithm.sampling, seed=int(chunk["seed"], 16)),
                prefix=chunk["prefix"],
            )
            assert [int(t) for t in tokens if t < engine.algorithm.start_speech_token] == chunk[
                "tokens"
            ]

        result = engine.synthesize(
            case["text"], voice, seed=case["seed"], language=case["language"]
        )
        assert list(result.tokens) == case["public_tokens"]

    if device == "coreml":
        import gc
        import time

        del engine
        gc.collect()
        time.sleep(1)  # Let native reset queues release their retained array views.


def _torch_generator_beside_coreml_tokens(
    checkpoint: Path, fixture: str, precision: Literal["fp32", "fp16"]
) -> None:
    import loudkit
    from loudkit.config import ExecutionConfig
    from loudkit.sampler import LRSamplerV1
    from loudkit.voice import VoiceProfile

    engine = loudkit.load(
        str(checkpoint),
        device="coreml",
        execution=ExecutionConfig(
            device="coreml",
            num_threads=1,
            generator_device="cpu",
            precision={"token_generator": precision},
        ),
    )
    rows = json.loads((FIXTURE_DIR / fixture).read_text(encoding="utf-8"))
    for case in rows["end_to_end"]:
        voice = VoiceProfile.load(FIXTURE_DIR / case["voice"])
        text = engine.frontend.encode(case["text"], case["language"])
        tokens = engine.token_generator.generate(
            text, voice, sampler=LRSamplerV1(engine.algorithm.sampling, seed=case["seed"])
        )
        got = [int(t) for t in tokens if t < engine.algorithm.start_speech_token]
        assert got == case["tokens"], f"{case['name']} at {precision}"


@requires("checkpoint")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_the_torch_generator_beside_coreml_keeps_the_fixture_tokens(
    precision: Literal["fp32", "fp16"],
) -> None:
    """``generator_device="cpu"`` keeps the torch generator under a CoreML renderer.

    The engine choice is a placement, and the precision under it is the one
    knob that could move a token. Measured on both fixture cases: fp16 and
    fp32 give the reference tokens exactly, so the flag buys speed on Apple
    silicon without changing what is said.
    """
    needs_module("coremltools")
    _torch_generator_beside_coreml_tokens(asset("checkpoint"), "vectors.json", precision)


@requires("turbo_checkpoint")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_the_torch_generator_beside_coreml_keeps_the_fusion_tokens(
    precision: Literal["fp32", "fp16"],
) -> None:
    """The same placement under ``fusion_mtp2``, held to its own fixture."""
    needs_module("coremltools")
    _torch_generator_beside_coreml_tokens(
        asset("turbo_checkpoint"), "vectors_fusion_mtp2.json", precision
    )
