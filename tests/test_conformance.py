"""The cross-language conformance vectors, checked against this implementation.

``tests/data/conformance`` is read by two test suites: this one, and the Swift
package's ``swift test``. The fixture is the contract — if Python drifts from
it, this file fails; if Swift drifts from it, the Swift tests fail; if both
pass, the two implementations agree without ever having met.

The weight-free sections (Philox, sampler, frontend, seeds) run everywhere.
The algorithm-identity and end-to-end sections need the packed checkpoint and
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

import numpy as np
import pytest

from .assets import asset, needs_module, requires, skip_or_fail

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
            got = philox_4x32_10(
                *(np.array([c], dtype=np.uint64) for c in case["counter"]),
                case["key"][0],
                case["key"][1],
            )
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
        from loudkit.engine import _derive

        for probe in fixture["seeds"]["derivation"]:
            assert hex(_derive(probe["seed"], probe["stream"])) == probe["derived"]


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
        from loudkit.frontend.polish import speech_text
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
