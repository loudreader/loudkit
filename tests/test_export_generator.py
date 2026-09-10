"""The exported pair occupies one cache slot and preserves both prediction heads."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from tools.export_generator import Head2, PairStep, Prefill, require_decode, staging_directory

from loudkit.config import AlgorithmConfig
from loudkit.models.generator import TorchTokenGenerator


def generator():
    torch.manual_seed(17)
    return TorchTokenGenerator(
        replace(AlgorithmConfig(), decode_mode="fusion_mtp2"),
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "intermediate_size": 128,
            "head_dim": 32,
        },
        attention="eager",
    ).eval()


@pytest.mark.parametrize("prefix_length", [3, 9])
def test_pair_graphs_preserve_slots_and_heads(prefix_length):
    gen = generator()
    embeds = torch.randn(1, prefix_length, 64)
    with torch.no_grad():
        logits, hidden, *cache = Prefill(gen)(embeds, torch.arange(prefix_length))
        expected_hidden, expected_cache = gen.tfmr(
            embeds, torch.arange(prefix_length), None, attention="eager"
        )
        torch.testing.assert_close(logits, gen.speech_head(expected_hidden))
        torch.testing.assert_close(hidden, expected_hidden[:, -1])
        second = Head2(gen)(hidden, torch.tensor([7]))
        np.testing.assert_allclose(
            second.numpy(), gen._pair_second_logits(hidden, 7), atol=1e-6
        )
        slot = gen._pair_slot_embed(7, 11, 5)
        expected_hidden, expected_cache = gen.tfmr(
            slot, torch.tensor([prefix_length]), expected_cache, attention="eager"
        )
        outputs = PairStep(gen)(
            torch.tensor([[7, 11]]), torch.tensor([5]), torch.tensor([prefix_length]), *cache
        )
        torch.testing.assert_close(outputs[0], gen.speech_head(expected_hidden[:, -1]))
        torch.testing.assert_close(outputs[1], expected_hidden[:, -1])
        for got, want in zip(
            outputs[2:], (x for pair in expected_cache for x in pair), strict=True
        ):
            assert got.shape[2] == prefix_length + 1
            torch.testing.assert_close(got, want)


def test_unknown_decode_refused():
    with pytest.raises(SystemExit, match="no export graph set"):
        require_decode("unimplemented")


def test_failed_staging_leaves_existing_graphs(tmp_path):
    destination = tmp_path / "graphs"
    destination.mkdir()
    (destination / "t3_prefill.onnx").write_bytes(b"verified")
    staging = staging_directory(destination)
    (staging / "t3_prefill.onnx").write_bytes(b"failed")
    assert (destination / "t3_prefill.onnx").read_bytes() == b"verified"


@pytest.mark.parametrize(
    ("decode", "stages"), [("fusion_mtp2", {"step"}), ("single", {"prefill"})]
)
def test_incomplete_generator_graph_set_refused(decode, stages):
    from tools.export_generator import validate_stages

    with pytest.raises(SystemExit):
        validate_stages(decode, stages)


@pytest.mark.parametrize("exporter", ["tools.export_onnx", "tools.export_coreml"])
def test_invalid_stage_never_creates_output(exporter, tmp_path, monkeypatch):
    import importlib
    import sys

    module = importlib.import_module(exporter)
    destination = tmp_path / "graphs"
    monkeypatch.setattr(module.Checkpoint, "open", lambda _: object())
    monkeypatch.setattr(module, "production_algorithm", lambda _: AlgorithmConfig())
    monkeypatch.setattr(
        sys,
        "argv",
        ["export", "--checkpoint", "unused", "--out", str(destination), "--stages", "unknown"],
    )
    with pytest.raises(SystemExit, match="do not belong"):
        module.main()
    assert not destination.exists()


def test_coreml_decode_attention_across_kernel_boundary(tmp_path, monkeypatch):
    """KV length 113 used to select an inaccurate CoreML CPU matmul kernel."""
    import sys
    from pathlib import Path

    if sys.platform != "darwin":
        pytest.skip("CoreML prediction requires macOS")
    pytest.importorskip("coremltools")
    from tools.export_generator import _convert

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))

    class AttentionProduct(torch.nn.Module):
        def forward(self, probabilities, values):
            return probabilities @ values

    rng = np.random.default_rng(73)

    def inputs(length):
        probabilities = rng.random((1, 16, 1, length), dtype=np.float32)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        values = rng.standard_normal((1, 16, length, 64), dtype=np.float32)
        return probabilities, values

    example = tuple(torch.from_numpy(v) for v in inputs(8))
    run = _convert(
        AttentionProduct(),
        example,
        ["probabilities", "values"],
        ["context"],
        {"probabilities": {3: "past"}, "values": {2: "past"}},
        tmp_path / "attention.mlpackage",
        "coreml",
    )
    for length in (112, 113, 114, 129, 256):
        probabilities, values = inputs(length)
        actual = run([probabilities, values])[0]
        np.testing.assert_allclose(actual, probabilities @ values, atol=1e-6, rtol=1e-5)
