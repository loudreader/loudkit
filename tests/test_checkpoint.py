"""What a checkpoint is allowed to claim about itself.

A packed checkpoint carries its weights and a manifest describing them, and the
manifest is read first. That ordering is the whole subject of this file: numbers
a manifest declares are acted on before the tensors that would contradict them
are looked at, so anything the manifest can inflate is inflated for free.
"""

from __future__ import annotations

import pytest


class TestTheManifestCannotOutrunTheWeights:
    """A manifest is external data, and three of its numbers drive allocation.

    `vocab_size`, `hidden_size` and `num_hidden_layers` are read out of the
    manifest and handed to the model constructor before a single tensor is
    touched. Measured on the shipped checkpoint: changing `num_hidden_layers`
    from 16 to 200 — four characters — takes peak RSS from 752 MB to 2.6 GB, in
    5.6 s, from a file that need not have grown by a byte. `load_state_dict`
    rejects the mismatch afterwards, which is after the allocation.

    The check is against the weights rather than against a ceiling. Any ceiling
    is either low enough to refuse a legitimate future checkpoint or high enough
    to still be worth the trouble; the tensors in the same file cannot be
    inflated without inflating the file.
    """

    @staticmethod
    def _checkpoint(tmp_path, layers: int = 3, hidden: int = 8, vocab: int = 5):
        """A checkpoint carrying a manifest and the tensors it describes."""
        import json

        import numpy as np
        from safetensors.numpy import save_file

        tensors = {
            "t3.tfmr.embed_tokens.weight": np.zeros((vocab, hidden), dtype=np.float32),
        }
        # `gate_proj`, `q_proj` and `k_proj` carry the three dimensions that
        # used to go unchecked, and `gate_proj` is the expensive one: the model
        # constructor builds the MLP from `intermediate_size` before any state
        # dict is read, so a manifest naming a huge one is spent the moment it
        # is believed.
        intermediate = hidden * 2
        heads, kv_heads = 2, 1
        head_dim = hidden // heads
        for i in range(layers):
            tensors[f"t3.tfmr.layers.{i}.mlp.up_proj.weight"] = np.zeros(
                (hidden, hidden), dtype=np.float32
            )
            tensors[f"t3.tfmr.layers.{i}.mlp.gate_proj.weight"] = np.zeros(
                (intermediate, hidden), dtype=np.float32
            )
            tensors[f"t3.tfmr.layers.{i}.self_attn.q_proj.weight"] = np.zeros(
                (heads * head_dim, hidden), dtype=np.float32
            )
            tensors[f"t3.tfmr.layers.{i}.self_attn.k_proj.weight"] = np.zeros(
                (kv_heads * head_dim, hidden), dtype=np.float32
            )
        manifest = {
            "format": "loudkit-checkpoint",
            "format_version": 1,
            "llama_config": {
                "vocab_size": vocab,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "intermediate_size": intermediate,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
            },
        }
        path = tmp_path / "ckpt.safetensors"
        save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})
        return path

    def test_shapes_are_read_without_loading_tensors(self, tmp_path) -> None:
        """The header is the whole point: it is available before allocation."""
        from loudkit.checkpoint import Checkpoint

        ckpt = Checkpoint.open(self._checkpoint(tmp_path))
        shapes = ckpt.shapes("t3.")
        assert shapes["tfmr.embed_tokens.weight"] == (5, 8)
        # Four tensors per layer: up/gate projections and q/k, which are what
        # corroborate `intermediate_size` and the head counts.
        assert sum(1 for k in shapes if k.startswith("tfmr.layers.")) == 4 * 3

    def test_an_honest_manifest_is_accepted(self, tmp_path) -> None:
        from loudkit.backends.torch_backend import _check_architecture_against_weights
        from loudkit.checkpoint import Checkpoint

        ckpt = Checkpoint.open(self._checkpoint(tmp_path))
        _check_architecture_against_weights(ckpt, ckpt.manifest["llama_config"])

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("num_hidden_layers", 100_000),
            ("hidden_size", 65_536),
            ("vocab_size", 1 << 20),
            # The three that used to be waved through on the grounds that
            # `load_state_dict` would catch them "before anything larger than
            # the file itself has been allocated". It does not: the constructor
            # builds the MLP first, so `intermediate_size = 16_000_000` beside
            # 2100 rows of `gate_proj` asked for 196 GB and got past preflight.
            ("intermediate_size", 16_000_000),
            ("num_attention_heads", 4096),
            ("num_key_value_heads", 1024),
        ],
    )
    def test_a_manifest_the_weights_contradict_is_refused(
        self, tmp_path, field: str, value: int
    ) -> None:
        """And refused *naming both numbers*, because one of them is a lie and
        the message has to say which file disagrees with which."""
        from loudkit.backends.torch_backend import _check_architecture_against_weights
        from loudkit.checkpoint import Checkpoint

        ckpt = Checkpoint.open(self._checkpoint(tmp_path))
        crafted = dict(ckpt.manifest["llama_config"])
        crafted[field] = value

        with pytest.raises(ValueError, match=field) as caught:
            _check_architecture_against_weights(ckpt, crafted)
        assert str(value) in str(caught.value)

    @pytest.mark.parametrize(
        "omitted", ["mlp.gate_proj", "self_attn.q_proj", "self_attn.k_proj"]
    )
    def test_deleting_the_corroborating_tensor_is_not_a_way_past(
        self, tmp_path, omitted: str
    ) -> None:
        """The bypass these checks shipped with, for one afternoon.

        The `intermediate_size` comparison was conditional on `gate_proj` being
        in the file, and the head counts on `q_proj` and `k_proj` — written that
        way so a checkpoint predating them would still load. A crafted file that
        simply *omits* the tensor therefore skips the comparison, and a
        twenty-six kilobyte checkpoint claiming `intermediate_size = 16_000_000`
        walks past preflight and asks the constructor for 197 GB.

        The earlier tests all supply the tensor, which is why they passed while
        the door was open. There is nothing to be compatible with: a layer
        without a gate projection is not one this engine can build.
        """
        import json

        import numpy as np
        from safetensors.numpy import save_file

        from loudkit.backends.torch_backend import _check_architecture_against_weights
        from loudkit.checkpoint import Checkpoint

        full = self._checkpoint(tmp_path)
        source = Checkpoint.open(full)
        manifest = dict(source.manifest)
        crafted = {
            f"t3.{name}": np.zeros(shape, dtype=np.float32)
            for name, shape in source.shapes("t3.").items()
            if omitted not in name
        }
        path = tmp_path / "crafted.safetensors"
        save_file(crafted, str(path), metadata={"manifest": json.dumps(manifest)})

        ckpt = Checkpoint.open(path)
        crafted_config = dict(ckpt.manifest["llama_config"])
        crafted_config["intermediate_size"] = 16_000_000
        with pytest.raises(ValueError, match=omitted.rsplit(".", maxsplit=1)[-1]):
            _check_architecture_against_weights(ckpt, crafted_config)

    @pytest.mark.parametrize(
        ("tensor", "degenerate"),
        [
            ("mlp.gate_proj", (16_000_000, 0)),
            ("self_attn.q_proj", (16, 0)),
            ("self_attn.k_proj", (8, 1)),
        ],
    )
    def test_a_tensor_with_the_right_rows_and_wrong_columns_is_refused(
        self, tmp_path, tensor: str, degenerate: tuple[int, int]
    ) -> None:
        """Comparing only `shape[0]` was a hole of its own.

        A `(16_000_000, 0)` matrix weighs almost nothing on disk and satisfied
        an `intermediate_size` of sixteen million, after which the constructor
        asked for 197 GB. A projection here is always `(something,
        hidden_size)`, so the columns are as knowable as the rows and there was
        never a reason to skip them.
        """
        import json

        import numpy as np
        from safetensors.numpy import save_file

        from loudkit.backends.torch_backend import _check_architecture_against_weights
        from loudkit.checkpoint import Checkpoint

        source = Checkpoint.open(self._checkpoint(tmp_path))
        crafted = {
            f"t3.{name}": np.zeros(
                degenerate if name.endswith(f"{tensor}.weight") else shape, dtype=np.float32
            )
            for name, shape in source.shapes("t3.").items()
        }
        path = tmp_path / "wrong-columns.safetensors"
        save_file(crafted, str(path), metadata={"manifest": json.dumps(dict(source.manifest))})

        ckpt = Checkpoint.open(path)
        config = dict(ckpt.manifest["llama_config"])
        if tensor == "mlp.gate_proj":
            config["intermediate_size"] = degenerate[0]
        with pytest.raises(ValueError):
            _check_architecture_against_weights(ckpt, config)

    def test_a_checkpoint_with_no_embedding_is_refused_rather_than_trusted(
        self, tmp_path
    ) -> None:
        """No matrix to check against is not the same as nothing to check."""
        import json

        import numpy as np
        from safetensors.numpy import save_file

        from loudkit.backends.torch_backend import _check_architecture_against_weights
        from loudkit.checkpoint import Checkpoint

        path = tmp_path / "headless.safetensors"
        save_file(
            {"t3.tfmr.layers.0.mlp.up_proj.weight": np.zeros((4, 4), dtype=np.float32)},
            str(path),
            metadata={
                "manifest": json.dumps({"format": "loudkit-checkpoint", "format_version": 1})
            },
        )
        with pytest.raises(ValueError, match="embed_tokens"):
            _check_architecture_against_weights(
                Checkpoint.open(path), {"hidden_size": 99, "num_hidden_layers": 99}
            )


class TestTheDeclaredVersionCoversTheDeclaredLoop:
    """A manifest may not name a decode loop its version does not carry.

    `format_version` is the **portable** half of the two-token contract. The
    Rust, Go, TypeScript and Swift engines gate on that number and none of them
    parses the `decode` block, so a manifest saying `format_version 1` beside
    `decode.mode: "fusion_mtp2"` loads in all five and is misread in four:
    Python builds the fusion loop, the others run the one-token loop over
    two-token weights and speak fluent nonsense — the failure this format calls
    worse than a crash.

    Until this check existed the coupling was written down in
    `docs/design/two-token-decode.md` and enforced by `tools/pack_turbo.py`
    remembering to write a 2.
    """

    @staticmethod
    def _write(tmp_path, manifest: dict):
        import json

        import numpy as np
        from safetensors.numpy import save_file

        path = tmp_path / "ckpt.safetensors"
        save_file(
            {"t3.speech_emb.weight": np.zeros((4, 4), dtype=np.float32)},
            str(path),
            metadata={"manifest": json.dumps(manifest)},
        )
        return path

    def test_fusion_under_version_one_is_refused(self, tmp_path) -> None:
        from loudkit.checkpoint import Checkpoint

        path = self._write(
            tmp_path,
            {
                "format": "loudkit-checkpoint",
                "format_version": 1,
                "decode": {"mode": "fusion_mtp2"},
            },
        )
        with pytest.raises(ValueError, match="needs 2"):
            Checkpoint.open(path)

    def test_fusion_under_version_two_loads(self, tmp_path) -> None:
        from loudkit.checkpoint import Checkpoint

        path = self._write(
            tmp_path,
            {
                "format": "loudkit-checkpoint",
                "format_version": 2,
                "decode": {"mode": "fusion_mtp2"},
            },
        )
        assert Checkpoint.open(path).manifest["format_version"] == 2

    @pytest.mark.parametrize(
        "decode", [None, {"mode": "single"}], ids=["absent", "explicit-single"]
    )
    def test_every_0_1_0_file_still_opens(self, tmp_path, decode) -> None:
        """The single-token loop needs no minimum, and an absent block is its
        synonym. Every checkpoint shipped before this table existed is one of
        these two shapes, and both must keep loading unchanged."""
        from loudkit.checkpoint import Checkpoint

        manifest: dict = {"format": "loudkit-checkpoint", "format_version": 1}
        if decode is not None:
            manifest["decode"] = decode
        assert Checkpoint.open(self._write(tmp_path, manifest)) is not None

    @pytest.mark.parametrize(
        "declared", [None, [1], {"v": 1}, "one"], ids=["null", "list", "object", "words"]
    )
    def test_an_unreadable_version_is_the_documented_refusal(self, tmp_path, declared) -> None:
        """``read_manifest`` documents ``ValueError``, and a caller catches it.

        ``int()`` over a manifest value raises ``TypeError`` for ``null``, a
        list and an object, so three of these four crashed past every handler
        written against the documented type.
        """
        from loudkit.checkpoint import Checkpoint

        path = self._write(
            tmp_path, {"format": "loudkit-checkpoint", "format_version": declared}
        )

        with pytest.raises(ValueError, match="expected a version number"):
            Checkpoint.open(path)

    @pytest.mark.parametrize(
        "declared", [1, 1.0, "1", True], ids=["int", "float", "str", "bool"]
    )
    def test_every_version_int_accepted_before_still_opens(self, tmp_path, declared) -> None:
        """Only the type of the refusal moved. Which files open did not, so
        the four ports' disagreements about these spellings stay where they
        were rather than being settled here by accident."""
        from loudkit.checkpoint import Checkpoint

        path = self._write(
            tmp_path, {"format": "loudkit-checkpoint", "format_version": declared}
        )

        assert Checkpoint.open(path) is not None

    def test_the_ports_gate_on_the_number_this_table_defends(self) -> None:
        """Why the check is in the loader and not in `AlgorithmConfig`.

        Python reads a mislabelled manifest correctly; the four ports cannot,
        because they read the version and nothing else. If a port ever starts
        reading `decode.mode` this assertion is what says the two mechanisms
        have to move together.
        """
        from loudkit.checkpoint import DECODE_FORMAT_VERSION, SUPPORTED_FORMAT_VERSIONS

        assert DECODE_FORMAT_VERSION["fusion_mtp2"] == 2
        assert 2 in SUPPORTED_FORMAT_VERSIONS, "Python is the engine that reads it"


class TestGraphSets:
    def test_decode_selects_its_graphs(self) -> None:
        from loudkit.backends.onnx_backend import graph_names
        from loudkit.config import AlgorithmConfig
        from loudkit.manifest import decode_from

        assert "t3_step.onnx" in graph_names(AlgorithmConfig())
        fused = graph_names(AlgorithmConfig(decode_mode=decode_from({"mode": "fusion_mtp2"})))
        assert "t3_step.onnx" not in fused
        assert {"t3_pair_step.onnx", "t3_head2.onnx"} <= set(fused)


class TestWhichBackendsRunWhichDecodeLoop:
    """One door, in front of every entry point, for "turbo is torch-only".

    Two callers ask the same question and used to answer it in different
    places or not at all: `hub.verify_release_inventory`, which judges a fetch,
    and `backends.build_engine`, which judges a local path. Without this the
    first answered a turbo repo with a list of exported graphs the turbo
    release was never going to carry, and the second failed inside a backend.

    The predicate lives in `checkpoint` because it is a statement about what a
    manifest declares, and because both callers may import this module.
    """

    def test_a_manifest_with_no_decode_block_is_single(self) -> None:
        from loudkit.checkpoint import decode_mode

        assert decode_mode({}) == "single"
        assert decode_mode({"decode": {"mode": "fusion_mtp2"}}) == "fusion_mtp2"
        assert decode_mode({"decode": "nonsense"}) == "single"

    def test_single_token_weights_run_everywhere(self) -> None:
        from loudkit.checkpoint import require_decode_support

        for target in ("torch", "onnx", "coreml", "cpu", "cuda:1", "mps"):
            require_decode_support("single", target)

    @pytest.mark.parametrize("target", ["torch", "cpu", "cuda", "cuda:1", "mps"])
    def test_fusion_runs_on_torch_and_its_devices(self, target: str) -> None:
        from loudkit.checkpoint import require_decode_support

        require_decode_support("fusion_mtp2", target)

    @pytest.mark.parametrize("target", ["onnx", "coreml"])
    def test_fusion_runs_on_graph_backends(self, target: str) -> None:
        from loudkit.checkpoint import require_decode_support

        require_decode_support("fusion_mtp2", target)

    def test_the_modes_it_accepts_are_the_modes_the_type_names(self) -> None:
        """One list, two spellings: the literal set inside
        `require_decode_support` and `config.DecodeMode`.

        `checkpoint` is a leaf and cannot import `config` to share the set, so
        the two are compared here, where both are in scope. A mode added to one
        and not the other is a checkpoint the config accepts and the reader
        refuses, or the reverse.
        """
        from typing import get_args

        from loudkit.checkpoint import _KNOWN_DECODE_MODES, require_decode_support
        from loudkit.config import DecodeMode

        # Both directions. Looping the type's members only would pass for a
        # reader that also accepted a mode the config has never heard of.
        assert get_args(DecodeMode) == _KNOWN_DECODE_MODES
        for mode in get_args(DecodeMode):
            require_decode_support(mode, "torch")
        with pytest.raises(ValueError, match="unsupported decode mode"):
            require_decode_support("fusion_mtp3", "torch")

    def test_the_engine_builder_refuses_before_it_builds(self, tmp_path) -> None:
        """The door is in `build_engine`, not in each backend.

        Driven against a backend registered for this test rather than against
        onnx or coreml, because the point is that nothing reached a builder at
        all: a test that needed onnxruntime installed would be exercising that
        backend's own guard instead of this one.
        """
        import json

        import numpy as np
        from safetensors.numpy import save_file

        from loudkit.backends import _REGISTRY, build_engine, register_backend

        path = tmp_path / "turbo.safetensors"
        save_file(
            {"t3.speech_emb.weight": np.zeros((4, 4), dtype=np.float32)},
            str(path),
            metadata={
                "manifest": json.dumps(
                    {
                        "format": "loudkit-checkpoint",
                        "format_version": 2,
                        "name": "loudkit-v0.1-turbo",
                        "decode": {"mode": "unsupported"},
                    }
                )
            },
        )

        reached: list[object] = []

        @register_backend("graphtest")
        def _never(ckpt, execution, algorithm):
            reached.append(ckpt)
            raise AssertionError("the refusal is the point")

        try:
            with pytest.raises(ValueError, match="this build decodes"):
                build_engine(str(path), device="graphtest")
        finally:
            _REGISTRY.pop("graphtest", None)
        assert not reached, "the backend was entered; the door is in the wrong place"


class TestThePayloadHashNamesTheHeaderItRefused:
    """`payload_sha256` reads an external header, so its refusals have to be
    distinguishable. Two of them are one condition apart: a `dtype` that is not a
    string at all, and a string naming a dtype the recipe does not cover.
    """

    @staticmethod
    def _one_tensor(tmp_path, entry: object):
        """A safetensors file whose header holds one tensor entry, verbatim."""
        import json
        import struct

        header = json.dumps({"t": entry}).encode()
        path = tmp_path / "one.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
        return path

    def test_a_non_string_dtype_is_not_reported_as_an_unsupported_one(self, tmp_path) -> None:
        from loudkit.checkpoint import payload_sha256

        path = self._one_tensor(tmp_path, {"dtype": 7, "shape": [4], "data_offsets": [0, 16]})
        with pytest.raises(ValueError, match="has no string dtype, got 7"):
            payload_sha256(path)

    def test_a_string_dtype_outside_the_recipe_names_the_dtype(self, tmp_path) -> None:
        from loudkit.checkpoint import payload_sha256

        path = self._one_tensor(
            tmp_path, {"dtype": "C64", "shape": [4], "data_offsets": [0, 16]}
        )
        with pytest.raises(ValueError, match="has unsupported dtype 'C64'"):
            payload_sha256(path)
