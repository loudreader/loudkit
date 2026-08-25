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
