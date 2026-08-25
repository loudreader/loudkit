"""Pins for the release-integrity and loopback-guard hardening.

Each test here locks behaviour that once was missing: checksums parsed
permissively enough to verify nothing, a ``Host`` header naming any origin
accepted by a server that trusts its bind, voice profiles written
world-readable, and an environment token losing to a missing flag.
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from loudkit.hub import _parse_sha256sums, _verify_sha256sums
from loudkit.transports.http import _host_is_loopback


def _write_sums(root: Path, entries: dict[str, str]) -> None:
    lines = [f"{digest}  {name}" for name, digest in entries.items()]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TestSha256SumsParsing:
    def test_a_well_formed_manifest_parses(self, tmp_path: Path) -> None:
        entries = {
            "loudr-1.safetensors": "a" * 64,
            "voices/kathleen.safetensors": "b" * 64,
        }
        _write_sums(tmp_path, entries)
        assert _parse_sha256sums(tmp_path / "SHA256SUMS") == entries

    def test_crlf_line_endings_are_tolerated(self, tmp_path: Path) -> None:
        (tmp_path / "SHA256SUMS").write_bytes(f"{'c' * 64}  loudr-1.safetensors\r\n".encode())
        assert _parse_sha256sums(tmp_path / "SHA256SUMS") == {"loudr-1.safetensors": "c" * 64}

    def test_a_mangled_line_is_refused_not_skipped(self, tmp_path: Path) -> None:
        # One space instead of two: sha256sum -c would read the whole line as a
        # filename, and a permissive parser here would verify nothing while
        # reporting success.
        line = f"{'d' * 64} loudr-1.safetensors\n"
        (tmp_path / "SHA256SUMS").write_text(line, encoding="utf-8")
        with pytest.raises(ValueError, match="malformed SHA256SUMS"):
            _parse_sha256sums(tmp_path / "SHA256SUMS")

    def test_a_short_digest_is_refused(self, tmp_path: Path) -> None:
        line = f"{'d' * 32}  loudr-1.safetensors\n"
        (tmp_path / "SHA256SUMS").write_text(line, encoding="utf-8")
        with pytest.raises(ValueError, match="malformed SHA256SUMS"):
            _parse_sha256sums(tmp_path / "SHA256SUMS")


class TestSnapshotVerification:
    def test_matching_files_pass(self, tmp_path: Path) -> None:
        payload = b"checkpoint bytes"
        (tmp_path / "loudr-1.safetensors").write_bytes(payload)
        _write_sums(tmp_path, {"loudr-1.safetensors": _digest(payload)})
        _verify_sha256sums(tmp_path)
        assert (tmp_path / ".loudkit-verified").is_file()

    def test_a_substituted_file_fails(self, tmp_path: Path) -> None:
        (tmp_path / "loudr-1.safetensors").write_bytes(b"evil")
        _write_sums(tmp_path, {"loudr-1.safetensors": _digest(b"checkpoint bytes")})
        with pytest.raises(ValueError, match="failed the release checksum"):
            _verify_sha256sums(tmp_path)

    def test_entries_for_unfetched_files_are_skipped(self, tmp_path: Path) -> None:
        """allow_patterns fetches a subset; absent entries are not failures."""
        payload = b"checkpoint bytes"
        (tmp_path / "loudr-1.safetensors").write_bytes(payload)
        _write_sums(
            tmp_path,
            {
                "loudr-1.safetensors": _digest(payload),
                "onnx/decoder.onnx": "e" * 64,  # never fetched by load()
            },
        )
        _verify_sha256sums(tmp_path)

    def test_the_marker_makes_a_cached_snapshot_hash_once(self, tmp_path: Path) -> None:
        payload = b"checkpoint bytes"
        target = tmp_path / "loudr-1.safetensors"
        target.write_bytes(payload)
        _write_sums(tmp_path, {"loudr-1.safetensors": _digest(payload)})
        _verify_sha256sums(tmp_path)
        # The cache semantics: verification happened once, and a snapshot whose
        # manifest, sizes and mtimes are unchanged is not re-hashed on every
        # load(). Written here with the size and the timestamp restored, which
        # is the only way a mutation stays invisible — the bytes differ, and
        # they are not read.
        before = target.stat()
        target.write_bytes(b"y" * len(payload))
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        _verify_sha256sums(tmp_path)  # must not raise

    def test_a_file_whose_stat_moved_is_hashed_again(self, tmp_path: Path) -> None:
        """Trust-on-first-use, minus the failures a stat can see.

        Truncation, replacement and rewriting all move the size or the mtime;
        the marker records both, so a sealed cache does not hide them.
        """
        payload = b"checkpoint bytes"
        target = tmp_path / "loudr-1.safetensors"
        target.write_bytes(payload)
        _write_sums(tmp_path, {"loudr-1.safetensors": _digest(payload)})
        _verify_sha256sums(tmp_path)
        target.write_bytes(b"truncated")
        with pytest.raises(ValueError, match="failed the release checksum"):
            _verify_sha256sums(tmp_path)

    def test_a_new_manifest_invalidates_the_marker(self, tmp_path: Path) -> None:
        payload = b"checkpoint bytes"
        (tmp_path / "loudr-1.safetensors").write_bytes(payload)
        _write_sums(tmp_path, {"loudr-1.safetensors": _digest(payload)})
        _verify_sha256sums(tmp_path)
        (tmp_path / "loudr-1.safetensors").write_bytes(b"changed after verification")
        _write_sums(tmp_path, {"loudr-1.safetensors": _digest(b"changed after verification")})
        _verify_sha256sums(tmp_path)  # new manifest, re-verified, marker refreshed


class TestLoopbackHostPin:
    @pytest.mark.parametrize(
        ("host", "allowed"),
        [
            ("localhost", True),
            ("localhost:8765", True),
            ("127.0.0.1", True),
            ("127.0.0.1:8765", True),
            ("::1", True),
            ("[::1]:8765", True),
            ("[::1]", True),
            ("evil.com", False),
            ("evil.com:80", False),
            ("127.0.0.1.evil.com", False),
            ("testserver", False),
        ],
    )
    def test_host_classification(self, host: str, allowed: bool) -> None:
        assert _host_is_loopback(host) is allowed

    def test_an_ipv6_bind_serves_its_own_loopback(self) -> None:
        """``serve --host ::1`` is legal; its Host header must not 403."""
        assert _host_is_loopback("::1")
        assert _host_is_loopback("[::1]:8765")


class TestProfilePermissions:
    @pytest.mark.skipif(os.name != "posix", reason="0600 is a POSIX mode, not a Windows ACL")
    def test_a_saved_profile_is_owner_only(self, tmp_path: Path) -> None:
        """A profile derives from a recording of a person: 0600, not 0644."""
        import numpy as np

        from loudkit import VoiceProfile

        rng = np.random.default_rng(7)
        profile = VoiceProfile(
            name="kathleen",
            source_sample_rate=16000,
            speaker_embedding=rng.normal(size=256).astype(np.float32),
            flow_embedding=rng.normal(size=192).astype(np.float32),
            prompt_tokens=np.array([1, 2, 3], dtype=np.int64),
            prompt_mel=rng.normal(size=(80, 10)).astype(np.float32),
            cond_prompt_tokens=np.array([1, 2, 3], dtype=np.int64),
        )
        path = tmp_path / "kathleen.safetensors"
        profile.save(path)
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


class TestTokenPrecedence:
    def test_the_flag_wins_over_the_environment(self, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
        import loudkit.transports.http as server_mod

        monkeypatch.setenv("LOUDKIT_TOKEN", "from-env")
        captured: dict[str, str | None] = {}
        original = server_mod.serve

        def capture(*args: object, **kwargs: object) -> None:
            captured["token"] = kwargs.get("token")
            captured["allow_public"] = kwargs.get("allow_public")

        monkeypatch.setattr(server_mod, "serve", capture)
        from loudkit.cli import main

        (tmp_path / "ckpt.safetensors").write_bytes(b"x")
        main(
            [
                "serve",
                "--checkpoint",
                str(tmp_path / "ckpt.safetensors"),
                "--allow-public",
                "--token",
                "from-flag",
            ]
        )
        assert captured["token"] == "from-flag"

        monkeypatch.delenv("LOUDKIT_TOKEN")
        main(
            [
                "serve",
                "--checkpoint",
                str(tmp_path / "ckpt.safetensors"),
                "--allow-public",
                "--token",
                "from-flag",
            ]
        )
        assert captured["token"] == "from-flag"

        monkeypatch.setenv("LOUDKIT_TOKEN", "from-env")
        main(["serve", "--checkpoint", str(tmp_path / "ckpt.safetensors"), "--allow-public"])
        assert captured["token"] == "from-env"

        monkeypatch.delenv("LOUDKIT_TOKEN")
        main(["serve", "--checkpoint", str(tmp_path / "ckpt.safetensors"), "--allow-public"])
        assert captured["token"] is None  # serve generates its own, printed to stderr
        assert original  # the real function was only ever replaced, not called
