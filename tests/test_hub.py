"""What the resolvers say when a fetch fails.

The translation from the Hugging Face client's exceptions into this library's
own is the entire user-facing surface of a failed first command: someone who
mistyped a repo id, or who is offline, sees one sentence and nothing else. It
has to name the thing that is actually wrong.

Driven against a fake client because the real failures need a network — and
because the translation matches on the exception's *name* (the client is an
optional extra, so its types cannot be imported here), which is exactly what a
fake reproduces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from loudkit import hub
from loudkit.errors import VoiceNotFoundError


class RepositoryNotFoundError(Exception):
    """The names below are the contract: `hub` matches by name, not by type."""


class RevisionNotFoundError(Exception):
    pass


class EntryNotFoundError(Exception):
    pass


class LocalEntryNotFoundError(FileNotFoundError):
    """Offline with nothing cached. A `FileNotFoundError` in the real client too."""


class _Client:
    """A hub client that fails one way, whatever it is asked for."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def hf_hub_download(self, **kwargs: object) -> str:
        raise self._exc

    def list_repo_files(self, **kwargs: object) -> list[str]:
        raise self._exc


@pytest.fixture
def failing(monkeypatch):  # type: ignore[no-untyped-def]
    def install(exc: BaseException) -> None:
        monkeypatch.setattr(hub, "_hub", lambda: _Client(exc))

    return install


class TestResolveVoiceNamesWhatIsWrong:
    def test_a_missing_repo_is_not_reported_as_a_missing_voice(self, failing) -> None:
        """``kathleen: no voice by that name in loudreader/typo`` sent the
        reader hunting for a misspelt voice in the one word they got right."""
        failing(RepositoryNotFoundError("401 Client Error: Invalid username or password"))
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_voice("kathleen", repo="loudreader/typo")
        assert not isinstance(caught.value, VoiceNotFoundError)
        assert "repository not found" in str(caught.value)
        assert "no voice by that name" not in str(caught.value)

    def test_an_unreachable_hub_is_not_reported_as_a_missing_voice(self, failing) -> None:
        failing(LocalEntryNotFoundError("Connection error, and we cannot find ..."))
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_voice("kathleen", repo="loudreader/loudr-1")
        assert not isinstance(caught.value, VoiceNotFoundError)
        assert "cannot be reached" in str(caught.value)

    def test_a_missing_revision_names_the_revision(self, failing) -> None:
        failing(RevisionNotFoundError("404"))
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_voice("kathleen", repo="loudreader/loudr-1", revision="v9")
        assert "revision 'v9' not found" in str(caught.value)

    def test_a_repo_that_answered_still_gives_a_voice_error(self, failing) -> None:
        """The repo is there and the file in it is not: that *is* a missing
        voice, and the library has an error for it."""
        failing(EntryNotFoundError("404 voices/kathleen.safetensors"))
        with pytest.raises(VoiceNotFoundError) as caught:
            hub.resolve_voice("kathleen", repo="loudreader/loudr-1")
        assert "no voice by that name in loudreader/loudr-1" in str(caught.value)

    def test_an_unrelated_failure_keeps_its_own_traceback(self, failing) -> None:
        """A timeout or a proxy 500 is not an answer to "is there such a
        voice"; swallowing it as one would report a network fault as a typo."""
        failing(TimeoutError("read timed out"))
        with pytest.raises(TimeoutError):
            hub.resolve_voice("kathleen", repo="loudreader/loudr-1")


class TestResolveCheckpointNamesWhatIsWrong:
    def test_an_unreachable_hub_says_so(self, monkeypatch) -> None:
        """The same diagnosis on the checkpoint path, where the client's own
        message is a paragraph about `HF_HUB_OFFLINE`."""

        class _Snapshot:
            def snapshot_download(self, **kwargs: object) -> str:
                raise LocalEntryNotFoundError("Connection error")

        client = _Snapshot()
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(FileNotFoundError, match="cannot be reached"):
            hub.resolve_checkpoint("loudreader/loudr-1")


class _SumsClient:
    """A hub client serving the release's bookkeeping files, for the one-file
    path.

    `resolve_voice` and `resolve_voice_encoder` fetch a single file and then
    ask the release's manifest — and, for an official repo, `release.json` —
    about it; these tests drive that second half directly, since the first
    half is `hf_hub_download` and belongs to the client.
    """

    def __init__(self, files: dict[str, str] | None) -> None:
        self.files = files or {}

    def hf_hub_download(self, *, filename: str, **kwargs: object) -> str:
        path = self.files.get(filename)
        if path is None:
            raise EntryNotFoundError(f"404 {filename}")
        return path


def _write_sums(root, entries: dict[str, str]) -> None:
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in entries.items()), encoding="utf-8"
    )


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _releasable(root) -> dict[str, str]:
    """A releasable ``release.json`` in ``root``; returns its checksum entry.

    An official snapshot has to say what it is, ``profile: full-0.1`` and
    ``verified: true``, before anything else about it is judged, so every
    official fixture in this file carries this and splices the returned entry
    into its ``SHA256SUMS``. What each test attacks is then the one property
    it names, not this precondition.
    """
    import json

    body = json.dumps({"profile": "full-0.1", "verified": True}) + "\n"
    # Exact bytes: text-mode writes translate LF to CRLF on Windows, while the
    # digest below is over UTF-8 with LF. A fixture must not disagree with
    # itself because of the host's newline convention.
    (root / "release.json").write_bytes(body.encode())
    return {"release.json": _sha256(body.encode())}


class TestSnapshotIntegrity:
    """What "verified" is allowed to mean for a downloaded release.

    A checksum file that is merely consulted proves nothing: before this, a
    release with no manifest verified silently, and a file the manifest did
    not mention was never looked at. That was the defect this regression test
    was added to prevent in future releases.
    """

    def test_a_good_manifest_passes_and_marks_the_snapshot(self, tmp_path) -> None:
        payload = b"weights"
        (tmp_path / "loudr-1.safetensors").write_bytes(payload)
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(payload), **_releasable(tmp_path)},
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        assert (tmp_path / ".loudkit-verified").is_file()

    def test_a_corrupted_file_fails_by_name(self, tmp_path) -> None:
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        (tmp_path / "voices" / "kathleen.safetensors").parent.mkdir()
        (tmp_path / "voices" / "kathleen.safetensors").write_bytes(b"tampered")
        _write_sums(
            tmp_path,
            {
                "loudr-1.safetensors": _sha256(b"weights"),
                "voices/kathleen.safetensors": _sha256(b"kathleen"),
                **_releasable(tmp_path),
            },
        )
        with pytest.raises(ValueError) as caught:
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        assert "voices/kathleen.safetensors" in str(caught.value)
        assert "loudr-1.safetensors," not in str(caught.value)  # the good one is not accused

    def test_an_official_release_without_a_manifest_is_refused(self, tmp_path) -> None:
        """Every loudreader release ships one; arriving without it is a defect,
        not an old-fashioned upload."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        with pytest.raises(ValueError, match="no SHA256SUMS"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_third_party_release_without_a_manifest_is_allowed(self, tmp_path) -> None:
        """There is nothing to check against and no expectation to violate."""
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        hub._verify_sha256sums(tmp_path, repo="somebody/their-model")
        hub._verify_sha256sums(tmp_path)  # a local tree, verified by hand

    def test_unlisted_weights_are_refused(self, tmp_path) -> None:
        """The bytes loudkit opens. An unlisted one is weights nothing vouches
        for, arriving inside a snapshot that otherwise verified."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        (tmp_path / "voices").mkdir()
        (tmp_path / "voices" / "extra.safetensors").write_bytes(b"unvouched")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        with pytest.raises(ValueError, match="voices/extra.safetensors"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_an_unlisted_non_weight_is_refused_in_an_official_release(self, tmp_path) -> None:
        """The builder checksums every file a release ships, so inside a
        loudreader snapshot an uncovered file has no honest origin, and a file
        another tool fetched from the same revision is in the manifest by the
        same rule. Only a third-party snapshot keeps the warning, because no
        builder promised coverage there."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "decoder.onnx").write_bytes(b"graph")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        with pytest.raises(ValueError, match="onnx/decoder.onnx"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_an_unlisted_non_weight_is_reported_for_a_third_party(self, tmp_path) -> None:
        """A stranger's repo made no coverage promise, so an uncovered file is
        named, not fatal."""
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        (tmp_path / "notes.txt").write_bytes(b"unvouched")
        _write_sums(tmp_path, {"model.safetensors": _sha256(b"weights")})
        with pytest.warns(UserWarning, match="notes.txt"):
            hub._verify_sha256sums(tmp_path, repo="somebody/their-model")

    def test_the_files_a_manifest_cannot_list_are_not_reported(self, tmp_path) -> None:
        """`SHA256SUMS` holds no digest of itself, `.loudkit-verified` is
        written here, and `.gitattributes` and `.cache/` are the hub client's
        own furniture. `release.json` is not on that list: the builder writes
        it before the manifest and covers it, so here it is verified like any
        other file."""
        import warnings

        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        (tmp_path / ".gitattributes").write_text("*.safetensors filter=lfs\n", encoding="utf-8")
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "download.metadata").write_text("x", encoding="utf-8")
        (tmp_path / ".cache" / "huggingface").mkdir()
        (tmp_path / ".cache" / "huggingface" / ".gitignore").write_text("*\n", encoding="utf-8")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_an_unchecksummed_release_json_is_refused(self, tmp_path) -> None:
        """It states the profile the bundle was built from and whether the
        checkpoint was verified. A copy nothing vouches for is the one worth
        hearing about, so far from an exemption it is fatal: an official
        release covers every file, this one included."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        _releasable(tmp_path)
        _write_sums(tmp_path, {"loudr-1.safetensors": _sha256(b"weights")})
        with pytest.raises(ValueError, match="release.json"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_hidden_checkpoint_is_refused_like_any_other(self, tmp_path) -> None:
        """The exemption used to be a shape, not a list: anything dot-prefixed
        skipped the inventory, and `Path.glob("*.safetensors")` matches
        `.hidden.safetensors`. So the one file class whose bytes matter most
        had the one route past the manifest."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        (tmp_path / ".hidden.safetensors").write_bytes(b"unvouched")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        with pytest.raises(ValueError, match=r"\.hidden\.safetensors"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_hidden_non_weight_is_not_waved_through(self, tmp_path) -> None:
        """Same rule, arriving dot-prefixed: unnamed bookkeeping inside an
        official snapshot is uncovered, and uncovered is fatal there."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        (tmp_path / ".env").write_text("TOKEN=x\n", encoding="utf-8")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        with pytest.raises(ValueError, match=r"\.env"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_hidden_checkpoint_is_not_a_candidate_for_the_loader(self, tmp_path) -> None:
        """Belt and braces on the other side of the glob, for the local case
        that has no manifest to be caught by: `resolve_checkpoint("./dir")`."""
        (tmp_path / ".hidden.safetensors").write_bytes(b"unvouched")
        with pytest.raises(FileNotFoundError, match="no \\*.safetensors here"):
            hub._only_checkpoint_in(tmp_path)
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        assert hub._only_checkpoint_in(tmp_path).name == "loudr-1.safetensors"

    def test_the_hash_is_chunked_and_agrees_with_a_whole_file_digest(self, tmp_path) -> None:
        """The checkpoint is 1.27 GB: hashing it whole costs more memory than
        loading it. Several blocks' worth, against the digest `sha256sum`
        prints for the same bytes."""
        from loudkit.checkpoint import file_sha256

        payload = bytes(range(256)) * 20_000  # ~5 MB, past the 1 MB block
        target = tmp_path / "loudr-1.safetensors"
        target.write_bytes(payload)
        assert file_sha256(target) == _sha256(payload)
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(payload), **_releasable(tmp_path)},
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")


class TestSingleFileIntegrity:
    """`resolve_voice` and `resolve_voice_encoder` fetch one file each.

    A profile is derived from a recording of a person, so these are the
    artefacts most worth guarding — and the ones where a manifest that stays
    silent about the file used to count as a pass.
    """

    def _official_client(self, tmp_path, entries: dict[str, str]) -> _SumsClient:
        """A client serving a releasable official repo whose sums hold ``entries``."""
        _write_sums(tmp_path, {**entries, **_releasable(tmp_path)})
        return _SumsClient(
            {
                "SHA256SUMS": str(tmp_path / "SHA256SUMS"),
                "release.json": str(tmp_path / "release.json"),
            }
        )

    def test_a_manifest_that_does_not_cover_the_file_is_refused(self, tmp_path) -> None:
        client = self._official_client(tmp_path, {"loudr-1.safetensors": "a" * 64})
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        with pytest.raises(ValueError, match="does not list voices/kathleen.safetensors"):
            hub._verify_against_release_sums(
                client,
                "loudreader/loudr-1",
                None,
                "voices/kathleen.safetensors",
                voice,
            )

    def test_a_covered_file_passes_and_a_tampered_one_does_not(self, tmp_path) -> None:
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        client = self._official_client(
            tmp_path, {"voices/kathleen.safetensors": _sha256(b"profile")}
        )
        hub._verify_against_release_sums(
            client, "loudreader/loudr-1", None, "voices/kathleen.safetensors", voice
        )
        voice.write_bytes(b"tampered")
        with pytest.raises(ValueError, match="failed the release checksum"):
            hub._verify_against_release_sums(
                client, "loudreader/loudr-1", None, "voices/kathleen.safetensors", voice
            )

    def test_an_official_fetch_requires_a_verified_release_record(self, tmp_path) -> None:
        """The snapshot path refuses an official bundle whose release.json
        does not say `verified: true`; a voice fetched alone used to skip that
        claim entirely, so the same repo was strict through one door and
        lenient through the other."""
        import json

        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        body = json.dumps({"profile": "full-0.1", "verified": False}) + "\n"
        (tmp_path / "release.json").write_bytes(body.encode())
        _write_sums(
            tmp_path,
            {
                "voices/kathleen.safetensors": _sha256(b"profile"),
                "release.json": _sha256(body.encode()),
            },
        )
        client = _SumsClient(
            {
                "SHA256SUMS": str(tmp_path / "SHA256SUMS"),
                "release.json": str(tmp_path / "release.json"),
            }
        )
        with pytest.raises(ValueError, match="verified"):
            hub._verify_against_release_sums(
                client, "loudreader/loudr-1", None, "voices/kathleen.safetensors", voice
            )

    def test_an_official_fetch_without_a_release_record_is_refused(self, tmp_path) -> None:
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        _write_sums(tmp_path, {"voices/kathleen.safetensors": _sha256(b"profile")})
        client = _SumsClient({"SHA256SUMS": str(tmp_path / "SHA256SUMS")})
        with pytest.raises(ValueError, match="release.json"):
            hub._verify_against_release_sums(
                client, "loudreader/loudr-1", None, "voices/kathleen.safetensors", voice
            )

    def test_a_third_party_fetch_needs_no_release_record(self, tmp_path) -> None:
        """No builder promised a record there; the digest is the whole claim."""
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        _write_sums(tmp_path, {"voices/kathleen.safetensors": _sha256(b"profile")})
        client = _SumsClient({"SHA256SUMS": str(tmp_path / "SHA256SUMS")})
        hub._verify_against_release_sums(
            client, "somebody/voices", None, "voices/kathleen.safetensors", voice
        )

    def test_an_official_release_without_a_manifest_is_refused(self, tmp_path) -> None:
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        with pytest.raises(ValueError, match="no SHA256SUMS"):
            hub._verify_against_release_sums(
                _SumsClient(None),
                "loudreader/loudr-1",
                None,
                "voices/kathleen.safetensors",
                voice,
            )

    def test_a_third_party_release_without_a_manifest_is_allowed(self, tmp_path) -> None:
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        hub._verify_against_release_sums(
            _SumsClient(None), "somebody/voices", None, "voices/kathleen.safetensors", voice
        )


class TestMarkerCoversTheWholeSnapshot:
    """The marker has to describe *the* snapshot, not a snapshot.

    The hub cache is shared per repo and revision and loudkit fetches in
    stages, so a checkpoint fetch seals the marker and a later `onnx/` fetch
    adds files it never saw. A marker that only re-stats what it already knew
    about waves those through unhashed.
    """

    def _sealed(self, tmp_path):
        """A verified snapshot whose manifest also covers a not-yet-fetched file."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        _write_sums(
            tmp_path,
            {
                "loudr-1.safetensors": _sha256(b"weights"),
                "onnx/t3_step.onnx": _sha256(b"graph"),
                **_releasable(tmp_path),
            },
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        assert (tmp_path / hub._VERIFIED_MARKER).is_file()

    def test_a_file_that_arrives_after_the_marker_is_hashed(self, tmp_path) -> None:
        self._sealed(tmp_path)
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "t3_step.onnx").write_bytes(b"substituted")
        with pytest.raises(ValueError) as caught:
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        assert "onnx/t3_step.onnx" in str(caught.value)

    def test_a_good_file_that_arrives_after_the_marker_passes(self, tmp_path) -> None:
        self._sealed(tmp_path)
        (tmp_path / "onnx").mkdir()
        (tmp_path / "onnx" / "t3_step.onnx").write_bytes(b"graph")
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_an_unlisted_file_that_arrives_after_the_marker_is_refused(self, tmp_path) -> None:
        """Same door, same verdict: arriving unlisted inside an official
        snapshot is uncovered, and uncovered is fatal there."""
        self._sealed(tmp_path)
        (tmp_path / "extra.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="extra.json"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_an_unchanged_inventory_is_not_rehashed(self, tmp_path) -> None:
        """The whole point of the marker: a cached 747 MB checkpoint is walked
        and stat-ed on the second load(), not read."""
        from loudkit import checkpoint

        read: list[str] = []
        real = checkpoint.file_sha256

        def spy(path, *args, **kwargs):
            read.append(Path(path).name)
            return real(path, *args, **kwargs)

        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(checkpoint, "file_sha256", spy)
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        # The manifest is hashed to identify the marker; the payload is not.
        assert read == ["SHA256SUMS"]


class TestManifestNamesAreConfinedToTheRelease:
    """A manifest name is joined onto the snapshot root and then read.

    So it has to be a normalised relative POSIX path, and it has to be listed
    once. A duplicate is the sharpest of the three: the manifest disagrees
    with itself, whichever entry the parser keeps decides whether verification
    passes, and the release can pass here and fail `shasum -c`.
    """

    def _write(self, tmp_path, body: str) -> Path:
        sums = tmp_path / "SHA256SUMS"
        sums.write_text(body, encoding="utf-8")
        return sums

    def test_nested_names_are_the_ordinary_case(self, tmp_path) -> None:
        sums = self._write(
            tmp_path,
            f"{'a' * 64}  loudr-1.safetensors\n"
            f"{'b' * 64}  onnx/t3_step.onnx\n"
            f"{'c' * 64}  voices/joe.safetensors\n",
        )
        assert hub._parse_sha256sums(sums) == {
            "loudr-1.safetensors": "a" * 64,
            "onnx/t3_step.onnx": "b" * 64,
            "voices/joe.safetensors": "c" * 64,
        }

    def test_traversal_is_refused_and_the_line_is_named(self, tmp_path) -> None:
        sums = self._write(
            tmp_path,
            f"{'a' * 64}  loudr-1.safetensors\n{'b' * 64}  ../etc/passwd\n",
        )
        with pytest.raises(ValueError) as caught:
            hub._parse_sha256sums(sums)
        assert "line 2" in str(caught.value)
        assert "../etc/passwd" in str(caught.value)

    def test_an_absolute_path_is_refused_and_the_line_is_named(self, tmp_path) -> None:
        sums = self._write(tmp_path, f"{'a' * 64}  /etc/passwd\n")
        with pytest.raises(ValueError) as caught:
            hub._parse_sha256sums(sums)
        assert "line 1" in str(caught.value)
        assert "absolute" in str(caught.value)

    def test_a_windows_absolute_path_is_refused_and_the_line_is_named(self, tmp_path) -> None:
        """`C:/weights/evil.safetensors` is absolute on the platform it is
        written for, and is a name relative to nothing on any other."""
        sums = self._write(
            tmp_path,
            f"{'a' * 64}  loudr-1.safetensors\n{'b' * 64}  C:/weights/evil.safetensors\n",
        )
        with pytest.raises(ValueError) as caught:
            hub._parse_sha256sums(sums)
        assert "line 2" in str(caught.value)
        assert "C:/weights/evil.safetensors" in str(caught.value)
        assert "absolute" in str(caught.value)

    def test_a_backslashed_windows_path_is_refused_and_the_line_is_named(
        self, tmp_path
    ) -> None:
        sums = self._write(tmp_path, f"{'a' * 64}  c:\\weights\\evil.safetensors\n")
        with pytest.raises(ValueError) as caught:
            hub._parse_sha256sums(sums)
        assert "line 1" in str(caught.value)
        # The message quotes the name with `!r`, which doubles the backslashes.
        assert repr("c:\\weights\\evil.safetensors") in str(caught.value)

    def test_a_colon_inside_a_relative_name_is_still_a_name(self, tmp_path) -> None:
        """The drive-letter rule is a prefix, not a ban on colons: a colon is
        a legal character in a POSIX filename, and refusing it would fail a
        release that legitimately holds one."""
        sums = self._write(
            tmp_path,
            f"{'a' * 64}  voices/de:formal.safetensors\n{'b' * 64}  c:notadrive.json\n",
        )
        assert hub._parse_sha256sums(sums) == {
            "voices/de:formal.safetensors": "a" * 64,
            "c:notadrive.json": "b" * 64,
        }

    def test_a_duplicate_name_is_refused_and_the_line_is_named(self, tmp_path) -> None:
        sums = self._write(
            tmp_path,
            f"{'a' * 64}  loudr-1.safetensors\n"
            f"{'b' * 64}  voices/joe.safetensors\n"
            f"{'c' * 64}  loudr-1.safetensors\n",
        )
        with pytest.raises(ValueError) as caught:
            hub._parse_sha256sums(sums)
        assert "line 3" in str(caught.value)
        assert "duplicate" in str(caught.value)

    def test_an_unnormalised_name_is_refused(self, tmp_path) -> None:
        """`voices/./joe.safetensors` resolves to a legal file, but the name
        checked is then not the name the filesystem walks."""
        sums = self._write(tmp_path, f"{'a' * 64}  voices/./joe.safetensors\n")
        with pytest.raises(ValueError, match="normalised"):
            hub._parse_sha256sums(sums)

    def test_a_snapshot_with_a_traversal_manifest_never_verifies(self, tmp_path) -> None:
        """The refusal reaches `load()`, not just the parser."""
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        self._write(
            tmp_path,
            f"{_sha256(b'weights')}  loudr-1.safetensors\n{'b' * 64}  ../etc/passwd\n",
        )
        with pytest.raises(ValueError, match="line 2"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")


class TestOfficialReleasesMustBeReleasable:
    """`SHA256SUMS` says the bytes arrived intact; `release.json` says what
    the bytes are. An official repo has to satisfy both, because a lenient
    development bundle carries a perfectly valid manifest and would otherwise
    download, verify and load exactly like the release. Third-party repos and
    local trees made no such claim, so they keep the lenient path.
    """

    def _snapshot(self, tmp_path, release: str | None) -> None:
        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        entries = {"loudr-1.safetensors": _sha256(b"weights")}
        if release is not None:
            (tmp_path / "release.json").write_bytes(release.encode())
            entries["release.json"] = _sha256(release.encode())
        _write_sums(tmp_path, entries)

    def test_a_missing_release_json_is_refused(self, tmp_path) -> None:
        self._snapshot(tmp_path, None)
        with pytest.raises(ValueError, match="no release.json"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_lenient_profile_is_refused_by_name(self, tmp_path) -> None:
        self._snapshot(tmp_path, '{"profile": "lenient", "verified": false}\n')
        with pytest.raises(ValueError) as caught:
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        assert "'lenient'" in str(caught.value)
        assert "full-0.1" in str(caught.value)

    def test_a_strict_profile_that_never_passed_the_gate_is_refused(self, tmp_path) -> None:
        self._snapshot(tmp_path, '{"profile": "full-0.1", "verified": false}\n')
        with pytest.raises(ValueError, match="verified"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_release_json_with_no_profile_is_refused(self, tmp_path) -> None:
        """The shape `release-dir/` in this repository has: entries, and no
        claim about what built them."""
        self._snapshot(tmp_path, '{"checkpoint": {"path": "loudr-1.safetensors"}}\n')
        with pytest.raises(ValueError, match="profile"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_third_party_bundle_is_not_held_to_the_profile(self, tmp_path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        _write_sums(tmp_path, {"model.safetensors": _sha256(b"weights")})
        hub._verify_sha256sums(tmp_path, repo="somebody/their-model")

    def test_a_local_tree_is_not_held_to_the_profile(self, tmp_path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        _write_sums(tmp_path, {"model.safetensors": _sha256(b"weights")})
        hub._verify_sha256sums(tmp_path)


class TestSymlinksDoNotEscapeTheSnapshot:
    """`_rejected_name` refuses traversal spelled in a manifest name. A
    symlink performs the same escape on the filesystem instead: the name is
    legal, and what it addresses is outside the root. The verifier asks every
    path component what it is, with `lstat` semantics, before hashing.
    """

    def test_a_directory_symlink_component_is_refused(self, tmp_path) -> None:
        """`voices/joe.safetensors` with `voices` pointing outside the root:
        the digest matches the outside bytes, so a verifier that follows the
        link vouches for bytes the snapshot does not contain."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "joe.safetensors").write_bytes(b"elsewhere")
        root = tmp_path / "snapshot"
        root.mkdir()
        (root / "loudr-1.safetensors").write_bytes(b"weights")
        (root / "voices").symlink_to(outside, target_is_directory=True)
        _write_sums(
            root,
            {
                "loudr-1.safetensors": _sha256(b"weights"),
                "voices/joe.safetensors": _sha256(b"elsewhere"),
                **_releasable(root),
            },
        )
        with pytest.raises(ValueError, match="symlink"):
            hub._verify_sha256sums(root, repo="loudreader/loudr-1")

    def test_a_file_symlink_is_refused(self, tmp_path) -> None:
        """Outside a hub-cache snapshot there is no blobs/ directory a link
        could legitimately point into, so a final-file symlink stays refused."""
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"elsewhere")
        root = tmp_path / "snapshot"
        root.mkdir()
        (root / "loudr-1.safetensors").symlink_to(outside)
        _write_sums(
            root,
            {"loudr-1.safetensors": _sha256(b"elsewhere"), **_releasable(root)},
        )
        with pytest.raises(ValueError, match="symlink"):
            hub._verify_sha256sums(root, repo="loudreader/loudr-1")

    def _hub_cache_snapshot(self, tmp_path):
        """An empty snapshot in the standard hub cache layout: names in
        ``snapshots/<rev>/`` are symlinks into the sibling ``blobs/``."""
        repo_dir = tmp_path / "hub" / "models--loudreader--loudr-1"
        blobs = repo_dir / "blobs"
        blobs.mkdir(parents=True)
        root = repo_dir / "snapshots" / "abc123"
        root.mkdir(parents=True)
        return root, blobs

    def _blob(self, blobs, root, name: str, data: bytes) -> str:
        """Store ``data`` as a blob and link ``name`` at it, as the cache does."""
        import os

        digest = _sha256(data)
        (blobs / digest).write_bytes(data)
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.symlink_to(os.path.relpath(blobs / digest, dest.parent))
        return digest

    def _linked_release(self, tmp_path) -> tuple:
        import json

        root, blobs = self._hub_cache_snapshot(tmp_path)
        record = json.dumps({"profile": "full-0.1", "verified": True}) + "\n"
        entries = {
            "loudr-1.safetensors": self._blob(blobs, root, "loudr-1.safetensors", b"weights"),
            "voices/joe.safetensors": self._blob(blobs, root, "voices/joe.safetensors", b"joe"),
            "release.json": self._blob(blobs, root, "release.json", record.encode()),
        }
        sums = "".join(f"{digest}  {name}\n" for name, digest in entries.items())
        self._blob(blobs, root, "SHA256SUMS", sums.encode())
        return root, blobs

    def test_the_hub_caches_own_blob_links_are_how_the_cache_works(self, tmp_path) -> None:
        """The standard cache stores every snapshot file as a symlink into the
        repo's own ``blobs/``. Refusing that refused every fresh
        ``lk.load("loudreader/loudr-1")`` — the first thing a new user does —
        while the escape the rule was written against is a *directory*
        symlink, which the cache never creates."""
        root, _ = self._linked_release(tmp_path)
        hub._verify_sha256sums(root, repo="loudreader/loudr-1")
        assert (root / hub._VERIFIED_MARKER).is_file()

    def test_a_final_link_out_of_the_blobs_dir_is_still_refused(self, tmp_path) -> None:
        root, _ = self._linked_release(tmp_path)
        outside = tmp_path / "elsewhere.bin"
        outside.write_bytes(b"joe")
        (root / "voices" / "joe.safetensors").unlink()
        (root / "voices" / "joe.safetensors").symlink_to(outside)
        with pytest.raises(ValueError, match="symlink"):
            hub._verify_sha256sums(root, repo="loudreader/loudr-1")

    def test_a_directory_symlink_is_refused_even_in_the_hub_cache(self, tmp_path) -> None:
        """The cache links files, never directories: a symlinked directory is
        the escape, whatever it points at."""
        root, blobs = self._linked_release(tmp_path)
        stash = tmp_path / "stash"
        (root / "voices" / "joe.safetensors").unlink()
        (root / "voices").rename(stash)
        (stash / "joe.safetensors").symlink_to(blobs / _sha256(b"joe"))
        (root / "voices").symlink_to(stash, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            hub._verify_sha256sums(root, repo="loudreader/loudr-1")

    def test_confinement_is_checked_before_the_marker_answers(self, tmp_path) -> None:
        """A symlink swapped in *after* verification, aimed at outside bytes
        whose size and mtime match the marker's record, used to ride the
        marker fast path straight past the rehash. Confinement now runs on
        every load, before the marker is consulted."""
        import os

        (tmp_path / "loudr-1.safetensors").write_bytes(b"weights")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(b"weights"), **_releasable(tmp_path)},
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        target = tmp_path / "loudr-1.safetensors"
        recorded = os.stat(target)
        outside = tmp_path.parent / f"{tmp_path.name}-outside.safetensors"
        outside.write_bytes(b"wEights")  # same size, different bytes
        os.utime(outside, ns=(recorded.st_atime_ns, recorded.st_mtime_ns))
        target.unlink()
        target.symlink_to(outside)
        with pytest.raises(ValueError, match="symlink"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

    def test_a_repointed_blob_link_invalidates_the_marker(self, tmp_path) -> None:
        """The marker records what each entry *is* and where a link points, so
        repointing a link at a different blob — even one whose stat matches —
        is a rehash, not a fast path."""
        import os

        root, blobs = self._linked_release(tmp_path)
        hub._verify_sha256sums(root, repo="loudreader/loudr-1")
        good = blobs / _sha256(b"weights")
        evil = blobs / _sha256(b"wEights")  # same length, different bytes
        evil.write_bytes(b"wEights")
        recorded = os.stat(good)
        os.utime(evil, ns=(recorded.st_atime_ns, recorded.st_mtime_ns))
        link = root / "loudr-1.safetensors"
        link.unlink()
        link.symlink_to(os.path.relpath(evil, link.parent))
        with pytest.raises(ValueError, match="failed the release checksum"):
            hub._verify_sha256sums(root, repo="loudreader/loudr-1")


# ------------------------------------------------- the release is two files

_SYNTHESIS = "loudr-1.safetensors"
_ENROLLMENT = "loudr-1-enrollment.safetensors"
"""The two names are the contract, so they are spelled out here rather than
read back from the module under test."""


def _pack(
    path,
    *,
    role: str | None = None,
    assets: tuple[str, ...] = (),
    source: str | None = None,
) -> None:
    """A file that reads as a loudkit checkpoint, with an optional role.

    ``role=None`` writes a **pre-split** checkpoint: the manifest carries no
    ``artifact_role`` at all, which is what every release built before the
    split holds — including the published one and ``release-dir/`` — and what
    the resolvers have to keep accepting.

    ``source`` is the ``split.source_payload_sha256`` the splitting tool
    stamps into both halves: the digest of the packed original, identical in
    the pair it produced and the only evidence two files were split together.
    """
    import json

    import numpy as np
    from safetensors.numpy import save_file

    manifest: dict = {"format": "loudkit-checkpoint", "format_version": 1}
    if role is not None:
        manifest["artifact_role"] = role
    if source is not None:
        manifest["split"] = {
            "source_payload_sha256": source,
            "roles": {"synthesis": _SYNTHESIS, "enrollment": _ENROLLMENT},
        }
    tensors = {"t3.dummy": np.zeros(2, np.float32)}
    for name in assets:
        tensors[f"assets.{name}"] = np.zeros(4, np.uint8)
    save_file(tensors, str(path), metadata={"manifest": json.dumps(manifest)})


class TestTheReleaseIsTwoArtefacts:
    """Resolution is by canonical name and ``artifact_role``, not by counting.

    The packed checkpoint is split in two, and the counting rule that used to
    stand in for resolution answers a perfectly ordinary split release with
    "2 checkpoints — name the one you mean": `lk.load("loudreader/loudr-1")`,
    the line in the README, stops working the moment the second artefact
    exists. And ``enroll`` handed the synthesis half to the enroller, which
    after the split holds none of the tensors a clone reads.

    A local directory someone assembled by hand is the interesting case, so
    every shape one can be in is named here: both halves, one half, and the
    pre-split single file that is what exists today.
    """

    def test_a_split_release_resolves_each_half(self, tmp_path) -> None:
        _pack(tmp_path / _SYNTHESIS, role="synthesis")
        _pack(tmp_path / _ENROLLMENT, role="enrollment")
        assert hub.resolve_checkpoint(str(tmp_path)).name == _SYNTHESIS
        assert hub.resolve_enrollment_checkpoint(str(tmp_path)).name == _ENROLLMENT

    def test_a_presplit_checkpoint_answers_for_both(self, tmp_path) -> None:
        """One file holding every tensor is not wrong; it is what the published
        release holds until the new bundle is uploaded. Both resolvers must
        answer with it, and neither may make it an error."""
        _pack(tmp_path / _SYNTHESIS)
        assert hub.resolve_checkpoint(str(tmp_path)).name == _SYNTHESIS
        assert hub.resolve_enrollment_checkpoint(str(tmp_path)).name == _SYNTHESIS

    def test_a_presplit_checkpoint_answers_for_both_by_file(self, tmp_path) -> None:
        """Same file, named directly rather than through its directory —
        `lk.enroll(wav, "./loudr-1.safetensors")`."""
        _pack(tmp_path / _SYNTHESIS)
        named = str(tmp_path / _SYNTHESIS)
        assert hub.resolve_enrollment_checkpoint(named) == tmp_path / _SYNTHESIS

    def test_a_named_synthesis_file_reaches_its_enrollment_sibling(self, tmp_path) -> None:
        _pack(tmp_path / _SYNTHESIS, role="synthesis")
        _pack(tmp_path / _ENROLLMENT, role="enrollment")
        got = hub.resolve_enrollment_checkpoint(str(tmp_path / _SYNTHESIS))
        assert got.name == _ENROLLMENT

    def test_a_synthesis_only_set_says_it_cannot_clone(self, tmp_path) -> None:
        """A synthesis fetch does not bring the enrollment artefact, and the
        checkpoint that did arrive says so about itself. The alternative was
        the enroller's own complaint about `s3gen.speaker_encoder`."""
        _pack(tmp_path / _SYNTHESIS, role="synthesis")
        assert hub.resolve_checkpoint(str(tmp_path)).name == _SYNTHESIS
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_enrollment_checkpoint(str(tmp_path))
        assert _ENROLLMENT in str(caught.value)
        assert "synthesis-only" in str(caught.value)

    def test_the_enrollment_half_alone_is_not_a_checkpoint(self, tmp_path) -> None:
        """A directory holding only the enrollment artefact used to resolve as
        "the one checkpoint here" and load as the engine's weights."""
        _pack(tmp_path / _ENROLLMENT, role="enrollment")
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_checkpoint(str(tmp_path))
        assert _SYNTHESIS in str(caught.value)

    def test_the_enrollment_file_is_refused_where_a_checkpoint_was_asked_for(
        self, tmp_path
    ) -> None:
        """Named directly, by a caller who picked the wrong half."""
        _pack(tmp_path / _ENROLLMENT, role="enrollment")
        with pytest.raises(FileNotFoundError, match="enrollment artefact"):
            hub.resolve_checkpoint(str(tmp_path / _ENROLLMENT))

    def test_a_renamed_pair_resolves_by_its_declared_roles(self, tmp_path) -> None:
        """Neither file has a canonical name, and the counting rule sees two
        checkpoints. Each manifest still says which half it is."""
        _pack(tmp_path / "mine.safetensors", role="synthesis")
        _pack(tmp_path / "mine-enrollment.safetensors", role="enrollment")
        assert hub.resolve_checkpoint(str(tmp_path)).name == "mine.safetensors"
        got = hub.resolve_enrollment_checkpoint(str(tmp_path))
        assert got.name == "mine-enrollment.safetensors"

    def test_two_undeclared_checkpoints_are_still_an_ambiguity(self, tmp_path) -> None:
        """The counting rule is the last resort, not the removed one: two files
        that make no claim are still a question only the caller can answer."""
        _pack(tmp_path / "one.safetensors")
        _pack(tmp_path / "two.safetensors")
        with pytest.raises(FileNotFoundError, match="name the one you mean"):
            hub.resolve_checkpoint(str(tmp_path))

    def test_the_voice_encoder_is_still_not_a_candidate(self, tmp_path) -> None:
        """The rule `ve.safetensors` exists for survives the rewrite."""
        _pack(tmp_path / _SYNTHESIS)
        (tmp_path / "ve.safetensors").write_bytes(b"encoder")
        assert hub.resolve_checkpoint(str(tmp_path)).name == _SYNTHESIS

    def test_halves_of_one_packing_run_pair_up(self, tmp_path) -> None:
        _pack(tmp_path / _SYNTHESIS, role="synthesis", source="a" * 64)
        _pack(tmp_path / _ENROLLMENT, role="enrollment", source="a" * 64)
        assert hub.resolve_enrollment_checkpoint(str(tmp_path)).name == _ENROLLMENT

    def test_halves_of_different_packing_runs_are_refused(self, tmp_path) -> None:
        """Both files load, both produce audio, and the voice is wrong: the
        one failure here that has no error and no obvious symptom. The
        ``split`` block records the packed original's digest in both halves
        precisely so it can be caught by reading two headers."""
        _pack(tmp_path / _SYNTHESIS, role="synthesis", source="a" * 64)
        _pack(tmp_path / _ENROLLMENT, role="enrollment", source="b" * 64)
        # A ValueError, not a FileNotFoundError: both halves are right here,
        # and "the enrollment artefact did not come" is the wrong diagnosis.
        with pytest.raises(ValueError, match="different .*packing runs") as caught:
            hub.resolve_enrollment_checkpoint(str(tmp_path))
        assert not isinstance(caught.value, FileNotFoundError)
        with pytest.raises(ValueError, match="different .*packing runs"):
            hub.resolve_enrollment_checkpoint(str(tmp_path / _SYNTHESIS))

    def test_a_mismatched_pair_is_not_reported_as_a_missing_file(self, tmp_path) -> None:
        """`verify_release_inventory` folds a missing enrollment half into its
        "missing:" list; a mismatched pair must not be folded in with it."""
        _pack(tmp_path / _SYNTHESIS, role="synthesis", source="a" * 64)
        _pack(tmp_path / _ENROLLMENT, role="enrollment", source="b" * 64)
        (tmp_path / "ve.safetensors").write_bytes(b"ve")
        for name in ("manifest.json", "tokenizer.json"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="different .*packing runs"):
            hub.verify_release_inventory(tmp_path, "torch", cloning=True)

    def test_a_half_with_no_split_block_is_not_second_guessed(self, tmp_path) -> None:
        """A build that records no ``split`` makes no claim to check, and the
        check may not turn "cannot tell" into a refusal."""
        _pack(tmp_path / _SYNTHESIS, role="synthesis", source="a" * 64)
        _pack(tmp_path / _ENROLLMENT, role="enrollment")
        assert hub.resolve_enrollment_checkpoint(str(tmp_path)).name == _ENROLLMENT


class TestTheHubsOwnNotFoundHierarchyIsHonoured:
    """A 404 has to be recognised through inheritance, not by one name.

    `RemoteEntryNotFoundError` is a subclass of `EntryNotFoundError`, five
    deep past HfHubHTTPError, HTTPError and OSError. Comparing the exact class
    name cannot see that, which is why a real 404 from a real client escaped
    as a raw error and the pre-split fallback failed against the published
    repository.
    """

    def test_the_named_classes_are_recognised(self) -> None:
        from loudkit.hub import _is_not_found

        for name in ("EntryNotFoundError", "RepositoryNotFoundError", "RevisionNotFoundError"):
            exc = type(name, (Exception,), {})()
            assert _is_not_found(exc), name

    def test_a_subclass_is_recognised_without_being_listed(self) -> None:
        from loudkit.hub import _is_not_found

        base = type("EntryNotFoundError", (Exception,), {})
        # The shape the real client has: several unrelated bases in front of
        # the one that carries the meaning.
        middle = type("HfHubHTTPError", (OSError, base), {})
        remote = type("SomethingTheClientAddsLater", (middle,), {})
        assert _is_not_found(remote())

    def test_anything_else_keeps_its_own_traceback(self) -> None:
        from loudkit.hub import _is_not_found

        for exc in (TimeoutError("slow"), OSError("proxy"), ValueError("nope")):
            assert not _is_not_found(exc)


class TestTheEnrollmentArtefactIsFetchedOnItsOwn:
    """A repo id: one file, not a second snapshot.

    Enrollment is asked for by `loudkit.enroll` and by nothing else, so the
    artefact is fetched the way a voice and the utterance encoder are — one
    named file, hashed against the release's own manifest.
    """

    def _client(self, monkeypatch, tmp_path, *, holds: bool):
        served = {}
        if holds:
            path = tmp_path / _ENROLLMENT
            _pack(path, role="enrollment")
            served[_ENROLLMENT] = str(path)
        asked: list[str] = []

        class _Client:
            def hf_hub_download(self, *, filename: str, **kwargs: object) -> str:
                asked.append(filename)
                target = served.get(filename)
                if target is None:
                    raise EntryNotFoundError(f"404 {filename}")
                return target

        client = _Client()
        monkeypatch.setattr(hub, "_hub", lambda: client)
        # A third-party repo, so the single-file path takes the lenient branch:
        # what is under test is which file is asked for, not the hashing.
        return asked

    def test_the_named_file_is_what_is_fetched(self, monkeypatch, tmp_path) -> None:
        asked = self._client(monkeypatch, tmp_path, holds=True)
        got = hub.resolve_enrollment_checkpoint("somebody/loudr-1")
        assert got.name == _ENROLLMENT
        assert asked[0] == _ENROLLMENT

    def test_a_presplit_repo_falls_back_to_its_one_checkpoint(
        self, monkeypatch, tmp_path
    ) -> None:
        """An older release may carry enrollment tensors in its sole checkpoint."""
        self._client(monkeypatch, tmp_path, holds=False)
        packed = tmp_path / _SYNTHESIS
        _pack(packed)
        monkeypatch.setattr(hub, "resolve_checkpoint", lambda *_a, **_k: packed)
        assert hub.resolve_enrollment_checkpoint("somebody/loudr-1") == packed

    def test_a_split_repo_missing_its_enrollment_half_says_so(
        self, monkeypatch, tmp_path
    ) -> None:
        self._client(monkeypatch, tmp_path, holds=False)
        packed = tmp_path / _SYNTHESIS
        _pack(packed, role="synthesis")
        monkeypatch.setattr(hub, "resolve_checkpoint", lambda *_a, **_k: packed)
        with pytest.raises(FileNotFoundError, match="synthesis-only"):
            hub.resolve_enrollment_checkpoint("somebody/loudr-1")
