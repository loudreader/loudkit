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

from collections.abc import Callable
from pathlib import Path
from typing import Any

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


class HfHubHTTPError(Exception):
    """The hub answered with a status that is not "not there"."""


@pytest.fixture(autouse=True)
def _fresh_process(monkeypatch) -> None:
    """Each test is its own process as far as the hub cache is concerned."""
    monkeypatch.setattr(hub, "_RESOLVED", {})


class _Client:
    """A hub client that fails one way, whatever it is asked for."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    def hf_hub_download(self, **kwargs: object) -> str:
        raise self._exc

    def list_repo_files(self, **kwargs: object) -> list[str]:
        raise self._exc


@pytest.fixture
def failing(monkeypatch):
    def install(exc: BaseException) -> None:
        monkeypatch.setattr(hub, "_hub", lambda: _Client(exc))

    return install


def _notices(caplog) -> str:
    """Every notice this process wrote, as one string.

    The offline notice travels as a ``loudkit.hub`` record at WARNING, which an
    unconfigured process still prints to stderr through ``logging.lastResort``
    and a host application can route. Read here rather than through `capsys`
    because pytest's logging plugin holds a handler on the root logger, so
    `lastResort` never fires under the suite.
    """
    return "\n".join(record.getMessage() for record in caplog.records)


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


class TestResolveVoiceEncoderNamesWhatIsWrong:
    """The one hub call that was unwrapped, so a release with no encoder
    surfaced the client's "Invalid username or password"."""

    def test_a_repo_that_answered_says_the_release_cannot_clone(self, failing) -> None:
        failing(EntryNotFoundError("404 ve.safetensors"))
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_voice_encoder("loudreader/loudr-1")
        assert "ve.safetensors" in str(caught.value)
        assert "synthesis-only" in str(caught.value)

    def test_a_missing_repo_is_not_reported_as_a_missing_encoder(self, failing) -> None:
        failing(RepositoryNotFoundError("401 Client Error: Invalid username or password"))
        with pytest.raises(FileNotFoundError) as caught:
            hub.resolve_voice_encoder("loudreader/typo")
        assert "repository not found" in str(caught.value)
        assert "synthesis-only" not in str(caught.value)

    def test_an_unrelated_failure_keeps_its_own_traceback(self, failing) -> None:
        failing(TimeoutError("read timed out"))
        with pytest.raises(TimeoutError):
            hub.resolve_voice_encoder("loudreader/loudr-1")


class TestResolveCheckpointNamesWhatIsWrong:
    def test_an_unreachable_hub_says_so(self, monkeypatch) -> None:
        """The same diagnosis on the checkpoint path, where the client's own
        message is a paragraph about `HF_HUB_OFFLINE`."""

        class _Snapshot:
            def snapshot_download(self, **kwargs: object) -> str:
                raise LocalEntryNotFoundError("Connection error")

            def hf_hub_download(self, **kwargs: object) -> str:
                raise LocalEntryNotFoundError("Connection error")

            def list_repo_files(self, **kwargs: object) -> list[str]:
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


class _AbsentRepoClient:
    """A hub with no such repository: every call answers the way the real
    client does for a private or mistyped repo, with one error class."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def snapshot_download(self, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        raise self.error

    def hf_hub_download(self, **kwargs: object) -> str:
        raise self.error

    def model_info(self, **kwargs: object) -> object:
        raise self.error

    def list_repo_files(self, **kwargs: object) -> list[str]:
        raise self.error


def _recording(real: Callable[[Any], str], seen: list[str], *, whole: bool = False):
    """``file_sha256``, with the name of every file it hashes appended to
    ``seen``. What a run hashed is the assertion in six tests: a receipt hit
    must reach no weight, and a moved revision must re-verify one."""

    def hashed(path: Any) -> str:
        seen.append(str(path) if whole else Path(path).name)
        return real(path)

    return hashed


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

    def test_a_good_manifest_passes_and_leaves_nothing_behind(self, tmp_path) -> None:
        """Verified once, at the fetch: no marker, and nothing to re-check."""
        payload = b"weights"
        (tmp_path / "loudr-1.safetensors").write_bytes(payload)
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(payload), **_releasable(tmp_path)},
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "SHA256SUMS",
            "loudr-1.safetensors",
            "release.json",
        ]

    def test_the_receipt_is_bookkeeping_the_manifest_cannot_list(self, tmp_path) -> None:
        payload = b"weights"
        (tmp_path / "loudr-1.safetensors").write_bytes(payload)
        (tmp_path / hub.RECEIPT_NAME).write_text("{}", encoding="utf-8")
        _write_sums(
            tmp_path,
            {"loudr-1.safetensors": _sha256(payload), **_releasable(tmp_path)},
        )
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1")

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
        """`SHA256SUMS` holds no digest of itself, and `.gitattributes` and
        `.cache/` are the hub client's own furniture. `release.json` is not on
        that list: the builder writes it before the manifest and covers it, so
        here it is verified like any other file."""
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

    def test_a_missing_repository_is_not_blamed_on_the_manifest(self, tmp_path) -> None:
        """ "No SHA256SUMS" is a repository that answered without the file. A
        repository or revision that is not there, and an offline miss, pass
        through to the diagnosis that names them."""
        voice = tmp_path / "kathleen.safetensors"
        voice.write_bytes(b"profile")
        for error in (
            RepositoryNotFoundError("401 Client Error: Invalid username or password"),
            RevisionNotFoundError("404 Client Error: Revision Not Found"),
            LocalEntryNotFoundError("offline, nothing cached"),
        ):
            with pytest.raises(type(error)):
                hub._verify_against_release_sums(
                    _AbsentRepoClient(error),
                    "loudreader/loudr-1",
                    None,
                    "voices/kathleen.safetensors",
                    voice,
                )


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

    def test_a_manifest_with_no_entries_is_refused(self, tmp_path) -> None:
        """A manifest that verifies nothing verifies everything.

        The refusal was an `assert`, which `python -O` strips: the class is
        pinned here rather than the message, because that is what the two
        callers catch.
        """
        sums = self._write(tmp_path, "\n   \n")
        with pytest.raises(ValueError, match="no checksum entries"):
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

    def test_the_turbo_profile_is_a_release_too(self, tmp_path) -> None:
        """`loudreader/loudr-1-turbo` is stamped `turbo-0.1`, and every fetch
        from it crosses this gate: the snapshot, each voice, the voice
        encoder. A gate that knew only `full-0.1` would refuse the whole
        second model as a development bundle."""
        self._snapshot(tmp_path, '{"profile": "turbo-0.1", "verified": true}\n')
        hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1-turbo")

    def test_an_unverified_turbo_snapshot_is_refused(self, tmp_path) -> None:
        self._snapshot(tmp_path, '{"profile": "turbo-0.1", "verified": false}\n')
        with pytest.raises(ValueError, match="verified"):
            hub._verify_sha256sums(tmp_path, repo="loudreader/loudr-1-turbo")

    def test_a_third_party_bundle_is_not_held_to_the_profile(self, tmp_path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        _write_sums(tmp_path, {"model.safetensors": _sha256(b"weights")})
        hub._verify_sha256sums(tmp_path, repo="somebody/their-model")

    def test_a_local_tree_is_not_held_to_the_profile(self, tmp_path) -> None:
        (tmp_path / "model.safetensors").write_bytes(b"weights")
        _write_sums(tmp_path, {"model.safetensors": _sha256(b"weights")})
        hub._verify_sha256sums(tmp_path)


class TestTheHubCacheIsMadeOfLinks:
    """The standard cache stores every snapshot file as a symlink into the
    repo's own ``blobs/``. The verifier hashes through them, because refusing
    them refused every fresh ``lk.load("loudreader/loudr-1")``."""

    def test_blob_links_verify(self, tmp_path) -> None:
        import json
        import os

        repo_dir = tmp_path / "hub" / "models--loudreader--loudr-1"
        blobs = repo_dir / "blobs"
        blobs.mkdir(parents=True)
        root = repo_dir / "snapshots" / "abc123"
        root.mkdir(parents=True)

        def blob(name: str, data: bytes) -> str:
            digest = _sha256(data)
            (blobs / digest).write_bytes(data)
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(blobs / digest, dest.parent))
            return digest

        record = json.dumps({"profile": "full-0.1", "verified": True}) + "\n"
        entries = {
            "loudr-1.safetensors": blob("loudr-1.safetensors", b"weights"),
            "voices/joe.safetensors": blob("voices/joe.safetensors", b"joe"),
            "release.json": blob("release.json", record.encode()),
        }
        blob("SHA256SUMS", "".join(f"{d}  {n}\n" for n, d in entries.items()).encode())
        hub._verify_sha256sums(root, repo="loudreader/loudr-1")


class _ReleaseClient:
    """A hub client for the checkpoint path: a listing, the two bookkeeping
    files by name, and a snapshot that records whether it was asked for."""

    def __init__(self, tmp_path, *, files: list[str], profile: str | None = "full-0.1") -> None:
        import json

        self.files = files
        self.snapshots: list[bool] = []
        self.root = tmp_path / "snap"
        self.root.mkdir()
        entries: dict[str, str] = {}
        if profile is not None:
            body = json.dumps({"profile": profile, "verified": True}) + "\n"
            (self.root / "release.json").write_bytes(body.encode())
            entries["release.json"] = _sha256(body.encode())
        _write_sums(self.root, entries)

    def list_repo_files(self, **kwargs: object) -> list[str]:
        return self.files

    def hf_hub_download(self, *, filename: str, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError(filename)
        path = self.root / filename
        if not path.is_file():
            raise EntryNotFoundError(f"404 {filename}")
        return str(path)

    def snapshot_download(self, **kwargs: object) -> str:
        self.snapshots.append(bool(kwargs.get("local_files_only")))
        if kwargs.get("local_files_only"):
            raise LocalEntryNotFoundError("not cached")
        return str(self.root)


class TestNothingMovesBeforeTheRepoIsJudged:
    """``release.json`` first, then the listing, then the weights."""

    def test_a_development_bundle_is_refused_before_the_snapshot(
        self, monkeypatch, tmp_path
    ) -> None:
        client = _ReleaseClient(tmp_path, files=["loudr-1.safetensors"], profile="lenient")
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(ValueError, match="development bundle"):
            hub.resolve_checkpoint("loudreader/loudr-1")
        assert client.snapshots == [True], "the network snapshot was asked for"

    def test_a_turbo_release_is_refused_for_a_graph_backend_before_the_snapshot(
        self, monkeypatch, tmp_path
    ) -> None:
        client = _ReleaseClient(
            tmp_path,
            files=["loudr-1-turbo.safetensors", "voices/joe.safetensors"],
            profile="turbo-0.1",
        )
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(ValueError, match="use loudreader/loudr-1 for onnx"):
            hub.resolve_checkpoint("loudreader/loudr-1-turbo", backend="onnx")
        assert client.snapshots == [True]

    def test_download_judges_the_repo_before_the_snapshot_too(
        self, monkeypatch, tmp_path
    ) -> None:
        """`loudkit download` fetched the whole set and judged it after: a
        turbo release for a graph backend cost 1.2 GB before the sentence."""
        client = _ReleaseClient(
            tmp_path,
            files=["loudr-1-turbo.safetensors", "voices/joe.safetensors"],
            profile="turbo-0.1",
        )
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(ValueError, match="use loudreader/loudr-1 for onnx"):
            hub.download("loudreader/loudr-1-turbo", backend="onnx")
        assert client.snapshots == [], "download fetched before it judged"

    def test_download_refuses_a_development_bundle_before_the_snapshot(
        self, monkeypatch, tmp_path
    ) -> None:
        client = _ReleaseClient(tmp_path, files=["loudr-1.safetensors"], profile="lenient")
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(ValueError, match="development bundle"):
            hub.download("loudreader/loudr-1")
        assert client.snapshots == []

    def test_a_turbo_release_passes_the_gate_for_torch(self, monkeypatch, tmp_path) -> None:
        """The profile gate is the same gate, and turbo is a release too. The
        fetch then fails on the empty snapshot, which is the next check's job."""
        client = _ReleaseClient(
            tmp_path, files=["loudr-1-turbo.safetensors"], profile="turbo-0.1"
        )
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises((FileNotFoundError, ValueError)) as caught:
            hub.resolve_checkpoint("loudreader/loudr-1-turbo")
        assert "development bundle" not in str(caught.value)
        assert client.snapshots == [True, False]


class _CacheClient:
    """A hub client with one snapshot in the cache, at ``cached``, whose
    ``main`` resolves to ``sha`` (or fails that way). A fetch lays out a
    second snapshot at ``sha`` and answers with it."""

    def __init__(self, tmp_path, cached: str, sha: str | BaseException) -> None:
        self.snapshots = tmp_path / "models--loudreader--loudr-1" / "snapshots"
        self.sha = sha
        self.ref = cached  # what refs/main points at: the last fetched commit
        self.asked: list[str] = []
        self.lay_out(cached)

    def lay_out(self, commit: str) -> Path:
        root = self.snapshots / commit
        (root / "voices").mkdir(parents=True)
        (root / "loudr-1.safetensors").write_bytes(b"weights")
        (root / "manifest.json").write_text("{}", encoding="utf-8")
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        (root / "voices" / "joe.safetensors").write_bytes(b"joe")
        return root

    def model_info(self, *, repo_id: str, revision: str | None) -> object:
        from types import SimpleNamespace

        self.asked.append("model_info")
        if isinstance(self.sha, BaseException):
            raise self.sha
        return SimpleNamespace(sha=self.sha)

    def snapshot_download(self, *, revision: str | None = None, **kwargs: object) -> str:
        if kwargs.get("local_files_only"):
            self.asked.append("cache")
            if not (self.snapshots / self.ref).is_dir():
                raise LocalEntryNotFoundError("not cached")
            return str(self.snapshots / self.ref)
        self.asked.append("fetch")
        if isinstance(self.sha, BaseException):
            raise LocalEntryNotFoundError("Connection error")
        root = self.snapshots / self.sha
        if not root.is_dir():
            self.lay_out(self.sha)
        self.ref = self.sha
        return str(root)

    def hf_hub_download(self, *, filename: str, **kwargs: object) -> str:
        """One file, by the same rules: the cache answers from refs/main, a
        fetch lays the file out under today's commit."""
        if kwargs.get("local_files_only"):
            self.asked.append(f"cache:{filename}")
            path = self.snapshots / self.ref / filename
            if not path.is_file():
                raise LocalEntryNotFoundError("not cached")
            return str(path)
        self.asked.append(f"fetch:{filename}")
        if isinstance(self.sha, BaseException):
            raise LocalEntryNotFoundError("Connection error")
        root = self.snapshots / self.sha
        if not root.is_dir():
            self.lay_out(self.sha)
        self.ref = self.sha
        return str(root / filename)

    def list_repo_files(self, **kwargs: object) -> list[str]:
        return ["loudr-1.safetensors"]


class TestACachedSnapshotIsNotReverified:
    """Verification happens when the bytes arrive. A later load of the same
    snapshot asks the hub what the revision names today, and when it is the
    cached commit, reads it, and neither hashes nor fetches."""

    def test_a_complete_cached_snapshot_loads_without_hashing(
        self, monkeypatch, tmp_path
    ) -> None:
        from loudkit import checkpoint

        client = _CacheClient(tmp_path, "a" * 40, "a" * 40)
        monkeypatch.setattr(hub, "_hub", lambda: client)
        read: list[str] = []
        monkeypatch.setattr(checkpoint, "file_sha256", lambda p: read.append(str(p)))
        got = hub.resolve_checkpoint("loudreader/loudr-1")
        assert got == client.snapshots / ("a" * 40) / "loudr-1.safetensors"
        assert read == []
        assert client.asked == ["cache", "model_info"]

    def test_a_moved_main_is_fetched_again(self, monkeypatch, tmp_path) -> None:
        """The drift check: a 0.1.0 cache must not answer for 0.1.1's main."""
        client = _CacheClient(tmp_path, "a" * 40, "b" * 40)
        monkeypatch.setattr(hub, "_hub", lambda: client)
        # The fetch's own gates are pinned elsewhere; here, the cache rule.
        monkeypatch.setattr(hub, "_refuse_before_fetching", lambda *_a, **_k: None)
        monkeypatch.setattr(hub, "_verify_sha256sums", lambda *_a, **_k: None)
        got = hub.resolve_checkpoint("loudreader/loudr-1")
        assert got == client.snapshots / ("b" * 40) / "loudr-1.safetensors"
        assert client.asked == ["cache", "model_info", "fetch"]

    def test_offline_with_a_cache_uses_it_and_says_so(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        client = _CacheClient(tmp_path, "a" * 40, ConnectionError("no route to host"))
        monkeypatch.setattr(hub, "_hub", lambda: client)
        got = hub.resolve_checkpoint("loudreader/loudr-1")
        assert got == client.snapshots / ("a" * 40) / "loudr-1.safetensors"
        assert client.asked == ["cache", "model_info"]
        said = _notices(caplog)
        assert "cannot be reached" in said
        assert "a" * 40 in said
        # The level is the pin: at INFO the notice disappears on an
        # unconfigured process, where `lastResort` starts at WARNING.
        assert [(r.name, r.levelname) for r in caplog.records] == [("loudkit.hub", "WARNING")]

    def test_offline_without_a_cache_says_so(self, monkeypatch, tmp_path) -> None:
        client = _CacheClient(tmp_path, "a" * 40, ConnectionError("no route to host"))
        import shutil

        shutil.rmtree(client.snapshots / ("a" * 40))
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(FileNotFoundError, match="cannot be reached"):
            hub.resolve_checkpoint("somebody/loudr-1")
        assert "model_info" not in client.asked

    def test_an_explicit_commit_needs_no_call(self, monkeypatch, tmp_path) -> None:
        client = _CacheClient(tmp_path, "a" * 40, ConnectionError("unasked"))
        monkeypatch.setattr(hub, "_hub", lambda: client)
        got = hub.resolve_checkpoint("loudreader/loudr-1", revision="a" * 40)
        assert got == client.snapshots / ("a" * 40) / "loudr-1.safetensors"
        assert client.asked == ["cache"]

    def test_a_hub_that_answers_is_an_error_not_an_outage(self, monkeypatch, tmp_path) -> None:
        client = _CacheClient(tmp_path, "a" * 40, RepositoryNotFoundError("401"))
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(FileNotFoundError, match="repository not found"):
            hub.resolve_checkpoint("loudreader/loudr-1")

    def test_an_unpublished_official_repo_is_named_as_such(self, monkeypatch) -> None:
        """Nothing cached and the repository private or mistyped: the first
        thing the resolver asks the hub for is SHA256SUMS, and the answer must
        say "not found or not public", not "no SHA256SUMS, retry"."""
        client = _AbsentRepoClient(RepositoryNotFoundError("401 Client Error"))
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(FileNotFoundError, match="repository not found or not public"):
            hub.resolve_checkpoint("loudreader/loudr-1-turbo")

    def test_a_cached_voice_follows_the_same_rule(self, monkeypatch, tmp_path) -> None:
        """A voice is fetched by file, and a moved main can move it too."""
        client = _CacheClient(tmp_path, "a" * 40, "b" * 40)
        monkeypatch.setattr(hub, "_hub", lambda: client)
        monkeypatch.setattr(hub, "_verify_against_release_sums", lambda *_a, **_k: None)
        got = hub.resolve_voice("joe", repo="loudreader/loudr-1")
        assert got == client.snapshots / ("b" * 40) / "voices" / "joe.safetensors"
        assert client.asked == [
            "cache:voices/joe.safetensors",
            "model_info",
            "fetch:voices/joe.safetensors",
        ]
        # Now the cache is at today's commit: the next voice is read, not fetched.
        assert hub.resolve_voice("joe", repo="loudreader/loudr-1") == got
        assert client.asked[3:] == ["cache:voices/joe.safetensors"]

    def test_one_process_asks_once_per_repo(self, monkeypatch, tmp_path) -> None:
        """``load`` then two voices then the voice encoder: one ``model_info``."""
        client = _CacheClient(tmp_path, "a" * 40, "a" * 40)
        (client.snapshots / ("a" * 40) / "ve.safetensors").write_bytes(b"ve")
        monkeypatch.setattr(hub, "_hub", lambda: client)
        hub.resolve_checkpoint("loudreader/loudr-1")
        hub.resolve_voice("joe", repo="loudreader/loudr-1")
        hub.resolve_voice("joe", repo="loudreader/loudr-1")
        hub.resolve_voice_encoder("loudreader/loudr-1")
        assert client.asked.count("model_info") == 1
        assert not any(call.startswith("fetch") for call in client.asked)
        # Another revision of the same repo is another question.
        hub.resolve_voice("joe", repo="loudreader/loudr-1", revision="v1")
        assert client.asked.count("model_info") == 2

    def test_offline_says_so_once_per_process(self, monkeypatch, tmp_path, caplog) -> None:
        client = _CacheClient(tmp_path, "a" * 40, ConnectionError("no route to host"))
        (client.snapshots / ("a" * 40) / "ve.safetensors").write_bytes(b"ve")
        monkeypatch.setattr(hub, "_hub", lambda: client)
        hub.resolve_checkpoint("loudreader/loudr-1")
        hub.resolve_voice("joe", repo="loudreader/loudr-1")
        hub.resolve_voice_encoder("loudreader/loudr-1")
        assert client.asked.count("model_info") == 1
        said = _notices(caplog)
        assert said.count("cannot be reached") == 1, said

    def test_a_hub_that_refuses_is_an_error_not_an_outage(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        """401, 403 and 500 are answers. The cache is not used and nothing
        says the hub was unreachable."""
        client = _CacheClient(tmp_path, "a" * 40, HfHubHTTPError("403 Forbidden"))
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(HfHubHTTPError):
            hub.resolve_checkpoint("loudreader/loudr-1")
        assert "cannot be reached" not in _notices(caplog)

    def test_the_cache_root_may_be_called_snapshots(self, monkeypatch, tmp_path) -> None:
        """``HF_HOME=/srv/snapshots/hf`` must not be read as the commit."""
        client = _CacheClient(tmp_path / "snapshots" / "hub", "a" * 40, "a" * 40)
        monkeypatch.setattr(hub, "_hub", lambda: client)
        got = hub.resolve_checkpoint("loudreader/loudr-1")
        assert got == client.snapshots / ("a" * 40) / "loudr-1.safetensors"
        assert client.asked == ["cache", "model_info"]


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


# ------------------------------------------------------------- the receipt

_FIXTURES = Path(__file__).resolve().parent / "data" / "conformance"


def _receipt_fixture() -> dict:
    import json

    return json.loads((_FIXTURES / "release_receipt.json").read_text(encoding="utf-8"))


def _receipt_on_disk(root: Path) -> dict:
    """The receipt file as written, keys in file order."""
    import json

    return json.loads((root / hub.RECEIPT_NAME).read_text(encoding="utf-8"))


def _lay_out_receipt_case(root: Path, case: dict) -> None:
    """A fixture case on disk: the receipt, the manifest and the files."""
    import json

    if case["receipt"] is not None:
        (root / hub.RECEIPT_NAME).write_text(json.dumps(case["receipt"]), encoding="utf-8")
    if case["sums"] is not None:
        (root / "SHA256SUMS").write_text(case["sums"], encoding="utf-8")
    for name in case["files"]:
        (root / name).write_text(name, encoding="utf-8")


class TestTheReceiptIsTheSharedContract:
    """``.loudkit-release.json`` is one contract for five downloaders: the
    fixture pins its shape, the predicate and the hit rule, and every port
    reads it."""

    def test_the_predicate_is_the_fixture(self, tmp_path) -> None:
        fixture = _receipt_fixture()
        assert fixture["name"] == hub.RECEIPT_NAME
        assert fixture["fields"] == list(hub.RECEIPT_FIELDS)
        assert list(fixture["example"]) == fixture["fields"]
        assert len(fixture["cases"]) >= 20
        assert {case["offline"] for case in fixture["cases"]} == {"use", "error"}
        assert {case["online"] for case in fixture["cases"]} == {"hit", "fetch"}
        for index, case in enumerate(fixture["cases"]):
            root = tmp_path / str(index)
            root.mkdir()
            _lay_out_receipt_case(root, case)
            valid = hub.read_receipt(root, case["repo"]) is not None
            assert valid is (case["offline"] == "use"), case["name"]
            hit = hub.receipt_hit(root, case["repo"], case["commit"])
            assert hit is (case["online"] == "hit"), case["name"]

    def test_the_predicate_hashes_no_weight(self, monkeypatch, tmp_path) -> None:
        from loudkit import checkpoint

        fixture = _receipt_fixture()
        _lay_out_receipt_case(tmp_path, fixture["cases"][0])
        real = checkpoint.file_sha256
        read: list[str] = []
        monkeypatch.setattr(checkpoint, "file_sha256", _recording(real, read))
        assert hub.read_receipt(tmp_path, fixture["cases"][0]["repo"]) is not None
        assert read == ["SHA256SUMS"]

    def test_a_written_receipt_has_the_fixture_shape(self, tmp_path) -> None:
        import re

        sums = f"{_sha256(b'{}')}  manifest.json\n"
        (tmp_path / "SHA256SUMS").write_text(sums, encoding="utf-8")
        (tmp_path / "manifest.json").write_bytes(b"{}")
        hub.write_receipt(tmp_path, repo="loudreader/loudr-1", revision=None, commit="a" * 40)
        written = _receipt_on_disk(tmp_path)
        assert list(written) == _receipt_fixture()["fields"]
        receipt = hub.read_receipt(tmp_path, "loudreader/loudr-1")
        assert receipt is not None
        assert receipt.revision == "main"
        assert receipt.sha256sums == _sha256(sums.encode())
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", receipt.fetched_at)
        assert hub.receipt_hit(tmp_path, "loudreader/loudr-1", "a" * 40)
        assert not hub.receipt_hit(tmp_path, "loudreader/loudr-1", "b" * 40)

    def test_an_unreadable_receipt_is_no_receipt(self, tmp_path) -> None:
        assert hub.read_receipt(tmp_path, "loudreader/loudr-1") is None
        (tmp_path / hub.RECEIPT_NAME).write_text("not json", encoding="utf-8")
        assert hub.read_receipt(tmp_path, "loudreader/loudr-1") is None
        (tmp_path / hub.RECEIPT_NAME).write_text("[1]", encoding="utf-8")
        assert hub.read_receipt(tmp_path, "loudreader/loudr-1") is None

    def test_a_receipt_that_is_not_a_record_is_nothing(self, tmp_path) -> None:
        """An empty file, garbage, and a file too large to be a receipt are no
        receipt, and the last is not read."""
        path = tmp_path / hub.RECEIPT_NAME
        record = (
            '{"repo": "someone/loudr-1", "revision": "main", "commit": "%s", '
            '"sha256sums": null, "fetched_at": "2026-09-02T12:00:00Z"' % ("a" * 40)
        )
        path.write_text(record + "}\n", encoding="utf-8")
        assert hub.read_receipt(tmp_path, "someone/loudr-1") is not None
        bodies = {
            "an empty file": b"",
            "garbage": b"\x00\xff{" * (1 << 15),
            "a record padded past the limit": (
                record + ', "pad": "' + "x" * (1 << 20) + '"}'
            ).encode(),
        }
        for name, body in bodies.items():
            path.write_bytes(body)
            assert hub.read_receipt(tmp_path, "someone/loudr-1") is None, name

    def test_the_fixture_is_what_the_generator_writes(self, tmp_path) -> None:
        import subprocess
        import sys

        repo = _FIXTURES.parents[2]
        result = subprocess.run(
            [
                sys.executable,
                str(repo / "tools" / "make_conformance.py"),
                "--fixtures-only",
                "--out",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            cwd=repo,
        )
        assert result.returncode == 0, result.stderr
        fresh = (tmp_path / "release_receipt.json").read_text(encoding="utf-8")
        committed = (_FIXTURES / "release_receipt.json").read_text(encoding="utf-8")
        assert fresh == committed, "release_receipt.json is stale: run the generator"


class _LocalDirClient:
    """A hub client for ``download(local_dir=...)``: answers ``model_info``
    with one commit (or one failure) and lays a signed release out in the
    directory it is asked to."""

    _FILES = (
        _SYNTHESIS,
        "manifest.json",
        "tokenizer.json",
        "voices/joe.safetensors",
        "voices/ann.safetensors",
    )

    def __init__(self, sha: str | BaseException | None, root: Path) -> None:
        self.sha = sha
        self.snapshots = 0
        self.fail_snapshot: BaseException | None = None
        # The release as the hub holds it: laid out once, so the bookkeeping
        # files served before the snapshot are the ones the snapshot brings.
        #
        # `root` comes from the caller's `tmp_path` rather than from
        # `tempfile.mkdtemp` so pytest reclaims it. Sixteen tests build one of
        # these, and an mkdtemp nothing removes left sixteen release layouts in
        # the system temp directory per run; this machine had accumulated 481.
        self.remote = root
        self.remote.mkdir(parents=True, exist_ok=True)
        (self.remote / "voices").mkdir()
        _pack(self.remote / _SYNTHESIS)
        (self.remote / "manifest.json").write_bytes(b"{}")
        (self.remote / "tokenizer.json").write_bytes(b"{}")
        (self.remote / "voices" / "joe.safetensors").write_bytes(b"joe")
        (self.remote / "voices" / "ann.safetensors").write_bytes(b"ann")
        entries = _releasable(self.remote)
        for rel in self._FILES:
            entries[rel] = _sha256((self.remote / rel).read_bytes())
        _write_sums(self.remote, entries)

    def model_info(self, *, repo_id: str, revision: str | None) -> object:
        from types import SimpleNamespace

        if isinstance(self.sha, BaseException):
            raise self.sha
        return SimpleNamespace(sha=self.sha)

    def list_repo_files(self, **kwargs: object) -> list[str]:
        return [*self._FILES, "release.json", "SHA256SUMS"]

    def hf_hub_download(self, *, filename: str, **kwargs: object) -> str:
        path = self.remote / filename
        if not path.is_file():
            raise EntryNotFoundError(f"404 {filename}")
        return str(path)

    def snapshot_download(self, *, local_dir: str | None, **kwargs: object) -> str:
        import shutil

        self.snapshots += 1
        if self.fail_snapshot is not None:
            raise self.fail_snapshot
        assert local_dir is not None
        root = Path(local_dir)
        root.mkdir(parents=True, exist_ok=True)
        for rel in (*self._FILES, "release.json", "SHA256SUMS"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.remote / rel, root / rel)
        return str(root)


class TestALocalDirCarriesAReceipt:
    """Python's ``--local-dir`` path and the four ports agree on what a
    verified directory looks like, and on when it is one."""

    _REPO = "loudreader/loudr-1"

    def _download(self, monkeypatch, client, target):
        monkeypatch.setattr(hub, "_hub", lambda: client)
        return hub.download(self._REPO, local_dir=target)

    def test_the_first_run_writes_a_receipt_and_a_hit_hashes_nothing(
        self, monkeypatch, tmp_path
    ) -> None:
        from loudkit import checkpoint

        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        assert self._download(monkeypatch, client, target) == target
        receipt = hub.read_receipt(target, self._REPO)
        assert receipt is not None
        assert receipt.repo == self._REPO
        assert receipt.commit == "a" * 40
        assert receipt.revision == "main"
        assert receipt.sha256sums == _sha256((target / "SHA256SUMS").read_bytes())

        read: list[str] = []
        real = checkpoint.file_sha256
        monkeypatch.setattr(checkpoint, "file_sha256", _recording(real, read))
        assert self._download(monkeypatch, client, target) == target
        assert client.snapshots == 1, "a hit reached for the network"
        assert read == ["SHA256SUMS"], "a hit hashed a weight"

    def test_a_receipt_for_another_commit_means_a_fetch(self, monkeypatch, tmp_path) -> None:
        """The drift check: the revision moved under the receipt."""
        from loudkit import checkpoint

        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        client.sha = "b" * 40
        read: list[str] = []
        real = checkpoint.file_sha256
        monkeypatch.setattr(checkpoint, "file_sha256", _recording(real, read, whole=True))
        self._download(monkeypatch, client, target)
        assert client.snapshots == 2
        assert read, "a moved revision was not re-verified"
        receipt = hub.read_receipt(target, self._REPO)
        assert receipt is not None
        assert receipt.commit == "b" * 40

    def test_a_directory_short_of_the_set_is_fetched_again(self, monkeypatch, tmp_path) -> None:
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        (target / "tokenizer.json").unlink()
        self._download(monkeypatch, client, target)
        assert client.snapshots == 2

    def test_a_listed_file_deleted_under_a_receipt_is_fetched_again(
        self, monkeypatch, tmp_path
    ) -> None:
        """The inventory a receipt vouches for is every file the plan
        selects from SHA256SUMS, not only the files a backend cannot run
        without: a voice is one of twenty, and its absence is still a miss."""
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        (target / "voices" / "ann.safetensors").unlink()
        self._download(monkeypatch, client, target)
        assert client.snapshots == 2
        assert (target / "voices" / "ann.safetensors").is_file()

    def test_a_file_edited_in_place_rides_on_a_matching_receipt(
        self, monkeypatch, tmp_path
    ) -> None:
        """The decided rule: a hit hashes nothing, so a listed file rewritten
        under a receipt that still matches is not looked at until the
        commit moves. Same length or not; size decides nothing either."""
        from loudkit import checkpoint

        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        (target / "voices" / "ann.safetensors").write_bytes(b"nna")
        read: list[str] = []
        real = checkpoint.file_sha256
        monkeypatch.setattr(checkpoint, "file_sha256", _recording(real, read))
        self._download(monkeypatch, client, target)
        assert client.snapshots == 1
        assert read == ["SHA256SUMS"]
        assert (target / "voices" / "ann.safetensors").read_bytes() == b"nna"

    def test_a_receipt_for_another_repo_is_a_miss(self, monkeypatch, tmp_path) -> None:
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        monkeypatch.setattr(hub, "_hub", lambda: client)
        assert hub.download("someone/loudr-1", local_dir=target) == target
        assert client.snapshots == 2
        receipt = hub.read_receipt(target, "someone/loudr-1")
        assert receipt is not None
        assert receipt.repo == "someone/loudr-1"

    def test_offline_with_a_receipt_uses_it_and_says_so(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        client.sha = ConnectionError("no route to host")
        assert self._download(monkeypatch, client, target) == target
        assert client.snapshots == 1
        said = _notices(caplog)
        assert "cannot be reached" in said
        assert "a" * 40 in said

    def test_offline_without_a_receipt_reaches_for_the_fetch(
        self, monkeypatch, tmp_path
    ) -> None:
        """Nothing to fall back on: the fetch runs and the client's own
        offline error is translated, as it always was."""
        client = _LocalDirClient(ConnectionError("no route to host"), tmp_path / "remote")
        client.fail_snapshot = LocalEntryNotFoundError("Connection error")
        with pytest.raises(FileNotFoundError, match="cannot be reached"):
            self._download(monkeypatch, client, tmp_path / "release")
        assert client.snapshots == 1

    def test_offline_a_forged_receipt_is_not_used(self, monkeypatch, tmp_path, capsys) -> None:
        """The hub is down and the directory holds the whole set, but the
        receipt is not one a download wrote: no commit, or a digest that
        is not the manifest's. Neither vouches for anything."""
        import json

        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        good = _receipt_on_disk(target)
        client.sha = ConnectionError("no route to host")
        client.fail_snapshot = LocalEntryNotFoundError("Connection error")
        for forged in (
            {k: v for k, v in good.items() if k != "commit"},
            {**good, "sha256sums": "0" * 64},
        ):
            (target / hub.RECEIPT_NAME).write_text(json.dumps(forged), encoding="utf-8")
            with pytest.raises(FileNotFoundError, match="cannot be reached"):
                self._download(monkeypatch, client, target)
        assert "using" not in capsys.readouterr().err

    def test_online_a_forged_receipt_naming_todays_commit_is_a_fetch(
        self, monkeypatch, tmp_path
    ) -> None:
        import json

        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        good = _receipt_on_disk(target)
        (target / hub.RECEIPT_NAME).write_text(
            json.dumps({**good, "sha256sums": "0" * 64}), encoding="utf-8"
        )
        self._download(monkeypatch, client, target)
        assert client.snapshots == 2

    def test_offline_a_receipt_for_another_repo_is_not_used(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        """A directory verified as one model must not answer for another
        because the hub is down: the fetch runs and fails as it would."""
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        client.sha = ConnectionError("no route to host")
        client.fail_snapshot = LocalEntryNotFoundError("Connection error")
        monkeypatch.setattr(hub, "_hub", lambda: client)
        with pytest.raises(FileNotFoundError, match="cannot be reached"):
            hub.download("someone/loudr-1", local_dir=target)
        assert client.snapshots == 2
        assert "using" not in capsys.readouterr().err

    def test_an_answer_with_a_status_is_an_error_not_an_outage(
        self, monkeypatch, tmp_path
    ) -> None:
        """A 500 is the hub answering, so the receipt does not stand in, the
        fetch does not run, and the receipt survives untouched."""
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        client.sha = HfHubHTTPError("500 Server Error")
        with pytest.raises(HfHubHTTPError):
            self._download(monkeypatch, client, target)
        assert client.snapshots == 1
        assert hub.read_receipt(target, self._REPO) is not None

    def test_an_answer_naming_no_commit_is_refused(self, monkeypatch, tmp_path) -> None:
        """One rule in five ports: a hub that answers without a sha is an
        error, never a fetch that writes no receipt."""
        client = _LocalDirClient(None, tmp_path / "remote")
        with pytest.raises(ValueError, match="named no commit"):
            self._download(monkeypatch, client, tmp_path / "release")
        assert client.snapshots == 0

    def test_a_missing_repo_is_an_error_not_an_outage(self, monkeypatch, tmp_path) -> None:
        client = _LocalDirClient(RepositoryNotFoundError("401"), tmp_path / "remote")
        with pytest.raises(FileNotFoundError, match="repository not found"):
            self._download(monkeypatch, client, tmp_path / "release")
        assert client.snapshots == 0

    def test_a_stale_receipt_does_not_outlive_a_failed_fetch(
        self, monkeypatch, tmp_path
    ) -> None:
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "release"
        self._download(monkeypatch, client, target)
        client.sha = "b" * 40
        client.fail_snapshot = TimeoutError("read timed out")
        with pytest.raises(TimeoutError):
            self._download(monkeypatch, client, target)
        assert not (target / hub.RECEIPT_NAME).exists()

    def test_the_hub_cache_path_writes_no_receipt(self, monkeypatch, tmp_path) -> None:
        """The hub cache is keyed by commit already; the receipt is for
        directories the caller named."""
        root = tmp_path / "snap"

        class _CacheClient(_LocalDirClient):
            def snapshot_download(self, *, local_dir: str | None, **kwargs: object) -> str:
                assert local_dir is None
                return super().snapshot_download(local_dir=str(root), **kwargs)

        client = _CacheClient("a" * 40, tmp_path / "remote")
        monkeypatch.setattr(hub, "_hub", lambda: client)
        assert hub.download(self._REPO) == root
        assert not (root / hub.RECEIPT_NAME).exists()


class TestAHubCacheSnapshotIsARelease:
    """The Hub cache lays a release out as a snapshot of symlinks into the cache
    entry's blob store. The resolver used to follow a voice's link, land in
    ``blobs/`` and call that an escape from ``voices/``, so no voice loaded by
    name from a downloaded release. The boundary is the entry."""

    def _entry(self, tmp_path: Path) -> tuple[Path, Path]:
        entry = tmp_path / "models--loudreader--loudr-1"
        snapshot = entry / "snapshots" / ("0" * 40)
        (snapshot / "voices").mkdir(parents=True)
        (entry / "blobs").mkdir()
        return entry, snapshot

    def test_a_voice_linked_into_the_entry_s_blobs_resolves(self, tmp_path) -> None:
        from loudkit.release import _voice_in_tree, release_confinement

        entry, snapshot = self._entry(tmp_path)
        blob = entry / "blobs" / ("ab" * 20)
        blob.write_bytes(b"not read here")
        try:
            link = snapshot / "voices" / "joe.safetensors"
            link.symlink_to(Path("../../../blobs") / blob.name)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"cannot create a symlink on this machine: {exc}")

        assert release_confinement(snapshot / "voices") == entry.resolve()
        assert release_confinement(tmp_path / "plain") == (tmp_path / "plain").resolve()
        assert _voice_in_tree(snapshot, "joe", "joe.safetensors") == blob.resolve()

    def test_a_link_out_of_the_entry_is_still_an_escape(self, tmp_path) -> None:
        from loudkit.release import _voice_in_tree

        _, snapshot = self._entry(tmp_path)
        outside = tmp_path / "secret.safetensors"
        outside.write_bytes(b"never handed out")
        try:
            (snapshot / "voices" / "leak.safetensors").symlink_to(outside)
        except OSError as exc:  # pragma: no cover - platform-dependent
            pytest.skip(f"cannot create a symlink on this machine: {exc}")

        with pytest.raises(VoiceNotFoundError, match="escapes"):
            _voice_in_tree(snapshot, "leak", "leak.safetensors")


class TestDownloadJudgesItsOwnArguments:
    """Both refusals happen before the client is reached, so both say
    something in loudkit's vocabulary rather than the client's."""

    @pytest.mark.parametrize("ref", ["../../etc", "/etc/passwd", "loudr-1", "a/b/c"])
    def test_a_ref_that_is_not_a_repo_id_is_refused_by_name(
        self, monkeypatch, ref: str
    ) -> None:
        """``resolve_checkpoint`` calls ``is_repo_id`` and all four ports call
        theirs; this did not, so a path-shaped ref reached huggingface_hub and
        came back as its sentence about ``repo_type`` -- a concept loudkit does
        not have, naming a fix that is not the fix.
        """

        def unreached() -> object:
            raise AssertionError(f"the client was reached for {ref!r}")

        monkeypatch.setattr(hub, "_hub", unreached)

        with pytest.raises(ValueError, match="not a Hugging Face repo id"):
            hub.download(ref)

    def test_a_local_dir_that_is_a_file_says_so(self, monkeypatch, tmp_path) -> None:
        """``unlink(missing_ok=True)`` swallows a missing file, not a parent
        that cannot hold one: a file here raised ``NotADirectoryError`` from
        inside the fetch, and the CLI reported it as a download that may be
        truncated -- advice about a fetch that had not started.
        """
        target = tmp_path / "not-a-directory"
        target.write_text("", encoding="utf-8")

        def unreached() -> object:
            raise AssertionError("the client was reached for a file --local-dir")

        monkeypatch.setattr(hub, "_hub", unreached)

        with pytest.raises(ValueError, match="has to be a directory"):
            hub.download("loudreader/loudr-1", local_dir=target)

    def test_a_local_dir_that_does_not_exist_yet_is_fine(self, monkeypatch, tmp_path) -> None:
        """The other half: ``--local-dir`` names where a release lands, and
        naming somewhere that does not exist yet is the ordinary case."""
        client = _LocalDirClient("a" * 40, tmp_path / "remote")
        target = tmp_path / "does" / "not" / "exist"
        monkeypatch.setattr(hub, "_hub", lambda: client)

        assert hub.download("loudreader/loudr-1", local_dir=target) == target
