"""What the engine computes, and how fast it gets there — kept apart on purpose.

Two frozen dataclasses, and the split between them is the load-bearing idea of
this library:

``AlgorithmConfig``
    Everything that determines *what comes out*. Identical on every backend,
    always. A backend that reads a different value here is broken, not tuned.

``ExecutionConfig``
    Everything that determines *how fast*. Free to differ per backend, judged
    only on speed and on staying inside the numerics band.

The test for which layer a new setting belongs to: *if I changed it on one
backend only, would the output be a different reading of the text, or the same
reading computed differently?* Different reading means algorithm.

The two layers are kept apart because algorithm values must be verifiable per
build. Guidance mode illustrates the cost of mixing them: applying dual-path
guidance to an estimator distilled for single-path use yields plausible audio
under a different algorithm, and no output comparison alone can distinguish
that from correct output. Every algorithm value therefore has exactly one home,
backends inherit it, and ``AlgorithmConfig.describe()`` exists to be logged on
every run.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import Any, Literal, cast, get_args

from .frontend.textconfig import TextConfig

# One direction only: `postprocess` holds the detectors and knows nothing about
# configuration assembly, so it imports nothing from here and there is no cycle.
from .postprocess import PostprocessConfig

__all__ = [
    "AlgorithmConfig",
    "ChunkConfig",
    "FINGERPRINT_SCHEMA",
    "ExecutionConfig",
    "ExecutionOverrides",
    "GuidanceMode",
    "PostprocessConfig",
    "SamplingConfig",
    "WindowConfig",
    "Device",
    "ONNXProvider",
    "Precision",
    "DEFAULT_ALGORITHM",
]

FINGERPRINT_SCHEMA = 1
"""Version of the fingerprint's canonical form, hashed alongside the values.

Bump only when the *serialisation* changes, never when a field is added — the
whole point is that adding a field with a default does not re-fingerprint an
algorithm that did not change.
"""


class _UnsetType:
    """The sentinel for "this optional field was never set".

    A plain ``object()`` cannot do this job: :func:`dataclasses.asdict`
    deep-copies field values, and ``deepcopy(object())`` is a *new* object —
    so the sentinel lost its identity on the way to
    :meth:`AlgorithmConfig.canonical_form` and the filter there could never
    match. This class survives both copy forms by returning itself, which is
    what makes "unset fields do not enter the fingerprint" actually true.
    """

    _instance: _UnsetType | None = None

    def __new__(cls) -> _UnsetType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __deepcopy__(self, memo: dict[int, object]) -> _UnsetType:
        return self

    def __copy__(self) -> _UnsetType:
        return self

    def __repr__(self) -> str:
        return "UNSET"


_UNSET: Any = _UnsetType()

RECIPE_VERSION = "loudkit-1"
"""The one recipe. There is no other, and nothing predates it."""


def _postprocess_from_manifest(pp: Mapping[str, Any]) -> PostprocessConfig:
    """Read the ``postprocess`` block, defaulting to the shipping detectors.

    Fields are read off the dataclass rather than through a wall of hand-written
    ``pp.get(...)`` lines. Not for brevity: a hand-written wall is a list that a
    new constant gets left out of, and a constant the loader silently ignores is
    a manifest declaring one recipe while the engine runs another. The coercion
    target is the default's own type, so a field added to
    :class:`~loudkit.postprocess.PostprocessConfig` is read here without being
    mentioned here.
    """
    default = PostprocessConfig()
    values: dict[str, Any] = {}
    for f in dataclass_fields(PostprocessConfig):
        if f.name not in pp:
            continue
        raw = pp[f.name]
        want = type(getattr(default, f.name))
        if want is str:
            if not isinstance(raw, str):
                raise ValueError(
                    f"manifest['postprocess'][{f.name!r}] must be a string, "
                    f"got {type(raw).__name__}"
                )
            values[f.name] = raw
            continue
        # `bool` is checked first because it is an `int` subclass: `true` would
        # otherwise sail through as 1 and set a token count to one.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(
                f"manifest['postprocess'][{f.name!r}] must be a number, got {raw!r}"
            )
        # `int(4.7)` is 4, silently. The manifest a human reads would say
        # 4.7 and the engine would run on 4, with the fingerprint recording the
        # truncated value — one file, two answers about what it means. A field
        # typed as a count takes a count.
        if want is int and isinstance(raw, float) and not raw.is_integer():
            raise ValueError(
                f"manifest['postprocess'][{f.name!r}] must be a whole number, got {raw!r}"
            )
        values[f.name] = want(raw)
    mode = values.get("mode", default.mode)
    if mode not in ("off", "report", "trim"):
        raise ValueError(f"manifest declares unknown postprocess mode {mode!r}")
    return replace(default, **values)


Device = Literal["cpu", "cuda", "mps", "onnx", "coreml"]
Precision = Literal["fp32", "fp16", "bf16"]

ONNXProvider = Literal["auto", "cpu", "cuda", "coreml", "directml"]
"""Which onnxruntime execution provider runs the exported graphs.

These five spellings are the cross-language contract: the Rust, Go and
TypeScript engines accept the same words for the same concept, because a port
that names one of them differently is indistinguishable from a port that does
something different. onnxruntime's own names (``CUDAExecutionProvider``,
``DmlExecutionProvider``) never leave the ONNX backend.
"""

ONNX_PROVIDERS: tuple[ONNXProvider, ...] = get_args(ONNXProvider)
"""The accepted values, for runtime validation. Not a preference order — the
order ``auto`` searches lives with the provider names, in the ONNX backend."""

GuidanceMode = Literal["single_path", "cfg_dual_path"]
"""How the flow estimator is driven.

``single_path``
    One estimator call per Euler step. Correct for a guidance-distilled
    estimator, where the guided velocity is already what the weights produce.
    This is what ships.

``cfg_dual_path``
    Two calls per step — conditional and unconditional — combined as
    ``(1 + w)·v_cond − w·v_uncond``. Correct only for a teacher that was never
    guidance-distilled. Applying it to a distilled student subtracts ``w`` times
    an output the student was never trained to produce.
"""


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """The sampling law. Algorithm layer — identical on every backend.

    Implemented by :mod:`loudkit.sampler` as LR-SAMPLER-v1, which is specified
    so that three independent implementations agree bit for bit. Two properties
    are not negotiable and both cost something to hold:

    * the RNG is **counter-based**, not sequential, so a token's random number
      depends on ``(seed, stream, step, index)`` alone. ``torch.multinomial`` is
      not portable — the same probability vector and the same generator stream
      produce different samples on x86 and arm64.
    * ``min_p`` is evaluated in **logit space**, so there is no softmax
      normalisation and no CDF scan, hence no reduction whose order a backend
      could vary.
    """

    temperature: float = 0.8
    repetition_penalty: float = 1.2
    min_p: float = 0.05
    max_new_tokens: int = 255

    silence_token_ids: tuple[int, ...] = ()
    """Tokens exempt from *both* the repetition penalty and the ``min_p`` floor.

    A reader pauses repeatedly, so penalising silence suppresses pausing. This
    is not cosmetic: swapping a 31-id production list for a plausible 19-id
    alternative moved pause ratio from 0.112 to 0.085 on a pause-heavy sentence.
    """

    min_tokens_floor: int = 0
    min_tokens_text_ratio: float = 0.0
    """EOS floor (len-prior gate): the stop token is masked to −inf until at
    least ``max(min_tokens_floor, int(n_text_tokens * min_tokens_text_ratio))``
    speech tokens have been emitted.

    This is the shipped engine's early-EOS guard (ChatterboxT3Runner:
    ``minTokens = max(10, textIds * 6/5)``), an algorithm-layer EOS policy that
    shared by every backend. Speech runs ~1.7–2.6 tokens per text token,
    so a 1.2x floor kills early-EOS truncations without forcing overlong reads.
    Defaults are 0/0.0 (off) so the bare sampling law stays the textbook one;
    the production values come from the backend that loads the checkpoint.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.temperature <= 4.0:
            raise ValueError(f"temperature out of range: {self.temperature}")
        if self.repetition_penalty < 1.0:
            raise ValueError(
                f"repetition_penalty below 1.0 rewards repetition: {self.repetition_penalty}"
            )
        if not 0.0 <= self.min_p < 1.0:
            raise ValueError(f"min_p out of range: {self.min_p}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive: {self.max_new_tokens}")
        if self.min_tokens_floor < 0:
            raise ValueError(f"min_tokens_floor must be >= 0: {self.min_tokens_floor}")
        if self.min_tokens_text_ratio < 0.0:
            raise ValueError(
                f"min_tokens_text_ratio must be >= 0: {self.min_tokens_text_ratio}"
            )


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    """How text longer than one window is split. Algorithm layer.

    A window carries ~10.2 s of speech, so anything longer than a couple of
    sentences has to be split, generated in pieces and joined. Where the splits
    fall and what each piece is conditioned on are *audible* decisions — they
    determine where the reader breathes — so they belong here, identical on
    every backend, rather than in whichever caller needs them first.

    Recorded from experience: generating each chunk independently makes every
    chunk restart its pitch contour like a new sentence, which is heard as a
    stutter at the join. Carrying a prefix of the previous chunk's tokens into
    the next one is what removes it, and it is why
    :meth:`~loudkit.contracts.TokenGenerator.generate` takes a ``prefix``.
    """

    enabled: bool = True

    max_tokens: int = 255
    """Longest run of speech tokens a single chunk may produce. Matches the
    window; exceeding it truncates, which is why it is checked rather than
    hoped."""

    prefix_tokens: int = 6
    """Speech tokens from the previous chunk fed back as context.

    Zero means chunks are independent, which is the simplest thing and the one
    that stutters at joins: measured on the reference voice, the pitch contour
    restarts ~74 Hz higher at the join when chunks are independent, vs ~7 Hz
    with a 6-token prefix (a natural phrase boundary). Non-zero costs
    generation time on tokens that are then discarded, and buys prosodic
    continuity. Default is 6 — small enough to cost little, large enough to
    remove the restart.
    """

    split_on: tuple[str, ...] = (". ", "! ", "? ", "; ", ", ")
    """Split candidates, strongest first. Sentence ends before clause ends
    before commas: a break at a full stop is inaudible, a break mid-clause is
    not."""

    first_chunk_max_tokens: int | None = _UNSET
    """Token budget for the *first* chunk only, or ``None``/unset for none.

    Time to first audio is the first chunk's generation plus its render, and
    both scale with its length — a full 255-token window is ~10 s of speech
    the listener waits through before hearing anything. Capping only the first
    chunk starts the stream at the first clause and lets every later chunk run
    long while earlier audio plays. Measured on an M3 Pro: a 96-token first
    budget cuts first audio from ~1.9 s to ~1.4 s, landing on a clause
    boundary; smaller budgets shave little more (the floor is the fixed
    prefill-plus-render cost) and cut mid-clause, which is heard. 96 is a good
    first value; below ~48 the first chunk stops being a phrase.

    This changes where the first split falls, which is audible — so it is an
    algorithm value: setting it re-fingerprints, and two engines that disagree
    about it are computing different things. Left unset it is absent from the
    fingerprint, and the split is exactly what it always was.

    Python-only for now; the ports follow the usual route (feature in the
    reference first, then the conformance fixture, then the ports). A value
    must be positive and no larger than ``max_tokens``.
    """

    def resolved_first_chunk_max_tokens(self) -> int | None:
        """The first-chunk budget, with the unset sentinel resolved to None."""
        value = self.first_chunk_max_tokens
        return None if value is _UNSET else value

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive: {self.max_tokens}")
        first = self.resolved_first_chunk_max_tokens()
        if first is not None and not (0 < first <= self.max_tokens):
            raise ValueError(
                f"first_chunk_max_tokens must be in 1..max_tokens ({self.max_tokens}): {first}"
            )
        # A positive max_tokens is not enough: split_text's character budget is
        # int(max_tokens * CHARS_PER_TOKEN), and a budget of zero makes it cut
        # nothing and loop forever. Refuse the config rather than hang.
        from .frontend.chunking import CHARS_PER_TOKEN

        if int(self.max_tokens * CHARS_PER_TOKEN) < 1:
            raise ValueError(
                f"max_tokens={self.max_tokens} leaves no character budget to split on "
                f"(int({self.max_tokens} * {CHARS_PER_TOKEN}) == 0); "
                f"needs at least {math.ceil(1 / CHARS_PER_TOKEN)}"
            )
        if not 0 <= self.prefix_tokens < self.max_tokens:
            raise ValueError(f"prefix_tokens must be in [0, max_tokens): {self.prefix_tokens}")
        if not self.split_on:
            raise ValueError("split_on cannot be empty: there would be nowhere to break")


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """How token sequences are framed for the mel decoder. Algorithm layer.

    This is here rather than in the backend because it was the other half of the
    same lesson. The entire measured deviation of the Neural Engine render —
    mel correlation 0.975 to 0.993, worst on the shortest sentence — traced not
    to the Neural Engine, not to fp16, and not to CoreML, but to the static
    window's pad-and-truncate recipe differing from the torch path's.

    A backend that needs fixed shapes may *pad* to them; it may not decide what
    padding means.
    """

    max_speech_tokens: int = 255
    """Longest token sequence a single window carries (~10.2 s at 25 Hz)."""

    static_length: int | None = None
    """Pad every window to exactly this many tokens, or ``None`` for ragged.

    Set it when a backend requires fixed shapes — but set it in the *algorithm*
    config so every backend pads identically, including the ones that did not
    need it.
    """

    pad_token_id: int | None = None
    """Token used for padding. ``None`` means "use the silence token".

    Not a free choice: padding the unused window with token 0 — an ordinary
    speech unit — measurably bleeds into the tail through the encoder's
    attention (+3 dB of high-band mel energy after the last real token). The
    shipped engine pads with silence unit 4254 for exactly that reason.
    """

    static_prompt_tokens: int | None = None
    """Fixed length of the reference-prompt window, or ``None`` for ragged.

    The other half of the production static recipe: the shipped engine frames
    the voice prompt at exactly 238 tokens — longer prompts truncate, shorter
    ones pad with the silence token — and the prompt's mel condition occupies
    exactly ``2 * static_prompt_tokens`` frames. This lives here, next to
    ``static_length``, because the pad/truncate recipe *is* the entire measured
    ANE-vs-torch mel deviation (corr 0.975–0.993); two backends that frame the
    prompt differently are different algorithms, whatever their hardware.
    """

    def __post_init__(self) -> None:
        if self.static_length is not None and self.static_length < self.max_speech_tokens:
            raise ValueError(
                f"static_length {self.static_length} cannot be shorter than "
                f"max_speech_tokens {self.max_speech_tokens}"
            )
        if self.static_prompt_tokens is not None and self.static_prompt_tokens <= 0:
            raise ValueError(
                f"static_prompt_tokens must be positive: {self.static_prompt_tokens}"
            )


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    """Everything that determines what the engine produces.

    Identical on every backend. Serialisable, hashable, and printed on every run
    so that "which mode was that measured in?" is never again unanswerable.
    """

    recipe_version: str = RECIPE_VERSION
    """Identifies the parts of the algorithm that are *code*, not settings.

    The sampling law, the Euler grid formula, the window framing recipe, the
    Box-Muller convention, the EOS arithmetic — none of these are fields here,
    yet all of them determine what comes out. Two builds could agree on every
    value below and still compute different things because one of them shipped
    LR-SAMPLER-v2.

    So the code's own version travels in the fingerprint. Bump it whenever the
    recipe changes, which re-bases goldens under the identity contract. Not
    bumping it after a recipe change reproduces the founding defect through the
    front door: two implementations, one config, different audio.
    """

    guidance: GuidanceMode = "single_path"
    guidance_rate: float = 0.0
    """Only read when ``guidance == "cfg_dual_path"``. Kept out of the way
    otherwise so a stray default cannot leak into a distilled model."""

    euler_steps: int = 2
    """Flow-matching integration steps. Distilled from six; see EXP-002."""

    euler_grid: tuple[float, ...] | None = None
    """Explicit time grid of ``euler_steps + 1`` values in [0, 1].

    ``None`` selects the cosine schedule. An explicit grid is preferred for
    anything that must match across implementations, because "cosine" is a
    formula two codebases can write two ways.
    """

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    chunking: ChunkConfig = field(default_factory=ChunkConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)

    text: TextConfig = field(default_factory=TextConfig)
    """What the text funnel is — see :mod:`loudkit.frontend.textconfig`.

    Here rather than beside the algorithm because the funnel decides what string
    the model is handed, and therefore what it says. It was outside the
    fingerprint until 2026-08-17, which meant a day's worth of funnel changes
    moved no hash at all."""

    sample_rate: int = 24_000
    token_rate_hz: float = 25.0
    speech_vocab_size: int = 8194
    start_speech_token: int = 6561
    stop_speech_token: int = 6562

    def __post_init__(self) -> None:
        if self.guidance == "single_path" and self.guidance_rate != 0.0:
            raise ValueError(
                "guidance_rate must be 0.0 in single_path mode — a non-zero rate "
                "here is the exact defect this config exists to prevent"
            )
        if self.guidance == "cfg_dual_path" and self.guidance_rate <= 0.0:
            raise ValueError("cfg_dual_path with a zero rate does twice the work for nothing")
        if self.euler_steps < 1:
            raise ValueError(f"euler_steps must be >= 1: {self.euler_steps}")
        self._validate_numeric_core()
        if self.euler_grid is not None:
            grid = self.euler_grid
            if len(grid) != self.euler_steps + 1:
                raise ValueError(
                    f"euler_grid has {len(grid)} points, expected {self.euler_steps + 1}"
                )
            if not all(b > a for a, b in zip(grid, grid[1:], strict=False)):
                raise ValueError("euler_grid must be strictly increasing")
            if abs(grid[0]) > 1e-6 or abs(grid[-1] - 1.0) > 1e-6:
                raise ValueError("euler_grid must run from 0.0 to 1.0")

        # The three token budgets live in three independent manifest blocks and
        # were validated in three independent `__post_init__`s, so nothing ever
        # asked whether they agreed. They have to: `_strip_specials` raises when
        # a chunk's tokens exceed the render window, and on the streaming path
        # that refusal lands *after* earlier chunks have been delivered and
        # played. A config that guarantees a mid-passage failure is a config
        # that should not load — the whole point of validating here rather than
        # at the first synthesis is that the manifest is checked once, at the
        # door, instead of per utterance.
        window = self.window.max_speech_tokens
        if self.chunking.enabled and self.chunking.max_tokens > window:
            raise ValueError(
                f"chunking.max_tokens {self.chunking.max_tokens} exceeds the render "
                f"window ({window}): every chunk would be sized past what the "
                "renderer accepts, and the refusal would land mid-stream, after "
                "audio had already been delivered"
            )
        if self.sampling.max_new_tokens > window:
            raise ValueError(
                f"sampling.max_new_tokens {self.sampling.max_new_tokens} exceeds the "
                f"render window ({window}): generation is allowed to produce more "
                "speech than the renderer will accept, so a long utterance fails "
                "after it has been generated rather than before"
            )

    # -- identity -----------------------------------------------------------

    def _validate_numeric_core(self) -> None:
        """The rates and token ids, which nothing checked.

        Every one of these constructs a config that looks fine in a log line and
        fails somewhere unrelated: ``sample_rate=0`` divides by zero in the
        vocoder and again in ``Result.duration``, ``token_rate_hz=0`` in every
        duration estimate, and ``start_speech_token == stop_speech_token``
        breaks the EOS contract itself — the generator waits for a token it has
        already emitted to open the speech. A manifest is data from outside this
        process, so it gets the same scrutiny as the rest of it.
        """
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0: {self.sample_rate}")
        if self.token_rate_hz <= 0:
            raise ValueError(f"token_rate_hz must be > 0: {self.token_rate_hz}")
        if self.speech_vocab_size < 1:
            raise ValueError(f"speech_vocab_size must be >= 1: {self.speech_vocab_size}")
        for name in ("start_speech_token", "stop_speech_token"):
            value = getattr(self, name)
            if not 0 <= value < self.speech_vocab_size:
                raise ValueError(f"{name} must be in [0, {self.speech_vocab_size}): {value}")
        if self.start_speech_token == self.stop_speech_token:
            raise ValueError(
                "start_speech_token and stop_speech_token must differ: both are "
                f"{self.start_speech_token} — generation would stop on the token "
                "that starts it"
            )

    def to_dict(self) -> dict[str, object]:
        """Plain-data form, suitable for a manifest or a log line."""
        return asdict(self)

    def canonical_form(self) -> str:
        """The exact string that gets hashed. Specified, not incidental.

        A fingerprint another language has to reproduce cannot be defined as
        "whatever ``json.dumps(asdict(...))`` happens to emit". Three rules make
        it portable:

        * **floats use ``repr``**, which is the shortest string that round-trips
          — the same value every IEEE-754 double produces, so a Swift port can
          match digit for digit rather than guessing at precision.
        * **numpy scalars are coerced**, because ``np.float32`` is not JSON
          serialisable and would otherwise raise from inside a hash function,
          three layers away from the user who built a grid with ``np.cos``.
        * **only fields the schema knows about are hashed**, in sorted order,
          alongside an explicit ``schema`` version. Adding a field with a
          default does not re-fingerprint a semantically identical algorithm.

        That last rule matters more than it looks. Adding ``min_tokens_floor``
        changed the hash of every existing config, and a check that cries wolf
        on every upgrade is a check people learn to override.
        """

        def norm(value: object) -> object:
            if isinstance(value, float):
                return repr(float(value))
            if isinstance(value, (list, tuple)):
                return [norm(v) for v in value]
            if isinstance(value, dict):
                # Unset optional fields drop out at every depth, not only the
                # top level — a nested config's new field must not re-hash
                # every existing config either.
                return {k: norm(v) for k, v in sorted(value.items()) if v is not _UNSET}
            if hasattr(value, "item") and not isinstance(value, (str, bytes)):
                return norm(value.item())  # numpy scalars
            return value

        body = {k: norm(v) for k, v in sorted(self.to_dict().items()) if v is not _UNSET}
        return json.dumps(
            {"schema": FINGERPRINT_SCHEMA, "algorithm": body},
            sort_keys=True,
            separators=(",", ":"),
        )

    def fingerprint(self) -> str:
        """Short stable hash of every algorithm value.

        Two backends whose fingerprints differ are computing different things,
        whatever their outputs happen to look like. Conformance compares these
        before it compares audio, because the guidance defect produced
        plausible audio on both sides of the mismatch.
        """
        return hashlib.sha256(self.canonical_form().encode()).hexdigest()[:16]

    def describe(self) -> str:
        """One-line summary for logs. Print this on every run."""
        g = self.guidance if self.guidance == "single_path" else f"cfg@{self.guidance_rate}"
        grid = "explicit" if self.euler_grid else "cosine"
        return (
            f"algo[{self.fingerprint()}] {self.recipe_version} {g} "
            f"euler={self.euler_steps}({grid}) "
            f"temp={self.sampling.temperature} rep={self.sampling.repetition_penalty} "
            f"min_p={self.sampling.min_p} sil={len(self.sampling.silence_token_ids)} "
            f"win={self.window.static_length or 'ragged'}"
        )

    def with_(self, **changes: object) -> AlgorithmConfig:
        """Copy with overrides, re-validated. Frozen configs change by copying."""
        # replace() wants each field's own type; a **kwargs passthrough cannot
        # express that, and the dataclass re-validates in __post_init__ anyway
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> AlgorithmConfig:
        """Build from a checkpoint manifest.

        The checkpoint is the authority on values that are properties of the
        weights — silence ids, vocab, step count — precisely so they cannot be
        re-guessed by whoever writes the next backend.
        """

        # Raised, not asserted: a manifest is external data, and `python -O`
        # strips asserts — which would turn a malformed checkpoint from a
        # named error into a TypeError deep in the config, or worse, into a
        # silently wrong algorithm. This function is where "the manifest is
        # the authority" is enforced, so it has to hold under -O too.
        #
        # Absent and present-but-wrong are different, and the difference is
        # load-bearing. `manifest.get(key) or default` treated them alike, so a
        # `window: []` or a `sampling_defaults: {}` from a truncated or
        # hand-edited pack loaded silently as the defaults — a checkpoint
        # running an algorithm no one chose, under a fingerprint that says it
        # was chosen. Missing keys still default (older packs predate several
        # blocks); present keys must be the right shape or the load fails.
        def block(key: str, kind: type, default: object) -> object:
            if key not in manifest:
                return default
            value = manifest[key]
            if not isinstance(value, kind):
                raise ValueError(
                    f"manifest key {key!r} must be {kind.__name__}, got {type(value).__name__}"
                )
            # `str` *is* a Sequence, so `"123"` passed the type check above and
            # was then iterated character by character: `silence_token_ids:
            # "123"` loaded as three arbitrary tokens exempted from both the
            # repetition penalty and the min_p floor. It loaded without a word,
            # under a fingerprint that faithfully recorded the wrong recipe.
            # `euler_grid` already guarded against this class a hundred lines
            # below; the guard belongs where every sequence field passes.
            if kind is Sequence and isinstance(value, (str, bytes)):
                raise ValueError(
                    f"manifest key {key!r} must be a list, got a "
                    f"{type(value).__name__} — a string is a sequence of "
                    "characters, which is not what this field means"
                )
            return value

        # Any, not object, for the same reason the asserts these replace gave
        # Any: the values are parsed JSON and are converted with int()/float()
        # a few lines below. The runtime check above is the part that matters.
        sampling_defaults = cast(Mapping[str, Any], block("sampling_defaults", Mapping, {}))
        sil = cast(Sequence[Any], block("silence_token_ids", Sequence, ()))
        speech = cast(Mapping[str, Any], block("speech_tokens", Mapping, {}))

        # The manifest is parsed JSON, so top-level reads are typed `object`;
        # these two narrow a numeric value with a check instead of an ignore.
        def geti(key: str, default: int) -> int:
            value = manifest.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"manifest[{key!r}] should be a number, got {value!r}")
            return int(value)

        def getf(key: str, default: float) -> float:
            value = manifest.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"manifest[{key!r}] should be a number, got {value!r}")
            return float(value)

        # Guidance mode is read, not defaulted. A teacher checkpoint loading
        # silently as single_path is the founding defect with its arrow
        # reversed, and it would be just as invisible.
        guidance = str(manifest.get("guidance", "single_path"))
        rate = getf("guidance_rate", 0.0)
        if guidance not in ("single_path", "cfg_dual_path"):
            raise ValueError(f"manifest declares unknown guidance mode {guidance!r}")

        # Window recipe and EOS floor: carried by amended checkpoints
        # (tools/amend_manifest.py). Absent on older packs, where the backend
        # supplies its production constants — the manifest is preferred
        # precisely so a future backend cannot re-guess either value.
        #
        # ``window: null`` is meaningful and is not an error: it declares the
        # ragged window explicitly, which is what ``WindowConfig()`` is. Any
        # other non-mapping is malformed and must be refused: falling through
        # to the ragged default would make a broken key indistinguishable from
        # a deliberate one — while ``backends.production_algorithm`` reads the
        # key's mere *presence* as authority and skips the production fallback.
        window = WindowConfig()
        if "window" in manifest and manifest["window"] is not None:
            win = manifest["window"]
            if not isinstance(win, Mapping):
                raise ValueError(
                    f"manifest['window'] must be a mapping or null (ragged), got "
                    f"{type(win).__name__}"
                )
            window = WindowConfig(
                max_speech_tokens=int(win.get("max_speech_tokens", 255)),
                static_length=(
                    None if win.get("static_length") is None else int(win["static_length"])
                ),
                pad_token_id=(
                    None if win.get("pad_token_id") is None else int(win["pad_token_id"])
                ),
                static_prompt_tokens=(
                    None
                    if win.get("static_prompt_tokens") is None
                    else int(win["static_prompt_tokens"])
                ),
            )
        eos = cast(Mapping[str, Any], block("eos_floor", Mapping, {}))
        eos_floor = int(eos.get("min_tokens_floor", 0))
        eos_ratio = float(eos.get("min_tokens_text_ratio", 0.0))

        # Chunking: parsed, not ignored. A manifest could declare `enabled`,
        # `max_tokens`, `prefix_tokens` and `split_on` and the runtime would
        # build a default ChunkConfig regardless — so a checkpoint and the
        # engine running it could share a recipe_version while breathing in
        # different places. `prefix_tokens` in particular is hashed into the
        # fingerprint, so the mismatch was invisible from both ends.
        chunk = cast(Mapping[str, Any], block("chunking", Mapping, {}))
        default_chunk = ChunkConfig()
        if "split_on" in chunk and isinstance(chunk["split_on"], (str, bytes)):
            # Same trap as `silence_token_ids`, and worse: `split_on: ". "`
            # loaded as `('.', ' ')`, which breaks at the full stop inside
            # "Version 3.14" and again at every space. A different breathing
            # recipe, accepted silently.
            raise ValueError(
                "manifest key 'chunking.split_on' must be a list of strings, got a "
                "string — a string is a sequence of characters, so each character "
                "would become its own split candidate"
            )
        chunking = ChunkConfig(
            enabled=bool(chunk.get("enabled", default_chunk.enabled)),
            max_tokens=int(chunk.get("max_tokens", default_chunk.max_tokens)),
            prefix_tokens=int(chunk.get("prefix_tokens", default_chunk.prefix_tokens)),
            first_chunk_max_tokens=(
                None
                if chunk.get("first_chunk_max_tokens") is None
                else int(chunk["first_chunk_max_tokens"])
            )
            if "first_chunk_max_tokens" in chunk
            else default_chunk.first_chunk_max_tokens,
            split_on=(
                tuple(str(s) for s in chunk["split_on"])
                if "split_on" in chunk
                else default_chunk.split_on
            ),
        )

        # Postprocess: the artifact detectors, parsed like every other block.
        # One recipe means one accepted value: a manifest naming any other tag
        # is a checkpoint this engine cannot claim to render faithfully, and
        # believing the string would put it in every Result and fingerprint.
        # Absence is not a tag; it is the shipping default left unstated.
        recipe = str(manifest.get("recipe_version", RECIPE_VERSION))
        if recipe != RECIPE_VERSION:
            raise ValueError(
                f"manifest declares recipe_version {recipe!r}; the only recipe "
                f"is {RECIPE_VERSION!r}"
            )
        postprocess = _postprocess_from_manifest(
            cast(Mapping[str, Any], block("postprocess", Mapping, {}))
        )

        # An explicit Euler grid overrides the cosine schedule; `null` means
        # "use the schedule", which is the default. Validated by
        # AlgorithmConfig.__post_init__ against euler_steps.
        grid_raw = manifest.get("euler_grid")
        if grid_raw is None:
            euler_grid: tuple[float, ...] | None = None
        elif isinstance(grid_raw, Sequence) and not isinstance(grid_raw, (str, bytes)):
            euler_grid = tuple(float(x) for x in grid_raw)
        else:
            raise ValueError(
                f"manifest['euler_grid'] must be a list of floats or null, got "
                f"{type(grid_raw).__name__}"
            )

        return cls(
            recipe_version=recipe,
            # narrowed by the membership check above; the Literal can't see that
            guidance=guidance,  # type: ignore[arg-type]
            guidance_rate=rate,
            euler_steps=geti("n_cfm_timesteps", 2),
            euler_grid=euler_grid,
            token_rate_hz=getf("token_rate_hz", 25.0),
            chunking=chunking,
            sample_rate=geti("sample_rate", 24_000),
            speech_vocab_size=geti("speech_vocab_size", 8194),
            start_speech_token=int(speech.get("start", 6561)),
            stop_speech_token=int(speech.get("stop", 6562)),
            sampling=SamplingConfig(
                temperature=float(sampling_defaults.get("temperature", 0.8)),
                repetition_penalty=float(sampling_defaults.get("repetition_penalty", 1.2)),
                min_p=float(sampling_defaults.get("min_p", 0.05)),
                max_new_tokens=int(sampling_defaults.get("max_new_tokens", 255)),
                silence_token_ids=tuple(int(t) for t in sil),
                min_tokens_floor=eos_floor,
                min_tokens_text_ratio=eos_ratio,
            ),
            window=window,
            postprocess=postprocess,
        )


DEFAULT_ALGORITHM = AlgorithmConfig()
"""The shipping algorithm. Anything else is a deliberate, declared deviation."""


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """How a backend gets there. Free to differ; never changes what is computed.

    Precision is the one honest grey zone: it lives here because it is a
    hardware constraint (the Neural Engine is fp16-native and has no bf16
    datapath at all), yet it does perturb the output. That is why it is
    declared per module and measured against bands rather than assumed free.
    """

    device: Device = "cpu"
    """Where stages run unless overridden below."""

    generator_device: Device | None = None
    renderer_device: Device | None = None
    """Per-stage placement, because the two stages want different hardware.

    The token generator is autoregressive: a few hundred tiny dispatches per
    token, at batch one. The renderer is one large parallel pass. Since the
    streaming pipeline the split's value is *parallelism*: the renderer renders
    window k on the GPU while the generator computes window k+1 on the CPU.
    Measured on an M3 Pro (torch 2.13), a six-window passage runs at RTF 3.37
    split against 2.15 with both stages on MPS, where they contend for one
    device.

    ``None`` means "use ``device``". Both are part of the execution layer: they
    change how fast, never what is computed, which is exactly why the components
    can be split at all — they pass arrays to each other, not live tensors on a
    particular device.
    """

    precision: Mapping[str, Precision] = field(
        default_factory=lambda: {
            "token_generator": "fp32",
            "mel_decoder.estimator": "fp32",
            "mel_decoder.encoder": "fp32",
            "vocoder": "fp32",
        }
    )
    """Per-module dtype. The defaults are conservative; the shipping map lives
    in the checkpoint manifest.

    Three of these are settled by measurement and should not be casually
    changed: the token generator tolerates fp16 easily (median KL 1.3e-06, top-1
    99.9%) because a sampling decision boundary annihilates sub-threshold error;
    the flow encoder does **not** (fp16 there gives mel correlation 0.619 and
    +22 dB of high-frequency energy); and the vocoder does not (fp16 produces an
    audible tone at Nyquist, from a cumulative phase accumulator in the source
    module).
    """

    compile_model: bool = False
    """Capture the decode step as a CUDA graph, exactly like ``cuda_graphs``.

    Both flags mean the same thing — the per-token decode as one captured graph
    instead of ~1442 kernel launches — and both run over the same static KV
    cache. ``torch.compile``'s own capture path hits an inductor mask-alignment
    bug on this model, so the flags share the manual ``torch.cuda.CUDAGraph``
    capture; kept as two flags because the config declared them as separate
    knobs. On non-CUDA devices (or Pascal) the static path runs eagerly with
    no launch win, and it shares the ``cuda_graphs`` token-drift caveat."""

    cuda_graphs: bool = False
    """Capture the decode step as a CUDA graph and replay it per token.

    The decode runs over a **static KV cache** (preallocated fixed-address
    buffers written in place via ``index_put_``), so the whole step — embed,
    16 layers, head — is one captured graph instead of ~1442 kernel launches
    (53% of the step was CPU-side launch overhead). Measured on an RTX 3090:
    token generator ~5x faster, end-to-end RTF 2.3x → 8.7x, determinism check
    passing.

    Three measured caveats. CUDA graphs need **sm_70+** (Volta): on Pascal
    (GTX 1080 Ti) the flag falls back to the static cache running eagerly,
    which is *slower* than eager there, so pass it only on Ampere or newer.
    And the static cache is the identity contract's ``equivalent`` class: the
    attention reduces over a **padded** buffer, which switches the cuBLAS
    kernel at large widths and drifts logits — ~2e-4 per layer at a 750-token
    prefill, ~1e-6 at short prefills, deterministic but *not* token-identical:
    a 255-token     utterance diverges from eager around token 26–130. Opt-in; the
    default (dynamic cache) stays bit-identical.

    One more caveat: the graph is captured **per synthesis**, not once per
    engine. The static KV buffers are sized to the utterance's prefill length,
    so a capture cached across utterances would bake in the wrong shapes; every
    ``generate()`` runs three warm-up steps plus a synchronise plus the capture
    before replay. That amortises over a long window (255 tokens ≈ 3 s of
    audio per synthesis) but can cost more than it saves on a short agent
    utterance of 20-30 tokens — measure before relying on it for
    time-to-first-audio."""

    attention: Literal["auto", "eager", "sdpa"] = "auto"
    """``auto`` resolves to ``eager`` on MPS, where the fused path aborts the
    process outright — ``LLVM ERROR: Failed to infer result type(s)`` from
    ``mps_matmul``, with no Python traceback — and to ``sdpa`` elsewhere."""

    onnx_provider: ONNXProvider = "auto"
    """Which execution provider the ONNX backend runs the graphs on.

    Read by the ONNX backend and by nothing else. ``auto`` takes the best
    provider the installed onnxruntime and this machine actually offer — cuda,
    then coreml, then directml, then cpu — and the backend rewrites this field
    to the provider it chose, so ``describe()`` reports what ran rather than
    what was asked for.

    An explicit provider this build does not offer is an error, never a quiet
    downgrade to cpu. Every ONNX session was pinned to ``CPUExecutionProvider``
    until this field existed, which is how the published figures came to
    describe the torch path only while an ONNX user got ~1.2x real time; a
    silent fallback would restore that failure with a benchmark row claiming a
    GPU it never touched.

    A GPU provider may change the sampled tokens. That is a per-provider
    measurement, not an assumption of freedom.
    """

    num_threads: int | None = None

    allow_tf32: bool = False
    """Whether Ampere-class hardware may use TF32 for matmuls and convolutions.

    Declared rather than inherited, because the defaults are a trap: PyTorch
    ships `cudnn.allow_tf32` **on** and `cuda.matmul.allow_tf32` **off**, so
    "plain fp32" is by default neither fp32 nor bit-reproducible against it. We
    measured that costing 5% and breaking bit-exactness on the renderer, and a
    baseline that inherited it silently made TF32 look worth 1.05x when it is
    worth 1.17x.

    Off by default. Turn it on knowingly, and it will show up in `describe()`.
    """

    deterministic: bool = True
    """Pin cuDNN algorithm selection and disable TF32.

    Costs about 5% end to end and buys the promise that the same seed on the
    same build gives a bit-identical waveform. Without it the vocoder's
    convolutions pick algorithms freely and two runs differ by ~5e-06.
    """

    def __post_init__(self) -> None:
        # The Literal is a promise to the type checker, not a runtime check,
        # and this value arrives from CLI flags and JSON bodies as often as
        # from Python. A typo ("CUDA", "metal") has to fail here: unvalidated,
        # it would reach the backend's provider lookup as a KeyError naming an
        # onnxruntime symbol, or worse, miss a comparison and run on cpu.
        if self.onnx_provider not in ONNX_PROVIDERS:
            raise ValueError(
                f"unknown onnx_provider {self.onnx_provider!r}; "
                f"expected one of {', '.join(ONNX_PROVIDERS)}"
            )

    def resolved_generator_device(self) -> Device:
        return self.generator_device or self.device

    def resolved_renderer_device(self) -> Device:
        return self.renderer_device or self.device

    def resolved_attention(self) -> Literal["eager", "sdpa"]:
        if self.attention != "auto":
            return self.attention
        # Judged by where the generator runs: it owns the attention, and MPS is
        # where the fused path aborts the process rather than raising.
        if self.resolved_generator_device() == "mps":
            return "eager"
        # SDPA on CUDA lowers to flash-attention, which exists on Ampere and
        # newer (compute capability >= 8.0). On older NVIDIA GPUs (Pascal,
        # Volta, Turing — 1080 Ti, V100, 2080) the fused path raises
        # "FlashAttention only supports Ampere or newer" at the first step,
        # with a traceback that names none of this code. Fall back to eager
        # rather than requiring the caller to know their GPU's generation.
        gen = self.resolved_generator_device()
        if gen.startswith("cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    idx = gen.split(":", 1)
                    device_index = int(idx[1]) if len(idx) > 1 else 0
                    cap = torch.cuda.get_device_capability(device_index)
                    if cap[0] < 8:
                        return "eager"
            except (ImportError, RuntimeError, AssertionError, ValueError, IndexError) as exc:
                # Named, not bare. `except Exception` here also swallowed a torch
                # that fails to import for a real reason — a broken CUDA
                # install, a version mismatch — and answered "sdpa", so a
                # misconfigured machine looked like a working one until the
                # first forward pass. These five are what a capability probe
                # actually raises; anything else is a defect worth surfacing.
                import warnings

                warnings.warn(
                    f"could not read the CUDA capability of {gen!r} ({exc}); "
                    "assuming SDPA is supported",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return "sdpa"
        return "sdpa"

    def describe(self) -> str:
        prec = ",".join(f"{k.split('.')[-1]}={v}" for k, v in sorted(self.precision.items()))
        placement: str = self.device
        if self.generator_device or self.renderer_device:
            placement = (
                f"gen={self.resolved_generator_device()}/"
                f"render={self.resolved_renderer_device()}"
            )
        flags = [
            placement,
            f"attn={self.resolved_attention()}",
            f"prec[{prec}]",
            f"tf32={'on' if self.allow_tf32 else 'off'}",
        ]
        # Only where it means something: every other backend ignores the field,
        # and the ONNX backend rewrites it to the provider it resolved, so this
        # names the provider that ran. An explicit value shows even off the onnx
        # device, because a caller who named one is owed the answer.
        if self.device.split(":", 1)[0] == "onnx" or self.onnx_provider != "auto":
            flags.append(f"provider={self.onnx_provider}")
        if self.compile_model:
            flags.append("compiled")
        if self.cuda_graphs:
            flags.append("graphs")
        if self.deterministic:
            flags.append("deterministic")
        return "exec[" + " ".join(flags) + "]"


@dataclass(frozen=True, slots=True)
class ExecutionOverrides:
    """A *partial* :class:`ExecutionConfig`: change one knob, inherit the rest.

    This exists because "unset" and "set to the value that happens to be the
    dataclass default" are different requests, and a plain ``ExecutionConfig``
    cannot tell them apart. The old merge compared each field against
    ``ExecutionConfig()`` and treated equality as "not specified", so:

    * asking for an all-fp32 map — the default map — kept the manifest's fp16
      generator, which changes the sampled tokens. A conformance run that
      explicitly requested fp32 measured fp16 and said fp32.
    * asking for ``device="cpu"`` on a build whose default device was CUDA was
      ignored, because ``"cpu"`` is the field default.

    Every field here is ``None`` until named, so an explicit value always wins
    even when it equals a default. ``precision`` merges per module rather than
    replacing the map, so ``ExecutionOverrides(precision={"vocoder": "fp32"})``
    means *that one module*, and naming all four modules replaces the map
    entirely — a partial dict can never silently drop a module's dtype.

    A full :class:`ExecutionConfig` passed to :func:`~loudkit.load` remains
    valid and is taken as **complete**: it is the configuration, not a patch.
    """

    device: Device | None = None
    generator_device: Device | None = None
    renderer_device: Device | None = None
    precision: Mapping[str, Precision] | None = None
    compile_model: bool | None = None
    cuda_graphs: bool | None = None
    attention: Literal["auto", "eager", "sdpa"] | None = None
    onnx_provider: ONNXProvider | None = None
    num_threads: int | None = None
    allow_tf32: bool | None = None
    deterministic: bool | None = None

    def applied_to(self, defaults: ExecutionConfig) -> ExecutionConfig:
        """Layer these overrides onto a backend's resolved defaults."""
        from dataclasses import fields, replace

        kwargs: dict[str, object] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            if f.name == "precision":
                merged = dict(defaults.precision)
                merged.update(cast("Mapping[str, Precision]", value))
                kwargs[f.name] = merged
            else:
                kwargs[f.name] = value
        return replace(defaults, **kwargs)  # type: ignore[arg-type]

    def describe(self) -> str:
        """Only the fields that were actually named."""
        from dataclasses import fields

        named = [
            f"{f.name}={getattr(self, f.name)!r}"
            for f in fields(self)
            if getattr(self, f.name) is not None
        ]
        return "overrides[" + (" ".join(named) or "none") + "]"
