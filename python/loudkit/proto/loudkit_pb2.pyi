from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesizeRequest(_message.Message):
    __slots__ = ("text", "voice", "seed", "language", "speed", "long_form", "previous_tokens", "audio_format")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    LONG_FORM_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FORMAT_FIELD_NUMBER: _ClassVar[int]
    text: str
    voice: str
    seed: int
    language: str
    speed: float
    long_form: bool
    previous_tokens: _containers.RepeatedScalarFieldContainer[int]
    audio_format: str
    def __init__(self, text: _Optional[str] = ..., voice: _Optional[str] = ..., seed: _Optional[int] = ..., language: _Optional[str] = ..., speed: _Optional[float] = ..., long_form: _Optional[bool] = ..., previous_tokens: _Optional[_Iterable[int]] = ..., audio_format: _Optional[str] = ...) -> None: ...

class SynthesizeResponse(_message.Message):
    __slots__ = ("audio", "media_type", "duration_seconds", "token_count", "truncated", "continuation", "fingerprint", "sample_rate")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TYPE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    CONTINUATION_FIELD_NUMBER: _ClassVar[int]
    FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    media_type: str
    duration_seconds: float
    token_count: int
    truncated: bool
    continuation: _containers.RepeatedScalarFieldContainer[int]
    fingerprint: str
    sample_rate: int
    def __init__(self, audio: _Optional[bytes] = ..., media_type: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., token_count: _Optional[int] = ..., truncated: _Optional[bool] = ..., continuation: _Optional[_Iterable[int]] = ..., fingerprint: _Optional[str] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class SynthesizeChunk(_message.Message):
    __slots__ = ("audio", "media_type", "duration_seconds", "token_count", "truncated", "continuation", "fingerprint", "sample_rate")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    MEDIA_TYPE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_COUNT_FIELD_NUMBER: _ClassVar[int]
    TRUNCATED_FIELD_NUMBER: _ClassVar[int]
    CONTINUATION_FIELD_NUMBER: _ClassVar[int]
    FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    media_type: str
    duration_seconds: float
    token_count: int
    truncated: bool
    continuation: _containers.RepeatedScalarFieldContainer[int]
    fingerprint: str
    sample_rate: int
    def __init__(self, audio: _Optional[bytes] = ..., media_type: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., token_count: _Optional[int] = ..., truncated: _Optional[bool] = ..., continuation: _Optional[_Iterable[int]] = ..., fingerprint: _Optional[str] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class DescribeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DescribeResponse(_message.Message):
    __slots__ = ("algorithm", "execution", "fingerprint", "version", "engine_held_seconds")
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    EXECUTION_FIELD_NUMBER: _ClassVar[int]
    FINGERPRINT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    ENGINE_HELD_SECONDS_FIELD_NUMBER: _ClassVar[int]
    algorithm: str
    execution: str
    fingerprint: str
    version: str
    engine_held_seconds: float
    def __init__(self, algorithm: _Optional[str] = ..., execution: _Optional[str] = ..., fingerprint: _Optional[str] = ..., version: _Optional[str] = ..., engine_held_seconds: _Optional[float] = ...) -> None: ...

class ListVoicesRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListVoicesResponse(_message.Message):
    __slots__ = ("voices",)
    VOICES_FIELD_NUMBER: _ClassVar[int]
    voices: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, voices: _Optional[_Iterable[str]] = ...) -> None: ...
