from __future__ import annotations

import importlib

import mlx.core as mx
import pytest


def test_expired_deadline_raises_typed_terminal_error() -> None:
    from mlx_whisper.decoding import DecodeTimeoutError, _check_decode_deadline

    with pytest.raises(DecodeTimeoutError) as exc_info:
        _check_decode_deadline(
            deadline=10.0,
            now=10.5,
            timeout=3.0,
            phase="decoder",
            window_index=2,
            temperature=0.4,
            token_count=17,
        )

    error = exc_info.value
    assert error.phase == "decoder"
    assert error.timeout == 3.0
    assert error.window_index == 2
    assert error.temperature == 0.4
    assert error.token_count == 17
    assert error.partial_result is False


def test_transcribe_reuses_one_deadline_across_windows_and_fallbacks(
    monkeypatch,
) -> None:
    transcribe_module = importlib.import_module("mlx_whisper.transcribe")

    class Dims:
        n_mels = 80
        n_audio_ctx = 1500

    class FakeModel:
        dims = Dims()
        is_multilingual = False
        num_languages = 1

        def __init__(self) -> None:
            self.options = []

        def decode(self, _segment, options):
            from mlx_whisper.decoding import DecodingResult

            self.options.append(options)
            attempt_in_window = (len(self.options) - 1) % 2
            return DecodingResult(
                audio_features=mx.zeros((1, 1)),
                language="en",
                tokens=[1],
                text="x",
                avg_logprob=-0.1,
                no_speech_prob=0.0,
                temperature=options.temperature,
                compression_ratio=3.0 if attempt_in_window == 0 else 1.0,
            )

    class FakeTokenizer:
        eot = 1000
        timestamp_begin = 2000

        @staticmethod
        def encode(_text):
            return []

        @staticmethod
        def decode(tokens):
            return "x" * len(tokens)

    model = FakeModel()
    monkeypatch.setattr(transcribe_module.ModelHolder, "get_model", lambda *_args: model)
    monkeypatch.setattr(
        transcribe_module,
        "log_mel_spectrogram",
        lambda *_args, **_kwargs: mx.zeros(
            (transcribe_module.N_FRAMES * 3, model.dims.n_mels)
        ),
    )
    monkeypatch.setattr(
        transcribe_module, "get_tokenizer", lambda *_args, **_kwargs: FakeTokenizer()
    )
    monkeypatch.setattr(transcribe_module.time, "monotonic", lambda: 100.0)

    events: list[dict] = []
    result = transcribe_module.transcribe(
        mx.zeros(1),
        language="en",
        temperature=(0.0, 0.2),
        decode_timeout=30.0,
        telemetry_callback=events.append,
    )

    assert result["text"]
    assert len(model.options) == 4
    assert {option.decode_deadline for option in model.options} == {130.0}
    assert [event["event"] for event in events].count("window_start") == 2
    assert [event["event"] for event in events].count("decode_attempt_end") == 4
    assert events[-1]["event"] == "transcription_complete"


def test_deadline_stops_temperature_fallback_instead_of_returning_partial_text(
    monkeypatch,
) -> None:
    from mlx_whisper.decoding import DecodeTimeoutError

    transcribe_module = importlib.import_module("mlx_whisper.transcribe")

    class Clock:
        now = 20.0

    clock = Clock()

    class Dims:
        n_mels = 80
        n_audio_ctx = 1500

    class FakeModel:
        dims = Dims()
        is_multilingual = False
        num_languages = 1

        def __init__(self) -> None:
            self.attempts = 0

        def decode(self, _segment, options):
            from mlx_whisper.decoding import DecodingResult

            self.attempts += 1
            clock.now += 6.0
            return DecodingResult(
                audio_features=mx.zeros((1, 1)),
                language="en",
                tokens=[1],
                text="partial",
                avg_logprob=-2.0,
                no_speech_prob=0.0,
                temperature=options.temperature,
                compression_ratio=3.0,
            )

    class FakeTokenizer:
        eot = 1000
        timestamp_begin = 2000

        @staticmethod
        def encode(_text):
            return []

        @staticmethod
        def decode(_tokens):
            return "partial"

    model = FakeModel()
    monkeypatch.setattr(transcribe_module.ModelHolder, "get_model", lambda *_args: model)
    monkeypatch.setattr(
        transcribe_module,
        "log_mel_spectrogram",
        lambda *_args, **_kwargs: mx.zeros(
            (transcribe_module.N_FRAMES * 2, model.dims.n_mels)
        ),
    )
    monkeypatch.setattr(
        transcribe_module, "get_tokenizer", lambda *_args, **_kwargs: FakeTokenizer()
    )
    monkeypatch.setattr(transcribe_module.time, "monotonic", lambda: clock.now)

    events: list[dict] = []
    with pytest.raises(DecodeTimeoutError) as exc_info:
        transcribe_module.transcribe(
            mx.zeros(1),
            language="en",
            temperature=(0.0, 0.2, 0.4),
            decode_timeout=10.0,
            telemetry_callback=events.append,
        )

    assert model.attempts == 2
    assert exc_info.value.partial_result is False
    assert events[-1]["event"] == "deadline_exceeded"
    assert events[-1]["phase"] == "temperature_fallback"
