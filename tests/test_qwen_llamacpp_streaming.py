from __future__ import annotations

import json

import tools.model.model_manager as model_manager
from tools.model.backends.qwen_llamacpp_backend import QwenLlamaCppBackend


class _FakeStreamingResponse:
    def __init__(self, lines: list[str]):
        self._lines = lines
        self.status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line if decode_unicode else line.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_qwen_llamacpp_server_streaming_emits_incremental_text(monkeypatch):
    backend = QwenLlamaCppBackend()

    monkeypatch.setattr(backend, "_ensure_server_ready", lambda: "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(
        "tools.model.backends.qwen_llamacpp_backend.mm.get_qwen_llamacpp_config",
        lambda: {"timeout_s": 120.0},
    )

    streamed_lines = [
        "data: " + json.dumps({"content": "Hello ", "stop": False}),
        "data: " + json.dumps({"content": "world", "stop": False}),
        "data: " + json.dumps({"content": "!", "stop_type": "eos", "tokens_evaluated": 12, "tokens_predicted": 3}),
        "data: [DONE]",
    ]

    monkeypatch.setattr(
        "tools.model.backends.qwen_llamacpp_backend.requests.post",
        lambda *args, **kwargs: _FakeStreamingResponse(streamed_lines),
    )

    stream = backend._run_server_completion_streaming(
        prompt="Say hello.",
        system_prompt="You are OmniChat.",
        temperature=0.2,
        max_new_tokens=32,
        enable_thinking=False,
    )
    parts = []
    while True:
        try:
            parts.append(next(stream))
        except StopIteration as stop:
            full_text = stop.value
            break

    assert parts == ["Hello ", "world", "!"]
    assert full_text == "Hello world!"
    assert backend.get_runtime_status()["last_server_stop_reason"] == "eos"


def test_qwen_llamacpp_chat_streaming_returns_final_override(monkeypatch):
    backend = QwenLlamaCppBackend()

    monkeypatch.setattr(backend, "_normalize_messages", lambda messages: messages)
    monkeypatch.setattr(backend, "_messages_are_text_only", lambda messages: True)
    monkeypatch.setattr(backend, "_prepare_conversation", lambda *args, **kwargs: ("Prompt", "System", None, None))
    monkeypatch.setattr(backend, "_format_chat_display", lambda text, reasoning: text)
    monkeypatch.setattr(backend, "_split_reasoning", lambda text: ("", text))
    monkeypatch.setattr(backend, "_repair_incomplete_response", lambda reasoning, text: (reasoning, "Hello world!"))
    monkeypatch.setattr(backend, "_response_satisfies_contract", lambda text, prompt: True)

    def _fake_stream(*args, **kwargs):
        yield "Hello "
        yield "world"
        return "Hello world"

    monkeypatch.setattr(backend, "_run_server_completion_streaming", _fake_stream)

    stream = backend.chat_streaming(
        messages=[{"role": "user", "content": ["Hi"]}],
        generate_audio=False,
        temperature=0.2,
        max_new_tokens=32,
        enable_thinking=False,
        trace_context={"request_id": "test-stream"},
    )
    chunks = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            final_payload = stop.value
            break

    assert chunks == [(None, "Hello "), (None, "world"), (None, "!")]
    assert final_payload == {"final_text": "Hello world!"}


def test_qwen_llamacpp_chat_streaming_passes_trace_context_to_minicpm_tts(monkeypatch):
    backend = QwenLlamaCppBackend()

    monkeypatch.setattr(backend, "_normalize_messages", lambda messages: messages)
    monkeypatch.setattr(backend, "_messages_are_text_only", lambda messages: True)
    monkeypatch.setattr(backend, "_prepare_conversation", lambda *args, **kwargs: ("Prompt", "System", None, None))
    def _fake_stream_response(**kwargs):
        if False:
            yield None, ""
        return "Hello world"

    monkeypatch.setattr(backend, "_stream_text_only_response", _fake_stream_response)
    captured_kwargs = {}

    def _fake_tts(text, **kwargs):
        captured_kwargs.update(kwargs)
        if False:
            yield None, ""
        return

    monkeypatch.setattr("tools.model.backends.qwen_llamacpp_backend.mm.stream_text_to_speech_with_minicpm", _fake_tts)

    stream = backend.chat_streaming(
        messages=[{"role": "user", "content": ["Hi"]}],
        generate_audio=True,
        temperature=0.2,
        max_new_tokens=32,
        enable_thinking=False,
        trace_context={"request_id": "trace-456"},
    )
    while True:
        try:
            next(stream)
        except StopIteration:
            break

    assert captured_kwargs["trace_context"] == {"request_id": "trace-456"}
    assert captured_kwargs["source_backend"] == "qwen_llamacpp"


def test_qwen_llamacpp_streaming_hides_leaked_think_blocks(monkeypatch):
    backend = QwenLlamaCppBackend()

    def _fake_stream(*args, **kwargs):
        yield "<think>"
        yield "secret plan"
        yield "</think>Hello "
        yield "world"
        return "<think>secret plan</think>Hello world"

    monkeypatch.setattr(backend, "_run_server_completion_streaming", _fake_stream)
    monkeypatch.setattr(backend, "_split_reasoning", QwenLlamaCppBackend._split_reasoning.__get__(backend, QwenLlamaCppBackend))
    monkeypatch.setattr(backend, "_response_satisfies_contract", lambda text, prompt: True)

    stream = backend._stream_text_only_response(
        request_id="test-think-filter",
        started=0.0,
        prompt="Prompt",
        system_prompt="System",
        temperature=0.2,
        max_new_tokens=32,
    )
    chunks = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            final_text = stop.value
            break

    assert chunks == [(None, "Hello"), (None, " world")]
    assert final_text == "Hello world"


def test_qwen_llamacpp_streaming_extracts_answer_inside_reasoning(monkeypatch):
    backend = QwenLlamaCppBackend()

    def _fake_stream(*args, **kwargs):
        yield "<think>"
        yield "Final answer: Hello"
        yield " world"
        yield "</think>"
        return "<think>Final answer: Hello world</think>"

    monkeypatch.setattr(backend, "_run_server_completion_streaming", _fake_stream)
    monkeypatch.setattr(backend, "_split_reasoning", QwenLlamaCppBackend._split_reasoning.__get__(backend, QwenLlamaCppBackend))
    monkeypatch.setattr(backend, "_response_satisfies_contract", lambda text, prompt: True)

    stream = backend._stream_text_only_response(
        request_id="test-answer-from-reasoning",
        started=0.0,
        prompt="Prompt",
        system_prompt="System",
        temperature=0.2,
        max_new_tokens=32,
    )
    chunks = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            final_text = stop.value
            break

    assert chunks == [(None, "Hello"), (None, " world")]
    assert final_text == "Hello world"


def test_qwen_llamacpp_text_only_response_restarts_server_before_cli(monkeypatch):
    backend = QwenLlamaCppBackend()

    server_calls = []
    shutdown_calls = []

    def _fake_server_completion(**kwargs):
        server_calls.append(kwargs)
        if len(server_calls) == 1:
            raise RuntimeError("server timed out")
        return "Recovered after restart."

    monkeypatch.setattr(backend, "_run_server_completion", _fake_server_completion)
    monkeypatch.setattr(backend, "_shutdown_server", lambda: shutdown_calls.append(True))
    monkeypatch.setattr(backend, "_run_cli", lambda **kwargs: (_ for _ in ()).throw(AssertionError("CLI fallback should not run")))

    reasoning, text = backend._complete_text_only_response(
        prompt="Say hello.",
        system_prompt="You are OmniChat.",
        temperature=0.2,
        max_new_tokens=64,
        enable_thinking=False,
    )

    assert reasoning == ""
    assert text == "Recovered after restart."
    assert len(server_calls) == 2
    assert shutdown_calls == [True]


def test_qwen_llamacpp_text_only_response_falls_back_to_cli_after_restart_failure(monkeypatch):
    backend = QwenLlamaCppBackend()

    shutdown_calls = []
    cli_calls = []

    monkeypatch.setattr(backend, "_run_server_completion", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("server timed out")))
    monkeypatch.setattr(backend, "_shutdown_server", lambda: shutdown_calls.append(True))

    def _fake_cli(**kwargs):
        cli_calls.append(kwargs)
        return "CLI fallback answer."

    monkeypatch.setattr(backend, "_run_cli", _fake_cli)

    reasoning, text = backend._complete_text_only_response(
        prompt="Say hello.",
        system_prompt="You are OmniChat.",
        temperature=0.2,
        max_new_tokens=64,
        enable_thinking=False,
    )

    assert reasoning == ""
    assert text == "CLI fallback answer."
    assert shutdown_calls == [True]
    assert len(cli_calls) == 1


def test_model_manager_chat_streaming_preserves_final_override(monkeypatch):
    class _Backend:
        def chat_streaming(self, **kwargs):
            yield None, "Hello"
            return {"final_text": "Hello world"}

    monkeypatch.setattr(model_manager, "_backend_name", "qwen_llamacpp")
    monkeypatch.setattr(model_manager, "get_backend", lambda: _Backend())

    stream = model_manager.chat_streaming(
        messages=[{"role": "user", "content": ["Hi"]}],
        generate_audio=False,
        trace_context={"request_id": "test-model-manager"},
    )
    chunks = []
    while True:
        try:
            chunks.append(next(stream))
        except StopIteration as stop:
            final_payload = stop.value
            break

    assert chunks == [(None, "Hello")]
    assert final_payload == {"final_text": "Hello world"}
