from __future__ import annotations

import base64
import io
import wave

import numpy as np

from tools.model import model_manager as mm
from tools.model.backends.gemma_llamacpp_backend import GemmaLlamaCppBackend


def test_gemma_llamacpp_capabilities_report_native_audio_input(monkeypatch):
    backend = GemmaLlamaCppBackend()

    monkeypatch.setattr(
        backend,
        "_get_backend_config",
        lambda: {"name": "Gemma 4 llama.cpp", "speech_backend": "none", "llama_root": "C:/tmp"},
    )
    monkeypatch.setattr(backend, "_resolve_server_path", lambda cfg: type("_Path", (), {"exists": lambda self: True})())

    capabilities = backend.get_capabilities()

    assert capabilities["backend"] == "gemma_llamacpp"
    assert capabilities["supports_audio_input"] is True
    assert capabilities["supports_audio_output"] is False
    assert capabilities["supports_image_input"] is True


def test_gemma_llamacpp_normalize_messages_bridges_audio_array_to_text(monkeypatch):
    backend = GemmaLlamaCppBackend()

    monkeypatch.setattr(mm, "transcribe_audio_array_with_minicpm", lambda audio, prompt=None: f"transcript:{len(audio)}")

    normalized = backend._normalize_messages(
        [{"role": "user", "content": [np.zeros(1600, dtype=np.float32), "hello"]}]
    )

    assert normalized[0]["content"][0] == "transcript:1600"
    assert normalized[0]["content"][1] == "hello"


def test_gemma_llamacpp_normalize_messages_bridges_input_audio_payload_to_text(monkeypatch):
    backend = GemmaLlamaCppBackend()

    samples = (np.zeros(800, dtype=np.int16)).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples)

    monkeypatch.setattr(mm, "transcribe_audio_array_with_minicpm", lambda audio, prompt=None: f"bridged:{len(audio)}")

    normalized = backend._normalize_messages([
        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode("ascii"), "format": "wav"}}]}
    ])

    assert normalized[0]["content"] == ["bridged:800"]


def test_gemma_llamacpp_chat_uses_multimodal_server_for_image(monkeypatch):
    backend = GemmaLlamaCppBackend()

    monkeypatch.setattr(backend, "_normalize_messages", lambda messages: messages)
    monkeypatch.setattr(backend, "_messages_are_text_only", lambda messages: False)
    monkeypatch.setattr(backend, "_messages_have_native_multimodal_input", lambda messages: True)
    monkeypatch.setattr(backend, "_prepare_conversation", lambda *args, **kwargs: ("Prompt", "System", None, None))
    monkeypatch.setattr(backend, "_run_server_chat_completion", lambda **kwargs: "Audio-aware answer")
    monkeypatch.setattr(backend, "_split_reasoning", lambda text: ("", text))

    result = backend.chat(
        messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]}],
        generate_audio=False,
        temperature=0.2,
        max_new_tokens=64,
    )

    assert result["text"] == "Audio-aware answer"
    assert result["audio"] is None