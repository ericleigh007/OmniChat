from __future__ import annotations

import base64
import io
import wave

import numpy as np
import torch

from tools.model import model_manager as mm
from tools.model.backends.gemma_transformers_backend import GemmaTransformersBackend


def test_gemma_transformers_capabilities_report_native_audio_input():
    backend = GemmaTransformersBackend()

    mm.set_gemma_transformers_config(
        checkpoint="google/gemma-4-E4B-it",
        speech_backend="none",
        local_files_only=True,
    )

    capabilities = backend.get_capabilities()

    assert capabilities["backend"] == "gemma_transformers"
    assert capabilities["supports_audio_input"] is True
    assert capabilities["supports_audio_output"] is False
    assert capabilities["supports_image_input"] is True
    assert capabilities["supports_video_input"] is True


def test_gemma_transformers_capabilities_enable_hybrid_tts():
    backend = GemmaTransformersBackend()

    mm.set_gemma_transformers_config(
        checkpoint="google/gemma-4-E4B-it",
        speech_backend="minicpm_streaming",
        local_files_only=True,
    )

    capabilities = backend.get_capabilities()

    assert capabilities["supports_audio_output"] is True
    assert capabilities["supports_streaming_audio"] is True
    assert capabilities["output_sample_rate"] == 24000


def test_gemma_transformers_normalize_audio_array_to_channel_first():
    backend = GemmaTransformersBackend()

    normalized = backend._normalize_content([np.zeros(1600, dtype=np.float32), "hello"])

    assert normalized[0]["type"] == "audio"
    assert normalized[0]["audio"].shape == (1600,)
    assert normalized[1] == {"type": "text", "text": "hello"}


def test_gemma_transformers_decodes_input_audio_payload(monkeypatch):
    backend = GemmaTransformersBackend()

    samples = (np.zeros(800, dtype=np.int16)).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples)

    normalized = backend._normalize_content([
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(buf.getvalue()).decode("ascii"), "format": "wav"}}
    ])

    assert normalized[0]["type"] == "audio"
    assert normalized[0]["audio"].shape == (800,)


def test_gemma_transformers_chat_hands_off_text_to_minicpm_tts(monkeypatch):
    backend = GemmaTransformersBackend()

    mm.set_gemma_transformers_config(
        checkpoint="google/gemma-4-E4B-it",
        speech_backend="minicpm_streaming",
        local_files_only=True,
    )

    monkeypatch.setattr(
        backend,
        "_generate_response",
        lambda **_kwargs: {"text": "hello from gemma", "audio": None, "audio_path": None, "sample_rate": None, "reasoning": ""},
    )
    monkeypatch.setattr(
        mm,
        "synthesize_text_with_minicpm",
        lambda text, **_kwargs: {"audio": np.zeros(2400, dtype=np.float32), "audio_path": None, "sample_rate": 24000},
    )

    result = backend.chat(messages=[{"role": "user", "content": "hello"}], generate_audio=True)

    assert result["text"] == "hello from gemma"
    assert result["audio"].shape == (2400,)
    assert result["sample_rate"] == 24000


def test_gemma_transformers_process_video_forces_configured_backend(monkeypatch):
    backend = GemmaTransformersBackend()

    mm.set_gemma_transformers_config(
        checkpoint="google/gemma-4-E4B-it",
        speech_backend="none",
        local_files_only=True,
        video_backend="pyav",
    )

    captured = {}

    class FakeProcessor:
        def apply_chat_template(self, conversation, **_kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

        def decode(self, _tokens, skip_special_tokens=False):
            return "decoded"

        def parse_response(self, _decoded):
            return {"content": "video ok", "thinking": ""}

    class FakeModel:
        def generate(self, **_kwargs):
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    class FakeHandle:
        processor = FakeProcessor()
        model = FakeModel()

    monkeypatch.setattr(backend, "_get_handle", lambda: FakeHandle())
    monkeypatch.setattr(backend, "_move_inputs", lambda inputs, _handle: inputs)
    monkeypatch.setattr(backend, "_collect_multimodal_inputs", lambda _conversation, video_backend: ([], [], ["decoded-video"], [{"fps": 24}]))

    result = backend.process_video("clip.mp4", prompt="Describe this clip.", generate_audio=False, max_new_tokens=8)

    assert captured["videos"] == ["decoded-video"]
    assert captured["videos_kwargs"] == {"video_metadata": [{"fps": 24}], "return_metadata": True}
    assert result["text"] == "video ok"