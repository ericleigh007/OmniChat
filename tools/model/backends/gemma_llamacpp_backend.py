"""Local Gemma GGUF backend using llama.cpp server for text/image plus MiniCPM audio bridging."""

from __future__ import annotations

import base64
import io

import numpy as np
import soundfile as sf

from tools.model import model_manager as mm
from tools.model.backends.qwen_llamacpp_backend import QwenLlamaCppBackend


class GemmaLlamaCppBackend(QwenLlamaCppBackend):
    """Adapter for local Gemma multimodal GGUF inference via llama.cpp."""

    name = "gemma_llamacpp"

    def _get_backend_config(self) -> dict:
        return mm.get_gemma_llamacpp_config()

    def _supports_native_audio_input(self) -> bool:
        return False

    def _supports_server_multimodal_chat(self) -> bool:
        return True

    def get_capabilities(self) -> dict[str, object]:
        capabilities = super().get_capabilities()
        capabilities["supports_audio_input"] = True
        capabilities["input_sample_rate"] = self._local_input_sample_rate()
        return capabilities

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        normalized_messages: list[dict] = []

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray)):
                content = [content]

            normalized_content: list[object] = []
            for item in content:
                if isinstance(item, np.ndarray) and not self._looks_like_image_array(item):
                    normalized_content.append(self._bridge_audio_array_to_text(item))
                    continue

                if isinstance(item, dict) and item.get("type") in {"audio", "input_audio"}:
                    normalized_content.append(self._bridge_audio_array_to_text(self._coerce_audio_item_to_array(item)))
                    continue

                normalized_content.append(item)

            normalized_messages.append({
                "role": message.get("role", "user"),
                "content": normalized_content,
            })

        return normalized_messages

    def transcribe_audio(
        self,
        video_path: str,
        prompt: str = "Transcribe this audio completely and verbatim.",
        temperature: float = 0.3,
        max_new_tokens: int = 4096,
        repetition_penalty: float = 1.05,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
        chunk_seconds: int = 30,
        on_chunk=None,
    ) -> dict[str, object]:
        del temperature, max_new_tokens, repetition_penalty, top_p, top_k, enable_thinking

        audio_path, cleanup_path = self._prepare_audio_path(video_path)
        try:
            audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != self._local_input_sample_rate():
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self._local_input_sample_rate())
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

        samples_per_chunk = int(chunk_seconds * self._local_input_sample_rate())
        chunks = [audio[start:start + samples_per_chunk] for start in range(0, len(audio), samples_per_chunk)] or [audio]
        all_parts: list[str] = []
        for index, chunk in enumerate(chunks):
            transcript = self._bridge_audio_array_to_text(chunk, prompt=prompt)
            all_parts.append(transcript)
            accumulated = " ".join(part for part in all_parts if part)
            if on_chunk is not None:
                on_chunk(index, len(chunks), accumulated)

        return {"text": " ".join(part for part in all_parts if part), "audio": None, "audio_path": None, "sample_rate": None}

    def _bridge_audio_array_to_text(self, audio: np.ndarray, *, prompt: str | None = None) -> str:
        transcript = mm.transcribe_audio_array_with_minicpm(
            audio,
            prompt=prompt or "Transcribe this audio completely and verbatim. Return only the transcript text.",
        )
        transcript = transcript.strip()
        if not transcript:
            raise RuntimeError("MiniCPM could not transcribe the microphone input for the gemma_llamacpp profile.")
        return transcript

    def _coerce_audio_item_to_array(self, item: dict) -> np.ndarray:
        if item.get("type") == "input_audio":
            audio_info = item.get("input_audio") or {}
            encoded = audio_info.get("data")
            if not encoded:
                raise ValueError("input_audio payload is missing data")
            audio, sample_rate = sf.read(io.BytesIO(base64.b64decode(encoded)), dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != self._local_input_sample_rate():
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self._local_input_sample_rate())
            return np.ascontiguousarray(audio, dtype=np.float32)

        raise RuntimeError(f"The {self.name} profile expects in-memory audio input through the app microphone path.")