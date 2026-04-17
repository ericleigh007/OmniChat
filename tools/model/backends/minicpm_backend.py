"""MiniCPM backend adapter for the existing OmniChat implementation."""

from typing import Any, Optional

import numpy as np

from tools.model.backends.base_backend import ModelBackend
from tools.model import model_manager as mm


class MiniCPMBackend(ModelBackend):
    """Adapter that exposes the current MiniCPM implementation via the backend interface."""

    name = "minicpm"

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "model_name": mm.MODEL_NAME,
            "supports_audio_input": True,
            "supports_audio_output": True,
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
            "supports_voice_reference": True,
            "input_sample_rate": 16000,
            "output_sample_rate": mm._STREAMING_SR,
        }

    def set_quantization(self, mode: str) -> None:
        mm._set_minicpm_quantization(mode)

    def set_auto_update(self, enabled: bool) -> None:
        mm._set_minicpm_auto_update(enabled)

    def get_model(self):
        return mm._get_minicpm_model()

    def chat(
        self,
        messages: list[dict],
        voice_ref: Optional[np.ndarray] = None,
        generate_audio: bool = True,
        output_audio_path: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.2,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
        trace_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        _ = trace_context
        return mm._minicpm_chat(
            messages=messages,
            voice_ref=voice_ref,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    def chat_streaming(
        self,
        messages: list[dict],
        voice_ref: Optional[np.ndarray] = None,
        generate_audio: bool = True,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.2,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
        trace_context: Optional[dict] = None,
    ):
        _ = trace_context
        if not generate_audio:
            result = mm._minicpm_chat(
                messages=messages,
                voice_ref=voice_ref,
                generate_audio=False,
                output_audio_path=None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            text = str((result or {}).get("text") or "")
            if text:
                yield None, text
            return
        yield from mm._minicpm_chat_streaming(
            messages=messages,
            voice_ref=voice_ref,
            generate_audio=generate_audio,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    def chat_streaming_with_playback(
        self,
        messages: list[dict],
        voice_ref: Optional[np.ndarray] = None,
        output_audio_path: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.2,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
        headless: bool = False,
        on_text_chunk=None,
    ) -> dict[str, Any]:
        return mm._minicpm_chat_streaming_with_playback(
            messages=messages,
            voice_ref=voice_ref,
            output_audio_path=output_audio_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            headless=headless,
            on_text_chunk=on_text_chunk,
        )

    def process_image(
        self,
        image,
        prompt: str = "Describe this image in detail.",
        generate_audio: bool = False,
        voice_ref: Optional[np.ndarray] = None,
        output_audio_path: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        max_slice_nums: int = 1,
        repetition_penalty: float = 1.2,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        return mm._minicpm_process_image(
            image=image,
            prompt=prompt,
            generate_audio=generate_audio,
            voice_ref=voice_ref,
            output_audio_path=output_audio_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_slice_nums=max_slice_nums,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    def process_video(
        self,
        video_path: str,
        prompt: str = "Describe what's happening in this video.",
        generate_audio: bool = False,
        voice_ref: Optional[np.ndarray] = None,
        output_audio_path: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.2,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
        max_frames: int = 64,
        on_progress=None,
    ) -> dict[str, Any]:
        return mm._minicpm_process_video(
            video_path=video_path,
            prompt=prompt,
            generate_audio=generate_audio,
            voice_ref=voice_ref,
            output_audio_path=output_audio_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            max_frames=max_frames,
            on_progress=on_progress,
        )

    def process_video_chunked(
        self,
        video_path: str,
        prompt: str = "Describe what's happening in this video.",
        generate_audio: bool = False,
        voice_ref: Optional[np.ndarray] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.2,
        top_p: float = 0.8,
        top_k: int = 100,
        enable_thinking: bool = False,
        max_frames: int = 64,
        chunk_seconds: int = 30,
        on_chunk=None,
        on_progress=None,
    ) -> dict[str, Any]:
        return mm._minicpm_process_video_chunked(
            video_path=video_path,
            prompt=prompt,
            generate_audio=generate_audio,
            voice_ref=voice_ref,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            max_frames=max_frames,
            chunk_seconds=chunk_seconds,
            on_chunk=on_chunk,
            on_progress=on_progress,
        )

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
    ) -> dict[str, Any]:
        return mm._minicpm_transcribe_audio(
            video_path=video_path,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            chunk_seconds=chunk_seconds,
            on_chunk=on_chunk,
        )
