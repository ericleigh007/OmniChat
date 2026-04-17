"""Local Qwen 3.5 GGUF backend using llama.cpp multimodal CLI/server."""

from __future__ import annotations

import atexit
import base64
import io
import json
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image

from tools.model.backends.base_backend import ModelBackend
from tools.model import model_manager as mm
from tools.shared.debug_trace import get_trace_logger


_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_ASSISTANT_PREFIX_RE = re.compile(r"^assistant\s*:\s*", re.IGNORECASE)
_REASONING_ANSWER_MARKER_RE = re.compile(
    r"^(?:revised|draft|drafting(?:\s+paragraph\s+\d+)?|final answer|final output)\s*:\s*(.*)$",
    re.IGNORECASE,
)
_LEAKED_REASONING_PREFIXES = (
    "the user wants me to",
    "the user is asking",
    "i should",
    "let me",
    "first,",
    "1.  **analyze",
    "1. **analyze",
)


logger = get_trace_logger()

_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".aac"}


class QwenLlamaCppBackend(ModelBackend):
    """Adapter for owned local Qwen 3.5 GGUF inference via llama.cpp."""

    name = "qwen_llamacpp"

    def __init__(self) -> None:
        self._handle: Optional[dict[str, Any]] = None
        self._server_process: Optional[subprocess.Popen] = None
        self._runtime_status = {
            "cli_path": None,
            "server_path": None,
            "server_url": None,
            "server_ready": False,
            "model_path": None,
            "mmproj_path": None,
            "last_return_code": None,
            "last_stderr_tail": None,
            "last_prompt_chars": 0,
            "last_prompt_tokens_est": 0,
            "last_text_chars": 0,
            "last_text_tokens_est": 0,
            "last_reasoning_chars": 0,
            "reasoning_detected": False,
            "last_n_predict": 0,
            "last_server_elapsed_s": 0.0,
            "last_server_stop_reason": None,
            "last_server_json_keys": [],
        }
        atexit.register(self._shutdown_server)

    def get_runtime_status(self) -> dict[str, Any]:
        return dict(self._runtime_status)

    def _get_backend_config(self) -> dict[str, Any]:
        return mm.get_qwen_llamacpp_config()

    def _supports_native_audio_input(self) -> bool:
        return False

    def _supports_server_multimodal_chat(self) -> bool:
        return False

    def _local_input_sample_rate(self) -> int:
        return 16000

    def get_capabilities(self) -> dict[str, Any]:
        cfg = self._get_backend_config()
        speech_backend = str(cfg.get("speech_backend") or "none")
        hybrid_tts = speech_backend == "minicpm_streaming"
        supports_audio_input = self._supports_native_audio_input() or hybrid_tts
        return {
            "backend": self.name,
            "model_name": cfg.get("name") or "",
            "supports_audio_input": supports_audio_input,
            "supports_audio_output": hybrid_tts,
            "supports_streaming_text": True,
            "supports_streaming_audio": hybrid_tts,
            "supports_voice_reference": hybrid_tts,
            "supports_image_input": True,
            "supports_video_input": False,
            "input_sample_rate": self._local_input_sample_rate() if supports_audio_input else None,
            "output_sample_rate": 24000 if hybrid_tts else None,
            "transport": {
                "protocol": "local_llamacpp_server" if self._resolve_server_path(cfg).exists() else "local_llamacpp_cli",
                "streaming_mode": self._describe_streaming_mode(server_available=self._resolve_server_path(cfg).exists(), hybrid_tts=hybrid_tts),
            },
        }

    def _describe_streaming_mode(self, *, server_available: bool, hybrid_tts: bool) -> str:
        if self._supports_server_multimodal_chat():
            if server_available and hybrid_tts:
                return "persistent_server_text+multimodal_chat+minicpm_tts"
            if server_available:
                return "persistent_server_text+multimodal_chat"
            return "oneshot_cli"
        if server_available and hybrid_tts:
            return "persistent_server_text+minicpm_tts"
        if server_available:
            return "persistent_server_text"
        if hybrid_tts:
            return "oneshot_cli+minicpm_tts"
        return "oneshot_cli"

    def set_quantization(self, mode: str) -> None:
        if str(mode or "none") != "none":
            raise ValueError(f"{self.name} uses a fixed pre-quantized GGUF artifact; leave quantization='none'.")

    def set_auto_update(self, enabled: bool) -> None:
        _ = enabled

    def get_model(self):
        if self._handle is not None:
            return self._handle, None

        cfg = self._get_backend_config()
        llama_root = Path(cfg["llama_root"]).expanduser()
        cli_path = Path(cfg["cli_path"]).expanduser() if cfg.get("cli_path") else llama_root / "build" / "bin" / "llama-mtmd-cli.exe"
        server_path = self._resolve_server_path(cfg)
        model_path = Path(cfg["model_path"]).expanduser()
        mmproj_path = Path(cfg["mmproj_path"]).expanduser()

        missing = [str(path) for path in (llama_root, cli_path, model_path, mmproj_path) if not path.exists()]
        if missing:
            raise RuntimeError(f"{self.name} backend is missing required files: {', '.join(missing)}")

        self._handle = {
            "llama_root": llama_root,
            "cli_path": cli_path,
            "server_path": server_path if server_path.exists() else None,
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "n_gpu_layers": int(cfg.get("n_gpu_layers", 99)),
            "flash_attn": bool(cfg.get("flash_attn", True)),
            "context_length": int(cfg.get("context_length", 8192)),
            "use_jinja": bool(cfg.get("use_jinja", True)),
        }
        self._runtime_status.update({
            "cli_path": str(cli_path),
            "server_path": str(server_path) if server_path.exists() else None,
            "model_path": str(model_path),
            "mmproj_path": str(mmproj_path),
        })
        if server_path.exists():
            self._ensure_server_ready()
        return self._handle, None

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
        del repetition_penalty, top_p, top_k
        normalized_messages = self._normalize_messages(messages)
        if self._messages_are_text_only(normalized_messages):
            request_id = self._get_request_id(trace_context)
            started = time.perf_counter()
            speech_backend = str(self._get_backend_config().get("speech_backend") or "none")
            logger.info(
                "trace_id=%s stage=backend event=request_start streaming=False transport=%s text_only=True speech_backend=%s message_count=%d temperature=%.3f max_tokens=%d enable_thinking=%s last_user_preview=%r",
                request_id,
                f"{self.name}_server",
                speech_backend,
                len(normalized_messages),
                float(temperature),
                int(max_new_tokens),
                bool(enable_thinking),
                self._last_user_preview(normalized_messages),
            )
            prompt, system_prompt, _image_path, _temp_image = self._prepare_conversation(
                normalized_messages,
                enable_thinking=enable_thinking,
            )
            try:
                reasoning, text = self._complete_text_only_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                )
            except Exception as exc:
                logger.exception("trace_id=%s stage=backend event=error error=%s", request_id, exc)
                raise
            if not enable_thinking:
                reasoning = ""
            if generate_audio and text:
                logger.info(
                    "trace_id=%s stage=backend event=audio_handoff target=minicpm_streaming source_backend=%s text_chars=%d response_tokens_est=%d",
                    request_id,
                    self.name,
                    len(text),
                    self._estimate_token_count(text),
                )
            audio_result = self._synthesize_speech_if_needed(
                text=text,
                voice_ref=voice_ref,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                trace_context=trace_context,
                source_backend=self.name,
            )
            self._runtime_status.update({
                "last_text_chars": len(text),
                "last_text_tokens_est": self._estimate_token_count(text),
                "last_reasoning_chars": len(reasoning),
                "reasoning_detected": bool(reasoning),
            })
            logger.info(
                "trace_id=%s stage=backend event=request_finish outcome=ok elapsed_s=%.3f response_chars=%d response_tokens_est=%d reasoning_chars=%d prompt_tokens_est=%d n_predict=%d stop_reason=%s preview=%r",
                request_id,
                time.perf_counter() - started,
                len(text),
                self._runtime_status.get("last_text_tokens_est", 0),
                len(reasoning),
                self._runtime_status.get("last_prompt_tokens_est", 0),
                self._runtime_status.get("last_n_predict", 0),
                self._runtime_status.get("last_server_stop_reason") or "n/a",
                text[:160].replace("\n", " "),
            )
            return {
                "text": text,
                "reasoning": reasoning,
                "audio": audio_result.get("audio"),
                "audio_path": audio_result.get("audio_path"),
                "sample_rate": audio_result.get("sample_rate"),
            }
        prompt, system_prompt, image_path, temp_image = self._prepare_conversation(normalized_messages, enable_thinking=enable_thinking)
        try:
            if self._supports_server_multimodal_chat() and self._messages_have_native_multimodal_input(normalized_messages):
                raw_text = self._run_server_chat_completion(
                    messages=normalized_messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                )
            else:
                raw_text = self._run_cli(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image_path=image_path,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                )
        finally:
            if temp_image is not None:
                temp_image.unlink(missing_ok=True)
        reasoning, text = self._split_reasoning(raw_text)
        if not enable_thinking:
            reasoning = ""
        audio_result = self._synthesize_speech_if_needed(
            text=text,
            voice_ref=voice_ref,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
        )
        self._runtime_status.update({
            "last_text_chars": len(text),
            "last_reasoning_chars": len(reasoning),
            "reasoning_detected": bool(reasoning),
        })
        return {
            "text": text,
            "reasoning": reasoning,
            "audio": audio_result.get("audio"),
            "audio_path": audio_result.get("audio_path"),
            "sample_rate": audio_result.get("sample_rate"),
        }

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
        normalized_messages = self._normalize_messages(messages)
        if self._messages_are_text_only(normalized_messages):
            request_id = self._get_request_id(trace_context)
            started = time.perf_counter()
            speech_backend = str(self._get_backend_config().get("speech_backend") or "none")
            logger.info(
                "trace_id=%s stage=backend event=request_start streaming=True transport=%s text_only=True speech_backend=%s message_count=%d temperature=%.3f max_tokens=%d requested_audio=%s enable_thinking=%s last_user_preview=%r",
                request_id,
                f"{self.name}_server",
                speech_backend,
                len(normalized_messages),
                float(temperature),
                int(max_new_tokens),
                bool(generate_audio),
                bool(enable_thinking),
                self._last_user_preview(normalized_messages),
            )
            prompt, system_prompt, _image_path, _temp_image = self._prepare_conversation(
                normalized_messages,
                enable_thinking=enable_thinking,
            )
            reasoning = ""
            text = ""
            display_text = ""
            try:
                if not enable_thinking:
                    text = yield from self._stream_text_only_response(
                        request_id=request_id,
                        started=started,
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                    )
                else:
                    reasoning, text = self._complete_text_only_response(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking,
                    )
            except Exception as exc:
                logger.exception("trace_id=%s stage=backend event=error error=%s", request_id, exc)
                raise
            if not enable_thinking:
                reasoning = ""
            display_text = self._format_chat_display(text, reasoning)
            if display_text and not text:
                logger.info(
                    "trace_id=%s stage=backend event=first_text first_text_s=%.3f",
                    request_id,
                    time.perf_counter() - started,
                )
                yield None, display_text
            self._runtime_status.update({
                "last_text_chars": len(text),
                "last_text_tokens_est": self._estimate_token_count(text),
                "last_reasoning_chars": len(reasoning),
                "reasoning_detected": bool(reasoning),
            })
            if generate_audio and text:
                logger.info(
                    "trace_id=%s stage=backend event=audio_handoff target=minicpm_streaming source_backend=%s text_chars=%d response_tokens_est=%d",
                    request_id,
                    self.name,
                    len(text),
                    self._runtime_status.get("last_text_tokens_est", 0),
                )
                for audio_chunk, _text_chunk in mm.stream_text_to_speech_with_minicpm(
                    text,
                    voice_ref=voice_ref,
                    temperature=min(max(temperature, 0.0), 0.3),
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=max(1, min(top_k, 20)),
                    trace_context=trace_context,
                    source_backend=self.name,
                ):
                    if audio_chunk is not None:
                        yield audio_chunk, ""
            logger.info(
                "trace_id=%s stage=backend event=request_finish outcome=ok elapsed_s=%.3f response_chars=%d response_tokens_est=%d reasoning_chars=%d prompt_tokens_est=%d n_predict=%d stop_reason=%s preview=%r",
                request_id,
                time.perf_counter() - started,
                len(text),
                self._runtime_status.get("last_text_tokens_est", 0),
                len(reasoning),
                self._runtime_status.get("last_prompt_tokens_est", 0),
                self._runtime_status.get("last_n_predict", 0),
                self._runtime_status.get("last_server_stop_reason") or "n/a",
                text[:160].replace("\n", " "),
            )
            return {"final_text": display_text}

        result = self.chat(
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
            trace_context=trace_context,
        )
        display_text = self._format_chat_display(result.get("text", ""), result.get("reasoning", ""))
        if display_text:
            yield None, display_text
        if generate_audio and result.get("text"):
            for audio_chunk, _text_chunk in mm.stream_text_to_speech_with_minicpm(
                result["text"],
                voice_ref=voice_ref,
                temperature=min(max(temperature, 0.0), 0.3),
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=max(1, min(top_k, 20)),
            ):
                if audio_chunk is not None:
                    yield audio_chunk, ""
        return {"final_text": display_text}

    def _stream_text_only_response(
        self,
        *,
        request_id: str,
        started: float,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
    ):
        emitted_visible_text = ""
        raw_text = ""
        first_text_logged = False

        try:
            stream = self._run_server_completion_streaming(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=False,
            )
            while True:
                try:
                    chunk = next(stream)
                except StopIteration as stop:
                    raw_text = stop.value or ""
                    break
                if not chunk:
                    continue
                raw_text += chunk
                reasoning, visible_text = self._split_reasoning(raw_text)
                candidate_visible_text = visible_text
                if not candidate_visible_text and reasoning:
                    candidate_visible_text = self._extract_answer_from_reasoning(reasoning)
                if not candidate_visible_text.startswith(emitted_visible_text):
                    continue
                delta = candidate_visible_text[len(emitted_visible_text):]
                if not delta:
                    continue
                if not first_text_logged:
                    logger.info(
                        "trace_id=%s stage=backend event=first_text first_text_s=%.3f",
                        request_id,
                        time.perf_counter() - started,
                    )
                    first_text_logged = True
                emitted_visible_text = candidate_visible_text
                yield None, delta
        except (requests.RequestException, RuntimeError) as exc:
            logger.info(
                "trace_id=%s stage=backend event=stream_fallback reason=%r",
                request_id,
                str(exc),
            )
            self._runtime_status["last_stderr_tail"] = f"server streaming failed; retrying via blocking path ({exc})"
            reasoning, text = self._complete_text_only_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=False,
            )
            _ = reasoning
            return text

        reasoning, repaired_text = self._split_reasoning(raw_text)
        reasoning, repaired_text = self._repair_incomplete_response(reasoning, repaired_text)
        if not self._response_satisfies_contract(repaired_text, prompt=prompt):
            retry_reasoning, retry_text = self._retry_server_text_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=False,
            )
            if retry_text and self._response_satisfies_contract(retry_text, prompt=prompt):
                reasoning, repaired_text = retry_reasoning, retry_text
            else:
                self._runtime_status["last_stderr_tail"] = "server returned no acceptable final text after stream; retrying via cli"
                raw_cli_text = self._run_cli(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image_path=None,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=False,
                )
                reasoning, repaired_text = self._split_reasoning(raw_cli_text)
                reasoning, repaired_text = self._repair_incomplete_response(reasoning, repaired_text)

        if repaired_text.startswith(emitted_visible_text):
            suffix = repaired_text[len(emitted_visible_text):]
            if suffix:
                if not first_text_logged:
                    logger.info(
                        "trace_id=%s stage=backend event=first_text first_text_s=%.3f",
                        request_id,
                        time.perf_counter() - started,
                    )
                yield None, suffix

        self._runtime_status.update({
            "last_text_chars": len(repaired_text),
            "last_text_tokens_est": self._estimate_token_count(repaired_text),
            "last_reasoning_chars": len(reasoning),
            "reasoning_detected": bool(reasoning),
        })
        return repaired_text

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
        del headless
        result = self.chat(
            messages=messages,
            voice_ref=voice_ref,
            generate_audio=False,
            output_audio_path=output_audio_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )
        display_text = self._format_chat_display(result.get("text", ""), result.get("reasoning", ""))
        if display_text and on_text_chunk is not None:
            on_text_chunk(display_text)
        audio_result = self._synthesize_speech_if_needed(
            text=result.get("text", ""),
            voice_ref=voice_ref,
            generate_audio=True,
            output_audio_path=output_audio_path,
        )
        result["audio"] = audio_result.get("audio")
        result["audio_path"] = audio_result.get("audio_path")
        result["sample_rate"] = audio_result.get("sample_rate")
        return result

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
        del max_slice_nums, repetition_penalty, top_p, top_k
        if self._supports_server_multimodal_chat():
            return self.chat(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image},
                        ],
                    }
                ],
                voice_ref=voice_ref,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.05,
                top_p=0.9,
                top_k=20,
                enable_thinking=enable_thinking,
            )
        image_path, temp_image = self._coerce_image_to_path(image)
        try:
            raw_text = self._run_cli(
                prompt=prompt,
                system_prompt=self._build_system_prompt(enable_thinking=enable_thinking),
                image_path=image_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
        finally:
            if temp_image is not None:
                temp_image.unlink(missing_ok=True)
        reasoning, text = self._split_reasoning(raw_text)
        audio_result = self._synthesize_speech_if_needed(
            text=text,
            voice_ref=voice_ref,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
        )
        self._runtime_status.update({
            "last_text_chars": len(text),
            "last_text_tokens_est": self._estimate_token_count(text),
            "last_reasoning_chars": len(reasoning),
            "reasoning_detected": bool(reasoning),
        })
        return {
            "text": text,
            "reasoning": reasoning,
            "audio": audio_result.get("audio"),
            "audio_path": audio_result.get("audio_path"),
            "sample_rate": audio_result.get("sample_rate"),
        }

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
        del video_path, prompt, generate_audio, voice_ref, output_audio_path, temperature, max_new_tokens, repetition_penalty, top_p, top_k, enable_thinking, max_frames, on_progress
        raise NotImplementedError(f"{self.name} does not support video in OmniChat yet.")

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
        del video_path, prompt, generate_audio, voice_ref, temperature, max_new_tokens, repetition_penalty, top_p, top_k, enable_thinking, max_frames, chunk_seconds, on_chunk, on_progress
        raise NotImplementedError(f"{self.name} does not support chunked video in OmniChat yet.")

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
        if not self._supports_native_audio_input():
            del video_path, prompt, temperature, max_new_tokens, repetition_penalty, top_p, top_k, enable_thinking, chunk_seconds, on_chunk
            raise NotImplementedError(f"{self.name} does not support audio transcription in OmniChat yet.")

        audio_path, cleanup_path = self._prepare_audio_path(video_path)
        try:
            import soundfile as sf

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
            result = self.chat(
                messages=[{"role": "user", "content": [chunk, prompt]}],
                generate_audio=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            )
            all_parts.append((result.get("text") or "").strip())
            accumulated = " ".join(part for part in all_parts if part)
            if on_chunk is not None:
                on_chunk(index, len(chunks), accumulated)
        return {"text": " ".join(part for part in all_parts if part), "audio": None, "audio_path": None, "sample_rate": None}

    def _prepare_conversation(
        self,
        messages: list[dict],
        *,
        enable_thinking: bool,
    ) -> tuple[str, str, Optional[str], Optional[Path]]:
        prompt_lines: list[str] = []
        image_path: Optional[str] = None
        temp_image: Optional[Path] = None

        for message in messages:
            role = str(message.get("role", "user")).capitalize()
            text_parts: list[str] = []
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray, Image.Image)):
                content = [content]

            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    item_type = item.get("type", "text")
                    if item_type == "text":
                        text_parts.append(str(item.get("text", "")))
                        continue
                    if item_type in {"image", "image_url"} and image_path is None:
                        raw_image = item.get("image")
                        if raw_image is None:
                            image_url = item.get("image_url")
                            raw_image = image_url.get("url") if isinstance(image_url, dict) else image_url
                        image_path, temp_image = self._coerce_image_to_path(raw_image)
                        text_parts.append("[image attached]")
                        continue
                    if item_type in {"audio", "input_audio", "video"}:
                        text_parts.append(f"[{item_type} omitted]")
                        continue
                if isinstance(item, Image.Image) and image_path is None:
                    image_path, temp_image = self._coerce_image_to_path(item)
                    text_parts.append("[image attached]")
                elif isinstance(item, np.ndarray) and self._looks_like_image_array(item) and image_path is None:
                    image_path, temp_image = self._coerce_image_to_path(item)
                    text_parts.append("[image attached]")

            prompt_text = " ".join(part for part in text_parts if part).strip()
            if prompt_text:
                prompt_lines.append(f"{role}: {prompt_text}")

        prompt = "\n\n".join(prompt_lines).strip()
        return prompt, self._build_system_prompt(enable_thinking=enable_thinking), image_path, temp_image

    def _build_system_prompt(self, *, enable_thinking: bool) -> str:
        base_prompt = "You are OmniChat, a helpful multimodal assistant."
        if enable_thinking:
            return (
                f"{base_prompt} If you need internal reasoning, keep it inside <think></think> "
                "and provide the final answer after the closing tag."
            )
        return (
            f"{base_prompt} Answer directly with the final response only. "
            "Do not reveal internal reasoning. Do not output <think> tags, thinking process, analysis, "
            "drafts, outlines, or self-corrections. Start immediately with the answer."
        )

    def _coerce_image_to_path(self, image: Any) -> tuple[str, Optional[Path]]:
        if isinstance(image, str):
            return str(Path(image).expanduser()), None

        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(array)
        else:
            raise NotImplementedError(f"Unsupported {self.name} image value type {type(image).__name__}.")

        temp_image = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
        pil_image.save(temp_image)
        return str(temp_image), temp_image

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        cfg = self._get_backend_config()
        allow_audio_bridge = (cfg.get("speech_backend") or "none") == "minicpm_streaming"
        normalized_messages: list[dict] = []

        for message in messages:
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray, Image.Image)):
                content = [content]

            normalized_content: list[Any] = []
            for item in content:
                if isinstance(item, np.ndarray) and not self._looks_like_image_array(item):
                    if self._supports_native_audio_input():
                        normalized_content.append(self._audio_array_to_input_audio_part(item))
                        continue
                    if not allow_audio_bridge:
                        raise RuntimeError(
                            f"The active {self.name} profile does not support microphone audio input. "
                            "Use a profile with native audio input or a MiniCPM-backed bridge profile."
                        )
                    transcript = mm.transcribe_audio_array_with_minicpm(item)
                    if not transcript:
                        raise RuntimeError(f"MiniCPM could not transcribe the microphone input for the {self.name} bridge profile.")
                    normalized_content.append(transcript)
                    continue

                if isinstance(item, dict) and item.get("type") in {"audio", "input_audio"}:
                    if self._supports_native_audio_input() and item.get("type") == "input_audio":
                        normalized_content.append(item)
                        continue
                    raise RuntimeError(f"The {self.name} profile expects in-memory audio input through the app microphone path.")

                normalized_content.append(item)

            normalized_messages.append({
                "role": message.get("role", "user"),
                "content": normalized_content,
            })

        return normalized_messages

    def _looks_like_image_array(self, value: np.ndarray) -> bool:
        array = np.asarray(value)
        if array.ndim == 2:
            return True
        if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
            return True
        return False

    def _run_cli(
        self,
        *,
        prompt: str,
        system_prompt: str,
        image_path: Optional[str],
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> str:
        handle, _ = self.get_model()
        think_prefix = "/think" if enable_thinking else "/no_think"
        final_prompt = prompt.strip()
        if not final_prompt:
            final_prompt = "Describe the provided input."
        final_prompt = f"System: {system_prompt}\n\n{final_prompt}"
        if not final_prompt.lstrip().startswith(("/think", "/no_think")):
            final_prompt = f"{think_prefix}\n{final_prompt}"
        cmd = [
            str(handle["cli_path"]),
            "-m", str(handle["model_path"]),
            "--mmproj", str(handle["mmproj_path"]),
            "-ngl", str(handle["n_gpu_layers"]),
            "--temp", str(temperature),
            "-n", str(max_new_tokens),
            "-c", str(handle["context_length"]),
        ]
        if handle["flash_attn"]:
            cmd.extend(["--flash-attn", "on"])
        if handle["use_jinja"]:
            cmd.append("--jinja")
        if image_path:
            cmd.extend(["--image", image_path])
        cmd.extend(["-p", final_prompt])

        timeout_s = float(self._get_backend_config().get("timeout_s", 45.0) or 45.0)

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(handle["llama_root"]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"llama-mtmd-cli timed out after {timeout_s:.1f}s") from exc

        stderr_tail = "\n".join((completed.stderr or "").splitlines()[-20:])
        prompt_tokens_est = self._estimate_token_count(final_prompt)
        text = (completed.stdout or "").strip()
        self._runtime_status.update({
            "last_return_code": completed.returncode,
            "last_stderr_tail": stderr_tail,
            "last_prompt_chars": len(final_prompt),
            "last_prompt_tokens_est": prompt_tokens_est,
            "last_text_chars": len(text),
            "last_text_tokens_est": self._estimate_token_count(text),
            "last_n_predict": int(max_new_tokens),
            "last_server_elapsed_s": 0.0,
            "last_server_stop_reason": "cli",
            "last_server_json_keys": [],
        })
        if completed.returncode != 0:
            raise RuntimeError(f"llama-mtmd-cli failed with exit code {completed.returncode}: {stderr_tail}")

        if not text:
            raise RuntimeError("llama-mtmd-cli returned no stdout text.")
        return text

    def _complete_text_only_response(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> tuple[str, str]:
        try:
            raw_text = self._run_server_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
        except (requests.RequestException, RuntimeError) as exc:
            self._runtime_status["last_stderr_tail"] = f"server completion failed; restarting server ({exc})"
            self._shutdown_server()
            try:
                raw_text = self._run_server_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                )
                self._runtime_status["last_stderr_tail"] = "server completion recovered after restart"
            except (requests.RequestException, RuntimeError) as restart_exc:
                self._runtime_status["last_stderr_tail"] = f"server completion failed after restart; retrying via cli ({restart_exc})"
                raw_text = self._run_cli(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    image_path=None,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    enable_thinking=enable_thinking,
                )
        reasoning, text = self._split_reasoning(raw_text)
        reasoning, text = self._repair_incomplete_response(reasoning, text)
        if not enable_thinking and not self._response_satisfies_contract(text, prompt=prompt):
            retry_reasoning, retry_text = self._retry_server_text_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
            )
            if retry_text and self._response_satisfies_contract(retry_text, prompt=prompt):
                reasoning, text = retry_reasoning, retry_text
        if enable_thinking or self._response_satisfies_contract(text, prompt=prompt):
            return reasoning, text

        self._runtime_status["last_stderr_tail"] = "server returned no acceptable final text; retrying via cli"
        raw_text = self._run_cli(
            prompt=prompt,
            system_prompt=system_prompt,
            image_path=None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )
        reasoning, text = self._split_reasoning(raw_text)
        reasoning, text = self._repair_incomplete_response(reasoning, text)
        if self._response_satisfies_contract(text, prompt=prompt):
            return reasoning, text

        raise RuntimeError(f"{self.name} returned no acceptable final text for this prompt.")

    def _retry_server_text_completion(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> tuple[str, str]:
        retry_temperature = min(float(temperature), 0.2)
        retry_max_new_tokens = max(int(max_new_tokens), 384)
        self._runtime_status["last_stderr_tail"] = "server returned truncated final text; retrying via server"
        try:
            raw_text = self._run_server_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=retry_temperature,
                max_new_tokens=retry_max_new_tokens,
                enable_thinking=enable_thinking,
            )
        except (requests.RequestException, RuntimeError) as exc:
            self._runtime_status["last_stderr_tail"] = f"server retry failed after truncated final text; restarting server ({exc})"
            self._shutdown_server()
            try:
                raw_text = self._run_server_completion(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=retry_temperature,
                    max_new_tokens=retry_max_new_tokens,
                    enable_thinking=enable_thinking,
                )
            except (requests.RequestException, RuntimeError) as restart_exc:
                self._runtime_status["last_stderr_tail"] = f"server retry failed after restart ({restart_exc})"
                return "", ""

        reasoning, text = self._split_reasoning(raw_text)
        return self._repair_incomplete_response(reasoning, text)

    def _split_reasoning(self, text: str) -> tuple[str, str]:
        stripped_text = self._strip_assistant_prefix(text or "")
        matches = list(_THINK_BLOCK_RE.finditer(stripped_text))
        if not matches:
            lowered = stripped_text.lower()
            if "<think>" in lowered and "</think>" not in lowered:
                return self._extract_incomplete_think_reasoning(stripped_text), ""
            incomplete_reasoning = self._extract_incomplete_think_reasoning(stripped_text)
            if incomplete_reasoning:
                return incomplete_reasoning, ""
            stripped = stripped_text.strip()
            if self._looks_like_leaked_reasoning(stripped):
                return stripped, ""
            return "", stripped

        reasoning_parts = [match.group(1).strip() for match in matches if match.group(1).strip()]
        final_text = _THINK_BLOCK_RE.sub("", stripped_text).strip()
        return "\n\n".join(reasoning_parts).strip(), final_text

    def _repair_incomplete_response(self, reasoning: str, text: str) -> tuple[str, str]:
        final_text = (text or "").strip()
        reasoning_text = (reasoning or "").strip()
        if not reasoning_text:
            return reasoning, text

        recovered = self._extract_answer_from_reasoning(reasoning_text)
        if not final_text:
            if recovered:
                self._runtime_status["last_stderr_tail"] = "server returned no final text; recovered answer from leaked reasoning"
                return reasoning, recovered
            return reasoning, text
        if not self._looks_truncated_response(final_text):
            return reasoning, text

        if not recovered or len(recovered) <= len(final_text):
            return reasoning, text

        self._runtime_status["last_stderr_tail"] = "server returned truncated final text; recovered answer from leaked reasoning"
        return reasoning, recovered

    def _extract_answer_from_reasoning(self, reasoning: str) -> str:
        candidates: list[str] = []
        lines = (reasoning or "").splitlines()
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            match = _REASONING_ANSWER_MARKER_RE.match(line.strip("* "))
            if not match:
                index += 1
                continue

            collected = [match.group(1).strip()] if match.group(1).strip() else []
            index += 1
            while index < len(lines):
                continuation = lines[index].strip()
                if not continuation:
                    break
                if _REASONING_ANSWER_MARKER_RE.match(continuation.strip("* ")):
                    break
                if re.match(r"^\d+\.\s", continuation):
                    break
                if continuation.startswith(("* ", "- ")):
                    break
                collected.append(continuation)
                index += 1

            candidate = " ".join(part for part in collected if part).strip().strip('"')
            if candidate:
                candidates.append(candidate)
            continue
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]
        return "\n\n".join(candidates)

    def _looks_truncated_response(self, text: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        words = normalized.split()
        if len(words) < 3:
            return normalized[-1].isalnum() and not normalized.endswith((".", "!", "?", '"', "'", ")", "]"))
        if len(words) < 8 and len(normalized) <= 64:
            return normalized[-1].isalnum() and not normalized.endswith((".", "!", "?", '"', "'", ")", "]"))
        if normalized.endswith((".", "!", "?", '"', "'", ")", "]")):
            return False
        trailing_word = words[-1].strip(".,!?;:'\"()[]{}").lower()
        if trailing_word in {"a", "an", "and", "as", "at", "because", "but", "by", "due", "for", "from", "if", "in", "into", "is", "of", "or", "than", "that", "the", "these", "this", "to", "when", "with"}:
            return True
        return normalized[-1].isalnum()

    def _response_satisfies_contract(self, text: str, *, prompt: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        if self._looks_truncated_response(normalized):
            return False

        if self._requires_long_form_response(prompt):
            if len(normalized) < 900:
                return False
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
            if len(paragraphs) < 3:
                return False

        return True

    def _requires_long_form_response(self, prompt: str) -> bool:
        normalized = (prompt or "").lower()
        return any(
            phrase in normalized
            for phrase in (
                "detailed manner",
                "at least three paragraphs",
                "three paragraphs",
                "do not stop mid-sentence",
                "long response",
            )
        )

    def _strip_assistant_prefix(self, text: str) -> str:
        stripped = (text or "").lstrip()
        return _ASSISTANT_PREFIX_RE.sub("", stripped, count=1)

    def _extract_incomplete_think_reasoning(self, text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        start = lowered.find("<think>")
        if start < 0:
            return ""
        if "</think>" in lowered[start:]:
            return ""
        return normalized[start + len("<think>"):].strip()

    def _looks_like_leaked_reasoning(self, text: str) -> bool:
        normalized = (text or "").strip().lower()
        if not normalized:
            return False
        if normalized.startswith(_LEAKED_REASONING_PREFIXES):
            return True
        if "**analyze the image" in normalized or "**analyze the prompt" in normalized:
            return True
        if normalized.startswith("the user") and "final answer:" not in normalized:
            return True
        return False

    def _format_chat_display(self, text: str, reasoning: str) -> str:
        final_text = (text or "").strip()
        reasoning_text = (reasoning or "").strip()
        if not reasoning_text:
            return final_text

        parts = [f"**Thinking**\n\n{reasoning_text}"]
        if final_text:
            parts.append(f"**Response**\n\n{final_text}")
        return "\n\n".join(parts)

    def _messages_have_native_multimodal_input(self, normalized_messages: list[dict]) -> bool:
        for message in normalized_messages:
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray, Image.Image)):
                content = [content]
            for item in content:
                if isinstance(item, dict) and item.get("type") in {"image", "image_url", "input_audio"}:
                    return True
                if isinstance(item, Image.Image):
                    return True
                if isinstance(item, np.ndarray) and (self._looks_like_image_array(item) or self._supports_native_audio_input()):
                    return True
        return False

    def _run_server_chat_completion(
        self,
        *,
        messages: list[dict],
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> str:
        del enable_thinking
        server_url = self._ensure_server_ready()
        timeout_s = float(self._get_backend_config().get("timeout_s", 45.0) or 45.0)
        payload = {
            "messages": self._build_openai_messages(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "modalities": ["text"],
            "stream": False,
        }
        started = time.perf_counter()
        response = requests.post(
            server_url.rstrip("/") + "/chat/completions",
            json=payload,
            timeout=(10.0, timeout_s),
        )
        response.raise_for_status()
        response_json = response.json()
        elapsed_s = time.perf_counter() - started
        choices = response_json.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = self._extract_text_from_content(message.get("content"))
        usage = response_json.get("usage") if isinstance(response_json.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or self._estimate_token_count(json.dumps(payload, ensure_ascii=False)))
        completion_tokens = int(usage.get("completion_tokens") or self._estimate_token_count(content))
        stop_reason = str((choices[0].get("finish_reason") if choices else None) or "chat")
        self._runtime_status.update({
            "last_prompt_chars": len(json.dumps(payload, ensure_ascii=False)),
            "last_prompt_tokens_est": prompt_tokens,
            "last_text_chars": len(content),
            "last_text_tokens_est": completion_tokens,
            "last_n_predict": int(max_new_tokens),
            "last_server_elapsed_s": elapsed_s,
            "last_server_stop_reason": stop_reason,
            "last_server_json_keys": sorted(response_json.keys()),
        })
        return content

    def _build_openai_messages(self, messages: list[dict]) -> list[dict[str, Any]]:
        rendered: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray, Image.Image)):
                content = [content]

            parts: list[dict[str, Any]] = []
            for item in content:
                parts.extend(self._content_item_to_openai_parts(item))
            rendered.append({
                "role": str(message.get("role") or "user"),
                "content": parts or [{"type": "text", "text": ""}],
            })
        return rendered

    def _content_item_to_openai_parts(self, item: Any) -> list[dict[str, Any]]:
        if isinstance(item, str):
            return [{"type": "text", "text": item}]
        if isinstance(item, Image.Image):
            return [{"type": "image_url", "image_url": {"url": self._image_to_data_url(item)}}]
        if isinstance(item, np.ndarray):
            if self._looks_like_image_array(item):
                return [{"type": "image_url", "image_url": {"url": self._image_to_data_url(item)}}]
            if self._supports_native_audio_input():
                return [self._audio_array_to_input_audio_part(item)]
            return [{"type": "text", "text": "[audio omitted]"}]
        if isinstance(item, dict):
            item_type = item.get("type", "text")
            if item_type == "text":
                return [{"type": "text", "text": str(item.get("text", ""))}]
            if item_type == "image":
                return [{"type": "image_url", "image_url": {"url": self._image_to_data_url(item.get("image"))}}]
            if item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    return [{"type": "image_url", "image_url": image_url}]
                return [{"type": "image_url", "image_url": {"url": str(image_url or "")}}]
            if item_type == "input_audio":
                return [item]
        return []

    def _audio_array_to_input_audio_part(self, audio: np.ndarray) -> dict[str, Any]:
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), self._local_input_sample_rate(), format="WAV", subtype="PCM_16")
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "format": "wav",
            },
        }

    def _image_to_data_url(self, image: Any) -> str:
        if isinstance(image, str):
            source = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            source = Image.fromarray(array)
        elif isinstance(image, Image.Image):
            source = image
        else:
            raise NotImplementedError(f"Unsupported {self.name} image value type {type(image).__name__}.")

        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _extract_text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        return str(content)

    def _prepare_audio_path(self, media_path: str) -> tuple[str, Optional[Path]]:
        path = Path(media_path)
        if path.suffix.lower() in _AUDIO_EXTENSIONS:
            return str(path), None

        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = Path(temp_audio.name)
        temp_audio.close()
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(self._local_input_sample_rate()), str(temp_audio_path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            temp_audio_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to extract audio for {self.name} transcription: {exc.stderr.decode(errors='replace')}"
            ) from exc
        return str(temp_audio_path), temp_audio_path

    def _synthesize_speech_if_needed(
        self,
        *,
        text: str,
        voice_ref: Optional[np.ndarray],
        generate_audio: bool,
        output_audio_path: Optional[str],
        trace_context: Optional[dict] = None,
        source_backend: Optional[str] = None,
    ) -> dict[str, Any]:
        cfg = self._get_backend_config()
        if not generate_audio or (cfg.get("speech_backend") or "none") != "minicpm_streaming":
            return {"audio": None, "audio_path": output_audio_path, "sample_rate": None}
        return mm.synthesize_text_with_minicpm(
            text,
            voice_ref=voice_ref,
            output_audio_path=output_audio_path,
            trace_context=trace_context,
            source_backend=source_backend,
        )

    def _resolve_server_path(self, cfg: dict[str, Any]) -> Path:
        llama_root = Path(
            cfg.get("llama_root")
            or (Path(tempfile.gettempdir()) / "llama-cpp-qwen35-test" / "llama.cpp")
        ).expanduser()
        return llama_root / "build" / "bin" / "llama-server.exe"

    def _reserve_server_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _shutdown_server(self) -> None:
        process = self._server_process
        self._server_process = None
        if self._handle is not None:
            self._handle["server_url"] = None
        self._runtime_status["server_url"] = None
        self._runtime_status["server_ready"] = False
        if process is None:
            return
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _ensure_server_ready(self) -> str:
        handle, _ = self.get_model() if self._handle is None else (self._handle, None)
        existing_url = handle.get("server_url")
        if existing_url and self._server_process is not None and self._server_process.poll() is None:
            return str(existing_url)

        server_path = handle.get("server_path")
        if server_path is None:
            raise RuntimeError(f"llama-server.exe is not available for the {self.name} backend.")

        last_error = "server did not report healthy status"
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        for attempt in range(1, 4):
            port = self._reserve_server_port()
            server_url = f"http://127.0.0.1:{port}/v1"
            cmd = [
                str(server_path),
                "-m", str(handle["model_path"]),
                "--host", "127.0.0.1",
                "--port", str(port),
                "-ngl", str(handle["n_gpu_layers"]),
                "-c", str(handle["context_length"]),
            ]
            if handle["flash_attn"]:
                cmd.extend(["--flash-attn", "on"])

            self._server_process = subprocess.Popen(
                cmd,
                cwd=str(handle["llama_root"]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            handle["server_url"] = server_url
            self._runtime_status.update({
                "server_url": server_url,
                "server_ready": False,
            })

            deadline = time.perf_counter() + 240.0
            health_url = server_url.rstrip("/") + "/health"
            models_url = server_url.rstrip("/") + "/models"
            while time.perf_counter() < deadline:
                if self._server_process.poll() is not None:
                    last_error = f"llama-server exited before becoming ready (attempt {attempt}/3)"
                    break
                try:
                    health = requests.get(health_url, timeout=2.0)
                    if health.status_code < 500:
                        self._runtime_status["server_ready"] = True
                        return server_url
                except requests.RequestException as exc:
                    last_error = str(exc)
                try:
                    models = requests.get(models_url, timeout=2.0)
                    if models.status_code == 200:
                        self._runtime_status["server_ready"] = True
                        return server_url
                except requests.RequestException as exc:
                    last_error = str(exc)
                time.sleep(0.25)

            self._shutdown_server()
            if attempt < 3:
                time.sleep(1.0)

        raise RuntimeError(f"Timed out waiting for llama-server to start: {last_error}")

    def _messages_are_text_only(self, normalized_messages: list[dict]) -> bool:
        for message in normalized_messages:
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray, Image.Image)):
                content = [content]
            for item in content:
                if isinstance(item, str):
                    continue
                if isinstance(item, dict) and item.get("type", "text") == "text":
                    continue
                return False
        return True

    def _run_server_completion(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ) -> str:
        server_url = self._ensure_server_ready()
        base_url = server_url[:-3] if server_url.endswith("/v1") else server_url
        think_prefix = "/think" if enable_thinking else "/no_think"
        final_prompt = prompt.strip() or "Describe the provided input."
        final_prompt = f"System: {system_prompt}\n\n{final_prompt}"
        if not final_prompt.lstrip().startswith(("/think", "/no_think")):
            final_prompt = f"{think_prefix}\n{final_prompt}"
        final_prompt = final_prompt.rstrip() + "\n\nAssistant:"
        no_think_budget = 1024
        if not enable_thinking and self._requires_long_form_response(prompt):
            no_think_budget = 1536
        payload = {
            "prompt": final_prompt,
            "temperature": float(temperature),
            "n_predict": int(max(max_new_tokens, no_think_budget if not enable_thinking else max_new_tokens * 2)),
            "stop": ["\nUser:", "\nSystem:"],
        }
        timeout_s = float(self._get_backend_config().get("timeout_s", 45.0) or 45.0)
        started = time.perf_counter()
        response = requests.post(
            base_url.rstrip("/") + "/completion",
            json=payload,
            timeout=(10.0, timeout_s),
        )
        response.raise_for_status()
        response_json = response.json()
        elapsed_s = time.perf_counter() - started
        content = str(response_json.get("content", "") or response_json.get("response", "") or "")
        prompt_tokens = self._extract_response_int(
            response_json,
            "prompt_tokens",
            "tokens_evaluated",
            "prompt_eval_count",
            fallback=self._estimate_token_count(final_prompt),
        )
        completion_tokens = self._extract_response_int(
            response_json,
            "completion_tokens",
            "tokens_predicted",
            "eval_count",
            fallback=self._estimate_token_count(content),
        )
        stop_reason = str(
            response_json.get("stop_type")
            or response_json.get("finish_reason")
            or response_json.get("stop_reason")
            or "server"
        )
        self._runtime_status.update({
            "last_prompt_chars": len(final_prompt),
            "last_prompt_tokens_est": prompt_tokens,
            "last_text_chars": len(content),
            "last_text_tokens_est": completion_tokens,
            "last_n_predict": int(payload["n_predict"]),
            "last_server_elapsed_s": elapsed_s,
            "last_server_stop_reason": stop_reason,
            "last_server_json_keys": sorted(response_json.keys()),
        })
        logger.info(
            "trace_id=n/a stage=backend event=server_completion_response elapsed_s=%.3f http_status=%d prompt_chars=%d prompt_tokens_est=%d completion_chars=%d completion_tokens_est=%d n_predict=%d stop_reason=%s response_keys=%r",
            elapsed_s,
            response.status_code,
            len(final_prompt),
            prompt_tokens,
            len(content),
            completion_tokens,
            int(payload["n_predict"]),
            stop_reason,
            sorted(response_json.keys()),
        )
        return content

    def _run_server_completion_streaming(
        self,
        *,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_new_tokens: int,
        enable_thinking: bool,
    ):
        server_url = self._ensure_server_ready()
        base_url = server_url[:-3] if server_url.endswith("/v1") else server_url
        think_prefix = "/think" if enable_thinking else "/no_think"
        final_prompt = prompt.strip() or "Describe the provided input."
        final_prompt = f"System: {system_prompt}\n\n{final_prompt}"
        if not final_prompt.lstrip().startswith(("/think", "/no_think")):
            final_prompt = f"{think_prefix}\n{final_prompt}"
        final_prompt = final_prompt.rstrip() + "\n\nAssistant:"
        no_think_budget = 1024
        if not enable_thinking and self._requires_long_form_response(prompt):
            no_think_budget = 1536
        payload = {
            "prompt": final_prompt,
            "temperature": float(temperature),
            "n_predict": int(max(max_new_tokens, no_think_budget if not enable_thinking else max_new_tokens * 2)),
            "stop": ["\nUser:", "\nSystem:"],
            "stream": True,
        }
        timeout_s = float(self._get_backend_config().get("timeout_s", 45.0) or 45.0)
        started = time.perf_counter()
        full_content = ""
        last_chunk: dict[str, Any] = {}

        with requests.post(
            base_url.rstrip("/") + "/completion",
            json=payload,
            timeout=(10.0, timeout_s),
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                last_chunk = chunk
                delta = self._extract_stream_delta(chunk, full_content)
                if delta:
                    full_content += delta
                    yield delta

        elapsed_s = time.perf_counter() - started
        prompt_tokens = self._extract_response_int(
            last_chunk,
            "prompt_tokens",
            "tokens_evaluated",
            "prompt_eval_count",
            fallback=self._estimate_token_count(final_prompt),
        )
        completion_tokens = self._extract_response_int(
            last_chunk,
            "completion_tokens",
            "tokens_predicted",
            "eval_count",
            fallback=self._estimate_token_count(full_content),
        )
        stop_reason = str(
            last_chunk.get("stop_type")
            or last_chunk.get("finish_reason")
            or last_chunk.get("stop_reason")
            or "stream"
        )
        self._runtime_status.update({
            "last_prompt_chars": len(final_prompt),
            "last_prompt_tokens_est": prompt_tokens,
            "last_text_chars": len(full_content),
            "last_text_tokens_est": completion_tokens,
            "last_n_predict": int(payload["n_predict"]),
            "last_server_elapsed_s": elapsed_s,
            "last_server_stop_reason": stop_reason,
            "last_server_json_keys": sorted(last_chunk.keys()),
        })
        logger.info(
            "trace_id=n/a stage=backend event=server_stream_response elapsed_s=%.3f prompt_chars=%d prompt_tokens_est=%d completion_chars=%d completion_tokens_est=%d n_predict=%d stop_reason=%s response_keys=%r",
            elapsed_s,
            len(final_prompt),
            prompt_tokens,
            len(full_content),
            completion_tokens,
            int(payload["n_predict"]),
            stop_reason,
            sorted(last_chunk.keys()),
        )
        return full_content

    def _extract_stream_delta(self, chunk: dict[str, Any], current_text: str) -> str:
        raw = chunk.get("content", "") or chunk.get("response", "") or ""
        if not raw:
            return ""
        text = str(raw)
        if current_text and text.startswith(current_text):
            return text[len(current_text):]
        return text

    def _estimate_token_count(self, text: str) -> int:
        normalized = (text or "").strip()
        if not normalized:
            return 0
        return max(1, round(len(normalized) / 4.0))

    def _extract_response_int(self, response_json: dict[str, Any], *keys: str, fallback: int = 0) -> int:
        for key in keys:
            value = response_json.get(key)
            if isinstance(value, (int, float)):
                return int(value)
        usage = response_json.get("usage")
        if isinstance(usage, dict):
            for key in keys:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    return int(value)
        return int(fallback)

    def _get_request_id(self, trace_context: Optional[dict]) -> str:
        if isinstance(trace_context, dict):
            return str(trace_context.get("request_id") or "n/a")
        return "n/a"

    def _last_user_preview(self, normalized_messages: list[dict]) -> str:
        for message in reversed(normalized_messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", [])
            if isinstance(content, (str, dict, np.ndarray, Image.Image)):
                content = [content]
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type", "text") == "text":
                    parts.append(str(item.get("text", "")))
            preview = " ".join(part.strip() for part in parts if str(part).strip())
            if preview:
                return preview[:160]
        return ""
