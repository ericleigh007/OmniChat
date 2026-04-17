"""Remote Qwen backend using an OpenAI-compatible HTTP API."""

import base64
import io
import logging
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from tools.model.backends.base_backend import ModelBackend
from tools.model.clients import QwenOmniClient, QwenOmniClientConfig
from tools.model import model_manager as mm
from tools.shared.debug_trace import get_trace_logger


_REMOTE_SPEECH_SR = 24000
_REMOTE_DEFAULT_VOICE = "chelsie"
_INLINE_AUDIO_RESPONSE_PROMPT = "Please listen to the provided audio and answer the user's spoken request."
logger = logging.getLogger(__name__)
trace_logger = get_trace_logger()


@dataclass
class _RemoteHandle:
    client: QwenOmniClient
    model_name: str


class QwenRemoteBackend(ModelBackend):
    """Adapter for a remote Qwen-compatible OpenAI-style chat service."""

    name = "qwen_remote"

    def __init__(self) -> None:
        self._handle: Optional[_RemoteHandle] = None
        self._supports_audio_transcriptions: Optional[bool] = None
        self._runtime_status = {
            "last_audio_delivery_mode": "none",
            "last_text_chars": 0,
            "last_stream_audio_chunks": 0,
            "last_stream_audio_bytes": 0,
        }

    def get_capabilities(self) -> dict[str, Any]:
        cfg = mm.get_qwen_remote_config()
        transport = QwenOmniClient(QwenOmniClientConfig(**cfg)).get_transport_contract()
        return {
            "backend": self.name,
            "model_name": cfg.get("model_name") or "",
            "supports_audio_input": True,
            "supports_audio_output": True,
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
            "supports_voice_reference": False,
            "supports_image_input": True,
            "supports_video_input": False,
            "input_sample_rate": 16000,
            "output_sample_rate": _REMOTE_SPEECH_SR,
            "transport": transport,
        }

    def get_runtime_status(self) -> dict[str, Any]:
        return dict(self._runtime_status)

    def set_quantization(self, mode: str) -> None:
        if mode not in ("none", "int8", "int4"):
            raise ValueError(f"Unknown quantization mode: {mode!r}")

    def set_auto_update(self, enabled: bool) -> None:
        _ = enabled

    def get_model(self):
        if self._handle is not None:
            return self._handle, None

        cfg = mm.get_qwen_remote_config()
        client = QwenOmniClient(QwenOmniClientConfig(**cfg))

        try:
            model_name = client.resolve_model_name()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to reach qwen_remote backend at {cfg['base_url']!r}: {exc}"
            ) from exc

        if not model_name:
            raise RuntimeError(
                "qwen_remote backend needs a configured model_name or a working /models endpoint."
            )

        self._handle = _RemoteHandle(client=client, model_name=model_name)
        return self._handle, None

    def warmup(self) -> dict[str, Any]:
        self.get_model()
        trace_context = {
            "request_id": f"warmup-{uuid4().hex[:8]}",
            "source": "qwen_remote_warmup",
        }
        messages = [{"role": "user", "content": ["Reply with exactly: OK"]}]
        started_at = time.perf_counter()
        first_text_s = None
        parts: list[str] = []

        for _audio_chunk, text_chunk in self.chat_streaming(
            messages=messages,
            generate_audio=False,
            temperature=0.0,
            max_new_tokens=8,
            repetition_penalty=1.0,
            top_p=1.0,
            top_k=1,
            enable_thinking=False,
            trace_context=trace_context,
        ):
            if text_chunk:
                parts.append(text_chunk)
                if first_text_s is None:
                    first_text_s = time.perf_counter() - started_at

        text = "".join(parts).strip()
        elapsed_s = time.perf_counter() - started_at
        result = {
            "ok": bool(text),
            "backend": self.name,
            "first_text_s": None if first_text_s is None else round(first_text_s, 3),
            "elapsed_s": round(elapsed_s, 3),
            "text": text,
        }
        logger.info(
            "qwen_remote warmup ok=%s first_text_s=%s elapsed_s=%.3f text=%r",
            result["ok"],
            "none" if first_text_s is None else f"{first_text_s:.3f}",
            elapsed_s,
            text,
        )
        return result

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
        del voice_ref
        payload = self._build_payload(
            messages=messages,
            modalities=["text"],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            stream=False,
        )
        self._log_request_start(
            trace_context,
            payload,
            streaming=False,
            cutoff_on_text_stop=False,
            requested_audio=generate_audio,
        )
        started_at = time.perf_counter()
        response = self._post_json(payload)
        text = self._extract_text_from_content(response["choices"][0]["message"].get("content"))
        self._log_request_finish(
            trace_context,
            started_at,
            text=text,
            first_text_s=None,
            text_finish_s=None,
            outcome="ok",
            cutoff_reason="blocking_chat",
            stream_audio_chunks=0,
            stream_audio_bytes=0,
        )
        result = {"text": text, "audio": None, "audio_path": output_audio_path, "sample_rate": None}
        self._record_audio_delivery(mode="none", text=text, stream_audio_chunks=0, stream_audio_bytes=0)
        if generate_audio and text:
            audio, sample_rate = self._synthesize_audio(text, output_audio_path=output_audio_path)
            result["audio"] = audio
            result["sample_rate"] = sample_rate
            self._record_audio_delivery(
                mode="blocking_speech_endpoint",
                text=text,
                stream_audio_chunks=1,
                stream_audio_bytes=int(audio.size * audio.dtype.itemsize),
            )
        return result

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
        del voice_ref
        cutoff_on_text_stop = not generate_audio
        payload = self._build_payload(
            messages=messages,
            modalities=["text", "audio"],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            stream=True,
        )
        self._log_request_start(
            trace_context,
            payload,
            streaming=True,
            cutoff_on_text_stop=cutoff_on_text_stop,
            requested_audio=generate_audio,
        )
        started_at = time.perf_counter()
        first_text_s = None
        text_finish_s = None
        cutoff_reason = "stream_complete"

        handle, _ = self.get_model()
        full_text = ""
        saw_stream_audio = False
        stream_audio_chunks = 0
        stream_audio_bytes = 0
        try:
            for event in handle.client.stream_chat(payload):
                if event.text_delta:
                    full_text += event.text_delta
                    if first_text_s is None:
                        first_text_s = time.perf_counter() - started_at
                        self._log_first_text(trace_context, first_text_s)
                    yield None, event.text_delta
                if event.modality == "text" and event.finish_reason == "stop":
                    text_finish_s = time.perf_counter() - started_at
                    self._log_text_finish(trace_context, text_finish_s, cutoff_on_text_stop=cutoff_on_text_stop)
                if cutoff_on_text_stop and event.modality == "text" and event.finish_reason == "stop":
                    cutoff_reason = "text_finish"
                    break
                if generate_audio and event.audio_delta:
                    audio_chunk = self._decode_stream_audio(event.audio_delta)
                    if audio_chunk is not None and audio_chunk.size:
                        saw_stream_audio = True
                        stream_audio_chunks += 1
                        stream_audio_bytes += len(event.audio_delta)
                        yield audio_chunk, ""
        except Exception:
            self._log_request_finish(
                trace_context,
                started_at,
                text=full_text,
                first_text_s=first_text_s,
                text_finish_s=text_finish_s,
                outcome="error",
                cutoff_reason=cutoff_reason,
                stream_audio_chunks=stream_audio_chunks,
                stream_audio_bytes=stream_audio_bytes,
            )
            raise
        if saw_stream_audio:
            self._record_audio_delivery(
                mode="inline_chat_stream",
                text=full_text,
                stream_audio_chunks=stream_audio_chunks,
                stream_audio_bytes=stream_audio_bytes,
            )
        else:
            self._record_audio_delivery(
                mode="text_only",
                text=full_text,
                stream_audio_chunks=stream_audio_chunks,
                stream_audio_bytes=stream_audio_bytes,
            )
        self._log_request_finish(
            trace_context,
            started_at,
            text=full_text,
            first_text_s=first_text_s,
            text_finish_s=text_finish_s,
            outcome="ok",
            cutoff_reason=cutoff_reason,
            stream_audio_chunks=stream_audio_chunks,
            stream_audio_bytes=stream_audio_bytes,
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
        full_audio_chunks: list[np.ndarray] = []
        player = None
        if not headless:
            from tools.audio.streaming_player import StreamingAudioPlayer

            player = StreamingAudioPlayer(sample_rate=_REMOTE_SPEECH_SR)
            player.start()

        full_text = ""
        try:
            for audio_chunk, text_chunk in self.chat_streaming(
                messages=messages,
                voice_ref=voice_ref,
                generate_audio=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
            ):
                if text_chunk:
                    full_text += text_chunk
                    if on_text_chunk:
                        on_text_chunk(text_chunk)
                if audio_chunk is not None:
                    full_audio_chunks.append(audio_chunk)
                    if player is not None:
                        player.push(audio_chunk)
        finally:
            if player is not None:
                player.finish()
                player.wait()
                player.stop()

        full_audio = np.concatenate(full_audio_chunks) if full_audio_chunks else None
        if output_audio_path and full_audio is not None:
            import soundfile as sf

            sf.write(output_audio_path, full_audio, _REMOTE_SPEECH_SR)

        return {
            "text": full_text,
            "audio": full_audio,
            "audio_path": output_audio_path,
            "sample_rate": _REMOTE_SPEECH_SR if full_audio is not None else None,
        }

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
        del generate_audio, voice_ref, output_audio_path, max_slice_nums
        image_url = self._image_to_data_url(image)
        payload = self._build_payload(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            stream=False,
        )
        response = self._post_json(payload)
        text = self._extract_text_from_content(response["choices"][0]["message"].get("content"))
        return {"text": text, "audio": None, "audio_path": None}

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
        del video_path, prompt, generate_audio, voice_ref, output_audio_path
        del temperature, max_new_tokens, repetition_penalty, top_p, top_k, enable_thinking, max_frames, on_progress
        raise NotImplementedError("qwen_remote video understanding is not implemented yet.")

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
        del video_path, prompt, generate_audio, voice_ref, temperature, max_new_tokens
        del repetition_penalty, top_p, top_k, enable_thinking, max_frames, chunk_seconds, on_chunk, on_progress
        raise NotImplementedError("qwen_remote chunked video understanding is not implemented yet.")

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
        del temperature, max_new_tokens, repetition_penalty, top_p, top_k, enable_thinking, chunk_seconds
        handle, _ = self.get_model()
        prepared_path, cleanup_path = self._prepare_audio_file(video_path)
        try:
            if self._supports_audio_transcriptions is False:
                text = self._transcribe_audio_via_chat_file(prepared_path, prompt=prompt)
            else:
                try:
                    response = handle.client.transcribe_audio_file(prepared_path, prompt=prompt)
                except Exception as exc:
                    if not self._should_fallback_to_inline_audio(exc):
                        raise
                    self._supports_audio_transcriptions = False
                    logger.warning(
                        "qwen_remote transcription endpoint unavailable; falling back to inline input_audio chat handling"
                    )
                    text = self._transcribe_audio_via_chat_file(prepared_path, prompt=prompt)
                else:
                    self._supports_audio_transcriptions = True
                    text = (response.get("text") or "").strip()
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

        if on_chunk:
            on_chunk(0, 1, text)
        return {"text": text, "audio": None, "audio_path": None, "sample_rate": None}

    def _build_payload(
        self,
        *,
        messages: list[dict],
        modalities: list[str],
        temperature: float,
        max_new_tokens: int,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
        stream: bool,
    ) -> dict[str, Any]:
        handle, _ = self.get_model()
        payload = {
            "model": handle.model_name,
            "messages": [self._normalize_message(message) for message in messages],
            "modalities": list(modalities),
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "stream": stream,
        }
        if enable_thinking:
            payload["enable_thinking"] = True
        return payload

    def _decode_stream_audio(self, audio_bytes: bytes) -> Optional[np.ndarray]:
        if not audio_bytes:
            return None
        if audio_bytes[:4] == b"RIFF":
            import soundfile as sf

            audio, _sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return np.asarray(audio, dtype=np.float32)

        sample_count = len(audio_bytes) // 2
        byte_count = sample_count * 2
        if byte_count == 0:
            return None
        return np.frombuffer(audio_bytes[:byte_count], dtype="<i2").astype(np.float32) / 32768.0

    def _record_audio_delivery(
        self,
        *,
        mode: str,
        text: str,
        stream_audio_chunks: int,
        stream_audio_bytes: int,
    ) -> None:
        self._runtime_status = {
            "last_audio_delivery_mode": mode,
            "last_text_chars": len(text),
            "last_stream_audio_chunks": int(stream_audio_chunks),
            "last_stream_audio_bytes": int(stream_audio_bytes),
        }
        logger.info(
            "qwen_remote audio delivery mode=%s text_chars=%d stream_audio_chunks=%d stream_audio_bytes=%d",
            mode,
            len(text),
            int(stream_audio_chunks),
            int(stream_audio_bytes),
        )

    def _log_request_start(
        self,
        trace_context: Optional[dict],
        payload: dict[str, Any],
        *,
        streaming: bool,
        cutoff_on_text_stop: bool = False,
        requested_audio: bool = False,
    ) -> None:
        trace_id = (trace_context or {}).get("request_id", "n/a")
        messages = payload.get("messages", [])
        last_user_preview = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                last_user_preview = self._extract_text_from_content(content)[:160].replace("\n", " ")
                if not last_user_preview and self._content_has_input_audio(content):
                    last_user_preview = "[audio input]"
                break
        trace_logger.info(
            "trace_id=%s stage=backend event=request_start streaming=%s modalities=%s requested_audio=%s cutoff_on_text_stop=%s message_count=%d temperature=%s max_tokens=%s top_p=%s top_k=%s repetition_penalty=%s enable_thinking=%s last_user_preview=%r",
            trace_id,
            streaming,
            payload.get("modalities"),
            requested_audio,
            cutoff_on_text_stop,
            len(messages),
            payload.get("temperature"),
            payload.get("max_tokens"),
            payload.get("top_p"),
            payload.get("top_k"),
            payload.get("repetition_penalty"),
            payload.get("enable_thinking", False),
            last_user_preview,
        )

    def _log_first_text(self, trace_context: Optional[dict], first_text_s: float) -> None:
        trace_id = (trace_context or {}).get("request_id", "n/a")
        trace_logger.info(
            "trace_id=%s stage=backend event=first_text first_text_s=%.3f",
            trace_id,
            first_text_s,
        )

    def _log_text_finish(
        self,
        trace_context: Optional[dict],
        text_finish_s: float,
        *,
        cutoff_on_text_stop: bool,
    ) -> None:
        trace_id = (trace_context or {}).get("request_id", "n/a")
        trace_logger.info(
            "trace_id=%s stage=backend event=text_finish text_finish_s=%.3f cutoff_on_text_stop=%s",
            trace_id,
            text_finish_s,
            cutoff_on_text_stop,
        )

    def _log_request_finish(
        self,
        trace_context: Optional[dict],
        started_at: float,
        *,
        text: str,
        first_text_s: Optional[float],
        text_finish_s: Optional[float],
        outcome: str,
        cutoff_reason: str,
        stream_audio_chunks: int = 0,
        stream_audio_bytes: int = 0,
    ) -> None:
        trace_id = (trace_context or {}).get("request_id", "n/a")
        trace_logger.info(
            "trace_id=%s stage=backend event=request_finish outcome=%s elapsed_s=%.3f first_text_s=%s text_finish_s=%s cutoff_reason=%s stream_audio_chunks=%d stream_audio_bytes=%d response_chars=%d preview=%r",
            trace_id,
            outcome,
            time.perf_counter() - started_at,
            "none" if first_text_s is None else f"{first_text_s:.3f}",
            "none" if text_finish_s is None else f"{text_finish_s:.3f}",
            cutoff_reason,
            int(stream_audio_chunks),
            int(stream_audio_bytes),
            len(text),
            text[:160].replace("\n", " "),
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        handle, _ = self.get_model()
        return handle.client.chat(payload)

    def _synthesize_audio(self, text: str, *, output_audio_path: Optional[str] = None) -> tuple[np.ndarray, int]:
        handle, _ = self.get_model()
        audio_bytes, _content_type = handle.client.synthesize_speech(
            text,
            voice=_REMOTE_DEFAULT_VOICE,
            task_type="CustomVoice",
            response_format="wav",
        )

        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if output_audio_path:
            Path(output_audio_path).write_bytes(audio_bytes)
        return audio, int(sample_rate)

    def _stream_audio(self, text: str):
        handle, _ = self.get_model()
        residual = b""
        for chunk in handle.client.stream_speech(
            text,
            voice=_REMOTE_DEFAULT_VOICE,
            task_type="CustomVoice",
            response_format="pcm",
        ):
            pcm = residual + chunk
            sample_count = len(pcm) // 2
            byte_count = sample_count * 2
            if byte_count == 0:
                residual = pcm
                continue
            audio_chunk = np.frombuffer(pcm[:byte_count], dtype="<i2").astype(np.float32) / 32768.0
            residual = pcm[byte_count:]
            if audio_chunk.size:
                yield audio_chunk

    def _normalize_message(self, message: dict) -> dict[str, Any]:
        return {
            "role": message["role"],
            "content": self._normalize_content(message.get("content", [])),
        }

    def _normalize_content(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, dict):
            return [content]
        if isinstance(content, np.ndarray):
            return self._ensure_audio_instruction(self._normalize_audio_content_item(content))
        if isinstance(content, list):
            normalized = []
            for item in content:
                if isinstance(item, str):
                    normalized.append({"type": "text", "text": item})
                elif isinstance(item, np.ndarray):
                    normalized.extend(self._normalize_audio_content_item(item))
                elif isinstance(item, dict):
                    normalized.append(item)
                else:
                    raise NotImplementedError(
                        f"qwen_remote does not support content item type {type(item).__name__}."
                    )
            if not normalized:
                raise RuntimeError("qwen_remote audio transcription produced no text.")
            return self._ensure_audio_instruction(normalized)
        raise NotImplementedError(f"qwen_remote does not support content type {type(content).__name__}.")

    def _ensure_audio_instruction(self, content_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        has_input_audio = any(
            isinstance(item, dict) and item.get("type") == "input_audio"
            for item in content_items
        )
        has_text = any(
            isinstance(item, dict) and item.get("type") == "text" and str(item.get("text", "")).strip()
            for item in content_items
        )
        if has_input_audio and not has_text:
            return [*content_items, {"type": "text", "text": _INLINE_AUDIO_RESPONSE_PROMPT}]
        return content_items

    def _transcribe_audio_array(self, audio: np.ndarray, prompt: str = "Transcribe this audio completely and verbatim.") -> str:
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        if self._supports_audio_transcriptions is False:
            return None

        handle, _ = self.get_model()
        try:
            response = handle.client.transcribe_audio_bytes(
                buffer.getvalue(),
                filename="mic.wav",
                prompt=prompt,
            )
        except Exception as exc:
            if not self._should_fallback_to_inline_audio(exc):
                raise
            self._supports_audio_transcriptions = False
            logger.warning(
                "qwen_remote transcription endpoint unavailable; falling back to inline input_audio chat handling"
            )
            return None
        self._supports_audio_transcriptions = True
        return (response.get("text") or "").strip()

    def _normalize_audio_content_item(self, audio: np.ndarray) -> list[dict[str, Any]]:
        transcript = self._transcribe_audio_array(audio)
        if transcript:
            return [{"type": "text", "text": transcript}]
        return [self._audio_array_to_input_audio_part(audio)]

    def _audio_array_to_input_audio_part(self, audio: np.ndarray) -> dict[str, Any]:
        if audio.ndim != 1:
            audio = np.asarray(audio).reshape(-1)

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.asarray(audio, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "format": "wav",
            },
        }

    def _transcribe_audio_via_chat_file(self, file_path: str, *, prompt: str) -> str:
        import soundfile as sf

        audio, sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != 16000:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return self._transcribe_audio_via_chat_array(audio, prompt=prompt)

    def _transcribe_audio_via_chat_array(self, audio: np.ndarray, *, prompt: str) -> str:
        payload = self._build_payload(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        self._audio_array_to_input_audio_part(audio),
                    ],
                }
            ],
            modalities=["text"],
            temperature=0.0,
            max_new_tokens=4096,
            repetition_penalty=1.0,
            top_p=1.0,
            top_k=1,
            enable_thinking=False,
            stream=False,
        )
        response = self._post_json(payload)
        return self._extract_text_from_content(response["choices"][0]["message"].get("content")).strip()

    def _should_fallback_to_inline_audio(self, exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code in {404, 405, 501}:
            return True
        message = str(exc).lower()
        if "audio/transcriptions" in message and ("404" in message or "not found" in message):
            return True
        return False

    def _prepare_audio_file(self, media_path: str) -> tuple[str, Optional[Path]]:
        path = Path(media_path)
        if path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".aac"}:
            return str(path), None

        import subprocess
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio_path = Path(temp_audio.name)
        temp_audio.close()
        try:
            subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(temp_audio_path)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            temp_audio_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to extract audio for qwen_remote transcription: {exc.stderr.decode(errors='replace')}"
            ) from exc
        return str(temp_audio_path), temp_audio_path

    def _extract_text_from_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    def _content_has_input_audio(self, content: Any) -> bool:
        if isinstance(content, dict):
            return content.get("type") == "input_audio"
        if isinstance(content, list):
            return any(self._content_has_input_audio(item) for item in content)
        return False

    def _image_to_data_url(self, image: Any) -> str:
        if not hasattr(image, "save"):
            raise NotImplementedError("qwen_remote image processing expects a PIL image.")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
