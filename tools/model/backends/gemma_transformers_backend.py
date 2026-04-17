"""Local Gemma 4 backend using native Transformers multimodal inference."""

from __future__ import annotations

import base64
import io
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from tools.model import model_manager as mm
from tools.model.backends.base_backend import ModelBackend


_LOCAL_GEMMA_INPUT_SR = 16000
_LOCAL_TTS_OUTPUT_SR = 24000
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".aac"}


@dataclass
class _LocalGemmaHandle:
    model: Any
    processor: Any
    checkpoint: str
    device: Any
    dtype: Any


def _require_local_gemma_dependencies():
    try:
        import soundfile as sf
        import torch
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Local gemma_transformers backend requires transformers>=4.57.3 and soundfile."
        ) from exc

    return {
        "sf": sf,
        "torch": torch,
        "processor_cls": AutoProcessor,
        "model_cls": Gemma4ForConditionalGeneration,
    }


class GemmaTransformersBackend(ModelBackend):
    """Adapter for local Gemma 4 inference via Transformers."""

    name = "gemma_transformers"

    def __init__(self) -> None:
        self._handle: Optional[_LocalGemmaHandle] = None
        self._runtime_status = {
            "attention_backend": "uninitialized",
            "requested_attention_backend": None,
            "torch_dtype": None,
            "tf32_matmul": None,
            "tf32_cudnn": None,
            "cuda_device_name": None,
            "cuda_capability": None,
            "checkpoint": None,
        }

    def get_capabilities(self) -> dict[str, Any]:
        cfg = mm.get_gemma_transformers_config()
        speech_backend = str(cfg.get("speech_backend") or "none")
        hybrid_tts = speech_backend == "minicpm_streaming"
        return {
            "backend": self.name,
            "model_name": cfg.get("checkpoint") or "",
            "supports_audio_input": True,
            "supports_audio_output": hybrid_tts,
            "supports_streaming_text": True,
            "supports_streaming_audio": hybrid_tts,
            "supports_voice_reference": hybrid_tts,
            "supports_image_input": True,
            "supports_video_input": True,
            "input_sample_rate": _LOCAL_GEMMA_INPUT_SR,
            "output_sample_rate": _LOCAL_TTS_OUTPUT_SR if hybrid_tts else None,
            "transport": {
                "protocol": "local_transformers",
                "streaming_mode": "emulated_after_generation+minicpm_tts" if hybrid_tts else "emulated_after_generation",
            },
        }

    def get_runtime_status(self) -> dict[str, Any]:
        return dict(self._runtime_status)

    def set_quantization(self, mode: str) -> None:
        if mode != "none":
            raise ValueError("gemma_transformers currently supports only quantization='none'.")

    def set_auto_update(self, enabled: bool) -> None:
        _ = enabled

    def get_model(self):
        if self._handle is not None:
            return self._handle.model, self._handle.processor

        cfg = mm.get_gemma_transformers_config()
        deps = _require_local_gemma_dependencies()
        torch = deps["torch"]

        self._apply_runtime_optimizations(torch)

        processor = deps["processor_cls"].from_pretrained(
            cfg["checkpoint"],
            local_files_only=bool(cfg.get("local_files_only", False)),
        )

        requested_attn_implementation = cfg.get("attn_implementation")
        resolved_attn_implementation = self._resolve_attention_implementation(requested_attn_implementation, torch)

        model_kwargs: dict[str, Any] = {
            "device_map": cfg.get("device_map", "auto"),
            "dtype": self._resolve_torch_dtype(cfg.get("torch_dtype", "auto"), torch),
            "local_files_only": bool(cfg.get("local_files_only", False)),
        }
        if resolved_attn_implementation:
            model_kwargs["attn_implementation"] = resolved_attn_implementation

        model = deps["model_cls"].from_pretrained(cfg["checkpoint"], **model_kwargs)
        device = self._infer_model_device(model, torch)
        dtype = getattr(model, "dtype", None)
        self._runtime_status.update({
            "attention_backend": resolved_attn_implementation or "backend_default",
            "requested_attention_backend": requested_attn_implementation,
            "torch_dtype": str(model_kwargs["dtype"]),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            "checkpoint": cfg.get("checkpoint"),
        })
        self._handle = _LocalGemmaHandle(
            model=model,
            processor=processor,
            checkpoint=cfg["checkpoint"],
            device=device,
            dtype=dtype,
        )
        return model, processor

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
        conversation = [self._normalize_message(message) for message in messages]
        response = self._generate_response(
            conversation=conversation,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )
        audio_result = self._synthesize_speech_if_needed(
            text=response["text"],
            voice_ref=voice_ref,
            generate_audio=generate_audio,
            output_audio_path=output_audio_path,
            trace_context=trace_context,
            source_backend=self.name,
        )
        response.update(audio_result)
        return response

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
        result = self.chat(
            messages=messages,
            voice_ref=voice_ref,
            generate_audio=generate_audio,
            output_audio_path=None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            trace_context=trace_context,
        )

        text = result.get("text", "")
        if text:
            yield None, text

        audio = result.get("audio")
        if audio is not None:
            chunk_size = _LOCAL_TTS_OUTPUT_SR
            for start in range(0, len(audio), chunk_size):
                yield audio[start:start + chunk_size], ""

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

            player = StreamingAudioPlayer(sample_rate=_LOCAL_TTS_OUTPUT_SR)
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
            deps = _require_local_gemma_dependencies()
            deps["sf"].write(output_audio_path, full_audio, _LOCAL_TTS_OUTPUT_SR)

        return {
            "text": full_text,
            "audio": full_audio,
            "audio_path": output_audio_path,
            "sample_rate": _LOCAL_TTS_OUTPUT_SR if full_audio is not None else None,
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
        del max_slice_nums
        return self.chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "image": self._normalize_image_value(image)},
                    {"type": "text", "text": prompt},
                ],
            }],
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
        del max_frames
        if on_progress is not None:
            on_progress("Preparing video", 0.1)

        result = self.chat(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": prompt},
                ],
            }],
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

        if on_progress is not None:
            on_progress("Complete", 1.0)
        return result

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
        del voice_ref, max_frames
        ffmpeg_exe = self._get_ffmpeg_exe()
        duration = self._probe_duration(video_path, ffmpeg_exe)
        if duration <= 0 or duration <= chunk_seconds:
            return self.process_video(
                video_path=video_path,
                prompt=prompt,
                generate_audio=generate_audio,
                output_audio_path=None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                on_progress=on_progress,
            )

        total_chunks = max(1, math.ceil(duration / float(chunk_seconds)))
        all_parts: list[str] = []
        if on_progress is not None:
            on_progress("Chunking video", 0.05)

        for index in range(total_chunks):
            start_s = index * chunk_seconds
            end_s = min((index + 1) * chunk_seconds, duration)
            if on_progress is not None:
                on_progress(f"Chunk {index + 1}/{total_chunks}", index / total_chunks)

            temp_chunk = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_chunk_path = Path(temp_chunk.name)
            temp_chunk.close()
            try:
                subprocess.run(
                    [
                        ffmpeg_exe,
                        "-y",
                        "-ss",
                        str(start_s),
                        "-t",
                        str(max(0.1, end_s - start_s)),
                        "-i",
                        str(video_path),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-c:a",
                        "aac",
                        "-ac",
                        "1",
                        str(temp_chunk_path),
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                temp_chunk_path.unlink(missing_ok=True)
                all_parts.append(f"[Chunk {index + 1}: failed to extract]")
                continue

            chunk_prompt = f"[Video segment {self._format_time_range(start_s, end_s)}] {prompt}"
            try:
                chunk_result = self.process_video(
                    video_path=str(temp_chunk_path),
                    prompt=chunk_prompt,
                    generate_audio=False,
                    output_audio_path=None,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                    top_p=top_p,
                    top_k=top_k,
                    enable_thinking=enable_thinking,
                )
                chunk_text = (chunk_result.get("text") or "").strip()
                if chunk_text:
                    all_parts.append(f"**[{self._format_time_range(start_s, end_s)}]** {chunk_text}")
            finally:
                temp_chunk_path.unlink(missing_ok=True)

            accumulated = "\n\n".join(all_parts)
            if on_chunk is not None:
                on_chunk(index, total_chunks, accumulated)

        if on_progress is not None:
            on_progress("Complete", 1.0)
        return {"text": "\n\n".join(all_parts), "audio": None, "audio_path": None, "sample_rate": None}

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
        audio_path, cleanup_path = self._prepare_audio_path(video_path)
        deps = _require_local_gemma_dependencies()
        try:
            audio, sample_rate = deps["sf"].read(audio_path, dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != _LOCAL_GEMMA_INPUT_SR:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_LOCAL_GEMMA_INPUT_SR)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

        samples_per_chunk = int(chunk_seconds * _LOCAL_GEMMA_INPUT_SR)
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

    def _generate_response(
        self,
        *,
        conversation: list[dict],
        temperature: float,
        max_new_tokens: int,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
    ) -> dict[str, Any]:
        handle = self._get_handle()
        cfg = mm.get_gemma_transformers_config()
        prompt_text = handle.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=enable_thinking,
        )

        images, audios, videos, video_metadata = self._collect_multimodal_inputs(
            conversation,
            video_backend=cfg.get("video_backend") or "pyav",
        )
        processor_kwargs: dict[str, Any] = {
            "text": prompt_text,
            "return_tensors": "pt",
        }
        if images:
            processor_kwargs["images"] = images
        if audios:
            processor_kwargs["audio"] = audios
        if videos:
            processor_kwargs["videos"] = videos
            processor_kwargs["videos_kwargs"] = {"video_metadata": video_metadata, "return_metadata": True}

        inputs = handle.processor(**processor_kwargs)
        inputs = self._move_inputs(inputs, handle)
        input_length = int(inputs["input_ids"].shape[1])

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(temperature > 0),
            "repetition_penalty": repetition_penalty,
            "top_p": top_p,
            "top_k": top_k,
        }
        if temperature > 0:
            generate_kwargs["temperature"] = temperature

        outputs = handle.model.generate(**inputs, **generate_kwargs)
        decoded = handle.processor.decode(outputs[0][input_length:], skip_special_tokens=False)
        parsed = handle.processor.parse_response(decoded)
        reasoning = ""
        text = decoded
        if isinstance(parsed, dict):
            reasoning = str(parsed.get("thinking") or "")
            text = str(parsed.get("content") or "")

        self._runtime_status.update({
            "last_prompt_chars": len(prompt_text),
            "last_prompt_tokens_est": self._estimate_token_count(prompt_text),
            "last_text_chars": len(text),
            "last_text_tokens_est": self._estimate_token_count(text),
            "last_reasoning_chars": len(reasoning),
            "reasoning_detected": bool(reasoning.strip()),
            "last_n_predict": int(max_new_tokens),
        })
        return {"text": text, "audio": None, "audio_path": None, "sample_rate": None, "reasoning": reasoning}

    def _normalize_message(self, message: dict[str, Any]) -> dict[str, Any]:
        return {
            "role": message["role"],
            "content": self._normalize_content(message.get("content", [])),
        }

    def _normalize_content(self, content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if isinstance(content, np.ndarray):
            return [{"type": "audio", "audio": self._normalize_audio_value(content)}]
        if isinstance(content, Image.Image):
            return [{"type": "image", "image": content}]
        if isinstance(content, dict):
            return [self._normalize_content_item(content)]
        if isinstance(content, list):
            normalized: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    normalized.append({"type": "text", "text": item})
                elif isinstance(item, np.ndarray):
                    normalized.append({"type": "audio", "audio": self._normalize_audio_value(item)})
                elif isinstance(item, Image.Image):
                    normalized.append({"type": "image", "image": item})
                elif isinstance(item, dict):
                    normalized.append(self._normalize_content_item(item))
                else:
                    raise NotImplementedError(
                        f"gemma_transformers does not support content item type {type(item).__name__}."
                    )
            return normalized
        raise NotImplementedError(f"gemma_transformers does not support content type {type(content).__name__}.")

    def _normalize_content_item(self, item: dict[str, Any]) -> dict[str, Any]:
        item_type = item.get("type", "text")
        if item_type == "text":
            return {"type": "text", "text": str(item.get("text", ""))}
        if item_type == "audio":
            return {
                "type": "audio",
                "audio": self._normalize_audio_value(item.get("audio") or item.get("audio_url")),
            }
        if item_type == "input_audio":
            audio_info = item.get("input_audio", {})
            data = audio_info.get("data")
            fmt = str(audio_info.get("format", "wav")).lower()
            if not data:
                raise ValueError("input_audio payload is missing data")
            return {"type": "audio", "audio": self._decode_base64_audio(data, fmt)}
        if item_type in {"image", "image_url"}:
            image_value = item.get("image")
            if image_value is None and "image_url" in item:
                image_url = item.get("image_url")
                if isinstance(image_url, dict):
                    image_value = image_url.get("url")
                else:
                    image_value = image_url
            return {"type": "image", "image": self._normalize_image_value(image_value)}
        if item_type == "video":
            video_value = item.get("video") or item.get("video_url")
            return {"type": "video", "video": str(video_value)}
        raise NotImplementedError(f"Unsupported gemma_transformers content item type: {item_type!r}")

    def _normalize_audio_value(self, audio: Any) -> Any:
        if isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                if audio.shape[0] <= 8:
                    audio = np.mean(audio, axis=0)
                else:
                    audio = np.mean(audio, axis=1)
            elif audio.ndim != 1:
                raise NotImplementedError(
                    f"Unsupported gemma_transformers audio array rank {audio.ndim}; expected mono or stereo waveform."
                )
            return audio
        if isinstance(audio, str):
            return audio
        raise NotImplementedError(f"Unsupported gemma_transformers audio value type {type(audio).__name__}.")

    def _normalize_image_value(self, image: Any) -> Any:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, np.ndarray):
            array = np.asarray(image)
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating):
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                else:
                    array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array)
        if isinstance(image, str):
            return image
        raise NotImplementedError(f"Unsupported gemma_transformers image value type {type(image).__name__}.")

    def _decode_base64_audio(self, data: str, fmt: str) -> np.ndarray:
        deps = _require_local_gemma_dependencies()
        raw = base64.b64decode(data)
        del fmt
        audio, sample_rate = deps["sf"].read(io.BytesIO(raw), dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != _LOCAL_GEMMA_INPUT_SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_LOCAL_GEMMA_INPUT_SR)
        return audio

    def _collect_multimodal_inputs(
        self,
        conversation: list[dict[str, Any]],
        *,
        video_backend: str,
    ) -> tuple[list[Any], list[Any], list[Any], list[Any]]:
        images: list[Any] = []
        audios: list[Any] = []
        videos: list[Any] = []
        video_metadata: list[Any] = []
        for message in conversation:
            for item in message.get("content", []):
                item_type = item.get("type")
                if item_type == "image":
                    images.append(item.get("image"))
                elif item_type == "audio":
                    audios.append(item.get("audio"))
                elif item_type == "video":
                    decoded_video, metadata = self._load_video_input(item.get("video"), backend=video_backend)
                    videos.append(decoded_video)
                    video_metadata.append(metadata)
        return images, audios, videos, video_metadata

    def _load_video_input(self, video: Any, *, backend: str) -> tuple[Any, Any]:
        if isinstance(video, str):
            from transformers.video_utils import load_video

            return load_video(video, backend=backend)
        return video, None

    def _move_inputs(self, inputs: Any, handle: _LocalGemmaHandle) -> Any:
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                tensor = value.to(handle.device)
                if handle.dtype is not None and getattr(tensor, "is_floating_point", lambda: False)():
                    tensor = tensor.to(handle.dtype)
                moved[key] = tensor
            else:
                moved[key] = value
        return moved

    def _prepare_audio_path(self, media_path: str) -> tuple[str, Optional[Path]]:
        path = Path(media_path)
        if path.suffix.lower() in _AUDIO_EXTENSIONS:
            return str(path), None

        ffmpeg_exe = self._get_ffmpeg_exe()
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
                f"Failed to extract audio for gemma_transformers transcription: {exc.stderr.decode(errors='replace')}"
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
        cfg = mm.get_gemma_transformers_config()
        if not generate_audio or (cfg.get("speech_backend") or "none") != "minicpm_streaming":
            return {"audio": None, "audio_path": output_audio_path, "sample_rate": None}
        return mm.synthesize_text_with_minicpm(
            text,
            voice_ref=voice_ref,
            output_audio_path=output_audio_path,
            trace_context=trace_context,
            source_backend=source_backend,
        )

    def _apply_runtime_optimizations(self, torch: Any) -> None:
        if hasattr(torch.backends, "cuda") and torch.cuda.is_available():
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = True
        self._runtime_status.update({
            "tf32_matmul": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)) if hasattr(torch.backends, "cuda") else None,
            "tf32_cudnn": bool(getattr(torch.backends.cudnn, "allow_tf32", False)) if hasattr(torch.backends, "cudnn") else None,
        })

    def _resolve_attention_implementation(self, requested: Optional[str], torch: Any) -> Optional[str]:
        if not requested or requested == "auto":
            if torch.cuda.is_available() and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                return "sdpa"
            return None
        return requested

    def _resolve_torch_dtype(self, value: str, torch: Any) -> Any:
        if value == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        resolved = mapping.get(str(value).lower())
        if resolved is None:
            raise ValueError(f"Unsupported torch_dtype for gemma_transformers: {value!r}")
        return resolved

    def _infer_model_device(self, model: Any, torch: Any):
        if hasattr(model, "device"):
            return model.device
        if hasattr(model, "hf_device_map") and model.hf_device_map:
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, str):
                return torch.device(first_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_handle(self) -> _LocalGemmaHandle:
        if self._handle is None:
            self.get_model()
        assert self._handle is not None
        return self._handle

    def _get_ffmpeg_exe(self) -> str:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()

    def _probe_duration(self, video_path: str, ffmpeg_exe: str) -> float:
        ffprobe = Path(ffmpeg_exe).with_name("ffprobe.exe")
        if not ffprobe.exists():
            return 0.0
        completed = subprocess.run(
            [
                str(ffprobe),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return 0.0
        try:
            return float((completed.stdout or "0").strip() or 0.0)
        except ValueError:
            return 0.0

    def _format_time_range(self, start_s: float, end_s: float) -> str:
        def _fmt(value: float) -> str:
            total = int(round(value))
            minutes, seconds = divmod(total, 60)
            hours, minutes = divmod(minutes, 60)
            if hours:
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            return f"{minutes:02d}:{seconds:02d}"

        return f"{_fmt(start_s)}-{_fmt(end_s)}"

    def _estimate_token_count(self, text: str) -> int:
        text = (text or "").strip()
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))