"""Local Qwen3-Omni backend using the official Transformers implementation."""

from __future__ import annotations

import base64
import io
import importlib.util
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

from tools.model.backends.base_backend import ModelBackend
from tools.model import model_manager as mm


_LOCAL_QWEN_OUTPUT_SR = 24000
_LOCAL_QWEN_INPUT_SR = 16000
_DEFAULT_SPEAKER = "Ethan"
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".oga", ".aac"}


@dataclass
class _LocalQwenHandle:
    model: Any
    processor: Any
    process_mm_info: Any
    checkpoint: str
    device: Any
    dtype: Any


def _require_local_qwen_dependencies():
    try:
        import soundfile as sf
        import torch
        from qwen_omni_utils import process_mm_info
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Local qwen_transformers backend requires transformers>=4.57.3, qwen-omni-utils, and soundfile."
        ) from exc

    return {
        "sf": sf,
        "torch": torch,
        "process_mm_info": process_mm_info,
        "model_cls": Qwen3OmniMoeForConditionalGeneration,
        "processor_cls": Qwen3OmniMoeProcessor,
    }


class QwenTransformersBackend(ModelBackend):
    """Adapter for local Qwen3-Omni inference via Transformers."""

    name = "qwen_transformers"

    def __init__(self) -> None:
        self._handle: Optional[_LocalQwenHandle] = None
        self._runtime_status = {
            "attention_backend": "uninitialized",
            "requested_attention_backend": None,
            "torch_dtype": None,
            "tf32_matmul": None,
            "tf32_cudnn": None,
            "cuda_device_name": None,
            "cuda_capability": None,
        }

    def get_capabilities(self) -> dict[str, Any]:
        cfg = mm.get_qwen_transformers_config()
        return {
            "backend": self.name,
            "model_name": cfg.get("checkpoint") or "",
            "supports_audio_input": True,
            "supports_audio_output": True,
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
            "supports_voice_reference": False,
            "supports_image_input": True,
            "supports_video_input": True,
            "input_sample_rate": _LOCAL_QWEN_INPUT_SR,
            "output_sample_rate": _LOCAL_QWEN_OUTPUT_SR,
            "transport": {
                "protocol": "local_transformers",
                "streaming_mode": "emulated_after_generation",
            },
        }

    def get_runtime_status(self) -> dict[str, Any]:
        return dict(self._runtime_status)

    def set_quantization(self, mode: str) -> None:
        if mode != "none":
            raise ValueError("qwen_transformers currently supports only quantization='none'.")

    def set_auto_update(self, enabled: bool) -> None:
        _ = enabled

    def get_model(self):
        if self._handle is not None:
            return self._handle.model, self._handle.processor

        cfg = mm.get_qwen_transformers_config()
        deps = _require_local_qwen_dependencies()
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
        })
        self._handle = _LocalQwenHandle(
            model=model,
            processor=processor,
            process_mm_info=deps["process_mm_info"],
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
        del voice_ref, trace_context
        conversation = [self._normalize_message(message) for message in messages]
        return self._generate_response(
            conversation=conversation,
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
        del voice_ref, trace_context
        result = self.chat(
            messages=messages,
            generate_audio=generate_audio,
            output_audio_path=None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

        text = result.get("text", "")
        if text:
            yield None, text

        audio = result.get("audio")
        if audio is not None:
            chunk_size = _LOCAL_QWEN_OUTPUT_SR
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

            player = StreamingAudioPlayer(sample_rate=_LOCAL_QWEN_OUTPUT_SR)
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
            deps = _require_local_qwen_dependencies()
            deps["sf"].write(output_audio_path, full_audio, _LOCAL_QWEN_OUTPUT_SR)

        return {
            "text": full_text,
            "audio": full_audio,
            "audio_path": output_audio_path,
            "sample_rate": _LOCAL_QWEN_OUTPUT_SR if full_audio is not None else None,
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
        del voice_ref, max_slice_nums
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": self._normalize_image_value(image)},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._generate_response(
            conversation=conversation,
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
        del voice_ref
        if on_progress is not None:
            on_progress("Preparing video", 0.1)

        cfg = mm.get_qwen_transformers_config()
        conversation = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path), "nframes": max_frames},
                {"type": "text", "text": prompt},
            ],
        }]

        try:
            result = self._generate_response(
                conversation=conversation,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                use_audio_in_video=bool(cfg.get("use_audio_in_video", True)),
            )
        except AssertionError as exc:
            if "audio track" not in str(exc).lower():
                raise
            result = self._generate_response(
                conversation=conversation,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                use_audio_in_video=False,
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
        del voice_ref
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
                max_frames=max_frames,
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
                    max_frames=max_frames,
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
        deps = _require_local_qwen_dependencies()
        try:
            audio, sample_rate = deps["sf"].read(audio_path, dtype="float32", always_2d=False)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sample_rate != _LOCAL_QWEN_INPUT_SR:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_LOCAL_QWEN_INPUT_SR)
        finally:
            if cleanup_path is not None:
                cleanup_path.unlink(missing_ok=True)

        samples_per_chunk = int(chunk_seconds * _LOCAL_QWEN_INPUT_SR)
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
        generate_audio: bool,
        output_audio_path: Optional[str],
        temperature: float,
        max_new_tokens: int,
        repetition_penalty: float,
        top_p: float,
        top_k: int,
        enable_thinking: bool,
        use_audio_in_video: Optional[bool] = None,
    ) -> dict[str, Any]:
        del enable_thinking
        handle = self._get_handle()
        cfg = mm.get_qwen_transformers_config()
        use_audio_in_video = bool(cfg.get("use_audio_in_video", True)) if use_audio_in_video is None else bool(use_audio_in_video)

        prompt_text = handle.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        mm_info = handle.process_mm_info(
            conversation,
            use_audio_in_video=use_audio_in_video,
            return_video_kwargs=True,
        )
        audios, images, videos, video_kwargs = mm_info
        if isinstance(video_kwargs.get("fps"), list):
            fps_values = video_kwargs.get("fps") or []
            video_kwargs["fps"] = fps_values[0] if fps_values else 1.0
        inputs = handle.processor(
            text=prompt_text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
            **video_kwargs,
        )
        inputs = self._move_inputs(inputs, handle)
        input_length = int(inputs["input_ids"].shape[1])

        thinker_kwargs: dict[str, Any] = {
            "thinker_return_dict_in_generate": True,
            "thinker_max_new_tokens": max_new_tokens,
            "thinker_do_sample": bool(temperature > 0),
            "thinker_top_p": top_p,
            "thinker_top_k": top_k,
            "thinker_repetition_penalty": repetition_penalty,
            "speaker": cfg.get("speaker", _DEFAULT_SPEAKER),
            "use_audio_in_video": use_audio_in_video,
            "return_audio": generate_audio,
        }
        if temperature > 0:
            thinker_kwargs["thinker_temperature"] = temperature

        text_ids, audio = handle.model.generate(**inputs, **thinker_kwargs)
        generated_text = handle.processor.batch_decode(
            text_ids.sequences[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        waveform = None
        if audio is not None:
            waveform = audio.reshape(-1).detach().cpu().numpy().astype(np.float32)
            if output_audio_path:
                deps = _require_local_qwen_dependencies()
                deps["sf"].write(output_audio_path, waveform, _LOCAL_QWEN_OUTPUT_SR)

        return {
            "text": generated_text,
            "audio": waveform,
            "audio_path": output_audio_path,
            "sample_rate": _LOCAL_QWEN_OUTPUT_SR if waveform is not None else None,
        }

    def _get_handle(self) -> _LocalQwenHandle:
        if self._handle is None:
            self.get_model()
        assert self._handle is not None
        return self._handle

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
                        f"qwen_transformers does not support content item type {type(item).__name__}."
                    )
            return normalized
        raise NotImplementedError(f"qwen_transformers does not support content type {type(content).__name__}.")

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
            normalized = {"type": "image", "image": self._normalize_image_value(image_value)}
            for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                if key in item:
                    normalized[key] = item[key]
            return normalized
        if item_type == "video":
            video_value = item.get("video") or item.get("video_url")
            normalized = {"type": "video", "video": str(video_value)}
            for key in ("video_start", "video_end", "fps", "nframes", "min_frames", "max_frames"):
                if key in item:
                    normalized[key] = item[key]
            return normalized
        raise NotImplementedError(f"Unsupported qwen_transformers content item type: {item_type!r}")

    def _normalize_audio_value(self, audio: Any) -> Any:
        if isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            return audio.reshape(-1)
        if isinstance(audio, str):
            return audio
        raise NotImplementedError(f"Unsupported qwen_transformers audio value type {type(audio).__name__}.")

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
        raise NotImplementedError(f"Unsupported qwen_transformers image value type {type(image).__name__}.")

    def _decode_base64_audio(self, data: str, fmt: str) -> np.ndarray:
        deps = _require_local_qwen_dependencies()
        raw = base64.b64decode(data)
        audio, sample_rate = deps["sf"].read(io.BytesIO(raw), dtype="float32", always_2d=False, format=fmt.upper())
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sample_rate != _LOCAL_QWEN_INPUT_SR:
            import librosa

            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_LOCAL_QWEN_INPUT_SR)
        return audio.reshape(-1)

    def _move_inputs(self, inputs: Any, handle: _LocalQwenHandle) -> Any:
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
                f"Failed to extract audio for qwen_transformers transcription: {exc.stderr.decode(errors='replace')}"
            ) from exc
        return str(temp_audio_path), temp_audio_path

    def _probe_duration(self, media_path: str, ffmpeg_exe: str) -> float:
        probe = subprocess.run([ffmpeg_exe, "-i", str(media_path)], capture_output=True, text=True)
        for line in probe.stderr.splitlines():
            if "Duration:" not in line:
                continue
            parts = line.split("Duration:", 1)[1].split(",", 1)[0].strip().split(":")
            try:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except (ValueError, IndexError):
                return 0.0
        return 0.0

    def _get_ffmpeg_exe(self) -> str:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()

    def _format_time_range(self, start_s: float, end_s: float) -> str:
        return f"{int(start_s // 60)}:{int(start_s % 60):02d}-{int(end_s // 60)}:{int(end_s % 60):02d}"

    def _resolve_torch_dtype(self, value: Any, torch_module: Any) -> Any:
        if value in (None, "auto"):
            if torch_module.cuda.is_available() and getattr(torch_module.cuda, "is_bf16_supported", lambda: False)():
                return torch_module.bfloat16
            return "auto"
        if value == "bfloat16":
            return torch_module.bfloat16
        if value == "float16":
            return torch_module.float16
        if value == "float32":
            return torch_module.float32
        return value

    def _resolve_attention_implementation(self, requested: Any, torch_module: Any) -> Optional[str]:
        if requested:
            return str(requested)
        if self._has_flash_attention_2():
            return "flash_attention_2"
        if torch_module.cuda.is_available():
            return "sdpa"
        return None

    def _has_flash_attention_2(self) -> bool:
        return importlib.util.find_spec("flash_attn") is not None

    def _apply_runtime_optimizations(self, torch_module: Any) -> None:
        if not torch_module.cuda.is_available():
            return

        if hasattr(torch_module, "set_float32_matmul_precision"):
            torch_module.set_float32_matmul_precision("high")

        cuda_backends = getattr(torch_module.backends, "cuda", None)
        if cuda_backends is not None:
            matmul = getattr(cuda_backends, "matmul", None)
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                matmul.allow_tf32 = True
                self._runtime_status["tf32_matmul"] = bool(matmul.allow_tf32)
            for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
                fn = getattr(cuda_backends, name, None)
                if callable(fn):
                    fn(True)

        cudnn_backend = getattr(torch_module.backends, "cudnn", None)
        if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
            cudnn_backend.allow_tf32 = True
            self._runtime_status["tf32_cudnn"] = bool(cudnn_backend.allow_tf32)

    def _infer_model_device(self, model: Any, torch_module: Any) -> Any:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
