"""
model_manager.py ΓÇö Load and manage MiniCPM-o 4.5 for multimodal inference.

Singleton pattern: call get_model() to load once, reuse everywhere.
Supports text, audio (with voice cloning), image, and video inputs.

Two generation paths:
  - chat()                        ΓÇö file-based, blocking (original)
  - chat_streaming()              ΓÇö yields (audio_chunk, text) with no file I/O
  - chat_streaming_with_playback()ΓÇö streaming + real-time speaker playback
"""

import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Callable, Generator, Optional

import torch
import numpy as np

from tools.shared.debug_trace import get_trace_logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_NAME = "openbmb/MiniCPM-o-4_5"

_QWEN_REMOTE_DEFAULTS = {
    "base_url": "http://127.0.0.1:8000/v1",
    "api_key": None,
    "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "timeout_s": 120.0,
    "endpoints": {
        "health": "/health",
        "models": "models",
        "chat_completions": "chat/completions",
        "audio_transcriptions": "audio/transcriptions",
        "audio_speech": "audio/speech",
        "audio_voices": "audio/voices",
        "responses": "responses",
        "realtime": "realtime",
    },
}

_QWEN_TRANSFORMERS_DEFAULTS = {
    "checkpoint": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "device_map": "auto",
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "speaker": "Ethan",
    "use_audio_in_video": True,
    "local_files_only": False,
}

_GEMMA_TRANSFORMERS_DEFAULTS = {
    "checkpoint": r"D:\OmniChatModels\gemma4-e4b-it-official\hf",
    "device_map": "auto",
    "torch_dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "speech_backend": "none",
    "local_files_only": False,
    "use_audio_in_video": True,
    "video_backend": "pyav",
}

_QWEN_LLAMACPP_DEFAULTS = {
    "name": "Qwen/Qwen3.5-27B",
    "llama_root": str((Path(tempfile.gettempdir()) / "llama-cpp-qwen35-test" / "llama.cpp").resolve()),
    "cli_path": None,
    "model_path": r"D:\OmniChatModels\qwen35-27b-official\gguf\Qwen-Qwen3.5-27B-27B-Q4_K_M-00001-of-00002.gguf",
    "mmproj_path": r"D:\OmniChatModels\qwen35-27b-official\gguf\mmproj-Qwen-Qwen3.5-27B-F16.gguf",
    "n_gpu_layers": 99,
    "flash_attn": True,
    "context_length": 8192,
    "use_jinja": True,
    "speech_backend": "none",
    "timeout_s": 120.0,
}

_GEMMA_LLAMACPP_DEFAULTS = {
    "name": "Gemma 4 (llama.cpp prototype)",
    "llama_root": str((Path(tempfile.gettempdir()) / "llama-cpp-gemma4-test" / "llama.cpp").resolve()),
    "cli_path": None,
    "model_path": r"D:\OmniChatModels\gemma4\gguf\model.gguf",
    "mmproj_path": r"D:\OmniChatModels\gemma4\gguf\mmproj-f16.gguf",
    "n_gpu_layers": 99,
    "flash_attn": True,
    "context_length": 8192,
    "use_jinja": True,
    "speech_backend": "none",
    "timeout_s": 120.0,
}

# Module-level singleton
_model = None
_tokenizer = None
_backend = None
_backend_name = "minicpm"
_qwen_remote_config = deepcopy(_QWEN_REMOTE_DEFAULTS)
_qwen_transformers_config = deepcopy(_QWEN_TRANSFORMERS_DEFAULTS)
_gemma_transformers_config = deepcopy(_GEMMA_TRANSFORMERS_DEFAULTS)
_qwen_llamacpp_config = deepcopy(_QWEN_LLAMACPP_DEFAULTS)
_gemma_llamacpp_config = deepcopy(_GEMMA_LLAMACPP_DEFAULTS)

logger = get_trace_logger()

_VALID_BACKENDS = ("minicpm", "qwen_remote", "qwen_transformers", "gemma_transformers", "qwen_llamacpp", "gemma_llamacpp")

# Quantization mode: "none" (bf16, ~19 GB), "int8" (~10-12 GB), "int4" (NF4, ~11 GB)
_quantization = "none"

# Auto-update: when True, check HuggingFace for newer model on each launch.
# When False, use local cache only (no network check) ΓÇö good for slow connections.
_auto_update = True


def set_backend(name: str) -> None:
    """Select the active model backend before the model is loaded."""
    global _backend, _backend_name

    if name not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend: {name!r}. Use one of {_VALID_BACKENDS}.")
    if _model is not None and name != _backend_name:
        raise RuntimeError("Cannot switch backends after the model is loaded.")

    _backend_name = name
    _backend = None


def get_backend_name() -> str:
    """Return the currently configured backend name."""
    return _backend_name


def get_backend():
    """Return the active backend adapter (singleton)."""
    global _backend

    if _backend is None:
        if _backend_name == "minicpm":
            from tools.model.backends.minicpm_backend import MiniCPMBackend

            _backend = MiniCPMBackend()
        elif _backend_name == "qwen_remote":
            from tools.model.backends.qwen_remote_backend import QwenRemoteBackend

            _backend = QwenRemoteBackend()
        elif _backend_name == "qwen_transformers":
            from tools.model.backends.qwen_transformers_backend import QwenTransformersBackend

            _backend = QwenTransformersBackend()
        elif _backend_name == "gemma_transformers":
            from tools.model.backends.gemma_transformers_backend import GemmaTransformersBackend

            _backend = GemmaTransformersBackend()
        elif _backend_name == "qwen_llamacpp":
            from tools.model.backends.qwen_llamacpp_backend import QwenLlamaCppBackend

            _backend = QwenLlamaCppBackend()
        elif _backend_name == "gemma_llamacpp":
            from tools.model.backends.gemma_llamacpp_backend import GemmaLlamaCppBackend

            _backend = GemmaLlamaCppBackend()
        else:
            raise RuntimeError(f"Unsupported backend: {_backend_name!r}")
    return _backend


def warmup_backend() -> dict:
    """Run an optional backend warmup pass and return its summary."""
    return get_backend().warmup()


def get_backend_capabilities() -> dict:
    """Return the active backend capability flags and media contract."""
    return get_backend().get_capabilities()


def get_backend_status() -> dict:
    """Return a summary of the active backend, configuration, and reachability."""
    capabilities = get_backend_capabilities()
    status = {
        "backend": _backend_name,
        "capabilities": capabilities,
        "configured_model_name": capabilities.get("model_name") or "",
    }

    if _backend_name == "minicpm":
        status.update({
            "ready": _model is not None,
            "summary": f"Local MiniCPM backend ({MODEL_NAME}, quantization={_quantization})",
            "transport": "in-process",
            "server_status": "local",
        })
        return status

    if _backend_name == "qwen_remote":
        from tools.model.clients import QwenOmniClient, QwenOmniClientConfig

        cfg = get_qwen_remote_config()
        backend = get_backend()
        client = QwenOmniClient(QwenOmniClientConfig(**cfg))
        health = client.healthcheck()
        status.update({
            "ready": health.get("ok", False),
            "summary": f"Remote Qwen backend ({cfg['base_url']})",
            "transport": client.get_transport_contract(),
            "health": health,
            "configured_model_name": cfg.get("model_name") or "",
            "server_status": "ok" if health.get("ok", False) else "unreachable",
            "server_url": cfg.get("base_url") or "",
        })
        if hasattr(backend, "get_runtime_status"):
            status["runtime"] = backend.get_runtime_status()
        return status

    if _backend_name == "qwen_transformers":
        cfg = get_qwen_transformers_config()
        backend = _backend
        status.update({
            "ready": _backend is not None,
            "summary": f"Local Qwen Transformers backend ({cfg['checkpoint']})",
            "transport": "local_transformers",
            "server_status": "local",
            "configured_model_name": cfg.get("checkpoint") or "",
        })
        if hasattr(backend, "get_runtime_status"):
            status["runtime"] = backend.get_runtime_status()
        return status

    if _backend_name == "gemma_transformers":
        cfg = get_gemma_transformers_config()
        backend = _backend
        status.update({
            "ready": _backend is not None,
            "summary": f"Local Gemma Transformers backend ({cfg['checkpoint']})",
            "transport": "local_transformers",
            "server_status": "local",
            "configured_model_name": cfg.get("checkpoint") or "",
        })
        if hasattr(backend, "get_runtime_status"):
            status["runtime"] = backend.get_runtime_status()
        return status

    if _backend_name == "qwen_llamacpp":
        cfg = get_qwen_llamacpp_config()
        backend = _backend
        status.update({
            "ready": _backend is not None,
            "summary": f"Local Qwen llama.cpp backend ({cfg['name']})",
            "transport": "local_llamacpp_cli",
            "server_status": "local",
            "configured_model_name": cfg.get("name") or "",
        })
        if hasattr(backend, "get_runtime_status"):
            status["runtime"] = backend.get_runtime_status()
        return status

    if _backend_name == "gemma_llamacpp":
        cfg = get_gemma_llamacpp_config()
        backend = _backend
        status.update({
            "ready": _backend is not None,
            "summary": f"Local Gemma llama.cpp backend ({cfg['name']})",
            "transport": "local_llamacpp_cli",
            "server_status": "local",
            "configured_model_name": cfg.get("name") or "",
        })
        if hasattr(backend, "get_runtime_status"):
            status["runtime"] = backend.get_runtime_status()
        return status

    status.update({
        "ready": False,
        "summary": f"Unknown backend: {_backend_name}",
        "server_status": "unknown",
    })
    return status


def summarize_backend_status(status: Optional[dict] = None) -> str:
    """Render the active model/transport/server summary as a single line."""
    if status is None:
        status = get_backend_status()

    capabilities = status.get("capabilities", {})
    model_name = capabilities.get("model_name") or status.get("configured_model_name", "") or "n/a"
    transport = status.get("transport")
    if isinstance(transport, dict):
        transport_name = transport.get("protocol", "n/a")
    else:
        transport_name = transport or "n/a"

    server_status = status.get("server_status")
    if not server_status:
        health = status.get("health")
        if health:
            server_status = "ok" if health.get("ok") else "unreachable"
        else:
            server_status = "n/a"

    return f"Model: {model_name} | Transport: {transport_name} | Server: {server_status}"


def format_backend_status(status: Optional[dict] = None) -> str:
    """Render backend status as a compact multiline string for UI and CLI output."""
    if status is None:
        status = get_backend_status()

    capabilities = status.get("capabilities", {})
    transport = status.get("transport")
    summary_line = summarize_backend_status(status)
    model_name = summary_line.split(" | ")[0].split(": ", 1)[1]
    transport_name = summary_line.split(" | ")[1].split(": ", 1)[1]
    server_status = summary_line.split(" | ")[2].split(": ", 1)[1]

    lines = [
        f"Backend: {status.get('backend', 'unknown')}",
        f"Model: {model_name}",
        f"Transport: {transport_name}",
        f"Server: {server_status}",
        f"Summary: {status.get('summary', 'n/a')}",
        f"Streaming text: {capabilities.get('supports_streaming_text', False)}",
        f"Streaming audio: {capabilities.get('supports_streaming_audio', False)}",
    ]

    health = status.get("health")
    if health:
        if status.get("server_url"):
            lines.append(f"Server URL: {status['server_url']}")
        if health.get("status_code") is not None:
            lines.append(f"Health status code: {health['status_code']}")
        if health.get("error"):
            lines.append(f"Health error: {health['error']}")

    if isinstance(transport, dict):
        lines.append(f"Realtime stub: {transport.get('realtime_ws_url', 'n/a')}")

    runtime = status.get("runtime")
    if isinstance(runtime, dict):
        mode = runtime.get("last_audio_delivery_mode")
        if mode:
            lines.append(f"Last audio delivery: {mode}")
            lines.append(f"Last text chars: {runtime.get('last_text_chars', 0)}")
            lines.append(f"Last audio chunks: {runtime.get('last_stream_audio_chunks', 0)}")

    return "\n".join(lines)


def set_qwen_remote_config(
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    timeout_s: Optional[float] = None,
    endpoints: Optional[dict] = None,
) -> None:
    """Configure the remote Qwen backend connection settings."""
    global _backend

    if base_url is not None:
        _qwen_remote_config["base_url"] = base_url.rstrip("/")
    if api_key is not None:
        _qwen_remote_config["api_key"] = api_key or None
    if model_name is not None:
        _qwen_remote_config["model_name"] = model_name
    if timeout_s is not None:
        _qwen_remote_config["timeout_s"] = float(timeout_s)
    if endpoints is not None:
        merged_endpoints = dict(_QWEN_REMOTE_DEFAULTS["endpoints"])
        merged_endpoints.update(endpoints)
        _qwen_remote_config["endpoints"] = merged_endpoints

    if _backend_name == "qwen_remote":
        _backend = None


def get_qwen_remote_config() -> dict:
    """Return a copy of the remote Qwen backend connection settings."""
    return deepcopy(_qwen_remote_config)


def set_qwen_transformers_config(
    *,
    checkpoint: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    speaker: Optional[str] = None,
    use_audio_in_video: Optional[bool] = None,
    local_files_only: Optional[bool] = None,
) -> None:
    """Configure the local Qwen Transformers backend before model load."""
    global _backend

    if checkpoint is not None:
        _qwen_transformers_config["checkpoint"] = checkpoint
    if device_map is not None:
        _qwen_transformers_config["device_map"] = device_map
    if torch_dtype is not None:
        _qwen_transformers_config["torch_dtype"] = torch_dtype
    if speaker is not None:
        _qwen_transformers_config["speaker"] = speaker
    if use_audio_in_video is not None:
        _qwen_transformers_config["use_audio_in_video"] = bool(use_audio_in_video)
    if local_files_only is not None:
        _qwen_transformers_config["local_files_only"] = bool(local_files_only)

    if attn_implementation is not None or attn_implementation is None:
        _qwen_transformers_config["attn_implementation"] = attn_implementation

    if _backend_name == "qwen_transformers":
        _backend = None


def get_qwen_transformers_config() -> dict:
    """Return a copy of the local Qwen Transformers backend settings."""
    return deepcopy(_qwen_transformers_config)


def set_gemma_transformers_config(
    *,
    checkpoint: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    attn_implementation: Optional[str] = None,
    speech_backend: Optional[str] = None,
    use_audio_in_video: Optional[bool] = None,
    local_files_only: Optional[bool] = None,
    video_backend: Optional[str] = None,
) -> None:
    """Configure the local Gemma Transformers backend before model load."""
    global _backend

    if checkpoint is not None:
        _gemma_transformers_config["checkpoint"] = checkpoint
    if device_map is not None:
        _gemma_transformers_config["device_map"] = device_map
    if torch_dtype is not None:
        _gemma_transformers_config["torch_dtype"] = torch_dtype
    if speech_backend is not None:
        _gemma_transformers_config["speech_backend"] = speech_backend
    if use_audio_in_video is not None:
        _gemma_transformers_config["use_audio_in_video"] = bool(use_audio_in_video)
    if local_files_only is not None:
        _gemma_transformers_config["local_files_only"] = bool(local_files_only)
    if video_backend is not None:
        _gemma_transformers_config["video_backend"] = video_backend

    if attn_implementation is not None or attn_implementation is None:
        _gemma_transformers_config["attn_implementation"] = attn_implementation

    if _backend_name == "gemma_transformers":
        _backend = None


def get_gemma_transformers_config() -> dict:
    """Return a copy of the local Gemma Transformers backend settings."""
    return deepcopy(_gemma_transformers_config)


def set_qwen_llamacpp_config(
    *,
    name: Optional[str] = None,
    llama_root: Optional[str] = None,
    cli_path: Optional[str] = None,
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    n_gpu_layers: Optional[int] = None,
    flash_attn: Optional[bool] = None,
    context_length: Optional[int] = None,
    use_jinja: Optional[bool] = None,
    speech_backend: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> None:
    """Configure the local Qwen llama.cpp backend before first use."""
    global _backend

    if name is not None:
        _qwen_llamacpp_config["name"] = name
    if llama_root is not None:
        _qwen_llamacpp_config["llama_root"] = llama_root
    if cli_path is not None:
        _qwen_llamacpp_config["cli_path"] = cli_path
    if model_path is not None:
        _qwen_llamacpp_config["model_path"] = model_path
    if mmproj_path is not None:
        _qwen_llamacpp_config["mmproj_path"] = mmproj_path
    if n_gpu_layers is not None:
        _qwen_llamacpp_config["n_gpu_layers"] = int(n_gpu_layers)
    if flash_attn is not None:
        _qwen_llamacpp_config["flash_attn"] = bool(flash_attn)
    if context_length is not None:
        _qwen_llamacpp_config["context_length"] = int(context_length)
    if use_jinja is not None:
        _qwen_llamacpp_config["use_jinja"] = bool(use_jinja)
    if speech_backend is not None:
        _qwen_llamacpp_config["speech_backend"] = str(speech_backend)
    if timeout_s is not None:
        _qwen_llamacpp_config["timeout_s"] = float(timeout_s)

    if _backend_name == "qwen_llamacpp":
        _backend = None


def get_qwen_llamacpp_config() -> dict:
    """Return a copy of the local Qwen llama.cpp backend settings."""
    return deepcopy(_qwen_llamacpp_config)


def set_gemma_llamacpp_config(
    *,
    name: Optional[str] = None,
    llama_root: Optional[str] = None,
    cli_path: Optional[str] = None,
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    n_gpu_layers: Optional[int] = None,
    flash_attn: Optional[bool] = None,
    context_length: Optional[int] = None,
    use_jinja: Optional[bool] = None,
    speech_backend: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> None:
    """Configure the local Gemma llama.cpp backend before first use."""
    global _backend

    if name is not None:
        _gemma_llamacpp_config["name"] = name
    if llama_root is not None:
        _gemma_llamacpp_config["llama_root"] = llama_root
    if cli_path is not None:
        _gemma_llamacpp_config["cli_path"] = cli_path
    if model_path is not None:
        _gemma_llamacpp_config["model_path"] = model_path
    if mmproj_path is not None:
        _gemma_llamacpp_config["mmproj_path"] = mmproj_path
    if n_gpu_layers is not None:
        _gemma_llamacpp_config["n_gpu_layers"] = int(n_gpu_layers)
    if flash_attn is not None:
        _gemma_llamacpp_config["flash_attn"] = bool(flash_attn)
    if context_length is not None:
        _gemma_llamacpp_config["context_length"] = int(context_length)
    if use_jinja is not None:
        _gemma_llamacpp_config["use_jinja"] = bool(use_jinja)
    if speech_backend is not None:
        _gemma_llamacpp_config["speech_backend"] = str(speech_backend)
    if timeout_s is not None:
        _gemma_llamacpp_config["timeout_s"] = float(timeout_s)

    if _backend_name == "gemma_llamacpp":
        _backend = None


def get_gemma_llamacpp_config() -> dict:
    """Return a copy of the local Gemma llama.cpp backend settings."""
    return deepcopy(_gemma_llamacpp_config)


def _build_minicpm_tts_prompt(text: str) -> str:
    spoken_text = (text or "").strip()
    return f"Speak this exactly, but convert any numbers to the words that represent them: {spoken_text}"


def stream_text_to_speech_with_minicpm(
    text: str,
    *,
    voice_ref: Optional[np.ndarray] = None,
    temperature: float = 0.2,
    repetition_penalty: float = 1.05,
    top_p: float = 0.9,
    top_k: int = 1,
    trace_context: Optional[dict] = None,
    source_backend: Optional[str] = None,
) -> Generator[tuple[Optional[np.ndarray], str], None, None]:
    """Stream MiniCPM speech for an exact assistant text turn without temp-file buffering."""
    spoken_text = (text or "").strip()
    if not spoken_text:
        return

    request_id = "n/a"
    if isinstance(trace_context, dict):
        request_id = str(trace_context.get("request_id") or "n/a")
    load_state = "reuse" if _model is not None else "cold-start"
    logger.info(
        "trace_id=%s stage=tts event=minicpm_tts_prepare source_backend=%s load_state=%s text_chars=%d voice_ref=%s",
        request_id,
        source_backend or "unknown",
        load_state,
        len(spoken_text),
        bool(voice_ref is not None),
    )

    tts_messages = [{
        "role": "user",
        "content": [_build_minicpm_tts_prompt(spoken_text)],
    }]

    yield from _minicpm_chat_streaming(
        messages=tts_messages,
        voice_ref=voice_ref,
        generate_audio=True,
        temperature=temperature,
        max_new_tokens=max(32, len(spoken_text.split()) * 6),
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=False,
    )


def synthesize_text_with_minicpm(
    text: str,
    *,
    voice_ref: Optional[np.ndarray] = None,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.2,
    repetition_penalty: float = 1.05,
    top_p: float = 0.9,
    top_k: int = 1,
    trace_context: Optional[dict] = None,
    source_backend: Optional[str] = None,
) -> dict:
    """Synthesize exact text with MiniCPM using streamed chunks collected in memory."""
    spoken_text = (text or "").strip()
    if not spoken_text:
        return {"text": "", "audio": None, "audio_path": output_audio_path, "sample_rate": None}

    collected_audio: list[np.ndarray] = []
    delivered_text = ""
    for audio_chunk, text_chunk in stream_text_to_speech_with_minicpm(
        spoken_text,
        voice_ref=voice_ref,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        trace_context=trace_context,
        source_backend=source_backend,
    ):
        if text_chunk:
            delivered_text += text_chunk
        if audio_chunk is not None and len(audio_chunk) > 0:
            collected_audio.append(audio_chunk.astype(np.float32, copy=False))

    full_audio = np.concatenate(collected_audio).astype(np.float32, copy=False) if collected_audio else None
    if output_audio_path and full_audio is not None:
        import soundfile as sf

        sf.write(output_audio_path, full_audio, _STREAMING_SR)

    return {
        "text": delivered_text or spoken_text,
        "audio": full_audio,
        "audio_path": output_audio_path,
        "sample_rate": _STREAMING_SR if full_audio is not None else None,
    }


def transcribe_audio_array_with_minicpm(
    audio: np.ndarray,
    *,
    prompt: str = "Transcribe this audio completely and verbatim. Return only the transcript text.",
) -> str:
    """Use MiniCPM to turn an in-memory user audio turn into plain text."""
    audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
    result = _minicpm_chat(
        messages=[{"role": "user", "content": [audio_array, prompt]}],
        generate_audio=False,
        temperature=0.1,
        max_new_tokens=256,
        repetition_penalty=1.02,
        top_p=0.8,
        top_k=1,
        enable_thinking=False,
    )
    return (result.get("text") or "").strip()


def _set_minicpm_quantization(mode: str) -> None:
    """Set the quantization mode before the model is loaded.

    Must be called before the first get_model() call.  Has no effect if the
    model is already loaded (singleton).

    Args:
        mode: "none" (bf16), "int8" (bitsandbytes 8-bit), or "int4" (bitsandbytes NF4).
    """
    global _quantization
    valid = ("none", "int8", "int4")
    if mode not in valid:
        raise ValueError(f"Unknown quantization mode: {mode!r}. Use one of {valid}.")
    _quantization = mode


def set_quantization(mode: str) -> None:
    """Set the quantization mode before the model is loaded."""
    if _backend_name == "qwen_transformers":
        get_backend().set_quantization(mode)
        return
    if _backend_name in {"qwen_llamacpp", "gemma_llamacpp"}:
        get_backend().set_quantization(mode)
        return
    _set_minicpm_quantization(mode)


def _set_minicpm_auto_update(enabled: bool) -> None:
    """Set whether to check HuggingFace for model updates on launch.

    When False, uses local cache only (local_files_only=True).
    Must be called before the first get_model() call.
    """
    global _auto_update
    _auto_update = enabled


def set_auto_update(enabled: bool) -> None:
    """Set whether to check HuggingFace for model updates on launch."""
    if _backend_name == "qwen_transformers":
        get_backend().set_auto_update(enabled)
        return
    if _backend_name in {"qwen_llamacpp", "gemma_llamacpp"}:
        get_backend().set_auto_update(enabled)
        return
    _set_minicpm_auto_update(enabled)

# Turn counter for diagnostics
_turn_count = 0

# Duration (in samples @ 24kHz) for fade-in on generated audio to smooth
# the HiFT vocoder's cold-start artifact (160ms zeros + no crossfade).
_FADE_IN_SAMPLES = 2400  # 100ms at 24kHz


class _DynamicCacheTensorListView:
    """List-like view that exposes DynamicCache layers via the old key/value API."""

    def __init__(self, cache, attr_name: str):
        self._cache = cache
        self._attr_name = attr_name

    def __len__(self) -> int:
        return len(self._cache.layers)

    def __iter__(self):
        for layer in self._cache.layers:
            yield getattr(layer, self._attr_name)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [getattr(layer, self._attr_name) for layer in self._cache.layers[index]]
        return getattr(self._cache.layers[index], self._attr_name)

    def __setitem__(self, index, value) -> None:
        if isinstance(index, slice):
            layers = self._cache.layers[index]
            if len(layers) != len(value):
                raise ValueError("slice assignment must preserve cache layer count")
            for layer, tensor in zip(layers, value):
                setattr(layer, self._attr_name, tensor)
                layer.is_initialized = tensor is not None
                if tensor is not None:
                    layer.device = tensor.device
                    layer.dtype = tensor.dtype
            return

        layer = self._cache.layers[index]
        setattr(layer, self._attr_name, value)
        layer.is_initialized = value is not None
        if value is not None:
            layer.device = value.device
            layer.dtype = value.dtype

    def __bool__(self) -> bool:
        return bool(len(self))


def _dynamic_cache_get_tensors(cache, attr_name: str) -> _DynamicCacheTensorListView:
    return _DynamicCacheTensorListView(cache, attr_name)


def _dynamic_cache_set_tensors(cache, attr_name: str, tensors) -> None:
    from transformers.cache_utils import DynamicLayer

    cache.layers = []
    for tensor in tensors:
        layer = DynamicLayer()
        setattr(layer, attr_name, tensor)
        other_attr = "values" if attr_name == "keys" else "keys"
        setattr(layer, other_attr, None)
        layer.is_initialized = tensor is not None
        if tensor is not None:
            layer.device = tensor.device
            layer.dtype = tensor.dtype
        cache.layers.append(layer)


def _patch_transformers_minicpm_compat() -> None:
    """Restore older cache/attention APIs expected by MiniCPM remote code."""
    from transformers.cache_utils import DynamicCache
    from transformers.cache_utils import EncoderDecoderCache
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.whisper.modeling_whisper import WhisperAttention

    sample_cache = DynamicCache()

    if not hasattr(sample_cache, "key_cache"):
        DynamicCache.key_cache = property(  # type: ignore[attr-defined]
            lambda self: _dynamic_cache_get_tensors(self, "keys"),
            lambda self, tensors: _dynamic_cache_set_tensors(self, "keys", tensors),
        )

    if not hasattr(sample_cache, "value_cache"):
        DynamicCache.value_cache = property(  # type: ignore[attr-defined]
            lambda self: _dynamic_cache_get_tensors(self, "values"),
            lambda self, tensors: _dynamic_cache_set_tensors(self, "values", tensors),
        )

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(  # type: ignore[attr-defined]
            lambda self: getattr(self, "_seen_tokens", self.get_seq_length()),
            lambda self, value: setattr(self, "_seen_tokens", int(value)),
        )

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *args, **kwargs: self.get_seq_length()  # type: ignore[attr-defined]

    if not hasattr(LlamaConfig, "rope_theta"):
        def _get_rope_theta(self):
            rope_parameters = getattr(self, "rope_parameters", None) or {}
            return rope_parameters.get("rope_theta", 10000.0)

        def _set_rope_theta(self, value):
            rope_parameters = dict(getattr(self, "rope_parameters", None) or {})
            rope_parameters["rope_theta"] = value
            self.rope_parameters = rope_parameters

        LlamaConfig.rope_theta = property(_get_rope_theta, _set_rope_theta)  # type: ignore[attr-defined]

    original_cache_getitem = getattr(EncoderDecoderCache, "__getitem__", None)
    if not getattr(original_cache_getitem, "_omnichat_minicpm_compat", False):

        def _compat_cache_getitem(self, layer_idx: int):
            if layer_idx < len(self):
                self_layer = self.self_attention_cache.layers[layer_idx]
                cross_keys = None
                cross_values = None
                if layer_idx < len(self.cross_attention_cache.layers):
                    cross_layer = self.cross_attention_cache.layers[layer_idx]
                    cross_keys = cross_layer.keys
                    cross_values = cross_layer.values
                return (self_layer.keys, self_layer.values, cross_keys, cross_values)
            if original_cache_getitem is not None:
                return original_cache_getitem(self, layer_idx)
            raise IndexError(layer_idx)

        _compat_cache_getitem._omnichat_minicpm_compat = True  # type: ignore[attr-defined]
        EncoderDecoderCache.__getitem__ = _compat_cache_getitem

    if not getattr(WhisperAttention.forward, "_omnichat_minicpm_compat", False):
        original_forward = WhisperAttention.forward

        def _compat_forward(self, *args, **kwargs):
            result = original_forward(self, *args, **kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                attn_output, attn_weights = result
                past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value"))
                return attn_output, attn_weights, past_key_values
            return result

        _compat_forward._omnichat_minicpm_compat = True  # type: ignore[attr-defined]
        WhisperAttention.forward = _compat_forward


def _ensure_minicpm_tts_sampling_defaults(config) -> None:
    """Restore TTS sampling defaults expected by MiniCPM remote code."""
    tts_config = getattr(config, "tts_config", None)
    if tts_config is None:
        return

    if not hasattr(tts_config, "top_p"):
        tts_config.top_p = 0.8
    if not hasattr(tts_config, "top_k"):
        tts_config.top_k = 50
    if not hasattr(tts_config, "repetition_penalty"):
        tts_config.repetition_penalty = 1.05


def _ensure_generation_cache_position(
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
):
    """Synthesize cache_position for MiniCPM remote code under newer Transformers."""
    if cache_position is not None:
        return cache_position

    past_length = 0
    if past_key_values is not None:
        if hasattr(past_key_values, "get_seq_length"):
            try:
                past_length = int(past_key_values.get_seq_length())
            except Exception:
                past_length = 0
        else:
            try:
                past_length = int(past_key_values[0][0].shape[2])
            except Exception:
                past_length = 0

    device = None
    seq_len = None
    if input_ids is not None:
        device = input_ids.device
        seq_len = max(1, int(input_ids.shape[1]))
    elif inputs_embeds is not None:
        device = inputs_embeds.device
        seq_len = max(1, int(inputs_embeds.shape[1]))
    elif attention_mask is not None:
        device = attention_mask.device
        mask_len = int(attention_mask.shape[1])
        seq_len = max(1, mask_len - past_length) if past_length else mask_len

    if device is None or seq_len is None:
        return cache_position

    return torch.arange(past_length, past_length + seq_len, device=device)


def _patch_minicpm_model_class_compat(model_name: str, config) -> None:
    """Patch remote MiniCPM model classes for newer Transformers expectations."""
    auto_map = getattr(config, "auto_map", None) or {}
    class_ref = auto_map.get("AutoModel")
    if not class_ref:
        return

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(class_ref, model_name)
    if not hasattr(model_class, "all_tied_weights_keys"):
        model_class.all_tied_weights_keys = {}
    if not hasattr(model_class, "_tied_weights_keys"):
        model_class._tied_weights_keys = []

    original_prepare_inputs = getattr(model_class, "prepare_inputs_for_generation", None)
    if original_prepare_inputs is not None and not getattr(
        original_prepare_inputs,
        "_omnichat_minicpm_cache_position_compat",
        False,
    ):

        def _compat_prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
        ):
            cache_position = _ensure_generation_cache_position(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )
            return original_prepare_inputs(
                self,
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )

        _compat_prepare_inputs_for_generation._omnichat_minicpm_cache_position_compat = True  # type: ignore[attr-defined]
        model_class.prepare_inputs_for_generation = _compat_prepare_inputs_for_generation


def _get_minicpm_model():
    """Load MiniCPM-o 4.5 (singleton ΓÇö loads once, reuses on subsequent calls).

    Respects the quantization mode set via set_quantization():
      - "none"  ΓÇö bf16, ~19 GB VRAM (default)
      - "int8"  ΓÇö bitsandbytes 8-bit, ~10-12 GB VRAM
      - "int4"  ΓÇö bitsandbytes NF4 with double quantization, ~11 GB VRAM
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    _patch_transformers_minicpm_compat()

    model_name = MODEL_NAME

    quant_label = {"none": "bf16 (full precision)", "int8": "int8 (bitsandbytes)", "int4": "int4 (bitsandbytes NF4)"}
    print(f"Loading {model_name}...")
    print(f"  Precision: {quant_label.get(_quantization, _quantization)}")
    print(f"  Auto-update: {'on' if _auto_update else 'off (local cache only)'}")
    print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")

    _local_only = not _auto_update

    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=_local_only
    )

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=_local_only,
    )
    _ensure_minicpm_tts_sampling_defaults(config)
    _patch_minicpm_model_class_compat(model_name, config)

    load_kwargs = {
        "config": config,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
        "init_vision": True,
        "init_audio": True,
        "init_tts": True,
        "local_files_only": _local_only,
    }

    if _quantization == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            # Only quantize LLM transformer layers (llm.model.layers.*).
            # Skip ALL non-LLM modules to avoid two problems:
            #
            # 1. SCB AttributeError: bitsandbytes Linear8bitLt._save_to_state_dict
            #    calls getattr(self.weight, "SCB") without a default. If any module's
            #    weight is a plain nn.Parameter (not Int8Params), this crashes.
            #    The TTS head_code modules use weight_norm (parametrize API), which
            #    is fundamentally incompatible with Linear8bitLt replacement.
            #
            # 2. Non-LLM modules (vision encoder, audio encoder, TTS, resampler,
            #    projection layers) are small relative to the LLM and benefit
            #    little from INT8 quantization.  Keeping them in bf16 preserves
            #    quality for audio/vision processing at negligible VRAM cost.
            #
            # The int4 (NF4) config uses the same skip list.
            llm_int8_skip_modules=[
                "lm_head",
                "apm",                    # Whisper audio encoder
                "tts",                    # TTS decoder (has weight_norm layers)
                "vpm",                    # SigLIP vision encoder
                "resampler",              # vision resampler
                "audio_projection_layer", # audio-to-LLM projector
                "audio_avg_pooler",       # audio pooling layer
            ],
        )
        load_kwargs["device_map"] = "auto"
    elif _quantization == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Same skip list as int8 ΓÇö only quantize LLM transformer layers.
            # Non-LLM modules stay in bf16 for audio/vision quality.
            llm_int8_skip_modules=[
                "lm_head",
                "apm",                    # Whisper audio encoder
                "tts",                    # TTS decoder (has weight_norm layers)
                "vpm",                    # SigLIP vision encoder
                "resampler",              # vision resampler
                "audio_projection_layer", # audio-to-LLM projector
                "audio_avg_pooler",       # audio pooling layer
            ],
        )
        load_kwargs["device_map"] = "auto"

    try:
        _model = AutoModel.from_pretrained(model_name, **load_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" not in message or not any(
            key in message for key in ("init_vision", "init_audio", "init_tts")
        ):
            raise
        compat_kwargs = dict(load_kwargs)
        compat_kwargs.pop("init_vision", None)
        compat_kwargs.pop("init_audio", None)
        compat_kwargs.pop("init_tts", None)
        _model = AutoModel.from_pretrained(model_name, **compat_kwargs)

    if not hasattr(_model, "all_tied_weights_keys"):
        _model.all_tied_weights_keys = {}
    if not hasattr(_model, "_tied_weights_keys"):
        _model._tied_weights_keys = []

    if not getattr(_model.prepare_generation_config, "_omnichat_minicpm_compat", False):
        original_prepare_generation_config = _model.prepare_generation_config

        def _compat_prepare_generation_config(do_sample, max_new_tokens=50, min_new_tokens=0, **kwargs):
            generation_config = original_prepare_generation_config(
                do_sample,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                **kwargs,
            )
            passthrough_keys = {
                "output_attentions",
                "output_hidden_states",
                "return_dict_in_generate",
                "use_cache",
            }
            for key in passthrough_keys:
                if key in kwargs:
                    generation_config[key] = kwargs[key]
            return generation_config

        _compat_prepare_generation_config._omnichat_minicpm_compat = True  # type: ignore[attr-defined]
        _model.prepare_generation_config = _compat_prepare_generation_config

    generation_model = getattr(_model, "llm", None)
    if generation_model is None:
        generation_model = _model

    prepare_inputs_for_generation = getattr(generation_model, "prepare_inputs_for_generation", None)
    if prepare_inputs_for_generation is not None and not getattr(
        prepare_inputs_for_generation,
        "_omnichat_minicpm_cache_position_compat",
        False,
    ):
        original_prepare_inputs_for_generation = prepare_inputs_for_generation

        def _compat_instance_prepare_inputs_for_generation(
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
        ):
            cache_position = _ensure_generation_cache_position(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
            )
            return original_prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                position_ids=position_ids,
                use_cache=use_cache,
                **kwargs,
            )

        _compat_instance_prepare_inputs_for_generation._omnichat_minicpm_cache_position_compat = True  # type: ignore[attr-defined]
        generation_model.prepare_inputs_for_generation = _compat_instance_prepare_inputs_for_generation

    if _quantization in ("int8", "int4"):
        # BNB with device_map="auto" places the model ΓÇö do NOT call .cuda()
        _model.eval()
    else:
        _model.eval().cuda()

    _model.init_tts()

    # Pre-initialize Token2wav cache with a short silent prompt so the default
    # voice TTS path works.  Without this, the first TTS call with no voice_ref
    # crashes in Token2wav._prepare_prompt(None) ΓåÆ TypeError("Invalid file: None")
    # because prompt_wav is None and cache hasn't been built yet.
    _init_audio = np.zeros(16000, dtype=np.float32)  # 1s silence at 16kHz
    _model.init_token2wav_cache(_init_audio)

    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM used: {vram_used:.1f} GB")
    print("  Model loaded and ready.")
    return _model, _tokenizer


def _reset_for_generation(model, generate_audio: bool, output_audio_path: Optional[str] = None):
    """Clear model state before a new generation call."""
    # Delete stale output audio so a failed TTS can't return the previous voice
    if output_audio_path:
        Path(output_audio_path).unlink(missing_ok=True)

    # Full session reset ΓÇö clears KV caches, round history, flags, etc.
    # Note: tts_last_turn_tokens is never actually populated by the model's
    # non-streaming path (TTSStreamingGenerator doesn't update it), so there's
    # nothing to preserve.
    model.reset_session()

    # Free stale GPU tensors that might interfere with the next generation
    torch.cuda.empty_cache()

    # NOTE: We do NOT clear Token2wav's internal cache (tok2wav.cache) here.
    # Token2wav.cache holds vocoder features for the current voice ΓÇö it is safe
    # to reuse across turns and in fact MUST be preserved, because the model's
    # default-voice path passes prompt_wav=None and Token2wav._prepare_prompt(None)
    # crashes with TypeError("Invalid file: None").  The model's own
    # reset_session() handles its separate token2wav_cache for streaming.
    # Voice switching (when voice_ref changes) is handled by the model's
    # get_sys_prompt() which embeds the ref audio, causing audio_prompt to be
    # non-None on the next _generate_speech_non_streaming call.


def _apply_fade_in(audio: np.ndarray, n_samples: int = _FADE_IN_SAMPLES) -> np.ndarray:
    """Apply a linear fade-in to smooth the HiFT vocoder's cold-start artifact."""
    if len(audio) <= n_samples:
        return audio
    fade = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    audio = audio.copy()
    audio[:n_samples] *= fade
    return audio


# ΓöÇΓöÇ Audio leveling ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
#
# Configurable via args/settings.yaml ΓåÆ audio.leveling.
# All levels in dBFS (0 = full scale).  Converted to linear at load time.
#
# Two processing stages:
#   1. Voice ref input  ΓÇö static RMS normalizer (short clips, one gain value)
#   2. Output audio     ΓÇö windowed RMS compressor with attack/release envelope
#
# Defaults match the YAML shipped with the project.  If the YAML is missing
# or the section is absent, these kick in so the code never crashes.

_LEVELING_DEFAULTS = {
    "enabled": True,
    # Voice reference (static normalizer)
    "ref_target_dbfs": -26.0,
    "ref_max_gain_db": 20.0,
    # Output compressor
    "output_threshold_dbfs": -20.0,   # level where compression engages
    "output_ratio": 3.0,             # compression ratio (3:1)
    "output_attack_ms": 15.0,        # how fast gain reduces on loud signal
    "output_release_ms": 150.0,      # how fast gain recovers after signal drops
    "output_knee_db": 6.0,           # soft knee width (0 = hard knee)
    "output_makeup_db": 4.0,         # post-compression boost
    "output_max_gain_db": 20.0,      # safety cap on total gain
    # Peak limiter (shared)
    "peak_ceiling_dbfs": -0.1,
}

# Cached config (loaded once on first use)
_leveling_cfg: dict | None = None


def _dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS to linear amplitude (0 dBFS = 1.0)."""
    return 10.0 ** (dbfs / 20.0)


def _db_to_ratio(db: float) -> float:
    """Convert a dB gain value to a linear ratio (20 dB = 10x)."""
    return 10.0 ** (db / 20.0)


def _linear_to_dbfs(linear: float) -> float:
    """Convert linear amplitude to dBFS.  Clamps to -120 dBFS for silence."""
    if linear < 1e-6:
        return -120.0
    return 20.0 * np.log10(linear)


def _get_leveling_config() -> dict:
    """Load leveling config from settings.yaml (cached after first call)."""
    global _leveling_cfg
    if _leveling_cfg is not None:
        return _leveling_cfg

    cfg = dict(_LEVELING_DEFAULTS)

    try:
        import yaml
        settings_path = Path(__file__).parent.parent.parent / "args" / "settings.yaml"
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
            leveling = settings.get("audio", {}).get("leveling", {})
            if leveling:
                for key in _LEVELING_DEFAULTS:
                    if key in leveling:
                        cfg[key] = leveling[key]
    except Exception as e:
        print(f"  [leveling] Warning: could not load settings, using defaults: {e}")

    # Pre-compute linear values for fast access
    cfg["_ref_target_linear"] = _dbfs_to_linear(cfg["ref_target_dbfs"])
    cfg["_ref_max_gain_linear"] = _db_to_ratio(cfg["ref_max_gain_db"])
    cfg["_output_threshold_linear"] = _dbfs_to_linear(cfg["output_threshold_dbfs"])
    cfg["_output_makeup_linear"] = _db_to_ratio(cfg["output_makeup_db"])
    cfg["_output_max_gain_linear"] = _db_to_ratio(cfg["output_max_gain_db"])
    cfg["_peak_ceiling_linear"] = _dbfs_to_linear(cfg["peak_ceiling_dbfs"])

    _leveling_cfg = cfg
    return cfg


def _normalize_rms(audio: np.ndarray, target_rms: float, max_gain: float, peak_ceiling: float) -> np.ndarray:
    """Static RMS normalizer ΓÇö single gain for the whole signal.

    Appropriate for short clips (voice references) where temporal dynamics
    don't matter.  NOT used for output audio ΓÇö see _compress_output().
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio  # silence ΓÇö don't amplify noise
    gain = target_rms / rms
    gain = min(gain, max_gain)
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > peak_ceiling:
        normalized = normalized * (peak_ceiling / peak)
    return normalized.astype(np.float32)


def _compress_output(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Windowed RMS compressor with attack/release envelope.

    Processes the signal through a proper dynamics curve:
      1. Compute windowed RMS envelope (~10ms windows)
      2. Convert to dB, apply compression curve (threshold + ratio + soft knee)
      3. Smooth the gain envelope with attack/release time constants
      4. Apply per-sample gain, then makeup gain
      5. Peak-limit to ceiling

    This preserves natural dynamics within the knee while controlling
    level differences between voices and utterances.
    """
    cfg = _get_leveling_config()
    if not cfg["enabled"]:
        return audio

    if len(audio) < 2:
        return audio

    threshold_db = cfg["output_threshold_dbfs"]
    ratio = cfg["output_ratio"]
    attack_ms = cfg["output_attack_ms"]
    release_ms = cfg["output_release_ms"]
    knee_db = cfg["output_knee_db"]
    makeup_linear = cfg["_output_makeup_linear"]
    max_gain = cfg["_output_max_gain_linear"]
    ceiling = cfg["_peak_ceiling_linear"]

    # ΓöÇΓöÇ Step 1: Windowed RMS envelope ΓöÇΓöÇ
    # ~10ms window for smooth envelope without losing transient detail
    window_samples = max(int(sample_rate * 0.010), 1)
    # Compute running mean of squared signal
    sq = audio.astype(np.float64) ** 2
    # Cumulative sum trick for fast windowed average
    cumsum = np.concatenate([[0.0], np.cumsum(sq)])
    # Pad with half-window lookahead for centered measurement
    windowed_mean = np.empty_like(sq)
    half_w = window_samples // 2
    for i in range(len(sq)):
        lo = max(0, i - half_w)
        hi = min(len(sq), i + half_w + 1)
        windowed_mean[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    env_rms = np.sqrt(windowed_mean).astype(np.float32)

    # ΓöÇΓöÇ Step 2: Convert to dB and apply compression curve ΓöÇΓöÇ
    # Use np.maximum to avoid log10(0) RuntimeWarning; values below 1e-6
    # are replaced with -120 dBFS anyway.
    env_db = np.where(
        env_rms > 1e-6,
        20.0 * np.log10(np.maximum(env_rms, 1e-10)),
        -120.0,
    )

    # Soft-knee compression curve:
    #   Below knee: no compression (gain_db = 0)
    #   In knee:    quadratic blend from 1:1 to ratio
    #   Above knee: full ratio compression
    half_knee = knee_db / 2.0
    knee_lo = threshold_db - half_knee
    knee_hi = threshold_db + half_knee

    gain_db = np.zeros_like(env_db)
    if knee_db > 0.01:
        # Below knee ΓÇö no compression
        # In knee ΓÇö quadratic interpolation
        in_knee = (env_db > knee_lo) & (env_db < knee_hi)
        x = env_db[in_knee] - knee_lo
        gain_db[in_knee] = -((1.0 - 1.0 / ratio) * x * x) / (2.0 * knee_db)
        # Above knee ΓÇö full compression
        above = env_db >= knee_hi
        gain_db[above] = (threshold_db - env_db[above]) * (1.0 - 1.0 / ratio)
    else:
        # Hard knee
        above = env_db > threshold_db
        gain_db[above] = (threshold_db - env_db[above]) * (1.0 - 1.0 / ratio)

    # ΓöÇΓöÇ Step 3: Smooth with attack/release time constants ΓöÇΓöÇ
    # Convert ms to per-sample smoothing coefficients
    attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0)) if attack_ms > 0 else 0.0
    release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000.0)) if release_ms > 0 else 0.0

    smoothed_db = np.empty_like(gain_db)
    smoothed_db[0] = gain_db[0]
    for i in range(1, len(gain_db)):
        if gain_db[i] < smoothed_db[i - 1]:
            # Signal getting louder ΓåÆ gain reducing ΓåÆ use attack (fast)
            coeff = attack_coeff
        else:
            # Signal getting quieter ΓåÆ gain recovering ΓåÆ use release (slow)
            coeff = release_coeff
        smoothed_db[i] = coeff * smoothed_db[i - 1] + (1.0 - coeff) * gain_db[i]

    # ΓöÇΓöÇ Step 4: Apply gain + makeup ΓöÇΓöÇ
    gain_linear = (10.0 ** (smoothed_db / 20.0)) * makeup_linear
    # Clamp to max gain to prevent over-amplification of quiet passages
    gain_linear = np.minimum(gain_linear, max_gain)
    compressed = audio * gain_linear.astype(np.float32)

    # ΓöÇΓöÇ Step 5: Peak limiter ΓöÇΓöÇ
    peak = np.max(np.abs(compressed))
    if peak > ceiling:
        compressed = compressed * (ceiling / peak)

    return compressed.astype(np.float32)


def _normalize_voice_ref(audio: np.ndarray) -> np.ndarray:
    """Normalize a voice reference to the configured target level.

    Uses static RMS normalization ΓÇö appropriate for short reference clips
    where temporal dynamics don't apply.
    """
    cfg = _get_leveling_config()
    if not cfg["enabled"]:
        return audio
    return _normalize_rms(
        audio,
        target_rms=cfg["_ref_target_linear"],
        max_gain=cfg["_ref_max_gain_linear"],
        peak_ceiling=cfg["_peak_ceiling_linear"],
    )


def _normalize_output(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Two-stage output processing: level-match, then compress dynamics.

    Stage 1 ΓÇö Static normalization: brings all audio to the threshold RMS
    regardless of source.  This closes the ~15 dB gap between the default
    voice (-40 dBFS) and cloned voices (-27 dBFS).

    Stage 2 ΓÇö Compressor: smooths within-utterance dynamics (attack/release)
    on the already-leveled signal.
    """
    cfg = _get_leveling_config()
    if not cfg["enabled"]:
        return audio
    # Stage 1: match levels ΓÇö bring everything to the threshold target
    audio = _normalize_rms(
        audio,
        target_rms=cfg["_output_threshold_linear"],
        max_gain=cfg["_output_max_gain_linear"],
        peak_ceiling=cfg["_peak_ceiling_linear"],
    )
    # Stage 2: dynamics compression (attack/release/knee)
    return _compress_output(audio, sample_rate)


def _build_voice_system_msg(model, voice_ref: Optional[np.ndarray]) -> dict:
    """
    Build the system message for TTS using the model's own get_sys_prompt().

    This guarantees exact format compatibility with the model's trained prompts.
    """
    return model.get_sys_prompt(ref_audio=voice_ref, mode="audio_assistant")


def _messages_contain_audio(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            content = [content]
        if any(isinstance(item, np.ndarray) for item in content):
            return True
    return False


def _minicpm_blocking_chat_kwargs(messages: list[dict], generate_audio: bool) -> dict[str, object]:
    """Return extra MiniCPM chat kwargs required for blocking multimodal turns."""
    if _messages_contain_audio(messages) and not generate_audio:
        return {
            "omni_mode": True,
            "max_slice_nums": 1,
            "use_image_id": False,
        }
    return {}


def _collect_minicpm_streaming_response(
    messages: list[dict],
    voice_ref: Optional[np.ndarray] = None,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.2,
    top_p: float = 0.8,
    top_k: int = 100,
    enable_thinking: bool = False,
) -> dict:
    """Collect streamed MiniCPM text/audio into the blocking chat() result shape."""
    collected_chunks: list[np.ndarray] = []
    collected_text: list[str] = []

    for audio_chunk, text_chunk in _minicpm_chat_streaming(
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
            collected_text.append(text_chunk)
        if audio_chunk is not None and len(audio_chunk) > 0:
            collected_chunks.append(audio_chunk.astype(np.float32, copy=False))

    full_audio = None
    if collected_chunks:
        full_audio = np.concatenate(collected_chunks).astype(np.float32, copy=False)
        full_audio = _apply_fade_in(full_audio)
        full_audio = _normalize_output(full_audio)
        if output_audio_path:
            import soundfile as sf

            sf.write(output_audio_path, full_audio, _STREAMING_SR)

    return {
        "text": "".join(collected_text),
        "audio": full_audio,
        "audio_path": output_audio_path,
        "sample_rate": _STREAMING_SR if full_audio is not None else None,
    }


def _minicpm_chat(
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
    allow_streaming_fallback: bool = True,
    prefer_streaming_audio: bool = True,
) -> dict:
    """
    Single-turn chat with optional voice cloning.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Content can include text strings, PIL Images, or numpy audio arrays.
        voice_ref: Optional 16kHz mono numpy array for voice cloning.
                   If provided, response will use this voice.
        generate_audio: If True, generate spoken audio response.
        output_audio_path: If set, save audio response to this path.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.
        repetition_penalty: Penalty for repeated tokens.
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling (1-500).
        enable_thinking: If True, enable the model's chain-of-thought reasoning.

    Returns:
        dict with keys:
            text: str ΓÇö the text response
            audio: np.ndarray | None ΓÇö audio waveform (24kHz) if generated
            audio_path: str | None ΓÇö path to saved audio file
    """
    if prefer_streaming_audio and generate_audio and not _messages_contain_audio(messages):
        return _collect_minicpm_streaming_response(
            messages=messages,
            voice_ref=voice_ref,
            output_audio_path=output_audio_path,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        )

    global _turn_count
    _turn_count += 1
    turn = _turn_count

    model, _tok = _get_minicpm_model()

    _reset_for_generation(model, generate_audio, output_audio_path)

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    voice_label = "custom" if voice_ref is not None else "default"
    print(f"  [chat] turn={turn} voice={voice_label} generate_audio={generate_audio}")

    # When voice cloning, re-initialize the Token2wav vocoder cache with the
    # reference audio so the vocoder synthesizes in that voice.  The system
    # prompt (via get_sys_prompt) tells the *language model* about the voice
    # style, but the *vocoder* needs its own reference waveform.
    if voice_ref is not None and generate_audio:
        model.init_token2wav_cache(voice_ref)

    # Build message list using model's own get_sys_prompt() for exact format
    msgs = []
    if generate_audio:
        msgs.append(_build_voice_system_msg(model, voice_ref))
    msgs.extend(messages)
    chat_kwargs = _minicpm_blocking_chat_kwargs(messages, generate_audio)

    # Generate response
    text = model.chat(
        msgs=msgs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        use_tts_template=generate_audio,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
        **chat_kwargs,
    )

    safe_text = text[:120].encode("ascii", errors="replace").decode("ascii")
    print(f"  [chat] turn={turn} response: {safe_text}{'...' if len(str(text)) > 120 else ''}")

    # If we used a custom voice, restore the default-voice cache so subsequent
    # calls without voice_ref don't accidentally use the cloned voice.
    if voice_ref is not None and generate_audio:
        _init_audio = np.zeros(16000, dtype=np.float32)
        model.init_token2wav_cache(_init_audio)

    result = {"text": text, "audio": None, "audio_path": output_audio_path}

    # If audio was saved to file, load it back for playback
    if output_audio_path and Path(output_audio_path).exists():
        import soundfile as sf

        audio_data, sr = sf.read(output_audio_path, dtype="float32")
        # Smooth the HiFT vocoder's cold-start glitch at the beginning
        audio_data = _apply_fade_in(audio_data)
        # Normalize output volume so default and cloned voices are similar
        audio_data = _normalize_output(audio_data)
        result["audio"] = audio_data
        result["sample_rate"] = sr
        print(f"  [chat] turn={turn} audio: {len(audio_data)} samples, {sr}Hz, "
              f"{len(audio_data)/sr:.1f}s")
    else:
        print(f"  [chat] turn={turn} NO AUDIO FILE at {output_audio_path}")

    if allow_streaming_fallback and generate_audio and result["audio"] is None and not _messages_contain_audio(messages):
        fallback_chunks: list[np.ndarray] = []
        fallback_text_parts: list[str] = []
        for audio_chunk, text_chunk in _minicpm_chat_streaming(
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
                fallback_text_parts.append(text_chunk)
            if audio_chunk is not None and len(audio_chunk) > 0:
                fallback_chunks.append(audio_chunk.astype(np.float32, copy=False))

        if fallback_chunks:
            fallback_audio = np.concatenate(fallback_chunks).astype(np.float32, copy=False)
            fallback_audio = _apply_fade_in(fallback_audio)
            fallback_audio = _normalize_output(fallback_audio)
            result["audio"] = fallback_audio
            result["sample_rate"] = _STREAMING_SR
            if output_audio_path:
                import soundfile as sf

                sf.write(output_audio_path, fallback_audio, _STREAMING_SR)
                result["audio_path"] = output_audio_path
            if fallback_text_parts:
                result["text"] = "".join(fallback_text_parts)
            print(f"  [chat] turn={turn} fallback audio: {len(fallback_audio)} samples, {_STREAMING_SR}Hz")

    return result


# ΓöÇΓöÇ Streaming generation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

# Sample rate for all MiniCPM-o TTS audio output.
_STREAMING_SR = 24000


def _minicpm_chat_streaming(
    messages: list[dict],
    voice_ref: Optional[np.ndarray] = None,
    generate_audio: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.2,
    top_p: float = 0.8,
    top_k: int = 100,
    enable_thinking: bool = False,
) -> Generator[tuple[Optional[np.ndarray], str], None, None]:
    """
    Streaming chat ΓÇö yields (audio_chunk, text_chunk) tuples as the model generates.

    Uses the model's streaming_prefill() + streaming_generate() API. Audio chunks
    are ~1 second of 24kHz float32 numpy arrays.  No file I/O is involved.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        voice_ref: Optional 16kHz mono numpy array for voice cloning.
        generate_audio: If True, generate spoken audio (yields waveform chunks).
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.
        repetition_penalty: Penalty for repeated tokens.
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling (1-500).
        enable_thinking: If True, enable chain-of-thought reasoning.

    Yields:
        (audio_chunk, text_chunk) where:
          - audio_chunk: np.ndarray (float32, 24kHz) or None
          - text_chunk: str ΓÇö incremental new text since last yield
    """
    global _turn_count
    _turn_count += 1
    turn = _turn_count
    has_audio_input = _messages_contain_audio(messages)
    pending_outputs: list[tuple[Optional[np.ndarray], str]] = []
    emitted_audio = False
    emitted_text = False
    streaming_started = False

    def _yield_blocking_fallback() -> Generator[tuple[Optional[np.ndarray], str], None, None]:
        result = _minicpm_chat(
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
            allow_streaming_fallback=False,
            prefer_streaming_audio=False,
        )

        text = str(result.get("text") or "")
        if not generate_audio or not text:
            yield None, text
            return

        yielded_audio = False
        yielded_text = False
        for audio_chunk, _text_chunk in stream_text_to_speech_with_minicpm(
            text,
            voice_ref=voice_ref,
        ):
            if audio_chunk is None or len(audio_chunk) == 0:
                continue
            yielded_audio = True
            yield audio_chunk, text if not yielded_text else ""
            yielded_text = True

        if not yielded_audio:
            yield None, text

    try:
        model, _tok = _get_minicpm_model()

        # Don't call _reset_for_generation ΓÇö streaming_prefill handles reset
        # internally when it detects a new session_id. But we do need to free
        # GPU cache and set up voice cloning before prefill.
        torch.cuda.empty_cache()

        # Normalize voice reference to consistent level before feeding to model
        if voice_ref is not None:
            voice_ref = _normalize_voice_ref(voice_ref)

        voice_label = "custom" if voice_ref is not None else "default"
        print(f"  [stream] turn={turn} voice={voice_label} generate_audio={generate_audio}")

        # Init Token2wav cache BEFORE streaming_prefill.
        # streaming_prefill calls reset_session() which clears the cache,
        # so we must (re)initialize it here for BOTH custom and default voice.
        if generate_audio:
            cache_audio = voice_ref if voice_ref is not None else np.zeros(16000, dtype=np.float32)
            model.init_token2wav_cache(cache_audio)

        # Build full message list with TTS system prompt
        msgs = []
        if generate_audio:
            msgs.append(_build_voice_system_msg(model, voice_ref))
        msgs.extend(messages)

        # Prefill each message one at a time (streaming_prefill requires len(msgs)==1).
        #
        # Audio handling: streaming_prefill uses online_streaming=True for user
        # messages containing audio.  This mode expects audio in proper-sized
        # chunks (FIRST_CHUNK_MS=1035 then CHUNK_MS=1000).  Sending a complete
        # recording as one chunk causes a tensor size mismatch because the
        # processor creates placeholder tokens based on streaming assumptions.
        #
        # Fix: split audio into chunks and call streaming_prefill once per chunk.
        # The model handles this natively ΓÇö after the first user segment it sets
        # new_user_msg=False so subsequent chunks skip the role prefix.
        FIRST_CHUNK_SAMPLES = int(1035 * 16000 / 1000)   # 16,560
        REGULAR_CHUNK_SAMPLES = int(1000 * 16000 / 1000)  # 16,000

        session_id = f"omnichat_stream_{turn}"
        for i, msg in enumerate(msgs):
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            # Separate audio arrays from other content (text, images)
            audio_arrays = [c for c in content if isinstance(c, np.ndarray)]
            other_content = [c for c in content if not isinstance(c, np.ndarray)]

            if not audio_arrays:
                # Text-only message ΓÇö single prefill call
                model.streaming_prefill(
                    session_id=session_id,
                    msgs=[{"role": msg["role"], "content": content}],
                    use_tts_template=generate_audio,
                    is_last_chunk=False,
                )
            else:
                # Audio message ΓÇö split into chunks for streaming prefill.
                # Concatenate all audio arrays (usually just one).
                full_audio = (np.concatenate(audio_arrays)
                              if len(audio_arrays) > 1 else audio_arrays[0])
                full_audio = full_audio.astype(np.float32)

                # Split: first chunk = FIRST_CHUNK_SAMPLES, rest = REGULAR_CHUNK_SAMPLES.
                # Every chunk MUST be exactly the expected size ΓÇö the model's
                # placeholder/embedding alignment requires it.  Zero-pad the
                # last chunk if it's shorter than REGULAR_CHUNK_SAMPLES.
                chunks: list[np.ndarray] = []
                n = len(full_audio)
                if n <= FIRST_CHUNK_SAMPLES:
                    # Short clip ΓÇö pad to exactly FIRST_CHUNK_SAMPLES
                    if n < FIRST_CHUNK_SAMPLES:
                        padded = np.zeros(FIRST_CHUNK_SAMPLES, dtype=np.float32)
                        padded[:n] = full_audio
                        chunks.append(padded)
                    else:
                        chunks.append(full_audio)
                else:
                    chunks.append(full_audio[:FIRST_CHUNK_SAMPLES])
                    pos = FIRST_CHUNK_SAMPLES
                    while pos < n:
                        chunk = full_audio[pos:pos + REGULAR_CHUNK_SAMPLES]
                        if len(chunk) < REGULAR_CHUNK_SAMPLES:
                            # Zero-pad to exact size
                            padded = np.zeros(REGULAR_CHUNK_SAMPLES, dtype=np.float32)
                            padded[:len(chunk)] = chunk
                            chunk = padded
                        chunks.append(chunk)
                        pos += REGULAR_CHUNK_SAMPLES

                n_chunks = len(chunks)
                print(f"  [stream] audio split into {n_chunks} chunk(s): "
                      f"{[len(c) for c in chunks]} samples")

                for ci, chunk in enumerate(chunks):
                    is_last = (ci == n_chunks - 1)
                    if ci == 0:
                        # First chunk: include text alongside audio
                        chunk_content = other_content + [chunk]
                    else:
                        # Subsequent chunks: audio only (model skips role prefix)
                        chunk_content = [chunk]
                    model.streaming_prefill(
                        session_id=session_id,
                        msgs=[{"role": msg["role"], "content": chunk_content}],
                        use_tts_template=generate_audio,
                        is_last_chunk=is_last,
                    )

        # Stream-generate text + audio
        _audio_chunks = 0
        _text_chunks = 0
        _total_audio_samples = 0
        for waveform_chunk, new_text in model.streaming_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            use_tts_template=generate_audio,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
        ):
            # streaming_generate yields (waveform_chunk, new_text) when generate_audio=True,
            # but (text_str, bool) when generate_audio=False.  Detect audio by type.
            if hasattr(waveform_chunk, "cpu"):
                # Torch tensor ΓÇö convert to 1D float32 numpy
                chunk_np = waveform_chunk.cpu().numpy().flatten().astype(np.float32)
                _audio_chunks += 1
                _total_audio_samples += len(chunk_np)
                emitted_audio = True
                text_out = new_text if isinstance(new_text, str) else ""
                if text_out:
                    emitted_text = True
                if generate_audio:
                    pending_outputs.append((chunk_np, text_out))
                    if emitted_audio and emitted_text:
                        streaming_started = True
                        while pending_outputs:
                            yield pending_outputs.pop(0)
                else:
                    yield chunk_np, text_out
            elif isinstance(waveform_chunk, np.ndarray):
                chunk_np = waveform_chunk.flatten().astype(np.float32)
                _audio_chunks += 1
                _total_audio_samples += len(chunk_np)
                emitted_audio = True
                text_out = new_text if isinstance(new_text, str) else ""
                if text_out:
                    emitted_text = True
                if generate_audio:
                    pending_outputs.append((chunk_np, text_out))
                    if emitted_audio and emitted_text:
                        streaming_started = True
                        while pending_outputs:
                            yield pending_outputs.pop(0)
                else:
                    yield chunk_np, text_out
            elif isinstance(waveform_chunk, str) and waveform_chunk:
                # Text-only mode: first element is text, second is bool (not str)
                _text_chunks += 1
                emitted_text = True
                if generate_audio:
                    pending_outputs.append((None, waveform_chunk))
                    if emitted_audio and emitted_text:
                        streaming_started = True
                        while pending_outputs:
                            yield pending_outputs.pop(0)
                else:
                    yield None, waveform_chunk
            elif isinstance(new_text, str) and new_text:
                _text_chunks += 1
                emitted_text = True
                if generate_audio:
                    pending_outputs.append((None, new_text))
                    if emitted_audio and emitted_text:
                        streaming_started = True
                        while pending_outputs:
                            yield pending_outputs.pop(0)
                else:
                    yield None, new_text

        print(f"  [stream] turn={turn} complete: {_audio_chunks} audio chunks "
              f"({_total_audio_samples} samples, {_total_audio_samples/24000:.1f}s), "
              f"{_text_chunks} text-only chunks")

        if generate_audio:
            if emitted_audio and emitted_text:
                while pending_outputs:
                    yield pending_outputs.pop(0)
            else:
                print(
                    f"  [stream] turn={turn} incomplete streaming output "
                    f"(audio={emitted_audio}, text={emitted_text}), falling back to blocking chat"
                )
                yield from _yield_blocking_fallback()

        # Restore default voice cache after voice cloning
        if voice_ref is not None and generate_audio:
            _init_audio = np.zeros(16000, dtype=np.float32)
            model.init_token2wav_cache(_init_audio)
    except Exception as exc:
        if not generate_audio and not has_audio_input:
            raise
        if generate_audio and (streaming_started or (emitted_audio and emitted_text)):
            print(f"  [stream] turn={turn} streaming interrupted after partial output: {exc}")
            while pending_outputs:
                yield pending_outputs.pop(0)
            return
        print(f"  [stream] turn={turn} streaming failed, falling back to blocking chat: {exc}")
        yield from _yield_blocking_fallback()


def _minicpm_chat_streaming_with_playback(
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
    on_text_chunk: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Stream-generate and play audio simultaneously through speakers.

    Audio starts playing as soon as the first chunk arrives (~1-2s), while the
    model continues generating. Returns the same dict interface as chat().

    Args:
        messages: List of message dicts.
        voice_ref: Optional voice reference for cloning.
        output_audio_path: If set, save complete audio here AFTER playback.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens.
        repetition_penalty: Penalty for repeated tokens.
        top_p: Nucleus sampling threshold (0.0-1.0).
        top_k: Top-k sampling (1-500).
        enable_thinking: If True, enable chain-of-thought reasoning.
        headless: If True, skip audio playback (collect audio silently).
        on_text_chunk: Optional callback for incremental text updates.

    Returns:
        dict with keys: text, audio, audio_path, sample_rate
    """
    player = None
    if not headless:
        from tools.audio.streaming_player import StreamingAudioPlayer
        player = StreamingAudioPlayer(sample_rate=_STREAMING_SR)
        player.start()

    full_text = ""
    collected_chunks: list[np.ndarray] = []

    is_first_chunk = True
    cfg = _get_leveling_config()

    try:
        for audio_chunk, text_chunk in _minicpm_chat_streaming(
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
            if audio_chunk is not None:
                # Smooth vocoder cold-start on the very first chunk
                if is_first_chunk:
                    audio_chunk = _apply_fade_in(audio_chunk)
                    is_first_chunk = False
                # Normalize each chunk before playback so the user hears
                # leveled audio in real time (not just in the archive file).
                # Uses the static RMS normalizer (Stage 1) per-chunk ΓÇö
                # appropriate since each ~1s chunk has consistent RMS.
                if cfg["enabled"]:
                    audio_chunk = _normalize_rms(
                        audio_chunk,
                        target_rms=cfg["_output_threshold_linear"],
                        max_gain=cfg["_output_max_gain_linear"],
                        peak_ceiling=cfg["_peak_ceiling_linear"],
                    )
                collected_chunks.append(audio_chunk)
                if player:
                    player.push(audio_chunk)

            if text_chunk:
                full_text += text_chunk
                if on_text_chunk:
                    on_text_chunk(text_chunk)
    finally:
        if player:
            player.finish()
            player.wait()
            player.stop()

    # Assemble result ΓÇö chunks are already per-chunk normalized;
    # run full-signal compressor on the concatenated audio for the archive.
    full_audio = np.concatenate(collected_chunks) if collected_chunks else None
    if full_audio is not None:
        full_audio = _normalize_output(full_audio)

    # Archival save AFTER playback (not before)
    if output_audio_path and full_audio is not None:
        import soundfile as sf
        sf.write(output_audio_path, full_audio, _STREAMING_SR)

    return {
        "text": full_text,
        "audio": full_audio,
        "audio_path": output_audio_path,
        "sample_rate": _STREAMING_SR,
    }


def _minicpm_process_image(
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
) -> dict:
    """
    Analyze an image with an optional text prompt.

    Args:
        image: PIL.Image.Image
        prompt: Text prompt describing what to do with the image.
        max_slice_nums: 1 for photos, 25 for documents/PDFs.

    Returns:
        dict with text (and optionally audio).
    """
    model, _tok = get_model()

    _reset_for_generation(model, generate_audio, output_audio_path)

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    msgs = []
    if generate_audio or voice_ref is not None:
        msgs.append(_build_voice_system_msg(model, voice_ref))

    msgs.append({"role": "user", "content": [image, prompt]})

    text = model.chat(
        msgs=msgs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        use_tts_template=generate_audio,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
        max_slice_nums=max_slice_nums,
        use_image_id=False,
    )

    result = {"text": text, "audio": None, "audio_path": output_audio_path}
    if output_audio_path and Path(output_audio_path).exists():
        import soundfile as sf

        audio_data, sr = sf.read(output_audio_path, dtype="float32")
        audio_data = _apply_fade_in(audio_data)
        # Normalize output volume so default and cloned voices are similar
        audio_data = _normalize_output(audio_data)
        result["audio"] = audio_data
        result["sample_rate"] = sr

    return result


def _minicpm_process_video(
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
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """
    Analyze a video file (with audio track).

    Args:
        video_path: Path to video file.
        prompt: Text prompt describing what to analyze.

    Returns:
        dict with text (and optionally audio).
    """
    def _progress(stage: str, pct: float):
        if on_progress:
            on_progress(stage, pct)

    _progress("Loading model", 0.0)
    model, _tok = get_model()

    _reset_for_generation(model, generate_audio, output_audio_path)

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    from minicpmo.utils import get_video_frame_audio_segments

    _progress("Extracting audio", 0.1)
    # Pre-extract audio using imageio-ffmpeg's bundled binary.
    # minicpmo's own audio extraction fails on Windows with MKV files:
    #   - use_ffmpeg=True needs ffprobe on PATH (not installed)
    #   - use_ffmpeg=False falls back to moviepy which uses
    #     NamedTemporaryFile(delete=True) ΓÇö Windows locks the file,
    #     so ffmpeg subprocess gets "Permission denied"
    # By extracting audio ourselves and passing audio_path, we bypass
    # both code paths entirely.
    import tempfile
    import subprocess
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()  # close immediately so ffmpeg can write to it
    has_audio = True
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-ac", "1",
             "-ar", "16000", temp_audio_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Video has no audio stream ΓÇö create a silent placeholder so
        # get_video_frame_audio_segments can still align frames.
        has_audio = False
        import soundfile as sf
        # Probe duration via ffmpeg to size the silence correctly
        probe = subprocess.run(
            [ffmpeg_exe, "-i", video_path],
            capture_output=True, text=True,
        )
        duration = 1.0  # fallback
        for line in probe.stderr.splitlines():
            if "Duration:" in line:
                parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
                try:
                    duration = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                except (ValueError, IndexError):
                    pass
                break
        silence = np.zeros(int(16000 * duration), dtype=np.float32)
        sf.write(temp_audio_path, silence, 16000)

    _progress("Extracting frames", 0.2)
    # Monkeypatch MAX_NUM_FRAMES so minicpmo uses our setting.
    # It's a module-level constant that can't be passed as a parameter.
    import minicpmo.utils as _mu
    _orig_max_frames = _mu.MAX_NUM_FRAMES
    _mu.MAX_NUM_FRAMES = max_frames
    try:
        video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
            video_path,
            audio_path=temp_audio_path,
            stack_frames=1,
            use_ffmpeg=False,
            adjust_audio_length=True,
        )
    finally:
        _mu.MAX_NUM_FRAMES = _orig_max_frames
        Path(temp_audio_path).unlink(missing_ok=True)

    _progress(f"Processing {len(video_frames)} frames", 0.4)
    # Build interleaved content: frame, audio, frame, audio, ...
    omni_contents = []
    for i in range(len(video_frames)):
        omni_contents.append(video_frames[i])
        if i < len(audio_segments):
            omni_contents.append(audio_segments[i])
        if stacked_frames is not None and i < len(stacked_frames) and stacked_frames[i] is not None:
            omni_contents.append(stacked_frames[i])

    omni_contents.append(prompt)

    msgs = []
    if generate_audio or voice_ref is not None:
        msgs.append(_build_voice_system_msg(model, voice_ref))

    msgs.append({"role": "user", "content": omni_contents})

    _progress("Analyzing video", 0.5)
    # model.chat() defaults max_inp_length=8192, but the model supports 40960
    # position embeddings. With many video frames, 8192 is too small ΓÇö the
    # processor blindly truncates input_ids[:max_inp_length], which can cut
    # paired audio tokens (audio_start without audio_end) causing an assertion
    # error in processing_minicpmo.py. Scale the limit to fit all frames.
    n_frames = len(video_frames)
    max_inp_length = max(8192, min(n_frames * 300 + 2048, 40960))

    text = model.chat(
        msgs=msgs,
        max_new_tokens=max_new_tokens,
        max_inp_length=max_inp_length,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        use_tts_template=generate_audio,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
        omni_mode=True,
        max_slice_nums=1,
        use_image_id=False,
    )

    _progress("Complete", 1.0)
    result = {"text": text, "audio": None, "audio_path": output_audio_path}
    if output_audio_path and Path(output_audio_path).exists():
        import soundfile as sf

        audio_data, sr = sf.read(output_audio_path, dtype="float32")
        audio_data = _apply_fade_in(audio_data)
        # Normalize output volume so default and cloned voices are similar
        audio_data = _normalize_output(audio_data)
        result["audio"] = audio_data
        result["sample_rate"] = sr

    return result


def _minicpm_process_video_chunked(
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
    on_chunk: Optional[Callable[[int, int, str], None]] = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """
    Analyze a video file in time-window chunks for thorough coverage.

    Splits the video into N-second segments, analyzes each with max_frames,
    and accumulates the descriptions. Each chunk gets the full frame budget,
    so a 2-minute video at chunk_seconds=30 gets 4 passes of 64 frames each
    instead of 64 frames spread across 2 minutes.

    Args:
        video_path: Path to video file.
        prompt: Text prompt for each chunk. A time context prefix is added automatically.
        chunk_seconds: Duration of each video chunk in seconds (default 30).
        on_chunk: Callback(chunk_index, total_chunks, accumulated_text) after each chunk.
        on_progress: Callback(stage_name, percent_0_to_1) for UI progress bars.

    Returns:
        dict with text (accumulated descriptions from all chunks).
    """
    import math
    import tempfile
    import subprocess
    import imageio_ffmpeg

    def _progress(stage: str, pct: float):
        if on_progress:
            on_progress(stage, pct)

    _progress("Probing video", 0.0)

    # Get video duration
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run(
        [ffmpeg_exe, "-i", video_path],
        capture_output=True, text=True,
    )
    duration = 0.0
    for line in probe.stderr.splitlines():
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip().split(":")
            try:
                duration = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except (ValueError, IndexError):
                pass
            break

    if duration <= 0:
        # Fallback: process as single chunk
        return _minicpm_process_video(
            video_path=video_path, prompt=prompt,
            generate_audio=generate_audio, voice_ref=voice_ref,
            temperature=temperature, max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k,
            enable_thinking=enable_thinking, max_frames=max_frames,
            on_progress=on_progress,
        )

    # If video is shorter than chunk_seconds, just process it whole
    if duration <= chunk_seconds:
        return _minicpm_process_video(
            video_path=video_path, prompt=prompt,
            generate_audio=generate_audio, voice_ref=voice_ref,
            temperature=temperature, max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k,
            enable_thinking=enable_thinking, max_frames=max_frames,
            on_progress=on_progress,
        )

    n_chunks = max(1, math.ceil(duration / chunk_seconds))
    dur_label = f"{int(duration // 60)}:{int(duration % 60):02d}"
    all_parts = []

    _progress(f"Video {dur_label} -- {n_chunks} chunks", 0.05)

    for i in range(n_chunks):
        start_s = i * chunk_seconds
        end_s = min((i + 1) * chunk_seconds, duration)
        chunk_dur = end_s - start_s

        chunk_pct = i / n_chunks
        time_range = f"{int(start_s // 60)}:{int(start_s % 60):02d}-{int(end_s // 60)}:{int(end_s % 60):02d}"
        _progress(f"Chunk {i + 1}/{n_chunks} [{time_range}] of {dur_label}", chunk_pct)

        # Extract chunk to temp file using ffmpeg -ss/-t
        chunk_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        chunk_path = chunk_file.name
        chunk_file.close()

        try:
            subprocess.run(
                [ffmpeg_exe, "-y",
                 "-ss", str(start_s), "-t", str(chunk_dur),
                 "-i", video_path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 "-c:a", "aac", "-ac", "1",
                 chunk_path],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError:
            # Try copy codec (faster, but may fail on some containers)
            try:
                subprocess.run(
                    [ffmpeg_exe, "-y",
                     "-ss", str(start_s), "-t", str(chunk_dur),
                     "-i", video_path,
                     "-c", "copy",
                     chunk_path],
                    check=True, capture_output=True,
                )
            except subprocess.CalledProcessError:
                Path(chunk_path).unlink(missing_ok=True)
                all_parts.append(f"[Chunk {i+1}: failed to extract]")
                continue

        # Add time context to the prompt
        time_label = f"{int(start_s // 60)}:{int(start_s % 60):02d}-{int(end_s // 60)}:{int(end_s % 60):02d}"
        chunk_prompt = f"[Video segment {time_label}] {prompt}"

        try:
            chunk_result = _minicpm_process_video(
                video_path=chunk_path,
                prompt=chunk_prompt,
                generate_audio=False,  # no audio gen for intermediate chunks
                voice_ref=voice_ref,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                top_k=top_k,
                enable_thinking=enable_thinking,
                max_frames=max_frames,
            )
            chunk_text = chunk_result.get("text", "").strip()
            if chunk_text:
                all_parts.append(f"**[{time_label}]** {chunk_text}")
        finally:
            Path(chunk_path).unlink(missing_ok=True)

        accumulated = "\n\n".join(all_parts)
        if on_chunk:
            on_chunk(i, n_chunks, accumulated)

    _progress("Complete", 1.0)
    full_text = "\n\n".join(all_parts)
    return {"text": full_text, "audio": None, "audio_path": None}


def _minicpm_transcribe_audio(
    video_path: str,
    prompt: str = "Transcribe this audio completely and verbatim.",
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    repetition_penalty: float = 1.05,
    top_p: float = 0.8,
    top_k: int = 100,
    enable_thinking: bool = False,
    chunk_seconds: int = 30,
    on_chunk: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Transcribe audio from a video/audio file by sending raw audio to the model.

    Pre-chunks audio into segments (default 30s) and transcribes each separately,
    calling on_chunk after each so the UI can show progressive results.

    Args:
        video_path: Path to video or audio file.
        prompt: Transcription instruction.
        temperature: Low values (0.2-0.4) recommended for faithful transcription.
        max_new_tokens: Max output tokens per chunk.
        chunk_seconds: Length of each audio chunk in seconds (default 30).
        on_chunk: Callback(chunk_index, total_chunks, accumulated_text) called
                  after each chunk is transcribed.

    Returns:
        dict with text (full transcription).
    """
    model, _tok = get_model()

    # Extract audio using imageio-ffmpeg's bundled binary
    import tempfile
    import subprocess
    import imageio_ffmpeg
    import soundfile as sf
    import math

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", video_path, "-vn", "-ac", "1",
             "-ar", "16000", temp_audio_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        Path(temp_audio_path).unlink(missing_ok=True)
        raise RuntimeError(
            f"Failed to extract audio: {e.stderr.decode(errors='replace')}"
        ) from e

    audio_data, sr = sf.read(temp_audio_path, dtype="float32")
    Path(temp_audio_path).unlink(missing_ok=True)

    # Split into chunks
    samples_per_chunk = chunk_seconds * sr
    total_samples = len(audio_data)
    n_chunks = max(1, math.ceil(total_samples / samples_per_chunk))

    chunks = []
    for i in range(n_chunks):
        start = int(i * samples_per_chunk)
        end = int(min((i + 1) * samples_per_chunk, total_samples))
        chunks.append(audio_data[start:end])

    # Transcribe each chunk
    all_parts = []
    for i, chunk in enumerate(chunks):
        _reset_for_generation(model, generate_audio=False, output_audio_path=None)

        msgs = [{"role": "user", "content": [chunk, prompt]}]

        text = model.chat(
            msgs=msgs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            enable_thinking=enable_thinking,
            generate_audio=False,
            omni_mode=True,
            max_slice_nums=1,
            use_image_id=False,
        )

        all_parts.append(text.strip() if text else "")
        accumulated = " ".join(all_parts)

        if on_chunk:
            on_chunk(i, n_chunks, accumulated)

    full_text = " ".join(all_parts)
    return {"text": full_text, "audio": None, "audio_path": None}


def get_model():
    """Load or return the active backend model singleton."""
    return get_backend().get_model()


def chat(
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
) -> dict:
    """Run a blocking chat request through the active backend."""
    return get_backend().chat(
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
        trace_context=trace_context,
    )


def chat_blocking_compare(
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
) -> dict:
    """Force end-of-turn delivery for latency comparison benchmarks."""
    if get_backend_name() == "minicpm":
        _ = trace_context
        return _minicpm_chat(
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
            allow_streaming_fallback=False,
            prefer_streaming_audio=False,
        )

    return get_backend().chat(
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
        trace_context=trace_context,
    )


def chat_streaming(
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
) -> Generator[tuple[Optional[np.ndarray], str], None, None]:
    """Run a streaming chat request through the active backend."""
    request_id = "n/a"
    if isinstance(trace_context, dict):
        request_id = str(trace_context.get("request_id") or "n/a")

    logger.info(
        "trace_id=%s stage=model_manager event=chat_streaming_enter backend=%s generate_audio=%s message_count=%d",
        request_id,
        _backend_name,
        bool(generate_audio),
        len(messages or []),
    )
    backend = get_backend()
    logger.info(
        "trace_id=%s stage=model_manager event=chat_streaming_backend_resolved backend=%s backend_type=%s",
        request_id,
        _backend_name,
        type(backend).__name__,
    )
    return (yield from backend.chat_streaming(
        messages=messages,
        voice_ref=voice_ref,
        generate_audio=generate_audio,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        trace_context=trace_context,
    ))


def chat_streaming_with_playback(
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
    on_text_chunk: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run streaming chat with local playback through the active backend."""
    return get_backend().chat_streaming_with_playback(
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
) -> dict:
    """Analyze an image through the active backend."""
    return get_backend().process_image(
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
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Analyze a video through the active backend."""
    return get_backend().process_video(
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
    on_chunk: Optional[Callable[[int, int, str], None]] = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Analyze a video in chunks through the active backend."""
    return get_backend().process_video_chunked(
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
    video_path: str,
    prompt: str = "Transcribe this audio completely and verbatim.",
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    repetition_penalty: float = 1.05,
    top_p: float = 0.8,
    top_k: int = 100,
    enable_thinking: bool = False,
    chunk_seconds: int = 30,
    on_chunk: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Transcribe audio through the active backend."""
    return get_backend().transcribe_audio(
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


# ΓöÇΓöÇ CLI test ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MiniCPM-o 4.5 model loading")
    parser.add_argument("--text", default="Hello! Tell me a short joke.", help="Text prompt")
    parser.add_argument("--audio", action="store_true", help="Generate audio response")
    parser.add_argument("--stream", action="store_true", help="Use streaming generation (real-time playback)")
    parser.add_argument("--output", default=None, help="Save audio to this path")
    args = parser.parse_args()

    model, tok = get_model()
    print(f"\nModel loaded successfully on {next(model.parameters()).device}")

    if args.stream:
        print(f"\nStreaming: '{args.text}'")
        result = chat_streaming_with_playback(
            messages=[{"role": "user", "content": [args.text]}],
            output_audio_path=args.output,
            headless=not args.audio,
        )
    else:
        result = chat(
            messages=[{"role": "user", "content": [args.text]}],
            generate_audio=args.audio,
            output_audio_path=args.output,
        )

    print(f"\nResponse: {result['text']}")
    if result.get("audio"):
        duration = len(result["audio"]) / result.get("sample_rate", 24000)
        print(f"Audio: {duration:.1f}s at {result.get('sample_rate', 24000)}Hz")
    if result.get("audio_path"):
        print(f"Audio saved to: {result['audio_path']}")
