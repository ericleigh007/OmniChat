"""
session.py -- Shared session helpers for both Gradio and PySide6 apps.

Extracted from main.py to avoid duplication between the two UI frontends.
All functions are pure logic with no UI framework dependency.
"""

import json
import re
import tempfile
from copy import deepcopy
from pathlib import Path

import yaml
import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent.resolve()


def _default_remote_endpoints() -> dict:
    return {
        "health": "/health",
        "models": "models",
        "chat_completions": "chat/completions",
        "audio_transcriptions": "audio/transcriptions",
        "audio_speech": "audio/speech",
        "audio_voices": "audio/voices",
        "responses": "responses",
        "realtime": "realtime",
    }


def _default_settings() -> dict:
    return {
        "model_profile": "gemma4_e4b_transformers_mincpm_tts",
        "model_profiles_file": "args/model_profiles.json",
        "model": {
            "quantization": "none",
            "auto_update": True,
        },
        "audio": {
            "input_sample_rate": 16000, "output_sample_rate": 24000, "default_voice": None, "voices_dir": "voices",
            "streaming": {
                "enabled": True,
                "blocking_compare_enabled": False,
            },
            "leveling": {
                "enabled": True, "ref_target_dbfs": -26.0, "ref_max_gain_db": 20.0,
                "output_threshold_dbfs": -20.0, "output_ratio": 3.0,
                "output_attack_ms": 15.0, "output_release_ms": 150.0,
                "output_knee_db": 6.0, "output_makeup_db": 4.0,
                "output_max_gain_db": 20.0, "peak_ceiling_dbfs": -0.1,
            },
        },
        "voice_commands": {"enabled": True, "fuzzy_threshold": 0.6},
        "inference": {
            "temperature": 0.7, "max_new_tokens": 2048, "do_sample": True,
            "repetition_penalty": 1.05, "top_p": 0.8, "top_k": 100,
            "enable_thinking": False, "max_frames": 64,
        },
        "output": {"default_format": "auto", "save_dir": "outputs"},
        "server": {"host": "localhost", "port": 7860, "share": False},
    }


def _default_model_profiles() -> dict:
    return {
        "default_profile": "gemma4_e4b_transformers_mincpm_tts",
        "profiles": {
            "minicpm_local": {
                "display_name": "MiniCPM-o 4.5 Local",
                "backend": "minicpm",
                "transport": "local_transformers",
                "name": "openbmb/MiniCPM-o-4_5",
                "dtype": "bfloat16",
                "device": "cuda",
                "quantization": "none",
                "auto_update": True,
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Loads MiniCPM directly inside the Windows app process.",
                },
            },
            "qwen3_omni_wsl": {
                "display_name": "Qwen3-Omni 30B A3B Instruct Remote (WSL/OpenAI API)",
                "backend": "qwen_remote",
                "transport": "openai_compatible",
                "name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "remote": {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "api_key": None,
                    "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    "timeout_s": 120,
                    "endpoints": _default_remote_endpoints(),
                },
                "launcher": {
                    "script": "launch_qwen_wsl.bat",
                    "host_os": "windows",
                    "server_env": "wsl",
                    "wsl_distro": "Ubuntu-22.04",
                    "stage_config": "args/qwen3_omni_wsl_stage_config.yaml",
                    "supports": ["--restart", "--stop"],
                },
            },
            "qwen3_omni_local": {
                "display_name": "Qwen3-Omni 30B A3B Instruct Local (Transformers)",
                "backend": "qwen_transformers",
                "transport": "local_transformers",
                "name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                "device_map": "auto",
                "torch_dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "speaker": "Ethan",
                "use_audio_in_video": True,
                "local_files_only": False,
                "quantization": "none",
                "auto_update": True,
                "inference": {
                    "temperature": 0.3,
                    "max_new_tokens": 192,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 1,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Loads Qwen3-Omni directly inside the Windows app process using Transformers.",
                },
            },
            "gemma4_e4b_transformers_local": {
                "display_name": "Gemma 4 E4B IT Local (Transformers, native audio)",
                "backend": "gemma_transformers",
                "transport": "local_transformers",
                "name": "Gemma 4 E4B IT",
                "checkpoint": r"D:\OmniChatModels\gemma4-e4b-it-official\hf",
                "device_map": "auto",
                "torch_dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "speech_backend": "none",
                "use_audio_in_video": True,
                "video_backend": "pyav",
                "local_files_only": False,
                "quantization": "none",
                "auto_update": False,
                "inference": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 20,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Loads Gemma 4 E4B IT directly inside the Windows app process using Transformers for native text, image, audio, and video input.",
                },
            },
            "gemma4_e4b_transformers_mincpm_tts": {
                "display_name": "Gemma 4 E4B IT Local (Transformers) + MiniCPM 4.5 Streaming TTS",
                "backend": "gemma_transformers",
                "transport": "local_transformers",
                "name": "Gemma 4 E4B IT",
                "checkpoint": r"D:\OmniChatModels\gemma4-e4b-it-official\hf",
                "device_map": "auto",
                "torch_dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "speech_backend": "minicpm_streaming",
                "use_audio_in_video": True,
                "video_backend": "pyav",
                "local_files_only": False,
                "quantization": "none",
                "auto_update": False,
                "inference": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 20,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Uses local Gemma 4 E4B IT Transformers inference and streams spoken output through MiniCPM 4.5 TTS.",
                },
            },
            "qwen35_27b_llamacpp_local": {
                "display_name": "Qwen3.5 27B Q4_K_M Local (llama.cpp, owned, text-only)",
                "backend": "qwen_llamacpp",
                "transport": "local_llamacpp_cli",
                "name": "Qwen/Qwen3.5-27B",
                "quantization": "none",
                "auto_update": False,
                "llama_cpp": {
                    "llama_root": str((Path(tempfile.gettempdir()) / "llama-cpp-qwen35-test" / "llama.cpp").resolve()),
                    "cli_path": None,
                    "model_path": r"D:\OmniChatModels\qwen35-27b-official\gguf\Qwen-Qwen3.5-27B-27B-Q4_K_M-00001-of-00002.gguf",
                    "mmproj_path": r"D:\OmniChatModels\qwen35-27b-official\gguf\mmproj-Qwen-Qwen3.5-27B-F16.gguf",
                    "n_gpu_layers": 99,
                    "flash_attn": True,
                    "context_length": 8192,
                    "use_jinja": True,
                    "speech_backend": "none",
                },
                "inference": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 20,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Uses the owned local llama.cpp GGUF + mmproj artifacts staged on D: for A/B comparison.",
                },
            },
            "qwen35_27b_llamacpp_local_minicpm_tts": {
                "display_name": "Qwen3.5 27B Q4_K_M Local (llama.cpp) + MiniCPM 4.5 Streaming TTS",
                "backend": "qwen_llamacpp",
                "transport": "local_llamacpp_cli",
                "name": "Qwen/Qwen3.5-27B",
                "quantization": "none",
                "auto_update": False,
                "llama_cpp": {
                    "llama_root": str((Path(tempfile.gettempdir()) / "llama-cpp-qwen35-test" / "llama.cpp").resolve()),
                    "cli_path": None,
                    "model_path": r"D:\OmniChatModels\qwen35-27b-official\gguf\Qwen-Qwen3.5-27B-27B-Q4_K_M-00001-of-00002.gguf",
                    "mmproj_path": r"D:\OmniChatModels\qwen35-27b-official\gguf\mmproj-Qwen-Qwen3.5-27B-F16.gguf",
                    "n_gpu_layers": 99,
                    "flash_attn": True,
                    "context_length": 8192,
                    "use_jinja": True,
                    "speech_backend": "minicpm_streaming",
                },
                "inference": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 20,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Uses owned local llama.cpp GGUF inference for Qwen3.5 27B and streams spoken output through MiniCPM 4.5 TTS for lower playback latency.",
                },
            },
            "gemma4_ssize_llamacpp_local": {
                "display_name": "Gemma 4 S-Size Local (llama.cpp, multimodal)",
                "backend": "gemma_llamacpp",
                "transport": "local_llamacpp_cli",
                "name": "Gemma 4 S-Size",
                "quantization": "none",
                "auto_update": False,
                "llama_cpp": {
                    "llama_root": str((Path(tempfile.gettempdir()) / "llama-cpp-gemma4-test" / "llama.cpp").resolve()),
                    "cli_path": None,
                    "model_path": r"D:\OmniChatModels\gemma4-ssize-official\gguf\gemma4-ssize-it-Q4_K_M.gguf",
                    "mmproj_path": r"D:\OmniChatModels\gemma4-ssize-official\gguf\mmproj-gemma4-ssize-it-f16.gguf",
                    "n_gpu_layers": 99,
                    "flash_attn": True,
                    "context_length": 8192,
                    "use_jinja": True,
                    "speech_backend": "none",
                },
                "inference": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 20,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Uses a local llama.cpp Gemma multimodal GGUF + mmproj path for typed, image, and audio-input turns.",
                },
            },
            "gemma4_ssize_llamacpp_mincpm_tts": {
                "display_name": "Gemma 4 S-Size Local (llama.cpp) + MiniCPM 4.5 Streaming TTS",
                "backend": "gemma_llamacpp",
                "transport": "local_llamacpp_cli",
                "name": "Gemma 4 S-Size",
                "quantization": "none",
                "auto_update": False,
                "llama_cpp": {
                    "llama_root": str((Path(tempfile.gettempdir()) / "llama-cpp-gemma4-test" / "llama.cpp").resolve()),
                    "cli_path": None,
                    "model_path": r"D:\OmniChatModels\gemma4-ssize-official\gguf\gemma4-ssize-it-Q4_K_M.gguf",
                    "mmproj_path": r"D:\OmniChatModels\gemma4-ssize-official\gguf\mmproj-gemma4-ssize-it-f16.gguf",
                    "n_gpu_layers": 99,
                    "flash_attn": True,
                    "context_length": 8192,
                    "use_jinja": True,
                    "speech_backend": "minicpm_streaming",
                },
                "inference": {
                    "temperature": 0.2,
                    "max_new_tokens": 256,
                    "repetition_penalty": 1.05,
                    "top_p": 0.9,
                    "top_k": 20,
                    "enable_thinking": False,
                },
                "launcher": {
                    "frontend_scripts": ["launch.bat", "launch_rt.bat"],
                    "notes": "Uses local Gemma multimodal GGUF inference through llama.cpp and streams spoken output through MiniCPM 4.5 TTS.",
                },
            },
        },
    }


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return deepcopy(override)

    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_model_profiles(*, profiles_file: str | None = None) -> dict:
    """Load JSON model profiles and merge them with built-in defaults."""
    defaults = _default_model_profiles()
    profiles_path = BASE_DIR / (profiles_file or _default_settings()["model_profiles_file"])
    if profiles_path.exists():
        with open(profiles_path, "r", encoding="utf-8") as f:
            loaded = json.load(f) or {}
        merged = _deep_merge(defaults, loaded)
    else:
        merged = defaults

    if merged.get("default_profile") not in merged.get("profiles", {}):
        merged["default_profile"] = defaults["default_profile"]
    return merged


def list_model_profiles() -> dict[str, dict]:
    """Return the configured model profiles keyed by profile id."""
    return load_model_profiles().get("profiles", {})


def resolve_model_settings(
    settings: dict,
    *,
    model_profile: str | None = None,
    profiles_data: dict | None = None,
) -> tuple[str, dict]:
    """Resolve the active model profile into the runtime model settings."""
    profiles_data = profiles_data or load_model_profiles()
    profiles = profiles_data.get("profiles", {})
    legacy_model = deepcopy(settings.get("model", {}) or {})

    selected_profile = model_profile or settings.get("model_profile") or settings.get("active_model_profile")
    if not selected_profile:
        legacy_backend = legacy_model.get("backend")
        if legacy_backend == "qwen_remote":
            selected_profile = "qwen3_omni_wsl"
        elif legacy_backend == "minicpm":
            selected_profile = "minicpm_local"
        else:
            selected_profile = profiles_data.get("default_profile", "minicpm_local")

    if selected_profile not in profiles:
        selected_profile = profiles_data.get("default_profile", "minicpm_local")

    # settings.yaml still carries historical MiniCPM defaults in model.*.
    # When an explicit non-MiniCPM profile is selected, those legacy values
    # should not override the chosen profile's model identity.
    if selected_profile != "minicpm_local" and not legacy_model.get("backend"):
        if legacy_model.get("name") == "openbmb/MiniCPM-o-4_5":
            legacy_model.pop("name", None)
        if legacy_model.get("dtype") == "bfloat16":
            legacy_model.pop("dtype", None)
        if legacy_model.get("device") == "cuda":
            legacy_model.pop("device", None)

    resolved_model = _deep_merge(profiles.get(selected_profile, {}), legacy_model)
    resolved_model.setdefault("backend", "minicpm")
    resolved_model.setdefault("quantization", "none")
    resolved_model.setdefault("auto_update", True)

    if resolved_model.get("backend") == "qwen_remote":
        remote = resolved_model.setdefault("remote", {})
        remote.setdefault("base_url", "http://127.0.0.1:8000/v1")
        remote.setdefault("api_key", None)
        remote.setdefault("timeout_s", 120)
        remote.setdefault("endpoints", _default_remote_endpoints())
        remote.setdefault("model_name", resolved_model.get("name"))
        resolved_model.setdefault("name", remote.get("model_name"))
    elif resolved_model.get("backend") == "qwen_transformers":
        resolved_model.setdefault("name", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        resolved_model.setdefault("device_map", "auto")
        resolved_model.setdefault("torch_dtype", "bfloat16")
        resolved_model.setdefault("attn_implementation", "sdpa")
        resolved_model.setdefault("speaker", "Ethan")
        resolved_model.setdefault("use_audio_in_video", True)
        resolved_model.setdefault("local_files_only", not resolved_model.get("auto_update", True))
    elif resolved_model.get("backend") == "gemma_transformers":
        resolved_model.setdefault("name", "Gemma 4 E4B IT")
        resolved_model.setdefault("checkpoint", r"D:\OmniChatModels\gemma4-e4b-it-official\hf")
        resolved_model.setdefault("device_map", "auto")
        resolved_model.setdefault("torch_dtype", "bfloat16")
        resolved_model.setdefault("attn_implementation", "sdpa")
        resolved_model.setdefault("speech_backend", "none")
        resolved_model.setdefault("use_audio_in_video", True)
        resolved_model.setdefault("local_files_only", not resolved_model.get("auto_update", True))
    elif resolved_model.get("backend") == "qwen_llamacpp":
        resolved_model.setdefault("name", "Qwen/Qwen3.5-27B")
        llama_cpp = resolved_model.setdefault("llama_cpp", {})
        llama_cpp.setdefault("llama_root", str((Path(tempfile.gettempdir()) / "llama-cpp-qwen35-test" / "llama.cpp").resolve()))
        llama_cpp.setdefault("cli_path", None)
        llama_cpp.setdefault("model_path", r"D:\OmniChatModels\qwen35-27b-official\gguf\Qwen-Qwen3.5-27B-27B-Q4_K_M-00001-of-00002.gguf")
        llama_cpp.setdefault("mmproj_path", r"D:\OmniChatModels\qwen35-27b-official\gguf\mmproj-Qwen-Qwen3.5-27B-F16.gguf")
        llama_cpp.setdefault("n_gpu_layers", 99)
        llama_cpp.setdefault("flash_attn", True)
        llama_cpp.setdefault("context_length", 8192)
        llama_cpp.setdefault("use_jinja", True)
        llama_cpp.setdefault("timeout_s", 120.0)
    elif resolved_model.get("backend") == "gemma_llamacpp":
        resolved_model.setdefault("name", "Gemma 4 S-Size")
        llama_cpp = resolved_model.setdefault("llama_cpp", {})
        llama_cpp.setdefault("llama_root", str((Path(tempfile.gettempdir()) / "llama-cpp-gemma4-test" / "llama.cpp").resolve()))
        llama_cpp.setdefault("cli_path", None)
        llama_cpp.setdefault("model_path", r"D:\OmniChatModels\gemma4-ssize-official\gguf\gemma4-ssize-it-Q4_K_M.gguf")
        llama_cpp.setdefault("mmproj_path", r"D:\OmniChatModels\gemma4-ssize-official\gguf\mmproj-gemma4-ssize-it-f16.gguf")
        llama_cpp.setdefault("n_gpu_layers", 99)
        llama_cpp.setdefault("flash_attn", True)
        llama_cpp.setdefault("context_length", 8192)
        llama_cpp.setdefault("use_jinja", True)
        llama_cpp.setdefault("timeout_s", 120.0)
    else:
        resolved_model.setdefault("name", "openbmb/MiniCPM-o-4_5")
        resolved_model.setdefault("dtype", "bfloat16")
        resolved_model.setdefault("device", "cuda")

    return selected_profile, resolved_model


def load_settings(*, model_profile: str | None = None) -> dict:
    """Load args/settings.yaml and resolve the active model profile."""
    settings = _default_settings()
    settings_path = BASE_DIR / "args" / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        settings = _deep_merge(settings, loaded)

    profiles_data = load_model_profiles(profiles_file=settings.get("model_profiles_file"))
    active_profile, resolved_model = resolve_model_settings(
        settings,
        model_profile=model_profile,
        profiles_data=profiles_data,
    )
    profile_inference = profiles_data.get("profiles", {}).get(active_profile, {}).get("inference", {})
    if isinstance(profile_inference, dict) and profile_inference:
        settings["inference"] = _deep_merge(settings.get("inference", {}), profile_inference)
    settings["model"] = resolved_model
    settings["model_profile"] = active_profile
    settings["active_model_profile"] = active_profile
    settings["available_model_profiles"] = {
        profile_id: {
            "display_name": profile.get("display_name", profile_id),
            "backend": profile.get("backend", "minicpm"),
            "name": profile.get("name"),
            "transport": profile.get("transport"),
        }
        for profile_id, profile in profiles_data.get("profiles", {}).items()
    }
    return settings


# ΓöÇΓöÇ Voice command detection ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

VOICE_COMMAND_PATTERNS = [
    re.compile(r"(?:change|switch)\s+(?:the\s+)?voice\s+to\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:change|switch)\s+to\s+(.+?)(?:'s)?\s+voice", re.IGNORECASE),
    re.compile(r"(?:use|try)\s+(.+?)(?:'s)?\s+voice", re.IGNORECASE),
    re.compile(r"sound\s+like\s+(.+)", re.IGNORECASE),
    re.compile(r"speak\s+(?:like|as)\s+(.+)", re.IGNORECASE),
    re.compile(r"(?:go\s+back\s+to|use)\s+(?:the\s+)?default\s+voice", re.IGNORECASE),
]


def detect_voice_command(text: str) -> str | None:
    """
    Check if text contains a voice-switch command.
    Returns the requested voice name, 'default' for default voice, or None.
    """
    # Check for default voice request
    if re.search(r"(?:use|go\s+back\s+to|switch\s+to)\s+(?:the\s+)?default\s+voice", text, re.IGNORECASE):
        return "default"

    for pattern in VOICE_COMMAND_PATTERNS:
        match = pattern.search(text)
        if match:
            name = match.group(1).strip().rstrip(".")
            return name

    return None


# ΓöÇΓöÇ Audio normalization ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def normalize_audio_input(sr: int, audio_data: np.ndarray) -> np.ndarray:
    """Convert raw audio (any sr, any dtype, any channels) to float32 mono 16kHz.

    Works for both Gradio (sr, ndarray) tuples and sounddevice (16kHz float32) input.
    """
    if audio_data.dtype != np.float32:
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        else:
            audio_data = audio_data.astype(np.float32)

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

    return audio_data


# ΓöÇΓöÇ Voice reference truncation ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def get_truncated_voice_ref(
    voice_ref: np.ndarray | None,
    sample_length_s: float,
) -> np.ndarray | None:
    """Truncate a voice reference to the configured sample length."""
    if voice_ref is None:
        return None
    max_samples = int(sample_length_s * 16000)
    if len(voice_ref) > max_samples:
        return voice_ref[:max_samples]
    return voice_ref


def build_chat_messages_from_history(
    chat_history: list[tuple[str, str]],
    current_user_content=None,
    *,
    max_entries: int = 20,
) -> list[dict]:
    """Build model messages from UI chat history.

    The UI stores display-only entries such as partial output, system notices,
    and voice placeholders. The model should only receive real user/assistant
    turns. When the current turn is non-textual (for example, live audio), pass
    it via ``current_user_content`` and it will be appended as the latest user
    message.
    """

    relevant_entries: list[tuple[str, str]] = []
    for role, text in chat_history:
        if role not in {"user", "assistant", "_assistant_hidden"}:
            continue
        if not text:
            continue
        if role == "user" and text == "[voice input]":
            continue
        relevant_entries.append((role, text))

    messages: list[dict] = []
    for role, text in relevant_entries[-max_entries:]:
        model_role = "assistant" if role == "_assistant_hidden" else role
        messages.append({"role": model_role, "content": [text]})

    if current_user_content is not None:
        if isinstance(current_user_content, list):
            content = current_user_content
        else:
            content = [current_user_content]
        if content:
            messages.append({"role": "user", "content": content})

    return messages
