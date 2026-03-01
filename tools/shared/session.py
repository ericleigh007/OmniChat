"""
session.py -- Shared session helpers for both Gradio and PySide6 apps.

Extracted from main.py to avoid duplication between the two UI frontends.
All functions are pure logic with no UI framework dependency.
"""

import re
from pathlib import Path

import yaml
import numpy as np

BASE_DIR = Path(__file__).parent.parent.parent.resolve()


def load_settings() -> dict:
    """Load args/settings.yaml. Return defaults if missing."""
    settings_path = BASE_DIR / "args" / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "model": {"name": "openbmb/MiniCPM-o-4_5", "dtype": "bfloat16", "device": "cuda", "quantization": "none"},
        "audio": {
            "input_sample_rate": 16000, "output_sample_rate": 24000, "default_voice": None, "voices_dir": "voices",
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
            "enable_thinking": False,
        },
        "output": {"default_format": "auto", "save_dir": "outputs"},
        "server": {"host": "localhost", "port": 7860, "share": False},
    }


# ── Voice command detection ───────────────────────────────────────────────────

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


# ── Audio normalization ───────────────────────────────────────────────────────

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


# ── Voice reference truncation ────────────────────────────────────────────────

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
