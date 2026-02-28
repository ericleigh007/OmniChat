"""Shared pytest fixtures for OmniChat tests."""

import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_voices_dir(tmp_path):
    """Temp directory with dummy voice WAV files for voice_manager tests."""
    import soundfile as sf

    voices = tmp_path / "voices"
    voices.mkdir()

    # Create 3 dummy voice files (1s silence at 16kHz)
    for name in ["morgan_freeman", "david_attenborough", "james_earl_jones"]:
        audio = np.zeros(16000, dtype=np.float32)
        sf.write(str(voices / f"{name}.wav"), audio, 16000)

    return voices


@pytest.fixture
def tmp_outputs_dir(tmp_path):
    """Temp directory for save_output tests."""
    out = tmp_path / "outputs"
    out.mkdir()
    return out


@pytest.fixture
def sample_audio_16k():
    """1 second of 440Hz sine wave at 16kHz, float32 mono."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sample_audio_48k_stereo():
    """1 second of 440Hz sine wave at 48kHz, float32 stereo."""
    t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
    mono = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return np.stack([mono, mono * 0.5], axis=-1)


@pytest.fixture
def sample_audio_int16():
    """1 second of 440Hz sine wave at 16kHz, int16 mono."""
    t = np.linspace(0, 1, 16000, endpoint=False, dtype=np.float64)
    return (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)


# ── GPU / Model fixtures (session-scoped — loads model once) ────────────────

@pytest.fixture(scope="session")
def loaded_model():
    """Load MiniCPM-o 4.5 once for the entire test session.

    Skips all dependent tests if CUDA is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    except ImportError:
        pytest.skip("torch not installed")

    from tools.model.model_manager import get_model
    model, tokenizer = get_model()
    return model, tokenizer
