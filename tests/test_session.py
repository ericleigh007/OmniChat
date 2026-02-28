"""Tests for tools/shared/session.py — shared helpers used by both Gradio and PySide6 apps."""

import numpy as np
import pytest
from pathlib import Path

from tools.shared.session import (
    load_settings,
    detect_voice_command,
    normalize_audio_input,
    get_truncated_voice_ref,
)


class TestLoadSettings:
    """Test YAML settings loader."""

    def test_returns_defaults_when_no_yaml(self, tmp_path, monkeypatch):
        """When settings.yaml doesn't exist, return hardcoded defaults."""
        import tools.shared.session as session_mod
        monkeypatch.setattr(session_mod, "BASE_DIR", tmp_path)
        settings = load_settings()
        assert settings["model"]["name"] == "openbmb/MiniCPM-o-4_5"
        assert settings["inference"]["temperature"] == 0.7
        assert settings["server"]["port"] == 7860

    def test_loads_from_yaml(self, tmp_path, monkeypatch):
        """When settings.yaml exists, parse and return it."""
        import tools.shared.session as session_mod
        (tmp_path / "args").mkdir()
        (tmp_path / "args" / "settings.yaml").write_text(
            "model:\n  name: test-model\ninference:\n  temperature: 0.5\n"
        )
        monkeypatch.setattr(session_mod, "BASE_DIR", tmp_path)
        settings = load_settings()
        assert settings["model"]["name"] == "test-model"
        assert settings["inference"]["temperature"] == 0.5


class TestNormalizeAudioInput:
    """Test audio format normalization."""

    def test_float32_16k_passthrough(self):
        """Float32 mono 16kHz should pass through unchanged."""
        audio = np.random.randn(16000).astype(np.float32)
        result = normalize_audio_input(16000, audio)
        np.testing.assert_array_equal(result, audio)

    def test_int16_conversion(self):
        """int16 audio should be converted to float32 in [-1, 1]."""
        audio = np.array([0, 16384, -16384, 32767], dtype=np.int16)
        result = normalize_audio_input(16000, audio)
        assert result.dtype == np.float32
        assert result[0] == 0.0
        assert abs(result[3] - 1.0) < 0.001  # 32767/32767 ~ 1.0

    def test_stereo_to_mono(self):
        """Stereo audio should be averaged to mono."""
        stereo = np.random.randn(16000, 2).astype(np.float32)
        result = normalize_audio_input(16000, stereo)
        assert result.ndim == 1
        assert len(result) == 16000
        np.testing.assert_array_almost_equal(result, stereo.mean(axis=-1))

    def test_resample_48k_to_16k(self):
        """48kHz audio should be resampled to 16kHz."""
        audio = np.random.randn(48000).astype(np.float32)
        result = normalize_audio_input(48000, audio)
        assert result.dtype == np.float32
        # Should be approximately 16000 samples (48000 * 16000/48000)
        assert abs(len(result) - 16000) < 10


class TestGetTruncatedVoiceRef:
    """Test voice reference truncation."""

    def test_none_returns_none(self):
        assert get_truncated_voice_ref(None, 5.0) is None

    def test_short_ref_not_truncated(self):
        """A 3-second clip with 5s limit should not be truncated."""
        ref = np.zeros(48000, dtype=np.float32)  # 3s at 16kHz
        result = get_truncated_voice_ref(ref, 5.0)
        assert len(result) == 48000

    def test_long_ref_truncated(self):
        """A 15-second clip with 5s limit should be truncated to 5s."""
        ref = np.zeros(240000, dtype=np.float32)  # 15s at 16kHz
        result = get_truncated_voice_ref(ref, 5.0)
        assert len(result) == 80000  # 5s * 16000

    def test_exact_length_not_truncated(self):
        """A clip exactly at the limit should not be truncated."""
        ref = np.zeros(80000, dtype=np.float32)  # 5s at 16kHz
        result = get_truncated_voice_ref(ref, 5.0)
        assert len(result) == 80000


class TestDetectVoiceCommand:
    """Test that voice command detection works from the shared module.

    Detailed pattern tests are in test_voice_commands.py — here we just
    verify the function is importable and functional from its new location.
    """

    def test_change_voice_to(self):
        assert detect_voice_command("change voice to Morgan Freeman") == "Morgan Freeman"

    def test_default_voice(self):
        assert detect_voice_command("use the default voice") == "default"

    def test_no_command(self):
        assert detect_voice_command("Hello, how are you?") is None
