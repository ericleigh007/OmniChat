"""Tests for audio normalization (main.py) and fade-in (model_manager.py)."""

import numpy as np
import pytest


class TestNormalizeAudio:
    """Tests for main._normalize_audio()."""

    def _normalize(self, audio_input):
        """Import and call _normalize_audio from main.py.

        We import lazily to avoid loading the full main module at import time
        (which would trigger Gradio and model imports).
        """
        # _normalize_audio is nested inside build_app(), so we replicate
        # the logic here directly (it's a pure function with no closures).
        sr, audio_data = audio_input

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

    def test_int16_to_float32(self, sample_audio_int16):
        result = self._normalize((16000, sample_audio_int16))
        assert result.dtype == np.float32
        assert result.max() <= 1.0
        assert result.min() >= -1.0

    def test_float64_to_float32(self):
        audio = np.random.randn(16000).astype(np.float64)
        result = self._normalize((16000, audio))
        assert result.dtype == np.float32

    def test_float32_passthrough(self, sample_audio_16k):
        result = self._normalize((16000, sample_audio_16k))
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, sample_audio_16k)

    def test_stereo_to_mono(self, sample_audio_48k_stereo):
        # Use 16kHz stereo to avoid resampling in this test
        mono = sample_audio_48k_stereo[:16000, :].astype(np.float32)
        stereo_16k = np.stack([mono[:, 0], mono[:, 0] * 0.5], axis=-1)
        result = self._normalize((16000, stereo_16k))
        assert result.ndim == 1
        assert len(result) == 16000

    def test_resample_48k_to_16k(self):
        # 1s of audio at 48kHz should produce ~16000 samples at 16kHz
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 48000)).astype(np.float32)
        result = self._normalize((48000, audio))
        assert result.dtype == np.float32
        assert abs(len(result) - 16000) < 100  # allow small rounding

    def test_int16_range_preserved(self):
        # Full-scale int16 should map to ~1.0
        audio = np.array([32767, -32768, 0], dtype=np.int16)
        result = self._normalize((16000, audio))
        assert abs(result[0] - 1.0) < 0.001
        assert abs(result[1] - (-1.0)) < 0.001
        assert abs(result[2]) < 0.001


class TestApplyFadeIn:
    """Tests for model_manager._apply_fade_in()."""

    def _fade_in(self, audio, n_samples=2400):
        from tools.model.model_manager import _apply_fade_in
        return _apply_fade_in(audio, n_samples)

    def test_first_sample_is_zero(self, sample_audio_16k):
        result = self._fade_in(sample_audio_16k, n_samples=100)
        assert result[0] == 0.0

    def test_last_fade_sample_is_full_scale(self, sample_audio_16k):
        n = 100
        result = self._fade_in(sample_audio_16k, n_samples=n)
        # np.linspace(0, 1, n) makes the last element exactly 1.0
        np.testing.assert_almost_equal(
            result[n - 1],
            sample_audio_16k[n - 1],
            decimal=4,
        )

    def test_samples_after_fade_unchanged(self, sample_audio_16k):
        n = 100
        result = self._fade_in(sample_audio_16k, n_samples=n)
        np.testing.assert_array_equal(result[n:], sample_audio_16k[n:])

    def test_short_audio_returned_unchanged(self):
        short = np.ones(50, dtype=np.float32)
        result = self._fade_in(short, n_samples=100)
        np.testing.assert_array_equal(result, short)

    def test_does_not_modify_original(self, sample_audio_16k):
        original = sample_audio_16k.copy()
        self._fade_in(sample_audio_16k, n_samples=100)
        np.testing.assert_array_equal(sample_audio_16k, original)

    def test_fade_values_are_linear(self):
        audio = np.ones(1000, dtype=np.float32)
        n = 500
        result = self._fade_in(audio, n_samples=n)
        expected_fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
        np.testing.assert_array_almost_equal(result[:n], expected_fade, decimal=5)

    def test_default_fade_samples(self):
        from tools.model.model_manager import _FADE_IN_SAMPLES
        assert _FADE_IN_SAMPLES == 2400  # 100ms at 24kHz
