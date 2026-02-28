"""Tests for model_manager.py — mock-based tests for reset logic and message construction."""

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


class TestApplyFadeIn:
    """Test _apply_fade_in with exact numerical verification."""

    def test_known_values(self):
        from tools.model.model_manager import _apply_fade_in

        audio = np.ones(10, dtype=np.float32)
        result = _apply_fade_in(audio, n_samples=5)

        # Fade: [0.0, 0.25, 0.5, 0.75, 1.0] applied to ones
        np.testing.assert_array_almost_equal(
            result[:5], [0.0, 0.25, 0.5, 0.75, 1.0], decimal=4
        )
        # After fade: unchanged
        np.testing.assert_array_equal(result[5:], np.ones(5, dtype=np.float32))

    def test_zero_audio_stays_zero(self):
        from tools.model.model_manager import _apply_fade_in

        audio = np.zeros(1000, dtype=np.float32)
        result = _apply_fade_in(audio, n_samples=100)
        np.testing.assert_array_equal(result, audio)


class TestResetForGeneration:
    """Test _reset_for_generation with a mock model."""

    def _make_mock_model(self):
        """Create a mock model with the expected attributes."""
        model = MagicMock()
        model.tts = MagicMock()
        model.tts.audio_tokenizer = MagicMock()
        model.tts.audio_tokenizer.cache = "some_cache"
        model.tts.audio_tokenizer.hift_cache_dict = {"mel": "data"}
        model.tts.audio_tokenizer.stream_cache = "stream_data"
        return model

    @patch("tools.model.model_manager.torch")
    def test_calls_reset_session(self, mock_torch):
        from tools.model.model_manager import _reset_for_generation

        model = self._make_mock_model()
        _reset_for_generation(model, generate_audio=True)
        model.reset_session.assert_called_once()

    @patch("tools.model.model_manager.torch")
    def test_calls_cuda_empty_cache(self, mock_torch):
        from tools.model.model_manager import _reset_for_generation

        model = self._make_mock_model()
        _reset_for_generation(model, generate_audio=True)
        mock_torch.cuda.empty_cache.assert_called_once()

    @patch("tools.model.model_manager.torch")
    def test_deletes_stale_output_file(self, mock_torch, tmp_path):
        from tools.model.model_manager import _reset_for_generation

        stale = tmp_path / "old_response.wav"
        stale.write_text("stale audio")
        assert stale.exists()

        model = self._make_mock_model()
        _reset_for_generation(model, generate_audio=True, output_audio_path=str(stale))
        assert not stale.exists()

    @patch("tools.model.model_manager.torch")
    def test_preserves_token2wav_cache(self, mock_torch):
        """Token2wav cache must NOT be cleared — clearing causes TypeError when
        prompt_wav is None (default voice path). Voice features are safe to reuse."""
        from tools.model.model_manager import _reset_for_generation

        model = self._make_mock_model()
        _reset_for_generation(model, generate_audio=True)

        # Token2wav caches should be preserved
        assert model.tts.audio_tokenizer.cache == "some_cache"
        assert model.tts.audio_tokenizer.hift_cache_dict == {"mel": "data"}

    @patch("tools.model.model_manager.torch")
    def test_preserves_token2wav_when_no_audio(self, mock_torch):
        from tools.model.model_manager import _reset_for_generation

        model = self._make_mock_model()
        _reset_for_generation(model, generate_audio=False)

        assert model.tts.audio_tokenizer.cache == "some_cache"
        assert model.tts.audio_tokenizer.hift_cache_dict == {"mel": "data"}


class TestBuildVoiceSystemMsg:
    """Test _build_voice_system_msg delegates to model.get_sys_prompt()."""

    def test_default_voice(self):
        from tools.model.model_manager import _build_voice_system_msg

        model = MagicMock()
        model.get_sys_prompt.return_value = {"role": "system", "content": "default"}

        result = _build_voice_system_msg(model, None)

        model.get_sys_prompt.assert_called_once_with(ref_audio=None, mode="audio_assistant")
        assert result == {"role": "system", "content": "default"}

    def test_custom_voice(self):
        from tools.model.model_manager import _build_voice_system_msg

        model = MagicMock()
        voice = np.zeros(16000, dtype=np.float32)
        model.get_sys_prompt.return_value = {"role": "system", "content": ["clone", voice]}

        result = _build_voice_system_msg(model, voice)

        # Verify ref_audio was the numpy array
        call_args = model.get_sys_prompt.call_args
        assert call_args.kwargs["mode"] == "audio_assistant"
        np.testing.assert_array_equal(call_args.kwargs["ref_audio"], voice)


class TestStaticNormalizer:
    """Test _normalize_rms (static normalizer for voice refs)."""

    def test_silence_unchanged(self):
        from tools.model.model_manager import _normalize_rms

        silence = np.zeros(1000, dtype=np.float32)
        result = _normalize_rms(silence, target_rms=0.05, max_gain=10.0, peak_ceiling=0.99)
        np.testing.assert_array_equal(result, silence)

    def test_reaches_target_rms(self):
        from tools.model.model_manager import _normalize_rms

        # Sine wave with known RMS
        t = np.linspace(0, 1, 24000, dtype=np.float32)
        audio = 0.1 * np.sin(2 * np.pi * 440 * t)  # RMS ≈ 0.0707
        target = 0.05
        result = _normalize_rms(audio, target_rms=target, max_gain=10.0, peak_ceiling=0.99)
        actual_rms = np.sqrt(np.mean(result ** 2))
        assert abs(actual_rms - target) < 0.005, f"RMS {actual_rms:.4f} not near target {target}"

    def test_max_gain_caps_boost(self):
        from tools.model.model_manager import _normalize_rms

        # Very quiet signal: RMS ≈ 0.001
        quiet = np.full(1000, 0.001, dtype=np.float32)
        target = 0.05  # would need 50x gain
        result = _normalize_rms(quiet, target_rms=target, max_gain=10.0, peak_ceiling=0.99)
        actual_peak = np.max(np.abs(result))
        # Gain capped at 10x: peak should be ≈ 0.01, not 0.05
        assert actual_peak < 0.015, f"Peak {actual_peak:.4f} suggests gain exceeded max"

    def test_peak_ceiling_clamps(self):
        from tools.model.model_manager import _normalize_rms

        # Loud signal that would clip after gain
        loud = np.full(1000, 0.8, dtype=np.float32)
        result = _normalize_rms(loud, target_rms=0.9, max_gain=10.0, peak_ceiling=0.95)
        assert np.max(np.abs(result)) <= 0.951, "Peak exceeded ceiling"


class TestCompressor:
    """Test _compress_output (windowed RMS compressor with attack/release)."""

    def _make_test_signal(self, sr=24000):
        """Create a signal with a quiet section followed by a loud section."""
        t = np.linspace(0, 1, sr, dtype=np.float32)
        quiet = 0.02 * np.sin(2 * np.pi * 440 * t)  # 1s at -34 dBFS
        loud = 0.5 * np.sin(2 * np.pi * 440 * t)     # 1s at -6 dBFS
        return np.concatenate([quiet, loud])

    def test_disabled_passthrough(self):
        """When leveling.enabled=false, audio passes through unchanged."""
        import tools.model.model_manager as mm

        audio = np.random.randn(24000).astype(np.float32) * 0.1
        # Force config to disabled
        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = {**mm._LEVELING_DEFAULTS, "enabled": False}
            result = mm._compress_output(audio, 24000)
            np.testing.assert_array_equal(result, audio)
        finally:
            mm._leveling_cfg = old_cfg

    def test_loud_signal_is_reduced(self):
        """Signal above threshold should have its peaks reduced."""
        from tools.model.model_manager import _compress_output
        import tools.model.model_manager as mm

        # Use known config
        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = None  # force reload from defaults
            mm._leveling_cfg = mm._get_leveling_config()

            # Loud sine: peak 0.8, RMS ≈ 0.566 (-5 dBFS), well above -28 dBFS threshold
            t = np.linspace(0, 1, 24000, dtype=np.float32)
            loud = 0.8 * np.sin(2 * np.pi * 440 * t)
            input_rms = np.sqrt(np.mean(loud ** 2))

            result = _compress_output(loud, 24000)
            output_rms = np.sqrt(np.mean(result ** 2))

            # Compressor should reduce the level (not amplify it)
            assert output_rms < input_rms, (
                f"Output RMS {output_rms:.4f} should be less than input {input_rms:.4f}"
            )
        finally:
            mm._leveling_cfg = old_cfg

    def test_quiet_signal_boosted_by_makeup(self):
        """Signal below threshold gets makeup gain applied."""
        from tools.model.model_manager import _compress_output
        import tools.model.model_manager as mm

        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = None
            mm._leveling_cfg = mm._get_leveling_config()

            # Very quiet sine: RMS ≈ 0.007 (-43 dBFS), well below threshold
            t = np.linspace(0, 1, 24000, dtype=np.float32)
            quiet = 0.01 * np.sin(2 * np.pi * 440 * t)
            input_rms = np.sqrt(np.mean(quiet ** 2))

            result = _compress_output(quiet, 24000)
            output_rms = np.sqrt(np.mean(result ** 2))

            # Below threshold: no compression, but makeup gain applies → louder
            assert output_rms > input_rms, (
                f"Output RMS {output_rms:.4f} should be greater than input {input_rms:.4f}"
            )
        finally:
            mm._leveling_cfg = old_cfg

    def test_dynamic_range_reduced(self):
        """A signal with quiet and loud sections should have reduced dynamic range."""
        from tools.model.model_manager import _compress_output
        import tools.model.model_manager as mm

        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = None
            mm._leveling_cfg = mm._get_leveling_config()

            signal = self._make_test_signal()
            sr = 24000
            result = _compress_output(signal, sr)

            # Measure RMS of each half
            in_quiet_rms = np.sqrt(np.mean(signal[:sr] ** 2))
            in_loud_rms = np.sqrt(np.mean(signal[sr:] ** 2))
            out_quiet_rms = np.sqrt(np.mean(result[:sr] ** 2))
            out_loud_rms = np.sqrt(np.mean(result[sr:] ** 2))

            input_ratio = in_loud_rms / max(in_quiet_rms, 1e-10)
            output_ratio = out_loud_rms / max(out_quiet_rms, 1e-10)

            # Dynamic range should be narrower after compression
            assert output_ratio < input_ratio, (
                f"Output ratio {output_ratio:.1f} should be less than input ratio {input_ratio:.1f}"
            )
        finally:
            mm._leveling_cfg = old_cfg

    def test_peak_ceiling_respected(self):
        """Output should never exceed the peak ceiling."""
        from tools.model.model_manager import _compress_output
        import tools.model.model_manager as mm

        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = None
            mm._leveling_cfg = mm._get_leveling_config()
            ceiling = mm._leveling_cfg["_peak_ceiling_linear"]

            # Hot signal near clipping
            t = np.linspace(0, 1, 24000, dtype=np.float32)
            hot = 0.95 * np.sin(2 * np.pi * 440 * t)
            result = _compress_output(hot, 24000)

            assert np.max(np.abs(result)) <= ceiling + 0.001, "Exceeded peak ceiling"
        finally:
            mm._leveling_cfg = old_cfg

    def test_short_audio_doesnt_crash(self):
        """Compressor handles very short signals gracefully."""
        from tools.model.model_manager import _compress_output
        import tools.model.model_manager as mm

        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = None
            mm._leveling_cfg = mm._get_leveling_config()

            # 1 sample, empty, and 10 samples
            assert len(_compress_output(np.array([0.5], dtype=np.float32), 24000)) == 1
            assert len(_compress_output(np.array([], dtype=np.float32), 24000)) == 0
            result = _compress_output(np.ones(10, dtype=np.float32) * 0.3, 24000)
            assert len(result) == 10
        finally:
            mm._leveling_cfg = old_cfg

    def test_silence_unchanged(self):
        """Pure silence should pass through without amplification."""
        from tools.model.model_manager import _compress_output
        import tools.model.model_manager as mm

        old_cfg = mm._leveling_cfg
        try:
            mm._leveling_cfg = None
            mm._leveling_cfg = mm._get_leveling_config()

            silence = np.zeros(24000, dtype=np.float32)
            result = _compress_output(silence, 24000)
            # Should remain essentially silent (makeup gain on ~0 is still ~0)
            assert np.max(np.abs(result)) < 0.001
        finally:
            mm._leveling_cfg = old_cfg


class TestTurnCounter:
    """Test that the turn counter increments."""

    def test_turn_counter_initial(self):
        import tools.model.model_manager as mm
        # Just verify the counter exists and is an int
        assert isinstance(mm._turn_count, int)


class TestChatStreamingPreprocessing:
    """Test chat_streaming() message preparation logic (mock-based)."""

    def _make_mock_model(self):
        """Create a mock model with streaming methods."""
        model = MagicMock()
        model.get_sys_prompt.return_value = {"role": "system", "content": ["tts system prompt"]}
        model.streaming_prefill.return_value = "prompt_text"
        # streaming_generate yields (waveform_chunk, text_chunk) then stops
        model.streaming_generate.return_value = iter([])
        return model

    @patch("tools.model.model_manager.get_model")
    @patch("tools.model.model_manager.torch")
    def test_prefills_system_then_user(self, mock_torch, mock_get_model):
        """System TTS prompt is prefilled first, then user message."""
        from tools.model.model_manager import chat_streaming

        model = self._make_mock_model()
        mock_get_model.return_value = (model, MagicMock())

        # Consume the generator
        list(chat_streaming(
            messages=[{"role": "user", "content": ["Hello"]}],
            generate_audio=True,
        ))

        # Should have 2 prefill calls: system + user
        assert model.streaming_prefill.call_count == 2
        calls = model.streaming_prefill.call_args_list

        # First call: system message
        assert calls[0].kwargs["msgs"][0]["role"] == "system"
        # Second call: user message
        assert calls[1].kwargs["msgs"][0]["role"] == "user"

    @patch("tools.model.model_manager.get_model")
    @patch("tools.model.model_manager.torch")
    def test_voice_ref_inits_cache(self, mock_torch, mock_get_model):
        """Voice reference triggers init_token2wav_cache before prefill."""
        from tools.model.model_manager import chat_streaming

        model = self._make_mock_model()
        mock_get_model.return_value = (model, MagicMock())
        voice = np.zeros(16000, dtype=np.float32)

        list(chat_streaming(
            messages=[{"role": "user", "content": ["Hello"]}],
            voice_ref=voice,
            generate_audio=True,
        ))

        # init_token2wav_cache should be called twice: once with voice, once to restore default
        assert model.init_token2wav_cache.call_count == 2
        # First call with voice ref
        first_call = model.init_token2wav_cache.call_args_list[0]
        np.testing.assert_array_equal(first_call.args[0], voice)
        # Second call restores default (silence)
        second_call = model.init_token2wav_cache.call_args_list[1]
        assert np.allclose(second_call.args[0], 0.0)

    @patch("tools.model.model_manager.get_model")
    @patch("tools.model.model_manager.torch")
    def test_no_audio_skips_system_prompt(self, mock_torch, mock_get_model):
        """When generate_audio=False, no system TTS prompt is prefilled."""
        from tools.model.model_manager import chat_streaming

        model = self._make_mock_model()
        mock_get_model.return_value = (model, MagicMock())

        list(chat_streaming(
            messages=[{"role": "user", "content": ["Hello"]}],
            generate_audio=False,
        ))

        # Only 1 prefill call: user message (no system prompt)
        assert model.streaming_prefill.call_count == 1
        calls = model.streaming_prefill.call_args_list
        assert calls[0].kwargs["msgs"][0]["role"] == "user"

    @patch("tools.model.model_manager.get_model")
    @patch("tools.model.model_manager.torch")
    def test_yields_audio_chunks(self, mock_torch, mock_get_model):
        """Generator yields audio chunks from streaming_generate."""
        import torch as real_torch
        from tools.model.model_manager import chat_streaming

        model = self._make_mock_model()
        mock_get_model.return_value = (model, MagicMock())

        # Mock streaming_generate to yield 2 chunks
        chunk1 = real_torch.ones(24000)
        chunk2 = real_torch.ones(12000) * 0.5
        model.streaming_generate.return_value = iter([
            (chunk1, "Hello "),
            (chunk2, "world!"),
        ])

        results = list(chat_streaming(
            messages=[{"role": "user", "content": ["Hi"]}],
            generate_audio=True,
        ))

        assert len(results) == 2
        assert len(results[0][0]) == 24000  # first audio chunk
        assert results[0][1] == "Hello "
        assert len(results[1][0]) == 12000  # second audio chunk
        assert results[1][1] == "world!"
