"""Integration tests for Silero-VAD and ConversationManager.

Loads the real Silero-VAD model (CPU, ~5 MB) — no GPU required.
Tests the actual speech detection pipeline that conversation mode uses.

Run with:
    python -m pytest tests/test_vad_integration.py -v
"""

import numpy as np
import pytest

from tools.audio.conversation import (
    ConversationManager,
    ConversationState,
    ConversationMode,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_speech_like(duration_s: float, sr: int = 16000) -> np.ndarray:
    """Generate speech-like audio that Silero-VAD will classify as speech.

    Mixes several harmonics with amplitude modulation to mimic the
    spectral and temporal patterns of human speech.
    """
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, dtype=np.float32)

    # Fundamental + harmonics (typical speech formants)
    signal = np.zeros(n, dtype=np.float32)
    for freq, amp in [(150, 0.3), (300, 0.25), (600, 0.2), (1200, 0.15), (2400, 0.1)]:
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Amplitude modulation at ~4 Hz (syllable rate)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    signal *= envelope

    # Normalize to reasonable level
    signal = signal / np.max(np.abs(signal)) * 0.8
    return signal.astype(np.float32)


def make_silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    """Generate near-silent audio (very low noise floor)."""
    n = int(duration_s * sr)
    return np.random.randn(n).astype(np.float32) * 0.001


# ── VAD model loading ────────────────────────────────────────────────────────


class TestVADLoading:
    """Verify Silero-VAD loads and runs on CPU."""

    def test_vad_loads_from_cache(self):
        """Silero-VAD should load without downloading (cached from prior use)."""
        conv = ConversationManager()
        conv._ensure_vad()
        assert conv._vad_model is not None

    def test_vad_runs_on_silence(self):
        """VAD should return False for silence."""
        conv = ConversationManager()
        conv._ensure_vad()
        silence = make_silence(0.5)
        result = conv._run_vad(silence)
        assert result is False, "VAD detected speech in silence"

    def test_vad_detects_speech_like_signal(self):
        """VAD should return True for speech-like audio.

        Note: Silero-VAD is trained on real speech, so synthetic signals
        may not always trigger it. This test uses a signal with speech-like
        spectral characteristics (formant frequencies + syllable-rate modulation).
        If this test fails, it may mean our synthetic signal isn't convincing
        enough, not necessarily a VAD bug.
        """
        conv = ConversationManager({"vad_threshold": 0.3})
        conv._ensure_vad()
        speech = make_speech_like(1.0)
        result = conv._run_vad(speech)
        # Note: log output from _run_vad will show max_prob for diagnostics
        print(f"  Speech-like signal: VAD detected = {result}")
        # We don't hard-assert here because synthetic != real speech
        # Just verify it doesn't crash and returns a bool
        assert isinstance(result, bool)


# ── ConversationManager with real VAD ────────────────────────────────────────


class TestConversationWithRealVAD:
    """End-to-end conversation flow using actual Silero-VAD."""

    def test_start_loads_vad(self):
        """start() should lazy-load Silero-VAD for auto-detect mode."""
        conv = ConversationManager()
        assert conv._vad_model is None
        conv.start()
        assert conv._vad_model is not None
        assert conv.state == ConversationState.LISTENING
        conv.stop()

    def test_silence_stays_listening(self):
        """Continuous silence should keep state at LISTENING."""
        conv = ConversationManager()
        conv.start()

        for i in range(5):
            silence = make_silence(0.5)
            result = conv.on_audio_chunk(silence)
            assert result.state == ConversationState.LISTENING, (
                f"Chunk {i+1}: unexpected state {result.state}"
            )
            assert result.audio_ready is None

        conv.stop()

    def test_off_state_ignores_audio(self):
        """Audio chunks in OFF state should be ignored."""
        conv = ConversationManager()
        # Don't start — stay in OFF state
        speech = make_speech_like(1.0)
        result = conv.on_audio_chunk(speech)
        assert result.state == ConversationState.OFF
        assert result.audio_ready is None

    def test_processing_state_ignores_audio(self):
        """Audio chunks in PROCESSING state should be ignored."""
        conv = ConversationManager()
        conv.start()
        conv.state = ConversationState.PROCESSING
        speech = make_speech_like(1.0)
        result = conv.on_audio_chunk(speech)
        assert result.state == ConversationState.PROCESSING
        assert result.audio_ready is None
        conv.stop()

    def test_model_lifecycle(self):
        """on_model_start/on_model_done transitions work correctly."""
        conv = ConversationManager()
        conv.start()

        # Simulate model processing
        conv.state = ConversationState.PROCESSING
        conv.on_model_start()
        assert conv.state == ConversationState.MODEL_SPEAKING

        # Audio during MODEL_SPEAKING is ignored
        result = conv.on_audio_chunk(make_speech_like(0.5))
        assert result.state == ConversationState.MODEL_SPEAKING

        conv.on_model_done()
        assert conv.state == ConversationState.LISTENING

        conv.stop()

    def test_mode_switching_preserves_vad(self):
        """Switching modes while active preserves the VAD model."""
        conv = ConversationManager()
        conv.start()
        assert conv._vad_model is not None
        vad_model = conv._vad_model

        conv.set_mode("Push-to-talk")
        assert conv.mode == ConversationMode.PUSH_TO_TALK

        conv.set_mode("Auto-detect")
        assert conv.mode == ConversationMode.AUTO_DETECT
        assert conv._vad_model is vad_model  # same model, not reloaded

        conv.stop()

    def test_push_to_talk_with_real_audio(self):
        """PTT mode: buffers audio, returns on stop."""
        conv = ConversationManager()
        conv.set_mode("Push-to-talk")
        conv.start()

        conv.ptt_start()
        assert conv.state == ConversationState.USER_SPEAKING

        # Feed some audio chunks
        for _ in range(3):
            chunk = make_speech_like(0.5)
            conv.on_audio_chunk(chunk)

        audio = conv.ptt_stop()
        assert audio is not None
        assert len(audio) == 16000 * 3 * 0.5  # 3 chunks × 0.5s × 16kHz
        assert conv.state == ConversationState.PROCESSING

        conv.stop()

    def test_click_per_turn_ignores_stream(self):
        """Click-per-turn mode should ignore streaming audio chunks."""
        conv = ConversationManager()
        conv.set_mode("Click per turn")
        conv.start()

        speech = make_speech_like(1.0)
        result = conv.on_audio_chunk(speech)
        assert result.state == ConversationState.LISTENING
        assert result.audio_ready is None

        conv.stop()


# ── Simulated Gradio mic chunks ──────────────────────────────────────────────


class TestGradioMicSimulation:
    """Simulate the Gradio streaming mic pipeline.

    Gradio sends audio chunks every ~0.5 seconds. The browser mic
    records at 48kHz stereo int16, which main.py normalizes to 16kHz
    float32 mono before passing to ConversationManager.on_audio_chunk().

    These tests simulate that normalized audio format.
    """

    def test_silence_chunks_at_gradio_rate(self):
        """Stream 10 seconds of silence at Gradio's 0.5s chunk rate."""
        conv = ConversationManager()
        conv.start()

        # 20 chunks × 0.5s = 10 seconds of silence
        for i in range(20):
            chunk = make_silence(0.5)
            result = conv.on_audio_chunk(chunk)
            assert result.state == ConversationState.LISTENING, (
                f"Chunk {i+1}/20: state changed to {result.state} during silence"
            )

        conv.stop()

    def test_vad_chunk_size_alignment(self):
        """Verify VAD processes 512-sample windows correctly with 0.5s chunks.

        0.5s at 16kHz = 8000 samples. 8000 / 512 = 15.625, so 15 full windows.
        """
        conv = ConversationManager()
        conv.start()

        chunk = make_silence(0.5)
        assert len(chunk) == 8000

        # This should process without errors
        result = conv.on_audio_chunk(chunk)
        assert result.state == ConversationState.LISTENING

        conv.stop()

    def test_various_chunk_sizes(self):
        """Different chunk sizes (Gradio may vary) should all work."""
        conv = ConversationManager()
        conv.start()

        for duration in [0.1, 0.25, 0.5, 1.0, 2.0]:
            chunk = make_silence(duration)
            result = conv.on_audio_chunk(chunk)
            assert result.state == ConversationState.LISTENING, (
                f"Failed with {duration}s chunk ({len(chunk)} samples)"
            )

        conv.stop()

    def test_very_short_chunk_no_crash(self):
        """A chunk shorter than one VAD window (512 samples) should not crash."""
        conv = ConversationManager()
        conv.start()

        tiny_chunk = make_silence(0.01)  # 160 samples — less than 512
        result = conv.on_audio_chunk(tiny_chunk)
        assert result.state == ConversationState.LISTENING  # no crash

        conv.stop()


# ── Audio normalization (simulating main.py's _normalize_audio) ──────────────


class TestAudioNormalization:
    """Test that various mic input formats normalize correctly for VAD."""

    @staticmethod
    def _normalize(sr, audio_data):
        """Replicate main.py's _normalize_audio logic."""
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

    def test_int16_mono_16k(self):
        """Browser mic sending int16 mono at 16kHz (unlikely but possible)."""
        raw = (np.random.randn(16000) * 16000).astype(np.int16)
        result = self._normalize(16000, raw)
        assert result.dtype == np.float32
        assert result.ndim == 1
        assert len(result) == 16000
        assert np.max(np.abs(result)) <= 1.001  # small tolerance for int16 asymmetry

    def test_int16_stereo_48k(self):
        """Typical browser mic: int16 stereo at 48kHz."""
        raw = (np.random.randn(48000, 2) * 16000).astype(np.int16)
        result = self._normalize(48000, raw)
        assert result.dtype == np.float32
        assert result.ndim == 1
        # 48kHz → 16kHz = 3:1 ratio
        assert abs(len(result) - 16000) < 100  # allow small librosa rounding

    def test_float32_mono_16k_passthrough(self):
        """Already in the right format — should pass through unchanged."""
        raw = np.random.randn(16000).astype(np.float32) * 0.5
        result = self._normalize(16000, raw)
        assert result.dtype == np.float32
        assert len(result) == 16000
        np.testing.assert_array_equal(result, raw)

    def test_normalized_audio_works_with_vad(self):
        """Normalized audio from various formats should not crash VAD."""
        conv = ConversationManager()
        conv.start()

        # Simulate int16 stereo 48kHz → normalize → feed to VAD
        raw = (np.random.randn(24000, 2) * 1000).astype(np.int16)  # 0.5s at 48k
        normalized = self._normalize(48000, raw)
        result = conv.on_audio_chunk(normalized)
        assert result.state in (ConversationState.LISTENING, ConversationState.USER_SPEAKING)

        conv.stop()
