"""Tests for ConversationManager state machine and VAD integration.

Mock-based — no GPU or model required.
"""

from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from tools.audio.conversation import (
    ConversationManager,
    ConversationState,
    ConversationMode,
    ChunkResult,
)


# ── State machine basics ──────────────────────────────────────────────────────


class TestConversationLifecycle:
    """Test start/stop lifecycle and initial state."""

    def test_initial_state_is_off(self):
        conv = ConversationManager()
        assert conv.state == ConversationState.OFF
        assert conv.active is False

    def test_start_transitions_to_listening(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()  # skip real VAD loading
        conv.start()
        assert conv.state == ConversationState.LISTENING
        assert conv.active is True

    def test_stop_transitions_to_off(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.start()
        conv.stop()
        assert conv.state == ConversationState.OFF
        assert conv.active is False

    def test_stop_clears_buffer(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.start()
        conv._audio_buffer = [np.zeros(1000, dtype=np.float32)]
        conv.stop()
        assert conv._audio_buffer == []

    def test_default_mode_is_auto_detect(self):
        conv = ConversationManager()
        assert conv.mode == ConversationMode.AUTO_DETECT


# ── Auto-detect mode (VAD) ────────────────────────────────────────────────────


class TestAutoDetect:
    """Test VAD-based speech boundary detection."""

    def _make_conv(self, silence_threshold_s=1.5):
        """Create a ConversationManager with a mocked VAD model."""
        conv = ConversationManager({"silence_threshold_s": silence_threshold_s})
        conv._vad_model = MagicMock()
        conv.start()
        return conv

    def _make_chunk(self, duration_s=0.5, sr=16000):
        """Create a dummy audio chunk."""
        return np.zeros(int(duration_s * sr), dtype=np.float32)

    def test_silence_stays_listening(self):
        conv = self._make_conv()
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)  # below threshold
        chunk = self._make_chunk()
        result = conv.on_audio_chunk(chunk)
        assert result.state == ConversationState.LISTENING
        assert result.audio_ready is None

    def test_speech_transitions_to_user_speaking(self):
        conv = self._make_conv()
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)  # above threshold
        chunk = self._make_chunk()
        result = conv.on_audio_chunk(chunk)
        assert result.state == ConversationState.USER_SPEAKING
        assert result.audio_ready is None

    def test_speech_then_silence_triggers_processing(self):
        conv = self._make_conv(silence_threshold_s=0.5)

        # Simulate speech
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        chunk_speech = self._make_chunk(0.5)
        result = conv.on_audio_chunk(chunk_speech)
        assert result.state == ConversationState.USER_SPEAKING

        # Simulate silence (enough to exceed threshold)
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
        chunk_silence = self._make_chunk(0.6)  # > 0.5s silence threshold
        result = conv.on_audio_chunk(chunk_silence)
        assert result.state == ConversationState.PROCESSING
        assert result.audio_ready is not None
        # Audio should contain both the speech and silence chunks
        expected_len = len(chunk_speech) + len(chunk_silence)
        assert len(result.audio_ready) == expected_len

    def test_speech_continues_no_trigger(self):
        conv = self._make_conv(silence_threshold_s=2.0)

        # First speech chunk
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(self._make_chunk(0.5))
        assert conv.state == ConversationState.USER_SPEAKING

        # Second speech chunk — still speaking
        conv.on_audio_chunk(self._make_chunk(0.5))
        assert conv.state == ConversationState.USER_SPEAKING
        assert conv._silence_chunks == 0

    def test_brief_silence_not_enough(self):
        conv = self._make_conv(silence_threshold_s=1.5)

        # Start speaking
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(self._make_chunk(0.5))
        assert conv.state == ConversationState.USER_SPEAKING

        # Brief silence (0.5s < 1.5s threshold)
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
        result = conv.on_audio_chunk(self._make_chunk(0.5))
        assert result.state == ConversationState.USER_SPEAKING
        assert result.audio_ready is None

    def test_silence_resets_on_new_speech(self):
        conv = self._make_conv(silence_threshold_s=1.5)

        # Start speaking
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(self._make_chunk(0.5))

        # Brief silence
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
        conv.on_audio_chunk(self._make_chunk(0.3))
        assert conv._silence_chunks > 0

        # Resume speaking — silence counter resets
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(self._make_chunk(0.5))
        assert conv._silence_chunks == 0

    def test_chunks_ignored_when_off(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        # Don't start — state is OFF
        result = conv.on_audio_chunk(self._make_chunk())
        assert result.state == ConversationState.OFF
        assert result.audio_ready is None

    def test_chunks_ignored_during_processing(self):
        conv = self._make_conv()
        conv.state = ConversationState.PROCESSING
        result = conv.on_audio_chunk(self._make_chunk())
        assert result.state == ConversationState.PROCESSING
        assert result.audio_ready is None

    def test_model_speaking_with_barge_in_disabled(self):
        """With barge-in disabled, chunks during MODEL_SPEAKING are ignored."""
        conv = ConversationManager({"barge_in_enabled": False})
        conv._vad_model = MagicMock()
        conv.start()
        conv.state = ConversationState.MODEL_SPEAKING
        result = conv.on_audio_chunk(self._make_chunk())
        assert result.state == ConversationState.MODEL_SPEAKING
        assert result.audio_ready is None
        assert result.barge_in is False


# ── Model lifecycle ───────────────────────────────────────────────────────────


class TestModelLifecycle:
    """Test on_model_start/on_model_done transitions."""

    def test_model_start_sets_state(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()
        assert conv.state == ConversationState.MODEL_SPEAKING

    def test_model_done_resumes_listening(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()
        conv.on_model_done()
        assert conv.state == ConversationState.LISTENING

    def test_model_done_resets_vad(self):
        conv = ConversationManager()
        mock_model = MagicMock()
        conv._vad_model = mock_model
        conv.start()
        mock_model.reset_states.reset_mock()  # clear the start() call

        conv.on_model_start()
        conv.on_model_done()
        mock_model.reset_states.assert_called_once()

    def test_idle_transitions_to_listening_on_chunk(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
        conv.start()
        conv.state = ConversationState.IDLE
        chunk = np.zeros(8000, dtype=np.float32)
        result = conv.on_audio_chunk(chunk)
        assert result.state == ConversationState.LISTENING


# ── Mode switching ────────────────────────────────────────────────────────────


class TestModeSwitching:
    """Test interaction mode changes."""

    def test_set_auto_detect(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.set_mode("Auto-detect")
        assert conv.mode == ConversationMode.AUTO_DETECT

    def test_set_push_to_talk(self):
        conv = ConversationManager()
        conv.set_mode("Push-to-talk")
        assert conv.mode == ConversationMode.PUSH_TO_TALK

    def test_set_click_per_turn(self):
        conv = ConversationManager()
        conv.set_mode("Click per turn")
        assert conv.mode == ConversationMode.CLICK_PER_TURN

    def test_unknown_mode_defaults_to_auto(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.set_mode("nonexistent")
        assert conv.mode == ConversationMode.AUTO_DETECT

    def test_set_mode_with_enum(self):
        """set_mode should accept ConversationMode enum directly."""
        conv = ConversationManager()
        conv.set_mode(ConversationMode.PUSH_TO_TALK)
        assert conv.mode == ConversationMode.PUSH_TO_TALK

    def test_set_mode_with_enum_auto_detect(self):
        """set_mode with AUTO_DETECT enum should load VAD if active."""
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.set_mode(ConversationMode.AUTO_DETECT)
        assert conv.mode == ConversationMode.AUTO_DETECT


# ── Push-to-talk ──────────────────────────────────────────────────────────────


class TestPushToTalk:
    """Test push-to-talk recording flow."""

    def test_ptt_start_sets_state(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.mode = ConversationMode.PUSH_TO_TALK
        conv.start()
        conv.ptt_start()
        assert conv.state == ConversationState.USER_SPEAKING
        assert conv._ptt_recording is True

    def test_ptt_stop_returns_audio(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.mode = ConversationMode.PUSH_TO_TALK
        conv.start()
        conv.ptt_start()

        chunk = np.ones(8000, dtype=np.float32) * 0.5
        conv.on_audio_chunk(chunk)
        conv.on_audio_chunk(chunk)

        audio = conv.ptt_stop()
        assert audio is not None
        assert len(audio) == 16000
        assert conv.state == ConversationState.PROCESSING

    def test_ptt_stop_empty_returns_none(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.mode = ConversationMode.PUSH_TO_TALK
        conv.start()
        conv.ptt_start()
        audio = conv.ptt_stop()  # no chunks pushed
        assert audio is None
        assert conv.state == ConversationState.LISTENING

    def test_ptt_ignored_when_off(self):
        conv = ConversationManager()
        conv.ptt_start()  # state is OFF
        assert conv.state == ConversationState.OFF


# ── State display ─────────────────────────────────────────────────────────────


class TestStateDisplay:
    """Test HTML state indicator rendering."""

    def test_off_state_html(self):
        conv = ConversationManager()
        html = conv.format_state_html()
        assert "Conversation off" in html
        assert "#888888" in html

    def test_listening_state_html(self):
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.start()
        html = conv.format_state_html()
        assert "Listening..." in html
        assert "#22c55e" in html
        assert "convpulse" in html  # pulsing animation

    def test_user_speaking_html(self):
        conv = ConversationManager()
        conv.state = ConversationState.USER_SPEAKING
        html = conv.format_state_html()
        assert "speaking" in html.lower()
        assert "#3b82f6" in html

    def test_mode_label_shown(self):
        conv = ConversationManager()
        html = conv.format_state_html()
        assert "Auto-detect" in html

    def test_processing_html(self):
        conv = ConversationManager()
        conv.state = ConversationState.PROCESSING
        html = conv.format_state_html()
        assert "Thinking..." in html
        assert "#eab308" in html

    def test_model_speaking_html(self):
        conv = ConversationManager()
        conv.state = ConversationState.MODEL_SPEAKING
        html = conv.format_state_html()
        assert "Responding..." in html
        assert "#a855f7" in html


# ── ChunkResult ───────────────────────────────────────────────────────────────


class TestChunkResult:
    """Test ChunkResult data class."""

    def test_default_no_audio(self):
        result = ChunkResult(ConversationState.LISTENING)
        assert result.state == ConversationState.LISTENING
        assert result.audio_ready is None
        assert result.barge_in is False

    def test_with_audio(self):
        audio = np.zeros(8000, dtype=np.float32)
        result = ChunkResult(ConversationState.PROCESSING, audio_ready=audio)
        assert result.audio_ready is not None
        assert len(result.audio_ready) == 8000

    def test_with_barge_in(self):
        result = ChunkResult(ConversationState.USER_SPEAKING, barge_in=True)
        assert result.barge_in is True


# ── VAD processing ────────────────────────────────────────────────────────────


class TestVADProcessing:
    """Test _run_vad internal method."""

    def test_vad_processes_512_sample_windows(self):
        conv = ConversationManager()
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=lambda: 0.1)
        conv._vad_model = mock_model

        # 2048 samples = 4 windows of 512
        chunk = np.zeros(2048, dtype=np.float32)
        conv._run_vad(chunk)

        assert mock_model.call_count == 4

    def test_vad_returns_true_on_speech(self):
        conv = ConversationManager()
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=lambda: 0.9)
        conv._vad_model = mock_model

        chunk = np.zeros(512, dtype=np.float32)
        assert conv._run_vad(chunk) is True

    def test_vad_returns_false_on_silence(self):
        conv = ConversationManager()
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(item=lambda: 0.1)
        conv._vad_model = mock_model

        chunk = np.zeros(512, dtype=np.float32)
        assert conv._run_vad(chunk) is False

    def test_vad_returns_false_without_model(self):
        conv = ConversationManager()
        chunk = np.zeros(512, dtype=np.float32)
        assert conv._run_vad(chunk) is False

    def test_vad_short_chunk_no_crash(self):
        """Chunks shorter than 512 samples should not crash."""
        conv = ConversationManager()
        mock_model = MagicMock()
        conv._vad_model = mock_model

        chunk = np.zeros(100, dtype=np.float32)
        result = conv._run_vad(chunk)
        assert result is False
        mock_model.assert_not_called()  # chunk too short for any window

    def test_vad_respects_threshold_override(self):
        """_run_vad with explicit threshold overrides the default."""
        conv = ConversationManager({"vad_threshold": 0.5})
        mock_model = MagicMock()
        # VAD returns 0.6 — above default 0.5, but below override 0.8
        mock_model.return_value = MagicMock(item=lambda: 0.6)
        conv._vad_model = mock_model

        chunk = np.zeros(512, dtype=np.float32)
        assert conv._run_vad(chunk, threshold=0.8) is False
        assert conv._run_vad(chunk) is True  # default threshold: 0.5


# ── Config ────────────────────────────────────────────────────────────────────


class TestConfig:
    """Test configuration handling."""

    def test_default_config(self):
        conv = ConversationManager()
        assert conv._vad_threshold == 0.5
        assert conv._silence_threshold_s == 1.5
        assert conv._echo_cooldown_s == 1.5
        assert conv._antivox_boost == 0.25
        assert conv._barge_in_enabled is True
        assert conv._barge_in_threshold == 0.75
        assert conv._barge_in_chunks == 3

    def test_custom_config(self):
        conv = ConversationManager({
            "vad_threshold": 0.7,
            "silence_threshold_s": 2.0,
            "wake_word": "hey computer",
            "echo_cooldown_s": 3.0,
            "antivox_boost": 0.3,
            "barge_in_enabled": False,
            "barge_in_threshold": 0.85,
            "barge_in_chunks": 5,
        })
        assert conv._vad_threshold == 0.7
        assert conv._silence_threshold_s == 2.0
        assert conv._wake_word == "hey computer"
        assert conv._echo_cooldown_s == 3.0
        assert conv._antivox_boost == 0.3
        assert conv._barge_in_enabled is False
        assert conv._barge_in_threshold == 0.85
        assert conv._barge_in_chunks == 5


# ── Full conversation flow ────────────────────────────────────────────────────


class TestFullFlow:
    """Test a complete auto-detect conversation turn."""

    def test_listen_speak_silence_process_done(self):
        """Simulate: listening -> speech -> silence -> processing -> model -> listening."""
        conv = ConversationManager({"silence_threshold_s": 0.5})
        conv._vad_model = MagicMock()
        conv.start()
        assert conv.state == ConversationState.LISTENING

        # Speech detected
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        chunk = np.random.randn(8000).astype(np.float32) * 0.1
        result = conv.on_audio_chunk(chunk)
        assert conv.state == ConversationState.USER_SPEAKING

        # More speech
        result = conv.on_audio_chunk(chunk)
        assert conv.state == ConversationState.USER_SPEAKING

        # Silence -> triggers processing
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
        silence = np.zeros(8000, dtype=np.float32)
        result = conv.on_audio_chunk(silence)
        assert result.state == ConversationState.PROCESSING
        assert result.audio_ready is not None
        # Should contain speech + speech + silence = 3 chunks
        assert len(result.audio_ready) == 8000 * 3

        # Model starts
        conv.on_model_start()
        assert conv.state == ConversationState.MODEL_SPEAKING

        # Model finishes
        conv.on_model_done()
        assert conv.state == ConversationState.LISTENING

    def test_multiple_turns(self):
        """Simulate two consecutive conversation turns."""
        conv = ConversationManager({
            "silence_threshold_s": 0.4,
            "echo_cooldown_s": 0,
        })
        conv._vad_model = MagicMock()
        conv.start()

        for turn in range(2):
            # Speech
            conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
            conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))

            # Silence -> processing
            conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
            result = conv.on_audio_chunk(np.zeros(8000, dtype=np.float32))
            assert result.state == ConversationState.PROCESSING
            assert result.audio_ready is not None

            # Model cycle
            conv.on_model_start()
            conv.on_model_done()
            assert conv.state == ConversationState.LISTENING


# ── Anti-vox echo suppression ────────────────────────────────────────────────


class TestAntiVox:
    """Anti-vox: elevated VAD threshold during cooldown after model speech."""

    def test_weak_speech_suppressed_during_cooldown(self):
        """Speaker bleed (below boosted threshold) doesn't trigger during cooldown."""
        conv = ConversationManager({
            "echo_cooldown_s": 5.0,
            "vad_threshold": 0.5,
            "antivox_boost": 0.25,  # effective threshold = 0.75 during cooldown
        })
        conv._vad_model = MagicMock()
        conv.start()

        conv.on_model_start()
        conv.on_model_done()
        assert conv.state == ConversationState.LISTENING

        # VAD returns 0.6 — above normal threshold (0.5) but below
        # boosted threshold (0.75). Should be suppressed.
        conv._vad_model.return_value = MagicMock(item=lambda: 0.6)
        result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert result.state == ConversationState.LISTENING  # not USER_SPEAKING

    def test_strong_speech_passes_during_cooldown(self):
        """Strong nearby speech (above boosted threshold) passes through cooldown."""
        conv = ConversationManager({
            "echo_cooldown_s": 5.0,
            "vad_threshold": 0.5,
            "antivox_boost": 0.25,  # effective threshold = 0.75 during cooldown
        })
        conv._vad_model = MagicMock()
        conv.start()

        conv.on_model_start()
        conv.on_model_done()

        # VAD returns 0.9 — above boosted threshold (0.75).
        # Should break through cooldown and be detected.
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert result.state == ConversationState.USER_SPEAKING

    def test_cooldown_cleared_on_strong_speech(self):
        """Once strong speech breaks through, cooldown is cancelled."""
        conv = ConversationManager({
            "echo_cooldown_s": 5.0,
            "vad_threshold": 0.5,
            "antivox_boost": 0.25,
        })
        conv._vad_model = MagicMock()
        conv.start()

        conv.on_model_start()
        conv.on_model_done()
        assert conv._cooldown_until > 0

        # Strong speech clears cooldown
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert conv._cooldown_until == 0.0

    def test_zero_cooldown_no_suppression(self):
        """With 0s cooldown, speech is detected immediately after model done."""
        conv = ConversationManager({"echo_cooldown_s": 0.0})
        conv._vad_model = MagicMock()
        conv.start()

        conv.on_model_start()
        conv.on_model_done()

        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert result.state == ConversationState.USER_SPEAKING

    def test_default_antivox_boost(self):
        conv = ConversationManager()
        assert conv._antivox_boost == 0.25

    def test_antivox_boost_configurable(self):
        conv = ConversationManager({"antivox_boost": 0.4})
        assert conv._antivox_boost == 0.4


# ── Barge-in (user interruption) ─────────────────────────────────────────────


class TestBargeIn:
    """Barge-in: user can interrupt model by speaking during MODEL_SPEAKING."""

    def test_single_speech_chunk_no_barge_in(self):
        """One speech chunk during model speech is not enough to trigger barge-in."""
        conv = ConversationManager({"barge_in_chunks": 3})
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()

        # VAD returns high confidence (above barge-in threshold)
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert result.state == ConversationState.MODEL_SPEAKING
        assert result.barge_in is False
        assert conv._barge_in_count == 1

    def test_consecutive_speech_triggers_barge_in(self):
        """N consecutive speech chunks trigger barge-in."""
        conv = ConversationManager({"barge_in_chunks": 3})
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()

        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        chunk = np.random.randn(8000).astype(np.float32)

        # Chunks 1 and 2: no barge-in yet
        result = conv.on_audio_chunk(chunk)
        assert result.barge_in is False
        result = conv.on_audio_chunk(chunk)
        assert result.barge_in is False

        # Chunk 3: barge-in triggered!
        result = conv.on_audio_chunk(chunk)
        assert result.barge_in is True
        assert result.state == ConversationState.USER_SPEAKING

    def test_silence_resets_barge_in_counter(self):
        """A silence chunk resets the consecutive speech counter."""
        conv = ConversationManager({"barge_in_chunks": 3})
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()

        chunk = np.random.randn(8000).astype(np.float32)

        # Two speech chunks
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(chunk)
        conv.on_audio_chunk(chunk)
        assert conv._barge_in_count == 2

        # One silence chunk — resets counter
        conv._vad_model.return_value = MagicMock(item=lambda: 0.1)
        conv.on_audio_chunk(chunk)
        assert conv._barge_in_count == 0

        # Two more speech chunks — still not enough (need 3 consecutive)
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(chunk)
        conv.on_audio_chunk(chunk)
        result = conv.on_audio_chunk(chunk)  # 3rd consecutive
        assert result.barge_in is True

    def test_barge_in_uses_elevated_threshold(self):
        """Barge-in uses barge_in_threshold, not the normal vad_threshold."""
        conv = ConversationManager({
            "vad_threshold": 0.5,
            "barge_in_threshold": 0.75,
            "barge_in_chunks": 1,
        })
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()

        # VAD returns 0.6 — above normal (0.5) but below barge-in (0.75)
        conv._vad_model.return_value = MagicMock(item=lambda: 0.6)
        result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert result.barge_in is False  # not strong enough

        # VAD returns 0.9 — above barge-in threshold
        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert result.barge_in is True

    def test_barge_in_disabled(self):
        """With barge-in disabled, model speaking ignores all chunks."""
        conv = ConversationManager({"barge_in_enabled": False})
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()

        conv._vad_model.return_value = MagicMock(item=lambda: 0.99)
        for _ in range(10):
            result = conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
            assert result.state == ConversationState.MODEL_SPEAKING
            assert result.barge_in is False

    def test_barge_in_transitions_to_user_speaking(self):
        """After barge-in, state is USER_SPEAKING with audio buffer started."""
        conv = ConversationManager({"barge_in_chunks": 1})
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()

        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        chunk = np.random.randn(8000).astype(np.float32)
        result = conv.on_audio_chunk(chunk)

        assert result.barge_in is True
        assert conv.state == ConversationState.USER_SPEAKING
        assert len(conv._audio_buffer) == 1  # chunk saved to buffer

    def test_barge_in_clears_cooldown(self):
        """Barge-in cancels any pending echo cooldown."""
        conv = ConversationManager({"barge_in_chunks": 1})
        conv._vad_model = MagicMock()
        conv.start()
        conv.on_model_start()
        conv._cooldown_until = 9999999999.0  # far future

        conv._vad_model.return_value = MagicMock(item=lambda: 0.9)
        conv.on_audio_chunk(np.random.randn(8000).astype(np.float32))
        assert conv._cooldown_until == 0.0

    def test_model_start_resets_barge_in_count(self):
        """on_model_start resets the barge-in counter."""
        conv = ConversationManager()
        conv._vad_model = MagicMock()
        conv.start()
        conv._barge_in_count = 5
        conv.on_model_start()
        assert conv._barge_in_count == 0
