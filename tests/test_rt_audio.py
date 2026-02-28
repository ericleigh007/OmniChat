"""Tests for rt_audio.py — real-time audio pipeline for PySide6 app.

All tests are mock-based: no GPU, no audio hardware, no PySide6 event loop.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ── MicInputStream tests ──────────────────────────────────────────────────────

class TestMicInputStream:
    """Test microphone input wrapper."""

    @patch("rt_audio.MicInputStream.__init_subclass__", lambda **kw: None)
    def test_start_opens_stream(self):
        """start() should open a sounddevice InputStream."""
        with patch("sounddevice.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream

            from rt_audio import MicInputStream
            # Bypass QObject init for testing
            mic = MicInputStream.__new__(MicInputStream)
            mic._sr = 16000
            mic._blocksize = 512
            mic._stream = None
            mic.chunk_received = MagicMock()

            mic.start()

            mock_stream_cls.assert_called_once_with(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=512,
                callback=mic._callback,
            )
            mock_stream.start.assert_called_once()

    def test_stop_closes_stream(self):
        """stop() should stop and close the stream."""
        from rt_audio import MicInputStream
        mic = MicInputStream.__new__(MicInputStream)
        mock_stream = MagicMock()
        mic._stream = mock_stream

        mic.stop()

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
        assert mic._stream is None

    def test_stop_when_no_stream(self):
        """stop() with no active stream should not crash."""
        from rt_audio import MicInputStream
        mic = MicInputStream.__new__(MicInputStream)
        mic._stream = None
        mic.stop()  # should not raise

    def test_callback_emits_flattened_chunk(self):
        """The callback should emit a 1D float32 array."""
        from rt_audio import MicInputStream
        mic = MicInputStream.__new__(MicInputStream)
        mic.chunk_received = MagicMock()

        # Simulate sounddevice callback: indata is (frames, channels)
        indata = np.random.randn(512, 1).astype(np.float32)
        mic._callback(indata, 512, {}, None)

        mic.chunk_received.emit.assert_called_once()
        emitted = mic.chunk_received.emit.call_args[0][0]
        assert emitted.ndim == 1
        assert emitted.dtype == np.float32
        assert len(emitted) == 512

    def test_is_active_property(self):
        """is_active should reflect stream state."""
        from rt_audio import MicInputStream
        mic = MicInputStream.__new__(MicInputStream)

        mic._stream = None
        assert not mic.is_active

        mic._stream = MagicMock()
        mic._stream.active = True
        assert mic.is_active

        mic._stream.active = False
        assert not mic.is_active


# ── ModelInferenceThread tests ────────────────────────────────────────────────

class TestModelInferenceThread:
    """Test model inference thread (mock-based, no GPU)."""

    def _make_thread(self, **kwargs):
        from rt_audio import ModelInferenceThread
        thread = ModelInferenceThread.__new__(ModelInferenceThread)
        thread._messages = kwargs.get("messages", [{"role": "user", "content": ["hello"]}])
        thread._voice_ref = kwargs.get("voice_ref", None)
        thread._settings = kwargs.get("settings", {"temperature": 0.7, "max_new_tokens": 2048, "repetition_penalty": 1.05})
        thread._generate_audio = kwargs.get("generate_audio", True)
        thread._stop_requested = False
        thread.chunk_ready = MagicMock()
        thread.finished_signal = MagicMock()
        return thread

    @patch("rt_audio.ModelInferenceThread.run")
    def test_request_stop_sets_flag(self, mock_run):
        """request_stop() should set the flag that breaks the generation loop."""
        thread = self._make_thread()
        assert not thread._stop_requested
        thread.request_stop()
        assert thread._stop_requested

    def test_run_emits_chunks(self):
        """run() should emit chunk_ready for each model output."""
        thread = self._make_thread()

        # Mock chat_streaming to yield 3 chunks
        fake_chunks = [
            (np.zeros(24000, dtype=np.float32), "Hello"),
            (np.zeros(24000, dtype=np.float32), " world"),
            (None, "!"),
        ]

        with patch("tools.model.model_manager.chat_streaming", return_value=iter(fake_chunks)), \
             patch("tools.model.model_manager._apply_fade_in", side_effect=lambda x: x), \
             patch("tools.model.model_manager._normalize_rms", side_effect=lambda x, **kw: x), \
             patch("tools.model.model_manager._get_leveling_config", return_value={"enabled": False}):

            thread.run()

        assert thread.chunk_ready.emit.call_count == 3
        thread.finished_signal.emit.assert_called_once_with("Hello world!")

    def test_run_stops_on_request(self):
        """run() should break early when stop is requested."""
        thread = self._make_thread()

        call_count = 0
        def fake_gen(**kwargs):
            nonlocal call_count
            for i in range(10):
                call_count += 1
                if call_count == 2:
                    thread._stop_requested = True
                yield (np.zeros(100, dtype=np.float32), f"chunk{i}")

        with patch("tools.model.model_manager.chat_streaming", side_effect=fake_gen), \
             patch("tools.model.model_manager._apply_fade_in", side_effect=lambda x: x), \
             patch("tools.model.model_manager._normalize_rms", side_effect=lambda x, **kw: x), \
             patch("tools.model.model_manager._get_leveling_config", return_value={"enabled": False}):

            thread.run()

        # Should have emitted only 1 chunk (stopped at iteration 2)
        assert thread.chunk_ready.emit.call_count == 1
        thread.finished_signal.emit.assert_called_once()

    def test_run_applies_fade_in_on_first_chunk(self):
        """First audio chunk should get fade-in applied."""
        thread = self._make_thread()

        fade_in_called = []
        def mock_fade_in(chunk):
            fade_in_called.append(True)
            return chunk

        fake_chunks = [
            (np.zeros(24000, dtype=np.float32), "a"),
            (np.zeros(24000, dtype=np.float32), "b"),
        ]

        with patch("tools.model.model_manager.chat_streaming", return_value=iter(fake_chunks)), \
             patch("tools.model.model_manager._apply_fade_in", side_effect=mock_fade_in), \
             patch("tools.model.model_manager._normalize_rms", side_effect=lambda x, **kw: x), \
             patch("tools.model.model_manager._get_leveling_config", return_value={"enabled": False}):

            thread.run()

        assert len(fade_in_called) == 1  # only first chunk


# ── AudioPipeline tests ──────────────────────────────────────────────────────

class TestAudioPipeline:
    """Test the pipeline orchestrator (mock-based)."""

    def _make_pipeline(self):
        """Create a pipeline with mocked ConversationManager and MicInputStream."""
        with patch("rt_audio.AudioPipeline.__init__", lambda self, cfg: None):
            from rt_audio import AudioPipeline
            pipe = AudioPipeline.__new__(AudioPipeline)

        # Set up mocks
        pipe.conv_mgr = MagicMock()
        pipe.conv_mgr.state = MagicMock()
        pipe.conv_mgr.state.name = "OFF"
        pipe.mic = MagicMock()
        pipe._player = None
        pipe._inference_thread = None
        pipe._is_generating = False
        pipe._prev_state = None

        # Mock signals
        pipe.state_changed = MagicMock()
        pipe.speech_ready = MagicMock()
        pipe.barge_in_detected = MagicMock()
        pipe.text_update = MagicMock()
        pipe.audio_chunk_ready = MagicMock()
        pipe.generation_started = MagicMock()
        pipe.generation_finished = MagicMock()

        return pipe

    def test_start_conversation_starts_mic_and_vad(self):
        """start_conversation should start mic and ConversationManager."""
        pipe = self._make_pipeline()
        pipe.start_conversation()

        pipe.conv_mgr.start.assert_called_once()
        pipe.mic.start.assert_called_once()

    def test_stop_conversation_stops_everything(self):
        """stop_conversation should stop mic and ConversationManager."""
        pipe = self._make_pipeline()
        pipe.stop_conversation()

        pipe.mic.stop.assert_called_once()
        pipe.conv_mgr.stop.assert_called_once()

    def test_on_mic_chunk_runs_vad(self):
        """_on_mic_chunk should call conv_mgr.on_audio_chunk."""
        pipe = self._make_pipeline()
        chunk = np.zeros(512, dtype=np.float32)
        result = MagicMock()
        result.audio_ready = None
        result.barge_in = False
        pipe.conv_mgr.on_audio_chunk.return_value = result

        pipe._on_mic_chunk(chunk)

        pipe.conv_mgr.on_audio_chunk.assert_called_once()
        np.testing.assert_array_equal(
            pipe.conv_mgr.on_audio_chunk.call_args[0][0], chunk
        )

    def test_on_mic_chunk_emits_speech_ready(self):
        """When VAD detects speech end, speech_ready signal should fire."""
        pipe = self._make_pipeline()
        speech_audio = np.random.randn(16000).astype(np.float32)
        result = MagicMock()
        result.audio_ready = speech_audio
        result.barge_in = False
        pipe.conv_mgr.on_audio_chunk.return_value = result

        pipe._on_mic_chunk(np.zeros(512, dtype=np.float32))

        pipe.speech_ready.emit.assert_called_once()
        emitted = pipe.speech_ready.emit.call_args[0][0]
        np.testing.assert_array_equal(emitted, speech_audio)

    def test_on_mic_chunk_detects_barge_in(self):
        """Barge-in should emit signal and stop generation."""
        pipe = self._make_pipeline()
        pipe._is_generating = True
        pipe._inference_thread = MagicMock()
        pipe._inference_thread.isRunning.return_value = True
        pipe._player = MagicMock()

        result = MagicMock()
        result.audio_ready = None
        result.barge_in = True
        pipe.conv_mgr.on_audio_chunk.return_value = result

        pipe._on_mic_chunk(np.zeros(512, dtype=np.float32))

        pipe.barge_in_detected.emit.assert_called_once()
        pipe._inference_thread.request_stop.assert_called_once()

    def test_on_model_chunk_pushes_to_player(self):
        """Audio chunks from model should be pushed to StreamingAudioPlayer."""
        pipe = self._make_pipeline()
        pipe._player = MagicMock()

        audio = np.zeros(24000, dtype=np.float32)
        pipe._on_model_chunk(audio, "hello")

        pipe._player.push.assert_called_once_with(audio)
        pipe.audio_chunk_ready.emit.assert_called_once()
        pipe.text_update.emit.assert_called_once_with("hello")

    def test_on_model_done_no_player_finalizes(self):
        """When generation finishes with no player, should finalize immediately."""
        pipe = self._make_pipeline()
        pipe._player = None
        pipe._is_generating = True

        pipe._on_model_done("Full response text")

        assert not pipe._is_generating
        pipe.conv_mgr.on_model_done.assert_called_once()
        pipe.generation_finished.emit.assert_called_once_with("Full response text")

    def test_on_model_done_skips_if_already_stopped(self):
        """Delayed finished_signal after barge-in should not reset state."""
        pipe = self._make_pipeline()
        pipe._is_generating = False  # already stopped via barge-in

        pipe._on_model_done("Partial text")

        # Should emit finished but NOT call on_model_done (state already reset)
        pipe.generation_finished.emit.assert_called_once_with("Partial text")
        pipe.conv_mgr.on_model_done.assert_not_called()

    def test_finalize_turn_cleans_up(self):
        """_finalize_turn should reset state and emit signals."""
        pipe = self._make_pipeline()
        pipe._is_generating = True

        pipe._finalize_turn("Full text")

        assert not pipe._is_generating
        pipe.conv_mgr.on_model_done.assert_called_once()
        pipe.generation_finished.emit.assert_called_once_with("Full text")

    def test_stop_generation_disconnects_signals(self):
        """_stop_generation should disconnect inference thread signals."""
        pipe = self._make_pipeline()
        pipe._is_generating = True
        pipe._inference_thread = MagicMock()
        pipe._inference_thread.isRunning.return_value = True
        pipe._player = MagicMock()

        pipe._stop_generation()

        pipe._inference_thread.chunk_ready.disconnect.assert_called_once()
        pipe._inference_thread.finished_signal.disconnect.assert_called_once()
        pipe._inference_thread.request_stop.assert_called_once()
        assert not pipe._is_generating

    def test_emit_state_skips_unchanged(self):
        """_emit_state should not emit when state hasn't changed."""
        pipe = self._make_pipeline()
        from unittest.mock import PropertyMock

        # First call: emits
        pipe._emit_state()
        assert pipe.state_changed.emit.call_count == 1

        # Second call with same state: should NOT emit
        pipe._emit_state()
        assert pipe.state_changed.emit.call_count == 1

        # Change state: should emit
        pipe.conv_mgr.state = MagicMock()
        pipe._emit_state()
        assert pipe.state_changed.emit.call_count == 2

    def test_process_turn_prevents_concurrent(self):
        """process_turn should not start if already generating."""
        pipe = self._make_pipeline()
        pipe._is_generating = True

        pipe.process_turn([{"role": "user", "content": ["hi"]}], None, {})

        # Should not have started anything
        pipe.conv_mgr.on_model_start.assert_not_called()
