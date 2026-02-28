"""Unit tests for StreamingAudioPlayer (mock-based, no GPU/audio hardware needed)."""

import queue
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tools.audio.streaming_player import StreamingAudioPlayer


class TestStreamingAudioPlayer:
    """Test the queue-based streaming audio player."""

    def test_push_collects_chunks(self):
        """Pushed chunks are stored for later retrieval via get_full_audio()."""
        player = StreamingAudioPlayer(sample_rate=24000)
        # Don't start() — we're testing collection logic, not audio output

        chunk1 = np.ones(1000, dtype=np.float32) * 0.5
        chunk2 = np.ones(2000, dtype=np.float32) * -0.3

        player.push(chunk1)
        player.push(chunk2)

        full = player.get_full_audio()
        assert full is not None
        assert len(full) == 3000
        assert np.allclose(full[:1000], 0.5)
        assert np.allclose(full[1000:], -0.3)

    def test_get_full_audio_empty_returns_none(self):
        """get_full_audio() returns None when nothing was pushed."""
        player = StreamingAudioPlayer()
        assert player.get_full_audio() is None

    def test_push_ignores_empty_chunks(self):
        """Empty or None chunks are silently ignored."""
        player = StreamingAudioPlayer()
        player.push(np.array([], dtype=np.float32))
        player.push(None)
        assert player.get_full_audio() is None

    def test_push_flattens_multidim(self):
        """Multi-dimensional chunks are flattened to 1D."""
        player = StreamingAudioPlayer()
        chunk_2d = np.ones((100, 1), dtype=np.float32)
        player.push(chunk_2d)
        full = player.get_full_audio()
        assert full.ndim == 1
        assert len(full) == 100

    def test_callback_fills_silence_on_empty_queue(self):
        """When the queue is empty and not finished, callback outputs zeros."""
        player = StreamingAudioPlayer(sample_rate=24000, blocksize=480)
        outdata = np.zeros((480, 1), dtype=np.float32)

        # Simulate callback with empty queue (not finished)
        player._callback(outdata, 480, None, None)

        # Should be all zeros (silence)
        assert np.allclose(outdata, 0.0)

    def test_callback_fills_from_queue(self):
        """Callback pulls data from the queue into the output buffer."""
        player = StreamingAudioPlayer(sample_rate=24000, blocksize=480)
        outdata = np.zeros((480, 1), dtype=np.float32)

        # Push a chunk
        chunk = np.ones(480, dtype=np.float32) * 0.7
        player._queue.put(chunk)

        player._callback(outdata, 480, None, None)

        assert np.allclose(outdata[:, 0], 0.7)

    def test_callback_handles_partial_chunk(self):
        """When chunk is smaller than frames, fills remainder from next chunk or silence."""
        player = StreamingAudioPlayer(sample_rate=24000, blocksize=480)
        outdata = np.zeros((480, 1), dtype=np.float32)

        # Push a small chunk (less than blocksize)
        chunk = np.ones(200, dtype=np.float32) * 0.5
        player._queue.put(chunk)

        player._callback(outdata, 480, None, None)

        # First 200 samples should be 0.5, rest silence
        assert np.allclose(outdata[:200, 0], 0.5)
        assert np.allclose(outdata[200:, 0], 0.0)

    def test_callback_stores_residual(self):
        """When chunk is larger than frames, remainder is saved as residual."""
        player = StreamingAudioPlayer(sample_rate=24000, blocksize=480)
        outdata = np.zeros((480, 1), dtype=np.float32)

        # Push a large chunk
        chunk = np.ones(700, dtype=np.float32) * 0.9
        player._queue.put(chunk)

        player._callback(outdata, 480, None, None)

        # Output should be filled
        assert np.allclose(outdata[:, 0], 0.9)
        # Residual should have 220 samples
        assert player._residual is not None
        assert len(player._residual) == 220

    def test_finish_and_drain_signals(self):
        """After finish(), the drained event is set when queue empties."""
        player = StreamingAudioPlayer(sample_rate=24000, blocksize=480)
        outdata = np.zeros((480, 1), dtype=np.float32)

        player.finish()

        # Callback should detect the sentinel and set _drained
        player._callback(outdata, 480, None, None)

        assert player._drained.is_set()

    @patch("tools.audio.streaming_player.StreamingAudioPlayer.start")
    @patch("tools.audio.streaming_player.StreamingAudioPlayer.stop")
    def test_lifecycle_without_hardware(self, mock_stop, mock_start):
        """Full push/finish/wait lifecycle works without audio hardware."""
        player = StreamingAudioPlayer(sample_rate=24000)
        mock_start.return_value = None
        mock_stop.return_value = None

        player.start()

        player.push(np.ones(1000, dtype=np.float32))
        player.push(np.ones(500, dtype=np.float32))
        player.finish()

        # Simulate drain
        player._drained.set()
        player.wait(timeout=1.0)

        player.stop()

        full = player.get_full_audio()
        assert full is not None
        assert len(full) == 1500
