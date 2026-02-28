"""StreamingAudioPlayer — queue-based real-time audio playback.

Plays numpy waveform chunks through speakers as they arrive, using
sounddevice.OutputStream with a callback. Designed for streaming TTS
where audio is generated incrementally by the model.

Usage:
    player = StreamingAudioPlayer(sample_rate=24000)
    player.start()

    for chunk in model_generates_audio():
        player.push(chunk)  # starts playing immediately

    player.finish()  # signal no more chunks
    player.wait()    # block until all audio played
    player.stop()    # close the stream

    # Optional: save all audio for archival
    full_audio = player.get_full_audio()
"""

import queue
import threading
from typing import Optional

import numpy as np


class StreamingAudioPlayer:
    """Queue-based streaming audio player using sounddevice.OutputStream.

    Producer thread pushes waveform chunks via push(). The sounddevice
    callback pulls data from an internal queue for real-time playback.
    Audio starts as soon as the first chunk arrives.
    """

    def __init__(self, sample_rate: int = 24000, channels: int = 1, blocksize: int = 4800):
        """
        Args:
            sample_rate: Audio sample rate in Hz (24000 for MiniCPM-o TTS).
            channels: Number of audio channels (1 = mono).
            blocksize: Frames per callback invocation. 4800 = 200ms at 24kHz.
        """
        self._sr = sample_rate
        self._channels = channels
        self._blocksize = blocksize

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._stream = None
        self._residual: Optional[np.ndarray] = None  # leftover from previous callback

        self._finished = threading.Event()  # set when producer calls finish()
        self._drained = threading.Event()   # set when queue is empty after finish
        self._collected: list[np.ndarray] = []

    def start(self) -> None:
        """Open the audio stream and begin the callback loop."""
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=self._channels,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def push(self, chunk: np.ndarray) -> None:
        """Add a waveform chunk to the playback queue.

        Thread-safe. Can be called from any thread while playback is active.
        Also stores the chunk for later archival via get_full_audio().
        """
        if chunk is not None and len(chunk) > 0:
            flat = chunk.flatten().astype(np.float32)
            self._collected.append(flat.copy())
            self._queue.put(flat)

    def finish(self) -> None:
        """Signal that no more chunks will be pushed.

        The player will continue playing until the queue is drained,
        then signal completion via the _drained event.
        """
        self._finished.set()
        self._queue.put(None)  # sentinel to unblock callback

    def wait(self, timeout: Optional[float] = None) -> None:
        """Block until all queued audio has been played.

        Must call finish() before wait().
        """
        self._drained.wait(timeout=timeout)

    def stop(self) -> None:
        """Close the audio stream and release resources."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def get_full_audio(self) -> Optional[np.ndarray]:
        """Return all pushed audio concatenated, for archival save.

        Returns None if no audio was pushed.
        """
        if not self._collected:
            return None
        return np.concatenate(self._collected)

    def _callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """sounddevice callback — fills output buffer from the queue.

        Called by the audio hardware thread. Must not block for long.
        Fills with silence if no data is available (prevents glitches).
        """
        needed = frames
        pos = 0

        # First, use any residual data from previous callback
        if self._residual is not None and len(self._residual) > 0:
            n = min(len(self._residual), needed)
            outdata[pos:pos + n, 0] = self._residual[:n]
            self._residual = self._residual[n:] if n < len(self._residual) else None
            pos += n
            needed -= n

        # Pull chunks from the queue to fill the rest
        while needed > 0:
            try:
                chunk = self._queue.get_nowait()
            except queue.Empty:
                # No data available — check if we're done
                if self._finished.is_set():
                    # Fill remainder with silence and signal drain
                    outdata[pos:, 0] = 0.0
                    self._drained.set()
                    return
                # Producer still running, fill with silence (underrun)
                outdata[pos:, 0] = 0.0
                return

            if chunk is None:
                # Sentinel from finish() — fill remainder with silence
                outdata[pos:, 0] = 0.0
                self._drained.set()
                return

            n = min(len(chunk), needed)
            outdata[pos:pos + n, 0] = chunk[:n]
            if n < len(chunk):
                self._residual = chunk[n:]
            pos += n
            needed -= n
