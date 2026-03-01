"""
rt_audio.py -- Real-time audio pipeline for PySide6 desktop app.

Three classes that orchestrate mic -> VAD -> model -> speaker:
  - MicInputStream: sounddevice.InputStream wrapper, emits Qt signals
  - ModelInferenceThread: runs chat_streaming() on QThread
  - AudioPipeline: coordinates everything, owns ConversationManager + StreamingAudioPlayer
"""

import numpy as np
from PySide6.QtCore import QObject, QThread, QTimer, Signal


class MicInputStream(QObject):
    """Real-time microphone capture via sounddevice.

    Opens a 16kHz mono float32 InputStream.  The sounddevice callback runs on
    a separate audio thread; we emit a Qt signal (thread-safe QueuedConnection)
    to deliver audio chunks to the main thread.
    """

    chunk_received = Signal(object)  # np.ndarray, 16kHz float32 mono

    def __init__(self, sample_rate: int = 16000, chunk_samples: int = 512):
        super().__init__()
        self._sr = sample_rate
        self._blocksize = chunk_samples  # 512 = 32ms at 16kHz (Silero-VAD size)
        self._stream = None

    @property
    def is_active(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self):
        """Open the microphone stream."""
        import sounddevice as sd
        if self._stream is not None:
            self.stop()
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        """Close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _callback(self, indata, frames, time_info, status):
        """sounddevice callback — runs on audio thread."""
        # indata is (frames, 1) — flatten to 1D and copy
        chunk = indata[:, 0].copy()
        self.chunk_received.emit(chunk)


class ModelInferenceThread(QThread):
    """Runs model.chat_streaming() on a background thread.

    GPU inference is blocking (1-30+ seconds).  Running on QThread keeps the UI
    responsive.  Audio chunks are emitted via signals as they arrive from the
    model's streaming generator.
    """

    chunk_ready = Signal(object, str)   # (audio_chunk or None, text_chunk)
    finished_signal = Signal(str)       # full response text

    def __init__(
        self,
        messages: list[dict],
        voice_ref,
        settings: dict,
        generate_audio: bool = True,
    ):
        super().__init__()
        self._messages = messages
        self._voice_ref = voice_ref
        self._settings = settings
        self._generate_audio = generate_audio
        self._stop_requested = False

    def request_stop(self):
        """Request early termination (barge-in)."""
        self._stop_requested = True

    def run(self):
        from tools.model.model_manager import (
            chat_streaming,
            _apply_fade_in,
            _normalize_rms,
            _get_leveling_config,
        )

        full_text = ""
        is_first = True
        cfg = _get_leveling_config()

        try:
            for audio_chunk, text_chunk in chat_streaming(
                messages=self._messages,
                voice_ref=self._voice_ref,
                generate_audio=self._generate_audio,
                temperature=self._settings.get("temperature", 0.7),
                max_new_tokens=self._settings.get("max_new_tokens", 2048),
                repetition_penalty=self._settings.get("repetition_penalty", 1.05),
                top_p=self._settings.get("top_p", 0.8),
                top_k=self._settings.get("top_k", 100),
                enable_thinking=self._settings.get("enable_thinking", False),
            ):
                if self._stop_requested:
                    break

                if audio_chunk is not None:
                    if is_first:
                        audio_chunk = _apply_fade_in(audio_chunk)
                        is_first = False
                    if cfg["enabled"]:
                        audio_chunk = _normalize_rms(
                            audio_chunk,
                            target_rms=cfg["_output_threshold_linear"],
                            max_gain=cfg["_output_max_gain_linear"],
                            peak_ceiling=cfg["_peak_ceiling_linear"],
                        )

                if isinstance(text_chunk, str) and text_chunk:
                    full_text += text_chunk

                self.chunk_ready.emit(audio_chunk, text_chunk if isinstance(text_chunk, str) else "")
        except Exception as e:
            print(f"  [rt_inference] Error: {e}")

        self.finished_signal.emit(full_text)


class AudioPipeline(QObject):
    """Orchestrates the mic -> VAD -> model -> speaker pipeline.

    Owns ConversationManager, MicInputStream, and coordinates with
    ModelInferenceThread for GPU work.

    Signals emitted are consumed by the UI (rt_app.py) to update widgets.
    """

    # Signals for UI updates
    state_changed = Signal(str, str)          # (state_name, color_hex)
    speech_ready = Signal(object)             # np.ndarray of user speech (16kHz)
    barge_in_detected = Signal()              # user interrupted model
    text_update = Signal(str)                 # incremental text from model
    audio_chunk_ready = Signal(object)        # np.ndarray audio chunk from model (for level meter)
    generation_started = Signal()
    generation_finished = Signal(str)         # full response text

    def __init__(self, chat_mode_cfg: dict):
        super().__init__()
        from tools.audio.conversation import ConversationManager
        self.conv_mgr = ConversationManager(chat_mode_cfg)
        self.mic = MicInputStream()
        self._player = None
        self._inference_thread = None
        self._is_generating = False
        self._prev_state = None

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    def start_conversation(self):
        """Begin continuous conversation mode — starts mic + VAD."""
        self.conv_mgr.start()
        # Disconnect first to prevent duplicate connections on repeated start/stop
        try:
            self.mic.chunk_received.disconnect(self._on_mic_chunk)
        except RuntimeError:
            pass
        self.mic.chunk_received.connect(self._on_mic_chunk)
        self.mic.start()
        self._emit_state()

    def stop_conversation(self):
        """End conversation mode — stops mic, cleans up."""
        self.mic.stop()
        try:
            self.mic.chunk_received.disconnect(self._on_mic_chunk)
        except RuntimeError:
            pass  # not connected
        self.conv_mgr.stop()
        self._stop_generation()
        self._emit_state()

    def process_turn(self, messages: list[dict], voice_ref, settings: dict, generate_audio: bool = True):
        """Start model inference for a turn (from conversation or single-turn)."""
        if self._is_generating:
            return  # already running

        self._is_generating = True
        self.conv_mgr.on_model_start()
        self.generation_started.emit()

        # Start audio player for this turn
        if generate_audio:
            from tools.audio.streaming_player import StreamingAudioPlayer
            self._player = StreamingAudioPlayer(sample_rate=24000)
            self._player.start()

        # Start inference thread
        self._inference_thread = ModelInferenceThread(
            messages=messages,
            voice_ref=voice_ref,
            settings=settings,
            generate_audio=generate_audio,
        )
        self._inference_thread.chunk_ready.connect(self._on_model_chunk)
        self._inference_thread.finished_signal.connect(self._on_model_done)
        self._inference_thread.start()
        self._emit_state()

    def _on_mic_chunk(self, chunk: np.ndarray):
        """Handle a mic chunk: run VAD, detect speech/barge-in."""
        result = self.conv_mgr.on_audio_chunk(chunk)

        if result.audio_ready is not None:
            self.speech_ready.emit(result.audio_ready)

        if result.barge_in:
            self.barge_in_detected.emit()
            self._stop_generation()

        # Emit state if it changed
        self._emit_state()

    def _on_model_chunk(self, audio_chunk, text_chunk: str):
        """Handle a chunk from the model inference thread."""
        if audio_chunk is not None and self._player is not None:
            self._player.push(audio_chunk)
            self.audio_chunk_ready.emit(audio_chunk)
        if text_chunk:
            self.text_update.emit(text_chunk)

    def _on_model_done(self, full_text: str):
        """Handle model inference completion."""
        if not self._is_generating:
            # Already stopped via barge-in or manual stop — don't reset state
            self.generation_finished.emit(full_text)
            return

        if self._player is not None:
            self._player.finish()
            # Poll for drain completion instead of blocking the UI thread
            self._pending_full_text = full_text
            self._drain_timer = QTimer()
            self._drain_timer.setInterval(50)
            self._drain_timer.timeout.connect(self._check_drain)
            self._drain_timer.start()
        else:
            self._finalize_turn(full_text)

    def _check_drain(self):
        """Poll until audio player finishes draining."""
        if self._player is None or self._player._drained.is_set():
            self._drain_timer.stop()
            if self._player is not None:
                self._player.stop()
                self._player = None
            self._finalize_turn(self._pending_full_text)

    def _finalize_turn(self, full_text: str):
        """Clean up after a completed generation turn."""
        self._is_generating = False
        self.conv_mgr.on_model_done()
        self.generation_finished.emit(full_text)
        self._emit_state()

    def _stop_generation(self):
        """Interrupt model generation (barge-in or manual stop)."""
        if self._inference_thread is not None:
            # Disconnect signals so delayed finished_signal won't corrupt state
            try:
                self._inference_thread.chunk_ready.disconnect(self._on_model_chunk)
            except RuntimeError:
                pass
            try:
                self._inference_thread.finished_signal.disconnect(self._on_model_done)
            except RuntimeError:
                pass
            if self._inference_thread.isRunning():
                self._inference_thread.request_stop()

        if self._player is not None:
            self._player.finish()
            self._player.stop()
            self._player = None

        self._is_generating = False

    def _emit_state(self, force: bool = False):
        """Emit current conversation state for UI display (only on change)."""
        from tools.audio.conversation import ConversationState
        state = self.conv_mgr.state
        if not force and state == self._prev_state:
            return
        self._prev_state = state
        colors = {
            ConversationState.OFF: "#888888",
            ConversationState.LISTENING: "#22c55e",
            ConversationState.USER_SPEAKING: "#3b82f6",
            ConversationState.PROCESSING: "#eab308",
            ConversationState.MODEL_SPEAKING: "#a855f7",
            ConversationState.IDLE: "#f97316",
        }
        self.state_changed.emit(state.name, colors.get(state, "#888888"))
