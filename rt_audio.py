"""
rt_audio.py -- Real-time audio pipeline for PySide6 desktop app.

Three classes that orchestrate mic -> VAD -> model -> speaker:
  - MicInputStream: sounddevice.InputStream wrapper, emits Qt signals
  - ModelInferenceThread: runs chat_streaming() on QThread
  - AudioPipeline: coordinates everything, owns ConversationManager + StreamingAudioPlayer
"""

import numpy as np
import time
from PySide6.QtCore import QObject, QThread, QTimer, Signal
from uuid import uuid4

from tools.shared.debug_trace import get_trace_logger


logger = get_trace_logger()


def _estimate_token_count(text: str) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, round(len(normalized) / 4.0))


def _estimate_token_count_from_chars(char_count: int) -> int:
    if char_count <= 0:
        return 0
    return max(1, round(char_count / 4.0))


def _safe_signal_emit(signal, *args) -> None:
    try:
        emitter = getattr(signal, "emit", None)
        if callable(emitter):
            emitter(*args)
    except RuntimeError:
        pass


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
    progress_signal = Signal(object)    # dict payload
    error_signal = Signal(str)

    def __init__(
        self,
        messages: list[dict],
        voice_ref,
        settings: dict,
        generate_audio: bool = True,
        trace_context: dict | None = None,
    ):
        super().__init__()
        self._messages = messages
        self._voice_ref = voice_ref
        self._settings = settings
        self._generate_audio = generate_audio
        self._stop_requested = False
        self._trace_context = dict(trace_context or {})

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

        trace_context = dict(getattr(self, "_trace_context", {}) or {})
        progress_signal = getattr(self, "progress_signal", None)
        error_signal = getattr(self, "error_signal", None)
        request_id = str(trace_context.get("request_id") or "n/a")
        full_text = ""
        is_first = True
        cfg = _get_leveling_config()
        started = time.perf_counter()
        text_chunk_count = 0
        audio_chunk_count = 0

        logger.info(
            "trace_id=%s stage=rt_inference event=thread_start generate_audio=%s temperature=%.3f max_tokens=%d message_count=%d",
            request_id,
            bool(self._generate_audio),
            float(self._settings.get("temperature", 0.7)),
            int(self._settings.get("max_new_tokens", 2048)),
            len(self._messages or []),
        )
        if progress_signal is not None:
            _safe_signal_emit(progress_signal, {
                "event": "thread_start",
                "request_id": request_id,
                "elapsed_s": 0.0,
                "text_chars": 0,
                "text_tokens_est": 0,
                "audio_chunks": 0,
                "message": "Inference thread started",
            })

        try:
            stream = chat_streaming(
                messages=self._messages,
                voice_ref=self._voice_ref,
                generate_audio=self._generate_audio,
                temperature=self._settings.get("temperature", 0.7),
                max_new_tokens=self._settings.get("max_new_tokens", 2048),
                repetition_penalty=self._settings.get("repetition_penalty", 1.05),
                top_p=self._settings.get("top_p", 0.8),
                top_k=self._settings.get("top_k", 100),
                enable_thinking=self._settings.get("enable_thinking", False),
                trace_context=trace_context,
            )
            final_override = None
            while True:
                try:
                    audio_chunk, text_chunk = next(stream)
                except StopIteration as stop:
                    if isinstance(stop.value, dict):
                        final_override = stop.value.get("final_text")
                    break
                if self._stop_requested:
                    break

                if audio_chunk is not None:
                    audio_chunk_count += 1
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
                    text_chunk_count += 1
                    logger.info(
                        "trace_id=%s stage=rt_inference event=text_chunk chunk_chars=%d total_chars=%d total_tokens_est=%d elapsed_s=%.3f",
                        request_id,
                        len(text_chunk),
                        len(full_text),
                        _estimate_token_count(full_text),
                        time.perf_counter() - started,
                    )
                    if progress_signal is not None:
                        _safe_signal_emit(progress_signal, {
                            "event": "text_chunk",
                            "request_id": request_id,
                            "elapsed_s": time.perf_counter() - started,
                            "text_chars": len(full_text),
                            "text_tokens_est": _estimate_token_count(full_text),
                            "audio_chunks": audio_chunk_count,
                            "message": "Text received from backend",
                        })

                _safe_signal_emit(self.chunk_ready, audio_chunk, text_chunk if isinstance(text_chunk, str) else "")
        except Exception as e:
            logger.exception(
                "trace_id=%s stage=rt_inference event=error elapsed_s=%.3f error=%s",
                request_id,
                time.perf_counter() - started,
                e,
            )
            if progress_signal is not None:
                _safe_signal_emit(progress_signal, {
                    "event": "error",
                    "request_id": request_id,
                    "elapsed_s": time.perf_counter() - started,
                    "text_chars": len(full_text),
                    "text_tokens_est": _estimate_token_count(full_text),
                    "audio_chunks": audio_chunk_count,
                    "message": str(e),
                })
            if error_signal is not None:
                _safe_signal_emit(error_signal, str(e))
            return

        if isinstance(final_override, str):
            full_text = final_override

        logger.info(
            "trace_id=%s stage=rt_inference event=thread_finish elapsed_s=%.3f stopped=%s text_chunks=%d audio_chunks=%d response_chars=%d response_tokens_est=%d",
            request_id,
            time.perf_counter() - started,
            bool(self._stop_requested),
            text_chunk_count,
            audio_chunk_count,
            len(full_text),
            _estimate_token_count(full_text),
        )
        if progress_signal is not None:
            _safe_signal_emit(progress_signal, {
                "event": "thread_finish",
                "request_id": request_id,
                "elapsed_s": time.perf_counter() - started,
                "text_chars": len(full_text),
                "text_tokens_est": _estimate_token_count(full_text),
                "audio_chunks": audio_chunk_count,
                "message": "Inference thread finished",
            })
        _safe_signal_emit(self.finished_signal, full_text)


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
    generation_progress = Signal(object)      # dict payload
    generation_error = Signal(str)

    def __init__(self, chat_mode_cfg: dict):
        super().__init__()
        from tools.audio.conversation import ConversationManager
        self.conv_mgr = ConversationManager(chat_mode_cfg)
        self.mic = MicInputStream()
        self._player = None
        self._inference_thread = None
        self._is_generating = False
        self._prev_state = None
        self._active_trace_context: dict = {}
        self._generation_started_at = 0.0
        self._generated_text_chars = 0
        self._generated_audio_chunks = 0
        self._pending_full_text = ""
        self._drain_timer = None
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(5000)
        self._progress_timer.timeout.connect(self._on_progress_heartbeat)

    def _cancel_drain_timer(self):
        timer = self._drain_timer
        self._drain_timer = None
        if timer is None:
            return
        try:
            timer.stop()
        except Exception:
            pass

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

    def process_turn(self, messages: list[dict], voice_ref, settings: dict, generate_audio: bool = True, trace_context: dict | None = None):
        """Start model inference for a turn (from conversation or single-turn)."""
        if self._is_generating:
            return  # already running

        self._cancel_drain_timer()
        self._active_trace_context = dict(trace_context or {})
        if not self._active_trace_context.get("request_id"):
            self._active_trace_context["request_id"] = f"chat-{uuid4().hex[:8]}"
        self._generation_started_at = time.perf_counter()
        self._generated_text_chars = 0
        self._generated_audio_chunks = 0

        self._is_generating = True
        self.conv_mgr.on_model_start()
        logger.info(
            "trace_id=%s stage=rt_audio event=turn_start generate_audio=%s max_tokens=%d temperature=%.3f",
            self._active_trace_context.get("request_id", "n/a"),
            bool(generate_audio),
            int(settings.get("max_new_tokens", 2048)),
            float(settings.get("temperature", 0.7)),
        )
        self.generation_started.emit()
        self._progress_timer.start()
        self.generation_progress.emit(self._build_progress_payload("turn_start", "Generation requested"))

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
            trace_context=self._active_trace_context,
        )
        self._inference_thread.chunk_ready.connect(self._on_model_chunk)
        self._inference_thread.finished_signal.connect(self._on_model_done)
        self._inference_thread.progress_signal.connect(self._on_thread_progress)
        self._inference_thread.error_signal.connect(self._on_model_error)
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
            self._generated_audio_chunks += 1
        if text_chunk:
            self._generated_text_chars += len(text_chunk)
            self.text_update.emit(text_chunk)
            self.generation_progress.emit(self._build_progress_payload("text_chunk", "Text chunk received"))

    def _on_thread_progress(self, payload: dict):
        if not isinstance(payload, dict):
            return
        merged = dict(payload)
        merged.setdefault("request_id", self._active_trace_context.get("request_id", "n/a"))
        self.generation_progress.emit(merged)

    def _on_progress_heartbeat(self):
        if not self._is_generating:
            return
        payload = self._build_progress_payload("heartbeat", "Generation still running")
        logger.info(
            "trace_id=%s stage=rt_audio event=generation_progress elapsed_s=%.3f text_chars=%d text_tokens_est=%d audio_chunks=%d",
            payload["request_id"],
            payload["elapsed_s"],
            payload["text_chars"],
            payload["text_tokens_est"],
            payload["audio_chunks"],
        )
        self.generation_progress.emit(payload)

    def _build_progress_payload(self, event: str, message: str) -> dict:
        elapsed_s = 0.0
        if self._generation_started_at:
            elapsed_s = time.perf_counter() - self._generation_started_at
        return {
            "event": event,
            "request_id": self._active_trace_context.get("request_id", "n/a"),
            "elapsed_s": elapsed_s,
            "text_chars": self._generated_text_chars,
            "text_tokens_est": _estimate_token_count_from_chars(self._generated_text_chars),
            "audio_chunks": self._generated_audio_chunks,
            "message": message,
        }

    def _on_model_done(self, full_text: str):
        """Handle model inference completion."""
        self._progress_timer.stop()
        if not self._is_generating:
            # Already stopped via barge-in or manual stop — don't reset state
            self.generation_finished.emit(full_text)
            return

        if self._player is not None:
            self._player.finish()
            # Poll for drain completion instead of blocking the UI thread
            self._cancel_drain_timer()
            self._pending_full_text = full_text
            self._drain_timer = QTimer()
            self._drain_timer.setInterval(50)
            self._drain_timer.timeout.connect(self._check_drain)
            self._drain_timer.start()
        else:
            self._finalize_turn(full_text)

    def _on_model_error(self, error_message: str):
        self._progress_timer.stop()
        self._cancel_drain_timer()
        request_id = self._active_trace_context.get("request_id", "n/a")
        logger.error(
            "trace_id=%s stage=rt_audio event=turn_error elapsed_s=%.3f error=%r text_chars=%d audio_chunks=%d",
            request_id,
            time.perf_counter() - self._generation_started_at if self._generation_started_at else 0.0,
            error_message,
            self._generated_text_chars,
            self._generated_audio_chunks,
        )

        if self._player is not None:
            self._player.finish()
            self._player.stop()
            self._player = None

        self._is_generating = False
        self.conv_mgr.on_model_done()
        self.generation_error.emit(error_message)
        self._emit_state()

    def _check_drain(self):
        """Poll until audio player finishes draining."""
        if not self._is_generating:
            self._cancel_drain_timer()
            return
        if self._player is None or self._player._drained.is_set():
            self._cancel_drain_timer()
            if self._player is not None:
                self._player.stop()
                self._player = None
            self._finalize_turn(self._pending_full_text)

    def _finalize_turn(self, full_text: str):
        """Clean up after a completed generation turn."""
        self._cancel_drain_timer()
        self._is_generating = False
        self.conv_mgr.on_model_done()
        logger.info(
            "trace_id=%s stage=rt_audio event=turn_finish elapsed_s=%.3f response_chars=%d response_tokens_est=%d audio_chunks=%d",
            self._active_trace_context.get("request_id", "n/a"),
            time.perf_counter() - self._generation_started_at if self._generation_started_at else 0.0,
            len(full_text or ""),
            _estimate_token_count(full_text or ""),
            self._generated_audio_chunks,
        )
        self.generation_finished.emit(full_text)
        self._emit_state()

    def _stop_generation(self):
        """Interrupt model generation (barge-in or manual stop)."""
        self._progress_timer.stop()
        self._cancel_drain_timer()
        self._pending_full_text = ""
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
            try:
                self._inference_thread.progress_signal.disconnect(self._on_thread_progress)
            except RuntimeError:
                pass
            try:
                self._inference_thread.error_signal.disconnect(self._on_model_error)
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
