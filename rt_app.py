"""
rt_app.py -- PySide6 QMainWindow for OmniChat real-time desktop client.

Contains the main window with 4 tabs: Voice Chat, Vision, Settings, About.
All audio I/O uses direct PCM via sounddevice (no HLS/browser overhead).
"""

import numpy as np
import os
import time
from pathlib import Path
from uuid import uuid4
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QFileDialog,
    QSplitter, QStatusBar, QProgressBar, QMessageBox, QScrollArea, QStackedLayout,
)
from PySide6.QtCore import Qt, QTimer, Signal, QUrl
from PySide6.QtGui import QFont, QIcon, QPixmap, QColor, QDesktopServices

try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget

    MULTIMEDIA_AVAILABLE = True
except Exception:
    QAudioOutput = None
    QMediaPlayer = None
    QVideoWidget = None
    MULTIMEDIA_AVAILABLE = False

from rt_audio import AudioPipeline
from tools.model import model_manager as mm
from tools.model.clients import QwenOmniClient, QwenOmniClientConfig
from tools.model.model_manager import (
    get_backend_capabilities,
    get_backend_name,
    get_backend_status,
    get_qwen_llamacpp_config,
    get_qwen_remote_config,
    set_qwen_remote_config,
)
from tools.shared.debug_trace import get_trace_logger, get_trace_log_path
from tools.shared.chat_render import render_chat_history_html
from tools.shared import session_playback
from tools.shared.session_window_recorder import SessionWindowRecorder
from tools.shared.session_recorder import SessionRecorder
from tools.shared.session import (
    detect_voice_command,
    get_truncated_voice_ref,
)
from tools.audio.voice_manager import list_voices, get_voice, add_voice, delete_voice

BASE_DIR = Path(__file__).parent.resolve()
logger = get_trace_logger()


def _default_voice_name_from_path(path: str) -> str:
    stem = Path(path).stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(part for part in stem.split() if part)


def load_voice_sample_from_media(path: str, *, target_sr: int = 16000) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    if np.asarray(audio).ndim > 1:
        audio = np.asarray(audio, dtype=np.float32).mean(axis=-1)
    if int(sample_rate) != int(target_sr):
        import librosa

        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sample_rate, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def record_voice_sample(*, duration_s: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    import sounddevice as sd

    frames = max(1, int(float(duration_s) * int(sample_rate)))
    recording = sd.rec(frames, samplerate=int(sample_rate), channels=1, dtype="float32")
    sd.wait()
    return np.asarray(recording, dtype=np.float32).reshape(-1)


def _estimate_token_count(text: str) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, round(len(normalized) / 4.0))


def _compact_model_specifier(value: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        return ""
    if any(sep in normalized for sep in ("/", "\\")):
        return Path(normalized).name or normalized.split("/")[-1]
    return normalized


class OmniChatWindow(QMainWindow):
    """Main application window with tabbed interface."""

    def __init__(self, settings: dict):
        super().__init__()
        self._settings = settings
        audio_cfg = settings.get("audio", {})
        inference = settings.get("inference", {})
        output_cfg = settings.get("output", {})
        streaming_cfg = audio_cfg.get("streaming", {})
        voice_cfg = settings.get("voice_commands", {})
        chat_mode_cfg = audio_cfg.get("chat_mode", {})

        # Session state (mirrors main.py's session_settings)
        self._session = {
            "temperature": inference.get("temperature", 0.7),
            "max_new_tokens": inference.get("max_new_tokens", 2048),
            "repetition_penalty": inference.get("repetition_penalty", 1.05),
            "top_p": inference.get("top_p", 0.8),
            "top_k": inference.get("top_k", 100),
            "enable_thinking": inference.get("enable_thinking", False),
            "max_frames": inference.get("max_frames", 64),
            "output_format": output_cfg.get("default_format", "auto"),
            "voice_sample_length_s": audio_cfg.get("voice_sample_length_s", 5.0),
            "font_size": settings.get("display", {}).get("font_size", 12),
        }
        self._voice_ref = None
        self._voice_name = None
        self._chat_history = []
        self._fuzzy_threshold = voice_cfg.get("fuzzy_threshold", 0.6)
        self._active_request_id = None
        self._active_response_is_spoken = False
        self._recording_enabled = False
        self._recording_mode = "metadata-only"
        self._transcript_policy = "full"
        self._active_turn_started_at = 0.0
        self._active_turn_first_text_s = None
        self._active_turn_modality = "text"
        self._active_turn_prompt_text = ""
        self._active_turn_user_audio = None
        self._active_turn_model_audio_chunks: list[np.ndarray] = []
        self._active_turn_user_audio_offset_s = 0.0
        self._recording_session_started_at = 0.0
        self._last_generation_progress: dict = {}
        self._response_status_phase = "idle"
        self._response_display_mode = "text"
        self._voice_backend_label = QLabel("")
        self._backend_summary_label = QLabel("")
        self._backend_label = QLabel("")
        self._about_backend_label = QLabel("")
        self._startup_status_label = QLabel("")
        self._startup_phase_label = QLabel("")
        self._remote_probe_label = QLabel("")
        remote_cfg = get_qwen_remote_config()
        endpoints = remote_cfg.get("endpoints", {})
        self._remote_base_url = QLineEdit(remote_cfg.get("base_url") or "")
        self._remote_model_name = QLineEdit(remote_cfg.get("model_name") or "")
        self._remote_api_key = QLineEdit(remote_cfg.get("api_key") or "")
        self._remote_timeout = QDoubleSpinBox()
        self._remote_timeout.setRange(1.0, 600.0)
        self._remote_timeout.setSingleStep(1.0)
        self._remote_timeout.setValue(float(remote_cfg.get("timeout_s") or 120.0))
        self._remote_health_ep = QLineEdit(endpoints.get("health") or "/health")
        self._remote_models_ep = QLineEdit(endpoints.get("models") or "models")
        self._remote_chat_ep = QLineEdit(endpoints.get("chat_completions") or "chat/completions")
        self._remote_responses_ep = QLineEdit(endpoints.get("responses") or "responses")
        self._remote_realtime_ep = QLineEdit(endpoints.get("realtime") or "realtime")
        self._playback_root = Path(output_cfg.get("save_dir", "outputs")) / "sessions"
        self._playback_session: session_playback.SessionPlaybackManifest | None = None
        self._playback_events: list[session_playback.SessionReplayEvent] = []
        self._playback_event_index = -1
        self._playback_audio_cache: dict[Path, tuple[np.ndarray, int]] = {}
        self._playback_is_running = False
        self._playback_media_player = None
        self._playback_audio_output = None
        self._playback_video_widget = None
        self._playback_timer = QTimer(self)
        self._playback_timer.setSingleShot(True)
        self._playback_timer.timeout.connect(self._advance_playback)
        self._recording_root = Path(output_cfg.get("save_dir", "outputs")) / "sessions"
        self._session_recorder: SessionRecorder | None = None
        self._session_video_recorder: SessionWindowRecorder | None = None

        # Load default voice if configured
        default_voice = audio_cfg.get("default_voice")
        if default_voice:
            result = get_voice(default_voice, self._fuzzy_threshold)
            if result["found"]:
                self._voice_ref = result["audio"]
                self._voice_name = result["name"]

        # Audio pipeline
        self._pipeline = AudioPipeline(chat_mode_cfg)
        self._pipeline.state_changed.connect(self._on_state_changed)
        self._pipeline.speech_ready.connect(self._on_speech_ready)
        self._pipeline.text_update.connect(self._on_text_update)
        self._pipeline.audio_chunk_ready.connect(self._on_audio_chunk_ready)
        self._pipeline.generation_started.connect(self._on_generation_started)
        self._pipeline.generation_finished.connect(self._on_generation_finished)
        if hasattr(self._pipeline, "generation_progress"):
            self._pipeline.generation_progress.connect(self._on_generation_progress)
        if hasattr(self._pipeline, "generation_error"):
            self._pipeline.generation_error.connect(self._on_generation_error)
        self._pipeline.barge_in_detected.connect(self._on_barge_in)

        self._setup_ui()
        self._setup_statusbar()
        self.setWindowTitle("OmniChat RT")
        icon_path = BASE_DIR / "assets" / "omnichat.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        # Apply configured font size, then set window size
        self._apply_font_size(self._session.get("font_size", 12))
        self.setMinimumSize(640, 480)
        self.resize(900, 700)
        self._refresh_backend_status(show_message=False)
        self._startup_phase_label.setText(self._startup_status_label.text())

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_voice_tab(), "Voice Chat")
        tabs.addTab(self._create_playback_tab(), "Playback")
        tabs.addTab(self._create_vision_tab(), "Vision")
        tabs.addTab(self._create_settings_tab(), "Settings")
        tabs.addTab(self._create_about_tab(), "About")
        self.setCentralWidget(tabs)

    def _setup_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(f"Ready | Trace: {get_trace_log_path()}")

    def _new_trace_context(self) -> dict:
        request_id = f"chat-{uuid4().hex[:8]}"
        self._active_request_id = request_id
        return {"request_id": request_id}

    def _current_backend_name(self) -> str:
        model_cfg = self._settings.get("model", {})
        try:
            backend_name = get_backend_name()
        except Exception:
            backend_name = None
        return str(backend_name or model_cfg.get("backend") or "minicpm")

    def _current_model_profile(self) -> str:
        return str(
            self._settings.get("active_model_profile")
            or self._settings.get("model_profile")
            or "unknown"
        )

    def _assistant_label(self) -> str:
        model_cfg = self._settings.get("model", {})
        specifier = (
            model_cfg.get("display_name")
            or model_cfg.get("name")
            or model_cfg.get("checkpoint")
            or self._settings.get("active_model_profile")
            or self._settings.get("model_profile")
            or self._current_backend_name()
        )
        specifier_text = _compact_model_specifier(str(specifier or "")) or self._current_backend_name()
        return f"OmniChat [{specifier_text}]"

    def _current_model_recording_metadata(self) -> dict:
        model_cfg = self._settings.get("model", {})
        voice_cfg = model_cfg.get("voice", {}) if isinstance(model_cfg.get("voice"), dict) else {}
        return {
            "profile": self._current_model_profile(),
            "backend": self._current_backend_name(),
            "display_name": model_cfg.get("display_name"),
            "name": model_cfg.get("name"),
            "checkpoint": model_cfg.get("checkpoint"),
            "quantization": model_cfg.get("quantization"),
            "transport": model_cfg.get("transport"),
            "speech_backend": self._current_speech_backend(),
            "voice": dict(voice_cfg),
        }

    def _current_voice_recording_metadata(self) -> dict:
        model_cfg = self._settings.get("model", {})
        voice_cfg = model_cfg.get("voice", {}) if isinstance(model_cfg.get("voice"), dict) else {}
        builtin_name = voice_cfg.get("builtin_voice_name") or model_cfg.get("speaker")

        if self._voice_ref is not None and self._voice_name:
            return {"mode": "cloned", "name": self._voice_name}
        if builtin_name:
            return {"mode": "builtin", "name": f"built-in {builtin_name}"}
        if voice_cfg.get("supports_voice_clone"):
            return {"mode": "default", "name": "default voice"}
        return {"mode": "none", "name": None}

    def _sync_record_button(self) -> None:
        if not hasattr(self, "_record_btn"):
            return
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(self._recording_enabled)
        mode_label = self._record_mode_label()
        self._record_btn.setText(f"Recording {mode_label}" if self._recording_enabled else "Record Off")
        self._record_btn.blockSignals(False)
        if hasattr(self, "_record_mode_combo"):
            self._record_mode_combo.blockSignals(True)
            index = self._record_mode_combo.findData(self._recording_mode)
            if index >= 0:
                self._record_mode_combo.setCurrentIndex(index)
            self._record_mode_combo.blockSignals(False)
        if hasattr(self, "_transcript_policy_combo"):
            self._transcript_policy_combo.blockSignals(True)
            index = self._transcript_policy_combo.findData(self._transcript_policy)
            if index >= 0:
                self._transcript_policy_combo.setCurrentIndex(index)
            self._transcript_policy_combo.blockSignals(False)

    def _record_mode_label(self) -> str:
        labels = {
            "metadata-only": "Metadata",
            "structured": "Structured",
            "video": "Video",
            "structured-video": "Structured + Video",
        }
        return labels.get(self._recording_mode, self._recording_mode)

    def _recording_status_summary(self) -> str:
        if not self._recording_enabled:
            return "Recording disabled"
        if self._session_recorder is None:
            return f"Recording enabled | Mode: {self._record_mode_label()}"
        return (
            f"Recording enabled | Mode: {self._record_mode_label()} | Policy: {self._transcript_policy} | "
            f"Session: {self._session_recorder.manifest_path}"
        )

    def _recording_uses_video(self) -> bool:
        return self._recording_mode in {"video", "structured-video"}

    def _recording_keeps_structured_audio(self) -> bool:
        return self._recording_mode in {"structured", "structured-video"}

    def _ensure_recording_session(self) -> None:
        if self._session_recorder is not None:
            self._session_recorder.set_recording_config(
                enabled=self._recording_enabled,
                recording_mode=self._recording_mode,
                transcript_policy=self._transcript_policy,
            )
            if self._recording_uses_video() and self._session_video_recorder is None:
                self._session_video_recorder = SessionWindowRecorder(output_dir=self._session_recorder.session_dir, widget=self)
                self._session_video_recorder.start()
            return
        self._session_recorder = SessionRecorder(
            output_root=self._recording_root,
            frontend="rt",
            session_metadata={"model": self._current_model_recording_metadata()},
            recording_enabled=self._recording_enabled,
            recording_mode=self._recording_mode,
            transcript_policy=self._transcript_policy,
        )
        self._recording_session_started_at = time.perf_counter()
        if self._recording_uses_video():
            self._session_video_recorder = SessionWindowRecorder(output_dir=self._session_recorder.session_dir, widget=self)
            self._session_video_recorder.start()

    def _current_recording_elapsed_s(self) -> float:
        if self._session_video_recorder is not None:
            return float(self._session_video_recorder.elapsed_s())
        if self._recording_session_started_at > 0:
            return max(0.0, time.perf_counter() - self._recording_session_started_at)
        return 0.0

    def _finalize_recording_session(self) -> None:
        recorder = self._session_recorder
        if self._session_video_recorder is not None and self._session_recorder is not None:
            metadata = self._session_video_recorder.stop()
            if metadata:
                self._session_recorder.register_session_video(
                    video_path=str(metadata.get("video_path") or "session.mp4"),
                    duration_s=float(metadata.get("duration_s") or 0.0),
                    fps=int(metadata.get("fps") or 0),
                    frame_count=int(metadata.get("frame_count") or 0),
                )
            self._session_video_recorder = None
        self._recording_session_started_at = 0.0
        self._session_recorder = None
        return recorder

    def _on_record_mode_changed(self, _index: int) -> None:
        if not hasattr(self, "_record_mode_combo"):
            return
        previous_uses_video = self._recording_uses_video()
        self._recording_mode = str(self._record_mode_combo.currentData() or "metadata-only")
        if self._session_recorder is not None:
            self._session_recorder.set_recording_config(recording_mode=self._recording_mode)
        if self._recording_enabled:
            if self._recording_uses_video():
                self._ensure_recording_session()
            elif previous_uses_video and self._session_video_recorder is not None:
                self._finalize_recording_session()
        self._sync_record_button()

    def _on_transcript_policy_changed(self, _index: int) -> None:
        if not hasattr(self, "_transcript_policy_combo"):
            return
        self._transcript_policy = str(self._transcript_policy_combo.currentData() or "full")
        if self._session_recorder is not None:
            self._session_recorder.set_recording_config(transcript_policy=self._transcript_policy)
        self._status.showMessage(self._recording_status_summary())

    def _current_speech_backend(self) -> str:
        model_cfg = self._settings.get("model", {})
        backend = self._current_backend_name()
        if backend in {"qwen_llamacpp", "gemma_llamacpp"}:
            try:
                return str(get_qwen_llamacpp_config().get("speech_backend") or model_cfg.get("llama_cpp", {}).get("speech_backend") or "none")
            except Exception:
                return str(model_cfg.get("llama_cpp", {}).get("speech_backend") or "none")
        if backend == "gemma_transformers":
            return str(model_cfg.get("speech_backend") or "none")
        if backend in {"minicpm", "qwen_remote", "qwen_transformers"}:
            return "native"
        return "none"

    def _set_response_phase(self, phase: str) -> None:
        self._response_status_phase = phase
        if phase == "thinking":
            message = "Thinking..."
        elif phase == "generating":
            message = "Generating..."
        elif phase == "text":
            message = "Responding (Text)..."
        elif phase == "tts":
            message = "Responding (TTS)..."
        elif phase == "audio":
            message = "Responding (Audio)..."
        else:
            message = "Ready"
        self._state_label.setText(message)
        self._status.showMessage(message)

    def _refresh_backend_status(self, *, show_message: bool = True) -> dict:
        status = get_backend_status()
        summary = mm.summarize_backend_status(status)
        detail = mm.format_backend_status(status)
        self._voice_backend_label.setText(summary)
        self._backend_summary_label.setText(summary)
        self._backend_label.setText(detail)
        self._about_backend_label.setText(detail)
        if status.get("backend") == "qwen_remote":
            startup_text = "Startup: Window ready"
        else:
            startup_text = "Startup: Ready"
        self._startup_status_label.setText(startup_text)
        if not self._startup_phase_label.text():
            self._startup_phase_label.setText(startup_text)
        if show_message:
            self._status.showMessage(summary)
        return status

    def _on_remote_warmup_complete(self, result: dict) -> None:
        if bool((result or {}).get("ok")):
            first_text_s = float((result or {}).get("first_text_s") or 0.0)
            self._startup_phase_label.setText("Startup: Remote backend ready")
            self._startup_status_label.setText(f"Startup: Remote backend ready | First text in {first_text_s:.1f}s")
        else:
            error = str((result or {}).get("error") or "unknown error")
            self._startup_phase_label.setText("Startup: Remote warmup failed")
            self._startup_status_label.setText(f"Startup: Remote warmup failed | {error}")

    def _backend_supports_audio_output(self) -> bool:
        backend = self._current_backend_name()
        speech_backend = self._current_speech_backend()
        if backend in {"qwen_llamacpp", "gemma_llamacpp"}:
            return speech_backend == "minicpm_streaming"
        return speech_backend != "none"

    # ── Voice Chat Tab ────────────────────────────────────────────────────

    def _create_voice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Top controls use two rows so buttons stay button-shaped instead of growing into tall wrapped blocks.
        controls_layout = QVBoxLayout()
        self._voice_controls_top_row = QHBoxLayout()
        self._voice_controls_bottom_row = QHBoxLayout()

        self._conv_start_btn = QPushButton("Start Conversation")
        self._conv_start_btn.clicked.connect(self._start_conversation)
        self._voice_controls_top_row.addWidget(self._conv_start_btn)

        self._conv_stop_btn = QPushButton("Stop")
        self._conv_stop_btn.setEnabled(False)
        self._conv_stop_btn.clicked.connect(self._stop_conversation)
        self._voice_controls_top_row.addWidget(self._conv_stop_btn)

        self._voice_controls_top_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Auto-detect", "Push-to-talk", "Click per turn"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._voice_controls_top_row.addWidget(self._mode_combo)

        # State indicator
        self._state_dot = QLabel("\u25cf")  # filled circle
        self._state_dot.setFont(QFont("Segoe UI", 16))
        self._state_dot.setStyleSheet("color: #888888;")
        self._voice_controls_top_row.addWidget(self._state_dot)

        self._state_label = QLabel("OFF")
        self._voice_controls_top_row.addWidget(self._state_label)

        self._voice_backend_label.setWordWrap(True)
        self._voice_controls_top_row.addWidget(self._voice_backend_label, stretch=1)

        self._record_btn = QPushButton("Record Off")
        self._record_btn.setCheckable(True)
        self._record_btn.toggled.connect(self._on_record_toggled)
        self._voice_controls_bottom_row.addWidget(self._record_btn)
        self._voice_controls_bottom_row.addWidget(QLabel("Mode:"))
        self._record_mode_combo = QComboBox()
        self._record_mode_combo.addItem("Metadata Only", userData="metadata-only")
        self._record_mode_combo.addItem("Structured", userData="structured")
        self._record_mode_combo.addItem("Video", userData="video")
        self._record_mode_combo.addItem("Structured + Video", userData="structured-video")
        self._record_mode_combo.currentIndexChanged.connect(self._on_record_mode_changed)
        self._voice_controls_bottom_row.addWidget(self._record_mode_combo)
        self._voice_controls_bottom_row.addWidget(QLabel("Transcript:"))
        self._transcript_policy_combo = QComboBox()
        self._transcript_policy_combo.addItem("Full", userData="full")
        self._transcript_policy_combo.addItem("Redacted", userData="redacted")
        self._transcript_policy_combo.addItem("Omitted", userData="omitted")
        self._transcript_policy_combo.currentIndexChanged.connect(self._on_transcript_policy_changed)
        self._voice_controls_bottom_row.addWidget(self._transcript_policy_combo)
        self._sync_record_button()

        self._voice_controls_bottom_row.addStretch()

        # Voice selector
        self._voice_controls_bottom_row.addWidget(QLabel("Voice:"))
        self._voice_combo = QComboBox()
        self._refresh_voice_list()
        self._voice_combo.currentTextChanged.connect(self._on_voice_changed)
        self._voice_controls_bottom_row.addWidget(self._voice_combo)

        controls_layout.addLayout(self._voice_controls_top_row)
        controls_layout.addLayout(self._voice_controls_bottom_row)
        layout.addLayout(controls_layout)

        # Push-to-talk button (visible only in PTT mode)
        self._ptt_btn = QPushButton("Hold to Talk")
        self._ptt_btn.setMinimumHeight(50)
        self._ptt_btn.setVisible(False)
        self._ptt_btn.pressed.connect(self._on_ptt_pressed)
        self._ptt_btn.released.connect(self._on_ptt_released)
        layout.addWidget(self._ptt_btn)

        # Chat history
        self._history_display = QTextEdit()
        self._history_display.setReadOnly(True)
        self._history_display.setFont(QFont("Segoe UI", 10))
        self._history_display.setPlaceholderText("Chat history will appear here...")
        layout.addWidget(self._history_display, stretch=1)

        # Text input row
        input_row = QHBoxLayout()
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("Type a message and press Enter...")
        self._text_input.returnPressed.connect(self._send_text)
        input_row.addWidget(self._text_input, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._send_text)
        input_row.addWidget(self._send_btn)

        layout.addLayout(input_row)

        return tab

    # ── Vision Tab ────────────────────────────────────────────────────────

    def _create_vision_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Image section
        img_group = QGroupBox("Image Analysis")
        img_layout = QVBoxLayout(img_group)

        img_controls = QHBoxLayout()
        self._img_upload_btn = QPushButton("Upload Image")
        self._img_upload_btn.clicked.connect(self._upload_image)
        img_controls.addWidget(self._img_upload_btn)

        self._doc_mode_check = QCheckBox("Document/OCR mode")
        img_controls.addWidget(self._doc_mode_check)

        self._img_prompt = QLineEdit()
        self._img_prompt.setPlaceholderText("Describe this image... (optional)")
        img_controls.addWidget(self._img_prompt, stretch=1)

        self._analyze_img_btn = QPushButton("Analyze")
        self._analyze_img_btn.clicked.connect(self._analyze_image)
        img_controls.addWidget(self._analyze_img_btn)
        img_layout.addLayout(img_controls)

        self._img_preview = QLabel()
        self._img_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_preview.setMaximumHeight(200)
        self._img_preview.setStyleSheet("background-color: #2e2e3e; border-radius: 4px;")
        img_layout.addWidget(self._img_preview)
        self._img_path = None

        layout.addWidget(img_group)

        # Video / Audio section
        vid_group = QGroupBox("Video / Audio")
        vid_layout = QHBoxLayout(vid_group)

        self._vid_upload_btn = QPushButton("Upload")
        self._vid_upload_btn.clicked.connect(self._upload_video)
        vid_layout.addWidget(self._vid_upload_btn)

        self._vid_label = QLabel("No file selected")
        vid_layout.addWidget(self._vid_label, stretch=1)

        self._vid_prompt = QLineEdit()
        self._vid_prompt.setPlaceholderText("What's happening? (optional)")
        vid_layout.addWidget(self._vid_prompt, stretch=1)

        self._analyze_vid_btn = QPushButton("Analyze")
        self._analyze_vid_btn.clicked.connect(self._analyze_video)
        vid_layout.addWidget(self._analyze_vid_btn)

        self._transcribe_vid_btn = QPushButton("Transcribe")
        self._transcribe_vid_btn.clicked.connect(self._transcribe_video)
        vid_layout.addWidget(self._transcribe_vid_btn)
        self._vid_path = None

        layout.addWidget(vid_group)

        # PDF section
        pdf_group = QGroupBox("PDF Document Scanning")
        pdf_layout = QVBoxLayout(pdf_group)

        pdf_controls = QHBoxLayout()
        self._pdf_upload_btn = QPushButton("Upload PDF(s)")
        self._pdf_upload_btn.clicked.connect(self._upload_pdf)
        pdf_controls.addWidget(self._pdf_upload_btn)

        self._bank_mode_check = QCheckBox("Bank statement mode")
        self._bank_mode_check.setChecked(True)
        pdf_controls.addWidget(self._bank_mode_check)

        self._accumulate_check = QCheckBox("Accumulate")
        self._accumulate_check.setToolTip("Append new scan results to previous table data")
        pdf_controls.addWidget(self._accumulate_check)

        self._pdf_prompt = QLineEdit()
        self._pdf_prompt.setPlaceholderText("Custom prompt (ignored in bank mode)...")
        pdf_controls.addWidget(self._pdf_prompt, stretch=1)

        self._scan_pdf_btn = QPushButton("Scan")
        self._scan_pdf_btn.clicked.connect(self._scan_pdf)
        pdf_controls.addWidget(self._scan_pdf_btn)

        self._clear_pdf_btn = QPushButton("Clear")
        self._clear_pdf_btn.clicked.connect(self._clear_pdf_results)
        pdf_controls.addWidget(self._clear_pdf_btn)
        pdf_layout.addLayout(pdf_controls)

        pdf_info_row = QHBoxLayout()
        self._pdf_label = QLabel("No PDF selected")
        pdf_info_row.addWidget(self._pdf_label, stretch=1)
        self._pdf_progress = QProgressBar()
        self._pdf_progress.setVisible(False)
        pdf_info_row.addWidget(self._pdf_progress, stretch=1)
        pdf_layout.addLayout(pdf_info_row)

        self._pdf_paths = []
        self._pdf_table = None

        layout.addWidget(pdf_group)

        # Output
        self._vision_output = QTextEdit()
        self._vision_output.setPlaceholderText("Analysis results will appear here...")
        layout.addWidget(self._vision_output, stretch=1)

        # Save row
        save_row = QHBoxLayout()
        save_row.addStretch()
        self._save_btn = QPushButton("Save As...")
        self._save_btn.clicked.connect(self._save_vision_output)
        save_row.addWidget(self._save_btn)
        layout.addLayout(save_row)

        return tab

    # ── Playback Tab ─────────────────────────────────────────────────────

    def _create_playback_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Session:"))
        self._playback_session_combo = QComboBox()
        self._playback_session_combo.currentIndexChanged.connect(self._load_selected_playback_session)
        selector_row.addWidget(self._playback_session_combo, stretch=1)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_playback_sessions)
        selector_row.addWidget(refresh_btn)
        browse_btn = QPushButton("Open JSON")
        browse_btn.clicked.connect(self._browse_playback_session)
        selector_row.addWidget(browse_btn)
        folder_btn = QPushButton("Open Folder")
        folder_btn.clicked.connect(self._open_playback_folder)
        selector_row.addWidget(folder_btn)
        layout.addLayout(selector_row)

        control_row = QHBoxLayout()
        self._playback_play_btn = QPushButton("Play From Start")
        self._playback_play_btn.clicked.connect(self._play_selected_session)
        control_row.addWidget(self._playback_play_btn)
        self._playback_stop_btn = QPushButton("Stop")
        self._playback_stop_btn.clicked.connect(self._stop_playback)
        control_row.addWidget(self._playback_stop_btn)
        prev_btn = QPushButton("Prev")
        prev_btn.clicked.connect(self._show_previous_playback_event)
        control_row.addWidget(prev_btn)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self._show_next_playback_event)
        control_row.addWidget(next_btn)
        export_audio_btn = QPushButton("Export Audio")
        export_audio_btn.clicked.connect(self._export_selected_session_audio)
        control_row.addWidget(export_audio_btn)
        export_video_btn = QPushButton("Export Video")
        export_video_btn.clicked.connect(self._export_selected_session_video)
        control_row.addWidget(export_video_btn)
        self._playback_open_video_btn = QPushButton("Open Video")
        self._playback_open_video_btn.clicked.connect(self._open_selected_playback_video)
        self._playback_open_video_btn.setEnabled(False)
        control_row.addWidget(self._playback_open_video_btn)
        layout.addLayout(control_row)

        self._playback_summary_label = QLabel("No session selected")
        self._playback_summary_label.setWordWrap(True)
        layout.addWidget(self._playback_summary_label)

        self._playback_progress = QProgressBar()
        self._playback_progress.setMinimum(0)
        self._playback_progress.setMaximum(1)
        self._playback_progress.setValue(0)
        layout.addWidget(self._playback_progress)

        video_group = QGroupBox("Session Video")
        video_layout = QVBoxLayout(video_group)
        if MULTIMEDIA_AVAILABLE:
            self._playback_video_host = QWidget()
            stack = QStackedLayout(self._playback_video_host)
            stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
            self._playback_video_widget = QVideoWidget()
            self._playback_video_widget.setMinimumHeight(220)
            stack.addWidget(self._playback_video_widget)
            self._playback_overlay_label = QLabel("No session video loaded")
            self._playback_overlay_label.setWordWrap(True)
            self._playback_overlay_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            self._playback_overlay_label.setStyleSheet("background-color: rgba(12, 18, 28, 170); color: white; padding: 10px; border-radius: 6px;")
            self._playback_overlay_label.setMargin(8)
            self._playback_overlay_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            stack.addWidget(self._playback_overlay_label)
            video_layout.addWidget(self._playback_video_host)
            self._playback_media_status = QLabel("Embedded video ready")
            video_layout.addWidget(self._playback_media_status)
            self._playback_audio_output = QAudioOutput(self)
            self._playback_media_player = QMediaPlayer(self)
            self._playback_media_player.setAudioOutput(self._playback_audio_output)
            self._playback_media_player.setVideoOutput(self._playback_video_widget)
            self._playback_media_player.positionChanged.connect(self._on_playback_video_position_changed)
        else:
            self._playback_overlay_label = QLabel("Embedded video playback is unavailable in this environment. Use Open Video instead.")
            self._playback_overlay_label.setWordWrap(True)
            video_layout.addWidget(self._playback_overlay_label)
            self._playback_media_status = QLabel("Embedded video unavailable")
            video_layout.addWidget(self._playback_media_status)
        layout.addWidget(video_group)

        self._playback_history_display = QTextEdit()
        self._playback_history_display.setReadOnly(True)
        self._playback_history_display.setFont(QFont("Segoe UI", 10))
        self._playback_history_display.setPlaceholderText("Recorded sessions will render here...")
        layout.addWidget(self._playback_history_display, stretch=1)

        self._refresh_playback_sessions()
        return tab

    # ── Settings Tab ──────────────────────────────────────────────────────

    def _create_settings_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        inner = QWidget()
        layout = QVBoxLayout(inner)

        # Inference settings
        inf_group = QGroupBox("Inference")
        inf_layout = QVBoxLayout(inf_group)

        self._temp_spin = self._add_double_spin(
            inf_layout, "Temperature:", 0.0, 1.5, 0.05,
            self._session["temperature"],
            lambda v: self._session.__setitem__("temperature", v),
        )
        self._tokens_spin = self._add_int_spin(
            inf_layout, "Max response tokens:", 128, 8192, 128,
            self._session["max_new_tokens"],
            lambda v: self._session.__setitem__("max_new_tokens", v),
        )
        self._rep_penalty_spin = self._add_double_spin(
            inf_layout, "Repetition penalty:", 1.0, 2.0, 0.05,
            self._session["repetition_penalty"],
            lambda v: self._session.__setitem__("repetition_penalty", v),
        )
        self._top_p_spin = self._add_double_spin(
            inf_layout, "Top-p (nucleus sampling):", 0.0, 1.0, 0.05,
            self._session["top_p"],
            lambda v: self._session.__setitem__("top_p", v),
        )
        self._top_k_spin = self._add_int_spin(
            inf_layout, "Top-k:", 1, 500, 10,
            self._session["top_k"],
            lambda v: self._session.__setitem__("top_k", v),
        )
        self._thinking_check = QCheckBox("Enable thinking mode (chain-of-thought)")
        self._thinking_check.setChecked(self._session["enable_thinking"])
        self._thinking_check.toggled.connect(
            lambda v: self._session.__setitem__("enable_thinking", v)
        )
        inf_layout.addWidget(self._thinking_check)
        layout.addWidget(inf_group)

        # Video settings
        vid_group = QGroupBox("Video")
        vid_layout = QVBoxLayout(vid_group)

        self._max_frames_spin = self._add_int_spin(
            vid_layout, "Max video frames:", 8, 512, 8,
            self._session["max_frames"],
            lambda v: self._update_max_frames(v),
        )

        self._vram_estimate_label = QLabel()
        vid_layout.addWidget(self._vram_estimate_label)
        self._update_vram_estimate(self._session["max_frames"])
        layout.addWidget(vid_group)

        # Audio settings
        audio_group = QGroupBox("Audio")
        audio_layout = QVBoxLayout(audio_group)

        self._voice_len_spin = self._add_double_spin(
            audio_layout, "Voice sample length (s):", 1.0, 30.0, 1.0,
            self._session["voice_sample_length_s"],
            lambda v: self._session.__setitem__("voice_sample_length_s", v),
        )
        layout.addWidget(audio_group)

        # Conversation settings
        conv_group = QGroupBox("Conversation Mode")
        conv_layout = QVBoxLayout(conv_group)

        mgr = self._pipeline.conv_mgr
        self._add_double_spin(
            conv_layout, "Silence threshold (s):", 0.5, 5.0, 0.1,
            mgr._silence_threshold_s,
            lambda v: setattr(mgr, "_silence_threshold_s", v),
        )
        self._add_double_spin(
            conv_layout, "VAD threshold:", 0.1, 0.95, 0.05,
            mgr._vad_threshold,
            lambda v: setattr(mgr, "_vad_threshold", v),
        )
        self._add_double_spin(
            conv_layout, "Echo cooldown (s):", 0.0, 10.0, 0.5,
            mgr._echo_cooldown_s,
            lambda v: setattr(mgr, "_echo_cooldown_s", v),
        )
        self._add_double_spin(
            conv_layout, "Anti-vox boost:", 0.0, 0.5, 0.05,
            mgr._antivox_boost,
            lambda v: setattr(mgr, "_antivox_boost", v),
        )

        barge_in_check = QCheckBox("Enable barge-in (interrupt model)")
        barge_in_check.setChecked(mgr._barge_in_enabled)
        barge_in_check.toggled.connect(lambda v: setattr(mgr, "_barge_in_enabled", v))
        conv_layout.addWidget(barge_in_check)
        layout.addWidget(conv_group)

        # Voice management
        voice_group = QGroupBox("Voice Management")
        voice_layout = QVBoxLayout(voice_group)

        add_row = QHBoxLayout()
        self._voice_add_name = QLineEdit()
        self._voice_add_name.setPlaceholderText("Voice name")
        add_row.addWidget(self._voice_add_name)
        add_from_file_btn = QPushButton("Add from WAV")
        add_from_file_btn.clicked.connect(self._add_voice_from_file)
        add_row.addWidget(add_from_file_btn)
        record_from_mic_btn = QPushButton("Record From Mic")
        record_from_mic_btn.clicked.connect(self._record_voice_from_mic)
        add_row.addWidget(record_from_mic_btn)
        voice_layout.addLayout(add_row)

        del_row = QHBoxLayout()
        self._voice_del_combo = QComboBox()
        self._voice_del_combo.addItems(list_voices())
        del_row.addWidget(self._voice_del_combo, stretch=1)
        del_btn = QPushButton("Delete Voice")
        del_btn.clicked.connect(self._delete_voice)
        del_row.addWidget(del_btn)
        voice_layout.addLayout(del_row)
        layout.addWidget(voice_group)

        backend_group = QGroupBox("Backend Status")
        backend_layout = QVBoxLayout(backend_group)
        self._backend_summary_label.setWordWrap(True)
        self._backend_label.setWordWrap(True)
        self._startup_status_label.setWordWrap(True)
        self._startup_phase_label.setWordWrap(True)
        backend_layout.addWidget(self._backend_summary_label)
        backend_layout.addWidget(self._backend_label)
        backend_layout.addWidget(self._startup_status_label)
        backend_layout.addWidget(self._startup_phase_label)
        layout.addWidget(backend_group)

        remote_group = QGroupBox("Remote Backend")
        remote_layout = QVBoxLayout(remote_group)
        for label_text, widget in [
            ("Base URL:", self._remote_base_url),
            ("Model Name:", self._remote_model_name),
            ("API Key:", self._remote_api_key),
            ("Health Endpoint:", self._remote_health_ep),
            ("Models Endpoint:", self._remote_models_ep),
            ("Chat Endpoint:", self._remote_chat_ep),
            ("Responses Endpoint:", self._remote_responses_ep),
            ("Realtime Endpoint:", self._remote_realtime_ep),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            row.addWidget(widget, stretch=1)
            remote_layout.addLayout(row)
        timeout_row = QHBoxLayout()
        timeout_row.addWidget(QLabel("Timeout (s):"))
        timeout_row.addWidget(self._remote_timeout)
        remote_layout.addLayout(timeout_row)
        remote_btn_row = QHBoxLayout()
        apply_remote_btn = QPushButton("Apply Remote Settings")
        apply_remote_btn.clicked.connect(self._apply_remote_settings)
        remote_btn_row.addWidget(apply_remote_btn)
        test_health_btn = QPushButton("Test Health")
        test_health_btn.clicked.connect(self._test_remote_health)
        remote_btn_row.addWidget(test_health_btn)
        test_realtime_btn = QPushButton("Test Realtime")
        test_realtime_btn.clicked.connect(self._test_remote_realtime)
        remote_btn_row.addWidget(test_realtime_btn)
        remote_layout.addLayout(remote_btn_row)
        self._remote_probe_label.setWordWrap(True)
        remote_layout.addWidget(self._remote_probe_label)
        layout.addWidget(remote_group)

        # Display settings
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self._font_size_spin = self._add_int_spin(
            display_layout, "Font size:", 8, 32, 1,
            self._session.get("font_size", 12),
            self._apply_font_size,
        )
        layout.addWidget(display_group)

        layout.addStretch()
        scroll.setWidget(inner)
        tab_layout.addWidget(scroll)
        return tab

    # ── About Tab ─────────────────────────────────────────────────────────

    def _create_about_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("OmniChat RT")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Real-time voice assistant with direct PCM audio")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        self._about_backend_label.setWordWrap(True)
        self._about_backend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._about_backend_label)

        info_lines = [
            "",
            "Model: MiniCPM-o 4.5 (bf16)",
            "Audio: sounddevice (PortAudio) -- direct PCM, no HLS",
            "VAD: Silero-VAD (CPU)",
            "",
            "Shares model, tools, and settings with the Gradio UI (main.py).",
            "Only one app can use the GPU at a time.",
        ]
        try:
            import torch
            if torch.cuda.is_available():
                info_lines.insert(3, f"GPU: {torch.cuda.get_device_name(0)}")
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                info_lines.insert(4, f"VRAM: {vram:.0f} GB")
        except Exception:
            pass

        for line in info_lines:
            lbl = QLabel(line)
            lbl.setFont(QFont("Segoe UI", 10))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)

        return tab

    # ── Helper: add spin boxes ────────────────────────────────────────────

    def _add_double_spin(self, layout, label, min_val, max_val, step, value, on_change):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(on_change)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _add_int_spin(self, layout, label, min_val, max_val, step, value, on_change):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.valueChanged.connect(on_change)
        row.addWidget(spin)
        layout.addLayout(row)
        return spin

    def _update_max_frames(self, val):
        self._session["max_frames"] = val
        self._update_vram_estimate(val)

    def _update_vram_estimate(self, frames):
        try:
            import torch
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            base_gb = torch.cuda.memory_allocated() / 1024**3
        except Exception:
            self._vram_estimate_label.setText(f"{frames} frames selected")
            return
        est_gb = base_gb + frames * 0.4
        text = f"~{est_gb:.0f} GB estimated for {frames} frames ({total_gb:.0f} GB available)"
        if est_gb > total_gb * 0.8:
            self._vram_estimate_label.setStyleSheet("color: #f38ba8;")
            text += " \u2014 may exceed VRAM!"
        else:
            self._vram_estimate_label.setStyleSheet("color: #a6adc8;")
        self._vram_estimate_label.setText(text)

    def _apply_font_size(self, size):
        self._session["font_size"] = size
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            font = app.font()
            font.setPointSize(size)
            app.setFont(font)

    # ── Voice Chat Handlers ───────────────────────────────────────────────

    def _start_conversation(self):
        self._pipeline.start_conversation()
        self._conv_start_btn.setEnabled(False)
        self._conv_stop_btn.setEnabled(True)
        self._status.showMessage("Conversation started — listening...")

    def _stop_conversation(self):
        self._pipeline.stop_conversation()
        self._conv_start_btn.setEnabled(True)
        self._conv_stop_btn.setEnabled(False)
        self._send_btn.setEnabled(True)
        self._text_input.setEnabled(True)
        if self._pipeline.is_generating and self._chat_history and self._chat_history[-1][0] == "_partial":
            interrupted_text = self._chat_history[-1][1] + " [interrupted]"
            self._chat_history[-1] = ("assistant", interrupted_text)
            self._complete_turn_recording(response_text=interrupted_text, error="interrupted by user", interrupted=True)
            self._update_history()
        self._response_display_mode = "text"
        self._set_response_phase("idle")
        self._status.showMessage("Conversation stopped")

    def _on_mode_changed(self, index):
        from tools.audio.conversation import ConversationMode
        modes = [ConversationMode.AUTO_DETECT, ConversationMode.PUSH_TO_TALK, ConversationMode.CLICK_PER_TURN]
        if index < len(modes):
            self._pipeline.conv_mgr.set_mode(modes[index])
            self._ptt_btn.setVisible(index == 1)  # show PTT button only in push-to-talk mode

    def _on_record_toggled(self, checked: bool):
        self._recording_enabled = bool(checked)
        if self._recording_enabled:
            self._ensure_recording_session()
        else:
            if self._session_recorder is not None:
                self._session_recorder.set_recording_config(enabled=False)
            self._finalize_recording_session()
        self._sync_record_button()
        self._status.showMessage(self._recording_status_summary())

    def _begin_turn_recording(self, *, request_id: str, modality: str, prompt_text: str, user_audio: np.ndarray | None = None):
        if not self._recording_enabled:
            return
        self._ensure_recording_session()
        self._active_turn_started_at = time.perf_counter()
        self._active_turn_first_text_s = None
        self._active_turn_modality = modality
        self._active_turn_prompt_text = prompt_text
        self._active_turn_user_audio = None if user_audio is None else np.asarray(user_audio, dtype=np.float32).reshape(-1).copy()
        self._active_turn_model_audio_chunks = []
        prompt_offset_s = self._current_recording_elapsed_s()
        self._active_turn_user_audio_offset_s = prompt_offset_s
        self._last_generation_progress = {}
        if self._session_recorder is None:
            return
        self._session_recorder.start_turn(
            request_id=request_id,
            turn_metadata={
                "modality": modality,
                "recording_enabled": self._recording_enabled,
                "recording_mode": self._recording_mode,
                "transcript_policy": self._transcript_policy,
                "prompt_text": prompt_text,
                "prompt_offset_s": prompt_offset_s,
                "prompt_chars": len(prompt_text or ""),
                "prompt_tokens_est": _estimate_token_count(prompt_text or ""),
                "model": self._current_model_recording_metadata(),
                "voice": self._current_voice_recording_metadata(),
            },
            user_audio=self._active_turn_user_audio,
            user_sample_rate=16000,
        )
        if self._session_video_recorder is not None and self._active_turn_user_audio is not None:
            self._session_video_recorder.add_audio_clip(
                self._active_turn_user_audio,
                sample_rate=16000,
                offset_s=self._active_turn_user_audio_offset_s,
            )

    def _complete_turn_recording(self, *, response_text: str, error: str | None = None, interrupted: bool = False):
        request_id = self._active_request_id
        if not request_id or self._session_recorder is None:
            return
        response_mode = "audio" if self._active_response_is_spoken else "text"
        progress = dict(self._last_generation_progress or {})
        elapsed_s = progress.get("elapsed_s")
        if elapsed_s is None and self._active_turn_started_at:
            elapsed_s = time.perf_counter() - self._active_turn_started_at
        if self._recording_keeps_structured_audio():
            for chunk in self._active_turn_model_audio_chunks:
                self._session_recorder.append_model_audio(request_id=request_id, audio_chunk=chunk, sample_rate=24000)
        if self._session_video_recorder is not None and self._active_turn_model_audio_chunks:
            self._session_video_recorder.add_audio_clip(
                np.concatenate(self._active_turn_model_audio_chunks).astype(np.float32, copy=False),
                sample_rate=24000,
            )
        self._session_recorder.complete_turn(
            request_id=request_id,
            response_text=response_text,
            response_mode=response_mode,
            interrupted=interrupted,
            error=error,
            first_text_s=self._active_turn_first_text_s,
            elapsed_s=elapsed_s,
            text_chars=progress.get("text_chars"),
            text_tokens_est=progress.get("text_tokens_est"),
            audio_chunks=progress.get("audio_chunks"),
        )
        self._active_turn_started_at = 0.0
        self._active_turn_first_text_s = None
        self._active_turn_prompt_text = ""
        self._active_turn_user_audio = None
        self._active_turn_model_audio_chunks = []
        self._active_turn_user_audio_offset_s = 0.0
        self._last_generation_progress = {}

    def _on_ptt_pressed(self):
        self._pipeline.conv_mgr.ptt_start()
        self._ptt_btn.setStyleSheet("background-color: #ef4444; color: white;")

    def _on_ptt_released(self):
        audio = self._pipeline.conv_mgr.ptt_stop()
        self._ptt_btn.setStyleSheet("")
        if audio is not None and len(audio) > 0:
            self._on_speech_ready(audio)

    def _send_text(self):
        text = self._text_input.text().strip()
        if not text:
            return
        self._text_input.clear()
        self._response_display_mode = "text"

        # Check for voice command first
        cmd = detect_voice_command(text)
        if cmd:
            self._apply_voice_command(cmd, text)
            return

        self._chat_history.append(("user", text))
        self._update_history()

        # Typed requests stay text-only; spoken output is reserved for audio-origin turns.
        from tools.audio.conversation import ConversationState
        conversation_state = getattr(self._pipeline.conv_mgr, "state", ConversationState.OFF)
        conversation_active = conversation_state != ConversationState.OFF
        generate_audio = False
        self._active_response_is_spoken = False

        voice_ref = get_truncated_voice_ref(self._voice_ref, self._session["voice_sample_length_s"]) if generate_audio else None
        msgs = [{"role": "user", "content": [text]}]
        trace_context = self._new_trace_context()
        self._begin_turn_recording(request_id=trace_context["request_id"], modality="text", prompt_text=text)
        logger.info(
            "trace_id=%s stage=ui event=send_text profile=%s backend=%s speech_backend=%s history_entries=%d assistant_turns=%d user_chars=%d conversation_active=%s generate_audio=%s preview=%r",
            trace_context["request_id"],
            self._current_model_profile(),
            self._current_backend_name(),
            self._current_speech_backend(),
            len(self._chat_history),
            sum(1 for role, _ in self._chat_history if role == "assistant"),
            len(text),
            bool(conversation_active),
            bool(generate_audio),
            text[:160],
        )
        self._pipeline.process_turn(msgs, voice_ref, dict(self._session), generate_audio=generate_audio, trace_context=trace_context)

    def _on_speech_ready(self, audio: np.ndarray):
        """VAD detected end of speech — process the turn."""
        self._chat_history.append(("user", "[voice input]"))
        self._update_history()
        self._active_response_is_spoken = True
        self._response_display_mode = "audio"

        voice_ref = get_truncated_voice_ref(self._voice_ref, self._session["voice_sample_length_s"])
        msgs = [{"role": "user", "content": [audio]}]
        settings = dict(self._session)
        if settings.get("speech_repetition_penalty") is not None:
            settings["repetition_penalty"] = settings["speech_repetition_penalty"]
        trace_context = self._new_trace_context()
        self._begin_turn_recording(request_id=trace_context["request_id"], modality="audio", prompt_text="[voice input]", user_audio=audio)
        logger.info(
            "trace_id=%s stage=ui event=send_speech profile=%s backend=%s speech_backend=%s history_entries=%d assistant_turns=%d audio_samples=%d",
            trace_context["request_id"],
            self._current_model_profile(),
            self._current_backend_name(),
            self._current_speech_backend(),
            len(self._chat_history),
            sum(1 for role, _ in self._chat_history if role == "assistant"),
            len(audio),
        )
        self._pipeline.process_turn(msgs, voice_ref, settings, trace_context=trace_context)

    def _on_state_changed(self, state_name: str, color: str):
        self._state_dot.setStyleSheet(f"color: {color};")
        display = "Thinking..." if state_name == "PROCESSING" else state_name.replace("_", " ").title()
        self._state_label.setText(display)

    def _on_generation_started(self):
        self._send_btn.setEnabled(False)
        self._text_input.setEnabled(False)
        logger.info("trace_id=%s stage=ui event=generation_started", self._active_request_id or "n/a")
        phase = "thinking"
        if self._response_display_mode == "text":
            try:
                transport = (get_backend_capabilities() or {}).get("transport") or {}
                if transport.get("streaming_mode") == "oneshot_cli":
                    phase = "generating"
            except Exception:
                pass
        self._set_response_phase(phase)

    def _on_generation_progress(self, payload: dict):
        if not isinstance(payload, dict):
            return
        elapsed_s = float(payload.get("elapsed_s", 0.0) or 0.0)
        text_chars = int(payload.get("text_chars", 0) or 0)
        text_tokens_est = int(payload.get("text_tokens_est", 0) or 0)
        audio_chunks = int(payload.get("audio_chunks", 0) or 0)
        message = str(payload.get("message") or "Generating response")
        self._last_generation_progress = dict(payload)
        self._status.showMessage(
            f"{message} | {elapsed_s:.1f}s | {text_chars} chars | ~{text_tokens_est} tok | audio chunks {audio_chunks}"
        )

    def _on_text_update(self, text_chunk: str):
        # Update history with partial text
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            prev = self._chat_history[-1][1]
            self._chat_history[-1] = ("_partial", prev + text_chunk)
        else:
            self._chat_history.append(("_partial", text_chunk))
        self._update_history()
        partial_text = self._chat_history[-1][1] if self._chat_history and self._chat_history[-1][0] == "_partial" else text_chunk
        if self._active_turn_started_at and self._active_turn_first_text_s is None:
            self._active_turn_first_text_s = time.perf_counter() - self._active_turn_started_at
        if self._response_display_mode == "text":
            self._set_response_phase("text")
        elif self._response_display_mode == "audio" and self._current_backend_name() in {"qwen_llamacpp", "gemma_llamacpp"} and self._current_speech_backend() == "minicpm_streaming":
            self._set_response_phase("tts")
        logger.info(
            "trace_id=%s stage=ui event=text_update chunk_chars=%d total_chars=%d total_tokens_est=%d",
            self._active_request_id or "n/a",
            len(text_chunk),
            len(partial_text),
            _estimate_token_count(partial_text),
        )

    def _on_audio_chunk_ready(self, audio_chunk: np.ndarray):
        if audio_chunk is None:
            return
        self._active_turn_model_audio_chunks.append(np.asarray(audio_chunk, dtype=np.float32).reshape(-1).copy())
        if self._response_display_mode == "audio":
            self._set_response_phase("audio")

    def _on_generation_finished(self, full_text: str):
        self._complete_turn_recording(response_text=full_text)
        # Replace partial with final
        final_role = "_assistant_hidden" if self._active_response_is_spoken else "assistant"
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            self._chat_history[-1] = (final_role, full_text)
        else:
            self._chat_history.append((final_role, full_text))
        self._update_history()
        self._send_btn.setEnabled(True)
        self._text_input.setEnabled(True)

        logger.info(
            "trace_id=%s stage=ui event=generation_finished response_chars=%d response_tokens_est=%d preview=%r",
            self._active_request_id or "n/a",
            len(full_text or ""),
            _estimate_token_count(full_text or ""),
            (full_text or "")[:160],
        )

        voice_label = self._voice_name or "Default"
        self._status.showMessage(f"Voice: {voice_label} | Ready")
        self._set_response_phase("idle")
        self._active_request_id = None
        self._active_response_is_spoken = False

    def _on_generation_error(self, error_message: str):
        partial_text = ""
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            partial_text = self._chat_history[-1][1]
        self._complete_turn_recording(response_text=partial_text, error=error_message)
        logger.error(
            "trace_id=%s stage=ui event=generation_error error=%r",
            self._active_request_id or "n/a",
            error_message,
        )
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            self._chat_history[-1] = ("system", f"Generation error: {error_message}")
        else:
            self._chat_history.append(("system", f"Generation error: {error_message}"))
        self._update_history()
        self._send_btn.setEnabled(True)
        self._text_input.setEnabled(True)
        self._set_response_phase("idle")
        self._status.showMessage(f"Generation error: {error_message} | Trace: {get_trace_log_path()}")
        self._active_request_id = None
        self._active_response_is_spoken = False

    def _on_barge_in(self):
        logger.info("trace_id=%s stage=ui event=barge_in", self._active_request_id or "n/a")
        self._status.showMessage("Interrupted by user")
        interrupted_text = ""
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            text = self._chat_history[-1][1]
            self._chat_history[-1] = ("assistant", text + " [interrupted]")
            interrupted_text = text + " [interrupted]"
            self._update_history()
        self._complete_turn_recording(response_text=interrupted_text, error="interrupted by user", interrupted=True)
        self._set_response_phase("idle")

    def _apply_remote_settings(self):
        set_qwen_remote_config(
            base_url=self._remote_base_url.text().strip(),
            api_key=self._remote_api_key.text().strip() or None,
            model_name=self._remote_model_name.text().strip(),
            timeout_s=self._remote_timeout.value(),
            endpoints={
                "health": self._remote_health_ep.text().strip(),
                "models": self._remote_models_ep.text().strip(),
                "chat_completions": self._remote_chat_ep.text().strip(),
                "responses": self._remote_responses_ep.text().strip(),
                "realtime": self._remote_realtime_ep.text().strip(),
            },
        )
        self._refresh_backend_status(show_message=False)
        self._remote_probe_label.setText(f"Base URL: {self._remote_base_url.text().strip()}")

    def _test_remote_health(self):
        status = self._refresh_backend_status(show_message=False)
        self._remote_probe_label.setText(mm.format_backend_status(status))

    def _test_remote_realtime(self):
        cfg = get_qwen_remote_config()
        result = QwenOmniClient(QwenOmniClientConfig(**cfg)).perform_realtime_handshake()
        self._refresh_backend_status(show_message=False)
        if result.get("ok"):
            self._remote_probe_label.setText(
                f"Realtime handshake OK | {result.get('url')} | {result.get('event_type')}"
            )
        else:
            self._remote_probe_label.setText(f"Realtime handshake failed | {result.get('error')}")

    def _update_history(self):
        """Render chat history to the display widget."""
        self._history_display.setHtml(
            render_chat_history_html(
                self._chat_history,
                limit=20,
                assistant_label=self._assistant_label(),
            )
        )
        # Scroll to bottom
        sb = self._history_display.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _apply_voice_command(self, cmd: str, source_text: str):
        if cmd == "default":
            self._voice_ref = None
            self._voice_name = None
            self._chat_history.append(("system", "Switched to default voice"))
        else:
            result = get_voice(cmd, self._fuzzy_threshold)
            if result["found"]:
                self._voice_ref = result["audio"]
                self._voice_name = result["name"]
                self._chat_history.append(("system", f"Switched to {result['name']}'s voice"))
            else:
                self._chat_history.append(("system", f"Voice '{cmd}' not found"))
        self._update_history()
        self._refresh_voice_list()

    def _refresh_voice_list(self):
        self._voice_combo.blockSignals(True)
        self._voice_combo.clear()
        self._voice_combo.addItem("Default")
        self._voice_combo.addItems(list_voices())
        if self._voice_name:
            idx = self._voice_combo.findText(self._voice_name, Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self._voice_combo.setCurrentIndex(idx)
        self._voice_combo.blockSignals(False)

    def _refresh_playback_sessions(self):
        current_path = self._playback_session.manifest_path if self._playback_session is not None else None
        manifests = session_playback.discover_session_manifests(self._playback_root)
        self._playback_session_combo.blockSignals(True)
        self._playback_session_combo.clear()
        for manifest_path in manifests:
            label = str(manifest_path.relative_to(self._playback_root)) if self._playback_root in manifest_path.parents else manifest_path.name
            self._playback_session_combo.addItem(label, userData=str(manifest_path))
        self._playback_session_combo.blockSignals(False)
        if manifests:
            selected_index = 0
            if current_path is not None:
                for index in range(self._playback_session_combo.count()):
                    if self._playback_session_combo.itemData(index) == str(current_path):
                        selected_index = index
                        break
            self._playback_session_combo.setCurrentIndex(selected_index)
            self._load_selected_playback_session(selected_index)
        else:
            self._playback_session = None
            self._playback_events = []
            self._playback_event_index = -1
            self._playback_history_display.clear()
            self._playback_summary_label.setText(f"No session manifests found under {self._playback_root}")
            self._playback_progress.setMaximum(1)
            self._playback_progress.setValue(0)

    def _browse_playback_session(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Session Manifest", str(self._playback_root), "JSON (*.json)")
        if not path:
            return
        self._set_playback_session(session_playback.load_session_manifest(path))

    def _open_playback_folder(self):
        target = self._playback_session.session_dir if self._playback_session is not None else self._playback_root
        target.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(target))
        except Exception as exc:
            self._status.showMessage(f"Could not open folder: {exc}")

    def _open_selected_playback_video(self):
        if self._playback_session is None or self._playback_session.video_path is None:
            self._status.showMessage("Selected session does not have a recorded video")
            return
        video_path = self._playback_session.video_path
        if not video_path.exists():
            self._status.showMessage(f"Recorded video missing: {video_path}")
            return
        QDesktopServices.openUrl(video_path.as_uri())
        self._status.showMessage(f"Opened recorded video: {video_path}")

    def _format_overlay_snapshot(self, snapshot: session_playback.SessionOverlaySnapshot | None) -> str:
        if snapshot is None:
            return "No synced stats available for this session yet."
        lines = [
            f"Turn {snapshot.turn_index} | {snapshot.phase.title()} | {snapshot.modality}/{snapshot.response_mode}",
            f"Prompt tok: {snapshot.prompt_tokens_est} | Response tok: {snapshot.response_tokens_est}",
            f"First text: {snapshot.first_text_s if snapshot.first_text_s is not None else 0.0:.2f}s | Total: {snapshot.elapsed_s if snapshot.elapsed_s is not None else 0.0:.2f}s",
            f"Prompt @ {snapshot.prompt_offset_s if snapshot.prompt_offset_s is not None else 0.0:.2f}s | Complete @ {snapshot.completed_offset_s if snapshot.completed_offset_s is not None else 0.0:.2f}s",
        ]
        if snapshot.interrupted:
            lines.append("Interrupted")
        if snapshot.error:
            lines.append(f"Error: {snapshot.error}")
        return "\n".join(lines)

    def _update_embedded_video_overlay(self, position_ms: int = 0):
        if not hasattr(self, "_playback_overlay_label") or self._playback_session is None:
            return
        snapshot = session_playback.get_overlay_snapshot(self._playback_session, max(0.0, float(position_ms) / 1000.0))
        self._playback_overlay_label.setText(self._format_overlay_snapshot(snapshot))

    def _on_playback_video_position_changed(self, position_ms: int):
        self._update_embedded_video_overlay(position_ms)
        if self._playback_media_status is not None:
            self._playback_media_status.setText(f"Playback position: {float(position_ms) / 1000.0:.2f}s")

    def _play_embedded_video(self):
        if self._playback_session is None or self._playback_session.video_path is None:
            self._status.showMessage("Selected session does not have a recorded video")
            return False
        if not MULTIMEDIA_AVAILABLE or self._playback_media_player is None:
            return False
        video_path = self._playback_session.video_path
        if not video_path.exists():
            self._status.showMessage(f"Recorded video missing: {video_path}")
            return False
        self._playback_media_player.setSource(QUrl.fromLocalFile(str(video_path.resolve())))
        self._playback_media_player.play()
        self._update_embedded_video_overlay(0)
        self._playback_media_status.setText(f"Playing embedded video: {video_path.name}")
        self._status.showMessage(f"Playing embedded video: {video_path}")
        return True

    def _load_selected_playback_session(self, _index: int):
        manifest_path = self._playback_session_combo.currentData()
        if not manifest_path:
            return
        self._set_playback_session(session_playback.load_session_manifest(manifest_path))

    def _set_playback_session(self, session: session_playback.SessionPlaybackManifest):
        self._stop_playback()
        self._playback_session = session
        self._playback_events = session_playback.build_replay_events(session)
        self._playback_event_index = len(self._playback_events) - 1
        self._playback_audio_cache.clear()
        total_messages = len(session_playback.build_chat_history(session))
        self._playback_summary_label.setText(
            f"{session.assistant_label} | {session.data.get('created_at') or ''} | {len(session.turns)} turns | {total_messages} messages | {session.recording_mode}"
        )
        self._playback_progress.setMaximum(max(1, len(self._playback_events)))
        self._playback_progress.setValue(max(0, len(self._playback_events)))
        self._render_playback_history(session_playback.build_chat_history(session), assistant_label=session.assistant_label)
        self._playback_open_video_btn.setEnabled(bool(session.video_path))
        self._update_embedded_video_overlay(0)
        if self._playback_media_status is not None:
            self._playback_media_status.setText("Embedded video ready" if session.video_path is not None else "No video artifact in this session")
        combo_index = self._playback_session_combo.findData(str(session.manifest_path))
        if combo_index >= 0 and self._playback_session_combo.currentIndex() != combo_index:
            self._playback_session_combo.blockSignals(True)
            self._playback_session_combo.setCurrentIndex(combo_index)
            self._playback_session_combo.blockSignals(False)

    def _render_playback_history(self, history: list[tuple[str, str]], *, assistant_label: str | None = None):
        label = assistant_label or (self._playback_session.assistant_label if self._playback_session is not None else self._assistant_label())
        self._playback_history_display.setHtml(
            render_chat_history_html(history, limit=max(20, len(history) + 5), assistant_label=label)
        )
        scroll_bar = self._playback_history_display.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _play_selected_session(self):
        if self._playback_session is None:
            self._status.showMessage("No session selected for playback")
            return
        if self._playback_session.video_path is not None:
            if self._play_embedded_video():
                return
            self._open_selected_playback_video()
            return
        self._stop_playback()
        self._playback_is_running = True
        self._playback_event_index = -1
        self._playback_progress.setValue(0)
        self._render_playback_history([], assistant_label=self._playback_session.assistant_label)
        self._advance_playback()

    def _advance_playback(self):
        if not self._playback_is_running or self._playback_session is None:
            return
        next_index = self._playback_event_index + 1
        if next_index >= len(self._playback_events):
            self._playback_is_running = False
            self._status.showMessage(f"Playback finished | {self._playback_session.manifest_path.name}")
            return
        self._playback_event_index = next_index
        self._show_playback_event(next_index)
        event = self._playback_events[next_index]
        self._playback_play_audio(event)
        self._playback_timer.start(max(50, int(event.duration_s * 1000)))

    def _show_playback_event(self, index: int):
        if self._playback_session is None or not (0 <= index < len(self._playback_events)):
            return
        event = self._playback_events[index]
        self._render_playback_history(event.history, assistant_label=self._playback_session.assistant_label)
        self._playback_progress.setValue(index + 1)
        self._playback_summary_label.setText(
            f"{self._playback_session.assistant_label} | {event.description} | {event.duration_s:.2f}s"
        )

    def _show_previous_playback_event(self):
        if not self._playback_events:
            return
        self._stop_playback()
        target_index = max(0, self._playback_event_index - 1 if self._playback_event_index >= 0 else len(self._playback_events) - 1)
        self._playback_event_index = target_index
        self._show_playback_event(target_index)

    def _show_next_playback_event(self):
        if not self._playback_events:
            return
        self._stop_playback()
        target_index = min(len(self._playback_events) - 1, self._playback_event_index + 1)
        self._playback_event_index = target_index
        self._show_playback_event(target_index)

    def _playback_play_audio(self, event: session_playback.SessionReplayEvent):
        if event.audio_path is None or not event.audio_path.exists():
            return
        try:
            import sounddevice as sd

            cached = self._playback_audio_cache.get(event.audio_path)
            if cached is None:
                cached = session_playback.load_audio_file(event.audio_path)
                self._playback_audio_cache[event.audio_path] = cached
            audio, sample_rate = cached
            sd.play(audio, samplerate=sample_rate)
        except Exception as exc:
            self._status.showMessage(f"Playback audio failed: {exc}")

    def _stop_playback(self):
        self._playback_is_running = False
        self._playback_timer.stop()
        if self._playback_media_player is not None:
            self._playback_media_player.stop()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def _export_selected_session_audio(self):
        if self._playback_session is None:
            self._status.showMessage("No session selected for export")
            return
        default_path = self._playback_session.session_dir / f"{self._playback_session.manifest_path.stem}_demo.mp3"
        path, _ = QFileDialog.getSaveFileName(self, "Export Stitched Audio", str(default_path), "MP3 (*.mp3);;WAV (*.wav)")
        if not path:
            return
        session_playback.export_stitched_audio(self._playback_session, path)
        self._status.showMessage(f"Exported audio: {path}")

    def _export_selected_session_video(self):
        if self._playback_session is None:
            self._status.showMessage("No session selected for export")
            return
        if self._playback_session.video_path is not None and self._playback_session.video_path.exists():
            path, _ = QFileDialog.getSaveFileName(self, "Export Demo Video", str(self._playback_session.video_path), "MP4 (*.mp4)")
            if not path:
                return
            Path(path).write_bytes(self._playback_session.video_path.read_bytes())
            self._status.showMessage(f"Exported video: {path}")
            return
        default_path = self._playback_session.session_dir / f"{self._playback_session.manifest_path.stem}_demo.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Export Demo Video", str(default_path), "MP4 (*.mp4)")
        if not path:
            return
        session_playback.export_demo_video(self._playback_session, path)
        self._status.showMessage(f"Exported video: {path}")

    def _on_voice_changed(self, name: str):
        if name == "Default" or not name:
            self._voice_ref = None
            self._voice_name = None
            self._status.showMessage("Voice: Default")
        else:
            result = get_voice(name, self._fuzzy_threshold)
            if result["found"]:
                self._voice_ref = result["audio"]
                self._voice_name = result["name"]
                self._status.showMessage(f"Voice: {result['name']}")
            else:
                self._status.showMessage(f"Voice '{name}' not found")

    # ── Vision Handlers ───────────────────────────────────────────────────

    def _upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        if path:
            resolved = Path(path).resolve()
            # Read bytes first — triggers OneDrive hydration for cloud-only files
            try:
                img_bytes = resolved.read_bytes()
            except OSError as e:
                self._img_preview.setText("File not available (cloud-only?)")
                self._status.showMessage(f"Cannot read file: {e}")
                self._img_path = None
                return
            self._img_path = str(resolved)
            pixmap = QPixmap()
            pixmap.loadFromData(img_bytes)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    self._img_preview.width(), self._img_preview.height(),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
                )
                self._img_preview.setPixmap(pixmap)
            else:
                self._img_preview.setText("Preview unavailable")

    def _upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video or Audio", "",
            "Media (*.mp4 *.avi *.mov *.mkv *.webm *.wav *.mp3 *.ogg *.flac *.m4a *.wma *.aac)"
        )
        if path:
            self._vid_path = path
            # Show filename + brief info in label; full details in tooltip
            name = Path(path).name
            tip_lines = [path]
            extra = ""
            try:
                from decord import VideoReader, cpu as decord_cpu
                vr = VideoReader(path, ctx=decord_cpu(0))
                fps = vr.get_avg_fps()
                n_frames = len(vr)
                dur = n_frames / fps if fps else 0
                w, h = vr[0].shape[1], vr[0].shape[0]
                mins, secs = divmod(int(dur), 60)
                hrs, mins = divmod(mins, 60)
                dur_str = f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins}:{secs:02d}"
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                extra = f" ({dur_str}, {w}x{h})"
                tip_lines.append(f"Duration: {dur_str}  ({n_frames} frames)")
                tip_lines.append(f"Resolution: {w}x{h}  |  FPS: {fps:.1f}")
                tip_lines.append(f"Size: {size_mb:.1f} MB")
            except Exception:
                # Audio-only files won't have video frames — show file size
                try:
                    size_mb = Path(path).stat().st_size / (1024 * 1024)
                    extra = f" ({size_mb:.1f} MB)"
                    tip_lines.append(f"Size: {size_mb:.1f} MB")
                except Exception:
                    pass
            self._vid_label.setText(name + extra)
            self._vid_label.setToolTip("\n".join(tip_lines))

    def _analyze_image(self):
        if not self._img_path:
            self._vision_output.setPlainText("No image selected.")
            return
        self._vision_output.setPlainText("Analyzing image...")
        self._status.showMessage("Analyzing image...")

        # Run on a thread to avoid blocking UI
        from PySide6.QtCore import QThread as _QT
        from tools.vision.process_media import scan_image, scan_document

        class _Worker(_QT):
            result_ready = Signal(str)
            def __init__(self, img_path, prompt, doc_mode, settings):
                super().__init__()
                self._img_path = img_path
                self._prompt = prompt
                self._doc_mode = doc_mode
                self._settings = settings
            def run(self):
                try:
                    from PIL import Image
                    from io import BytesIO
                    img_bytes = Path(self._img_path).read_bytes()
                    image = Image.open(BytesIO(img_bytes)).convert("RGB")
                    s = self._settings
                    if self._doc_mode:
                        prompt = self._prompt or "Extract all text from this document."
                        result = scan_document(image, prompt=prompt,
                                               temperature=s["temperature"],
                                               max_new_tokens=s["max_new_tokens"],
                                               top_p=s.get("top_p", 0.8),
                                               top_k=s.get("top_k", 100),
                                               enable_thinking=s.get("enable_thinking", False))
                    else:
                        prompt = self._prompt or "Describe this image in detail."
                        result = scan_image(image, prompt=prompt,
                                            temperature=s["temperature"],
                                            max_new_tokens=s["max_new_tokens"],
                                            top_p=s.get("top_p", 0.8),
                                            top_k=s.get("top_k", 100),
                                            enable_thinking=s.get("enable_thinking", False))
                    self.result_ready.emit(result["text"])
                except Exception as e:
                    self.result_ready.emit(f"Error: {e}\nPath: {self._img_path}")

        worker = _Worker(self._img_path, self._img_prompt.text().strip(),
                         self._doc_mode_check.isChecked(), self._session)
        worker.result_ready.connect(lambda t: (
            self._vision_output.setPlainText(t),
            self._status.showMessage("Image analysis complete"),
        ))
        worker.start()
        self._img_worker = worker  # prevent GC

    def _analyze_video(self):
        if not self._vid_path:
            self._vision_output.setPlainText("No video selected.")
            return
        self._vision_output.setPlainText("Analyzing video...")
        self._status.showMessage("Analyzing video...")

        from PySide6.QtCore import QThread as _QT
        from tools.vision.process_media import analyze_video

        class _Worker(_QT):
            result_ready = Signal(str)
            def __init__(self, vid_path, prompt, settings):
                super().__init__()
                self._vid_path = vid_path
                self._prompt = prompt
                self._settings = settings
            def run(self):
                prompt = self._prompt or "Describe what's happening in this video."
                s = self._settings
                result = analyze_video(self._vid_path, prompt=prompt,
                                       temperature=s["temperature"],
                                       max_new_tokens=s["max_new_tokens"],
                                       top_p=s.get("top_p", 0.8),
                                       top_k=s.get("top_k", 100),
                                       enable_thinking=s.get("enable_thinking", False),
                                       max_frames=s.get("max_frames", 64))
                self.result_ready.emit(result["text"])

        worker = _Worker(self._vid_path, self._vid_prompt.text().strip(), self._session)
        worker.result_ready.connect(lambda t: (
            self._vision_output.setPlainText(t),
            self._status.showMessage("Video analysis complete"),
        ))
        worker.start()
        self._vid_worker = worker

    def _transcribe_video(self):
        if not self._vid_path:
            self._vision_output.setPlainText("No video selected.")
            return
        self._vision_output.setPlainText("Transcribing audio...")
        self._status.showMessage("Transcribing audio...")

        from PySide6.QtCore import QThread as _QT
        from tools.vision.process_media import transcribe_video

        class _Worker(_QT):
            chunk_update = Signal(str, str)  # (accumulated_text, status_msg)
            result_ready = Signal(str)
            def __init__(self, vid_path, prompt, settings):
                super().__init__()
                self._vid_path = vid_path
                self._prompt = prompt
                self._settings = settings
            def run(self):
                prompt = self._prompt or "Transcribe this audio completely and verbatim."
                s = self._settings

                def _on_chunk(idx, total, accumulated):
                    self.chunk_update.emit(
                        accumulated,
                        f"Transcribing chunk {idx + 1}/{total}...",
                    )

                result = transcribe_video(self._vid_path, prompt=prompt,
                                          temperature=s["temperature"],
                                          max_new_tokens=s["max_new_tokens"],
                                          top_p=s.get("top_p", 0.8),
                                          top_k=s.get("top_k", 100),
                                          enable_thinking=s.get("enable_thinking", False),
                                          on_chunk=_on_chunk)
                self.result_ready.emit(result["text"])

        worker = _Worker(self._vid_path, self._vid_prompt.text().strip(), self._session)
        worker.chunk_update.connect(lambda text, status: (
            self._vision_output.setPlainText(text),
            self._status.showMessage(status),
        ))
        worker.result_ready.connect(lambda t: (
            self._vision_output.setPlainText(t),
            self._status.showMessage("Audio transcription complete"),
        ))
        worker.start()
        self._transcribe_worker = worker

    def _upload_pdf(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF(s)", "", "PDF Files (*.pdf)"
        )
        if paths:
            self._pdf_paths = paths
            from tools.vision.pdf_processor import get_page_count
            total_pages = sum(get_page_count(p) for p in paths)
            if len(paths) == 1:
                self._pdf_label.setText(
                    f"{Path(paths[0]).name} ({total_pages} pages)"
                )
            else:
                self._pdf_label.setText(
                    f"{len(paths)} files ({total_pages} pages total)"
                )

    def _scan_pdf(self):
        if not self._pdf_paths:
            self._vision_output.setPlainText("No PDF selected.")
            return
        self._vision_output.setPlainText("Scanning PDF...")
        self._status.showMessage("Scanning PDF...")
        self._pdf_progress.setVisible(True)
        self._pdf_progress.setValue(0)
        self._scan_pdf_btn.setEnabled(False)

        from PySide6.QtCore import QThread as _QT

        class _PDFWorker(_QT):
            result_ready = Signal(str, object)
            page_progress = Signal(int, int)

            def __init__(self, pdf_paths, prompt, bank_mode, settings):
                super().__init__()
                self._pdf_paths = pdf_paths
                self._prompt = prompt
                self._bank_mode = bank_mode
                self._settings = settings

            def run(self):
                from tools.vision.pdf_processor import (
                    scan_pdf, scan_bank_statement, scan_multiple_pdfs,
                )

                def on_page(page, total):
                    self.page_progress.emit(page, total)

                if len(self._pdf_paths) == 1:
                    if self._bank_mode:
                        result = scan_bank_statement(
                            self._pdf_paths[0], on_page_start=on_page,
                        )
                    else:
                        result = scan_pdf(
                            self._pdf_paths[0],
                            prompt=self._prompt or None,
                            temperature=self._settings["temperature"],
                            max_new_tokens=self._settings["max_new_tokens"],
                            on_page_start=on_page,
                        )
                    self.result_ready.emit(
                        result["combined_text"], result["combined_table"]
                    )
                else:
                    result = scan_multiple_pdfs(
                        self._pdf_paths,
                        prompt=self._prompt or None,
                        bank_mode=self._bank_mode,
                        temperature=self._settings["temperature"],
                        max_new_tokens=self._settings["max_new_tokens"],
                        on_page_start=on_page,
                    )
                    self.result_ready.emit(
                        result["combined_text"], result["combined_table"]
                    )

        worker = _PDFWorker(
            self._pdf_paths,
            self._pdf_prompt.text().strip(),
            self._bank_mode_check.isChecked(),
            self._session,
        )
        worker.page_progress.connect(
            lambda cur, total: self._pdf_progress.setValue(
                int((cur + 1) / total * 100)
            )
        )
        worker.result_ready.connect(self._on_pdf_done)
        worker.start()
        self._pdf_worker = worker

    def _on_pdf_done(self, text, table):
        from tools.vision.pdf_processor import merge_tables

        self._vision_output.setPlainText(text)

        if self._accumulate_check.isChecked() and self._pdf_table:
            self._pdf_table = merge_tables(self._pdf_table, table)
        else:
            self._pdf_table = table

        self._pdf_progress.setVisible(False)
        self._scan_pdf_btn.setEnabled(True)
        rows = (len(self._pdf_table) - 1) if self._pdf_table else 0
        suffix = " (accumulated)" if self._accumulate_check.isChecked() else ""
        self._status.showMessage(
            f"PDF scan complete | {rows} data rows{suffix}"
        )

    def _clear_pdf_results(self):
        self._pdf_table = None
        self._pdf_paths = []
        self._pdf_label.setText("No PDF selected")
        self._vision_output.clear()
        self._status.showMessage("PDF results cleared")

    def _save_vision_output(self):
        text = self._vision_output.toPlainText()
        has_table = bool(self._pdf_table)
        if not text.strip() and not has_table:
            self._status.showMessage("Nothing to save")
            return

        # Build format filters — table formats only when table data exists
        filters = []
        if has_table:
            filters.append("Excel Workbook (*.xlsx)")
            filters.append("CSV - Comma separated (*.csv)")
            filters.append("TSV - Tab separated (*.tsv)")
        filters.append("Text file (*.txt)")
        filters.append("Markdown (*.md)")

        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Output", "", ";;".join(filters),
        )
        if not file_path:
            return

        from tools.output.save_output import (
            save_as_excel, save_as_csv, save_as_text, save_as_markdown,
        )

        # Check if file exists — offer to append
        append = False
        if Path(file_path).exists():
            reply = QMessageBox.question(
                self,
                "File exists",
                f"{Path(file_path).name} already exists.\n\n"
                "Append to the existing file?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                return
            append = reply == QMessageBox.Yes

        if selected_filter.startswith("Excel") and has_table:
            save_as_excel(self._pdf_table, path=file_path, append=append)
        elif selected_filter.startswith("CSV") and has_table:
            save_as_csv(self._pdf_table, path=file_path, delimiter=",", append=append)
        elif selected_filter.startswith("TSV") and has_table:
            save_as_csv(self._pdf_table, path=file_path, delimiter="\t", append=append)
        elif selected_filter.startswith("Markdown"):
            save_as_markdown(text, path=file_path, append=append)
        else:
            save_as_text(text, path=file_path, append=append)

        action = "Appended to" if append else "Saved to"
        self._status.showMessage(f"{action}: {file_path}")

    # ── Voice Management ──────────────────────────────────────────────────

    def _add_voice_from_file(self):
        name = self._voice_add_name.text().strip()
        path, _ = QFileDialog.getOpenFileName(self, "Select Voice WAV", "", "Audio (*.wav *.mp3 *.flac)")
        if not path:
            return
        if not name:
            name = _default_voice_name_from_path(path)
            self._voice_add_name.setText(name)
        audio = load_voice_sample_from_media(path)
        add_voice(name, audio, sample_rate=16000)
        self._refresh_voice_list()
        self._voice_del_combo.clear()
        self._voice_del_combo.addItems(list_voices())
        self._voice_combo.blockSignals(True)
        idx = self._voice_combo.findText(name, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._voice_combo.setCurrentIndex(idx)
        self._voice_combo.blockSignals(False)
        self._voice_del_combo.blockSignals(True)
        idx = self._voice_del_combo.findText(name, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._voice_del_combo.setCurrentIndex(idx)
        self._voice_del_combo.blockSignals(False)
        self._status.showMessage(f"Voice '{name}' imported from {Path(path).name}")

    def _record_voice_from_mic(self):
        name = self._voice_add_name.text().strip()
        if not name:
            self._status.showMessage("Enter a voice name first")
            return
        audio = record_voice_sample(duration_s=self._session["voice_sample_length_s"], sample_rate=16000)
        add_voice(name, audio, sample_rate=16000)
        self._refresh_voice_list()
        self._voice_del_combo.clear()
        self._voice_del_combo.addItems(list_voices())
        self._voice_combo.blockSignals(True)
        idx = self._voice_combo.findText(name, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._voice_combo.setCurrentIndex(idx)
        self._voice_combo.blockSignals(False)
        self._voice_del_combo.blockSignals(True)
        idx = self._voice_del_combo.findText(name, Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self._voice_del_combo.setCurrentIndex(idx)
        self._voice_del_combo.blockSignals(False)
        self._status.showMessage(f"Voice '{name}' recorded from microphone")

    def _delete_voice(self):
        name = self._voice_del_combo.currentText()
        if not name:
            return
        delete_voice(name)
        self._status.showMessage(f"Voice '{name}' deleted")
        self._refresh_voice_list()
        self._voice_del_combo.clear()
        self._voice_del_combo.addItems(list_voices())

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_playback()
        self._finalize_recording_session()
        self._pipeline.stop_conversation()
        # Wait for inference thread to finish so CUDA cleans up safely
        if self._pipeline._inference_thread is not None:
            self._pipeline._inference_thread.wait(5000)
        super().closeEvent(event)
