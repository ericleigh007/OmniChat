"""
rt_app.py -- PySide6 QMainWindow for OmniChat real-time desktop client.

Contains the main window with 4 tabs: Voice Chat, Vision, Settings, About.
All audio I/O uses direct PCM via sounddevice (no HLS/browser overhead).
"""

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit, QComboBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QGroupBox, QFileDialog,
    QSplitter, QStatusBar, QProgressBar, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QIcon, QPixmap, QColor

from rt_audio import AudioPipeline
from tools.shared.session import (
    detect_voice_command,
    get_truncated_voice_ref,
)
from tools.audio.voice_manager import list_voices, get_voice, add_voice, delete_voice

BASE_DIR = Path(__file__).parent.resolve()


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
            "output_format": output_cfg.get("default_format", "auto"),
            "voice_sample_length_s": audio_cfg.get("voice_sample_length_s", 5.0),
        }
        self._voice_ref = None
        self._voice_name = None
        self._chat_history = []
        self._fuzzy_threshold = voice_cfg.get("fuzzy_threshold", 0.6)

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
        self._pipeline.generation_started.connect(self._on_generation_started)
        self._pipeline.generation_finished.connect(self._on_generation_finished)
        self._pipeline.barge_in_detected.connect(self._on_barge_in)

        self._setup_ui()
        self._setup_statusbar()
        self.setWindowTitle("OmniChat RT")
        icon_path = BASE_DIR / "assets" / "omnichat.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(900, 700)

    # ── UI Setup ──────────────────────────────────────────────────────────

    def _setup_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self._create_voice_tab(), "Voice Chat")
        tabs.addTab(self._create_vision_tab(), "Vision")
        tabs.addTab(self._create_settings_tab(), "Settings")
        tabs.addTab(self._create_about_tab(), "About")
        self.setCentralWidget(tabs)

    def _setup_statusbar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ── Voice Chat Tab ────────────────────────────────────────────────────

    def _create_voice_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Top bar: conversation controls
        controls = QHBoxLayout()

        self._conv_start_btn = QPushButton("Start Conversation")
        self._conv_start_btn.clicked.connect(self._start_conversation)
        controls.addWidget(self._conv_start_btn)

        self._conv_stop_btn = QPushButton("Stop")
        self._conv_stop_btn.setEnabled(False)
        self._conv_stop_btn.clicked.connect(self._stop_conversation)
        controls.addWidget(self._conv_stop_btn)

        controls.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Auto-detect", "Push-to-talk", "Click per turn"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        controls.addWidget(self._mode_combo)

        # State indicator
        self._state_dot = QLabel("\u25cf")  # filled circle
        self._state_dot.setFont(QFont("Segoe UI", 16))
        self._state_dot.setStyleSheet("color: #888888;")
        controls.addWidget(self._state_dot)

        self._state_label = QLabel("OFF")
        controls.addWidget(self._state_label)

        controls.addStretch()

        # Voice selector
        controls.addWidget(QLabel("Voice:"))
        self._voice_combo = QComboBox()
        self._refresh_voice_list()
        self._voice_combo.currentTextChanged.connect(self._on_voice_changed)
        controls.addWidget(self._voice_combo)

        layout.addLayout(controls)

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
        self._img_preview.setFixedHeight(200)
        self._img_preview.setStyleSheet("background-color: #2e2e3e; border-radius: 4px;")
        img_layout.addWidget(self._img_preview)
        self._img_path = None

        layout.addWidget(img_group)

        # Video section
        vid_group = QGroupBox("Video Analysis")
        vid_layout = QHBoxLayout(vid_group)

        self._vid_upload_btn = QPushButton("Upload Video")
        self._vid_upload_btn.clicked.connect(self._upload_video)
        vid_layout.addWidget(self._vid_upload_btn)

        self._vid_label = QLabel("No video selected")
        vid_layout.addWidget(self._vid_label, stretch=1)

        self._vid_prompt = QLineEdit()
        self._vid_prompt.setPlaceholderText("What's happening? (optional)")
        vid_layout.addWidget(self._vid_prompt, stretch=1)

        self._analyze_vid_btn = QPushButton("Analyze")
        self._analyze_vid_btn.clicked.connect(self._analyze_video)
        vid_layout.addWidget(self._analyze_vid_btn)
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

    # ── Settings Tab ──────────────────────────────────────────────────────

    def _create_settings_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

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

        layout.addStretch()
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
        self._status.showMessage("Conversation stopped")

    def _on_mode_changed(self, index):
        from tools.audio.conversation import ConversationMode
        modes = [ConversationMode.AUTO_DETECT, ConversationMode.PUSH_TO_TALK, ConversationMode.CLICK_PER_TURN]
        if index < len(modes):
            self._pipeline.conv_mgr.set_mode(modes[index])
            self._ptt_btn.setVisible(index == 1)  # show PTT button only in push-to-talk mode

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

        # Check for voice command first
        cmd = detect_voice_command(text)
        if cmd:
            self._apply_voice_command(cmd, text)
            return

        self._chat_history.append(("user", text))
        self._update_history()

        # Generate audio only when conversation mode is active (mic running)
        from tools.audio.conversation import ConversationState
        generate_audio = self._pipeline.conv_mgr.state != ConversationState.OFF

        voice_ref = get_truncated_voice_ref(self._voice_ref, self._session["voice_sample_length_s"]) if generate_audio else None
        msgs = [{"role": "user", "content": [text]}]
        self._pipeline.process_turn(msgs, voice_ref, self._session, generate_audio=generate_audio)

    def _on_speech_ready(self, audio: np.ndarray):
        """VAD detected end of speech — process the turn."""
        self._chat_history.append(("user", "[voice input]"))
        self._update_history()

        voice_ref = get_truncated_voice_ref(self._voice_ref, self._session["voice_sample_length_s"])
        msgs = [{"role": "user", "content": [audio]}]
        self._pipeline.process_turn(msgs, voice_ref, self._session)

    def _on_state_changed(self, state_name: str, color: str):
        self._state_dot.setStyleSheet(f"color: {color};")
        display = state_name.replace("_", " ").title()
        self._state_label.setText(display)

    def _on_generation_started(self):
        self._send_btn.setEnabled(False)
        self._status.showMessage("Generating response...")

    def _on_text_update(self, text_chunk: str):
        # Update history with partial text
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            prev = self._chat_history[-1][1]
            self._chat_history[-1] = ("_partial", prev + text_chunk)
        else:
            self._chat_history.append(("_partial", text_chunk))
        self._update_history()

    def _on_generation_finished(self, full_text: str):
        # Replace partial with final
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            self._chat_history[-1] = ("assistant", full_text)
        else:
            self._chat_history.append(("assistant", full_text))
        self._update_history()
        self._send_btn.setEnabled(True)

        voice_label = self._voice_name or "Default"
        self._status.showMessage(f"Voice: {voice_label} | Ready")

    def _on_barge_in(self):
        self._status.showMessage("Interrupted by user")
        if self._chat_history and self._chat_history[-1][0] == "_partial":
            text = self._chat_history[-1][1]
            self._chat_history[-1] = ("assistant", text + " [interrupted]")
            self._update_history()

    def _update_history(self):
        """Render chat history to the display widget."""
        lines = []
        for role, text in self._chat_history[-20:]:
            if role == "user":
                lines.append(f"<b>You:</b> {text}")
            elif role == "assistant":
                lines.append(f"<b>OmniChat:</b> {text}")
            elif role == "_partial":
                lines.append(f"<b>OmniChat:</b> {text}...")
            elif role == "system":
                lines.append(f"<i>{text}</i>")
        self._history_display.setHtml("<br><br>".join(lines))
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
            self._img_path = path
            pixmap = QPixmap(path).scaled(
                self._img_preview.width(), self._img_preview.height(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
            self._img_preview.setPixmap(pixmap)

    def _upload_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv *.webm)"
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
                from PIL import Image
                image = Image.open(self._img_path).convert("RGB")
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
                                       enable_thinking=s.get("enable_thinking", False))
                self.result_ready.emit(result["text"])

        worker = _Worker(self._vid_path, self._vid_prompt.text().strip(), self._session)
        worker.result_ready.connect(lambda t: (
            self._vision_output.setPlainText(t),
            self._status.showMessage("Video analysis complete"),
        ))
        worker.start()
        self._vid_worker = worker

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
        if not name:
            self._status.showMessage("Enter a voice name first")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select Voice WAV", "", "Audio (*.wav *.mp3 *.flac)")
        if not path:
            return
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        add_voice(name, audio, sample_rate=16000)
        self._status.showMessage(f"Voice '{name}' added")
        self._refresh_voice_list()
        self._voice_del_combo.clear()
        self._voice_del_combo.addItems(list_voices())

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
        self._pipeline.stop_conversation()
        # Wait for inference thread to finish so CUDA cleans up safely
        if self._pipeline._inference_thread is not None:
            self._pipeline._inference_thread.wait(5000)
        super().closeEvent(event)
