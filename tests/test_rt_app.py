"""GUI-focused tests for the PySide6 RT application.

These tests use pytest-qt plus a fake audio pipeline so the real UI can be
exercised without GPU inference, live audio devices, or a running remote model.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QWidget

import rt_app
from tools.model import model_manager as mm
from tools.shared import session_playback


class FakeSessionRecorder:
    def __init__(self, output_root, frontend, session_metadata=None, recording_enabled=False, recording_mode="structured", transcript_policy="full"):
        self.output_root = output_root
        self.frontend = frontend
        self.session_metadata = dict(session_metadata or {})
        self.recording_enabled = recording_enabled
        self.recording_mode = recording_mode
        self.transcript_policy = transcript_policy
        self.manifest_path = Path(output_root) / "fake-session.json"
        self.session_dir = Path(output_root) / "fake-session"
        self.recording_toggles = [recording_enabled]
        self.config_updates = []
        self.started_turns = []
        self.model_audio_appends = []
        self.completed_turns = []
        self.session_video = None

    def set_recording_enabled(self, enabled):
        self.recording_enabled = bool(enabled)
        self.recording_toggles.append(bool(enabled))

    def set_recording_config(self, *, enabled=None, recording_mode=None, transcript_policy=None):
        if enabled is not None:
            self.recording_enabled = bool(enabled)
            self.recording_toggles.append(bool(enabled))
        if recording_mode is not None:
            self.recording_mode = recording_mode
        if transcript_policy is not None:
            self.transcript_policy = transcript_policy
        self.config_updates.append({
            "enabled": enabled,
            "recording_mode": recording_mode,
            "transcript_policy": transcript_policy,
        })

    def register_session_video(self, **kwargs):
        self.session_video = dict(kwargs)

    def start_turn(self, *, request_id, turn_metadata, user_audio=None, user_sample_rate=16000):
        self.started_turns.append({
            "request_id": request_id,
            "turn_metadata": dict(turn_metadata),
            "user_audio": None if user_audio is None else np.asarray(user_audio).copy(),
            "user_sample_rate": user_sample_rate,
        })

    def append_model_audio(self, *, request_id, audio_chunk, sample_rate=24000):
        self.model_audio_appends.append({
            "request_id": request_id,
            "audio_chunk": np.asarray(audio_chunk).copy(),
            "sample_rate": sample_rate,
        })

    def complete_turn(self, **kwargs):
        self.completed_turns.append(dict(kwargs))
        return dict(kwargs)


class FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


@dataclass
class FakeCall:
    messages: list[dict]
    voice_ref: np.ndarray | None
    settings: dict
    generate_audio: bool
    trace_context: dict | None


class FakeConversationManager:
    def __init__(self):
        self._silence_threshold_s = 1.0
        self._vad_threshold = 0.5
        self._echo_cooldown_s = 2.0
        self._antivox_boost = 0.1
        self._barge_in_enabled = True
        self._wake_word_enabled = False
        self._wake_word_model = ""
        self._wake_word_threshold = 0.5
        self._dormant_timeout_s = 30.0

    def start(self):
        return None

    def stop(self):
        return None

    def set_mode(self, _mode):
        return None

    def ptt_start(self):
        return None

    def ptt_stop(self):
        return None

    def on_model_start(self):
        return None

    def on_model_done(self):
        return None

    def set_wake_word_enabled(self, value):
        self._wake_word_enabled = value

    def set_wake_word_model(self, value):
        self._wake_word_model = value

    def set_wake_word_threshold(self, value):
        self._wake_word_threshold = value

    def set_dormant_timeout(self, value):
        self._dormant_timeout_s = value


class FakePipeline:
    def __init__(self, _chat_mode_cfg):
        self.state_changed = FakeSignal()
        self.speech_ready = FakeSignal()
        self.text_update = FakeSignal()
        self.audio_chunk_ready = FakeSignal()
        self.generation_started = FakeSignal()
        self.generation_finished = FakeSignal()
        self.generation_error = FakeSignal()
        self.barge_in_detected = FakeSignal()
        self.conv_mgr = FakeConversationManager()
        self._is_generating = False
        self.calls: list[FakeCall] = []
        self.next_response = "stub response"
        self.next_chunks = ["stub ", "response"]
        self.next_audio_chunks: list[np.ndarray] = []

    @property
    def is_generating(self):
        return self._is_generating

    def start_conversation(self):
        return None

    def stop_conversation(self):
        return None

    def process_turn(self, messages, voice_ref, settings, generate_audio=True, trace_context=None):
        self.calls.append(
            FakeCall(
                messages=messages,
                voice_ref=voice_ref,
                settings=dict(settings),
                generate_audio=generate_audio,
                trace_context=dict(trace_context or {}),
            )
        )
        self._is_generating = True
        self.generation_started.emit()
        for chunk in self.next_chunks:
            self.text_update.emit(chunk)
        for chunk in self.next_audio_chunks:
            self.audio_chunk_ready.emit(chunk)
        self._is_generating = False
        self.generation_finished.emit(self.next_response)


class FakeSessionWindowRecorder:
    def __init__(self, *, output_dir, widget, fps=8):
        self.output_dir = Path(output_dir)
        self.widget = widget
        self.fps = fps
        self.started = False
        self.clips = []
        self.stop_result = {"video_path": "session.mp4", "duration_s": 3.0, "fps": fps, "frame_count": 24}

    def start(self):
        self.started = True

    def elapsed_s(self):
        return 0.25

    def add_audio_clip(self, audio, *, sample_rate, offset_s=None):
        self.clips.append({"audio": np.asarray(audio).copy(), "sample_rate": sample_rate, "offset_s": offset_s})

    def stop(self):
        return dict(self.stop_result)


class FakeAudioOutput:
    def __init__(self, *_args, **_kwargs):
        self.volume = 1.0


class FakeVideoWidget(QWidget):
    pass


class FakeMediaPlayer:
    def __init__(self, *_args, **_kwargs):
        self.positionChanged = FakeSignal()
        self.audio_output = None
        self.video_output = None
        self.source = None
        self.play_calls = 0
        self.stop_calls = 0

    def setAudioOutput(self, audio_output):
        self.audio_output = audio_output

    def setVideoOutput(self, video_output):
        self.video_output = video_output

    def setSource(self, source):
        self.source = source

    def play(self):
        self.play_calls += 1
        self.positionChanged.emit(0)

    def stop(self):
        self.stop_calls += 1


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def rt_window_factory(monkeypatch, qapp):
    monkeypatch.setattr(rt_app, "AudioPipeline", FakePipeline)
    monkeypatch.setattr(rt_app, "SessionRecorder", FakeSessionRecorder)
    monkeypatch.setattr(rt_app, "SessionWindowRecorder", FakeSessionWindowRecorder)
    monkeypatch.setattr(rt_app, "QAudioOutput", FakeAudioOutput, raising=False)
    monkeypatch.setattr(rt_app, "QMediaPlayer", FakeMediaPlayer, raising=False)
    monkeypatch.setattr(rt_app, "QVideoWidget", FakeVideoWidget, raising=False)
    monkeypatch.setattr(rt_app, "MULTIMEDIA_AVAILABLE", True, raising=False)
    monkeypatch.setattr(rt_app, "list_voices", lambda: [])
    monkeypatch.setattr(rt_app.QTimer, "singleShot", staticmethod(lambda *_args, **_kwargs: None))
    monkeypatch.setitem(
        sys.modules,
        "tools.audio.wake_word",
        types.SimpleNamespace(list_wake_word_models=lambda: []),
    )
    monkeypatch.setattr(rt_app, "set_qwen_remote_config", lambda **_kwargs: None, raising=False)
    monkeypatch.setattr(
        rt_app,
        "get_qwen_remote_config",
        lambda: {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": None,
            "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "timeout_s": 120.0,
            "endpoints": {
                "health": "/health",
                "models": "models",
                "chat_completions": "chat/completions",
                "responses": "responses",
                "realtime": "realtime",
            },
        },
        raising=False,
    )

    def _settings():
        return {
            "audio": {
                "streaming": {},
                "chat_mode": {},
                "voice_sample_length_s": 5.0,
                "default_voice": None,
            },
            "inference": {
                "temperature": 0.7,
                "max_new_tokens": 2048,
                "repetition_penalty": 1.05,
                "top_p": 0.8,
                "top_k": 100,
                "enable_thinking": False,
            },
            "output": {"default_format": "auto"},
            "display": {"font_size": 12},
            "voice_commands": {"fuzzy_threshold": 0.6},
        }

    def _build(status: dict, backend_name: str):
        status_provider = status if callable(status) else (lambda: status)
        monkeypatch.setattr(rt_app, "get_backend_status", status_provider, raising=False)
        monkeypatch.setattr(rt_app, "get_backend_name", lambda: backend_name, raising=False)

        window = rt_app.OmniChatWindow(_settings())
        window.show()
        qapp.processEvents()
        return window

    return _build


def _default_rt_settings() -> dict:
    return {
        "audio": {
            "streaming": {},
            "chat_mode": {},
            "voice_sample_length_s": 5.0,
            "default_voice": None,
        },
        "inference": {
            "temperature": 0.7,
            "max_new_tokens": 2048,
            "repetition_penalty": 1.05,
            "top_p": 0.8,
            "top_k": 100,
            "enable_thinking": False,
        },
        "output": {"default_format": "auto"},
        "display": {"font_size": 12},
        "voice_commands": {"fuzzy_threshold": 0.6},
    }


def _minicpm_status() -> dict:
    return {
        "backend": "minicpm",
        "summary": "Local MiniCPM backend (openbmb/MiniCPM-o-4_5, quantization=none)",
        "configured_model_name": "openbmb/MiniCPM-o-4_5",
        "server_status": "local",
        "transport": "in-process",
        "capabilities": {
            "model_name": "openbmb/MiniCPM-o-4_5",
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
        },
    }


def _qwen_status() -> dict:
    return {
        "backend": "qwen_remote",
        "summary": "Remote Qwen backend (http://127.0.0.1:8000/v1)",
        "configured_model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "server_status": "ok",
        "server_url": "http://127.0.0.1:8000/v1",
        "transport": {
            "protocol": "http+sse",
            "realtime_ws_url": "ws://127.0.0.1:8000/v1/realtime",
        },
        "health": {"ok": True, "status_code": 200},
        "capabilities": {
            "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
        },
    }


def _qwen_status_with(**overrides) -> dict:
    status = _qwen_status()
    status.update(overrides)
    return status


@pytest.mark.parametrize(
    ("backend_name", "status"),
    [
        ("minicpm", _minicpm_status()),
        ("qwen_remote", _qwen_status()),
    ],
)
def test_rt_window_shows_active_backend_in_main_ui(rt_window_factory, backend_name, status):
    window = rt_window_factory(status, backend_name)

    expected_summary = mm.summarize_backend_status(status)
    expected_detail = mm.format_backend_status(status)

    assert window._voice_backend_label.text() == expected_summary
    assert window._backend_summary_label.text() == expected_summary
    assert window._backend_label.text() == expected_detail
    assert window._about_backend_label.text() == expected_detail


@pytest.mark.parametrize(
    ("backend_name", "status", "expected_startup_text"),
    [
        ("minicpm", _minicpm_status(), "Startup: Ready"),
        ("qwen_remote", _qwen_status(), "Startup: Window ready"),
    ],
)
def test_rt_window_shows_startup_readiness_state(rt_window_factory, backend_name, status, expected_startup_text):
    window = rt_window_factory(status, backend_name)

    assert expected_startup_text in window._startup_status_label.text()
    assert expected_startup_text in window._startup_phase_label.text()


def test_rt_remote_warmup_completion_updates_startup_indicator(rt_window_factory):
    window = rt_window_factory(_qwen_status(), "qwen_remote")

    window._on_remote_warmup_complete({"ok": True, "first_text_s": 1.2, "elapsed_s": 2.4})

    assert "Startup: Remote backend ready" in window._startup_status_label.text()
    assert "First text in 1.2s" in window._startup_status_label.text()
    assert window._startup_phase_label.text() == "Startup: Remote backend ready"


def test_rt_remote_warmup_failure_updates_startup_indicator(rt_window_factory):
    window = rt_window_factory(_qwen_status(), "qwen_remote")

    window._on_remote_warmup_complete({"ok": False, "error": "boot timeout"})

    assert "Startup: Remote warmup failed" in window._startup_status_label.text()
    assert "boot timeout" in window._startup_status_label.text()
    assert window._startup_phase_label.text() == "Startup: Remote warmup failed"


def test_rt_window_starts_and_refreshes_with_real_local_qwen_status(monkeypatch, qapp):
    class DummyQwenTransformersBackend:
        name = "qwen_transformers"

        def get_capabilities(self):
            return {
                "backend": self.name,
                "model_name": "Qwen/test-local",
                "supports_audio_input": True,
                "supports_audio_output": True,
                "supports_streaming_text": True,
                "supports_streaming_audio": True,
                "supports_voice_reference": False,
                "input_sample_rate": 16000,
                "output_sample_rate": 24000,
                "transport": {
                    "protocol": "local_transformers",
                    "streaming_mode": "emulated_after_generation",
                },
            }

        def get_runtime_status(self):
            return {
                "attention_backend": "sdpa",
                "requested_attention_backend": "sdpa",
                "torch_dtype": "torch.bfloat16",
            }

    monkeypatch.setattr(rt_app, "AudioPipeline", FakePipeline)
    monkeypatch.setattr(rt_app, "list_voices", lambda: [])
    monkeypatch.setattr(rt_app.QTimer, "singleShot", staticmethod(lambda *_args, **_kwargs: None))
    monkeypatch.setitem(
        sys.modules,
        "tools.audio.wake_word",
        types.SimpleNamespace(list_wake_word_models=lambda: []),
    )
    monkeypatch.setattr(rt_app, "set_qwen_remote_config", lambda **_kwargs: None, raising=False)
    monkeypatch.setattr(
        rt_app,
        "get_qwen_remote_config",
        lambda: {
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": None,
            "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "timeout_s": 120.0,
            "endpoints": {
                "health": "/health",
                "models": "models",
                "chat_completions": "chat/completions",
                "responses": "responses",
                "realtime": "realtime",
            },
        },
        raising=False,
    )
    monkeypatch.setattr(mm, "_backend_name", "qwen_transformers")
    monkeypatch.setattr(mm, "_backend", DummyQwenTransformersBackend())
    monkeypatch.setattr(
        mm,
        "_qwen_transformers_config",
        {
            **mm.get_qwen_transformers_config(),
            "checkpoint": "Qwen/test-local",
            "attn_implementation": "sdpa",
            "torch_dtype": "bfloat16",
        },
    )

    window = rt_app.OmniChatWindow(_default_rt_settings())
    window.show()
    qapp.processEvents()

    expected_status = mm.get_backend_status()
    expected_summary = mm.summarize_backend_status(expected_status)
    expected_detail = mm.format_backend_status(expected_status)

    assert expected_status["runtime"]["attention_backend"] == "sdpa"

    assert window._voice_backend_label.text() == expected_summary
    assert window._backend_summary_label.text() == expected_summary
    assert window._backend_label.text() == expected_detail
    assert window._about_backend_label.text() == expected_detail

    window._refresh_backend_status(show_message=False)
    qapp.processEvents()

    assert window._backend_label.text() == expected_detail
    assert "Backend: qwen_transformers" in window._backend_label.text()


@pytest.mark.parametrize(
    ("backend_name", "status", "model_cfg", "expected_generate_audio"),
    [
        (
            "minicpm",
            _minicpm_status(),
            {"backend": "minicpm"},
            False,
        ),
        (
            "qwen_remote",
            _qwen_status(),
            {"backend": "qwen_remote"},
            False,
        ),
        (
            "qwen_llamacpp",
            _qwen_status_with(backend="qwen_llamacpp"),
            {"backend": "qwen_llamacpp", "llama_cpp": {"speech_backend": "none"}},
            False,
        ),
    ],
)
def test_rt_typed_chat_dispatches_correct_profile_request(
    rt_window_factory,
    backend_name,
    status,
    model_cfg,
    expected_generate_audio,
):
    window = rt_window_factory(status, backend_name)
    window._settings["model"] = model_cfg
    window._chat_history = [
        ("user", "First question"),
        ("assistant", "First answer"),
        ("user", "Second question"),
        ("assistant", "Second answer"),
        ("user", "Third question"),
        ("assistant", "Third answer"),
    ]
    window._update_history()

    window._pipeline.next_chunks = ["Hello", " there"]
    window._pipeline.next_response = "Hello there"

    window._text_input.setText("Fourth question")
    window._send_text()

    assert len(window._pipeline.calls) == 1
    call = window._pipeline.calls[0]

    assert call.generate_audio is expected_generate_audio
    assert call.messages == [{"role": "user", "content": ["Fourth question"]}]
    assert call.settings["max_new_tokens"] == 2048
    assert window._chat_history[-1] == ("assistant", "Hello there")
    assert window._send_btn.isEnabled() is True
    assert window._text_input.isEnabled() is True


@pytest.mark.parametrize(
    ("backend_name", "status"),
    [
        ("minicpm", _minicpm_status()),
        ("qwen_remote", _qwen_status()),
    ],
)
def test_rt_voice_turn_uses_audio_mode_and_preserves_hidden_response(rt_window_factory, backend_name, status):
    window = rt_window_factory(status, backend_name)
    window._record_mode_combo.setCurrentIndex(window._record_mode_combo.findData("structured"))
    window._record_btn.setChecked(True)
    window._pipeline.next_chunks = ["Spoken ", "reply"]
    window._pipeline.next_response = "Spoken reply"
    window._pipeline.next_audio_chunks = [np.zeros(2400, dtype=np.float32)]

    audio = np.zeros(16000, dtype=np.float32)
    window._on_speech_ready(audio)

    assert len(window._pipeline.calls) == 1
    call = window._pipeline.calls[0]
    assert call.generate_audio is True
    assert call.trace_context["request_id"].startswith("chat-")
    assert call.messages[-1]["role"] == "user"
    assert isinstance(call.messages[-1]["content"][0], np.ndarray)
    assert window._session_recorder.started_turns[0]["turn_metadata"]["modality"] == "audio"
    assert window._session_recorder.completed_turns[0]["response_mode"] == "audio"
    assert len(window._session_recorder.model_audio_appends) == 1
    assert window._chat_history[-1] == ("_assistant_hidden", "Spoken reply")
    assert ("user", "[voice input]") in window._chat_history
    assert all(role != "_partial" for role, _ in window._chat_history)
    history_html = window._history_display.toHtml()
    assert "[Speech Input]" in history_html
    assert "OmniChat [minicpm] [Spoken Text]:" in history_html or "OmniChat [qwen_remote] [Spoken Text]:" in history_html
    assert "font-style:italic" in history_html
    assert "Spoken reply" in history_html


def test_rt_voice_turn_renders_spoken_markdown(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._pipeline.next_chunks = ["## Heading\n\n- one\n- two"]
    window._pipeline.next_response = "## Heading\n\n- one\n- two"

    audio = np.zeros(16000, dtype=np.float32)
    window._on_speech_ready(audio)

    history_html = window._history_display.toHtml()

    assert "OmniChat [minicpm] [Spoken Text]:" in history_html
    assert "Heading" in history_html
    assert "<ul" in history_html
    assert "<li" in history_html


def test_rt_history_uses_model_qualified_assistant_label(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._settings["model"] = {"backend": "gemma_llamacpp", "display_name": "Gemma 4 S-Size Local"}
    window._chat_history = [("assistant", "Hello there")]

    window._update_history()

    history_html = window._history_display.toHtml()
    assert "OmniChat [Gemma 4 S-Size Local]:" in history_html


def test_rt_record_toggle_updates_session_recorder(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")

    window._record_btn.setChecked(True)

    assert window._recording_enabled is True
    assert "Recording Metadata" in window._record_btn.text()
    assert window._session_recorder.recording_toggles[-1] is True


def test_rt_voice_chat_controls_use_two_rows(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")

    assert window._voice_controls_top_row.count() > 0
    assert window._voice_controls_bottom_row.count() > 0
    assert window._voice_controls_top_row.indexOf(window._conv_start_btn) >= 0
    assert window._voice_controls_bottom_row.indexOf(window._record_btn) >= 0
    assert window._voice_controls_bottom_row.indexOf(window._voice_combo) >= 0


def test_rt_record_mode_video_starts_window_recorder(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._record_mode_combo.setCurrentIndex(window._record_mode_combo.findData("video"))

    window._record_btn.setChecked(True)

    assert window._session_recorder.recording_mode == "video"
    assert window._session_video_recorder is not None
    assert window._session_video_recorder.started is True


def test_rt_record_toggle_off_finalizes_video_session(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._record_mode_combo.setCurrentIndex(window._record_mode_combo.findData("video"))
    window._record_btn.setChecked(True)
    recorder = window._session_recorder

    window._record_btn.setChecked(False)

    assert recorder.session_video["video_path"] == "session.mp4"
    assert window._session_recorder is None


def test_rt_record_toggle_on_after_off_creates_new_session(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")

    window._record_btn.setChecked(True)
    first_recorder = window._session_recorder
    window._record_btn.setChecked(False)
    window._record_btn.setChecked(True)

    assert first_recorder is not None
    assert window._session_recorder is not None
    assert window._session_recorder is not first_recorder


def test_rt_text_turn_records_text_only_metadata(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._record_btn.setChecked(True)
    window._text_input.setText("Hello")

    window._send_text()

    assert window._session_recorder.started_turns[0]["turn_metadata"]["modality"] == "text"
    assert window._session_recorder.started_turns[0]["turn_metadata"]["prompt_text"] == "Hello"
    assert window._session_recorder.completed_turns[0]["response_mode"] == "text"


def test_rt_audio_turn_records_voice_metadata(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._record_btn.setChecked(True)
    window._voice_ref = np.ones(16000, dtype=np.float32)
    window._voice_name = "Morgan Freeman"
    window._pipeline.next_audio_chunks = [np.ones(1200, dtype=np.float32)]

    window._on_speech_ready(np.zeros(16000, dtype=np.float32))

    turn = window._session_recorder.started_turns[0]["turn_metadata"]
    assert turn["voice"]["mode"] == "cloned"
    assert turn["voice"]["name"] == "Morgan Freeman"


def test_rt_voice_turn_uses_speech_repetition_penalty_override(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._session["repetition_penalty"] = 1.5
    window._session["speech_repetition_penalty"] = 1.05

    audio = np.zeros(16000, dtype=np.float32)
    window._on_speech_ready(audio)

    assert len(window._pipeline.calls) == 1
    call = window._pipeline.calls[0]
    assert call.settings["repetition_penalty"] == pytest.approx(1.05)


def test_rt_text_turn_shows_responding_text_status(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._response_display_mode = "text"

    window._on_generation_started()
    assert window._response_status_phase == "thinking"
    assert window._status.currentMessage() == "Thinking..."
    assert window._state_label.text() == "Thinking..."

    window._on_text_update("Hello")

    assert window._response_status_phase == "text"
    assert window._status.currentMessage() == "Responding (Text)..."
    assert window._state_label.text() == "Responding (Text)..."


def test_rt_audio_turn_shows_tts_phase_for_qwen_bridge(rt_window_factory, monkeypatch):
    monkeypatch.setattr(rt_app, "get_qwen_llamacpp_config", lambda: {"speech_backend": "minicpm_streaming"}, raising=False)
    window = rt_window_factory(_minicpm_status(), "qwen_llamacpp")
    window._response_display_mode = "audio"
    window._response_status_phase = "thinking"

    window._on_text_update("Bridge text")

    assert window._response_status_phase == "tts"
    assert window._status.currentMessage() == "Responding (TTS)..."
    assert window._state_label.text() == "Responding (TTS)..."


def test_rt_audio_chunk_switches_status_to_audio(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._response_display_mode = "audio"
    window._response_status_phase = "thinking"

    window._on_audio_chunk_ready(np.zeros(128, dtype=np.float32))

    assert window._response_status_phase == "audio"
    assert window._status.currentMessage() == "Responding (Audio)..."
    assert window._state_label.text() == "Responding (Audio)..."


def test_rt_pipeline_processing_state_displays_thinking_label(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")

    window._on_state_changed("PROCESSING", "#eab308")

    assert window._state_label.text() == "Thinking..."


def test_rt_stop_conversation_restores_text_entry_after_manual_interrupt(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._chat_history = [("user", "Prompt"), ("_partial", "Partial reply")]
    window._response_display_mode = "text"
    window._on_generation_started()
    window._pipeline._is_generating = True

    window._stop_conversation()

    assert window._send_btn.isEnabled() is True
    assert window._text_input.isEnabled() is True
    assert window._chat_history[-1] == ("assistant", "Partial reply [interrupted]")
    assert window._response_status_phase == "idle"
    assert window._response_display_mode == "text"


def test_rt_oneshot_text_backend_promotes_to_generating(rt_window_factory, monkeypatch):
    monkeypatch.setattr(
        rt_app,
        "get_backend_capabilities",
        lambda: {"transport": {"streaming_mode": "oneshot_cli"}},
        raising=False,
    )
    monkeypatch.setattr(rt_app.QTimer, "singleShot", staticmethod(lambda _ms, callback: callback()))
    window = rt_window_factory(_qwen_status_with(backend="qwen_llamacpp"), "qwen_llamacpp")
    window._response_display_mode = "text"
    window._pipeline._is_generating = True

    window._on_generation_started()

    assert window._response_status_phase == "generating"
    assert window._status.currentMessage() == "Generating..."
    assert window._state_label.text() == "Generating..."


@pytest.mark.parametrize(
    ("backend_name", "status"),
    [
        ("minicpm", _minicpm_status()),
        ("qwen_remote", _qwen_status()),
    ],
)
def test_rt_ptt_release_routes_audio_into_speech_turn(rt_window_factory, backend_name, status):
    window = rt_window_factory(status, backend_name)
    window._pipeline.next_chunks = ["PTT ", "reply"]
    window._pipeline.next_response = "PTT reply"

    recorded_audio = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    window._pipeline.conv_mgr.ptt_stop = lambda: recorded_audio

    window._on_ptt_pressed()
    assert "#ef4444" in window._ptt_btn.styleSheet()

    window._on_ptt_released()

    assert window._ptt_btn.styleSheet() == ""
    assert len(window._pipeline.calls) == 1
    call = window._pipeline.calls[0]
    assert call.generate_audio is True
    np.testing.assert_array_equal(call.messages[-1]["content"][0], recorded_audio)
    assert window._chat_history[-1] == ("_assistant_hidden", "PTT reply")


def test_rt_remote_settings_actions_refresh_visible_backend_summary(rt_window_factory, monkeypatch):
    status_holder = {
        "value": _qwen_status_with(
            server_status="unreachable",
            health={"ok": False, "status_code": 503, "error": "booting"},
        )
    }
    monkeypatch.setattr(rt_app, "get_backend_status", lambda: status_holder["value"])

    window = rt_window_factory(lambda: status_holder["value"], "qwen_remote")

    window._remote_base_url.setText("http://127.0.0.1:9000/v1")
    window._remote_model_name.setText("Qwen/Test")
    window._remote_api_key.setText("secret")
    window._remote_timeout.setValue(42.0)
    window._remote_health_ep.setText("/healthz")
    window._remote_models_ep.setText("models")
    window._remote_chat_ep.setText("chat/completions")
    window._remote_responses_ep.setText("responses")
    window._remote_realtime_ep.setText("realtime")

    status_holder["value"] = _qwen_status_with(
        configured_model_name="Qwen/Test",
        capabilities={
            "model_name": "Qwen/Test",
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
        },
        server_status="starting",
        server_url="http://127.0.0.1:9000/v1",
        transport={
            "protocol": "http+sse",
            "realtime_ws_url": "ws://127.0.0.1:9000/v1/realtime",
        },
        health={"ok": False, "status_code": 503, "error": "booting"},
    )
    window._apply_remote_settings()
    assert window._voice_backend_label.text() == mm.summarize_backend_status(status_holder["value"])
    assert "Base URL: http://127.0.0.1:9000/v1" in window._remote_probe_label.text()

    status_holder["value"] = _qwen_status_with(
        configured_model_name="Qwen/Test",
        capabilities={
            "model_name": "Qwen/Test",
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
        },
        server_status="ok",
        server_url="http://127.0.0.1:9000/v1",
        transport={
            "protocol": "http+sse",
            "realtime_ws_url": "ws://127.0.0.1:9000/v1/realtime",
        },
        health={"ok": True, "status_code": 200},
    )
    window._test_remote_health()
    assert window._voice_backend_label.text() == mm.summarize_backend_status(status_holder["value"])
    assert window._remote_probe_label.text() == mm.format_backend_status(status_holder["value"])

    class FakeRealtimeClient:
        def __init__(self, _config):
            pass

        def perform_realtime_handshake(self):
            return {
                "ok": True,
                "url": "ws://127.0.0.1:9000/v1/realtime",
                "event_type": "session.created",
            }

    monkeypatch.setattr(rt_app, "QwenOmniClient", FakeRealtimeClient)
    status_holder["value"] = _qwen_status_with(
        configured_model_name="Qwen/Test",
        capabilities={
            "model_name": "Qwen/Test",
            "supports_streaming_text": True,
            "supports_streaming_audio": True,
        },
        server_status="ok",
        server_url="http://127.0.0.1:9000/v1",
        transport={
            "protocol": "http+ws",
            "realtime_ws_url": "ws://127.0.0.1:9000/v1/realtime",
        },
        health={"ok": True, "status_code": 200},
    )
    window._test_remote_realtime()
    assert window._voice_backend_label.text() == mm.summarize_backend_status(status_holder["value"])
    assert "Realtime handshake OK" in window._remote_probe_label.text()
    assert "ws://127.0.0.1:9000/v1/realtime" in window._remote_probe_label.text()


@pytest.mark.parametrize(
    ("backend_name", "status"),
    [
        ("minicpm", _minicpm_status()),
        ("qwen_remote", _qwen_status()),
    ],
)
def test_rt_typed_chat_writes_expected_trace_events(rt_window_factory, monkeypatch, tmp_path, backend_name, status):
    trace_log = tmp_path / f"trace_{backend_name}.log"
    logger = logging.getLogger(f"omnichat.trace.test.{backend_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(trace_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    monkeypatch.setattr(rt_app, "logger", logger)

    window = rt_window_factory(status, backend_name)
    window._pipeline.next_chunks = ["Hello", " there"]
    window._pipeline.next_response = "Hello there"
    window._text_input.setText("Fourth question")
    window._send_text()

    trace_id = window._pipeline.calls[0].trace_context["request_id"]
    handler.flush()
    contents = trace_log.read_text(encoding="utf-8")
    logger.removeHandler(handler)
    handler.close()

    assert f"trace_id={trace_id} stage=ui event=send_text" in contents
    assert f"trace_id={trace_id} stage=ui event=generation_started" in contents
    assert f"trace_id={trace_id} stage=ui event=generation_finished" in contents
    assert "response_chars=11" in contents


def test_rt_typed_chat_disables_audio_for_plain_qwen_even_in_conversation_mode(rt_window_factory):
    from tools.audio.conversation import ConversationState

    window = rt_window_factory(_qwen_status_with(backend="qwen_llamacpp"), "qwen_llamacpp")
    window._settings["active_model_profile"] = "qwen35_27b_llamacpp_local"
    window._settings["model"] = {
        "backend": "qwen_llamacpp",
        "llama_cpp": {"speech_backend": "none"},
    }
    window._pipeline.conv_mgr.state = ConversationState.LISTENING
    window._text_input.setText("Hello")

    window._send_text()

    assert len(window._pipeline.calls) == 1
    assert window._pipeline.calls[0].generate_audio is False


def test_rt_typed_chat_keeps_audio_disabled_for_qwen_hybrid_in_conversation_mode(rt_window_factory):
    window = rt_window_factory(_qwen_status_with(backend="qwen_llamacpp"), "qwen_llamacpp")
    window._settings["active_model_profile"] = "qwen35_27b_llamacpp_local_minicpm_tts"
    window._settings["model"] = {
        "backend": "qwen_llamacpp",
        "llama_cpp": {"speech_backend": "minicpm_streaming"},
    }
    window._text_input.setText("Hello")

    window._send_text()

    assert len(window._pipeline.calls) == 1
    assert window._pipeline.calls[0].generate_audio is False


def test_rt_typed_chat_disables_audio_for_plain_gemma_even_in_conversation_mode(rt_window_factory):
    from tools.audio.conversation import ConversationState

    window = rt_window_factory(_qwen_status_with(backend="gemma_llamacpp"), "gemma_llamacpp")
    window._settings["active_model_profile"] = "gemma4_ssize_llamacpp_local"
    window._settings["model"] = {
        "backend": "gemma_llamacpp",
        "llama_cpp": {"speech_backend": "none"},
    }
    window._pipeline.conv_mgr.state = ConversationState.LISTENING
    window._text_input.setText("Hello")

    window._send_text()

    assert len(window._pipeline.calls) == 1
    assert window._pipeline.calls[0].generate_audio is False


def test_rt_typed_chat_keeps_audio_disabled_for_gemma_hybrid_in_conversation_mode(rt_window_factory):
    window = rt_window_factory(_qwen_status_with(backend="gemma_llamacpp"), "gemma_llamacpp")
    window._settings["active_model_profile"] = "gemma4_ssize_llamacpp_mincpm_tts"
    window._settings["model"] = {
        "backend": "gemma_llamacpp",
        "llama_cpp": {"speech_backend": "minicpm_streaming"},
    }
    window._text_input.setText("Hello")

    window._send_text()

    assert len(window._pipeline.calls) == 1
    assert window._pipeline.calls[0].generate_audio is False


def test_rt_typed_chat_keeps_audio_disabled_for_minicpm_without_conversation_mode(rt_window_factory):
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._text_input.setText("Hello")

    window._send_text()

    assert len(window._pipeline.calls) == 1
    assert window._pipeline.calls[0].generate_audio is False


@pytest.mark.parametrize(
    ("backend_name", "status"),
    [
        ("minicpm", _minicpm_status()),
        ("qwen_remote", _qwen_status()),
    ],
)
def test_rt_voice_turn_writes_expected_trace_events(rt_window_factory, monkeypatch, tmp_path, backend_name, status):
    trace_log = tmp_path / f"trace_voice_{backend_name}.log"
    logger = logging.getLogger(f"omnichat.trace.test.voice.{backend_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(trace_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    monkeypatch.setattr(rt_app, "logger", logger)

    window = rt_window_factory(status, backend_name)
    window._pipeline.next_chunks = ["Spoken ", "reply"]
    window._pipeline.next_response = "Spoken reply"

    audio = np.zeros(16000, dtype=np.float32)
    window._on_speech_ready(audio)

    trace_id = window._pipeline.calls[0].trace_context["request_id"]
    handler.flush()
    contents = trace_log.read_text(encoding="utf-8")
    logger.removeHandler(handler)
    handler.close()

    assert f"trace_id={trace_id} stage=ui event=send_speech" in contents
    assert "audio_samples=16000" in contents
    assert f"trace_id={trace_id} stage=ui event=generation_started" in contents
    assert f"trace_id={trace_id} stage=ui event=generation_finished" in contents
    assert "response_chars=12" in contents


@pytest.mark.parametrize(
    ("backend_name", "status"),
    [
        ("minicpm", _minicpm_status()),
        ("qwen_remote", _qwen_status()),
    ],
)
def test_rt_voice_barge_in_writes_trace_event(rt_window_factory, monkeypatch, tmp_path, backend_name, status):
    trace_log = tmp_path / f"trace_barge_{backend_name}.log"
    logger = logging.getLogger(f"omnichat.trace.test.barge.{backend_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(trace_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    monkeypatch.setattr(rt_app, "logger", logger)

    window = rt_window_factory(status, backend_name)

    def process_turn(messages, voice_ref, settings, generate_audio=True, trace_context=None):
        window._pipeline.calls.append(
            FakeCall(
                messages=messages,
                voice_ref=voice_ref,
                settings=dict(settings),
                generate_audio=generate_audio,
                trace_context=dict(trace_context or {}),
            )
        )
        window._pipeline._is_generating = True
        window._pipeline.generation_started.emit()

    monkeypatch.setattr(window._pipeline, "process_turn", process_turn)

    audio = np.zeros(8000, dtype=np.float32)
    window._on_speech_ready(audio)
    trace_id = window._pipeline.calls[0].trace_context["request_id"]
    window._pipeline.barge_in_detected.emit()

    handler.flush()
    contents = trace_log.read_text(encoding="utf-8")
    logger.removeHandler(handler)
    handler.close()

    assert f"trace_id={trace_id} stage=ui event=send_speech" in contents
    assert f"trace_id={trace_id} stage=ui event=generation_started" in contents
    assert f"trace_id={trace_id} stage=ui event=barge_in" in contents


def test_rt_import_voice_uses_filename_when_name_is_blank(rt_window_factory, monkeypatch):
    voices = []
    imported = {}

    monkeypatch.setattr(rt_app, "list_voices", lambda: list(voices))
    monkeypatch.setattr(rt_app.QFileDialog, "getOpenFileName", staticmethod(lambda *_args, **_kwargs: (r"C:\temp\Morgan_Freeman.mp4", "Media (*.mp4)")))
    monkeypatch.setattr(rt_app, "load_voice_sample_from_media", lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32))

    def fake_add_voice(name, audio, sample_rate=16000):
        imported["name"] = name
        imported["samples"] = len(audio)
        imported["sample_rate"] = sample_rate
        voices.append(name)
        return fr"C:\voices\{name}.wav"

    monkeypatch.setattr(rt_app, "add_voice", fake_add_voice)

    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._voice_add_name.setText("")
    window._add_voice_from_file()

    assert imported == {"name": "Morgan Freeman", "samples": 16000, "sample_rate": 16000}
    assert window._voice_add_name.text() == "Morgan Freeman"
    assert window._voice_combo.currentText() == "Morgan Freeman"
    assert window._voice_del_combo.currentText() == "Morgan Freeman"
    assert window.statusBar().currentMessage() == "Voice 'Morgan Freeman' imported from Morgan_Freeman.mp4"


def test_rt_import_voice_prefers_explicit_name(rt_window_factory, monkeypatch):
    voices = []
    imported = {}

    monkeypatch.setattr(rt_app, "list_voices", lambda: list(voices))
    monkeypatch.setattr(rt_app.QFileDialog, "getOpenFileName", staticmethod(lambda *_args, **_kwargs: (r"C:\temp\clip.wav", "Audio (*.wav)")))
    monkeypatch.setattr(rt_app, "load_voice_sample_from_media", lambda *_args, **_kwargs: np.zeros(16000, dtype=np.float32))

    def fake_add_voice(name, audio, sample_rate=16000):
        imported["name"] = name
        imported["samples"] = len(audio)
        imported["sample_rate"] = sample_rate
        voices.append(name)
        return fr"C:\voices\{name}.wav"

    monkeypatch.setattr(rt_app, "add_voice", fake_add_voice)

    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._voice_add_name.setText("Custom Voice")
    window._add_voice_from_file()

    assert imported == {"name": "Custom Voice", "samples": 16000, "sample_rate": 16000}
    assert window._voice_combo.currentText() == "Custom Voice"


def test_rt_record_voice_from_mic_uses_explicit_name(rt_window_factory, monkeypatch):
    voices = []
    imported = {}

    monkeypatch.setattr(rt_app, "list_voices", lambda: list(voices))
    monkeypatch.setattr(rt_app, "record_voice_sample", lambda **_kwargs: np.ones(16000, dtype=np.float32))

    def fake_add_voice(name, audio, sample_rate=16000):
        imported["name"] = name
        imported["samples"] = len(audio)
        imported["sample_rate"] = sample_rate
        voices.append(name)
        return fr"C:\voices\{name}.wav"

    monkeypatch.setattr(rt_app, "add_voice", fake_add_voice)

    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._voice_add_name.setText("Desk Mic")
    window._record_voice_from_mic()

    assert imported == {"name": "Desk Mic", "samples": 16000, "sample_rate": 16000}
    assert window._voice_combo.currentText() == "Desk Mic"
    assert window._voice_del_combo.currentText() == "Desk Mic"
    assert window.statusBar().currentMessage() == "Voice 'Desk Mic' recorded from microphone"


def test_rt_playback_tab_loads_manifest_and_renders_history(rt_window_factory, monkeypatch, tmp_path):
    manifest_path = tmp_path / "session.json"
    video_path = tmp_path / "session.mp4"
    fake_session = session_playback.SessionPlaybackManifest(
        manifest_path=manifest_path,
        session_dir=tmp_path,
        data={"created_at": "2026-04-16T22:00:00Z"},
        turns=[
            {
                "turn_index": 1,
                "modality": "audio",
                "prompt": {"tokens_est": 3},
                "response": {"tokens_est": 5, "mode": "audio"},
                "timing": {"prompt_offset_s": 0.5, "first_text_s": 0.4, "elapsed_s": 1.6, "first_text_offset_s": 0.9, "completed_offset_s": 2.1},
            }
        ],
        assistant_label="OmniChat [Demo Model]",
        recording_mode="video",
        transcript_policy="full",
        video_path=video_path,
    )
    fake_history = [("user", "Hello"), ("assistant", "World")]
    fake_events = [
        session_playback.SessionReplayEvent(0, 1, "user", "Hello", [("user", "Hello")], None, 1.0, "Turn 1: user"),
        session_playback.SessionReplayEvent(1, 1, "assistant", "World", fake_history, None, 1.0, "Turn 1: assistant"),
    ]

    monkeypatch.setattr(rt_app.session_playback, "discover_session_manifests", lambda _root: [manifest_path])
    monkeypatch.setattr(rt_app.session_playback, "load_session_manifest", lambda _path: fake_session)
    monkeypatch.setattr(rt_app.session_playback, "build_chat_history", lambda _session: fake_history)
    monkeypatch.setattr(rt_app.session_playback, "build_replay_events", lambda _session: fake_events)

    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._refresh_playback_sessions()

    assert window._playback_session is fake_session
    assert window._playback_session_combo.count() == 1
    assert window._playback_open_video_btn.isEnabled() is True
    html = window._playback_history_display.toHtml()
    assert "Hello" in html
    assert "World" in html


def test_rt_playback_embedded_video_uses_overlay_timeline(rt_window_factory, tmp_path):
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"video")
    window = rt_window_factory(_minicpm_status(), "minicpm")
    fake_session = session_playback.SessionPlaybackManifest(
        manifest_path=tmp_path / "session.json",
        session_dir=tmp_path,
        data={"created_at": "2026-04-16T22:00:00Z"},
        turns=[
            {
                "turn_index": 1,
                "modality": "audio",
                "prompt": {"tokens_est": 3},
                "response": {"tokens_est": 4, "mode": "audio"},
                "timing": {"prompt_offset_s": 0.5, "first_text_s": 0.4, "elapsed_s": 1.7, "first_text_offset_s": 0.9, "completed_offset_s": 2.2},
                "interrupted": False,
                "error": None,
            }
        ],
        assistant_label="OmniChat [Demo Model]",
        recording_mode="video",
        transcript_policy="full",
        video_path=video_path,
    )

    window._set_playback_session(fake_session)
    window._play_selected_session()
    window._on_playback_video_position_changed(1400)

    assert window._playback_media_player.play_calls == 1
    assert "Turn 1" in window._playback_overlay_label.text()
    assert "Responding" in window._playback_overlay_label.text()
    assert "Prompt tok: 3" in window._playback_overlay_label.text()


def test_rt_playback_export_actions_call_session_exporters(rt_window_factory, monkeypatch, tmp_path):
    manifest_path = tmp_path / "session.json"
    video_path = tmp_path / "session.mp4"
    video_path.write_bytes(b"video")
    fake_session = session_playback.SessionPlaybackManifest(
        manifest_path=manifest_path,
        session_dir=tmp_path,
        data={"created_at": "2026-04-16T22:00:00Z"},
        turns=[],
        assistant_label="OmniChat [Demo Model]",
        recording_mode="video",
        transcript_policy="full",
        video_path=video_path,
    )
    monkeypatch.setattr(rt_app.session_playback, "discover_session_manifests", lambda _root: [])
    window = rt_window_factory(_minicpm_status(), "minicpm")
    window._playback_session = fake_session

    exported = {}
    monkeypatch.setattr(rt_app.QFileDialog, "getSaveFileName", staticmethod(lambda *_args, **_kwargs: (str(tmp_path / "demo.mp3"), "MP3 (*.mp3)")))
    monkeypatch.setattr(rt_app.session_playback, "export_stitched_audio", lambda session, output_path: exported.update({"audio_session": session, "audio_path": Path(output_path)}))

    window._export_selected_session_audio()

    monkeypatch.setattr(rt_app.QFileDialog, "getSaveFileName", staticmethod(lambda *_args, **_kwargs: (str(tmp_path / "demo.mp4"), "MP4 (*.mp4)")))

    window._export_selected_session_video()

    assert exported["audio_session"] is fake_session
    assert exported["audio_path"] == tmp_path / "demo.mp3"
    assert (tmp_path / "demo.mp4").read_bytes() == b"video"