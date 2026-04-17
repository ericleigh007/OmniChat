from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget

import rt_app
from tools.shared import session_playback


SCREENSHOT_NAME = "OmniChat-RT-Playback.png"
GIF_NAME = "OmniChat-RT-Playback.gif"


class FakeSignal:
    def __init__(self):
        self._callbacks: list[Any] = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class FakeConversationManager:
    def __init__(self):
        self._silence_threshold_s = 1.0
        self._vad_threshold = 0.5
        self._echo_cooldown_s = 2.0
        self._antivox_boost = 0.1
        self._barge_in_enabled = True


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
        self._inference_thread = None

    def start_conversation(self):
        return None

    def stop_conversation(self):
        return None


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


def _configure_demo_font(app: QApplication) -> None:
    candidate_paths = [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
    ]
    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        font_id = QFontDatabase.addApplicationFont(str(candidate))
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            app.setFont(QFont(families[0], 10))
            return


@dataclass(frozen=True)
class GeneratedPlaybackAssets:
    screenshot_path: Path
    gif_path: Path


def _default_settings(output_dir: Path) -> dict[str, Any]:
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
        "output": {
            "default_format": "auto",
            "save_dir": str(output_dir.parent),
        },
        "display": {"font_size": 12},
        "voice_commands": {"fuzzy_threshold": 0.6},
    }


def _backend_status() -> dict[str, Any]:
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


def _demo_session(session_dir: Path) -> session_playback.SessionPlaybackManifest:
    video_path = session_dir / "session.mp4"
    video_path.write_bytes(b"demo")
    turns = [
        {
            "turn_index": 1,
            "modality": "audio",
            "prompt": {
                "text": "[voice input] Give me a short rundown of session playback.",
                "tokens_est": 10,
                "audio_duration_s": 1.1,
                "audio_path": None,
            },
            "response": {
                "text": "Playback now re-renders the transcript, syncs overlay stats, and can export audio or video.",
                "mode": "audio",
                "tokens_est": 16,
                "audio_duration_s": 1.8,
                "audio_path": None,
            },
            "timing": {
                "prompt_offset_s": 0.5,
                "first_text_s": 0.4,
                "elapsed_s": 1.8,
                "first_text_offset_s": 0.9,
                "completed_offset_s": 2.3,
            },
        },
        {
            "turn_index": 2,
            "modality": "text",
            "prompt": {
                "text": "Show me the second turn controls.",
                "tokens_est": 7,
                "audio_duration_s": 0.0,
                "audio_path": None,
            },
            "response": {
                "text": "The playback tab can step events, replay from the beginning, and open the captured MP4.",
                "mode": "text",
                "tokens_est": 15,
                "audio_duration_s": 0.0,
                "audio_path": None,
            },
            "timing": {
                "prompt_offset_s": 2.8,
                "first_text_s": 0.3,
                "elapsed_s": 1.2,
                "first_text_offset_s": 3.1,
                "completed_offset_s": 4.0,
            },
        },
    ]
    return session_playback.SessionPlaybackManifest(
        manifest_path=session_dir / "session.json",
        session_dir=session_dir,
        data={"created_at": "2026-04-17T12:00:00Z"},
        turns=turns,
        assistant_label="OmniChat [Playback Demo]",
        recording_mode="structured-video",
        transcript_policy="full",
        video_path=video_path,
    )


def _capture_frame(window: rt_app.OmniChatWindow, path: Path) -> Image.Image:
    window.repaint()
    QApplication.processEvents()
    window.grab().save(str(path), "PNG")
    return Image.open(path).convert("RGB")


def _apply_demo_state(window: rt_app.OmniChatWindow, session: session_playback.SessionPlaybackManifest, event_index: int, position_ms: int) -> None:
    if event_index >= 0:
        window._show_playback_event(event_index)
    window._on_playback_video_position_changed(position_ms)
    if window._playback_media_status is not None:
        window._playback_media_status.setText(f"Playback position: {position_ms / 1000.0:.2f}s")
    QApplication.processEvents()


def generate_playback_demo_assets(output_dir: str | Path) -> GeneratedPlaybackAssets:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    temp_dir = destination / ".playback_asset_frames"
    temp_dir.mkdir(parents=True, exist_ok=True)

    originals = {
        "AudioPipeline": rt_app.AudioPipeline,
        "QAudioOutput": rt_app.QAudioOutput,
        "QMediaPlayer": rt_app.QMediaPlayer,
        "QVideoWidget": rt_app.QVideoWidget,
        "MULTIMEDIA_AVAILABLE": rt_app.MULTIMEDIA_AVAILABLE,
        "list_voices": rt_app.list_voices,
        "get_backend_status": rt_app.get_backend_status,
        "get_backend_name": rt_app.get_backend_name,
        "get_qwen_remote_config": rt_app.get_qwen_remote_config,
        "set_qwen_remote_config": rt_app.set_qwen_remote_config,
    }

    created_app = False
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
        created_app = True
    _configure_demo_font(app)

    try:
        rt_app.AudioPipeline = FakePipeline
        rt_app.QAudioOutput = FakeAudioOutput
        rt_app.QMediaPlayer = FakeMediaPlayer
        rt_app.QVideoWidget = FakeVideoWidget
        rt_app.MULTIMEDIA_AVAILABLE = True
        rt_app.list_voices = lambda: []
        rt_app.get_backend_status = _backend_status
        rt_app.get_backend_name = lambda: "minicpm"
        rt_app.get_qwen_remote_config = lambda: {
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
        }
        rt_app.set_qwen_remote_config = lambda **_kwargs: None

        with tempfile.TemporaryDirectory() as session_dir_str:
            session = _demo_session(Path(session_dir_str))

            window = rt_app.OmniChatWindow(_default_settings(destination))
            window.resize(1360, 960)
            window.show()
            tabs = window.centralWidget()
            if tabs is not None:
                tabs.setCurrentIndex(1)
            QApplication.processEvents()

            window._set_playback_session(session)
            window._play_selected_session()
            QApplication.processEvents()

            frame_specs = [
                (0, 0),
                (0, 600),
                (1, 1400),
                (2, 3200),
                (3, 3900),
            ]
            frames: list[Image.Image] = []
            for frame_index, (event_index, position_ms) in enumerate(frame_specs):
                _apply_demo_state(window, session, event_index, position_ms)
                frame_path = temp_dir / f"frame_{frame_index:02d}.png"
                frames.append(_capture_frame(window, frame_path).copy())

            screenshot_path = destination / SCREENSHOT_NAME
            frames[2].save(screenshot_path)

            gif_path = destination / GIF_NAME
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=[900, 900, 1100, 1100, 1000],
                loop=0,
                optimize=False,
            )

            window.close()
            QApplication.processEvents()
            return GeneratedPlaybackAssets(screenshot_path=screenshot_path, gif_path=gif_path)
    finally:
        rt_app.AudioPipeline = originals["AudioPipeline"]
        rt_app.QAudioOutput = originals["QAudioOutput"]
        rt_app.QMediaPlayer = originals["QMediaPlayer"]
        rt_app.QVideoWidget = originals["QVideoWidget"]
        rt_app.MULTIMEDIA_AVAILABLE = originals["MULTIMEDIA_AVAILABLE"]
        rt_app.list_voices = originals["list_voices"]
        rt_app.get_backend_status = originals["get_backend_status"]
        rt_app.get_backend_name = originals["get_backend_name"]
        rt_app.get_qwen_remote_config = originals["get_qwen_remote_config"]
        rt_app.set_qwen_remote_config = originals["set_qwen_remote_config"]
        for frame_path in temp_dir.glob("*.png"):
            frame_path.unlink(missing_ok=True)
        temp_dir.rmdir()
        if created_app:
            app.quit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate RT playback screenshot and GIF assets.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "media"),
        help="Directory that will receive the screenshot and GIF.",
    )
    args = parser.parse_args()

    assets = generate_playback_demo_assets(args.output_dir)
    print(f"Wrote screenshot: {assets.screenshot_path}")
    print(f"Wrote gif: {assets.gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())