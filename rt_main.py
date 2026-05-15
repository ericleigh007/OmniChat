"""
rt_main.py -- PySide6 desktop app entry point for OmniChat.

Real-time voice assistant with direct PCM audio (no HLS latency).
Uses the same model, tools, and settings as the Gradio UI (main.py).

Usage:
    .venv/Scripts/python.exe rt_main.py
"""

import sys
import os
import time
from pathlib import Path

# Ensure we run from the project root
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from PySide6.QtWidgets import QApplication, QSplashScreen, QLabel
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont, QIcon

from tools.shared.debug_trace import get_trace_log_path, get_trace_logger
from tools.shared.session import configure_model_runtime, list_model_profiles, load_settings


def _console_startup(message: str) -> None:
    """Print a flush-on-write startup milestone for launcher visibility."""
    stamp = time.strftime("%H:%M:%S")
    print(f"[OmniChat RT {stamp}] {message}", flush=True)


class ModelLoadThread(QThread):
    """Load the model singleton on a background thread."""

    progress = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def __init__(self, *, backend: str, model_label: str):
        super().__init__()
        self._backend = backend
        self._model_label = model_label

    def run(self):
        started_at = time.perf_counter()
        try:
            if self._backend == "qwen_remote":
                self.progress.emit(f"Connecting to remote backend...\n{self._model_label}")
                _console_startup(f"Connecting to remote backend for {self._model_label}.")
            else:
                self.progress.emit(f"Loading local model into memory...\n{self._model_label}")
                _console_startup(f"Loading local model for {self._model_label}.")
            from tools.model.model_manager import get_model
            get_model()
            elapsed_s = time.perf_counter() - started_at
            self.progress.emit(f"Model ready.\n{self._model_label}\n({elapsed_s:.1f}s)")
            _console_startup(f"Backend initialization finished in {elapsed_s:.1f}s.")
            self.finished_ok.emit()
        except Exception as e:
            _console_startup(f"Backend initialization failed: {e}")
            self.finished_err.emit(str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OmniChat RT ΓÇö Real-time desktop voice assistant")
    parser.add_argument("--voices-dir", default=None, help="Path to voice WAV samples directory")
    parser.add_argument("--model-profile", default=None, help="Configured model profile id from args/model_profiles.json")
    parser.add_argument("--list-models", action="store_true", help="List configured model profiles and exit")
    parser.add_argument("--backend", default=None, choices=["minicpm", "qwen_remote", "qwen_transformers", "gemma_transformers", "qwen_llamacpp", "gemma_llamacpp"],
                        help="Model backend: MiniCPM, remote Qwen, local Qwen Transformers, local Gemma Transformers, local Qwen llama.cpp GGUF, or local Gemma llama.cpp GGUF")
    parser.add_argument("--quantization", default=None, choices=["none", "int8", "int4"],
                        help="Model quantization: none (bf16, ~19GB), int8 (~10-12GB), int4 (~11GB)")
    args = parser.parse_args()

    _console_startup("Desktop startup requested.")
    get_trace_logger()
    _console_startup(f"Trace log: {get_trace_log_path()}")

    if args.list_models:
        for profile_id, profile in list_model_profiles().items():
            label = profile.get("display_name", profile_id)
            backend = profile.get("backend", "minicpm")
            model_name = profile.get("name", "")
            print(f"{profile_id}: {label} [{backend}] {model_name}")
        return

    settings = load_settings(model_profile=args.model_profile)
    _console_startup(
        "Settings loaded"
        + (f" for profile {args.model_profile}." if args.model_profile else " using the default profile.")
    )
    resolved_profile = settings.get("active_model_profile") or settings.get("model_profile") or "unknown"
    resolved_model = settings.get("model", {})
    resolved_backend = resolved_model.get("backend", "minicpm")
    resolved_speech_backend = "none"
    if resolved_backend in {"qwen_llamacpp", "gemma_llamacpp"}:
        resolved_speech_backend = resolved_model.get("llama_cpp", {}).get("speech_backend", "none")
    elif resolved_backend == "gemma_transformers":
        resolved_speech_backend = resolved_model.get("speech_backend", "none")
    elif resolved_backend in {"minicpm", "qwen_remote", "qwen_transformers"}:
        resolved_speech_backend = "native"
    _console_startup(
        f"Resolved profile={resolved_profile} backend={resolved_backend} speech_backend={resolved_speech_backend}."
    )
    get_trace_logger().info(
        "trace_id=startup stage=app event=profile_resolved requested_profile=%s resolved_profile=%s backend=%s speech_backend=%s",
        args.model_profile or "default",
        resolved_profile,
        resolved_backend,
        resolved_speech_backend,
    )

    # Configure voices directory (CLI overrides settings.yaml)
    from tools.audio.voice_manager import set_voices_dir
    voices_dir = args.voices_dir or settings.get("audio", {}).get("voices_dir", "voices")
    set_voices_dir(voices_dir)

    # Configure model loading (CLI overrides settings.yaml)
    model_settings = settings.get("model", {})
    runtime = configure_model_runtime(
        settings,
        backend_override=args.backend,
        quantization_override=args.quantization,
    )
    backend = runtime["backend"]
    quant = runtime["quantization"]
    _console_startup(
        f"Configured backend={backend} quantization={quant} model={model_settings.get('display_name') or model_settings.get('name', 'unknown')}."
    )

    # Set Windows AppUserModelID so the taskbar shows our icon (not generic Python)
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("omnichat.rt.desktop")
    except Exception:
        pass

    app = QApplication([])
    app.setApplicationName("OmniChat RT")
    app.setStyle("Fusion")

    # Set application-wide icon (taskbar, window title bar, Alt-Tab)
    icon_path = BASE_DIR / "assets" / "omnichat.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Show a simple splash while the model loads
    splash_label = QLabel("OmniChat RT\n\nLoading model...")
    splash_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    splash_label.setFont(QFont("Segoe UI", 14))
    splash_label.setFixedSize(400, 200)
    splash_label.setWindowTitle("OmniChat RT")
    splash_label.setStyleSheet(
        "background-color: #1e1e2e; color: #cdd6f4; border-radius: 10px; padding: 20px;"
    )
    splash_label.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    splash_label.show()
    app.processEvents()
    _console_startup("Splash screen shown. Waiting for backend initialization.")

    # Load model in background
    model_label = model_settings.get("display_name") or model_settings.get("name", "MiniCPM-o 4.5")

    loader = ModelLoadThread(backend=backend, model_label=model_label)

    def on_progress(msg):
        splash_label.setText(f"OmniChat RT\n\n{msg}")
        app.processEvents()
        _console_startup(msg.replace("\n", " | "))

    def on_model_ready():
        splash_label.close()
        # Import here so PySide6 classes are available
        from rt_app import OmniChatWindow
        window = OmniChatWindow(settings)
        window.show()
        app.processEvents()
        # Store reference to prevent GC
        app._main_window = window
        if backend == "qwen_remote":
            _console_startup("Desktop window shown. Remote backend warmup may still continue inside the app.")
        else:
            _console_startup("Desktop window shown. The app should now be ready.")

    def on_model_error(err):
        splash_label.setText(f"OmniChat RT\n\nModel load failed:\n{err}")
        splash_label.setStyleSheet(
            "background-color: #1e1e2e; color: #f38ba8; border-radius: 10px; padding: 20px;"
        )
        _console_startup("Startup stopped at the model-load phase.")

    loader.progress.connect(on_progress)
    loader.finished_ok.connect(on_model_ready)
    loader.finished_err.connect(on_model_error)
    loader.start()
    _console_startup("Background backend initialization thread started.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
