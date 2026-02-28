"""
rt_main.py -- PySide6 desktop app entry point for OmniChat.

Real-time voice assistant with direct PCM audio (no HLS latency).
Uses the same model, tools, and settings as the Gradio UI (main.py).

Usage:
    .venv/Scripts/python.exe rt_main.py
"""

import sys
import os
from pathlib import Path

# Ensure we run from the project root
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

from PySide6.QtWidgets import QApplication, QSplashScreen, QLabel
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont

from tools.shared.session import load_settings


class ModelLoadThread(QThread):
    """Load the model singleton on a background thread."""

    progress = Signal(str)
    finished_ok = Signal()
    finished_err = Signal(str)

    def run(self):
        try:
            self.progress.emit("Loading MiniCPM-o 4.5 model...")
            from tools.model.model_manager import get_model
            get_model()
            self.progress.emit("Model ready.")
            self.finished_ok.emit()
        except Exception as e:
            self.finished_err.emit(str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OmniChat RT — Real-time desktop voice assistant")
    parser.add_argument("--voices-dir", default=None, help="Path to voice WAV samples directory")
    parser.add_argument("--quantization", default=None, choices=["none", "int8", "int4"],
                        help="Model quantization: none (bf16, ~19GB), int8 (~10-12GB), int4 (~11GB)")
    args = parser.parse_args()

    settings = load_settings()

    # Configure voices directory (CLI overrides settings.yaml)
    from tools.audio.voice_manager import set_voices_dir
    voices_dir = args.voices_dir or settings.get("audio", {}).get("voices_dir", "voices")
    set_voices_dir(voices_dir)

    # Configure quantization (CLI overrides settings.yaml)
    from tools.model.model_manager import set_quantization
    quant = args.quantization or settings.get("model", {}).get("quantization", "none")
    set_quantization(quant)

    app = QApplication([])
    app.setApplicationName("OmniChat RT")
    app.setStyle("Fusion")

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

    # Load model in background
    loader = ModelLoadThread()

    def on_progress(msg):
        splash_label.setText(f"OmniChat RT\n\n{msg}")
        app.processEvents()

    def on_model_ready():
        splash_label.close()
        # Import here so PySide6 classes are available
        from rt_app import OmniChatWindow
        window = OmniChatWindow(settings)
        window.show()
        # Store reference to prevent GC
        app._main_window = window

    def on_model_error(err):
        splash_label.setText(f"OmniChat RT\n\nModel load failed:\n{err}")
        splash_label.setStyleSheet(
            "background-color: #1e1e2e; color: #f38ba8; border-radius: 10px; padding: 20px;"
        )

    loader.progress.connect(on_progress)
    loader.finished_ok.connect(on_model_ready)
    loader.finished_err.connect(on_model_error)
    loader.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
