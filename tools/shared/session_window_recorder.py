from __future__ import annotations

import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QTimer


_VIDEO_AUDIO_SR = 24000


class SessionWindowRecorder:
    def __init__(self, *, output_dir: str | Path, widget, fps: int = 8) -> None:
        self._output_dir = Path(output_dir)
        self._widget = widget
        self._fps = max(1, int(fps))
        self._frames_dir = self._output_dir / "video_frames"
        self._video_path = self._output_dir / "session.mp4"
        self._timer = QTimer(widget)
        self._timer.timeout.connect(self._capture_frame)
        self._started_at = 0.0
        self._frame_index = 0
        self._audio_segments: list[tuple[float, np.ndarray, int]] = []

    @property
    def video_path(self) -> Path:
        return self._video_path

    def start(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.perf_counter()
        self._frame_index = 0
        self._audio_segments.clear()
        self._capture_frame()
        self._timer.start(max(40, int(round(1000 / self._fps))))

    def add_audio_clip(self, audio: np.ndarray | None, *, sample_rate: int, offset_s: float | None = None) -> None:
        if audio is None:
            return
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return
        clip_offset = float(offset_s if offset_s is not None else self.elapsed_s())
        self._audio_segments.append((max(0.0, clip_offset), array.copy(), int(sample_rate)))

    def elapsed_s(self) -> float:
        if self._started_at <= 0:
            return 0.0
        return max(0.0, time.perf_counter() - self._started_at)

    def stop(self) -> dict[str, float | int | str] | None:
        if self._started_at <= 0:
            return None
        self._timer.stop()
        self._capture_frame()
        duration_s = max(self.elapsed_s(), self._frame_index / float(self._fps), self._audio_duration_s())
        if self._frame_index == 0:
            return None
        self._encode_video(duration_s)
        self._started_at = 0.0
        return {
            "video_path": self._video_path.name,
            "duration_s": duration_s,
            "fps": self._fps,
            "frame_count": self._frame_index,
        }

    def _capture_frame(self) -> None:
        if self._widget is None:
            return
        pixmap = self._widget.grab()
        if pixmap.isNull():
            return
        frame_path = self._frames_dir / f"frame_{self._frame_index:06d}.png"
        pixmap.save(str(frame_path), "PNG")
        self._frame_index += 1

    def _audio_duration_s(self) -> float:
        max_duration = 0.0
        for offset_s, audio, sample_rate in self._audio_segments:
            max_duration = max(max_duration, offset_s + (len(audio) / float(sample_rate or 1)))
        return max_duration

    def _encode_video(self, duration_s: float) -> None:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = Path(temp_wav.name)
        try:
            audio = self._mix_audio(max(0.2, float(duration_s)))
            sf.write(str(wav_path), audio, _VIDEO_AUDIO_SR, format="WAV", subtype="PCM_16")
            subprocess.run(
                [
                    ffmpeg_exe,
                    "-y",
                    "-framerate",
                    str(self._fps),
                    "-i",
                    str(self._frames_dir / "frame_%06d.png"),
                    "-i",
                    str(wav_path),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(self._video_path),
                ],
                check=True,
                capture_output=True,
            )
        finally:
            wav_path.unlink(missing_ok=True)

    def _mix_audio(self, duration_s: float) -> np.ndarray:
        total_frames = max(1, int(math.ceil(float(duration_s) * _VIDEO_AUDIO_SR)))
        mixed = np.zeros(total_frames, dtype=np.float32)
        for offset_s, audio, sample_rate in self._audio_segments:
            clip = np.asarray(audio, dtype=np.float32).reshape(-1)
            if int(sample_rate) != _VIDEO_AUDIO_SR:
                import librosa

                clip = librosa.resample(clip, orig_sr=int(sample_rate), target_sr=_VIDEO_AUDIO_SR).astype(np.float32)
            start = max(0, int(round(float(offset_s) * _VIDEO_AUDIO_SR)))
            end = min(total_frames, start + len(clip))
            if end <= start:
                continue
            mixed[start:end] += clip[: end - start]
        return np.clip(mixed, -1.0, 1.0)