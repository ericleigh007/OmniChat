"""
extract_voice.py — Extract a voice sample from an MP4 (or any video/audio file).

Takes the first N seconds of audio, converts to 16kHz mono WAV,
and saves it as a voice reference sample ready for voice cloning.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _get_ffmpeg() -> str:
    """Get the ffmpeg binary path (bundled with imageio-ffmpeg)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        # Fall back to system ffmpeg
        return "ffmpeg"


def extract_audio_from_video(
    video_path: str,
    duration: float = 15.0,
    start: float = 0.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Extract audio from a video file and return as a numpy array.

    Args:
        video_path: Path to the video file (MP4, AVI, MOV, MKV, etc.).
        duration: Seconds of audio to extract (default 15s).
        start: Start time in seconds (default 0).
        sample_rate: Target sample rate (default 16000 for voice cloning).

    Returns:
        numpy float32 array of mono audio at the target sample rate.
    """
    ffmpeg = _get_ffmpeg()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            ffmpeg,
            "-y",                       # overwrite
            "-ss", str(start),          # seek to start time
            "-i", str(video_path),      # input file
            "-t", str(duration),        # duration
            "-vn",                      # no video
            "-ac", "1",                 # mono
            "-ar", str(sample_rate),    # target sample rate
            "-acodec", "pcm_s16le",     # 16-bit PCM
            tmp_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        audio_data, sr = sf.read(tmp_path, dtype="float32")

        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=-1)

        return audio_data

    finally:
        Path(tmp_path).unlink(missing_ok=True)


def extract_and_save_voice(
    video_path: str,
    voice_name: str,
    duration: float = 15.0,
    start: float = 0.0,
) -> str:
    """
    Extract audio from a video and save as a voice sample.

    Args:
        video_path: Path to the video file.
        voice_name: Name for the voice (e.g., "Morgan Freeman").
        duration: Seconds of audio to extract.
        start: Start time in seconds.

    Returns:
        Path to the saved voice WAV file.
    """
    from tools.audio.voice_manager import add_voice

    audio = extract_audio_from_video(video_path, duration=duration, start=start)
    return add_voice(voice_name, audio, sample_rate=16000)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract voice sample from video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--name", required=True, help="Voice name")
    parser.add_argument("--duration", type=float, default=15.0, help="Seconds to extract")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    args = parser.parse_args()

    path = extract_and_save_voice(args.video, args.name, args.duration, args.start)
    print(f"Voice '{args.name}' saved to: {path}")
