"""
voice_manager.py — Manage voice reference samples for voice cloning.

Scans a configurable voices directory for .wav files, provides fuzzy matching
when a requested voice name doesn't exactly match, and handles
loading audio at the correct sample rate (16kHz mono).

The voices directory defaults to ./voices/ relative to the project root,
but can be overridden via settings.yaml (audio.voices_dir) or by calling
set_voices_dir() at startup.
"""

import sys
from pathlib import Path
from difflib import get_close_matches
from typing import Optional

import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

VOICES_DIR = _PROJECT_ROOT / "voices"
SAMPLE_RATE = 16000  # MiniCPM-o expects 16kHz input


def set_voices_dir(path: str | Path) -> None:
    """Override the voices directory (called at startup from settings or CLI args)."""
    global VOICES_DIR
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    VOICES_DIR = p


def list_voices() -> list[str]:
    """Return sorted list of available voice names (without .wav extension)."""
    if not VOICES_DIR.exists():
        return []
    return sorted(
        p.stem.replace("_", " ")
        for p in VOICES_DIR.glob("*.wav")
        if p.stem != "README"
    )


def _normalize_name(name: str) -> str:
    """Normalize a voice name for matching: lowercase, strip, collapse spaces."""
    return " ".join(name.lower().strip().split())


def _name_to_filename(name: str) -> str:
    """Convert a display name to filename form: 'Morgan Freeman' → 'morgan_freeman'."""
    return _normalize_name(name).replace(" ", "_")


def get_voice(name: str, fuzzy_threshold: float = 0.6) -> dict:
    """
    Look up a voice sample by name.

    Args:
        name: Requested voice name (e.g., "Morgan Freeman").
        fuzzy_threshold: Cutoff for difflib fuzzy matching (0.0-1.0).

    Returns:
        dict with keys:
            found: bool — whether a voice was found
            name: str — the matched voice name (may differ from input if fuzzy)
            audio: np.ndarray | None — 16kHz mono audio array
            exact: bool — True if exact match, False if fuzzy
            message: str — human-readable status message
    """
    available = list_voices()
    if not available:
        return {
            "found": False,
            "name": name,
            "audio": None,
            "exact": False,
            "message": "No voice samples available. Add .wav files to the voices/ directory.",
        }

    # Normalize for matching
    requested = _normalize_name(name)
    available_normalized = {_normalize_name(v): v for v in available}

    # Exact match
    if requested in available_normalized:
        display_name = available_normalized[requested]
        audio = _load_voice(display_name)
        return {
            "found": True,
            "name": display_name,
            "audio": audio,
            "exact": True,
            "message": f"Voice switched to {display_name}.",
        }

    # Fuzzy match
    matches = get_close_matches(
        requested,
        list(available_normalized.keys()),
        n=1,
        cutoff=fuzzy_threshold,
    )
    if matches:
        matched_normalized = matches[0]
        display_name = available_normalized[matched_normalized]
        audio = _load_voice(display_name)
        return {
            "found": True,
            "name": display_name,
            "audio": audio,
            "exact": False,
            "message": f'I don\'t have "{name}" exactly, but I found "{display_name}". Using that voice.',
        }

    # No match
    voice_list = ", ".join(available)
    return {
        "found": False,
        "name": name,
        "audio": None,
        "exact": False,
        "message": f'I don\'t know the voice "{name}". Available voices: {voice_list}',
    }


def _load_voice(display_name: str) -> np.ndarray:
    """Load a voice sample as a 16kHz mono numpy array."""
    import librosa

    filename = _name_to_filename(display_name) + ".wav"
    path = VOICES_DIR / filename
    if not path.exists():
        # Try the display name directly (with spaces as underscores)
        for p in VOICES_DIR.glob("*.wav"):
            if _normalize_name(p.stem.replace("_", " ")) == _normalize_name(display_name):
                path = p
                break

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


def add_voice(name: str, audio_data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Save a new voice sample to the voices/ directory.

    Args:
        name: Voice name (e.g., "Morgan Freeman").
        audio_data: Audio waveform as numpy array.
        sample_rate: Sample rate of the input audio.

    Returns:
        Path to the saved file.
    """
    import soundfile as sf
    import librosa

    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    # Resample to 16kHz if needed
    if sample_rate != SAMPLE_RATE:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    # Ensure mono
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)

    filename = _name_to_filename(name) + ".wav"
    path = VOICES_DIR / filename
    sf.write(str(path), audio_data, SAMPLE_RATE)
    return str(path)


def delete_voice(name: str) -> bool:
    """Delete a voice sample. Returns True if deleted."""
    filename = _name_to_filename(name) + ".wav"
    path = VOICES_DIR / filename
    if path.exists():
        path.unlink()
        return True
    return False


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Voices directory: {VOICES_DIR}")
    print(f"Available voices: {list_voices()}")

    import argparse

    parser = argparse.ArgumentParser(description="Voice manager test")
    parser.add_argument("--lookup", default=None, help="Look up a voice by name")
    args = parser.parse_args()

    if args.lookup:
        result = get_voice(args.lookup)
        print(f"  Found: {result['found']}")
        print(f"  Name: {result['name']}")
        print(f"  Exact: {result['exact']}")
        print(f"  Message: {result['message']}")
        if result["audio"] is not None:
            print(f"  Audio: {result['audio'].shape} samples @ {SAMPLE_RATE}Hz")
