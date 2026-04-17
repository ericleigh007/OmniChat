from __future__ import annotations

import json
import math
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

from tools.shared.session_recorder import _ARTIFACT_OMITTED, _RECORDING_OFF


_VIDEO_AUDIO_SR = 24000
_GAP_AFTER_AUDIO_S = 0.12
_FALLBACK_TEXT_STEP_S = 1.2
_MAX_TEXT_STEP_S = 6.0


@dataclass(slots=True)
class SessionPlaybackManifest:
    manifest_path: Path
    session_dir: Path
    data: dict[str, Any]
    turns: list[dict[str, Any]]
    assistant_label: str
    recording_mode: str
    transcript_policy: str
    video_path: Path | None


@dataclass(slots=True)
class SessionReplayEvent:
    event_index: int
    turn_index: int
    role: str
    text: str
    history: list[tuple[str, str]]
    audio_path: Path | None
    duration_s: float
    description: str


@dataclass(slots=True)
class SessionOverlaySnapshot:
    position_s: float
    turn_index: int
    phase: str
    modality: str
    prompt_tokens_est: int
    response_tokens_est: int
    first_text_s: float | None
    elapsed_s: float | None
    prompt_offset_s: float | None
    first_text_offset_s: float | None
    completed_offset_s: float | None
    response_mode: str
    interrupted: bool
    error: str | None


def discover_session_manifests(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []

    manifests: list[Path] = []
    for candidate in root_path.rglob("*.json"):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("turns"), list):
            manifests.append(candidate)
    return sorted(manifests, key=lambda path: path.stat().st_mtime, reverse=True)


def load_session_manifest(path: str | Path) -> SessionPlaybackManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("turns"), list):
        raise ValueError(f"Invalid session manifest: {manifest_path}")
    turns = [turn for turn in payload.get("turns", []) if isinstance(turn, dict)]
    return SessionPlaybackManifest(
        manifest_path=manifest_path,
        session_dir=manifest_path.parent,
        data=payload,
        turns=turns,
        assistant_label=_assistant_label_from_manifest(payload, turns),
        recording_mode=str(payload.get("recording_mode") or "structured"),
        transcript_policy=str(payload.get("transcript_policy") or "full"),
        video_path=_resolve_video_path(manifest_path.parent, ((payload.get("artifacts") or {}).get("video") or {}).get("path")),
    )


def build_chat_history(session: SessionPlaybackManifest, upto_turn: int | None = None) -> list[tuple[str, str]]:
    history: list[tuple[str, str]] = []
    limit = upto_turn if upto_turn is not None else len(session.turns)
    for turn in session.turns[:limit]:
        prompt_text = str((turn.get("prompt") or {}).get("text") or "")
        if prompt_text:
            history.append(("user", prompt_text))
        response = turn.get("response") or {}
        response_text = str(response.get("text") or "")
        if response_text:
            response_role = "_assistant_hidden" if str(response.get("mode") or "text") == "audio" else "assistant"
            history.append((response_role, response_text))
    return history


def build_replay_events(session: SessionPlaybackManifest) -> list[SessionReplayEvent]:
    history: list[tuple[str, str]] = []
    events: list[SessionReplayEvent] = []
    event_index = 0

    for turn in session.turns:
        turn_index = int(turn.get("turn_index") or (len(events) + 1))
        prompt = turn.get("prompt") or {}
        prompt_text = str(prompt.get("text") or "")
        if prompt_text:
            history.append(("user", prompt_text))
            events.append(
                SessionReplayEvent(
                    event_index=event_index,
                    turn_index=turn_index,
                    role="user",
                    text=prompt_text,
                    history=list(history),
                    audio_path=_resolve_audio_path(session, prompt.get("audio_path")),
                    duration_s=_event_duration(prompt_text, prompt.get("audio_duration_s")),
                    description=f"Turn {turn_index}: user",
                )
            )
            event_index += 1

        response = turn.get("response") or {}
        response_text = str(response.get("text") or "")
        if response_text:
            response_role = "_assistant_hidden" if str(response.get("mode") or "text") == "audio" else "assistant"
            history.append((response_role, response_text))
            events.append(
                SessionReplayEvent(
                    event_index=event_index,
                    turn_index=turn_index,
                    role=response_role,
                    text=response_text,
                    history=list(history),
                    audio_path=_resolve_audio_path(session, response.get("audio_path")),
                    duration_s=_event_duration(response_text, response.get("audio_duration_s")),
                    description=f"Turn {turn_index}: assistant",
                )
            )
            event_index += 1

    return events


def get_overlay_snapshot(session: SessionPlaybackManifest, position_s: float) -> SessionOverlaySnapshot | None:
    turns_with_offsets: list[dict[str, Any]] = []
    for turn in session.turns:
        timing = turn.get("timing") or {}
        prompt_offset = timing.get("prompt_offset_s")
        if prompt_offset is None:
            continue
        turns_with_offsets.append(turn)
    if not turns_with_offsets:
        return None

    pos = max(0.0, float(position_s))
    active_turn = turns_with_offsets[0]
    for turn in turns_with_offsets:
        timing = turn.get("timing") or {}
        prompt_offset = timing.get("prompt_offset_s")
        if prompt_offset is not None and float(prompt_offset) <= pos:
            active_turn = turn
        else:
            break

    timing = active_turn.get("timing") or {}
    prompt = active_turn.get("prompt") or {}
    response = active_turn.get("response") or {}
    prompt_offset = _optional_float(timing.get("prompt_offset_s"))
    first_text_offset = _optional_float(timing.get("first_text_offset_s"))
    completed_offset = _optional_float(timing.get("completed_offset_s"))

    if prompt_offset is None:
        phase = "unknown"
    elif pos < prompt_offset:
        phase = "upcoming"
    elif first_text_offset is None or pos < first_text_offset:
        phase = "waiting"
    elif completed_offset is None or pos < completed_offset:
        phase = "responding"
    else:
        phase = "completed"

    return SessionOverlaySnapshot(
        position_s=pos,
        turn_index=int(active_turn.get("turn_index") or 0),
        phase=phase,
        modality=str(active_turn.get("modality") or "text"),
        prompt_tokens_est=int(prompt.get("tokens_est") or 0),
        response_tokens_est=int(response.get("tokens_est") or 0),
        first_text_s=_optional_float(timing.get("first_text_s")),
        elapsed_s=_optional_float(timing.get("elapsed_s")),
        prompt_offset_s=prompt_offset,
        first_text_offset_s=first_text_offset,
        completed_offset_s=completed_offset,
        response_mode=str(response.get("mode") or "text"),
        interrupted=bool(active_turn.get("interrupted")),
        error=active_turn.get("error"),
    )


def load_audio_file(path: str | Path) -> tuple[np.ndarray, int]:
    audio_path = Path(path)
    try:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    except Exception:
        return _decode_audio_via_ffmpeg(audio_path)
    return _normalize_audio(audio, int(sample_rate))


def stitch_session_audio(session: SessionPlaybackManifest) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    for event in build_replay_events(session):
        if event.audio_path is not None and event.audio_path.exists():
            audio, sample_rate = load_audio_file(event.audio_path)
            if sample_rate != _VIDEO_AUDIO_SR:
                import librosa

                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_VIDEO_AUDIO_SR).astype(np.float32)
            chunks.append(audio.astype(np.float32, copy=False))
            chunks.append(_silence(_GAP_AFTER_AUDIO_S, _VIDEO_AUDIO_SR))
        else:
            chunks.append(_silence(max(0.2, float(event.duration_s)), _VIDEO_AUDIO_SR))
    if not chunks:
        chunks.append(_silence(1.0, _VIDEO_AUDIO_SR))
    return np.concatenate(chunks).astype(np.float32, copy=False), _VIDEO_AUDIO_SR


def export_stitched_audio(session: SessionPlaybackManifest, output_path: str | Path) -> Path:
    destination = Path(output_path)
    audio, sample_rate = stitch_session_audio(session)
    if destination.suffix.lower() == ".wav":
        sf.write(str(destination), audio, sample_rate, format="WAV", subtype="PCM_16")
    else:
        _write_audio_mp3(destination, audio, sample_rate)
    return destination


def render_session_poster(session: SessionPlaybackManifest, output_path: str | Path, *, size: tuple[int, int] = (1280, 720)) -> Path:
    destination = Path(output_path)
    width, height = size
    image = Image.new("RGB", (width, height), (20, 24, 33))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()

    title = session.assistant_label
    metadata = session.data.get("session") or {}
    model = metadata.get("model") or {}
    subtitle = f"{model.get('display_name') or model.get('backend') or 'OmniChat Session'} | {session.data.get('created_at') or ''}"
    draw.text((40, 28), title, fill=(235, 241, 255), font=title_font)
    draw.text((40, 54), subtitle, fill=(152, 163, 179), font=body_font)

    history = build_chat_history(session)
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for role, text in history:
        prefix = "You" if role == "user" else session.assistant_label if role == "assistant" else f"{session.assistant_label} [Spoken Text]"
        color = (220, 228, 240) if role == "user" else (181, 220, 255)
        wrapped = textwrap.wrap(f"{prefix}: {text}", width=max(40, width // 18)) or [f"{prefix}: {text}"]
        for wrapped_line in wrapped:
            lines.append((wrapped_line, color))
        lines.append(("", color))

    y = 100
    line_height = 18
    max_lines = max(10, (height - 140) // line_height)
    for line, color in lines[-max_lines:]:
        draw.text((40, y), line, fill=color, font=body_font)
        y += line_height

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return destination


def export_demo_video(session: SessionPlaybackManifest, output_path: str | Path) -> Path:
    import imageio_ffmpeg

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        poster_path = render_session_poster(session, temp_dir / "poster.png")
        wav_path = temp_dir / "stitched.wav"
        audio, sample_rate = stitch_session_audio(session)
        sf.write(str(wav_path), audio, sample_rate, format="WAV", subtype="PCM_16")
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loop",
                "1",
                "-i",
                str(poster_path),
                "-i",
                str(wav_path),
                "-c:v",
                "libx264",
                "-tune",
                "stillimage",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
    return destination


def _assistant_label_from_manifest(payload: dict[str, Any], turns: list[dict[str, Any]]) -> str:
    model = ((payload.get("session") or {}).get("model") or {}) if isinstance(payload, dict) else {}
    display_name = model.get("display_name") or model.get("name") or model.get("backend") or "OmniChat"
    if turns:
        turn_model = (turns[0].get("model") or {}) if isinstance(turns[0], dict) else {}
        display_name = turn_model.get("display_name") or turn_model.get("name") or display_name
    return f"OmniChat [{display_name}]"


def _resolve_audio_path(session: SessionPlaybackManifest, relative_path: Any) -> Path | None:
    if not relative_path:
        return None
    if str(relative_path) in {_RECORDING_OFF, _ARTIFACT_OMITTED}:
        return None
    candidate = session.session_dir / str(relative_path)
    return candidate if candidate.exists() else None


def _resolve_video_path(session_dir: Path, relative_path: Any) -> Path | None:
    if not relative_path:
        return None
    candidate = session_dir / str(relative_path)
    return candidate if candidate.exists() else candidate


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _event_duration(text: str, audio_duration_s: Any) -> float:
    try:
        duration = float(audio_duration_s or 0.0)
    except Exception:
        duration = 0.0
    if duration > 0:
        return duration
    char_based = max(_FALLBACK_TEXT_STEP_S, min(_MAX_TEXT_STEP_S, len(text or "") / 18.0))
    return float(char_based)


def _normalize_audio(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=-1)
    return array.reshape(-1), int(sample_rate)


def _decode_audio_via_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    import imageio_ffmpeg

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_path = Path(temp_wav.name)
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", str(path), str(temp_path)],
            check=True,
            capture_output=True,
        )
        audio, sample_rate = sf.read(str(temp_path), dtype="float32")
        return _normalize_audio(audio, int(sample_rate))
    finally:
        temp_path.unlink(missing_ok=True)


def _silence(duration_s: float, sample_rate: int) -> np.ndarray:
    frame_count = max(1, int(math.ceil(max(0.0, float(duration_s)) * int(sample_rate))))
    return np.zeros(frame_count, dtype=np.float32)


def _write_audio_mp3(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import imageio_ffmpeg

    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        temp_wav_path = Path(temp_wav.name)
    try:
        sf.write(str(temp_wav_path), np.asarray(audio, dtype=np.float32), int(sample_rate), format="WAV", subtype="PCM_16")
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(temp_wav_path),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        temp_wav_path.unlink(missing_ok=True)