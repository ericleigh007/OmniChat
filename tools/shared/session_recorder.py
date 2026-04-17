"""Session recording helpers for RT conversations and later model comparison."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf


_RECORDING_OFF = "recording was off"
_ARTIFACT_OMITTED = "artifact omitted by recording mode"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _estimate_token_count(text: str) -> int:
    normalized = (text or "").strip()
    if not normalized:
        return 0
    return max(1, round(len(normalized) / 4.0))


def _session_id(prefix: str = "rt") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-session-{stamp}"


def _normalize_audio_array(audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if audio is None:
        return None
    array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return None
    return array


class SessionRecorder:
    """Capture turn metadata and optional per-turn audio artifacts."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        frontend: str,
        session_metadata: Optional[dict[str, Any]] = None,
        recording_enabled: bool = False,
        recording_mode: str = "structured",
        transcript_policy: str = "full",
    ) -> None:
        self._output_root = Path(output_root)
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._frontend = frontend
        self._recording_enabled = bool(recording_enabled)
        self._recording_mode = str(recording_mode or "structured")
        self._transcript_policy = str(transcript_policy or "full")
        self._session_id = _session_id(frontend)
        self._session_dir = self._output_root / self._session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._session_dir / "session.json"
        self._turn_counter = 0
        self._pending_turns: dict[str, dict[str, Any]] = {}
        self._manifest: dict[str, Any] = {
            "session_id": self._session_id,
            "frontend": self._frontend,
            "created_at": _utc_now_iso(),
            "recording_enabled": self._recording_enabled,
            "recording_mode": self._recording_mode,
            "transcript_policy": self._transcript_policy,
            "session": dict(session_metadata or {}),
            "artifacts": {},
            "turns": [],
        }
        self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def session_dir(self) -> Path:
        return self._session_dir

    @property
    def recording_mode(self) -> str:
        return self._recording_mode

    @property
    def transcript_policy(self) -> str:
        return self._transcript_policy

    @property
    def stores_audio_artifacts(self) -> bool:
        return self._recording_enabled and self._recording_mode in {"structured", "structured-video"}

    def set_recording_enabled(self, enabled: bool) -> None:
        self._recording_enabled = bool(enabled)
        self._manifest["recording_enabled"] = self._recording_enabled
        self._write_manifest()

    def set_recording_config(
        self,
        *,
        enabled: Optional[bool] = None,
        recording_mode: Optional[str] = None,
        transcript_policy: Optional[str] = None,
    ) -> None:
        if enabled is not None:
            self._recording_enabled = bool(enabled)
            self._manifest["recording_enabled"] = self._recording_enabled
        if recording_mode is not None:
            self._recording_mode = str(recording_mode or self._recording_mode)
            self._manifest["recording_mode"] = self._recording_mode
        if transcript_policy is not None:
            self._transcript_policy = str(transcript_policy or self._transcript_policy)
            self._manifest["transcript_policy"] = self._transcript_policy
        self._write_manifest()

    def register_session_video(
        self,
        *,
        video_path: str,
        duration_s: Optional[float] = None,
        fps: Optional[int] = None,
        frame_count: Optional[int] = None,
    ) -> None:
        artifacts = self._manifest.setdefault("artifacts", {})
        artifacts["video"] = {
            "path": video_path,
            "duration_s": duration_s,
            "fps": fps,
            "frame_count": frame_count,
        }
        self._write_manifest()

    def start_turn(
        self,
        *,
        request_id: str,
        turn_metadata: dict[str, Any],
        user_audio: Optional[np.ndarray] = None,
        user_sample_rate: int = 16000,
    ) -> None:
        self._turn_counter += 1
        self._pending_turns[request_id] = {
            "turn_index": self._turn_counter,
            "turn_metadata": dict(turn_metadata),
            "user_audio": _normalize_audio_array(user_audio),
            "user_sample_rate": int(user_sample_rate),
            "model_audio_chunks": [],
            "started_at": _utc_now_iso(),
            "recording_enabled": bool(turn_metadata.get("recording_enabled", self._recording_enabled)),
        }

    def append_model_audio(
        self,
        *,
        request_id: str,
        audio_chunk: Optional[np.ndarray],
        sample_rate: int = 24000,
    ) -> None:
        pending = self._pending_turns.get(request_id)
        if pending is None:
            return
        normalized = _normalize_audio_array(audio_chunk)
        if normalized is None:
            return
        pending.setdefault("model_sample_rate", int(sample_rate))
        pending["model_audio_chunks"].append(normalized.copy())

    def complete_turn(
        self,
        *,
        request_id: str,
        response_text: str,
        response_mode: str,
        interrupted: bool = False,
        error: Optional[str] = None,
        first_text_s: Optional[float] = None,
        elapsed_s: Optional[float] = None,
        text_chars: Optional[int] = None,
        text_tokens_est: Optional[int] = None,
        audio_chunks: Optional[int] = None,
    ) -> dict[str, Any] | None:
        pending = self._pending_turns.pop(request_id, None)
        if pending is None:
            return None

        response_text = str(response_text or "")
        stored_prompt_text = _apply_transcript_policy(str(pending["turn_metadata"].get("prompt_text") or ""), self._transcript_policy)
        stored_response_text = _apply_transcript_policy(response_text, self._transcript_policy)
        user_audio = pending.get("user_audio")
        model_chunks = pending.get("model_audio_chunks") or []
        model_audio = np.concatenate(model_chunks).astype(np.float32, copy=False) if model_chunks else None
        modality = str(pending["turn_metadata"].get("modality") or "text")
        recording_enabled = bool(pending.get("recording_enabled", False))
        turn_index = int(pending["turn_index"])

        user_audio_path: str | None = None
        model_audio_path: str | None = None
        if modality == "audio":
            if self.stores_audio_artifacts:
                if user_audio is not None:
                    user_path = self._session_dir / f"turn_{turn_index:03d}_user.mp3"
                    self._write_audio_mp3(user_path, user_audio, pending.get("user_sample_rate", 16000))
                    user_audio_path = user_path.name
                if model_audio is not None:
                    model_path = self._session_dir / f"turn_{turn_index:03d}_model.mp3"
                    self._write_audio_mp3(model_path, model_audio, pending.get("model_sample_rate", 24000))
                    model_audio_path = model_path.name
            elif recording_enabled:
                user_audio_path = _ARTIFACT_OMITTED
                model_audio_path = _ARTIFACT_OMITTED
            else:
                user_audio_path = _RECORDING_OFF
                model_audio_path = _RECORDING_OFF

        response_chars = int(text_chars if text_chars is not None else len(response_text))
        response_tokens = int(text_tokens_est if text_tokens_est is not None else _estimate_token_count(response_text))
        output_samples = int(model_audio.size) if model_audio is not None else 0
        output_sample_rate = int(pending.get("model_sample_rate", 24000)) if model_audio is not None else None
        output_duration_s = float(output_samples / output_sample_rate) if model_audio is not None and output_sample_rate else 0.0

        turn_record = {
            "turn_index": turn_index,
            "request_id": request_id,
            "started_at": pending.get("started_at"),
            "completed_at": _utc_now_iso(),
            "recording_enabled": recording_enabled,
            "modality": modality,
            "model": dict(pending["turn_metadata"].get("model") or {}),
            "voice": dict(pending["turn_metadata"].get("voice") or {}),
            "prompt": {
                "text": stored_prompt_text,
                "chars": int(pending["turn_metadata"].get("prompt_chars") or 0),
                "tokens_est": int(pending["turn_metadata"].get("prompt_tokens_est") or 0),
                "audio_samples": int(user_audio.size) if user_audio is not None else 0,
                "audio_duration_s": float((user_audio.size / pending.get("user_sample_rate", 16000)) if user_audio is not None else 0.0),
                "audio_path": user_audio_path,
            },
            "response": {
                "text": stored_response_text,
                "mode": response_mode,
                "chars": response_chars,
                "tokens_est": response_tokens,
                "audio_chunks": int(audio_chunks if audio_chunks is not None else len(model_chunks)),
                "audio_samples": output_samples,
                "audio_duration_s": output_duration_s,
                "audio_path": model_audio_path,
            },
            "timing": {
                "prompt_offset_s": pending["turn_metadata"].get("prompt_offset_s"),
                "first_text_s": first_text_s,
                "elapsed_s": elapsed_s,
                "first_text_offset_s": _relative_offset(pending["turn_metadata"].get("prompt_offset_s"), first_text_s),
                "completed_offset_s": _relative_offset(pending["turn_metadata"].get("prompt_offset_s"), elapsed_s),
            },
            "interrupted": bool(interrupted),
            "error": error,
        }
        self._manifest["turns"].append(turn_record)
        self._write_manifest()
        return turn_record

    def _write_manifest(self) -> None:
        temp_path = self._manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(self._manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(self._manifest_path)

    def _write_audio_mp3(self, path: Path, audio: np.ndarray, sample_rate: int) -> None:
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


def _apply_transcript_policy(text: str, policy: str) -> str | None:
    normalized = str(policy or "full").strip().lower()
    if normalized == "omitted":
        return None
    if normalized == "redacted":
        return "[redacted]" if text else ""
    return text


def _relative_offset(base_s: Any, delta_s: Any) -> float | None:
    try:
        if base_s is None or delta_s is None:
            return None
        return float(base_s) + float(delta_s)
    except Exception:
        return None