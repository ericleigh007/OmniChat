from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.shared.session_recorder import SessionRecorder


def test_session_recorder_writes_text_only_turn_manifest(tmp_path):
    recorder = SessionRecorder(
        output_root=tmp_path,
        frontend="rt",
        session_metadata={"model": {"profile": "minicpm_local"}},
        recording_enabled=False,
    )

    recorder.start_turn(
        request_id="req-1",
        turn_metadata={
            "modality": "text",
            "recording_enabled": False,
            "prompt_text": "Hello",
            "prompt_chars": 5,
            "prompt_tokens_est": 1,
            "model": {"profile": "minicpm_local"},
            "voice": {"mode": "default", "name": "default voice"},
        },
    )
    recorder.complete_turn(
        request_id="req-1",
        response_text="World",
        response_mode="text",
        elapsed_s=1.25,
        first_text_s=0.5,
    )

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    turn = manifest["turns"][0]
    assert turn["modality"] == "text"
    assert turn["prompt"]["text"] == "Hello"
    assert turn["response"]["text"] == "World"
    assert turn["prompt"]["audio_path"] is None
    assert turn["response"]["audio_path"] is None
    assert turn["timing"]["prompt_offset_s"] is None
    assert turn["timing"]["first_text_offset_s"] is None
    assert turn["timing"]["completed_offset_s"] is None


def test_session_recorder_marks_audio_paths_when_recording_off(tmp_path):
    recorder = SessionRecorder(output_root=tmp_path, frontend="rt", recording_enabled=False)
    user_audio = np.zeros(16000, dtype=np.float32)

    recorder.start_turn(
        request_id="req-2",
        turn_metadata={
            "modality": "audio",
            "recording_enabled": False,
            "prompt_text": "[voice input]",
            "prompt_chars": 13,
            "prompt_tokens_est": 3,
            "model": {},
            "voice": {},
        },
        user_audio=user_audio,
        user_sample_rate=16000,
    )
    recorder.append_model_audio(request_id="req-2", audio_chunk=np.ones(2400, dtype=np.float32), sample_rate=24000)
    recorder.complete_turn(request_id="req-2", response_text="Hi", response_mode="audio")

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    turn = manifest["turns"][0]
    assert turn["prompt"]["audio_path"] == "recording was off"
    assert turn["response"]["audio_path"] == "recording was off"


def test_session_recorder_writes_mp3_paths_when_recording_enabled(tmp_path, monkeypatch):
    recorder = SessionRecorder(output_root=tmp_path, frontend="rt", recording_enabled=True)

    written_paths: list[Path] = []

    def _fake_write_audio_mp3(path: Path, audio: np.ndarray, sample_rate: int):
        written_paths.append(path)
        path.write_bytes(b"mp3")

    monkeypatch.setattr(recorder, "_write_audio_mp3", _fake_write_audio_mp3)

    recorder.start_turn(
        request_id="req-3",
        turn_metadata={
            "modality": "audio",
            "recording_enabled": True,
            "prompt_text": "[voice input]",
            "prompt_chars": 13,
            "prompt_tokens_est": 3,
            "model": {},
            "voice": {},
        },
        user_audio=np.zeros(16000, dtype=np.float32),
        user_sample_rate=16000,
    )
    recorder.append_model_audio(request_id="req-3", audio_chunk=np.ones(2400, dtype=np.float32), sample_rate=24000)
    recorder.complete_turn(request_id="req-3", response_text="Hi", response_mode="audio")

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    turn = manifest["turns"][0]
    assert turn["prompt"]["audio_path"] == "turn_001_user.mp3"
    assert turn["response"]["audio_path"] == "turn_001_model.mp3"
    assert len(written_paths) == 2


def test_session_recorder_omits_audio_artifacts_in_video_mode_and_registers_video(tmp_path):
    recorder = SessionRecorder(
        output_root=tmp_path,
        frontend="rt",
        recording_enabled=True,
        recording_mode="video",
        transcript_policy="redacted",
    )

    recorder.start_turn(
        request_id="req-4",
        turn_metadata={
            "modality": "audio",
            "recording_enabled": True,
            "prompt_text": "secret prompt",
            "prompt_chars": 13,
            "prompt_tokens_est": 3,
            "model": {},
            "voice": {},
        },
        user_audio=np.zeros(16000, dtype=np.float32),
        user_sample_rate=16000,
    )
    recorder.append_model_audio(request_id="req-4", audio_chunk=np.ones(2400, dtype=np.float32), sample_rate=24000)
    recorder.complete_turn(request_id="req-4", response_text="secret response", response_mode="audio")
    recorder.register_session_video(video_path="session.mp4", duration_s=4.0, fps=8, frame_count=32)

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    turn = manifest["turns"][0]
    assert manifest["recording_mode"] == "video"
    assert manifest["transcript_policy"] == "redacted"
    assert turn["prompt"]["text"] == "[redacted]"
    assert turn["response"]["text"] == "[redacted]"
    assert turn["prompt"]["audio_path"] == "artifact omitted by recording mode"
    assert turn["response"]["audio_path"] == "artifact omitted by recording mode"
    assert manifest["artifacts"]["video"]["path"] == "session.mp4"


def test_session_recorder_records_relative_timing_offsets(tmp_path):
    recorder = SessionRecorder(output_root=tmp_path, frontend="rt", recording_enabled=True)

    recorder.start_turn(
        request_id="req-5",
        turn_metadata={
            "modality": "text",
            "recording_enabled": True,
            "prompt_text": "Hello",
            "prompt_offset_s": 1.25,
            "prompt_chars": 5,
            "prompt_tokens_est": 1,
            "model": {},
            "voice": {},
        },
    )
    recorder.complete_turn(
        request_id="req-5",
        response_text="World",
        response_mode="text",
        first_text_s=0.5,
        elapsed_s=1.75,
    )

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    timing = manifest["turns"][0]["timing"]
    assert timing["prompt_offset_s"] == 1.25
    assert timing["first_text_offset_s"] == 1.75
    assert timing["completed_offset_s"] == 3.0
