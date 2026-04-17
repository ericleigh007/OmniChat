from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.shared import session_playback


def _write_manifest(session_dir: Path) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "session.json"
    payload = {
        "session_id": "rt-session-test",
        "frontend": "rt",
        "created_at": "2026-04-16T22:00:00Z",
        "recording_mode": "structured",
        "transcript_policy": "full",
        "session": {
            "model": {
                "display_name": "Gemma Local",
                "backend": "gemma_llamacpp",
            }
        },
        "artifacts": {},
        "turns": [
            {
                "turn_index": 1,
                "modality": "audio",
                "prompt": {
                    "text": "[voice input]",
                    "tokens_est": 3,
                    "audio_duration_s": 1.0,
                    "audio_path": "turn_001_user.mp3",
                },
                "response": {
                    "text": "Hello there",
                    "mode": "audio",
                    "tokens_est": 4,
                    "audio_duration_s": 1.5,
                    "audio_path": "turn_001_model.mp3",
                },
                "timing": {"prompt_offset_s": 0.5, "first_text_s": 0.4, "elapsed_s": 1.7, "first_text_offset_s": 0.9, "completed_offset_s": 2.2},
            },
            {
                "turn_index": 2,
                "modality": "text",
                "prompt": {"text": "Tell me more", "tokens_est": 3, "audio_duration_s": 0.0, "audio_path": None},
                "response": {"text": "A text reply", "mode": "text", "tokens_est": 3, "audio_duration_s": 0.0, "audio_path": None},
                "timing": {"prompt_offset_s": 2.8, "first_text_s": 0.2, "elapsed_s": 1.1, "first_text_offset_s": 3.0, "completed_offset_s": 3.9},
            },
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    (session_dir / "turn_001_user.mp3").write_bytes(b"user")
    (session_dir / "turn_001_model.mp3").write_bytes(b"model")
    return manifest_path


def test_discover_and_load_session_manifest(tmp_path):
    manifest_path = _write_manifest(tmp_path / "demo-session")

    manifests = session_playback.discover_session_manifests(tmp_path)

    assert manifests == [manifest_path]
    loaded = session_playback.load_session_manifest(manifest_path)
    assert loaded.manifest_path == manifest_path
    assert loaded.assistant_label == "OmniChat [Gemma Local]"
    assert loaded.video_path is None


def test_build_replay_events_and_history(tmp_path):
    manifest = session_playback.load_session_manifest(_write_manifest(tmp_path / "demo-session"))

    history = session_playback.build_chat_history(manifest)
    events = session_playback.build_replay_events(manifest)

    assert history == [
        ("user", "[voice input]"),
        ("_assistant_hidden", "Hello there"),
        ("user", "Tell me more"),
        ("assistant", "A text reply"),
    ]
    assert len(events) == 4
    assert events[0].audio_path == manifest.session_dir / "turn_001_user.mp3"
    assert events[1].role == "_assistant_hidden"
    assert events[-1].duration_s >= 1.0


def test_export_stitched_audio_and_video(tmp_path, monkeypatch):
    manifest = session_playback.load_session_manifest(_write_manifest(tmp_path / "demo-session"))
    monkeypatch.setattr(session_playback, "load_audio_file", lambda _path: (np.ones(24000, dtype=np.float32), 24000))

    written_audio = {}
    monkeypatch.setattr(session_playback, "_write_audio_mp3", lambda path, audio, sample_rate: written_audio.update({"path": Path(path), "samples": len(audio), "sr": sample_rate}) or Path(path).write_bytes(b"mp3"))

    ffmpeg_calls = []

    def _fake_run(command, check, capture_output):
        ffmpeg_calls.append(command)
        Path(command[-1]).write_bytes(b"mp4")
        return None

    monkeypatch.setattr(session_playback.subprocess, "run", _fake_run)

    audio_path = tmp_path / "demo.mp3"
    video_path = tmp_path / "demo.mp4"
    session_playback.export_stitched_audio(manifest, audio_path)
    session_playback.export_demo_video(manifest, video_path)

    assert written_audio["path"] == audio_path
    assert written_audio["sr"] == 24000
    assert video_path.exists()
    assert ffmpeg_calls


def test_load_session_manifest_resolves_video_path(tmp_path):
    manifest_path = _write_manifest(tmp_path / "video-session")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["recording_mode"] = "video"
    payload["artifacts"] = {"video": {"path": "session.mp4", "duration_s": 2.0}}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    (manifest_path.parent / "session.mp4").write_bytes(b"mp4")

    loaded = session_playback.load_session_manifest(manifest_path)

    assert loaded.recording_mode == "video"
    assert loaded.video_path == manifest_path.parent / "session.mp4"


def test_get_overlay_snapshot_uses_manifest_timeline(tmp_path):
    manifest = session_playback.load_session_manifest(_write_manifest(tmp_path / "overlay-session"))

    snapshot_waiting = session_playback.get_overlay_snapshot(manifest, 0.6)
    snapshot_responding = session_playback.get_overlay_snapshot(manifest, 1.4)
    snapshot_second_turn = session_playback.get_overlay_snapshot(manifest, 3.2)

    assert snapshot_waiting is not None
    assert snapshot_waiting.turn_index == 1
    assert snapshot_waiting.phase == "waiting"
    assert snapshot_waiting.prompt_tokens_est == 3
    assert snapshot_waiting.response_tokens_est == 4

    assert snapshot_responding is not None
    assert snapshot_responding.turn_index == 1
    assert snapshot_responding.phase == "responding"
    assert snapshot_responding.completed_offset_s == 2.2

    assert snapshot_second_turn is not None
    assert snapshot_second_turn.turn_index == 2
    assert snapshot_second_turn.phase == "responding"
    assert snapshot_second_turn.response_mode == "text"