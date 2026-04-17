"""Live offscreen desktop verification for local llama.cpp RT GUI profiles.

These tests exercise the real PySide6 desktop app path through the existing
`outputs/debug/qwen_desktop_probe.py` hook. They are intentionally marked as
GPU tests because they require staged local llama.cpp artifacts and a real
CUDA-capable runtime.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs" / "debug" / "qwen_desktop_probe.py"
PROMPT = "Why is the sky blue? Answer in a detailed manner with at least three paragraphs and do not stop mid-sentence."


def _kill_stale_probe_processes() -> None:
    if os.name != "nt":
        return

    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'qwen_desktop_probe.py' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _kill_llama_server() -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/IM", "llama-server.exe", "/F"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        return

    subprocess.run(
        ["pkill", "-f", "llama-server"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def _require_probe_assets(profile: str) -> None:
    from tools.shared.session import load_settings

    try:
        import torch
    except ImportError as exc:
        pytest.skip(f"torch is unavailable: {exc}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available for live llama.cpp GUI verification")

    settings = load_settings(model_profile=profile)
    llama_cpp = settings.get("model", {}).get("llama_cpp", {})
    required_paths = [
        Path(llama_cpp.get("llama_root") or ""),
        Path(llama_cpp.get("model_path") or ""),
        Path(llama_cpp.get("mmproj_path") or ""),
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        pytest.skip(f"Required local llama.cpp artifacts are missing for {profile}: {missing}")


def _run_desktop_probe(profile: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], dict]:
    _require_probe_assets(profile)
    _kill_stale_probe_processes()
    _kill_llama_server()

    output_path = tmp_path / f"{profile}_desktop_probe.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--profile",
            profile,
            "--prompt",
            PROMPT,
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )

    assert completed.returncode == 0, (
        f"Desktop probe failed for {profile}\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )
    assert output_path.exists(), f"Desktop probe did not write {output_path} for {profile}"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    return completed, payload


@pytest.mark.parametrize(
    ("profile", "min_text_len"),
    [
        ("qwen35_27b_llamacpp_local", 1500),
        ("qwen35_27b_llamacpp_local_minicpm_tts", 1800),
        ("gemma4_ssize_llamacpp_local", 400),
        ("gemma4_ssize_llamacpp_mincpm_tts", 400),
    ],
)
def test_rt_llamacpp_gui_profiles_complete_desktop_turn(profile: str, min_text_len: int, tmp_path: Path):
    completed, payload = _run_desktop_probe(profile, tmp_path)

    assert payload["profile"] == profile
    assert payload["finished"] is True
    assert payload["last_entry_role"] == "assistant"
    assert isinstance(payload["last_entry_text"], str) and payload["last_entry_text"].strip()
    assert int(payload["last_entry_len"] or 0) >= min_text_len

    runtime_status = payload.get("runtime_status") or {}
    assert runtime_status.get("server_ready") is True
    assert int(runtime_status.get("last_prompt_tokens_est") or 0) > 0
    assert int(runtime_status.get("last_text_tokens_est") or 0) > 0

    trace_text = f"{completed.stdout}\n{completed.stderr}"
    assert "stage=ui event=generation_started" in trace_text
    assert "stage=ui event=generation_finished" in trace_text
    assert "stage=rt_audio event=generation_progress" in trace_text