"""Live app-borne full demo verification for the PySide6 desktop path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "outputs" / "debug" / "rt_full_demo_probe.py"


def test_rt_app_full_demo_mtp_profile(tmp_path):
    output_dir = tmp_path / "rt_app_full_demo"
    output_path = output_dir / "rt_full_demo_probe.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["PYTHONIOENCODING"] = "utf-8"

    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--profile",
            "gemma4_e4b_transformers_mtp_mincpm_tts",
            "--output-dir",
            str(output_dir),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
    )

    assert completed.returncode == 0, (
        f"RT full demo probe failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
    assert output_path.exists(), f"Probe did not write {output_path}"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["overall_passed"] is True
    assert payload["passed"] == 7
    assert payload["failed"] == 0
    assert payload.get("runtime_status", {}).get("last_mtp_used") is True
    assert payload.get("runtime_status", {}).get("assistant_loaded") is True
    for path in payload["screenshots"].values():
        assert Path(path).exists()
