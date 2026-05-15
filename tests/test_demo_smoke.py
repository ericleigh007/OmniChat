"""Pytest smoke test for the OmniChat demo -- runs headless as a regression test.

Run with:
    python -m pytest tests/test_demo_smoke.py -v -s
"""

import pytest

# Every test in this module needs the GPU
pytestmark = pytest.mark.gpu


class TestDemoSmoke:
    """Run the full demo in headless mode and check for hard failures."""

    def test_demo_headless(self, tmp_path):
        """Run the full Gemma MTP hybrid demo in headless mode."""
        import subprocess
        import sys

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "demos.run_demo",
                "--model-profile",
                "gemma4_e4b_transformers_mtp_mincpm_tts",
                "--headless",
                "--strict",
                "--output-dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

        assert completed.returncode == 0, (
            f"Demo failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
        assert "Overall: PASS" in completed.stdout
        assert "Total: 7 passed, 0 soft-failed, 0 failed" in completed.stdout
