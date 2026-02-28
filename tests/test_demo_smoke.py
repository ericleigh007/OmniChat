"""Pytest smoke test for the OmniChat demo -- runs headless as a regression test.

Run with:
    python -m pytest tests/test_demo_smoke.py -v -s
"""

import pytest

# Every test in this module needs the GPU
pytestmark = pytest.mark.gpu


class TestDemoSmoke:
    """Run the full demo in headless mode and check for hard failures."""

    def test_demo_headless(self, loaded_model, tmp_path):
        """Run all 6 demo acts in headless mode."""
        from demos.run_demo import DemoContext, run_all_acts

        ctx = DemoContext(
            headless=True,
            no_audio=True,
            strict=False,
            output_dir=str(tmp_path),
        )
        results = run_all_acts(ctx)

        # Every act should have run
        assert len(results) == 6, f"Expected 6 acts, got {len(results)}"

        # No hard failures (soft failures like voice cloning are OK)
        hard_failures = [r for r in results if not r.passed and not r.soft_fail]
        if hard_failures:
            details = [(r.act_num, r.title, r.detail) for r in hard_failures]
            pytest.fail(f"Demo hard failures: {details}")
