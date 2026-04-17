from __future__ import annotations

from PIL import Image

from tools.capture_rt_playback_assets import generate_playback_demo_assets


def test_generate_playback_demo_assets_creates_screenshot_and_gif(tmp_path):
    assets = generate_playback_demo_assets(tmp_path)

    assert assets.screenshot_path.exists()
    assert assets.gif_path.exists()

    with Image.open(assets.screenshot_path) as screenshot:
        assert screenshot.width >= 1000
        assert screenshot.height >= 700

    with Image.open(assets.gif_path) as animation:
        assert getattr(animation, "is_animated", False) is True
        assert animation.n_frames >= 4