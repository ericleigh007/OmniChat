"""
setup.py — OmniChat first-run setup (idempotent, safe to re-run).

1. Creates directory structure
2. Installs Python packages from requirements.txt
3. Downloads MiniCPM-o 4.5 model (cached by HuggingFace)
4. Verifies CUDA + GPU availability
"""

import sys
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
MODEL_NAME = "openbmb/MiniCPM-o-4_5"


def ok(msg):
    print(f"  [OK]  {msg}")


def warn(msg):
    print(f"  WARN  {msg}")


def step(msg):
    print(f"\n>> {msg}...")


def create_dirs():
    step("Ensuring directory structure")
    dirs = [
        BASE_DIR / "args",
        BASE_DIR / "voices",
        BASE_DIR / "outputs",
        BASE_DIR / ".tmp",
        BASE_DIR / "tools" / "model",
        BASE_DIR / "tools" / "audio",
        BASE_DIR / "tools" / "vision",
        BASE_DIR / "tools" / "output",
        BASE_DIR / "goals",
        BASE_DIR / "context",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    ok("All directories present")


def install_packages():
    step("Installing Python packages")
    req = BASE_DIR / "requirements.txt"
    if not req.exists():
        warn("requirements.txt not found — skipping")
        return

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok("All packages installed")
    else:
        warn(f"pip install had issues:\n{result.stderr[-500:]}")


def check_cuda():
    step("Checking CUDA and GPU")
    try:
        import torch

        if not torch.cuda.is_available():
            warn("CUDA not available — model will run on CPU (very slow)")
            return

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"GPU: {gpu_name} ({vram_gb:.0f} GB VRAM)")

        if vram_gb < 12:
            warn(
                f"GPU has {vram_gb:.0f} GB VRAM. MiniCPM-o 4.5 needs ~19 GB in bf16. "
                "Use int4 quantization (~11 GB): --quantization int4"
            )
        elif vram_gb < 20:
            warn(
                f"GPU has {vram_gb:.0f} GB VRAM. MiniCPM-o 4.5 needs ~19 GB in bf16. "
                "Use int8 quantization (~10-12 GB): --quantization int8"
            )
    except ImportError:
        warn("PyTorch not installed — cannot check GPU")


def download_model():
    step(f"Checking model: {MODEL_NAME}")
    try:
        from huggingface_hub import snapshot_download

        # This is a no-op if already cached
        cache_dir = snapshot_download(
            MODEL_NAME,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.txt", "*.model"],
        )
        ok(f"Model cached at: {cache_dir}")
    except ImportError:
        warn("huggingface_hub not installed — model will download on first run")
    except Exception as e:
        warn(f"Model download issue: {e}")
        print("  The model will download automatically on first launch.")


def main():
    print("=" * 50)
    print("  OmniChat Setup")
    print("=" * 50)

    create_dirs()
    install_packages()
    check_cuda()
    download_model()

    print()
    print("=" * 50)
    print("  Setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
