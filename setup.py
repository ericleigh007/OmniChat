"""
setup.py — OmniChat first-run setup (idempotent, safe to re-run).

1. Creates or reuses the repository-local .venv
2. Installs CUDA-enabled PyTorch in that .venv
3. Installs Python packages from requirements.txt
4. Downloads MiniCPM-o 4.5 model (cached by HuggingFace)
5. Verifies CUDA + GPU availability
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
MODEL_NAME = "openbmb/MiniCPM-o-4_5"
SUPPORTED_PYTHON_MINORS = ("3.12", "3.11", "3.10")
CUDA_TORCH_VERSION = "2.8.0"
CUDA_WHEEL_INDEX = "https://download.pytorch.org/whl/cu129"


def err(msg):
    print(f"  FAIL  {msg}")


def venv_dir():
    return BASE_DIR / ".venv"


def venv_python():
    if os.name == "nt":
        return venv_dir() / "Scripts" / "python.exe"
    return venv_dir() / "bin" / "python"


def in_repo_venv():
    try:
        return Path(sys.executable).resolve() == venv_python().resolve()
    except FileNotFoundError:
        return False


def run_checked(cmd, *, capture_output=False):
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or f"command failed: {' '.join(str(part) for part in cmd)}"
        raise RuntimeError(detail)
    return result


def find_supported_python():
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if current_version in SUPPORTED_PYTHON_MINORS:
        return [sys.executable]

    candidates = []
    if os.name == "nt":
        for version in SUPPORTED_PYTHON_MINORS:
            launcher = ["py", f"-{version}"]
            result = subprocess.run(
                [*launcher, "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return launcher
            candidates.append(f"py -{version}")
    else:
        for executable in ("python3.12", "python3.11", "python3.10"):
            result = subprocess.run(
                [executable, "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return [executable]
            candidates.append(executable)

    raise RuntimeError(
        "No supported Python interpreter found. Install one of: "
        + ", ".join(SUPPORTED_PYTHON_MINORS)
        + ". Tried: "
        + ", ".join(candidates)
    )


def ensure_local_venv():
    if not venv_python().exists():
        step("Creating repository-local .venv")
        bootstrap_python = find_supported_python()
        run_checked([*bootstrap_python, "-m", "venv", str(venv_dir())])
        ok(f"Created {venv_dir()}")

    if not in_repo_venv():
        step("Re-launching setup inside repository-local .venv")
        os.execv(str(venv_python()), [str(venv_python()), str(BASE_DIR / "setup.py")])


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


def install_cuda_pytorch():
    step("Installing CUDA-enabled PyTorch into .venv")
    run_checked([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_checked(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"torch=={CUDA_TORCH_VERSION}",
            f"torchaudio=={CUDA_TORCH_VERSION}",
            "--index-url",
            CUDA_WHEEL_INDEX,
        ]
    )
    ok(f"Installed torch/torchaudio {CUDA_TORCH_VERSION} from {CUDA_WHEEL_INDEX}")


def install_packages():
    step("Installing Python packages")
    req = BASE_DIR / "requirements.txt"
    if not req.exists():
        warn("requirements.txt not found — skipping")
        return

    run_checked([sys.executable, "-m", "pip", "install", "-r", str(req)])
    ok("All packages installed")


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

    try:
        ensure_local_venv()
        create_dirs()
        install_cuda_pytorch()
        install_packages()
        check_cuda()
        download_model()
    except Exception as exc:
        print()
        err(str(exc))
        print("=" * 50)
        print("  Setup failed")
        print("=" * 50)
        sys.exit(1)

    print()
    print("=" * 50)
    print("  Setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
