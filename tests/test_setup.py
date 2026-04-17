from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_setup_module():
    setup_path = Path(__file__).resolve().parent.parent / "setup.py"
    spec = importlib.util.spec_from_file_location("omnichat_setup", setup_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cuda_pytorch_satisfied_accepts_local_cuda_build(monkeypatch):
    setup_mod = _load_setup_module()

    versions = {
        "torch": "2.8.0+cu129",
        "torchaudio": "2.8.0+cu129",
    }

    monkeypatch.setattr(setup_mod, "_installed_version", lambda name: versions.get(name))

    assert setup_mod.cuda_pytorch_satisfied() is True


def test_requirements_satisfied_accepts_matching_versions(monkeypatch, tmp_path):
    setup_mod = _load_setup_module()
    req_path = tmp_path / "requirements.txt"
    req_path.write_text("transformers>=4.57.3\ntorch<=2.8.0\naccelerate\n", encoding="utf-8")

    versions = {
        "transformers": "4.57.3",
        "torch": "2.8.0+cu129",
        "accelerate": "1.10.1",
    }

    monkeypatch.setattr(setup_mod, "_installed_version", lambda name: versions.get(name))

    ready, detail = setup_mod.requirements_satisfied(req_path)

    assert ready is True
    assert detail == "requirements satisfied"


def test_requirements_satisfied_reports_missing_package(monkeypatch, tmp_path):
    setup_mod = _load_setup_module()
    req_path = tmp_path / "requirements.txt"
    req_path.write_text("transformers>=4.57.3\nPySide6>=6.6\n", encoding="utf-8")

    versions = {
        "transformers": "4.57.3",
    }

    monkeypatch.setattr(setup_mod, "_installed_version", lambda name: versions.get(name))

    ready, detail = setup_mod.requirements_satisfied(req_path)

    assert ready is False
    assert detail == "Missing package: PySide6"