"""Smoke tests for rt_main.py profile selection and startup wiring."""

from __future__ import annotations

import types

import pytest

import rt_main


class FakeSignal:
    def __init__(self):
        self._callbacks = []

    def connect(self, callback):
        self._callbacks.append(callback)

    def emit(self, *args, **kwargs):
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class FakeApp:
    def __init__(self, _args):
        self._main_window = None
        self.icon = None

    def setApplicationName(self, _name):
        return None

    def setStyle(self, _style):
        return None

    def setWindowIcon(self, icon):
        self.icon = icon

    def processEvents(self):
        return None

    def exec(self):
        return 0


class FakeLabel:
    def __init__(self, text=""):
        self.text = text

    def setAlignment(self, _value):
        return None

    def setFont(self, _value):
        return None

    def setFixedSize(self, *_value):
        return None

    def setWindowTitle(self, _value):
        return None

    def setStyleSheet(self, _value):
        return None

    def setWindowFlags(self, _value):
        return None

    def show(self):
        return None

    def close(self):
        return None

    def setText(self, value):
        self.text = value


class FakeWindow:
    def __init__(self, settings):
        self.settings = settings
        self.shown = False

    def show(self):
        self.shown = True


class FakeLoader:
    instances = []

    def __init__(self, *, backend: str, model_label: str):
        self.backend = backend
        self.model_label = model_label
        self.progress = FakeSignal()
        self.finished_ok = FakeSignal()
        self.finished_err = FakeSignal()
        FakeLoader.instances.append(self)

    def start(self):
        self.progress.emit(f"booting {self.model_label}")
        self.finished_ok.emit()


@pytest.fixture(autouse=True)
def reset_loader_instances():
    FakeLoader.instances.clear()
    yield
    FakeLoader.instances.clear()


def test_rt_main_lists_models(monkeypatch, capsys):
    monkeypatch.setattr(rt_main.sys, "argv", ["rt_main.py", "--list-models"])
    monkeypatch.setattr(
        rt_main,
        "list_model_profiles",
        lambda: {
            "minicpm_local": {
                "display_name": "MiniCPM-o 4.5 Local",
                "backend": "minicpm",
                "name": "openbmb/MiniCPM-o-4_5",
            },
            "qwen3_omni_wsl": {
                "display_name": "Qwen3 Omni WSL",
                "backend": "qwen_remote",
                "name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            },
        },
    )

    rt_main.main()
    out = capsys.readouterr().out
    assert "minicpm_local: MiniCPM-o 4.5 Local [minicpm] openbmb/MiniCPM-o-4_5" in out
    assert "qwen3_omni_wsl: Qwen3 Omni WSL [qwen_remote] Qwen/Qwen3-Omni-30B-A3B-Instruct" in out


@pytest.mark.parametrize(
    ("profile_id", "resolved_settings", "expected_backend"),
    [
        (
            "minicpm_local",
            {
                "audio": {"voices_dir": "voices"},
                "model": {
                    "backend": "minicpm",
                    "quantization": "int4",
                    "auto_update": False,
                    "display_name": "MiniCPM Local",
                    "name": "openbmb/MiniCPM-o-4_5",
                },
            },
            "minicpm",
        ),
        (
            "qwen3_omni_wsl",
            {
                "audio": {"voices_dir": "voices"},
                "model": {
                    "backend": "qwen_remote",
                    "quantization": "none",
                    "auto_update": True,
                    "display_name": "Qwen Remote",
                    "name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                    "remote": {
                        "base_url": "http://127.0.0.1:8000/v1",
                        "api_key": "token",
                        "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
                        "timeout_s": 45.0,
                        "endpoints": {"health": "/health", "models": "models", "chat_completions": "chat/completions", "responses": "responses", "realtime": "realtime"},
                    },
                },
            },
            "qwen_remote",
        ),
        (
            "gemma4_ssize_llamacpp_local",
            {
                "audio": {"voices_dir": "voices"},
                "model": {
                    "backend": "gemma_llamacpp",
                    "quantization": "none",
                    "auto_update": False,
                    "display_name": "Gemma Local",
                    "name": "Gemma 4 S-Size",
                    "llama_cpp": {
                        "llama_root": "C:/tmp/llama.cpp",
                        "cli_path": None,
                        "model_path": "D:/OmniChatModels/gemma4/model.gguf",
                        "mmproj_path": "D:/OmniChatModels/gemma4/mmproj.gguf",
                        "n_gpu_layers": 60,
                        "flash_attn": False,
                        "context_length": 4096,
                        "use_jinja": True,
                        "speech_backend": "none",
                    },
                },
            },
            "gemma_llamacpp",
        ),
        (
            "gemma4_e4b_transformers_local",
            {
                "audio": {"voices_dir": "voices"},
                "model": {
                    "backend": "gemma_transformers",
                    "quantization": "none",
                    "auto_update": False,
                    "display_name": "Gemma 4 E4B IT Local",
                    "name": "Gemma 4 E4B IT",
                    "checkpoint": "D:/OmniChatModels/gemma4-e4b-it-official/hf",
                    "device_map": "auto",
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                    "speech_backend": "none",
                    "use_audio_in_video": True,
                    "video_backend": "pyav",
                    "local_files_only": True,
                },
            },
            "gemma_transformers",
        ),
    ],
)
def test_rt_main_smoke_starts_selected_profile(monkeypatch, profile_id, resolved_settings, expected_backend):
    calls = {
        "load_settings": None,
        "voices_dir": None,
        "set_backend": [],
        "set_quantization": [],
        "set_auto_update": [],
        "set_qwen_remote_config": [],
        "set_gemma_transformers_config": [],
        "set_gemma_llamacpp_config": [],
    }

    monkeypatch.setattr(rt_main.sys, "argv", ["rt_main.py", "--model-profile", profile_id])
    monkeypatch.setattr(rt_main, "QApplication", FakeApp)
    monkeypatch.setattr(rt_main, "QLabel", FakeLabel)
    monkeypatch.setattr(rt_main, "QIcon", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rt_main, "ModelLoadThread", FakeLoader)
    monkeypatch.setattr(
        rt_main,
        "load_settings",
        lambda model_profile=None: calls.__setitem__("load_settings", model_profile) or resolved_settings,
    )
    monkeypatch.setattr("tools.audio.voice_manager.set_voices_dir", lambda value: calls.__setitem__("voices_dir", value))
    monkeypatch.setattr("tools.model.model_manager.set_backend", lambda value: calls["set_backend"].append(value))
    monkeypatch.setattr("tools.model.model_manager.set_quantization", lambda value: calls["set_quantization"].append(value))
    monkeypatch.setattr("tools.model.model_manager.set_auto_update", lambda value: calls["set_auto_update"].append(value))
    monkeypatch.setattr("tools.model.model_manager.set_qwen_remote_config", lambda **kwargs: calls["set_qwen_remote_config"].append(kwargs))
    monkeypatch.setattr("tools.model.model_manager.set_gemma_transformers_config", lambda **kwargs: calls["set_gemma_transformers_config"].append(kwargs))
    monkeypatch.setattr("tools.model.model_manager.set_gemma_llamacpp_config", lambda **kwargs: calls["set_gemma_llamacpp_config"].append(kwargs))
    monkeypatch.setattr("rt_app.OmniChatWindow", FakeWindow)

    exit_codes = []
    monkeypatch.setattr(rt_main.sys, "exit", lambda code=0: exit_codes.append(code))

    rt_main.main()

    assert calls["load_settings"] == profile_id
    assert calls["voices_dir"] == "voices"
    assert calls["set_backend"] == [expected_backend]
    assert calls["set_quantization"] == [resolved_settings["model"]["quantization"]]
    assert calls["set_auto_update"] == [resolved_settings["model"]["auto_update"]]
    assert FakeLoader.instances[-1].backend == expected_backend
    assert FakeLoader.instances[-1].model_label == resolved_settings["model"]["display_name"]
    assert exit_codes == [0]

    if expected_backend == "qwen_remote":
        assert calls["set_qwen_remote_config"] == [resolved_settings["model"]["remote"]]
    else:
        assert calls["set_qwen_remote_config"] == []

    if expected_backend == "gemma_transformers":
        assert calls["set_gemma_transformers_config"] == [{
            "checkpoint": resolved_settings["model"].get("checkpoint"),
            "device_map": resolved_settings["model"].get("device_map"),
            "torch_dtype": resolved_settings["model"].get("torch_dtype"),
            "attn_implementation": resolved_settings["model"].get("attn_implementation"),
            "speech_backend": resolved_settings["model"].get("speech_backend"),
            "use_audio_in_video": resolved_settings["model"].get("use_audio_in_video"),
            "local_files_only": resolved_settings["model"].get("local_files_only", not resolved_settings["model"].get("auto_update", True)),
            "video_backend": resolved_settings["model"].get("video_backend"),
        }]
    else:
        assert calls["set_gemma_transformers_config"] == []

    if expected_backend == "gemma_llamacpp":
        assert calls["set_gemma_llamacpp_config"] == [resolved_settings["model"]["llama_cpp"] | {"name": resolved_settings["model"]["name"]}]
    else:
        assert calls["set_gemma_llamacpp_config"] == []


def test_rt_main_emits_startup_console_milestones(monkeypatch):
    messages = []

    monkeypatch.setattr(rt_main.sys, "argv", ["rt_main.py", "--model-profile", "minicpm_local"])
    monkeypatch.setattr(rt_main, "QApplication", FakeApp)
    monkeypatch.setattr(rt_main, "QLabel", FakeLabel)
    monkeypatch.setattr(rt_main, "QIcon", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rt_main, "ModelLoadThread", FakeLoader)
    monkeypatch.setattr(rt_main, "_console_startup", lambda message: messages.append(message))
    monkeypatch.setattr(
        rt_main,
        "load_settings",
        lambda model_profile=None: {
            "active_model_profile": "minicpm_local",
            "model_profile": "minicpm_local",
            "audio": {"voices_dir": "voices"},
            "model": {
                "backend": "minicpm",
                "quantization": "none",
                "auto_update": False,
                "display_name": "MiniCPM Local",
                "name": "openbmb/MiniCPM-o-4_5",
            },
        },
    )
    monkeypatch.setattr("tools.audio.voice_manager.set_voices_dir", lambda _value: None)
    monkeypatch.setattr("tools.model.model_manager.set_backend", lambda _value: None)
    monkeypatch.setattr("tools.model.model_manager.set_quantization", lambda _value: None)
    monkeypatch.setattr("tools.model.model_manager.set_auto_update", lambda _value: None)
    monkeypatch.setattr("rt_app.OmniChatWindow", FakeWindow)
    monkeypatch.setattr(rt_main.sys, "exit", lambda code=0: None)

    rt_main.main()

    assert any("Desktop startup requested." in message for message in messages)
    assert any("Resolved profile=minicpm_local backend=minicpm speech_backend=native." in message for message in messages)
    assert any("Background backend initialization thread started." in message for message in messages)
    assert any("Desktop window shown. The app should now be ready." in message for message in messages)


def test_rt_main_uses_default_gemma_profile_without_cli_override(monkeypatch):
    calls = {
        "load_settings": None,
        "voices_dir": None,
        "set_backend": [],
        "set_quantization": [],
        "set_auto_update": [],
        "set_gemma_transformers_config": [],
    }

    resolved_settings = {
        "active_model_profile": "gemma4_e4b_transformers_mincpm_tts",
        "model_profile": "gemma4_e4b_transformers_mincpm_tts",
        "audio": {"voices_dir": "voices"},
        "model": {
            "backend": "gemma_transformers",
            "quantization": "none",
            "auto_update": False,
            "display_name": "Gemma 4 E4B IT Local + MiniCPM TTS",
            "name": "Gemma 4 E4B IT",
            "checkpoint": "D:/OmniChatModels/gemma4-e4b-it-official/hf",
            "device_map": "auto",
            "torch_dtype": "bfloat16",
            "attn_implementation": "sdpa",
            "speech_backend": "minicpm_streaming",
            "use_audio_in_video": True,
            "video_backend": "pyav",
            "local_files_only": True,
        },
    }

    monkeypatch.setattr(rt_main.sys, "argv", ["rt_main.py"])
    monkeypatch.setattr(rt_main, "QApplication", FakeApp)
    monkeypatch.setattr(rt_main, "QLabel", FakeLabel)
    monkeypatch.setattr(rt_main, "QIcon", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(rt_main, "ModelLoadThread", FakeLoader)
    monkeypatch.setattr(
        rt_main,
        "load_settings",
        lambda model_profile=None: calls.__setitem__("load_settings", model_profile) or resolved_settings,
    )
    monkeypatch.setattr("tools.audio.voice_manager.set_voices_dir", lambda value: calls.__setitem__("voices_dir", value))
    monkeypatch.setattr("tools.model.model_manager.set_backend", lambda value: calls["set_backend"].append(value))
    monkeypatch.setattr("tools.model.model_manager.set_quantization", lambda value: calls["set_quantization"].append(value))
    monkeypatch.setattr("tools.model.model_manager.set_auto_update", lambda value: calls["set_auto_update"].append(value))
    monkeypatch.setattr("tools.model.model_manager.set_gemma_transformers_config", lambda **kwargs: calls["set_gemma_transformers_config"].append(kwargs))
    monkeypatch.setattr("rt_app.OmniChatWindow", FakeWindow)

    exit_codes = []
    monkeypatch.setattr(rt_main.sys, "exit", lambda code=0: exit_codes.append(code))

    rt_main.main()

    assert calls["load_settings"] is None
    assert calls["voices_dir"] == "voices"
    assert calls["set_backend"] == ["gemma_transformers"]
    assert calls["set_quantization"] == ["none"]
    assert calls["set_auto_update"] == [False]
    assert calls["set_gemma_transformers_config"] == [{
        "checkpoint": "D:/OmniChatModels/gemma4-e4b-it-official/hf",
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "speech_backend": "minicpm_streaming",
        "use_audio_in_video": True,
        "video_backend": "pyav",
        "local_files_only": True,
    }]
    assert FakeLoader.instances[-1].backend == "gemma_transformers"
    assert FakeLoader.instances[-1].model_label == "Gemma 4 E4B IT Local + MiniCPM TTS"
    assert exit_codes == [0]