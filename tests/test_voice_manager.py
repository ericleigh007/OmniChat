"""Tests for voice_manager.py — fuzzy matching, name normalization, CRUD."""

import numpy as np
import pytest

import tools.audio.voice_manager as vm


class TestNameNormalization:
    """Test string normalization helpers."""

    def test_normalize_basic(self):
        assert vm._normalize_name("Morgan Freeman") == "morgan freeman"

    def test_normalize_extra_spaces(self):
        assert vm._normalize_name("  Morgan   Freeman  ") == "morgan freeman"

    def test_normalize_already_lower(self):
        assert vm._normalize_name("bob") == "bob"

    def test_name_to_filename(self):
        assert vm._name_to_filename("Morgan Freeman") == "morgan_freeman"

    def test_name_to_filename_extra_spaces(self):
        assert vm._name_to_filename("  David   Attenborough  ") == "david_attenborough"

    def test_name_to_filename_single_name(self):
        assert vm._name_to_filename("Siri") == "siri"


class TestListVoices:
    """Test voice listing from directory."""

    def test_list_voices_populated(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        voices = vm.list_voices()
        assert len(voices) == 3
        assert "morgan freeman" in voices
        assert "david attenborough" in voices
        assert "james earl jones" in voices

    def test_list_voices_sorted(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        voices = vm.list_voices()
        assert voices == sorted(voices)

    def test_list_voices_empty_dir(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_voices"
        empty.mkdir()
        monkeypatch.setattr(vm, "VOICES_DIR", empty)
        assert vm.list_voices() == []

    def test_list_voices_nonexistent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_path / "does_not_exist")
        assert vm.list_voices() == []


class TestGetVoice:
    """Test voice lookup with fuzzy matching."""

    def test_exact_match(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        result = vm.get_voice("morgan freeman")
        assert result["found"] is True
        assert result["exact"] is True
        assert result["name"] == "morgan freeman"
        assert result["audio"] is not None

    def test_exact_match_case_insensitive(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        result = vm.get_voice("Morgan Freeman")
        assert result["found"] is True
        assert result["exact"] is True

    def test_fuzzy_match(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        result = vm.get_voice("morgan freemn", fuzzy_threshold=0.6)
        assert result["found"] is True
        assert result["exact"] is False
        assert "morgan freeman" in result["name"]

    def test_no_match(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        result = vm.get_voice("completely unknown person")
        assert result["found"] is False
        assert result["audio"] is None

    def test_no_voices_available(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty_voices"
        empty.mkdir()
        monkeypatch.setattr(vm, "VOICES_DIR", empty)
        result = vm.get_voice("anyone")
        assert result["found"] is False
        assert "No voice samples available" in result["message"]

    def test_high_threshold_rejects_fuzzy(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        result = vm.get_voice("morgen freemen", fuzzy_threshold=0.99)
        assert result["found"] is False


class TestDeleteVoice:
    """Test voice deletion."""

    def test_delete_existing(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        assert vm.delete_voice("morgan freeman") is True
        assert not (tmp_voices_dir / "morgan_freeman.wav").exists()

    def test_delete_nonexistent(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        assert vm.delete_voice("nobody") is False


class TestAddVoice:
    """Test voice sample saving."""

    def test_add_voice(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        audio = np.zeros(16000, dtype=np.float32)
        path = vm.add_voice("New Person", audio, sample_rate=16000)
        assert "new_person.wav" in path
        assert (tmp_voices_dir / "new_person.wav").exists()

    def test_add_voice_resamples(self, tmp_voices_dir, monkeypatch):
        monkeypatch.setattr(vm, "VOICES_DIR", tmp_voices_dir)
        # 48kHz audio should be resampled to 16kHz
        audio = np.zeros(48000, dtype=np.float32)
        path = vm.add_voice("Resampled", audio, sample_rate=48000)
        assert (tmp_voices_dir / "resampled.wav").exists()
        # Verify the saved file is at 16kHz
        import soundfile as sf
        _, sr = sf.read(path)
        assert sr == 16000
