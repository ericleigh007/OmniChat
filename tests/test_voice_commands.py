"""Tests for voice command detection."""

import pytest

from tools.shared.session import detect_voice_command


class TestDetectVoiceCommand:
    """Test the regex-based voice command parser."""

    # ── Positive matches ──────────────────────────────────────────────

    def test_change_voice_to(self):
        assert detect_voice_command("change voice to Morgan Freeman") == "Morgan Freeman"

    def test_switch_voice_to(self):
        assert detect_voice_command("switch voice to Morgan Freeman") == "Morgan Freeman"

    def test_change_the_voice_to(self):
        assert detect_voice_command("change the voice to David Attenborough") == "David Attenborough"

    def test_switch_to_possessive_voice(self):
        assert detect_voice_command("switch to Morgan Freeman's voice") == "Morgan Freeman"

    def test_use_possessive_voice(self):
        assert detect_voice_command("use David Attenborough's voice") == "David Attenborough"

    def test_try_possessive_voice(self):
        assert detect_voice_command("try James Earl Jones's voice") == "James Earl Jones"

    def test_sound_like(self):
        assert detect_voice_command("sound like James Earl Jones") == "James Earl Jones"

    def test_speak_like(self):
        assert detect_voice_command("speak like Batman") == "Batman"

    def test_speak_as(self):
        assert detect_voice_command("speak as Gandalf") == "Gandalf"

    # ── Default voice ─────────────────────────────────────────────────

    def test_go_back_to_default(self):
        assert detect_voice_command("go back to default voice") == "default"

    def test_use_default_voice(self):
        assert detect_voice_command("use the default voice") == "default"

    def test_switch_to_default(self):
        assert detect_voice_command("switch to default voice") == "default"

    # ── No match (negative cases) ────────────────────────────────────

    def test_no_command_greeting(self):
        assert detect_voice_command("hello how are you") is None

    def test_no_command_unrelated(self):
        assert detect_voice_command("I want to change my shirt") is None

    def test_no_command_empty(self):
        assert detect_voice_command("") is None

    def test_no_command_just_voice(self):
        assert detect_voice_command("voice") is None

    # ── Edge cases ────────────────────────────────────────────────────

    def test_case_insensitive(self):
        assert detect_voice_command("CHANGE VOICE TO Bob") == "Bob"

    def test_trailing_period(self):
        result = detect_voice_command("change voice to Morgan Freeman.")
        assert result == "Morgan Freeman"

    def test_mixed_case(self):
        assert detect_voice_command("Switch Voice To Alice") == "Alice"

    def test_single_name(self):
        assert detect_voice_command("change voice to Siri") == "Siri"

    def test_name_with_hyphen(self):
        assert detect_voice_command("change voice to Mary-Jane") == "Mary-Jane"
