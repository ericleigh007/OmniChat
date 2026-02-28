"""Integration tests for OmniChat — loads the real model, no Gradio.

Run with:
    python -m pytest tests/test_integration.py -v          # all integration tests
    python -m pytest tests/test_integration.py -v -k text  # only text tests (faster)
    python -m pytest tests/ -v -m "not gpu"                # skip GPU tests entirely

These tests require:
- CUDA GPU with sufficient VRAM (~19 GB for bf16 model)
- MiniCPM-o 4.5 downloaded (auto-downloads on first run)
"""

import difflib
from pathlib import Path

import numpy as np
import pytest

# Every test in this module needs the GPU
pytestmark = pytest.mark.gpu


# ── Helpers ──────────────────────────────────────────────────────────────────

def similarity(a: str, b: str) -> float:
    """Normalized text similarity ratio (0.0 = completely different, 1.0 = identical)."""
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def is_echo(response_text: str, input_text: str, threshold: float = 0.9) -> bool:
    """Check if the model's response is a near-verbatim echo of the input.

    A true echo (model repeating input back) scores >0.95 similarity.
    Normal Q&A can score 0.6-0.8 when the answer restates part of the question
    (e.g. "What is the capital of Japan?" -> "The capital of Japan is Tokyo."),
    so threshold is set high at 0.9 to avoid false positives.
    """
    return similarity(response_text, input_text) > threshold


# ── Text-only chat (no TTS, faster) ─────────────────────────────────────────

class TestTextChat:
    """Text-in -> text-out, no audio generation. Fastest GPU tests."""

    def test_basic_response(self, loaded_model):
        from tools.model.model_manager import chat

        result = chat(
            messages=[{"role": "user", "content": ["What is 2 + 2?"]}],
            generate_audio=False,
        )
        assert result["text"] is not None
        assert len(result["text"]) > 0
        print(f"  Response: {result['text'][:200]}")

    def test_response_is_not_echo(self, loaded_model):
        from tools.model.model_manager import chat

        prompt = "Tell me a short joke about robots."
        result = chat(
            messages=[{"role": "user", "content": [prompt]}],
            generate_audio=False,
        )
        assert not is_echo(result["text"], prompt), (
            f"Model echoed the input!\n"
            f"  Input:    {prompt}\n"
            f"  Response: {result['text'][:200]}"
        )
        print(f"  Prompt:   {prompt}")
        print(f"  Response: {result['text'][:200]}")

    def test_response_has_substance(self, loaded_model):
        from tools.model.model_manager import chat

        result = chat(
            messages=[{"role": "user", "content": ["Explain what a neural network is in two sentences."]}],
            generate_audio=False,
        )
        # Response should be longer than the prompt (actual explanation)
        assert len(result["text"]) > 20, f"Response too short: '{result['text']}'"
        print(f"  Response: {result['text'][:200]}")


# ── Multi-turn echo test (the key bug diagnostic) ───────────────────────────

class TestMultiTurnEcho:
    """Specifically tests the 1st-OK / 2nd-echo / 3rd-OK pattern.

    Each call goes through the full _reset_for_generation -> model.chat() path,
    exactly like the Gradio UI does.
    """

    def test_three_turns_no_echo(self, loaded_model):
        from tools.model.model_manager import chat

        # Math prompts that yield only a number — zero overlap with question text
        prompts = [
            "If I add 100 to 1, what number do I get? Answer only with the number.",
            "What is 50 times 2? Answer only with the number.",
            "What is 1000 minus 1? Answer only with the number.",
        ]
        results = []

        for i, prompt in enumerate(prompts, 1):
            result = chat(
                messages=[{"role": "user", "content": [prompt]}],
                generate_audio=False,
            )
            echo = is_echo(result["text"], prompt)
            status = "ECHO" if echo else "OK"
            print(f"  Turn {i} [{status}]: '{prompt}' -> '{result['text'][:100]}'")
            results.append((prompt, result["text"], echo))

        # Check each turn
        for i, (prompt, response, echo) in enumerate(results, 1):
            assert not echo, (
                f"Turn {i} echoed!\n"
                f"  Prompt:   {prompt}\n"
                f"  Response: {response[:200]}"
            )

    def test_five_turns_text_only(self, loaded_model):
        from tools.model.model_manager import chat

        # Math prompts — answers are just numbers, zero overlap with question text
        prompts = [
            "What is 7 times 8? Answer only with the number.",
            "What is 200 plus 50? Answer only with the number.",
            "What is 144 divided by 12? Answer only with the number.",
            "What is 99 minus 33? Answer only with the number.",
            "What is 25 squared? Answer only with the number.",
        ]
        echo_count = 0

        for i, prompt in enumerate(prompts, 1):
            result = chat(
                messages=[{"role": "user", "content": [prompt]}],
                generate_audio=False,
            )
            echo = is_echo(result["text"], prompt)
            status = "ECHO" if echo else "OK"
            print(f"  Turn {i} [{status}]: '{prompt}' -> '{result['text'][:80]}'")
            if echo:
                echo_count += 1

        assert echo_count == 0, f"{echo_count}/5 turns were echoes"


# ── Audio generation tests ───────────────────────────────────────────────────

class TestAudioGeneration:
    """Test TTS output — file creation, format, duration."""

    def test_audio_file_created(self, loaded_model, tmp_path):
        from tools.model.model_manager import chat

        output_path = str(tmp_path / "response.wav")
        result = chat(
            messages=[{"role": "user", "content": ["Hello, how are you?"]}],
            generate_audio=True,
            output_audio_path=output_path,
        )
        assert Path(output_path).exists(), "Audio file was not created"
        assert result["audio"] is not None, "Audio array not returned"
        print(f"  Text:     {result['text'][:100]}")
        print(f"  Audio:    {len(result['audio'])} samples")
        print(f"  Duration: {len(result['audio']) / result.get('sample_rate', 24000):.1f}s")

    def test_audio_sample_rate_24k(self, loaded_model, tmp_path):
        from tools.model.model_manager import chat
        import soundfile as sf

        output_path = str(tmp_path / "response.wav")
        result = chat(
            messages=[{"role": "user", "content": ["Test."]}],
            generate_audio=True,
            output_audio_path=output_path,
        )
        assert result.get("sample_rate") == 24000, (
            f"Expected 24kHz, got {result.get('sample_rate')}"
        )
        # Also verify the file on disk
        _, sr = sf.read(output_path)
        assert sr == 24000

    def test_audio_is_float32(self, loaded_model, tmp_path):
        from tools.model.model_manager import chat

        output_path = str(tmp_path / "response.wav")
        result = chat(
            messages=[{"role": "user", "content": ["Hello."]}],
            generate_audio=True,
            output_audio_path=output_path,
        )
        assert result["audio"].dtype == np.float32

    def test_audio_has_fade_in(self, loaded_model, tmp_path):
        from tools.model.model_manager import chat

        output_path = str(tmp_path / "response.wav")
        result = chat(
            messages=[{"role": "user", "content": ["Good morning."]}],
            generate_audio=True,
            output_audio_path=output_path,
        )
        audio = result["audio"]
        if audio is not None and len(audio) > 100:
            # First sample should be 0 or very close (fade-in applied)
            assert abs(audio[0]) < 0.01, (
                f"First sample is {audio[0]:.4f} -- fade-in may not be applied"
            )
            print(f"  First 5 samples: {audio[:5]}")
            print(f"  Samples at 2400: {audio[2395:2405]}")


# ── Multi-turn with audio (the full echo test) ──────────────────────────────

class TestMultiTurnAudio:
    """Tests with generate_audio=True — the exact path where echo occurs."""

    def test_three_turns_with_audio_no_echo(self, loaded_model, tmp_path):
        from tools.model.model_manager import chat

        prompts = [
            "What is 15 times 3? Answer only with the number.",
            "What is 500 minus 123? Answer only with the number.",
            "What is 64 divided by 8? Answer only with the number.",
        ]

        for i, prompt in enumerate(prompts, 1):
            output_path = str(tmp_path / f"response_turn{i}.wav")
            result = chat(
                messages=[{"role": "user", "content": [prompt]}],
                generate_audio=True,
                output_audio_path=output_path,
            )
            echo = is_echo(result["text"], prompt)
            has_audio = Path(output_path).exists()
            status = "ECHO" if echo else "OK"
            audio_status = "audio=YES" if has_audio else "audio=NO"

            print(f"  Turn {i} [{status}] [{audio_status}]: '{prompt}' -> '{result['text'][:80]}'")

            assert not echo, (
                f"Turn {i} echoed with audio generation!\n"
                f"  Prompt:   {prompt}\n"
                f"  Response: {result['text'][:200]}"
            )
            assert has_audio, f"Turn {i}: audio file not created"


# ── Voice switching ──────────────────────────────────────────────────────────

class TestVoiceSwitching:
    """Test voice reference audio is accepted by the model."""

    def test_default_voice_consecutive(self, loaded_model, tmp_path):
        """Multiple turns with default voice — no voice_ref, just TTS."""
        from tools.model.model_manager import chat

        prompts = [
            "What is 9 times 9? Answer only with the number.",
            "What is 300 plus 45? Answer only with the number.",
        ]
        for i, prompt in enumerate(prompts, 1):
            out = str(tmp_path / f"default_turn{i}.wav")
            result = chat(
                messages=[{"role": "user", "content": [prompt]}],
                generate_audio=True,
                output_audio_path=out,
            )
            assert Path(out).exists(), f"Turn {i}: audio not generated"
            assert not is_echo(result["text"], prompt), (
                f"Turn {i} echoed during default voice test"
            )
            print(f"  Turn {i}: {result['text'][:80]}")

    @pytest.mark.xfail(
        reason="Token2wav _prepare_prompt passes None as prompt_wav — "
               "voice cloning pipeline bug in stepaudio2",
        strict=False,
    )
    def test_custom_voice_ref(self, loaded_model, tmp_path):
        """Custom voice reference — tests voice cloning path.

        Known issue: Token2wav's _prepare_prompt expects a WAV file path
        for the voice prompt but receives None, causing TypeError.
        """
        import soundfile as sf
        from tools.model.model_manager import chat

        # Create a proper WAV file as voice reference (3s of 200Hz tone at 16kHz)
        voice_audio = np.sin(
            np.linspace(0, 2 * np.pi * 200 * 3, 16000 * 3, dtype=np.float32)
        )
        voice_wav = str(tmp_path / "voice_ref.wav")
        sf.write(voice_wav, voice_audio, 16000)
        # Load it back as numpy array (what chat() expects)
        voice_ref, _ = sf.read(voice_wav, dtype="float32")

        out = str(tmp_path / "custom_voice.wav")
        result = chat(
            messages=[{"role": "user", "content": ["Describe what a sunset looks like."]}],
            voice_ref=voice_ref,
            generate_audio=True,
            output_audio_path=out,
        )
        assert Path(out).exists(), "Custom voice audio not generated"
        print(f"  Custom voice: {result['text'][:80]}")


# ── Audio input test (simulated mic) ────────────────────────────────────────

# ── Streaming audio generation ────────────────────────────────────────────

class TestStreamingAudio:
    """Test streaming generation pipeline (streaming_prefill + streaming_generate)."""

    def test_streaming_produces_audio_and_text(self, loaded_model):
        """chat_streaming() yields non-empty audio chunks and text."""
        from tools.model.model_manager import chat_streaming

        chunks = []
        full_text = ""
        for audio_chunk, text_chunk in chat_streaming(
            messages=[{"role": "user", "content": ["Please respond in English. Say hello in a friendly way."]}],
            generate_audio=True,
        ):
            if audio_chunk is not None:
                chunks.append(audio_chunk)
            if text_chunk:
                full_text += text_chunk

        assert len(chunks) > 0, "No audio chunks yielded by streaming"
        assert len(full_text) > 0, "No text yielded by streaming"

        total_samples = sum(len(c) for c in chunks)
        duration = total_samples / 24000
        print(f"  Chunks: {len(chunks)}, total samples: {total_samples}, duration: {duration:.1f}s")
        print(f"  Text: {full_text[:120]}")

        # Audio should be at least 0.5s (model says something)
        assert duration > 0.5, f"Audio too short: {duration:.2f}s"

    def test_streaming_text_only(self, loaded_model):
        """chat_streaming() with generate_audio=False yields text but no audio."""
        from tools.model.model_manager import chat_streaming

        chunks = []
        full_text = ""
        for audio_chunk, text_chunk in chat_streaming(
            messages=[{"role": "user", "content": ["What is 7 plus 3? Answer only with the number."]}],
            generate_audio=False,
        ):
            if audio_chunk is not None:
                chunks.append(audio_chunk)
            if text_chunk:
                full_text += text_chunk

        assert len(chunks) == 0, f"Got {len(chunks)} audio chunks with generate_audio=False"
        assert len(full_text) > 0, "No text yielded"
        print(f"  Text: {full_text[:120]}")

    def test_streaming_with_playback_headless(self, loaded_model, tmp_path):
        """chat_streaming_with_playback() in headless mode collects audio and saves to file."""
        from tools.model.model_manager import chat_streaming_with_playback

        output_path = str(tmp_path / "streaming_output.wav")
        result = chat_streaming_with_playback(
            messages=[{"role": "user", "content": ["Please respond in English. What is 5 times 5? Say the answer in a short sentence."]}],
            output_audio_path=output_path,
            headless=True,
        )

        assert result["text"] is not None and len(result["text"]) > 0
        assert result["audio"] is not None
        assert result["sample_rate"] == 24000
        assert Path(output_path).exists(), "Audio file not saved"

        duration = len(result["audio"]) / result["sample_rate"]
        print(f"  Text: {result['text'][:120]}")
        print(f"  Audio: {len(result['audio'])} samples, {duration:.1f}s")
        print(f"  File: {output_path}")

    def test_streaming_consecutive_turns(self, loaded_model):
        """Multiple streaming turns don't interfere with each other (no stale state)."""
        from tools.model.model_manager import chat_streaming

        prompts = [
            "Please respond in English. What is 10 plus 10? Say the answer in a short sentence.",
            "Please respond in English. What is 100 minus 1? Say the answer in a short sentence.",
        ]

        for i, prompt in enumerate(prompts, 1):
            chunks = []
            full_text = ""
            for audio_chunk, text_chunk in chat_streaming(
                messages=[{"role": "user", "content": [prompt]}],
                generate_audio=True,
            ):
                if audio_chunk is not None:
                    chunks.append(audio_chunk)
                if text_chunk:
                    full_text += text_chunk

            total_samples = sum(len(c) for c in chunks)
            duration = total_samples / 24000
            print(f"  Turn {i}: {len(chunks)} chunks, {duration:.1f}s, text: {full_text[:80]}")
            assert len(chunks) > 0, f"Turn {i}: no audio chunks"
            assert len(full_text) > 0, f"Turn {i}: no text"


# ── Audio input test (simulated mic) ────────────────────────────────────

class TestAudioInput:
    """Send audio as user input (simulating microphone recording)."""

    def test_audio_input_gets_response(self, loaded_model, tmp_path):
        """Send a sine wave as 'speech' — model should respond, not crash."""
        from tools.model.model_manager import chat

        # 2 seconds of 440Hz tone at 16kHz (not real speech, but tests the pipeline)
        audio_input = np.sin(
            np.linspace(0, 2 * np.pi * 440 * 2, 32000, dtype=np.float32)
        )

        output_path = str(tmp_path / "audio_input_response.wav")
        result = chat(
            messages=[{"role": "user", "content": [audio_input]}],
            generate_audio=True,
            output_audio_path=output_path,
        )

        assert result["text"] is not None, "No text response from audio input"
        assert len(result["text"]) > 0, "Empty text response from audio input"
        print(f"  Audio input response: {result['text'][:200]}")


# ── Streaming with audio input (the conversation-mode path) ──────────────

class TestStreamingAudioInput:
    """Test chat_streaming() with audio input — the exact pipeline that
    conversation mode and single-turn mic streaming use.

    This exercises the audio chunking fix: complete audio recordings are
    split into FIRST_CHUNK (1035ms) + REGULAR_CHUNK (1000ms) pieces before
    being sent to streaming_prefill.
    """

    def test_streaming_audio_input_produces_response(self, loaded_model):
        """Send audio through chat_streaming() — should yield text + audio."""
        from tools.model.model_manager import chat_streaming

        # 2 seconds of 440Hz tone at 16kHz (simulating a mic recording)
        audio_input = np.sin(
            np.linspace(0, 2 * np.pi * 440 * 2, 32000, dtype=np.float32)
        )
        msgs = [{"role": "user", "content": [audio_input]}]

        chunks = []
        full_text = ""
        for audio_chunk, text_chunk in chat_streaming(
            messages=msgs, generate_audio=True,
        ):
            if audio_chunk is not None:
                chunks.append(audio_chunk)
            if text_chunk:
                full_text += text_chunk

        assert len(full_text) > 0, "No text from streaming audio input"
        total_samples = sum(len(c) for c in chunks)
        duration = total_samples / 24000 if total_samples else 0
        print(f"  Audio input streaming: {len(chunks)} chunks, "
              f"{duration:.1f}s audio, text: {full_text[:120]}")

    def test_streaming_audio_input_short_clip(self, loaded_model):
        """Short audio (<1035ms) should work — model pads to FIRST_CHUNK_MS."""
        from tools.model.model_manager import chat_streaming

        # 0.5 seconds = 8000 samples (shorter than FIRST_CHUNK of 16560)
        audio_input = np.sin(
            np.linspace(0, 2 * np.pi * 440 * 0.5, 8000, dtype=np.float32)
        )
        msgs = [{"role": "user", "content": [audio_input]}]

        full_text = ""
        for audio_chunk, text_chunk in chat_streaming(
            messages=msgs, generate_audio=True,
        ):
            if text_chunk:
                full_text += text_chunk

        assert len(full_text) > 0, "No text from short streaming audio input"
        print(f"  Short audio (0.5s) streaming text: {full_text[:120]}")

    def test_streaming_audio_input_long_clip(self, loaded_model):
        """Long audio (>3s) splits into 4+ chunks — all should process correctly."""
        from tools.model.model_manager import chat_streaming

        # 5 seconds = 80000 samples → should split into 5 chunks
        audio_input = np.sin(
            np.linspace(0, 2 * np.pi * 440 * 5, 80000, dtype=np.float32)
        )
        msgs = [{"role": "user", "content": [audio_input]}]

        chunks = []
        full_text = ""
        for audio_chunk, text_chunk in chat_streaming(
            messages=msgs, generate_audio=True,
        ):
            if audio_chunk is not None:
                chunks.append(audio_chunk)
            if text_chunk:
                full_text += text_chunk

        assert len(full_text) > 0, "No text from long streaming audio input"
        print(f"  Long audio (5s) streaming: {len(chunks)} chunks, "
              f"text: {full_text[:120]}")

    def test_streaming_audio_with_text_prompt(self, loaded_model):
        """Audio + text together in the same message (mixed content)."""
        from tools.model.model_manager import chat_streaming

        audio_input = np.sin(
            np.linspace(0, 2 * np.pi * 440 * 2, 32000, dtype=np.float32)
        )
        msgs = [{"role": "user", "content": [
            "Describe what you hear in this audio.",
            audio_input,
        ]}]

        full_text = ""
        for audio_chunk, text_chunk in chat_streaming(
            messages=msgs, generate_audio=True,
        ):
            if text_chunk:
                full_text += text_chunk

        assert len(full_text) > 0, "No text from audio+text streaming input"
        print(f"  Audio+text streaming: {full_text[:120]}")

    def test_streaming_audio_input_consecutive_turns(self, loaded_model):
        """Multiple audio-input streaming turns don't interfere."""
        from tools.model.model_manager import chat_streaming

        for turn in range(2):
            audio_input = np.sin(
                np.linspace(0, 2 * np.pi * (440 + turn * 100) * 2,
                            32000, dtype=np.float32)
            )
            msgs = [{"role": "user", "content": [audio_input]}]

            full_text = ""
            for audio_chunk, text_chunk in chat_streaming(
                messages=msgs, generate_audio=True,
            ):
                if text_chunk:
                    full_text += text_chunk

            assert len(full_text) > 0, f"Turn {turn+1}: no text"
            print(f"  Turn {turn+1}: {full_text[:80]}")
