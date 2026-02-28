"""Fixed prompt definitions for quantization benchmarks.

Each prompt is a dict with:
  id            — filename prefix for outputs
  messages      — chat messages list (same format as model_manager.chat())
  generate_audio — whether to request TTS output
  description   — human-readable purpose
  use_streaming — if True, run through chat_streaming() instead of chat()
  needs_voice_ref — if True, only run when --voice-ref is provided
"""

# ── Echo Prompts ─────────────────────────────────────────────────────────
# Same intended output across quant levels.  Primary comparison targets.

ECHO_PROMPTS = [
    {
        "id": "echo_pangram",
        "messages": [{"role": "user", "content": [
            "Read this aloud exactly as written: "
            "'The quick brown fox jumps over the lazy dog.'"
        ]}],
        "generate_audio": True,
        "use_streaming": False,
        "needs_voice_ref": False,
        "description": "Pangram — covers all phonemes, identical intended audio",
    },
    {
        "id": "echo_counting",
        "messages": [{"role": "user", "content": [
            "Read this aloud exactly as written: "
            "'One, two, three, four, five, six, seven, eight, nine, ten.'"
        ]}],
        "generate_audio": True,
        "use_streaming": False,
        "needs_voice_ref": False,
        "description": "Counting — simple predictable cadence",
    },
    {
        "id": "echo_audio",
        "messages": None,  # Built at runtime with voice ref audio
        "generate_audio": True,
        "use_streaming": False,
        "needs_voice_ref": True,
        "description": "Audio echo — repeat back a reference WAV (full audio pipeline test)",
    },
]

# ── Free Prompts ─────────────────────────────────────────────────────────
# Sanity checks — verify model still works at each quant level.

TEXT_PROMPTS = [
    {
        "id": "text_math",
        "messages": [{"role": "user", "content": [
            "What is 7 times 8? Answer only with the number."
        ]}],
        "generate_audio": False,
        "use_streaming": False,
        "needs_voice_ref": False,
        "description": "Deterministic math — expected answer: 56",
        "expected_substring": "56",
    },
    {
        "id": "text_long",
        "messages": [{"role": "user", "content": [
            "Explain in exactly three sentences what photosynthesis is and why it matters."
        ]}],
        "generate_audio": False,
        "use_streaming": False,
        "needs_voice_ref": False,
        "description": "Longer text — tests coherence under quantization",
    },
]

TTS_PROMPTS = [
    {
        "id": "tts_greeting",
        "messages": [{"role": "user", "content": [
            "Please respond in English. Say hello and introduce yourself in one sentence."
        ]}],
        "generate_audio": True,
        "use_streaming": False,
        "needs_voice_ref": False,
        "description": "Short TTS — basic functionality check (default voice)",
    },
    {
        "id": "tts_story",
        "messages": [{"role": "user", "content": [
            "Please respond in English. "
            "Tell me a very short story about a cat in exactly three sentences."
        ]}],
        "generate_audio": True,
        "use_streaming": False,
        "needs_voice_ref": False,
        "description": "Longer TTS — reveals degradation over time",
    },
]

STREAMING_PROMPTS = [
    {
        "id": "stream_greeting",
        "messages": [{"role": "user", "content": [
            "Please respond in English. Say hello and tell me what you can do in one sentence."
        ]}],
        "generate_audio": True,
        "use_streaming": True,
        "needs_voice_ref": False,
        "description": "Streaming TTS — tests streaming pipeline quality",
    },
]


def get_all_prompts():
    """Return all prompt lists in execution order."""
    return ECHO_PROMPTS + TEXT_PROMPTS + TTS_PROMPTS + STREAMING_PROMPTS


def build_echo_audio_messages(voice_ref_audio):
    """Build messages for the echo_audio prompt with the actual audio data.

    Args:
        voice_ref_audio: numpy array of 16kHz mono audio to echo back.

    Returns:
        List of message dicts suitable for chat().
    """
    return [{"role": "user", "content": [
        voice_ref_audio,
        "Repeat exactly what you just heard. Say the same words aloud."
    ]}]
