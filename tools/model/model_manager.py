"""
model_manager.py — Load and manage MiniCPM-o 4.5 for multimodal inference.

Singleton pattern: call get_model() to load once, reuse everywhere.
Supports text, audio (with voice cloning), image, and video inputs.

Two generation paths:
  - chat()                        — file-based, blocking (original)
  - chat_streaming()              — yields (audio_chunk, text) with no file I/O
  - chat_streaming_with_playback()— streaming + real-time speaker playback
"""

import sys
from pathlib import Path
from typing import Callable, Generator, Optional

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODEL_NAME = "openbmb/MiniCPM-o-4_5"

# Module-level singleton
_model = None
_tokenizer = None

# Quantization mode: "none" (bf16, ~19 GB), "int8" (~10-12 GB), "int4" (NF4, ~11 GB)
_quantization = "none"


def set_quantization(mode: str) -> None:
    """Set the quantization mode before the model is loaded.

    Must be called before the first get_model() call.  Has no effect if the
    model is already loaded (singleton).

    Args:
        mode: "none" (bf16), "int8" (bitsandbytes 8-bit), or "int4" (bitsandbytes NF4).
    """
    global _quantization
    valid = ("none", "int8", "int4")
    if mode not in valid:
        raise ValueError(f"Unknown quantization mode: {mode!r}. Use one of {valid}.")
    _quantization = mode

# Turn counter for diagnostics
_turn_count = 0

# Duration (in samples @ 24kHz) for fade-in on generated audio to smooth
# the HiFT vocoder's cold-start artifact (160ms zeros + no crossfade).
_FADE_IN_SAMPLES = 2400  # 100ms at 24kHz


def get_model():
    """Load MiniCPM-o 4.5 (singleton — loads once, reuses on subsequent calls).

    Respects the quantization mode set via set_quantization():
      - "none"  — bf16, ~19 GB VRAM (default)
      - "int8"  — bitsandbytes 8-bit, ~10-12 GB VRAM
      - "int4"  — bitsandbytes NF4 with double quantization, ~11 GB VRAM
    """
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModel, AutoTokenizer

    model_name = MODEL_NAME

    quant_label = {"none": "bf16 (full precision)", "int8": "int8 (bitsandbytes)", "int4": "int4 (bitsandbytes NF4)"}
    print(f"Loading {model_name}...")
    print(f"  Precision: {quant_label.get(_quantization, _quantization)}")
    print(f"  Device: cuda ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")

    _tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )

    load_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
        "init_vision": True,
        "init_audio": True,
        "init_tts": True,
    }

    if _quantization == "int8":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            # Only quantize LLM transformer layers (llm.model.layers.*).
            # Skip ALL non-LLM modules to avoid two problems:
            #
            # 1. SCB AttributeError: bitsandbytes Linear8bitLt._save_to_state_dict
            #    calls getattr(self.weight, "SCB") without a default. If any module's
            #    weight is a plain nn.Parameter (not Int8Params), this crashes.
            #    The TTS head_code modules use weight_norm (parametrize API), which
            #    is fundamentally incompatible with Linear8bitLt replacement.
            #
            # 2. Non-LLM modules (vision encoder, audio encoder, TTS, resampler,
            #    projection layers) are small relative to the LLM and benefit
            #    little from INT8 quantization.  Keeping them in bf16 preserves
            #    quality for audio/vision processing at negligible VRAM cost.
            #
            # The int4 (NF4) config uses the same skip list.
            llm_int8_skip_modules=[
                "lm_head",
                "apm",                    # Whisper audio encoder
                "tts",                    # TTS decoder (has weight_norm layers)
                "vpm",                    # SigLIP vision encoder
                "resampler",              # vision resampler
                "audio_projection_layer", # audio-to-LLM projector
                "audio_avg_pooler",       # audio pooling layer
            ],
        )
        load_kwargs["device_map"] = "auto"
    elif _quantization == "int4":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            # Same skip list as int8 — only quantize LLM transformer layers.
            # Non-LLM modules stay in bf16 for audio/vision quality.
            llm_int8_skip_modules=[
                "lm_head",
                "apm",                    # Whisper audio encoder
                "tts",                    # TTS decoder (has weight_norm layers)
                "vpm",                    # SigLIP vision encoder
                "resampler",              # vision resampler
                "audio_projection_layer", # audio-to-LLM projector
                "audio_avg_pooler",       # audio pooling layer
            ],
        )
        load_kwargs["device_map"] = "auto"

    _model = AutoModel.from_pretrained(model_name, **load_kwargs)

    if _quantization in ("int8", "int4"):
        # BNB with device_map="auto" places the model — do NOT call .cuda()
        _model.eval()
    else:
        _model.eval().cuda()

    _model.init_tts()

    # Pre-initialize Token2wav cache with a short silent prompt so the default
    # voice TTS path works.  Without this, the first TTS call with no voice_ref
    # crashes in Token2wav._prepare_prompt(None) → TypeError("Invalid file: None")
    # because prompt_wav is None and cache hasn't been built yet.
    _init_audio = np.zeros(16000, dtype=np.float32)  # 1s silence at 16kHz
    _model.init_token2wav_cache(_init_audio)

    vram_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM used: {vram_used:.1f} GB")
    print("  Model loaded and ready.")
    return _model, _tokenizer


def _reset_for_generation(model, generate_audio: bool, output_audio_path: Optional[str] = None):
    """Clear model state before a new generation call."""
    # Delete stale output audio so a failed TTS can't return the previous voice
    if output_audio_path:
        Path(output_audio_path).unlink(missing_ok=True)

    # Full session reset — clears KV caches, round history, flags, etc.
    # Note: tts_last_turn_tokens is never actually populated by the model's
    # non-streaming path (TTSStreamingGenerator doesn't update it), so there's
    # nothing to preserve.
    model.reset_session()

    # Free stale GPU tensors that might interfere with the next generation
    torch.cuda.empty_cache()

    # NOTE: We do NOT clear Token2wav's internal cache (tok2wav.cache) here.
    # Token2wav.cache holds vocoder features for the current voice — it is safe
    # to reuse across turns and in fact MUST be preserved, because the model's
    # default-voice path passes prompt_wav=None and Token2wav._prepare_prompt(None)
    # crashes with TypeError("Invalid file: None").  The model's own
    # reset_session() handles its separate token2wav_cache for streaming.
    # Voice switching (when voice_ref changes) is handled by the model's
    # get_sys_prompt() which embeds the ref audio, causing audio_prompt to be
    # non-None on the next _generate_speech_non_streaming call.


def _apply_fade_in(audio: np.ndarray, n_samples: int = _FADE_IN_SAMPLES) -> np.ndarray:
    """Apply a linear fade-in to smooth the HiFT vocoder's cold-start artifact."""
    if len(audio) <= n_samples:
        return audio
    fade = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    audio = audio.copy()
    audio[:n_samples] *= fade
    return audio


# ── Audio leveling ────────────────────────────────────────────────────────
#
# Configurable via args/settings.yaml → audio.leveling.
# All levels in dBFS (0 = full scale).  Converted to linear at load time.
#
# Two processing stages:
#   1. Voice ref input  — static RMS normalizer (short clips, one gain value)
#   2. Output audio     — windowed RMS compressor with attack/release envelope
#
# Defaults match the YAML shipped with the project.  If the YAML is missing
# or the section is absent, these kick in so the code never crashes.

_LEVELING_DEFAULTS = {
    "enabled": True,
    # Voice reference (static normalizer)
    "ref_target_dbfs": -26.0,
    "ref_max_gain_db": 20.0,
    # Output compressor
    "output_threshold_dbfs": -20.0,   # level where compression engages
    "output_ratio": 3.0,             # compression ratio (3:1)
    "output_attack_ms": 15.0,        # how fast gain reduces on loud signal
    "output_release_ms": 150.0,      # how fast gain recovers after signal drops
    "output_knee_db": 6.0,           # soft knee width (0 = hard knee)
    "output_makeup_db": 4.0,         # post-compression boost
    "output_max_gain_db": 20.0,      # safety cap on total gain
    # Peak limiter (shared)
    "peak_ceiling_dbfs": -0.1,
}

# Cached config (loaded once on first use)
_leveling_cfg: dict | None = None


def _dbfs_to_linear(dbfs: float) -> float:
    """Convert dBFS to linear amplitude (0 dBFS = 1.0)."""
    return 10.0 ** (dbfs / 20.0)


def _db_to_ratio(db: float) -> float:
    """Convert a dB gain value to a linear ratio (20 dB = 10x)."""
    return 10.0 ** (db / 20.0)


def _linear_to_dbfs(linear: float) -> float:
    """Convert linear amplitude to dBFS.  Clamps to -120 dBFS for silence."""
    if linear < 1e-6:
        return -120.0
    return 20.0 * np.log10(linear)


def _get_leveling_config() -> dict:
    """Load leveling config from settings.yaml (cached after first call)."""
    global _leveling_cfg
    if _leveling_cfg is not None:
        return _leveling_cfg

    cfg = dict(_LEVELING_DEFAULTS)

    try:
        import yaml
        settings_path = Path(__file__).parent.parent.parent / "args" / "settings.yaml"
        if settings_path.exists():
            with open(settings_path, "r", encoding="utf-8") as f:
                settings = yaml.safe_load(f) or {}
            leveling = settings.get("audio", {}).get("leveling", {})
            if leveling:
                for key in _LEVELING_DEFAULTS:
                    if key in leveling:
                        cfg[key] = leveling[key]
    except Exception as e:
        print(f"  [leveling] Warning: could not load settings, using defaults: {e}")

    # Pre-compute linear values for fast access
    cfg["_ref_target_linear"] = _dbfs_to_linear(cfg["ref_target_dbfs"])
    cfg["_ref_max_gain_linear"] = _db_to_ratio(cfg["ref_max_gain_db"])
    cfg["_output_threshold_linear"] = _dbfs_to_linear(cfg["output_threshold_dbfs"])
    cfg["_output_makeup_linear"] = _db_to_ratio(cfg["output_makeup_db"])
    cfg["_output_max_gain_linear"] = _db_to_ratio(cfg["output_max_gain_db"])
    cfg["_peak_ceiling_linear"] = _dbfs_to_linear(cfg["peak_ceiling_dbfs"])

    _leveling_cfg = cfg
    return cfg


def _normalize_rms(audio: np.ndarray, target_rms: float, max_gain: float, peak_ceiling: float) -> np.ndarray:
    """Static RMS normalizer — single gain for the whole signal.

    Appropriate for short clips (voice references) where temporal dynamics
    don't matter.  NOT used for output audio — see _compress_output().
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio  # silence — don't amplify noise
    gain = target_rms / rms
    gain = min(gain, max_gain)
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > peak_ceiling:
        normalized = normalized * (peak_ceiling / peak)
    return normalized.astype(np.float32)


def _compress_output(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Windowed RMS compressor with attack/release envelope.

    Processes the signal through a proper dynamics curve:
      1. Compute windowed RMS envelope (~10ms windows)
      2. Convert to dB, apply compression curve (threshold + ratio + soft knee)
      3. Smooth the gain envelope with attack/release time constants
      4. Apply per-sample gain, then makeup gain
      5. Peak-limit to ceiling

    This preserves natural dynamics within the knee while controlling
    level differences between voices and utterances.
    """
    cfg = _get_leveling_config()
    if not cfg["enabled"]:
        return audio

    if len(audio) < 2:
        return audio

    threshold_db = cfg["output_threshold_dbfs"]
    ratio = cfg["output_ratio"]
    attack_ms = cfg["output_attack_ms"]
    release_ms = cfg["output_release_ms"]
    knee_db = cfg["output_knee_db"]
    makeup_linear = cfg["_output_makeup_linear"]
    max_gain = cfg["_output_max_gain_linear"]
    ceiling = cfg["_peak_ceiling_linear"]

    # ── Step 1: Windowed RMS envelope ──
    # ~10ms window for smooth envelope without losing transient detail
    window_samples = max(int(sample_rate * 0.010), 1)
    # Compute running mean of squared signal
    sq = audio.astype(np.float64) ** 2
    # Cumulative sum trick for fast windowed average
    cumsum = np.concatenate([[0.0], np.cumsum(sq)])
    # Pad with half-window lookahead for centered measurement
    windowed_mean = np.empty_like(sq)
    half_w = window_samples // 2
    for i in range(len(sq)):
        lo = max(0, i - half_w)
        hi = min(len(sq), i + half_w + 1)
        windowed_mean[i] = (cumsum[hi] - cumsum[lo]) / (hi - lo)
    env_rms = np.sqrt(windowed_mean).astype(np.float32)

    # ── Step 2: Convert to dB and apply compression curve ──
    # Use np.maximum to avoid log10(0) RuntimeWarning; values below 1e-6
    # are replaced with -120 dBFS anyway.
    env_db = np.where(
        env_rms > 1e-6,
        20.0 * np.log10(np.maximum(env_rms, 1e-10)),
        -120.0,
    )

    # Soft-knee compression curve:
    #   Below knee: no compression (gain_db = 0)
    #   In knee:    quadratic blend from 1:1 to ratio
    #   Above knee: full ratio compression
    half_knee = knee_db / 2.0
    knee_lo = threshold_db - half_knee
    knee_hi = threshold_db + half_knee

    gain_db = np.zeros_like(env_db)
    if knee_db > 0.01:
        # Below knee — no compression
        # In knee — quadratic interpolation
        in_knee = (env_db > knee_lo) & (env_db < knee_hi)
        x = env_db[in_knee] - knee_lo
        gain_db[in_knee] = -((1.0 - 1.0 / ratio) * x * x) / (2.0 * knee_db)
        # Above knee — full compression
        above = env_db >= knee_hi
        gain_db[above] = (threshold_db - env_db[above]) * (1.0 - 1.0 / ratio)
    else:
        # Hard knee
        above = env_db > threshold_db
        gain_db[above] = (threshold_db - env_db[above]) * (1.0 - 1.0 / ratio)

    # ── Step 3: Smooth with attack/release time constants ──
    # Convert ms to per-sample smoothing coefficients
    attack_coeff = np.exp(-1.0 / (sample_rate * attack_ms / 1000.0)) if attack_ms > 0 else 0.0
    release_coeff = np.exp(-1.0 / (sample_rate * release_ms / 1000.0)) if release_ms > 0 else 0.0

    smoothed_db = np.empty_like(gain_db)
    smoothed_db[0] = gain_db[0]
    for i in range(1, len(gain_db)):
        if gain_db[i] < smoothed_db[i - 1]:
            # Signal getting louder → gain reducing → use attack (fast)
            coeff = attack_coeff
        else:
            # Signal getting quieter → gain recovering → use release (slow)
            coeff = release_coeff
        smoothed_db[i] = coeff * smoothed_db[i - 1] + (1.0 - coeff) * gain_db[i]

    # ── Step 4: Apply gain + makeup ──
    gain_linear = (10.0 ** (smoothed_db / 20.0)) * makeup_linear
    # Clamp to max gain to prevent over-amplification of quiet passages
    gain_linear = np.minimum(gain_linear, max_gain)
    compressed = audio * gain_linear.astype(np.float32)

    # ── Step 5: Peak limiter ──
    peak = np.max(np.abs(compressed))
    if peak > ceiling:
        compressed = compressed * (ceiling / peak)

    return compressed.astype(np.float32)


def _normalize_voice_ref(audio: np.ndarray) -> np.ndarray:
    """Normalize a voice reference to the configured target level.

    Uses static RMS normalization — appropriate for short reference clips
    where temporal dynamics don't apply.
    """
    cfg = _get_leveling_config()
    if not cfg["enabled"]:
        return audio
    return _normalize_rms(
        audio,
        target_rms=cfg["_ref_target_linear"],
        max_gain=cfg["_ref_max_gain_linear"],
        peak_ceiling=cfg["_peak_ceiling_linear"],
    )


def _normalize_output(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Two-stage output processing: level-match, then compress dynamics.

    Stage 1 — Static normalization: brings all audio to the threshold RMS
    regardless of source.  This closes the ~15 dB gap between the default
    voice (-40 dBFS) and cloned voices (-27 dBFS).

    Stage 2 — Compressor: smooths within-utterance dynamics (attack/release)
    on the already-leveled signal.
    """
    cfg = _get_leveling_config()
    if not cfg["enabled"]:
        return audio
    # Stage 1: match levels — bring everything to the threshold target
    audio = _normalize_rms(
        audio,
        target_rms=cfg["_output_threshold_linear"],
        max_gain=cfg["_output_max_gain_linear"],
        peak_ceiling=cfg["_peak_ceiling_linear"],
    )
    # Stage 2: dynamics compression (attack/release/knee)
    return _compress_output(audio, sample_rate)


def _build_voice_system_msg(model, voice_ref: Optional[np.ndarray]) -> dict:
    """
    Build the system message for TTS using the model's own get_sys_prompt().

    This guarantees exact format compatibility with the model's trained prompts.
    """
    return model.get_sys_prompt(ref_audio=voice_ref, mode="audio_assistant")


def chat(
    messages: list[dict],
    voice_ref: Optional[np.ndarray] = None,
    generate_audio: bool = True,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.2,
) -> dict:
    """
    Single-turn chat with optional voice cloning.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                  Content can include text strings, PIL Images, or numpy audio arrays.
        voice_ref: Optional 16kHz mono numpy array for voice cloning.
                   If provided, response will use this voice.
        generate_audio: If True, generate spoken audio response.
        output_audio_path: If set, save audio response to this path.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.

    Returns:
        dict with keys:
            text: str — the text response
            audio: np.ndarray | None — audio waveform (24kHz) if generated
            audio_path: str | None — path to saved audio file
    """
    global _turn_count
    _turn_count += 1
    turn = _turn_count

    model, _tok = get_model()

    _reset_for_generation(model, generate_audio, output_audio_path)

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    voice_label = "custom" if voice_ref is not None else "default"
    print(f"  [chat] turn={turn} voice={voice_label} generate_audio={generate_audio}")

    # When voice cloning, re-initialize the Token2wav vocoder cache with the
    # reference audio so the vocoder synthesizes in that voice.  The system
    # prompt (via get_sys_prompt) tells the *language model* about the voice
    # style, but the *vocoder* needs its own reference waveform.
    if voice_ref is not None and generate_audio:
        model.init_token2wav_cache(voice_ref)

    # Build message list using model's own get_sys_prompt() for exact format
    msgs = []
    if generate_audio:
        msgs.append(_build_voice_system_msg(model, voice_ref))
    msgs.extend(messages)

    # Generate response
    text = model.chat(
        msgs=msgs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        use_tts_template=generate_audio,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
    )

    safe_text = text[:120].encode("ascii", errors="replace").decode("ascii")
    print(f"  [chat] turn={turn} response: {safe_text}{'...' if len(str(text)) > 120 else ''}")

    # If we used a custom voice, restore the default-voice cache so subsequent
    # calls without voice_ref don't accidentally use the cloned voice.
    if voice_ref is not None and generate_audio:
        _init_audio = np.zeros(16000, dtype=np.float32)
        model.init_token2wav_cache(_init_audio)

    result = {"text": text, "audio": None, "audio_path": output_audio_path}

    # If audio was saved to file, load it back for playback
    if output_audio_path and Path(output_audio_path).exists():
        import soundfile as sf

        audio_data, sr = sf.read(output_audio_path, dtype="float32")
        # Smooth the HiFT vocoder's cold-start glitch at the beginning
        audio_data = _apply_fade_in(audio_data)
        # Normalize output volume so default and cloned voices are similar
        audio_data = _normalize_output(audio_data)
        result["audio"] = audio_data
        result["sample_rate"] = sr
        print(f"  [chat] turn={turn} audio: {len(audio_data)} samples, {sr}Hz, "
              f"{len(audio_data)/sr:.1f}s")
    else:
        print(f"  [chat] turn={turn} NO AUDIO FILE at {output_audio_path}")

    return result


# ── Streaming generation ─────────────────────────────────────────────────

# Sample rate for all MiniCPM-o TTS audio output.
_STREAMING_SR = 24000


def chat_streaming(
    messages: list[dict],
    voice_ref: Optional[np.ndarray] = None,
    generate_audio: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.2,
) -> Generator[tuple[Optional[np.ndarray], str], None, None]:
    """
    Streaming chat — yields (audio_chunk, text_chunk) tuples as the model generates.

    Uses the model's streaming_prefill() + streaming_generate() API. Audio chunks
    are ~1 second of 24kHz float32 numpy arrays.  No file I/O is involved.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        voice_ref: Optional 16kHz mono numpy array for voice cloning.
        generate_audio: If True, generate spoken audio (yields waveform chunks).
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.

    Yields:
        (audio_chunk, text_chunk) where:
          - audio_chunk: np.ndarray (float32, 24kHz) or None
          - text_chunk: str — incremental new text since last yield
    """
    global _turn_count
    _turn_count += 1
    turn = _turn_count

    model, _tok = get_model()

    # Don't call _reset_for_generation — streaming_prefill handles reset
    # internally when it detects a new session_id. But we do need to free
    # GPU cache and set up voice cloning before prefill.
    torch.cuda.empty_cache()

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    voice_label = "custom" if voice_ref is not None else "default"
    print(f"  [stream] turn={turn} voice={voice_label} generate_audio={generate_audio}")

    # Init Token2wav cache BEFORE streaming_prefill.
    # streaming_prefill calls reset_session() which clears the cache,
    # so we must (re)initialize it here for BOTH custom and default voice.
    if generate_audio:
        cache_audio = voice_ref if voice_ref is not None else np.zeros(16000, dtype=np.float32)
        model.init_token2wav_cache(cache_audio)

    # Build full message list with TTS system prompt
    msgs = []
    if generate_audio:
        msgs.append(_build_voice_system_msg(model, voice_ref))
    msgs.extend(messages)

    # Prefill each message one at a time (streaming_prefill requires len(msgs)==1).
    #
    # Audio handling: streaming_prefill uses online_streaming=True for user
    # messages containing audio.  This mode expects audio in proper-sized
    # chunks (FIRST_CHUNK_MS=1035 then CHUNK_MS=1000).  Sending a complete
    # recording as one chunk causes a tensor size mismatch because the
    # processor creates placeholder tokens based on streaming assumptions.
    #
    # Fix: split audio into chunks and call streaming_prefill once per chunk.
    # The model handles this natively — after the first user segment it sets
    # new_user_msg=False so subsequent chunks skip the role prefix.
    FIRST_CHUNK_SAMPLES = int(1035 * 16000 / 1000)   # 16,560
    REGULAR_CHUNK_SAMPLES = int(1000 * 16000 / 1000)  # 16,000

    session_id = f"omnichat_stream_{turn}"
    for i, msg in enumerate(msgs):
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Separate audio arrays from other content (text, images)
        audio_arrays = [c for c in content if isinstance(c, np.ndarray)]
        other_content = [c for c in content if not isinstance(c, np.ndarray)]

        if not audio_arrays:
            # Text-only message — single prefill call
            model.streaming_prefill(
                session_id=session_id,
                msgs=[{"role": msg["role"], "content": content}],
                use_tts_template=generate_audio,
                is_last_chunk=False,
            )
        else:
            # Audio message — split into chunks for streaming prefill.
            # Concatenate all audio arrays (usually just one).
            full_audio = (np.concatenate(audio_arrays)
                          if len(audio_arrays) > 1 else audio_arrays[0])
            full_audio = full_audio.astype(np.float32)

            # Split: first chunk = FIRST_CHUNK_SAMPLES, rest = REGULAR_CHUNK_SAMPLES.
            # Every chunk MUST be exactly the expected size — the model's
            # placeholder/embedding alignment requires it.  Zero-pad the
            # last chunk if it's shorter than REGULAR_CHUNK_SAMPLES.
            chunks: list[np.ndarray] = []
            n = len(full_audio)
            if n <= FIRST_CHUNK_SAMPLES:
                # Short clip — pad to exactly FIRST_CHUNK_SAMPLES
                if n < FIRST_CHUNK_SAMPLES:
                    padded = np.zeros(FIRST_CHUNK_SAMPLES, dtype=np.float32)
                    padded[:n] = full_audio
                    chunks.append(padded)
                else:
                    chunks.append(full_audio)
            else:
                chunks.append(full_audio[:FIRST_CHUNK_SAMPLES])
                pos = FIRST_CHUNK_SAMPLES
                while pos < n:
                    chunk = full_audio[pos:pos + REGULAR_CHUNK_SAMPLES]
                    if len(chunk) < REGULAR_CHUNK_SAMPLES:
                        # Zero-pad to exact size
                        padded = np.zeros(REGULAR_CHUNK_SAMPLES, dtype=np.float32)
                        padded[:len(chunk)] = chunk
                        chunk = padded
                    chunks.append(chunk)
                    pos += REGULAR_CHUNK_SAMPLES

            n_chunks = len(chunks)
            print(f"  [stream] audio split into {n_chunks} chunk(s): "
                  f"{[len(c) for c in chunks]} samples")

            for ci, chunk in enumerate(chunks):
                is_last = (ci == n_chunks - 1)
                if ci == 0:
                    # First chunk: include text alongside audio
                    chunk_content = other_content + [chunk]
                else:
                    # Subsequent chunks: audio only (model skips role prefix)
                    chunk_content = [chunk]
                model.streaming_prefill(
                    session_id=session_id,
                    msgs=[{"role": msg["role"], "content": chunk_content}],
                    use_tts_template=generate_audio,
                    is_last_chunk=is_last,
                )

    # Stream-generate text + audio
    _audio_chunks = 0
    _text_chunks = 0
    _total_audio_samples = 0
    for waveform_chunk, new_text in model.streaming_generate(
        session_id=session_id,
        generate_audio=generate_audio,
        max_new_tokens=max_new_tokens,
        use_tts_template=generate_audio,
        do_sample=True,
        repetition_penalty=repetition_penalty,
    ):
        # streaming_generate yields (waveform_chunk, new_text) when generate_audio=True,
        # but (text_str, bool) when generate_audio=False.  Detect audio by type.
        if hasattr(waveform_chunk, "cpu"):
            # Torch tensor — convert to 1D float32 numpy
            chunk_np = waveform_chunk.cpu().numpy().flatten().astype(np.float32)
            _audio_chunks += 1
            _total_audio_samples += len(chunk_np)
            yield chunk_np, new_text if isinstance(new_text, str) else ""
        elif isinstance(waveform_chunk, np.ndarray):
            chunk_np = waveform_chunk.flatten().astype(np.float32)
            _audio_chunks += 1
            _total_audio_samples += len(chunk_np)
            yield chunk_np, new_text if isinstance(new_text, str) else ""
        elif isinstance(waveform_chunk, str) and waveform_chunk:
            # Text-only mode: first element is text, second is bool (not str)
            _text_chunks += 1
            yield None, waveform_chunk
        elif isinstance(new_text, str) and new_text:
            _text_chunks += 1
            yield None, new_text

    print(f"  [stream] turn={turn} complete: {_audio_chunks} audio chunks "
          f"({_total_audio_samples} samples, {_total_audio_samples/24000:.1f}s), "
          f"{_text_chunks} text-only chunks")

    # Restore default voice cache after voice cloning
    if voice_ref is not None and generate_audio:
        _init_audio = np.zeros(16000, dtype=np.float32)
        model.init_token2wav_cache(_init_audio)


def chat_streaming_with_playback(
    messages: list[dict],
    voice_ref: Optional[np.ndarray] = None,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.2,
    headless: bool = False,
    on_text_chunk: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Stream-generate and play audio simultaneously through speakers.

    Audio starts playing as soon as the first chunk arrives (~1-2s), while the
    model continues generating. Returns the same dict interface as chat().

    Args:
        messages: List of message dicts.
        voice_ref: Optional voice reference for cloning.
        output_audio_path: If set, save complete audio here AFTER playback.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens.
        headless: If True, skip audio playback (collect audio silently).
        on_text_chunk: Optional callback for incremental text updates.

    Returns:
        dict with keys: text, audio, audio_path, sample_rate
    """
    player = None
    if not headless:
        from tools.audio.streaming_player import StreamingAudioPlayer
        player = StreamingAudioPlayer(sample_rate=_STREAMING_SR)
        player.start()

    full_text = ""
    collected_chunks: list[np.ndarray] = []

    is_first_chunk = True
    cfg = _get_leveling_config()

    try:
        for audio_chunk, text_chunk in chat_streaming(
            messages=messages,
            voice_ref=voice_ref,
            generate_audio=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        ):
            if audio_chunk is not None:
                # Smooth vocoder cold-start on the very first chunk
                if is_first_chunk:
                    audio_chunk = _apply_fade_in(audio_chunk)
                    is_first_chunk = False
                # Normalize each chunk before playback so the user hears
                # leveled audio in real time (not just in the archive file).
                # Uses the static RMS normalizer (Stage 1) per-chunk —
                # appropriate since each ~1s chunk has consistent RMS.
                if cfg["enabled"]:
                    audio_chunk = _normalize_rms(
                        audio_chunk,
                        target_rms=cfg["_output_threshold_linear"],
                        max_gain=cfg["_output_max_gain_linear"],
                        peak_ceiling=cfg["_peak_ceiling_linear"],
                    )
                collected_chunks.append(audio_chunk)
                if player:
                    player.push(audio_chunk)

            if text_chunk:
                full_text += text_chunk
                if on_text_chunk:
                    on_text_chunk(text_chunk)
    finally:
        if player:
            player.finish()
            player.wait()
            player.stop()

    # Assemble result — chunks are already per-chunk normalized;
    # run full-signal compressor on the concatenated audio for the archive.
    full_audio = np.concatenate(collected_chunks) if collected_chunks else None
    if full_audio is not None:
        full_audio = _normalize_output(full_audio)

    # Archival save AFTER playback (not before)
    if output_audio_path and full_audio is not None:
        import soundfile as sf
        sf.write(output_audio_path, full_audio, _STREAMING_SR)

    return {
        "text": full_text,
        "audio": full_audio,
        "audio_path": output_audio_path,
        "sample_rate": _STREAMING_SR,
    }


def process_image(
    image,
    prompt: str = "Describe this image in detail.",
    generate_audio: bool = False,
    voice_ref: Optional[np.ndarray] = None,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    max_slice_nums: int = 1,
    repetition_penalty: float = 1.2,
) -> dict:
    """
    Analyze an image with an optional text prompt.

    Args:
        image: PIL.Image.Image
        prompt: Text prompt describing what to do with the image.
        max_slice_nums: 1 for photos, 25 for documents/PDFs.

    Returns:
        dict with text (and optionally audio).
    """
    model, _tok = get_model()

    _reset_for_generation(model, generate_audio, output_audio_path)

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    msgs = []
    if generate_audio or voice_ref is not None:
        msgs.append(_build_voice_system_msg(model, voice_ref))

    msgs.append({"role": "user", "content": [image, prompt]})

    text = model.chat(
        msgs=msgs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        use_tts_template=generate_audio,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
        max_slice_nums=max_slice_nums,
        use_image_id=False,
    )

    result = {"text": text, "audio": None, "audio_path": output_audio_path}
    if output_audio_path and Path(output_audio_path).exists():
        import soundfile as sf

        audio_data, sr = sf.read(output_audio_path, dtype="float32")
        audio_data = _apply_fade_in(audio_data)
        # Normalize output volume so default and cloned voices are similar
        audio_data = _normalize_output(audio_data)
        result["audio"] = audio_data
        result["sample_rate"] = sr

    return result


def process_video(
    video_path: str,
    prompt: str = "Describe what's happening in this video.",
    generate_audio: bool = False,
    voice_ref: Optional[np.ndarray] = None,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    repetition_penalty: float = 1.2,
) -> dict:
    """
    Analyze a video file (with audio track).

    Args:
        video_path: Path to video file.
        prompt: Text prompt describing what to analyze.

    Returns:
        dict with text (and optionally audio).
    """
    model, _tok = get_model()

    _reset_for_generation(model, generate_audio, output_audio_path)

    # Normalize voice reference to consistent level before feeding to model
    if voice_ref is not None:
        voice_ref = _normalize_voice_ref(voice_ref)

    from minicpmo.utils import get_video_frame_audio_segments

    video_frames, audio_segments, stacked_frames = get_video_frame_audio_segments(
        video_path,
        stack_frames=1,
        use_ffmpeg=True,
        adjust_audio_length=True,
    )

    # Build interleaved content: frame, audio, frame, audio, ...
    omni_contents = []
    for i in range(len(video_frames)):
        omni_contents.append(video_frames[i])
        if i < len(audio_segments):
            omni_contents.append(audio_segments[i])
        if stacked_frames is not None and i < len(stacked_frames) and stacked_frames[i] is not None:
            omni_contents.append(stacked_frames[i])

    omni_contents.append(prompt)

    msgs = []
    if generate_audio or voice_ref is not None:
        msgs.append(_build_voice_system_msg(model, voice_ref))

    msgs.append({"role": "user", "content": omni_contents})

    text = model.chat(
        msgs=msgs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        use_tts_template=generate_audio,
        generate_audio=generate_audio,
        output_audio_path=output_audio_path,
        omni_mode=True,
        max_slice_nums=1,
        use_image_id=False,
    )

    result = {"text": text, "audio": None, "audio_path": output_audio_path}
    if output_audio_path and Path(output_audio_path).exists():
        import soundfile as sf

        audio_data, sr = sf.read(output_audio_path, dtype="float32")
        audio_data = _apply_fade_in(audio_data)
        # Normalize output volume so default and cloned voices are similar
        audio_data = _normalize_output(audio_data)
        result["audio"] = audio_data
        result["sample_rate"] = sr

    return result


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MiniCPM-o 4.5 model loading")
    parser.add_argument("--text", default="Hello! Tell me a short joke.", help="Text prompt")
    parser.add_argument("--audio", action="store_true", help="Generate audio response")
    parser.add_argument("--stream", action="store_true", help="Use streaming generation (real-time playback)")
    parser.add_argument("--output", default=None, help="Save audio to this path")
    args = parser.parse_args()

    model, tok = get_model()
    print(f"\nModel loaded successfully on {next(model.parameters()).device}")

    if args.stream:
        print(f"\nStreaming: '{args.text}'")
        result = chat_streaming_with_playback(
            messages=[{"role": "user", "content": [args.text]}],
            output_audio_path=args.output,
            headless=not args.audio,
        )
    else:
        result = chat(
            messages=[{"role": "user", "content": [args.text]}],
            generate_audio=args.audio,
            output_audio_path=args.output,
        )

    print(f"\nResponse: {result['text']}")
    if result.get("audio"):
        duration = len(result["audio"]) / result.get("sample_rate", 24000)
        print(f"Audio: {duration:.1f}s at {result.get('sample_rate', 24000)}Hz")
    if result.get("audio_path"):
        print(f"Audio saved to: {result['audio_path']}")
