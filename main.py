"""
main.py — OmniChat entry point.
Loads settings, initializes the model, and launches the Gradio web UI.
"""

import sys
import os
import re
import uuid
from pathlib import Path

# Ensure we run from the project root regardless of working directory
BASE_DIR = Path(__file__).parent.resolve()
os.chdir(BASE_DIR)
sys.path.insert(0, str(BASE_DIR))

import yaml
import numpy as np
import gradio as gr

from tools.shared.session import (
    load_settings,
    detect_voice_command,
    normalize_audio_input,
    get_truncated_voice_ref,
)

# Configure pydub to use imageio-ffmpeg's bundled binary for Gradio streaming.
# Gradio's Audio streaming converts chunks to ADTS/AAC via pydub, which needs
# both ffmpeg (converter) and ffprobe (prober). imageio-ffmpeg only ships ffmpeg.
# Fix: monkeypatch pydub.AudioSegment.from_file to auto-detect WAV from the RIFF
# header, which skips the ffprobe call entirely (pydub only probes when format=None).
try:
    import imageio_ffmpeg, pydub
    pydub.AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

    _orig_from_file = pydub.AudioSegment.from_file.__func__

    @classmethod
    def _from_file_with_wav_detect(cls, file, format=None, **kwargs):
        if format is None and hasattr(file, "read"):
            pos = file.tell()
            header = file.read(4)
            file.seek(pos)
            if header == b"RIFF":
                format = "wav"
        return _orig_from_file(cls, file, format=format, **kwargs)

    pydub.AudioSegment.from_file = _from_file_with_wav_detect
except ImportError:
    pass  # streaming audio may not work without ffmpeg


# load_settings, detect_voice_command, normalize_audio_input, get_truncated_voice_ref
# are imported from tools.shared.session (shared with rt_main.py)


# ── Gradio app ────────────────────────────────────────────────────────────────

def build_app(settings: dict) -> gr.Blocks:
    """Build the Gradio web UI with Voice Chat, Vision, and Settings tabs."""

    from tools.model.model_manager import (
        get_model, chat, chat_streaming,
        _apply_fade_in, _normalize_rms, _get_leveling_config,
    )
    from tools.audio.voice_manager import get_voice, list_voices, add_voice, delete_voice
    from tools.audio.extract_voice import extract_audio_from_video
    from tools.audio.conversation import ConversationManager
    from tools.vision.process_media import scan_image, scan_document, analyze_video
    from tools.output.save_output import save_auto

    inference = settings.get("inference", {})
    voice_cfg = settings.get("voice_commands", {})
    audio_cfg = settings.get("audio", {})
    output_cfg = settings.get("output", {})
    streaming_cfg = audio_cfg.get("streaming", {})

    # Session state
    current_voice_ref = [None]   # mutable container for current voice audio ref
    current_voice_name = [None]  # mutable container for current voice name
    chat_history = []            # list of (role, text) tuples
    session_settings = {         # mutable settings the user can tweak in the UI
        "temperature": inference.get("temperature", 0.7),
        "max_new_tokens": inference.get("max_new_tokens", 2048),
        "repetition_penalty": inference.get("repetition_penalty", 1.05),
        "output_format": output_cfg.get("default_format", "auto"),
        "streaming_enabled": streaming_cfg.get("enabled", True),
        "voice_sample_length_s": audio_cfg.get("voice_sample_length_s", 5.0),
    }

    def _get_voice_ref():
        """Return the voice ref truncated to the configured sample length."""
        return get_truncated_voice_ref(
            current_voice_ref[0],
            session_settings["voice_sample_length_s"],
        )

    # Conversation mode (continuous voice chat with VAD)
    chat_mode_cfg = audio_cfg.get("chat_mode", {})
    conv_mgr = ConversationManager(chat_mode_cfg)
    conv_ready_audio = [None]  # mutable container for accumulated speech
    conv_interrupt = [False]   # set True by barge-in to stop model output

    # Load default voice if configured
    default_voice = audio_cfg.get("default_voice")
    if default_voice:
        result = get_voice(default_voice, voice_cfg.get("fuzzy_threshold", 0.6))
        if result["found"]:
            current_voice_ref[0] = result["audio"]
            current_voice_name[0] = result["name"]

    def _normalize_audio(audio_input):
        """Convert Gradio audio tuple to float32 mono 16kHz numpy array."""
        sr, audio_data = audio_input
        return normalize_audio_input(sr, audio_data)

    # ── Helper: extract audio result for playback ──────────────────────

    def _get_audio_output(result, output_path):
        """Extract playable audio tuple from chat result."""
        if result.get("audio") is not None:
            audio = result["audio"]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            return (result.get("sample_rate", 24000), audio)
        elif Path(output_path).exists():
            import soundfile as sf
            data, out_sr = sf.read(output_path, dtype="float32")
            return (out_sr, data)
        return None

    # ── Streaming helpers ─────────────────────────────────────────────

    def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
        """Convert numpy audio chunk to WAV bytes for Gradio streaming output."""
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _normalize_chunk(audio_chunk: np.ndarray, is_first: bool) -> np.ndarray:
        """Per-chunk fade-in + RMS normalization (mirrors chat_streaming_with_playback)."""
        cfg = _get_leveling_config()
        if is_first:
            audio_chunk = _apply_fade_in(audio_chunk)
        if cfg["enabled"]:
            audio_chunk = _normalize_rms(
                audio_chunk,
                target_rms=cfg["_output_threshold_linear"],
                max_gain=cfg["_output_max_gain_linear"],
                peak_ceiling=cfg["_peak_ceiling_linear"],
            )
        return audio_chunk

    # Minimum samples to accumulate before yielding to Gradio.
    # Each pydub -> ADTS conversion spawns an ffmpeg subprocess (~50-100ms on Windows).
    # Accumulating reduces the number of conversions and eliminates inter-chunk gaps.
    # First chunk needs the MOST buffer since HLS has zero runway at start.
    # bufferStalledError occurs when playback outpaces segment arrival.
    _STREAM_FIRST_MIN_SAMPLES = 96000  # 4 seconds at 24kHz (first chunk — needs most runway)
    _STREAM_MIN_SAMPLES = 48000        # 2 seconds at 24kHz (subsequent — HLS has buffer by now)

    def _buffered_streaming(chat_gen):
        """Wrap chat_streaming() to accumulate audio into larger blocks.

        Yields (wav_bytes_or_None, full_text) tuples. The first audio chunk
        yields immediately; subsequent chunks accumulate to _STREAM_MIN_SAMPLES
        before yielding. Fade-in and normalization are applied per model chunk.
        Text-only updates are yielded between audio flushes so the user sees
        text appearing even while audio accumulates.
        """
        buffer = []
        buffer_samples = 0
        first_audio_yielded = False
        is_first_model_chunk = True
        full_text = ""
        last_text_yielded = ""

        for audio_chunk, text_chunk in chat_gen:
            if text_chunk:
                full_text += text_chunk
            if audio_chunk is not None:
                audio_chunk = _normalize_chunk(audio_chunk, is_first_model_chunk)
                is_first_model_chunk = False
                buffer.append(audio_chunk)
                buffer_samples += len(audio_chunk)

                threshold = _STREAM_FIRST_MIN_SAMPLES if not first_audio_yielded else _STREAM_MIN_SAMPLES
                if buffer_samples >= threshold:
                    combined = np.concatenate(buffer)
                    buffer = []
                    buffer_samples = 0
                    first_audio_yielded = True
                    last_text_yielded = full_text
                    yield _numpy_to_wav_bytes(combined), full_text
            elif full_text != last_text_yielded:
                # Text arrived without audio — yield a text-only update
                last_text_yielded = full_text
                yield None, full_text

        # Flush remaining buffer
        if buffer:
            combined = np.concatenate(buffer)
            yield _numpy_to_wav_bytes(combined), full_text
        elif not first_audio_yielded:
            # No audio at all — yield text only
            yield None, full_text

    def format_history_with_partial(partial_text: str) -> str:
        """Format chat history with a partial assistant response appended."""
        entries = list(chat_history[-19:])
        entries.append(("assistant", partial_text + "..."))
        lines = []
        for role, text in entries:
            if role == "user":
                lines.append(f"**You:** {text}")
            elif role == "assistant":
                lines.append(f"**OmniChat:** {text}")
            elif role == "system":
                lines.append(f"*{text}*")
        return "\n\n".join(lines)

    # ── Voice Chat handlers ────────────────────────────────────────────

    def _detect_and_apply_voice_cmd(text_response):
        """Detect voice command in response text, apply if found.
        Returns (status_msg, dropdown_update)."""
        voice_cmd = detect_voice_command(text_response)
        if not voice_cmd or not voice_cfg.get("enabled", True):
            return "", gr.update()
        if voice_cmd == "default":
            current_voice_ref[0] = None
            current_voice_name[0] = None
            return "Voice reset to default.", gr.update(value="Default")
        voice_result = get_voice(voice_cmd, voice_cfg.get("fuzzy_threshold", 0.6))
        if voice_result["found"]:
            current_voice_ref[0] = voice_result["audio"]
            current_voice_name[0] = voice_result["name"]
            return voice_result["message"], gr.update(value=voice_result["name"])
        return voice_result["message"], gr.update()

    # ── Conversation mode handlers ─────────────────────────────────────

    def start_conversation():
        """Start continuous conversation mode."""
        _conv_chunk_count[0] = 0  # reset diagnostic counter
        conv_mgr.start()
        print("  [conversation] Started — mode:", conv_mgr.MODE_LABELS[conv_mgr.mode])
        print("  [conversation] Now click the Record button on the conversation mic")
        return conv_mgr.format_state_html()

    def stop_conversation():
        """Stop continuous conversation mode."""
        conv_mgr.stop()
        print("  [conversation] Stopped")
        return conv_mgr.format_state_html()

    def change_conv_mode(mode_str):
        """Handle mode dropdown change."""
        conv_mgr.set_mode(mode_str)
        print(f"  [conversation] Mode changed to: {mode_str}")
        return conv_mgr.format_state_html()

    def update_conv_state():
        """Timer handler — poll conversation state for display."""
        return conv_mgr.format_state_html()

    _conv_chunk_count = [0]  # mutable counter for diagnostic logging

    def on_conv_chunk(audio_tuple, trigger_val):
        """Stream handler: fast VAD processing on each mic chunk.

        Returns the updated trigger value. When speech ends and audio is
        ready for processing, the trigger increments, which fires the
        process_conv_turn handler via .change().
        """
        if audio_tuple is None or not conv_mgr.active:
            return trigger_val

        _conv_chunk_count[0] += 1
        cnt = _conv_chunk_count[0]
        try:
            sr, raw = audio_tuple
            # Log first 5 chunks and then every 20th for diagnostics
            if cnt <= 5 or cnt % 20 == 0:
                print(f"  [conv_chunk #{cnt}] sr={sr} shape={raw.shape} "
                      f"dtype={raw.dtype} state={conv_mgr.state.value}")
            audio_data = _normalize_audio(audio_tuple)
        except Exception as e:
            print(f"  [conv_chunk #{cnt}] ERROR normalizing audio: {e}")
            return trigger_val

        prev_state = conv_mgr.state
        result = conv_mgr.on_audio_chunk(audio_data)

        # Log state transitions
        if result.state != prev_state:
            print(f"  [conv_chunk #{cnt}] STATE: {prev_state.value} -> {result.state.value}")

        # Barge-in: user interrupted model mid-response
        if result.barge_in:
            conv_interrupt[0] = True
            print(f"  [conv_chunk #{cnt}] BARGE-IN — interrupting model output")
            # Don't trigger a new turn yet; the user is still speaking.
            # The audio will accumulate in conv_mgr's buffer and trigger
            # normally when the user stops speaking (silence detected).
            return trigger_val

        if result.audio_ready is not None:
            conv_ready_audio[0] = result.audio_ready
            dur = len(result.audio_ready) / 16000
            print(f"  [conversation] Speech ended — {dur:.1f}s buffered, "
                  f"trigger {trigger_val} -> {trigger_val + 1}")
            return trigger_val + 1
        return trigger_val

    def process_conv_turn(trigger_val):
        """Model processing for conversation turns. Generator for streaming audio."""
        print(f"  [conversation] process_conv_turn fired — trigger={trigger_val}")
        audio = conv_ready_audio[0]
        if audio is None:
            print(f"  [conversation] No audio ready — skipping")
            return
        conv_ready_audio[0] = None
        conv_interrupt[0] = False  # reset interrupt flag for this turn

        conv_mgr.on_model_start()
        voice_name = current_voice_name[0] or "Default"
        msgs = [{"role": "user", "content": [audio]}]

        print(f"  [conversation] Processing turn — voice='{voice_name}'")

        if session_settings["streaming_enabled"]:
            # ── Streaming path: yield WAV bytes progressively via HLS ──
            stream_gen = chat_streaming(
                messages=msgs,
                voice_ref=_get_voice_ref(),
                generate_audio=True,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
                repetition_penalty=session_settings["repetition_penalty"],
            )
            full_text = ""
            interrupted = False
            chunk_count = 0
            for wav_bytes, text in _buffered_streaming(stream_gen):
                # Check barge-in flag — user is interrupting
                if conv_interrupt[0]:
                    print(f"  [conv_stream] INTERRUPTED by barge-in at chunk #{chunk_count}")
                    interrupted = True
                    break
                chunk_count += 1
                full_text = text
                if wav_bytes is not None:
                    yield (wav_bytes,
                           format_history_with_partial(full_text),
                           f"Voice: {voice_name} | Streaming...",
                           gr.update())
                else:
                    yield (gr.update(),
                           format_history_with_partial(full_text),
                           f"Voice: {voice_name} | Generating...",
                           gr.update())

            suffix = " [interrupted]" if interrupted else ""
            print(f"  [conv_stream] done: {chunk_count} chunks{suffix}")

            # Post-streaming: voice command detection
            status_msg, dropdown_update = _detect_and_apply_voice_cmd(full_text)
            chat_history.append(("assistant", full_text + suffix))
            if status_msg:
                chat_history.append(("system", status_msg))
            conv_mgr.on_model_done()
            voice_label = current_voice_name[0] or "Default"
            status = f"Voice: {voice_label} | Conversation"
            if status_msg:
                status += f" | {status_msg}"
            print(f"  [conversation] Turn complete: {full_text[:80]}...")
            yield gr.update(), format_history(), status, dropdown_update

        else:
            # Blocking path — can't interrupt mid-generation, but on_model_done
            # will still use anti-vox cooldown for the next turn.
            output_path = str(BASE_DIR / ".tmp" / f"conv_{uuid.uuid4().hex[:8]}.wav")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result = chat(
                messages=msgs,
                voice_ref=_get_voice_ref(),
                generate_audio=True,
                output_audio_path=output_path,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
                repetition_penalty=session_settings["repetition_penalty"],
            )
            audio_output = _get_audio_output(result, output_path)
            has_audio = audio_output is not None
            print(f"  [conv_blocking] result has audio={has_audio}, "
                  f"text={result['text'][:60]}...")
            status_msg, dropdown_update = _detect_and_apply_voice_cmd(result["text"])
            chat_history.append(("assistant", result["text"]))
            if status_msg:
                chat_history.append(("system", status_msg))
            conv_mgr.on_model_done()
            voice_label = current_voice_name[0] or "Default"
            status = f"Voice: {voice_label} | Conversation"
            if status_msg:
                status += f" | {status_msg}"
            yield (
                audio_output,
                format_history(),
                status,
                dropdown_update,
            )

    def conv_ptt_press():
        """Push-to-talk: start recording."""
        conv_mgr.ptt_start()
        return conv_mgr.format_state_html()

    def conv_ptt_release(trigger_val):
        """Push-to-talk: stop recording, trigger processing."""
        audio = conv_mgr.ptt_stop()
        if audio is not None:
            conv_ready_audio[0] = audio
            print(f"  [conversation] PTT released — {len(audio)/16000:.1f}s")
            return trigger_val + 1
        return trigger_val

    # ── Single-turn Voice Chat handlers ─────────────────────────────────

    def process_audio(audio_input):
        """Handle audio input from microphone. Generator for streaming."""
        if audio_input is None:
            yield None, "No audio received.", format_history(), gr.update()
            return

        audio_data = _normalize_audio(audio_input)

        voice_name = current_voice_name[0] or "Default"
        has_ref = current_voice_ref[0] is not None
        duration = len(audio_data) / 16000
        print(f"  [process_audio] voice='{voice_name}' has_ref={has_ref} "
              f"audio={duration:.1f}s streaming={session_settings['streaming_enabled']}")

        msgs = [{"role": "user", "content": [audio_data]}]

        if session_settings["streaming_enabled"]:
            # ── Streaming path: yield WAV bytes progressively via HLS ──
            stream_gen = chat_streaming(
                messages=msgs,
                voice_ref=_get_voice_ref(),
                generate_audio=True,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
                repetition_penalty=session_settings["repetition_penalty"],
            )
            full_text = ""
            for wav_bytes, text in _buffered_streaming(stream_gen):
                full_text = text
                if wav_bytes is not None:
                    yield (wav_bytes,
                           f"Voice: {voice_name} | Streaming...",
                           format_history_with_partial(full_text),
                           gr.update())
                else:
                    yield (gr.update(),
                           f"Voice: {voice_name} | Generating...",
                           format_history_with_partial(full_text),
                           gr.update())

            # Post-streaming: voice command detection on response
            status_msg, dropdown_update = _detect_and_apply_voice_cmd(full_text)
            chat_history.append(("assistant", full_text))
            if status_msg:
                chat_history.append(("system", status_msg))
            voice_label = current_voice_name[0] or "Default"
            status = f"Voice: {voice_label}"
            if status_msg:
                status += f" | {status_msg}"
            yield gr.update(), status, format_history(), dropdown_update

        else:
            # ── Blocking path ──
            output_path = str(BASE_DIR / ".tmp" / f"response_{uuid.uuid4().hex[:8]}.wav")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            result = chat(
                messages=msgs,
                voice_ref=_get_voice_ref(),
                generate_audio=True,
                output_audio_path=output_path,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
                repetition_penalty=session_settings["repetition_penalty"],
            )

            status_msg, dropdown_update = _detect_and_apply_voice_cmd(result["text"])
            chat_history.append(("assistant", result["text"]))
            if status_msg:
                chat_history.append(("system", status_msg))
            voice_label = current_voice_name[0] or "Default"
            status = f"Voice: {voice_label}"
            if status_msg:
                status += f" | {status_msg}"
            yield _get_audio_output(result, output_path), status, format_history(), dropdown_update

    def process_text(text_input):
        """Handle text input. Generator for streaming audio."""
        if not text_input or not text_input.strip():
            yield None, "No text entered.", format_history(), gr.update()
            return

        # Check for voice command first (before model call)
        status_msg, dropdown_update = _detect_and_apply_voice_cmd(text_input)
        if status_msg:
            chat_history.append(("user", text_input))
            chat_history.append(("system", status_msg))
            voice_label = current_voice_name[0] or "Default"
            yield None, f"Voice: {voice_label} | {status_msg}", format_history(), dropdown_update
            return

        chat_history.append(("user", text_input))

        voice_name = current_voice_name[0] or "Default"
        has_ref = current_voice_ref[0] is not None
        print(f"  [process_text] voice='{voice_name}' has_ref={has_ref}"
              f" streaming={session_settings['streaming_enabled']}")

        msgs = [{"role": "user", "content": [text_input]}]

        if session_settings["streaming_enabled"]:
            # ── Streaming path: yield WAV bytes progressively via HLS ──
            stream_gen = chat_streaming(
                messages=msgs,
                voice_ref=_get_voice_ref(),
                generate_audio=True,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
                repetition_penalty=session_settings["repetition_penalty"],
            )
            full_text = ""
            for wav_bytes, text in _buffered_streaming(stream_gen):
                full_text = text
                if wav_bytes is not None:
                    yield (wav_bytes,
                           f"Voice: {voice_name} | Streaming...",
                           format_history_with_partial(full_text),
                           gr.update())
                else:
                    yield (gr.update(),
                           f"Voice: {voice_name} | Generating...",
                           format_history_with_partial(full_text),
                           gr.update())

            chat_history.append(("assistant", full_text))
            voice_label = current_voice_name[0] or "Default"
            yield gr.update(), f"Voice: {voice_label}", format_history(), gr.update()

        else:
            # ── Blocking path ──
            output_path = str(BASE_DIR / ".tmp" / f"response_{uuid.uuid4().hex[:8]}.wav")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            result = chat(
                messages=msgs,
                voice_ref=_get_voice_ref(),
                generate_audio=True,
                output_audio_path=output_path,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
                repetition_penalty=session_settings["repetition_penalty"],
            )

            chat_history.append(("assistant", result["text"]))
            voice_label = current_voice_name[0] or "Default"
            yield _get_audio_output(result, output_path), f"Voice: {voice_label}", format_history(), gr.update()

    def format_history() -> str:
        """Format chat history as markdown."""
        lines = []
        for role, text in chat_history[-20:]:
            if role == "user":
                lines.append(f"**You:** {text}")
            elif role == "assistant":
                lines.append(f"**OmniChat:** {text}")
            elif role == "system":
                lines.append(f"*{text}*")
        return "\n\n".join(lines)

    # ── Vision handlers ────────────────────────────────────────────────

    def process_image_upload(image, prompt, is_document):
        """Handle image upload for analysis."""
        if image is None:
            return "No image provided.", ""

        prompt = prompt.strip() if prompt else None

        if is_document:
            prompt = prompt or "Extract all text from this document. Preserve formatting, tables, and structure."
            result = scan_document(
                image,
                prompt=prompt,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
            )
        else:
            prompt = prompt or "Describe this image in detail."
            result = scan_image(
                image,
                prompt=prompt,
                temperature=session_settings["temperature"],
                max_new_tokens=session_settings["max_new_tokens"],
            )

        fmt_label = f"Detected format: {result['format']}"
        if result["table"]:
            fmt_label += f" ({len(result['table'])} rows)"

        return result["text"], fmt_label

    def process_video_upload(video_path, prompt):
        """Handle video upload for analysis."""
        if video_path is None:
            return "No video provided.", ""

        prompt = prompt.strip() if prompt else "Describe what's happening in this video."

        result = analyze_video(
            video_path,
            prompt=prompt,
            temperature=session_settings["temperature"],
            max_new_tokens=session_settings["max_new_tokens"],
        )

        fmt_label = f"Detected format: {result['format']}"
        return result["text"], fmt_label

    def save_vision_output(text, filename):
        """Save vision output to file."""
        if not text or not text.strip():
            return "Nothing to save."

        filename = filename.strip() if filename else None
        path = save_auto(
            text,
            fmt=session_settings["output_format"],
            filename=filename,
        )
        return f"Saved to: {path}"

    # ── Settings handlers ──────────────────────────────────────────────

    def change_voice(voice_name):
        """Handle voice dropdown change. Returns (status, dropdown_update)."""
        if not voice_name or voice_name == "Default":
            current_voice_ref[0] = None
            current_voice_name[0] = None
            print(f"  [voice] Switched to Default voice")
            return "Voice: Default", gr.update()

        result = get_voice(voice_name, voice_cfg.get("fuzzy_threshold", 0.6))
        if result["found"]:
            current_voice_ref[0] = result["audio"]
            current_voice_name[0] = result["name"]
            print(f"  [voice] Switched to '{result['name']}' "
                  f"({len(result['audio'])} samples, {len(result['audio'])/16000:.1f}s)")
            return f"Voice: {result['name']}", gr.update(value=result["name"])
        else:
            print(f"  [voice] Not found: '{voice_name}'")
            return f"Voice: Default | {result['message']}", gr.update(value="Default")

    def upload_voice(audio, name):
        """Handle voice sample upload."""
        if audio is None or not name:
            return "Please provide both audio and a name.", gr.update(choices=["Default"] + list_voices())

        sr, data = audio
        if data.dtype != np.float32:
            if np.issubdtype(data.dtype, np.integer):
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            else:
                data = data.astype(np.float32)

        path = add_voice(name, data, sr)
        voices = ["Default"] + list_voices()
        return f"Voice '{name}' saved to {path}", gr.update(choices=voices)

    def remove_voice(voice_name):
        """Delete a voice sample."""
        if not voice_name or voice_name == "Default":
            return "Cannot delete the default voice.", gr.update(choices=["Default"] + list_voices())

        deleted = delete_voice(voice_name)
        voices = ["Default"] + list_voices()
        if deleted:
            if current_voice_name[0] == voice_name:
                current_voice_ref[0] = None
                current_voice_name[0] = None
            return f"Voice '{voice_name}' deleted.", gr.update(choices=voices, value="Default")
        return f"Voice '{voice_name}' not found.", gr.update(choices=voices)

    def upload_voice_from_video(video_path, name, duration, start):
        """Extract voice sample from an MP4/video file."""
        if video_path is None or not name:
            return "Please provide both a video and a name.", gr.update(choices=["Default"] + list_voices())

        name = name.strip()
        try:
            audio = extract_audio_from_video(
                video_path,
                duration=duration or 15.0,
                start=start or 0.0,
            )
            path = add_voice(name, audio, sample_rate=16000)
            voices = ["Default"] + list_voices()
            secs = len(audio) / 16000
            return f"Voice '{name}' saved ({secs:.1f}s extracted) to {path}", gr.update(choices=voices)
        except Exception as e:
            return f"Error: {e}", gr.update(choices=["Default"] + list_voices())

    def update_temperature(val):
        """Update session temperature."""
        session_settings["temperature"] = val
        return f"Temperature: {val}"

    def update_max_tokens(val):
        """Update session max tokens."""
        session_settings["max_new_tokens"] = int(val)
        return f"Max tokens: {int(val)}"

    def update_output_format(val):
        """Update session output format."""
        session_settings["output_format"] = val
        return f"Output format: {val}"

    def update_streaming(val):
        """Toggle streaming mode."""
        session_settings["streaming_enabled"] = val
        mode = "Streaming" if val else "Full response"
        return f"Audio mode: {mode}"

    def update_repetition_penalty(val):
        """Update repetition penalty — higher values reduce voice sample echo."""
        session_settings["repetition_penalty"] = val
        return f"Repetition penalty: {val}"

    def update_voice_sample_length(val):
        """Update how many seconds of the voice clip are sent to the model."""
        session_settings["voice_sample_length_s"] = val
        ref = current_voice_ref[0]
        clip_info = ""
        if ref is not None:
            clip_s = len(ref) / 16000
            used_s = min(val, clip_s)
            clip_info = f" (clip: {clip_s:.1f}s, using: {used_s:.1f}s)"
        return f"Voice sample length: {val}s{clip_info}"

    def update_silence_threshold(val):
        """Update VAD silence threshold for turn detection."""
        conv_mgr._silence_threshold_s = val
        return f"Silence threshold: {val}s"

    def update_vad_threshold(val):
        """Update VAD speech confidence threshold."""
        conv_mgr._vad_threshold = val
        return f"VAD threshold: {val}"

    def update_echo_cooldown(val):
        """Update echo cooldown duration."""
        conv_mgr._echo_cooldown_s = val
        return f"Echo cooldown: {val}s"

    def update_antivox_boost(val):
        """Update anti-vox threshold boost."""
        conv_mgr._antivox_boost = val
        return f"Anti-vox boost: +{val}"

    def update_barge_in(val):
        """Toggle barge-in (interruption) support."""
        conv_mgr._barge_in_enabled = val
        return f"Barge-in: {'enabled' if val else 'disabled'}"

    # ── Build UI ──────────────────────────────────────────────────────────

    # JavaScript: force-play audio on every source change.
    # Gradio's autoplay can stall after the first play in some browsers.
    # This polls for new blob URLs and calls .play() as a belt-and-suspenders fix.
    _AUTOPLAY_JS = """
    () => {
        let lastSrc = '';
        setInterval(() => {
            const c = document.getElementById('audio_output');
            if (!c) return;
            const a = c.querySelector('audio');
            if (a && a.src && a.src !== lastSrc) {
                lastSrc = a.src;
                a.play().catch(() => {});
            }
        }, 300);
    }
    """

    with gr.Blocks(title="OmniChat", js=_AUTOPLAY_JS) as app:
        gr.Markdown("# OmniChat\nMultimodal voice assistant powered by MiniCPM-o 4.5")

        with gr.Tabs():
            # ── Tab 1: Voice Chat ──────────────────────────────────────
            with gr.Tab("Voice Chat"):
                # ── Conversation mode controls ──
                with gr.Group():
                    with gr.Row():
                        conv_start_btn = gr.Button(
                            "Start Conversation", variant="primary", scale=2,
                        )
                        conv_stop_btn = gr.Button(
                            "Stop", variant="stop", scale=1,
                        )
                        conv_mode = gr.Dropdown(
                            choices=["Auto-detect", "Push-to-talk", "Click per turn"],
                            value="Auto-detect",
                            label="Mode",
                            scale=2,
                        )
                    conv_state_html = gr.HTML(
                        value=conv_mgr.format_state_html(),
                    )
                    conv_mic = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Conversation mic (press Record, then speak normally)",
                    )
                    conv_trigger = gr.Number(value=0, visible=False)
                    conv_timer = gr.Timer(value=0.5)

                # ── Single-turn controls ──
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_in = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            label="Speak (single turn)",
                        )
                        with gr.Row():
                            text_in = gr.Textbox(
                                placeholder="Or type a message...",
                                label="Text input",
                                scale=4,
                            )
                            send_btn = gr.Button("Send", scale=1)

                        audio_out = gr.Audio(
                            label="Response",
                            type="numpy",
                            streaming=True,
                            autoplay=True,
                            elem_id="audio_output",
                        )

                        chat_status = gr.Textbox(
                            label="Status",
                            value=f"Voice: {current_voice_name[0] or 'Default'} | Streaming: {'On' if session_settings['streaming_enabled'] else 'Off'}",
                            interactive=False,
                        )

                    with gr.Column(scale=2):
                        history_display = gr.Markdown(
                            value="*Conversation will appear here...*",
                            label="Chat History",
                        )

                        voice_choices = ["Default"] + list_voices()
                        voice_dropdown = gr.Dropdown(
                            choices=voice_choices,
                            value="Default",
                            label="Voice",
                        )

            # ── Tab 2: Vision ──────────────────────────────────────────
            with gr.Tab("Vision"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("Image"):
                                vision_image = gr.Image(
                                    type="pil",
                                    label="Upload image",
                                    sources=["upload", "webcam"],
                                )
                                vision_doc_mode = gr.Checkbox(
                                    label="Document/OCR mode (better for dense text, tables, PDFs)",
                                    value=False,
                                )
                                vision_image_prompt = gr.Textbox(
                                    placeholder="Describe this image... (optional — leave blank for default)",
                                    label="Prompt",
                                )
                                vision_image_btn = gr.Button("Analyze Image", variant="primary")

                            with gr.Tab("Video"):
                                vision_video = gr.Video(
                                    label="Upload video",
                                    sources=["upload"],
                                )
                                vision_video_prompt = gr.Textbox(
                                    placeholder="Describe what's happening... (optional)",
                                    label="Prompt",
                                )
                                vision_video_btn = gr.Button("Analyze Video", variant="primary")

                    with gr.Column(scale=2):
                        vision_output = gr.Textbox(
                            label="Analysis Output",
                            lines=20,
                            interactive=True,
                        )
                        vision_format_label = gr.Textbox(
                            label="Format",
                            interactive=False,
                        )
                        with gr.Row():
                            vision_save_name = gr.Textbox(
                                placeholder="Filename (optional)",
                                label="Save as",
                                scale=3,
                            )
                            vision_save_btn = gr.Button("Save", scale=1)
                        vision_save_status = gr.Textbox(
                            label="Save status",
                            interactive=False,
                        )

            # ── Tab 3: Settings ────────────────────────────────────────
            with gr.Tab("Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Inference")
                        temp_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            step=0.05,
                            value=session_settings["temperature"],
                            label="Temperature",
                        )
                        tokens_slider = gr.Slider(
                            minimum=128,
                            maximum=8192,
                            step=128,
                            value=session_settings["max_new_tokens"],
                            label="Max response tokens",
                        )
                        format_dropdown = gr.Dropdown(
                            choices=["auto", "markdown", "text", "excel"],
                            value=session_settings["output_format"],
                            label="Output format (for saved files)",
                        )
                        streaming_toggle = gr.Checkbox(
                            label="Enable streaming audio",
                            value=session_settings["streaming_enabled"],
                            info="Audio plays as it generates. Disable for full-response mode.",
                        )
                        rep_penalty_slider = gr.Slider(
                            minimum=1.0,
                            maximum=2.0,
                            step=0.05,
                            value=session_settings["repetition_penalty"],
                            label="Repetition penalty",
                            info="Higher values reduce voice sample echo. Model default is 1.05, recommended 1.3-1.5 with voice cloning.",
                        )
                        voice_sample_slider = gr.Slider(
                            minimum=1.0,
                            maximum=30.0,
                            step=1.0,
                            value=session_settings["voice_sample_length_s"],
                            label="Voice sample length (seconds)",
                            info="How many seconds of the voice clip to send for cloning. Longer may improve quality.",
                        )
                        settings_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                        )

                        gr.Markdown("### Conversation Mode")
                        silence_slider = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            step=0.1,
                            value=conv_mgr._silence_threshold_s,
                            label="Silence threshold (seconds)",
                            info="How long to wait after speech stops before sending to the model.",
                        )
                        vad_slider = gr.Slider(
                            minimum=0.1,
                            maximum=0.95,
                            step=0.05,
                            value=conv_mgr._vad_threshold,
                            label="VAD sensitivity",
                            info="Speech detection confidence (lower = more sensitive, higher = fewer false triggers).",
                        )
                        echo_slider = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            step=0.5,
                            value=conv_mgr._echo_cooldown_s,
                            label="Echo cooldown (seconds)",
                            info="After model responds, use elevated VAD threshold for this long to filter speaker bleed.",
                        )
                        antivox_slider = gr.Slider(
                            minimum=0.0,
                            maximum=0.5,
                            step=0.05,
                            value=conv_mgr._antivox_boost,
                            label="Anti-vox boost",
                            info="How much to raise the VAD threshold during cooldown. Higher = stricter filtering of speaker echo.",
                        )
                        barge_in_toggle = gr.Checkbox(
                            label="Enable barge-in (interruption)",
                            value=conv_mgr._barge_in_enabled,
                            info="Let the user interrupt model speech by talking. Uses elevated VAD threshold to distinguish user from speakers.",
                        )

                    with gr.Column():
                        gr.Markdown("### Voice Management")
                        with gr.Accordion("From Audio Recording", open=True):
                            voice_upload_audio = gr.Audio(
                                sources=["microphone", "upload"],
                                type="numpy",
                                label="Voice sample (5-30 seconds of speech)",
                            )
                            voice_upload_name = gr.Textbox(
                                placeholder="Person's name",
                                label="Voice name",
                            )
                            voice_upload_btn = gr.Button("Save Voice")
                            voice_upload_status = gr.Textbox(
                                label="Upload status",
                                interactive=False,
                            )

                        with gr.Accordion("From Video Clip (MP4)", open=True):
                            gr.Markdown("*Extract voice from the first N seconds of a video clip.*")
                            voice_video_file = gr.Video(
                                label="Video file",
                                sources=["upload"],
                            )
                            voice_video_name = gr.Textbox(
                                placeholder="Person's name",
                                label="Voice name",
                            )
                            with gr.Row():
                                voice_video_start = gr.Number(
                                    value=0,
                                    label="Start (seconds)",
                                    minimum=0,
                                    precision=1,
                                )
                                voice_video_duration = gr.Number(
                                    value=15,
                                    label="Duration (seconds)",
                                    minimum=1,
                                    maximum=60,
                                    precision=1,
                                )
                            voice_video_btn = gr.Button("Extract Voice")
                            voice_video_status = gr.Textbox(
                                label="Extraction status",
                                interactive=False,
                            )

                        with gr.Accordion("Delete Voice", open=False):
                            voice_delete_dropdown = gr.Dropdown(
                                choices=["Default"] + list_voices(),
                                label="Select voice to delete",
                            )
                            voice_delete_btn = gr.Button("Delete Voice", variant="stop")
                            voice_delete_status = gr.Textbox(
                                label="Delete status",
                                interactive=False,
                            )

        # ── Wire events ────────────────────────────────────────────────

        # Voice Chat tab
        audio_in.stop_recording(
            fn=process_audio,
            inputs=[audio_in],
            outputs=[audio_out, chat_status, history_display, voice_dropdown],
        )

        send_btn.click(
            fn=process_text,
            inputs=[text_in],
            outputs=[audio_out, chat_status, history_display, voice_dropdown],
        ).then(lambda: "", outputs=[text_in])

        text_in.submit(
            fn=process_text,
            inputs=[text_in],
            outputs=[audio_out, chat_status, history_display, voice_dropdown],
        ).then(lambda: "", outputs=[text_in])

        voice_dropdown.change(
            fn=change_voice,
            inputs=[voice_dropdown],
            outputs=[chat_status, voice_dropdown],
        )

        # Conversation mode
        conv_start_btn.click(
            fn=start_conversation,
            outputs=[conv_state_html],
        )

        conv_stop_btn.click(
            fn=stop_conversation,
            outputs=[conv_state_html],
        )

        conv_mode.change(
            fn=change_conv_mode,
            inputs=[conv_mode],
            outputs=[conv_state_html],
        )

        conv_mic.stream(
            fn=on_conv_chunk,
            inputs=[conv_mic, conv_trigger],
            outputs=[conv_trigger],
            stream_every=0.5,
        )

        conv_trigger.change(
            fn=process_conv_turn,
            inputs=[conv_trigger],
            outputs=[audio_out, history_display, chat_status, voice_dropdown],
        )

        conv_timer.tick(
            fn=update_conv_state,
            outputs=[conv_state_html],
        )

        # Vision tab
        vision_image_btn.click(
            fn=process_image_upload,
            inputs=[vision_image, vision_image_prompt, vision_doc_mode],
            outputs=[vision_output, vision_format_label],
        )

        vision_video_btn.click(
            fn=process_video_upload,
            inputs=[vision_video, vision_video_prompt],
            outputs=[vision_output, vision_format_label],
        )

        vision_save_btn.click(
            fn=save_vision_output,
            inputs=[vision_output, vision_save_name],
            outputs=[vision_save_status],
        )

        # Settings tab
        temp_slider.release(
            fn=update_temperature,
            inputs=[temp_slider],
            outputs=[settings_status],
        )

        tokens_slider.release(
            fn=update_max_tokens,
            inputs=[tokens_slider],
            outputs=[settings_status],
        )

        format_dropdown.change(
            fn=update_output_format,
            inputs=[format_dropdown],
            outputs=[settings_status],
        )

        streaming_toggle.change(
            fn=update_streaming,
            inputs=[streaming_toggle],
            outputs=[settings_status],
        )

        rep_penalty_slider.release(
            fn=update_repetition_penalty,
            inputs=[rep_penalty_slider],
            outputs=[settings_status],
        )

        voice_sample_slider.release(
            fn=update_voice_sample_length,
            inputs=[voice_sample_slider],
            outputs=[settings_status],
        )

        silence_slider.release(
            fn=update_silence_threshold,
            inputs=[silence_slider],
            outputs=[settings_status],
        )

        vad_slider.release(
            fn=update_vad_threshold,
            inputs=[vad_slider],
            outputs=[settings_status],
        )

        echo_slider.release(
            fn=update_echo_cooldown,
            inputs=[echo_slider],
            outputs=[settings_status],
        )

        antivox_slider.release(
            fn=update_antivox_boost,
            inputs=[antivox_slider],
            outputs=[settings_status],
        )

        barge_in_toggle.change(
            fn=update_barge_in,
            inputs=[barge_in_toggle],
            outputs=[settings_status],
        )

        voice_upload_btn.click(
            fn=upload_voice,
            inputs=[voice_upload_audio, voice_upload_name],
            outputs=[voice_upload_status, voice_dropdown],
        )

        voice_video_btn.click(
            fn=upload_voice_from_video,
            inputs=[voice_video_file, voice_video_name, voice_video_duration, voice_video_start],
            outputs=[voice_video_status, voice_dropdown],
        )

        voice_delete_btn.click(
            fn=remove_voice,
            inputs=[voice_delete_dropdown],
            outputs=[voice_delete_status, voice_delete_dropdown],
        )

    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="OmniChat — Multimodal voice assistant (Gradio)")
    parser.add_argument("--voices-dir", default=None, help="Path to voice WAV samples directory")
    parser.add_argument("--quantization", default=None, choices=["none", "int8", "int4"],
                        help="Model quantization: none (bf16, ~19GB), int8 (~10-12GB), int4 (~11GB)")
    args = parser.parse_args()

    settings = load_settings()

    # Configure voices directory (CLI overrides settings.yaml)
    from tools.audio.voice_manager import set_voices_dir
    voices_dir = args.voices_dir or settings.get("audio", {}).get("voices_dir", "voices")
    set_voices_dir(voices_dir)

    # Configure quantization (CLI overrides settings.yaml)
    from tools.model.model_manager import set_quantization, get_model
    quant = args.quantization or settings.get("model", {}).get("quantization", "none")
    set_quantization(quant)

    print("Loading model (this may take a minute on first run)...")
    get_model()  # Pre-load so it's ready when the UI starts

    print("Building Gradio UI...")
    app = build_app(settings)

    server = settings.get("server", {})
    app.launch(
        server_name=server.get("host", "127.0.0.1"),
        server_port=server.get("port", 7860),
        share=server.get("share", False),
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
