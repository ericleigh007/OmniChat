"""OmniChat Live Demo -- showcase every capability of MiniCPM-o 4.5.

Run:
    cd OmniChat
    python -m demos.run_demo                    # full demo (file-based audio)
    python -m demos.run_demo --stream           # streaming audio (real-time playback)
    python -m demos.run_demo --headless         # no audio playback, no image display
    python -m demos.run_demo --no-audio         # skip audio playback only
    python -m demos.run_demo --strict           # exit on first failure
    python -m demos.run_demo --acts 1,3,5       # run specific acts only
"""

import argparse
import difflib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from demos.demo_assets import create_fake_invoice, create_geometric_scene
from demos.demo_narrative import (
    ActResult,
    banner,
    narrate,
    pass_fail,
    show_result,
    summary_table,
)


# ── Context object ───────────────────────────────────────────────────────────

@dataclass
class DemoContext:
    """Shared state passed to every act."""
    headless: bool = False
    no_audio: bool = False
    stream: bool = False
    strict: bool = False
    output_dir: str = ""
    narrator_voice: Optional[np.ndarray] = None  # voice ref for narrator (distinct from response voice)
    results: list = field(default_factory=list)


# ── Audio / display helpers ──────────────────────────────────────────────────

def _speak_narrator(text: str, ctx: DemoContext) -> None:
    """Have the model speak narration text using the narrator voice.

    Uses a distinct voice reference (e.g. eric_snyder) so the audience can
    tell narrator speech apart from model responses (which use the default voice).
    With --stream: audio starts playing as soon as the first chunk is ready.
    Without --stream: generates full audio to file then plays.
    Skipped in headless/no-audio mode.
    """
    if ctx.headless or ctx.no_audio:
        return
    try:
        msgs = [{"role": "user", "content": [f"Read this aloud exactly in English: {text}"]}]
        if ctx.stream:
            from tools.model.model_manager import chat_streaming_with_playback
            chat_streaming_with_playback(
                messages=msgs,
                voice_ref=ctx.narrator_voice,
                max_new_tokens=512,
            )
        else:
            from tools.model.model_manager import chat
            result = chat(
                messages=msgs,
                voice_ref=ctx.narrator_voice,
                generate_audio=True,
                output_audio_path=str(Path(ctx.output_dir) / "_narrator_tmp.wav"),
                max_new_tokens=512,
            )
            if result.get("audio") is not None:
                _play_audio(result["audio"], result.get("sample_rate", 24000), ctx)
    except Exception:
        pass  # narrator is best-effort


def _play_audio(audio: np.ndarray, sample_rate: int, ctx: DemoContext) -> None:
    """Play audio through speakers unless suppressed."""
    if ctx.headless or ctx.no_audio:
        narrate(f"(audio skipped -- {len(audio)} samples at {sample_rate}Hz)")
        return
    try:
        import sounddevice as sd
        sd.play(audio, sample_rate)
        sd.wait()
    except Exception as e:
        narrate(f"Audio playback failed: {e}")


def _show_image(img, ctx: DemoContext, save_path: Optional[str] = None):
    """Display image on screen unless headless. Always saves to output dir.

    Returns a subprocess handle (or None) that can be passed to _close_image()
    to dismiss the viewer after the model has answered.
    """
    if save_path:
        img.save(save_path)
        narrate(f"Image saved: {save_path}")
    if ctx.headless:
        narrate("(image display skipped -- headless mode)")
        return None
    try:
        import subprocess
        # Use mspaint on Windows — gives us a killable process handle.
        # Falls back to PIL's img.show() on other platforms.
        view_path = save_path
        if view_path is None:
            import tempfile
            fd, view_path = tempfile.mkstemp(suffix=".png")
            import os; os.close(fd)
            img.save(view_path)
        if sys.platform == "win32":
            proc = subprocess.Popen(["mspaint", view_path])
            narrate("Image displayed.")
            return proc
        else:
            img.show()
            narrate("Image displayed.")
            return None
    except Exception as e:
        narrate(f"Image display failed: {e}")
        return None


def _close_image(proc) -> None:
    """Close an image viewer opened by _show_image()."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        pass


def _save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save audio array to WAV file."""
    import soundfile as sf
    sf.write(path, audio, sample_rate)
    narrate(f"Audio saved: {path}")


def _chat_with_audio(
    messages: list[dict],
    ctx: DemoContext,
    output_path: str,
    voice_ref: Optional[np.ndarray] = None,
    max_new_tokens: int = 2048,
) -> dict:
    """Generate a response with audio — streaming or file-based depending on ctx.stream.

    With --stream: real-time playback via chat_streaming_with_playback().
    Without --stream: file-based chat() then _play_audio().
    Returns the same dict either way: {text, audio, audio_path, sample_rate}.
    """
    if ctx.stream:
        from tools.model.model_manager import chat_streaming_with_playback
        return chat_streaming_with_playback(
            messages=messages,
            voice_ref=voice_ref,
            output_audio_path=output_path,
            headless=ctx.headless or ctx.no_audio,
            max_new_tokens=max_new_tokens,
        )
    else:
        from tools.model.model_manager import chat
        result = chat(
            messages=messages,
            voice_ref=voice_ref,
            generate_audio=True,
            output_audio_path=output_path,
            max_new_tokens=max_new_tokens,
        )
        if result.get("audio") is not None:
            _play_audio(result["audio"], result.get("sample_rate", 24000), ctx)
        return result


# ── Similarity helpers (same logic as test_integration.py) ───────────────────

def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _is_echo(response: str, prompt: str, threshold: float = 0.9) -> bool:
    return _similarity(response, prompt) > threshold


# ── ACT 1: Text Chat ────────────────────────────────────────────────────────

def act_1_text_chat(ctx: DemoContext) -> ActResult:
    banner(1, "Text Chat")
    _speak_narrator("Act 1: Text Chat. Three math questions, text only.", ctx)

    from tools.model.model_manager import chat

    prompts = [
        ("If I add 100 to 1, what number do I get? Answer only with the number.", "101"),
        ("What is 50 times 2? Answer only with the number.", "100"),
        ("What is 1000 minus 1? Answer only with the number.", "999"),
    ]

    all_ok = True
    for i, (prompt, expected) in enumerate(prompts, 1):
        show_result(f"Q{i}", prompt)
        _speak_narrator(f"Question {i}: {prompt}", ctx)

        result = chat(
            messages=[{"role": "user", "content": [prompt]}],
            generate_audio=False,
        )
        text = result["text"].strip()
        echo = _is_echo(text, prompt)
        has_expected = expected in text

        show_result(f"A{i}", text[:120])

        if echo:
            show_result(f"Q{i}", "ECHO DETECTED!")
            all_ok = False
        elif not has_expected:
            show_result(f"Q{i}", f"Expected '{expected}' in response")
        else:
            show_result(f"Q{i}", "OK")

    title = "Text Chat"
    pass_fail(1, title, all_ok)
    return ActResult(1, title, all_ok)


# ── ACT 2: Text-to-Speech ───────────────────────────────────────────────────

def act_2_tts(ctx: DemoContext) -> ActResult:
    mode = "Streaming" if ctx.stream else "File-based"
    banner(2, f"Text-to-Speech ({mode})")
    _speak_narrator("Act 2: Text to Speech. Asking for a fun fact about the ocean.", ctx)

    prompt = "Please respond in English. Tell me a fun fact about the ocean."
    show_result("Prompt", prompt)

    out_wav = str(Path(ctx.output_dir) / "act2_tts.wav")
    result = _chat_with_audio(
        messages=[{"role": "user", "content": [prompt]}],
        ctx=ctx,
        output_path=out_wav,
    )

    text = result["text"]
    audio = result.get("audio")
    sr = result.get("sample_rate", 24000)

    show_result("Text", text[:200] if text else "(no text)")

    passed = True
    detail = ""

    if not text or len(text) < 5:
        passed = False
        detail = "No text response"
    elif audio is None:
        passed = False
        detail = "No audio generated"
    else:
        show_result("Audio", f"{len(audio)} samples, {sr}Hz, {len(audio)/sr:.1f}s")

    title = f"Text-to-Speech ({mode})"
    pass_fail(2, title, passed, detail=detail)
    return ActResult(2, title, passed, detail=detail)


# ── ACT 3: Voice Cloning ────────────────────────────────────────────────────

def act_3_voice_cloning(ctx: DemoContext) -> ActResult:
    mode = "Streaming" if ctx.stream else "File-based"
    banner(3, f"Voice Cloning ({mode})")
    _speak_narrator("Act 3: Voice Cloning. Using Morgan Freeman's voice to describe a sunset.", ctx)

    import soundfile as sf

    voice_path = Path(__file__).parent.parent / "voices" / "morgan_freeman.wav"
    if not voice_path.exists():
        narrate(f"Voice file not found: {voice_path}")
        title = "Voice Cloning"
        pass_fail(3, title, False, soft_fail=True, detail="Voice file missing")
        return ActResult(3, title, False, soft_fail=True, detail="Voice file missing")

    voice_ref, _ = sf.read(str(voice_path), dtype="float32")
    narrate(f"Voice reference loaded: {len(voice_ref)} samples")

    out_wav = str(Path(ctx.output_dir) / "act3_voice_clone.wav")

    try:
        result = _chat_with_audio(
            messages=[{"role": "user", "content": ["Please respond in English. Describe what a sunset looks like in two sentences."]}],
            ctx=ctx,
            output_path=out_wav,
            voice_ref=voice_ref,
        )

        text = result["text"]
        audio = result.get("audio")
        sr = result.get("sample_rate", 24000)

        show_result("Text", text[:200] if text else "(no text)")

        if audio is not None:
            show_result("Audio", f"{len(audio)} samples, {sr}Hz, {len(audio)/sr:.1f}s")
            passed = True
            detail = ""
        else:
            passed = False
            detail = "No audio generated"

    except Exception as e:
        narrate(f"Voice cloning failed: {e}")
        passed = False
        detail = str(e)[:100]

    title = f"Voice Cloning ({mode})"
    pass_fail(3, title, passed, soft_fail=not passed, detail=detail)
    return ActResult(3, title, passed, soft_fail=not passed, detail=detail)


# ── ACT 4: Vision -- Image Description ──────────────────────────────────────

def act_4_vision_image(ctx: DemoContext) -> ActResult:
    banner(4, "Vision -- Image Description")
    _speak_narrator("Act 4: Vision. Showing the model a test image and asking it to describe what it sees.", ctx)

    from tools.model.model_manager import process_image

    img = create_geometric_scene()
    img_path = str(Path(ctx.output_dir) / "act4_geometric_scene.png")
    viewer = _show_image(img, ctx, save_path=img_path)

    out_wav = str(Path(ctx.output_dir) / "act4_vision.wav")
    generate_audio = not ctx.no_audio and not ctx.headless

    result = process_image(
        image=img,
        prompt=(
            "Describe everything you see in this image in English. "
            "Read all text, identify every shape and its color, "
            "and describe them from left to right."
        ),
        generate_audio=generate_audio,
        output_audio_path=out_wav if generate_audio else None,
    )

    text = result["text"]
    show_result("Description", text[:300] if text else "(no text)")

    passed = text is not None and len(text) > 10

    if generate_audio and result.get("audio") is not None:
        sr = result.get("sample_rate", 24000)
        _save_wav(result["audio"], sr, out_wav)
        _play_audio(result["audio"], sr, ctx)

    _close_image(viewer)

    title = "Vision -- Image"
    pass_fail(4, title, passed, detail="" if passed else "Response too short")
    return ActResult(4, title, passed)


# ── ACT 5: Vision -- Document OCR ───────────────────────────────────────────

def act_5_vision_ocr(ctx: DemoContext) -> ActResult:
    banner(5, "Vision -- Document OCR")
    _speak_narrator("Act 5: Document OCR. Showing the model a fake invoice and extracting the text.", ctx)

    from tools.vision.process_media import scan_document

    img = create_fake_invoice()
    img_path = str(Path(ctx.output_dir) / "act5_fake_invoice.png")
    viewer = _show_image(img, ctx, save_path=img_path)

    result = scan_document(img, prompt="Extract all text from this invoice. Preserve the table structure.")

    _close_image(viewer)

    text = result["text"]
    fmt = result["format"]
    table = result.get("table")

    show_result("Format detected", fmt)
    # Show full extracted text so table isn't truncated
    if text:
        narrate("Extracted text:")
        for line in text.splitlines():
            show_result("  ", line)
    else:
        show_result("Extracted text", "(no text)")
    if table:
        show_result("Table rows", str(len(table)))
        for row in table:
            show_result("  Row", " | ".join(row))

    passed = text is not None and len(text) > 20
    detail = f"format={fmt}"
    if table:
        detail += f", {len(table)} table rows"

    title = "Vision -- OCR"
    pass_fail(5, title, passed, detail=detail)
    return ActResult(5, title, passed, detail=detail)


# ── ACT 6: Multi-Turn Conversation ──────────────────────────────────────────

def act_6_multi_turn(ctx: DemoContext) -> ActResult:
    mode = "Streaming" if ctx.stream else "File-based"
    banner(6, f"Multi-Turn Conversation ({mode})")
    _speak_narrator("Act 6: Multi-turn conversation. Five math questions with speech, testing for echo bugs.", ctx)

    # Use natural prompts so the model gives spoken-length answers (not bare
    # numbers).  Bare "56" is too short for the vocoder and sounds garbled.
    # Each tuple: (prompt, expected_substring_in_text)
    prompts = [
        ("Please respond in English only in a short sentence. What is 7 times 8?", "56"),
        ("Please respond in English only in a short sentence. What is 200 plus 50?", "250"),
        ("Please respond in English only in a short sentence. What is 144 divided by 12?", "12"),
        ("Please respond in English only in a short sentence. What is 99 minus 33?", "66"),
        ("Please respond in English only in a short sentence. What is 25 squared?", "625"),
    ]

    echo_count = 0
    for i, (prompt, expected) in enumerate(prompts, 1):
        # Clear visual separator between turns
        narrate(f"--- Turn {i} of {len(prompts)} ---")
        show_result(f"Turn {i}", f"Q: {prompt}")

        # Narrator reads the question in the narrator voice
        _speak_narrator(f"Question: {prompt}", ctx)

        out_wav = str(Path(ctx.output_dir) / f"act6_turn{i}.wav")
        result = _chat_with_audio(
            messages=[{"role": "user", "content": [prompt]}],
            ctx=ctx,
            output_path=out_wav,
        )

        text = result["text"].strip()
        echo = _is_echo(text, prompt)

        if echo:
            status = "ECHO!"
            echo_count += 1
        elif expected in text:
            status = "OK"
        else:
            status = "OK?"  # model answered but didn't include expected number

        show_result(f"Turn {i}", f"[{status}] A: {text[:120]}")

    passed = echo_count == 0
    detail = f"{echo_count}/5 echoes" if echo_count else "0 echoes"

    title = f"Multi-Turn ({mode})"
    pass_fail(6, title, passed, detail=detail)
    return ActResult(6, title, passed, detail=detail)


# ── Orchestration ────────────────────────────────────────────────────────────

ALL_ACTS = {
    1: act_1_text_chat,
    2: act_2_tts,
    3: act_3_voice_cloning,
    4: act_4_vision_image,
    5: act_5_vision_ocr,
    6: act_6_multi_turn,
}


def run_all_acts(ctx: DemoContext, acts: Optional[list[int]] = None) -> list[ActResult]:
    """Run selected acts (or all) and return results."""
    selected = acts or sorted(ALL_ACTS.keys())

    for act_num in selected:
        if act_num not in ALL_ACTS:
            narrate(f"Unknown act {act_num}, skipping.")
            continue

        try:
            result = ALL_ACTS[act_num](ctx)
        except Exception as e:
            title = f"Act {act_num}"
            narrate(f"Act {act_num} crashed: {e}")
            result = ActResult(act_num, title, False, detail=str(e)[:100])
            pass_fail(act_num, title, False, detail=str(e)[:100])

        ctx.results.append(result)

        if ctx.strict and not result.passed and not result.soft_fail:
            narrate("Strict mode: stopping on first hard failure.")
            break

    return ctx.results


def main():
    parser = argparse.ArgumentParser(description="OmniChat Live Demo")
    parser.add_argument("--headless", action="store_true", help="No audio playback, no image display")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio playback only")
    parser.add_argument("--stream", action="store_true", help="Use streaming audio (real-time playback during generation)")
    parser.add_argument("--strict", action="store_true", help="Exit on first hard failure")
    parser.add_argument("--acts", default=None, help="Comma-separated act numbers (e.g. 1,3,5)")
    parser.add_argument("--output-dir", default=None, help="Custom output directory")
    args = parser.parse_args()

    # Parse act selection
    selected_acts = None
    if args.acts:
        selected_acts = [int(x.strip()) for x in args.acts.split(",")]

    # Create output directory
    if args.output_dir:
        out_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = str(Path(__file__).parent.parent / "demo_outputs" / f"demo_{timestamp}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load narrator voice reference (distinct voice for prompts/narration)
    narrator_voice = None
    narrator_path = Path(__file__).parent.parent / "voices" / "eric_snyder.wav"
    if narrator_path.exists() and not args.headless and not args.no_audio:
        import soundfile as sf
        narrator_voice, _ = sf.read(str(narrator_path), dtype="float32")

    ctx = DemoContext(
        headless=args.headless,
        no_audio=args.no_audio,
        stream=args.stream,
        strict=args.strict,
        output_dir=out_dir,
        narrator_voice=narrator_voice,
    )

    audio_mode = "Streaming" if args.stream else "File-based"
    print("")
    print("=" * 52)
    print("  OMNICHAT LIVE DEMO")
    print(f"  MiniCPM-o 4.5 Capabilities Showcase")
    print(f"  Audio mode: {audio_mode}")
    print(f"  Output: {out_dir}")
    print("=" * 52)

    narrate("Loading model (this may take a minute on first run)...")
    start = time.time()

    # Trigger model load
    from tools.model.model_manager import get_model
    get_model()

    load_time = time.time() - start
    narrate(f"Model loaded in {load_time:.1f}s")

    # Run acts
    demo_start = time.time()
    results = run_all_acts(ctx, selected_acts)
    demo_time = time.time() - demo_start

    # Summary
    summary_table(results)
    narrate(f"Demo completed in {demo_time:.1f}s")
    narrate(f"All outputs saved to: {out_dir}")

    # Exit code: 0 if no hard failures
    hard_failures = [r for r in results if not r.passed and not r.soft_fail]
    sys.exit(len(hard_failures))


if __name__ == "__main__":
    main()
