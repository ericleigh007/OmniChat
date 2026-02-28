"""Run all benchmark prompts at a single quantization level.

Invoked as a subprocess by run_benchmark.py:
    python -m benchmarks.run_single_quant --quant none --output-dir benchmark_outputs/bench_xxx/bf16

Loads the model ONCE at the specified quantization, runs all prompts, saves
raw (pre-leveling) text and audio outputs, then exits so GPU memory is freed.
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from benchmarks.prompts import (
    get_all_prompts, build_echo_audio_messages,
)


def _run_chat_prompt(prompt_def, output_dir, temperature, voice_ref=None):
    """Run a prompt through chat() and save raw outputs.

    Returns a result dict with timing, audio info, and success status.
    """
    from tools.model.model_manager import chat

    pid = prompt_def["id"]
    generate_audio = prompt_def["generate_audio"]

    # For audio prompts, use a temp path so the model writes raw audio there
    audio_tmp = output_dir / f"_raw_{pid}.wav" if generate_audio else None

    start = time.time()
    result = chat(
        messages=prompt_def["messages"],
        voice_ref=voice_ref,
        generate_audio=generate_audio,
        output_audio_path=str(audio_tmp) if audio_tmp else None,
        temperature=temperature,
    )
    elapsed = time.time() - start

    text = result.get("text", "")

    # Save text output
    (output_dir / f"{pid}.txt").write_text(text, encoding="utf-8")

    info = {
        "id": pid,
        "description": prompt_def["description"],
        "generation_time_s": round(elapsed, 2),
        "text_length": len(text),
        "audio_generated": False,
        "error": None,
    }

    # Save raw audio (the file on disk IS the raw model output — chat() only
    # applies leveling to the in-memory copy, not the file)
    if audio_tmp and audio_tmp.exists():
        raw_audio, sr = sf.read(str(audio_tmp), dtype="float32")
        final_path = output_dir / f"{pid}.wav"
        sf.write(str(final_path), raw_audio, sr)
        audio_tmp.unlink()  # Clean up temp file

        info["audio_generated"] = True
        info["audio_duration_s"] = round(len(raw_audio) / sr, 2)
        info["audio_samples"] = len(raw_audio)
        info["audio_sample_rate"] = sr
        print(f"    Audio: {info['audio_duration_s']}s, {sr}Hz, {len(raw_audio)} samples")
    elif generate_audio:
        print(f"    WARNING: Audio was requested but no file was produced")

    return info


def _run_streaming_prompt(prompt_def, output_dir, temperature, voice_ref=None):
    """Run a prompt through chat_streaming() and save raw outputs.

    Collects raw audio chunks directly from the streaming generator (no leveling).
    """
    from tools.model.model_manager import chat_streaming

    pid = prompt_def["id"]

    chunks = []
    full_text = ""

    start = time.time()
    for audio_chunk, text_chunk in chat_streaming(
        messages=prompt_def["messages"],
        voice_ref=voice_ref,
        generate_audio=prompt_def["generate_audio"],
        temperature=temperature,
    ):
        if audio_chunk is not None:
            chunks.append(audio_chunk)
        if text_chunk:
            full_text += text_chunk
    elapsed = time.time() - start

    # Save text
    (output_dir / f"{pid}.txt").write_text(full_text, encoding="utf-8")

    info = {
        "id": pid,
        "description": prompt_def["description"],
        "generation_time_s": round(elapsed, 2),
        "text_length": len(full_text),
        "audio_generated": False,
        "error": None,
    }

    # Save raw audio from concatenated chunks
    if chunks:
        raw_audio = np.concatenate(chunks).astype(np.float32)
        sr = 24000  # MiniCPM-o output rate
        final_path = output_dir / f"{pid}.wav"
        sf.write(str(final_path), raw_audio, sr)

        info["audio_generated"] = True
        info["audio_duration_s"] = round(len(raw_audio) / sr, 2)
        info["audio_samples"] = len(raw_audio)
        info["audio_sample_rate"] = sr
        print(f"    Audio: {info['audio_duration_s']}s, {sr}Hz, {len(raw_audio)} samples")

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark prompts at a single quantization level"
    )
    parser.add_argument(
        "--quant", required=True, choices=["none", "int8", "int4"],
        help="Quantization mode"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--voice-ref", default=None,
        help="Path to voice reference WAV for echo_audio test"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    quant_label = {"none": "bf16", "int8": "int8", "int4": "int4"}[args.quant]
    print(f"\n{'='*52}")
    print(f"  BENCHMARK WORKER: {quant_label.upper()}")
    print(f"  Quantization: {args.quant}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Output: {output_dir}")
    print(f"{'='*52}\n")

    # ── Collect system info ──────────────────────────────────────────────
    metadata = {
        "quantization": args.quant,
        "quantization_label": quant_label,
        "temperature": args.temperature,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_vram_total_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        ) if torch.cuda.is_available() else 0,
    }

    # ── Set quantization and load model ──────────────────────────────────
    from tools.model.model_manager import set_quantization, get_model
    set_quantization(args.quant)

    print(f"Loading model with quantization={args.quant}...")
    load_start = time.time()
    model, tokenizer = get_model()
    load_time = time.time() - load_start

    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    metadata["load_time_s"] = round(load_time, 1)
    metadata["vram_after_load_gb"] = round(vram_after_load, 1)
    print(f"  Model loaded in {load_time:.1f}s, VRAM: {vram_after_load:.1f} GB\n")

    # ── Load voice reference (optional) ──────────────────────────────────
    voice_ref = None
    if args.voice_ref and Path(args.voice_ref).exists():
        voice_ref, vr_sr = sf.read(args.voice_ref, dtype="float32")
        # Resample to 16kHz if needed
        if vr_sr != 16000:
            import librosa
            voice_ref = librosa.resample(voice_ref, orig_sr=vr_sr, target_sr=16000)
        # Mono
        if voice_ref.ndim > 1:
            voice_ref = voice_ref.mean(axis=1)
        metadata["voice_ref"] = args.voice_ref
        print(f"  Voice ref: {args.voice_ref} ({len(voice_ref)/16000:.1f}s)")

    # ── Run prompts ──────────────────────────────────────────────────────
    all_prompts = get_all_prompts()
    results = []

    for i, prompt_def in enumerate(all_prompts, 1):
        pid = prompt_def["id"]

        # Skip voice-ref-dependent prompts if no voice ref
        if prompt_def.get("needs_voice_ref") and voice_ref is None:
            print(f"[{i}/{len(all_prompts)}] {pid} -- SKIPPED (no --voice-ref)")
            results.append({
                "id": pid,
                "description": prompt_def["description"],
                "skipped": True,
                "reason": "no voice reference provided",
            })
            continue

        print(f"[{i}/{len(all_prompts)}] {pid}: {prompt_def['description']}")

        try:
            # Build messages for echo_audio (needs voice ref injected)
            if pid == "echo_audio":
                prompt_def = dict(prompt_def)  # Don't mutate original
                prompt_def["messages"] = build_echo_audio_messages(voice_ref)

            if prompt_def.get("use_streaming"):
                info = _run_streaming_prompt(
                    prompt_def, output_dir, args.temperature, voice_ref=None
                )
            else:
                info = _run_chat_prompt(
                    prompt_def, output_dir, args.temperature,
                    voice_ref=None,  # Echo prompts use default voice for fair comparison
                )
            results.append(info)
            print(f"    OK ({info['generation_time_s']}s)")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"    ERROR: {e}")
            print(f"    {tb}")
            results.append({
                "id": pid,
                "description": prompt_def["description"],
                "error": str(e),
                "traceback": tb,
            })

    # ── Save metadata ────────────────────────────────────────────────────
    metadata["results"] = results
    metadata["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 1)

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    ok_count = sum(1 for r in results if not r.get("error") and not r.get("skipped"))
    err_count = sum(1 for r in results if r.get("error"))
    skip_count = sum(1 for r in results if r.get("skipped"))
    print(f"\n{'='*52}")
    print(f"  {quant_label.upper()} COMPLETE: {ok_count} OK, {err_count} errors, {skip_count} skipped")
    print(f"  Peak VRAM: {metadata['vram_peak_gb']} GB")
    print(f"{'='*52}\n")

    if err_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
