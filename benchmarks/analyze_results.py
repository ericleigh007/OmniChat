"""Analyze benchmark results and generate comparison report.

Reads WAV files from each quantization directory, computes audio metrics,
generates mel spectrogram comparison images, and writes a markdown report.

No GPU needed — runs on CPU only.

Usage:
    python -m benchmarks.analyze_results --bench-dir benchmark_outputs/bench_2026-02-28_143000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import librosa.display

try:
    # Force non-interactive backend BEFORE importing pyplot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

SAMPLE_RATE = 24000  # MiniCPM-o output rate


# ── Audio Metrics ────────────────────────────────────────────────────────

def compute_audio_metrics(audio: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compute spectral and amplitude metrics for a WAV file."""
    if len(audio) == 0:
        return {"error": "empty audio"}

    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_dbfs = 20 * np.log10(rms) if rms > 1e-10 else -120.0
    peak = float(np.max(np.abs(audio)))
    peak_dbfs = 20 * np.log10(peak) if peak > 1e-10 else -120.0

    # Spectral centroid — "brightness"
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = float(np.mean(centroid))

    # Zero-crossing rate — noise/texture
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_mean = float(np.mean(zcr))

    # Spectral bandwidth — frequency spread
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    bandwidth_mean = float(np.mean(bandwidth))

    # Spectral rolloff — high-frequency cutoff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rolloff_mean = float(np.mean(rolloff))

    return {
        "duration_s": round(duration, 2),
        "rms": round(rms, 6),
        "rms_dbfs": round(rms_dbfs, 1),
        "peak": round(peak, 6),
        "peak_dbfs": round(peak_dbfs, 1),
        "spectral_centroid_hz": round(centroid_mean, 1),
        "zero_crossing_rate": round(zcr_mean, 6),
        "spectral_bandwidth_hz": round(bandwidth_mean, 1),
        "spectral_rolloff_hz": round(rolloff_mean, 1),
    }


# ── Spectrogram Generation ──────────────────────────────────────────────

def generate_spectrogram_comparison(
    wav_paths: dict[str, Path],
    output_path: Path,
    title: str,
):
    """Generate side-by-side mel spectrograms for visual comparison.

    Args:
        wav_paths: {"bf16": Path, "int8": Path, ...} — ordered.
        output_path: Where to save the PNG.
        title: Plot title.
    """
    if not HAS_MATPLOTLIB:
        print(f"    Skipping spectrogram (matplotlib not installed): {output_path.name}")
        return

    n_plots = len(wav_paths)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4), squeeze=False)

    for idx, (label, wav_path) in enumerate(wav_paths.items()):
        audio, sr = sf.read(str(wav_path), dtype="float32")

        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr // 2)
        S_db = librosa.power_to_db(S, ref=np.max)

        ax = axes[0][idx]
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel",
            ax=ax, cmap="magma",
        )
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)" if idx == 0 else "")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Report Generation ────────────────────────────────────────────────────

def _write_markdown_report(
    report_dir: Path,
    all_metadata: dict,
    all_metrics: dict,
    all_texts: dict,
    wav_ids: set,
    spec_dir: Path,
):
    """Write the comparison report as a markdown file."""
    lines = []
    lines.append("# OmniChat Quantization Benchmark Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # GPU info from first available metadata
    for meta in all_metadata.values():
        gpu = meta.get("gpu_name", "Unknown")
        vram = meta.get("gpu_vram_total_gb", "?")
        lines.append(f"GPU: {gpu} ({vram} GB VRAM)\n")
        break

    lines.append(f"Audio: **Raw model output** (pre-leveling, pre-fade-in) for clean comparison\n")

    # ── System summary table ──
    labels = sorted(all_metadata.keys())
    lines.append("\n## System Summary\n")
    lines.append("| Metric | " + " | ".join(labels) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(labels)) + "|")

    for key, display in [
        ("vram_after_load_gb", "VRAM after load (GB)"),
        ("load_time_s", "Load time (s)"),
        ("vram_peak_gb", "Peak VRAM (GB)"),
    ]:
        vals = []
        for label in labels:
            v = all_metadata.get(label, {}).get(key, "N/A")
            vals.append(str(v))
        lines.append(f"| {display} | " + " | ".join(vals) + " |")

    # ── Text responses ──
    lines.append("\n## Text Responses\n")
    for pid in sorted(all_texts.keys()):
        texts_for_pid = all_texts[pid]
        lines.append(f"\n### {pid}\n")
        lines.append("| Quant | Response (first 300 chars) |")
        lines.append("|-------|---------------------------|")
        for label in labels:
            text = texts_for_pid.get(label, "*not available*")
            # Truncate and escape pipes for markdown table
            display = text[:300].replace("|", "\\|").replace("\n", " ")
            if len(text) > 300:
                display += "..."
            lines.append(f"| {label} | {display} |")

    # ── Audio metrics ──
    lines.append("\n## Audio Metrics\n")
    lines.append("All measurements are on **raw model output** (no leveling/normalization).\n")

    metric_display = [
        ("duration_s", "Duration (s)"),
        ("rms_dbfs", "RMS (dBFS)"),
        ("peak_dbfs", "Peak (dBFS)"),
        ("spectral_centroid_hz", "Spectral centroid (Hz)"),
        ("spectral_bandwidth_hz", "Spectral bandwidth (Hz)"),
        ("spectral_rolloff_hz", "Spectral rolloff (Hz)"),
        ("zero_crossing_rate", "Zero-crossing rate"),
    ]

    for pid in sorted(all_metrics.keys()):
        metrics_for_pid = all_metrics[pid]
        avail_labels = [l for l in labels if l in metrics_for_pid]
        if not avail_labels:
            continue

        lines.append(f"\n### {pid}\n")
        lines.append("| Metric | " + " | ".join(avail_labels) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(avail_labels)) + "|")

        for key, display in metric_display:
            vals = []
            for label in avail_labels:
                v = metrics_for_pid.get(label, {}).get(key, "N/A")
                vals.append(str(v))
            lines.append(f"| {display} | " + " | ".join(vals) + " |")

        # Link spectrogram if it exists
        spec_png = spec_dir / f"{pid}_comparison.png"
        if spec_png.exists():
            rel_path = spec_png.relative_to(report_dir)
            lines.append(f"\n![{pid} spectrograms]({rel_path})\n")

    # ── Audio file listing ──
    lines.append("\n## Audio Files for Listening\n")
    lines.append("WAV files are raw model output. Listen and compare:\n")
    for pid in sorted(wav_ids):
        for label in labels:
            lines.append(f"- `{label}/{pid}.wav`")

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report written to: {report_path}")


def generate_report(bench_dir: Path):
    """Main entry — generate the full comparison report."""
    report_dir = bench_dir / "report"
    spec_dir = report_dir / "spectrograms"
    report_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    # Discover quantization directories (in display order)
    quant_dirs = {}
    for label in ["bf16", "int8", "int4"]:
        d = bench_dir / label
        if d.exists() and (d / "metadata.json").exists():
            quant_dirs[label] = d

    if len(quant_dirs) < 1:
        print("  ERROR: No valid quantization results found.")
        sys.exit(1)

    print(f"  Found results for: {', '.join(quant_dirs.keys())}")

    if not HAS_MATPLOTLIB:
        print("  WARNING: matplotlib not installed -- spectrograms will be skipped")
        print("  Install with: pip install matplotlib")

    # Load metadata
    all_metadata = {}
    for label, d in quant_dirs.items():
        with open(d / "metadata.json", encoding="utf-8") as f:
            all_metadata[label] = json.load(f)

    # Discover WAV files and compute metrics
    all_metrics = {}   # {prompt_id: {quant_label: metrics_dict}}
    all_texts = {}     # {prompt_id: {quant_label: text_response}}
    wav_ids = set()

    for label, d in quant_dirs.items():
        for wav_file in sorted(d.glob("*.wav")):
            pid = wav_file.stem
            if pid.startswith("_raw_"):
                continue  # Skip temp files if they somehow remain
            wav_ids.add(pid)
            if pid not in all_metrics:
                all_metrics[pid] = {}

            print(f"  Computing metrics: {label}/{pid}.wav")
            audio, sr = sf.read(str(wav_file), dtype="float32")
            all_metrics[pid][label] = compute_audio_metrics(audio, sr)

        for txt_file in sorted(d.glob("*.txt")):
            pid = txt_file.stem
            if pid not in all_texts:
                all_texts[pid] = {}
            all_texts[pid][label] = txt_file.read_text(encoding="utf-8")

    # Generate spectrogram comparisons
    for pid in sorted(wav_ids):
        wav_paths = {}
        for label, d in quant_dirs.items():
            wav_path = d / f"{pid}.wav"
            if wav_path.exists():
                wav_paths[label] = wav_path

        if len(wav_paths) >= 2:
            out_png = spec_dir / f"{pid}_comparison.png"
            print(f"  Generating spectrogram: {out_png.name}")
            generate_spectrogram_comparison(
                wav_paths, out_png,
                title=f"Mel Spectrogram: {pid}",
            )

    # Save structured metrics
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": all_metadata, "audio_metrics": all_metrics},
            f, indent=2,
        )

    # Write markdown report
    _write_markdown_report(
        report_dir, all_metadata, all_metrics, all_texts, wav_ids, spec_dir,
    )

    print(f"\n  Analysis complete. Open {report_dir / 'report.md'} to review.")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze quantization benchmark results"
    )
    parser.add_argument(
        "--bench-dir", required=True,
        help="Path to benchmark output directory (contains bf16/, int8/, int4/ subdirs)"
    )
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    if not bench_dir.exists():
        print(f"ERROR: Directory not found: {bench_dir}")
        sys.exit(1)

    print(f"\n{'='*52}")
    print("  BENCHMARK ANALYSIS")
    print(f"  Source: {bench_dir}")
    print(f"{'='*52}\n")

    generate_report(bench_dir)


if __name__ == "__main__":
    main()
