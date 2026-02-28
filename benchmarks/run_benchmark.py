"""Quantization benchmark orchestrator.

Spawns one Python subprocess per quantization level (sequential — one GPU model
at a time), then generates the comparison report with spectrograms.

Usage:
    python -m benchmarks.run_benchmark
    python -m benchmarks.run_benchmark --quants none,int8
    python -m benchmarks.run_benchmark --voice-ref voices/my_voice.wav
    python -m benchmarks.run_benchmark --skip-analysis
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PYTHON = sys.executable
BASE_DIR = Path(__file__).parent.parent.resolve()

QUANT_LABELS = {"none": "bf16", "int8": "int8", "int4": "int4"}


def main():
    parser = argparse.ArgumentParser(description="OmniChat Quantization Benchmark")
    parser.add_argument(
        "--quants", default="none,int8,int4",
        help="Comma-separated quantization levels (default: none,int8,int4)"
    )
    parser.add_argument(
        "--voice-ref", default=None,
        help="Path to voice reference WAV for echo_audio and cloned-voice tests"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Sampling temperature (default: 0.3 for deterministic comparison)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Custom output directory (default: benchmark_outputs/bench_TIMESTAMP)"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip report generation (run inference only)"
    )
    args = parser.parse_args()

    quants = [q.strip() for q in args.quants.split(",")]
    for q in quants:
        if q not in QUANT_LABELS:
            print(f"ERROR: Unknown quantization level: {q!r}")
            print(f"  Valid levels: {', '.join(QUANT_LABELS.keys())}")
            sys.exit(1)

    # Create timestamped output directory
    if args.output_dir:
        bench_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        bench_dir = BASE_DIR / "benchmark_outputs" / f"bench_{timestamp}"

    bench_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 56)
    print("  OMNICHAT QUANTIZATION BENCHMARK")
    print("=" * 56)
    print(f"  Quant levels : {', '.join(QUANT_LABELS[q] for q in quants)}")
    print(f"  Temperature  : {args.temperature}")
    print(f"  Voice ref    : {args.voice_ref or 'none'}")
    print(f"  Output       : {bench_dir}")
    print("=" * 56)

    # Run each quantization level in a separate subprocess
    run_results = {}
    for quant in quants:
        label = QUANT_LABELS[quant]
        quant_dir = bench_dir / label

        cmd = [
            PYTHON, "-m", "benchmarks.run_single_quant",
            "--quant", quant,
            "--output-dir", str(quant_dir),
            "--temperature", str(args.temperature),
        ]
        if args.voice_ref:
            cmd.extend(["--voice-ref", args.voice_ref])

        print(f"\n{'='*56}")
        print(f"  STARTING: {label.upper()}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*56}\n")

        result = subprocess.run(cmd, cwd=str(BASE_DIR))
        run_results[label] = result.returncode

        if result.returncode != 0:
            print(f"\n  WARNING: {label} exited with code {result.returncode}")
            print(f"  Continuing with remaining quantization levels...\n")

    # Summary of subprocess results
    print(f"\n{'='*56}")
    print("  INFERENCE COMPLETE")
    print("=" * 56)
    for label, rc in run_results.items():
        status = "OK" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  {label:6s} : {status}")
    print()

    # Generate comparison report
    if not args.skip_analysis:
        successful = [label for label, rc in run_results.items() if rc == 0]
        if len(successful) < 2:
            print("  Skipping analysis — need at least 2 successful runs to compare.")
        else:
            print(f"{'='*56}")
            print("  GENERATING COMPARISON REPORT")
            print(f"{'='*56}\n")

            result = subprocess.run([
                PYTHON, "-m", "benchmarks.analyze_results",
                "--bench-dir", str(bench_dir),
            ], cwd=str(BASE_DIR))

            if result.returncode != 0:
                print(f"\n  WARNING: Analysis exited with code {result.returncode}")

    print(f"\nBenchmark complete. Results in: {bench_dir}")

    # Exit with failure if any quant level failed
    if any(rc != 0 for rc in run_results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
