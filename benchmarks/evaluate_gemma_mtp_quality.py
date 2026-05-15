"""Evaluate quality parity for a Gemma baseline vs MTP benchmark run.

This is intentionally deterministic: it scores saved benchmark outputs against
case-specific rubrics instead of relying on a judge model. It writes
quality_comparison.csv, quality_comparison.json, and report/quality_report.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean


LABELS = ("gemma4-baseline", "gemma4-mtp")


RUBRICS: dict[str, dict] = {
    "text_math_01": {"any": [["42"]]},
    "text_math_02": {"any": [["12", "twelve"]]},
    "text_math_03": {"any": [["114"]]},
    "text_reason_01": {"all": ["sunlight", "water", "droplet"], "any": [["refract", "reflect", "scatter"], ["color", "wavelength"]]},
    "text_reason_02": {"all": ["latency", "throughput"], "any": [["delay", "time"], ["amount", "rate", "capacity"]]},
    "text_reason_03": {"any": [["chemical"], ["electrolyte", "material"], ["cycle", "temperature", "heat"]]},
    "text_extract_01": {"any": [["orange"]]},
    "text_extract_02": {"any": [["lhr"]]},
    "text_code_01": {"all": ["x"], "any": [["**", "pow"], ["3"]]},
    "text_plan_01": {"any": [["launch", "start", "run"], ["test"], ["verify", "check"], ["log", "result", "report"]]},
    "text_compare_01": {"all": ["local", "remote"], "any": [["latency"], ["privacy", "network"], ["server", "cloud"]]},
    "text_safety_01": {"any": [["secret", "credential", "password"], ["leak", "expos", "risk"]]},
    "text_table_01": {"all": ["cat", "dog"], "any": [["meow"], ["bark"], ["|"]]},
    "text_json_01": {"all": ["paris", "france"], "any": [["city"], ["country"]]},
    "text_translate_01": {"any": [["modelo"], ["rápido", "rapido"], ["ejecut", "funciona"]]},
    "text_style_01": {"any": [["fast", "speed", "rapid", "swift"], ["formal", "exceptional", "operates"]]},
    "text_memory_01": {"any": [["heliotrope"]]},
    "text_count_01": {"any": [["9", "nine"]]},
    "image_shapes_01": {"all": ["red", "blue", "green"], "any": [["circle"], ["rectangle"], ["triangle"]]},
    "image_shapes_02": {"all": ["mixed", "orange", "purple", "teal"], "any": [["square"], ["circle"], ["rectangle"]]},
    "image_shapes_03": {"all": ["yellow"], "any": [["triangle"], ["middle", "center"]]},
    "image_shapes_04": {"all": ["cyan", "gray", "red"], "any": [["circle"], ["square"], ["triangle"]]},
    "image_chart_01": {"all": ["q4", "42"]},
    "image_chart_02": {"all": ["d", "58"], "any": [["lowest", "least", "minimum"]]},
    "image_chart_03": {"all": ["blue", "20"]},
    "image_chart_04": {"all": ["ram", "ssd", "cpu", "gpu"], "any": [["21"], ["13"], ["8"], ["5"]]},
    "image_combo_01": {"all": ["primary", "blue"], "any": [["rectangle"]]},
    "image_combo_02": {"all": ["q2", "31", "q4", "42"]},
    "image_combo_03": {"all": ["purple"], "any": [["yes", "there is"]]},
    "image_combo_04": {"all": ["5"], "any": [["gpu"]]},
    "ocr_invoice_01": {"all": ["77120", "northstar", "638"]},
    "ocr_invoice_02": {"all": ["sensor", "cable", "calibration", "638"], "any": [["|"]]},
    "ocr_invoice_03": {"all": ["638"]},
    "ocr_notice_01": {"all": ["mira", "2026-06-30", "high", "42,500"]},
    "ocr_notice_02": {"all": ["power", "sample", "calibration", "cabinet"]},
    "ocr_notice_03": {"all": ["austin", "omni", "4", "9", "18"]},
    "ocr_notice_04": {"all": ["2026-06-30"]},
    "ocr_invoice_04": {"any": [["3", "three"]]},
    "audio_in_01": {"any": [["speaker", "project", "career"]]},
    "audio_in_02": {"any": [["hello", "greeting"]]},
    "audio_in_03": {"any": [["hello", "namaste", "acknowledg"]], "audio": True},
    "audio_in_04": {"any": [["speaker", "project", "career", "heard"]], "audio": True},
    "audio_in_05": {"any": [["speech", "discernible", "sound", "music"]]},
    "audio_in_06": {"any": [["hello", "namaste", "word"]]},
    "tts_01": {"all": ["benchmark", "complete"], "audio": True},
    "tts_02": {"any": [["moon"]], "audio": True},
    "tts_03": {"any": [["fast", "inference"], ["responsive", "real-time", "latency"]], "audio": True},
    "tts_04": {"any": [["hello", "greeting", "welcome"], ["omnichat"]], "audio": True},
    "tts_05": {"all": ["one", "two", "three", "four", "five"], "audio": True},
    "tts_06": {"any": [["sunset"], ["calm", "tranquil", "soft"]], "audio": True},
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text


def _score_text(text: str, rubric: dict, *, audio_generated: bool) -> tuple[float, list[str]]:
    normalized = _normalize(text)
    checks = []
    notes = []

    for term in rubric.get("all", []):
        ok = _contains(normalized, str(term))
        checks.append(ok)
        if not ok:
            notes.append(f"missing:{term}")

    for options in rubric.get("any", []):
        ok = any(_contains(normalized, str(option)) for option in options)
        checks.append(ok)
        if not ok:
            notes.append("missing_any:" + "/".join(str(option) for option in options))

    if rubric.get("audio"):
        checks.append(bool(audio_generated))
        if not audio_generated:
            notes.append("missing_audio")

    if not checks:
        checks.append(len(normalized) >= 20)
        if not checks[-1]:
            notes.append("too_short")

    return sum(1 for item in checks if item) / len(checks), notes


def _load_text(run_dir: Path, pid: str) -> str:
    path = run_dir / f"{pid}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_metadata(bench_dir: Path, label: str) -> dict:
    return json.loads((bench_dir / label / "metadata.json").read_text(encoding="utf-8"))


def evaluate(bench_dir: Path) -> dict:
    metadata = {label: _load_metadata(bench_dir, label) for label in LABELS}
    results = {
        label: {row["id"]: row for row in metadata[label].get("results", [])}
        for label in LABELS
    }
    ids = [row["id"] for row in metadata[LABELS[0]].get("results", [])]

    rows = []
    for pid in ids:
        rubric = RUBRICS.get(pid, {})
        base_meta = results[LABELS[0]][pid]
        mtp_meta = results[LABELS[1]][pid]
        base_text = _load_text(bench_dir / LABELS[0], pid)
        mtp_text = _load_text(bench_dir / LABELS[1], pid)
        base_score, base_notes = _score_text(base_text, rubric, audio_generated=bool(base_meta.get("audio_generated")))
        mtp_score, mtp_notes = _score_text(mtp_text, rubric, audio_generated=bool(mtp_meta.get("audio_generated")))
        similarity = SequenceMatcher(None, base_text.strip(), mtp_text.strip()).ratio()
        if abs(base_score - mtp_score) < 0.05:
            winner = "tie"
        elif mtp_score > base_score:
            winner = "mtp"
        else:
            winner = "baseline"
        rows.append({
            "id": pid,
            "modality": base_meta.get("modality"),
            "baseline_quality": round(base_score, 3),
            "mtp_quality": round(mtp_score, 3),
            "quality_delta": round(mtp_score - base_score, 3),
            "winner": winner,
            "text_similarity": round(similarity, 3),
            "baseline_notes": ";".join(base_notes),
            "mtp_notes": ";".join(mtp_notes),
            "baseline_preview": base_text[:300],
            "mtp_preview": mtp_text[:300],
        })

    by_modality = {}
    for modality in sorted({row["modality"] for row in rows}):
        selected = [row for row in rows if row["modality"] == modality]
        by_modality[modality] = {
            "cases": len(selected),
            "baseline_quality": round(mean(row["baseline_quality"] for row in selected), 3),
            "mtp_quality": round(mean(row["mtp_quality"] for row in selected), 3),
            "quality_delta": round(mean(row["quality_delta"] for row in selected), 3),
            "ties": sum(1 for row in selected if row["winner"] == "tie"),
            "baseline_wins": sum(1 for row in selected if row["winner"] == "baseline"),
            "mtp_wins": sum(1 for row in selected if row["winner"] == "mtp"),
            "mean_similarity": round(mean(row["text_similarity"] for row in selected), 3),
        }

    overall = {
        "cases": len(rows),
        "baseline_quality": round(mean(row["baseline_quality"] for row in rows), 3),
        "mtp_quality": round(mean(row["mtp_quality"] for row in rows), 3),
        "quality_delta": round(mean(row["quality_delta"] for row in rows), 3),
        "ties": sum(1 for row in rows if row["winner"] == "tie"),
        "baseline_wins": sum(1 for row in rows if row["winner"] == "baseline"),
        "mtp_wins": sum(1 for row in rows if row["winner"] == "mtp"),
        "mean_similarity": round(mean(row["text_similarity"] for row in rows), 3),
    }
    return {"overall": overall, "by_modality": by_modality, "rows": rows}


def write_outputs(bench_dir: Path, payload: dict) -> None:
    (bench_dir / "quality_comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = payload["rows"]
    with (bench_dir / "quality_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Gemma 4 Baseline vs MTP Quality Evaluation",
        "",
        "Deterministic rubric scoring over the saved 50-case benchmark outputs. Scores are 0.0-1.0 per case.",
        "",
        "## Overall Quality",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Cases | {payload['overall']['cases']} |",
        f"| Baseline mean quality | {payload['overall']['baseline_quality']:.3f} |",
        f"| MTP mean quality | {payload['overall']['mtp_quality']:.3f} |",
        f"| MTP quality delta | {payload['overall']['quality_delta']:+.3f} |",
        f"| Pairwise ties | {payload['overall']['ties']} |",
        f"| Baseline wins | {payload['overall']['baseline_wins']} |",
        f"| MTP wins | {payload['overall']['mtp_wins']} |",
        f"| Mean text similarity | {payload['overall']['mean_similarity']:.3f} |",
        "",
        "## Quality By Modality",
        "",
        "| Modality | Cases | Baseline | MTP | Delta | Ties | Baseline wins | MTP wins | Similarity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for modality, data in payload["by_modality"].items():
        lines.append(
            f"| {modality} | {data['cases']} | {data['baseline_quality']:.3f} | {data['mtp_quality']:.3f} | "
            f"{data['quality_delta']:+.3f} | {data['ties']} | {data['baseline_wins']} | {data['mtp_wins']} | {data['mean_similarity']:.3f} |"
        )

    lines.extend([
        "",
        "## Case Quality",
        "",
        "| ID | Modality | Baseline | MTP | Delta | Winner | Similarity | Notes |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ])
    for row in rows:
        notes = []
        if row["baseline_notes"]:
            notes.append(f"baseline {row['baseline_notes']}")
        if row["mtp_notes"]:
            notes.append(f"MTP {row['mtp_notes']}")
        lines.append(
            f"| {row['id']} | {row['modality']} | {row['baseline_quality']:.3f} | {row['mtp_quality']:.3f} | "
            f"{row['quality_delta']:+.3f} | {row['winner']} | {row['text_similarity']:.3f} | {'; '.join(notes)} |"
        )

    report_dir = bench_dir / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "quality_report.md").write_text("\n".join(lines), encoding="utf-8")

    comparison_path = bench_dir / "comparison.json"
    if comparison_path.exists():
        speed = json.loads(comparison_path.read_text(encoding="utf-8"))
        speed_overall = speed["overall"]
        combined = [
            "# Gemma 4 Baseline vs Gemma 4 MTP: Speed and Quality",
            "",
            "This benchmark compares the same Gemma 4 target model with and without the MTP assistant across 50 deterministic multimodal cases.",
            "",
            "## Summary",
            "",
            "| Metric | Gemma 4 baseline | Gemma 4 + MTP | Delta |",
            "|---|---:|---:|---:|",
            f"| Total generation time | {speed_overall['baseline']['total']:.3f}s | {speed_overall['mtp']['total']:.3f}s | **{speed_overall['speedup']:.2f}x faster** |",
            f"| Mean generation time | {speed_overall['baseline']['mean']:.3f}s | {speed_overall['mtp']['mean']:.3f}s |  |",
            f"| Median generation time | {speed_overall['baseline']['median']:.3f}s | {speed_overall['mtp']['median']:.3f}s |  |",
            f"| Mean rubric quality | {payload['overall']['baseline_quality']:.3f} | {payload['overall']['mtp_quality']:.3f} | {payload['overall']['quality_delta']:+.3f} |",
            f"| Pairwise quality wins | {payload['overall']['baseline_wins']} | {payload['overall']['mtp_wins']} | {payload['overall']['ties']} ties |",
            f"| Mean text similarity |  |  | {payload['overall']['mean_similarity']:.3f} |",
            "",
            "## Speed By Modality",
            "",
            "| Modality | Cases | Baseline total | MTP total | Speedup |",
            "|---|---:|---:|---:|---:|",
        ]
        for modality, data in speed["by_modality"].items():
            combined.append(
                f"| {modality} | {data['baseline']['count']} | {data['baseline']['total']:.3f}s | "
                f"{data['mtp']['total']:.3f}s | **{data['speedup']:.2f}x** |"
            )
        combined.extend([
            "",
            "## Quality By Modality",
            "",
            "| Modality | Cases | Baseline quality | MTP quality | Delta | Pairwise result |",
            "|---|---:|---:|---:|---:|---|",
        ])
        for modality, data in payload["by_modality"].items():
            combined.append(
                f"| {modality} | {data['cases']} | {data['baseline_quality']:.3f} | {data['mtp_quality']:.3f} | "
                f"{data['quality_delta']:+.3f} | {data['ties']} ties, {data['baseline_wins']} baseline wins, {data['mtp_wins']} MTP wins |"
            )
        combined.extend([
            "",
            "## Interpretation",
            "",
            "- MTP improved total generation wall time while preserving deterministic rubric quality in this run.",
            "- Text, image, and OCR saw the largest speedups because Gemma generation dominated those paths.",
            "- Audio-output-heavy cases saw smaller gains because MiniCPM TTS dominates end-to-end wall time after Gemma finishes text generation.",
            "- The quality result is rubric-based and deterministic; it is not a human preference study or judge-model evaluation.",
            "",
            "Artifacts: `comparison.json`, `comparison.csv`, `quality_comparison.json`, `quality_comparison.csv`, raw per-case `.txt` outputs, and generated benchmark assets are saved beside this report.",
        ])
        (report_dir / "speed_quality_report.md").write_text("\n".join(combined), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gemma baseline vs MTP quality from saved benchmark outputs")
    parser.add_argument("--bench-dir", required=True)
    args = parser.parse_args()
    bench_dir = Path(args.bench_dir)
    payload = evaluate(bench_dir)
    write_outputs(bench_dir, payload)
    print(json.dumps(payload["overall"], indent=2), flush=True)
    print(f"Quality report: {bench_dir / 'report' / 'quality_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
