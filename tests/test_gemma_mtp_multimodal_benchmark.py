import json
from pathlib import Path

from benchmarks import evaluate_gemma_mtp_quality
from benchmarks import gemma_mtp_multimodal as bench


def test_build_cases_creates_varied_suite(tmp_path):
    cases = bench.build_cases(tmp_path / "assets")

    assert len(cases) >= 44
    assert len(cases) <= 50
    modalities = {case["modality"] for case in cases}
    assert {"text", "image", "ocr", "audio_output"}.issubset(modalities)
    assert (tmp_path / "assets" / "shapes_primary.png").exists()
    assert (tmp_path / "assets" / "invoice_a.png").exists()


def test_write_reports_creates_readme_ready_outputs(tmp_path):
    baseline = tmp_path / "gemma4-baseline"
    mtp = tmp_path / "gemma4-mtp"
    baseline.mkdir()
    mtp.mkdir()

    baseline_rows = [
        {
            "id": "text_math_01",
            "modality": "text",
            "description": "Math prompt",
            "expected": "42",
            "contains_expected": True,
            "generation_time_s": 2.0,
            "text_length": 2,
            "text_preview": "42",
            "audio_generated": False,
            "error": None,
        },
        {
            "id": "image_shapes_01",
            "modality": "image",
            "description": "Image prompt",
            "expected": None,
            "contains_expected": None,
            "generation_time_s": 4.0,
            "text_length": 20,
            "text_preview": "red circle and blue rectangle",
            "audio_generated": False,
            "error": None,
        },
    ]
    mtp_rows = [
        {**baseline_rows[0], "generation_time_s": 1.0},
        {**baseline_rows[1], "generation_time_s": 2.0},
    ]

    (baseline / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in baseline_rows),
        encoding="utf-8",
    )
    (mtp / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in mtp_rows),
        encoding="utf-8",
    )

    bench.write_reports(tmp_path)

    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["overall"]["case_count"] == 2
    assert comparison["overall"]["speedup"] == 2.0
    assert (tmp_path / "comparison.csv").exists()

    report = (tmp_path / "report" / "report.md").read_text(encoding="utf-8")
    assert "Gemma 4 Baseline vs Gemma 4 MTP Multimodal Benchmark" in report
    assert "2.00x" in report


def test_quality_evaluator_scores_saved_outputs(tmp_path):
    for label in ("gemma4-baseline", "gemma4-mtp"):
        run_dir = tmp_path / label
        run_dir.mkdir()
        (run_dir / "metadata.json").write_text(
            json.dumps({
                "results": [
                    {
                        "id": "text_math_01",
                        "modality": "text",
                        "audio_generated": False,
                    },
                    {
                        "id": "tts_01",
                        "modality": "audio_output",
                        "audio_generated": True,
                    },
                ]
            }),
            encoding="utf-8",
        )
        (run_dir / "text_math_01.txt").write_text("42", encoding="utf-8")
        (run_dir / "tts_01.txt").write_text("Benchmark test one is complete.", encoding="utf-8")

    (tmp_path / "comparison.json").write_text(
        json.dumps({
            "overall": {
                "baseline": {"total": 3.0, "mean": 1.5, "median": 1.5},
                "mtp": {"total": 2.0, "mean": 1.0, "median": 1.0},
                "speedup": 1.5,
            },
            "by_modality": {
                "text": {
                    "baseline": {"count": 1, "total": 1.0},
                    "mtp": {"total": 0.5},
                    "speedup": 2.0,
                }
            },
        }),
        encoding="utf-8",
    )

    payload = evaluate_gemma_mtp_quality.evaluate(tmp_path)
    evaluate_gemma_mtp_quality.write_outputs(tmp_path, payload)

    assert payload["overall"]["baseline_quality"] == 1.0
    assert payload["overall"]["mtp_quality"] == 1.0
    assert payload["overall"]["ties"] == 2
    assert (tmp_path / "quality_comparison.json").exists()
    assert (tmp_path / "report" / "quality_report.md").exists()
    assert (tmp_path / "report" / "speed_quality_report.md").exists()
