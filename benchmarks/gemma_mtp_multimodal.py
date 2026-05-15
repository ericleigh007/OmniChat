"""Head-to-head Gemma 4 baseline vs MTP multimodal benchmark.

Runs a fixed 50-case suite across text, image, OCR/document, audio-input, and
audio-output paths. The parent process launches one worker per model profile so
each variant loads cleanly and releases GPU memory before the next run.

Example:
    .venv\\Scripts\\python.exe -m benchmarks.gemma_mtp_multimodal --output-dir benchmark_outputs\\gemma4_mtp_multimodal_2026-05-15
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VARIANTS = [
    {
        "label": "gemma4-baseline",
        "profile": "gemma4_e4b_transformers_mincpm_tts",
        "mtp": "off",
    },
    {
        "label": "gemma4-mtp",
        "profile": "gemma4_e4b_transformers_mtp_mincpm_tts",
        "mtp": "on",
    },
]


def _font(size: int = 28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _save_shapes(path: Path, title: str, shapes: list[tuple[str, str, tuple[int, int, int, int]]]) -> None:
    img = Image.new("RGB", (720, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 25), title, fill="black", font=_font(34))
    for name, color, box in shapes:
        if "circle" in name.lower():
            draw.ellipse(box, fill=color, outline="black", width=3)
        elif "triangle" in name.lower():
            x1, y1, x2, y2 = box
            draw.polygon([(x1 + (x2 - x1) // 2, y1), (x1, y2), (x2, y2)], fill=color, outline="black")
        else:
            draw.rectangle(box, fill=color, outline="black", width=3)
        draw.text((box[0], box[3] + 12), name, fill="black", font=_font(22))
    img.save(path)


def _save_chart(path: Path, title: str, bars: list[tuple[str, int, str]]) -> None:
    img = Image.new("RGB", (760, 460), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 22), title, fill="black", font=_font(34))
    origin_x, origin_y = 80, 390
    draw.line((origin_x, 90, origin_x, origin_y), fill="black", width=3)
    draw.line((origin_x, origin_y, 720, origin_y), fill="black", width=3)
    max_value = max(v for _, v, _ in bars)
    for i, (label, value, color) in enumerate(bars):
        x = origin_x + 55 + i * 115
        h = int(250 * value / max_value)
        draw.rectangle((x, origin_y - h, x + 62, origin_y), fill=color, outline="black")
        draw.text((x - 8, origin_y + 12), label, fill="black", font=_font(18))
        draw.text((x + 8, origin_y - h - 28), str(value), fill="black", font=_font(18))
    img.save(path)


def _save_invoice(path: Path, invoice_id: str, total: str) -> None:
    img = Image.new("RGB", (820, 620), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 35), f"INVOICE #{invoice_id}", fill="black", font=_font(36))
    draw.text((40, 90), "Date: 2026-05-15", fill="black", font=_font(24))
    draw.text((40, 125), "Customer: Northstar Research", fill="black", font=_font(24))
    headers = ["Item", "Qty", "Price", "Total"]
    rows = [
        ["Sensor Kit", "3", "$120.00", "$360.00"],
        ["Cable Pack", "6", "$15.50", "$93.00"],
        ["Calibration", "1", "$185.00", "$185.00"],
        ["TOTAL", "", "", total],
    ]
    x0, y0 = 40, 190
    widths = [270, 90, 150, 150]
    row_h = 54
    x = x0
    for header, width in zip(headers, widths):
        draw.rectangle((x, y0, x + width, y0 + row_h), outline="black", width=2)
        draw.text((x + 10, y0 + 15), header, fill="black", font=_font(22))
        x += width
    for r, row in enumerate(rows, 1):
        x = x0
        for cell, width in zip(row, widths):
            y = y0 + r * row_h
            draw.rectangle((x, y, x + width, y + row_h), outline="black", width=2)
            draw.text((x + 10, y + 15), cell, fill="black", font=_font(22))
            x += width
    img.save(path)


def _save_notice(path: Path, title: str, lines: list[str]) -> None:
    img = Image.new("RGB", (820, 520), "white")
    draw = ImageDraw.Draw(img)
    draw.text((45, 38), title, fill="black", font=_font(34))
    y = 110
    for line in lines:
        draw.text((55, y), line, fill="black", font=_font(24))
        y += 48
    img.save(path)


def _ensure_assets(asset_dir: Path) -> dict[str, Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[str, Path] = {}

    shape_defs = [
        ("shapes_primary", "PRIMARY SHAPES", [("Red Circle", "red", (60, 130, 190, 260)), ("Blue Rectangle", "royalblue", (260, 145, 430, 245)), ("Green Triangle", "limegreen", (500, 125, 650, 265))]),
        ("shapes_mixed", "MIXED SYMBOLS", [("Orange Square", "orange", (70, 135, 190, 255)), ("Purple Circle", "purple", (285, 130, 420, 265)), ("Teal Rectangle", "teal", (500, 150, 665, 245))]),
        ("shapes_order", "LEFT TO RIGHT", [("Black Rectangle", "black", (60, 135, 205, 235)), ("Yellow Triangle", "gold", (290, 120, 430, 260)), ("Pink Circle", "pink", (520, 130, 655, 265))]),
        ("shapes_small", "SMALL DETAILS", [("Cyan Circle", "cyan", (70, 150, 170, 250)), ("Gray Square", "gray", (305, 150, 405, 250)), ("Red Triangle", "red", (540, 140, 655, 260))]),
    ]
    for key, title, shapes in shape_defs:
        path = asset_dir / f"{key}.png"
        _save_shapes(path, title, shapes)
        assets[key] = path

    charts = [
        ("chart_sales", "Quarterly Sales", [("Q1", 18, "steelblue"), ("Q2", 31, "seagreen"), ("Q3", 24, "orange"), ("Q4", 42, "crimson")]),
        ("chart_latency", "Latency by Build", [("A", 95, "gray"), ("B", 72, "royalblue"), ("C", 63, "seagreen"), ("D", 58, "purple")]),
        ("chart_votes", "Survey Votes", [("Red", 12, "red"), ("Blue", 20, "blue"), ("Green", 16, "green"), ("Gold", 9, "gold")]),
        ("chart_inventory", "Inventory Count", [("CPU", 8, "teal"), ("GPU", 5, "orange"), ("RAM", 21, "slateblue"), ("SSD", 13, "brown")]),
    ]
    for key, title, bars in charts:
        path = asset_dir / f"{key}.png"
        _save_chart(path, title, bars)
        assets[key] = path

    invoices = [("invoice_a", "77120", "$638.00"), ("invoice_b", "88145", "$638.00"), ("invoice_c", "99201", "$638.00")]
    for key, invoice_id, total in invoices:
        path = asset_dir / f"{key}.png"
        _save_invoice(path, invoice_id, total)
        assets[key] = path

    notices = [
        ("notice_alpha", "PROJECT NOTICE", ["Owner: Mira Chen", "Deadline: 2026-06-30", "Priority: High", "Budget: $42,500"]),
        ("notice_beta", "LAB CHECKLIST", ["1. Power on analyzer", "2. Confirm sample tray", "3. Record calibration ID", "4. Lock cabinet"]),
        ("notice_gamma", "SHIPPING LABEL", ["Destination: Austin TX", "Carrier: Omni Freight", "Box: 4 of 9", "Weight: 18 lb"]),
    ]
    for key, title, lines in notices:
        path = asset_dir / f"{key}.png"
        _save_notice(path, title, lines)
        assets[key] = path

    audio = ROOT / "outputs" / "debug" / "qwen_transformers_probe_input.wav"
    if audio.exists():
        assets["audio_probe"] = audio
    return assets


def _text_cases() -> list[dict]:
    prompts = [
        ("text_math_01", "What is 17 plus 25? Answer only with the number.", "42"),
        ("text_math_02", "What is 144 divided by 12? Answer only with the number.", "12"),
        ("text_math_03", "What is 19 times 6? Answer only with the number.", "114"),
        ("text_reason_01", "In two concise sentences, explain why rainbows appear after rain.", None),
        ("text_reason_02", "Summarize the difference between latency and throughput in two sentences.", None),
        ("text_reason_03", "Give three short bullet points about why batteries degrade over time.", None),
        ("text_extract_01", "Return only the second word from this phrase: silver orange candle river.", "orange"),
        ("text_extract_02", "Return only the airport code in this sentence: The flight lands at LHR tonight.", "LHR"),
        ("text_code_01", "Write a Python expression that squares x and adds 3. Return only the expression.", None),
        ("text_plan_01", "Give a four-step checklist for testing a desktop chat app.", None),
        ("text_compare_01", "In one paragraph, compare local inference and remote inference.", None),
        ("text_safety_01", "In one sentence, explain why logs should avoid storing secrets.", None),
        ("text_table_01", "Create a tiny markdown table with columns Animal and Sound for cat and dog.", None),
        ("text_json_01", "Return compact JSON with keys city and country for Paris, France.", "Paris"),
        ("text_translate_01", "Translate to Spanish: The model is running quickly.", None),
        ("text_style_01", "Rewrite this sentence more formally: this thing is super fast.", None),
        ("text_memory_01", "Remember the word heliotrope. What word did I ask you to remember? Answer only that word.", "heliotrope"),
        ("text_count_01", "How many letters are in the word benchmark? Answer only with the number.", "9"),
    ]
    return [
        {
            "id": pid,
            "modality": "text",
            "description": prompt,
            "prompt": prompt,
            "expected": expected,
            "generate_audio": False,
        }
        for pid, prompt, expected in prompts
    ]


def _image_cases(assets: dict[str, Path]) -> list[dict]:
    items = [
        ("image_shapes_01", assets["shapes_primary"], "List the shapes and colors from left to right."),
        ("image_shapes_02", assets["shapes_mixed"], "What title text is shown and what are the three labeled objects?"),
        ("image_shapes_03", assets["shapes_order"], "Identify the object in the middle and its color."),
        ("image_shapes_04", assets["shapes_small"], "Describe the three shapes in a concise sentence."),
        ("image_chart_01", assets["chart_sales"], "Which quarter has the highest value and what is it?"),
        ("image_chart_02", assets["chart_latency"], "Which build has the lowest latency?"),
        ("image_chart_03", assets["chart_votes"], "Which survey option received the most votes?"),
        ("image_chart_04", assets["chart_inventory"], "Rank the inventory items from highest count to lowest."),
        ("image_combo_01", assets["shapes_primary"], "Read the title and name the blue object."),
        ("image_combo_02", assets["chart_sales"], "Give the Q2 and Q4 values only."),
        ("image_combo_03", assets["shapes_mixed"], "Is there a purple object? Answer briefly."),
        ("image_combo_04", assets["chart_inventory"], "How many GPU items are shown?"),
    ]
    return [
        {
            "id": pid,
            "modality": "image",
            "description": prompt,
            "prompt": prompt,
            "asset_path": str(path),
            "generate_audio": False,
        }
        for pid, path, prompt in items
    ]


def _ocr_cases(assets: dict[str, Path]) -> list[dict]:
    items = [
        ("ocr_invoice_01", assets["invoice_a"], "Extract the invoice number, customer, and total."),
        ("ocr_invoice_02", assets["invoice_b"], "Extract the full table as markdown."),
        ("ocr_invoice_03", assets["invoice_c"], "What is the total amount? Answer briefly."),
        ("ocr_notice_01", assets["notice_alpha"], "Extract all fields and values."),
        ("ocr_notice_02", assets["notice_beta"], "Return the checklist as numbered lines."),
        ("ocr_notice_03", assets["notice_gamma"], "Extract destination, carrier, box count, and weight."),
        ("ocr_notice_04", assets["notice_alpha"], "What is the deadline?"),
        ("ocr_invoice_04", assets["invoice_a"], "How many line items appear before TOTAL?"),
    ]
    return [
        {
            "id": pid,
            "modality": "ocr",
            "description": prompt,
            "prompt": prompt,
            "asset_path": str(path),
            "generate_audio": False,
        }
        for pid, path, prompt in items
    ]


def _audio_cases(assets: dict[str, Path]) -> list[dict]:
    path = assets.get("audio_probe")
    if not path:
        return []
    prompts = [
        ("audio_in_01", "Transcribe or summarize the audio in one short sentence.", False),
        ("audio_in_02", "What language or greeting do you hear? Answer briefly.", False),
        ("audio_in_03", "Respond to the audio with a short English acknowledgement.", True),
        ("audio_in_04", "Give a one sentence reply to what you heard.", True),
        ("audio_in_05", "Identify whether the clip contains speech or music. Answer briefly.", False),
        ("audio_in_06", "Repeat the main word you heard, if any.", False),
    ]
    return [
        {
            "id": pid,
            "modality": "audio_input" if not generate_audio else "audio_input_output",
            "description": prompt,
            "prompt": prompt,
            "asset_path": str(path),
            "generate_audio": generate_audio,
        }
        for pid, prompt, generate_audio in prompts
    ]


def _tts_cases() -> list[dict]:
    prompts = [
        ("tts_01", "Please respond in English. Say: Benchmark test one is complete.", True),
        ("tts_02", "Please respond in English. Give one fun fact about the moon.", True),
        ("tts_03", "Please respond in English. Explain in one sentence why fast inference matters.", True),
        ("tts_04", "Please respond in English. Say a short greeting to an OmniChat user.", True),
        ("tts_05", "Please respond in English. Count from one to five.", True),
        ("tts_06", "Please respond in English. Describe a calm sunset in one sentence.", True),
    ]
    return [
        {
            "id": pid,
            "modality": "audio_output",
            "description": prompt,
            "prompt": prompt,
            "generate_audio": generate_audio,
        }
        for pid, prompt, generate_audio in prompts
    ]


def build_cases(asset_dir: Path) -> list[dict]:
    assets = _ensure_assets(asset_dir)
    cases = _text_cases() + _image_cases(assets) + _ocr_cases(assets) + _audio_cases(assets) + _tts_cases()
    return cases[:50]


def _load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if np.asarray(audio).ndim > 1:
        audio = np.asarray(audio, dtype=np.float32).mean(axis=-1)
    if int(sr) != 16000:
        import librosa

        audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=int(sr), target_sr=16000)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def run_case(case: dict, output_dir: Path, *, temperature: float, max_new_tokens: int) -> dict:
    from tools.model.model_manager import chat, process_image
    from tools.vision.process_media import scan_document

    pid = case["id"]
    started = time.perf_counter()
    text = ""
    audio_generated = False
    audio_samples = 0
    audio_sample_rate = None
    error = None
    try:
        if case["modality"] == "text" or case["modality"] == "audio_output":
            result = chat(
                messages=[{"role": "user", "content": [case["prompt"]]}],
                generate_audio=bool(case.get("generate_audio")),
                output_audio_path=str(output_dir / f"{pid}.wav") if case.get("generate_audio") else None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=0.9,
                top_k=20,
                enable_thinking=False,
            )
        elif case["modality"] == "image":
            image = Image.open(case["asset_path"]).convert("RGB")
            result = process_image(
                image=image,
                prompt=case["prompt"],
                generate_audio=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=0.9,
                top_k=20,
                enable_thinking=False,
            )
        elif case["modality"] == "ocr":
            image = Image.open(case["asset_path"]).convert("RGB")
            result = scan_document(
                image,
                prompt=case["prompt"],
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=0.9,
                top_k=20,
                enable_thinking=False,
            )
        elif case["modality"] in {"audio_input", "audio_input_output"}:
            audio = _load_audio(case["asset_path"])
            result = chat(
                messages=[{"role": "user", "content": [audio, case["prompt"]]}],
                generate_audio=bool(case.get("generate_audio")),
                output_audio_path=str(output_dir / f"{pid}.wav") if case.get("generate_audio") else None,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=0.9,
                top_k=20,
                enable_thinking=False,
            )
        else:
            raise ValueError(f"Unknown modality: {case['modality']}")

        text = result.get("text") or ""
        audio = result.get("audio")
        if audio is not None:
            audio_generated = True
            audio_samples = int(len(audio))
            audio_sample_rate = int(result.get("sample_rate") or 24000)
    except Exception as exc:
        error = repr(exc)

    elapsed = time.perf_counter() - started
    (output_dir / f"{pid}.txt").write_text(text, encoding="utf-8")
    expected = case.get("expected")
    return {
        "id": pid,
        "modality": case["modality"],
        "description": case["description"],
        "expected": expected,
        "contains_expected": (str(expected).lower() in text.lower()) if expected else None,
        "generation_time_s": round(elapsed, 3),
        "text_length": len(text),
        "text_preview": text[:500],
        "audio_generated": audio_generated,
        "audio_samples": audio_samples,
        "audio_sample_rate": audio_sample_rate,
        "error": error,
    }


def worker_main(args: argparse.Namespace) -> int:
    from tools.model.model_manager import get_backend, get_model
    from tools.shared.session import configure_model_runtime, load_settings

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(Path(args.asset_dir))
    if args.limit:
        cases = cases[: args.limit]

    settings = load_settings(model_profile=args.worker_profile)
    settings = dict(settings)
    model_settings = dict(settings.get("model", {}))
    model_settings["mtp_enabled"] = args.worker_mtp == "on"
    settings["model"] = model_settings
    configure_model_runtime(settings)

    metadata = {
        "label": args.worker_label,
        "profile": args.worker_profile,
        "mtp": args.worker_mtp,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
    }

    load_started = time.perf_counter()
    get_model()
    metadata["load_time_s"] = round(time.perf_counter() - load_started, 3)
    backend = get_backend()
    if hasattr(backend, "get_runtime_status"):
        metadata["runtime_status_after_load"] = backend.get_runtime_status()

    results = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, 1):
            print(f"[{args.worker_label}] {index:02d}/{len(cases)} {case['id']} ({case['modality']})", flush=True)
            result = run_case(case, output_dir, temperature=args.temperature, max_new_tokens=args.max_new_tokens)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()

    if hasattr(backend, "get_runtime_status"):
        metadata["runtime_status_after_run"] = backend.get_runtime_status()
    metadata["results"] = results
    metadata["total_generation_time_s"] = round(sum(r["generation_time_s"] for r in results if not r.get("error")), 3)
    metadata["error_count"] = sum(1 for r in results if r.get("error"))
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return 1 if metadata["error_count"] else 0


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "total": round(sum(values), 3),
        "mean": round(mean(values), 3) if values else 0,
        "median": round(median(values), 3) if values else 0,
    }


def write_reports(bench_dir: Path) -> None:
    labels = [variant["label"] for variant in VARIANTS]
    results_by_label = {label: _read_jsonl(bench_dir / label / "results.jsonl") for label in labels}
    by_id = {
        label: {row["id"]: row for row in rows}
        for label, rows in results_by_label.items()
    }
    ids = [row["id"] for row in results_by_label[labels[0]]]

    comparison_rows = []
    for pid in ids:
        base = by_id[labels[0]][pid]
        mtp = by_id[labels[1]][pid]
        base_time = float(base["generation_time_s"])
        mtp_time = float(mtp["generation_time_s"])
        speedup = base_time / mtp_time if mtp_time else 0
        comparison_rows.append({
            "id": pid,
            "modality": base["modality"],
            "description": base["description"],
            "baseline_s": round(base_time, 3),
            "mtp_s": round(mtp_time, 3),
            "speedup": round(speedup, 3),
            "baseline_chars": base["text_length"],
            "mtp_chars": mtp["text_length"],
            "baseline_audio": base["audio_generated"],
            "mtp_audio": mtp["audio_generated"],
            "baseline_expected": base.get("contains_expected"),
            "mtp_expected": mtp.get("contains_expected"),
            "baseline_error": base.get("error"),
            "mtp_error": mtp.get("error"),
            "baseline_preview": base.get("text_preview", ""),
            "mtp_preview": mtp.get("text_preview", ""),
        })

    csv_path = bench_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    summary = {}
    for modality in sorted({row["modality"] for row in comparison_rows}):
        rows = [row for row in comparison_rows if row["modality"] == modality and not row["baseline_error"] and not row["mtp_error"]]
        summary[modality] = {
            "baseline": _summarize([row["baseline_s"] for row in rows]),
            "mtp": _summarize([row["mtp_s"] for row in rows]),
        }
        total_base = summary[modality]["baseline"]["total"]
        total_mtp = summary[modality]["mtp"]["total"]
        summary[modality]["speedup"] = round(total_base / total_mtp, 3) if total_mtp else 0

    all_base = [row["baseline_s"] for row in comparison_rows if not row["baseline_error"] and not row["mtp_error"]]
    all_mtp = [row["mtp_s"] for row in comparison_rows if not row["baseline_error"] and not row["mtp_error"]]
    overall = {
        "baseline": _summarize(all_base),
        "mtp": _summarize(all_mtp),
        "speedup": round(sum(all_base) / sum(all_mtp), 3) if sum(all_mtp) else 0,
        "case_count": len(comparison_rows),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    report_json = {
        "overall": overall,
        "by_modality": summary,
        "rows": comparison_rows,
    }
    (bench_dir / "comparison.json").write_text(json.dumps(report_json, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Gemma 4 Baseline vs Gemma 4 MTP Multimodal Benchmark",
        "",
        f"Generated: {overall['generated_at']}",
        "",
        "## Overall",
        "",
        "| Metric | Gemma 4 baseline | Gemma 4 + MTP | MTP speedup |",
        "|---|---:|---:|---:|",
        f"| Total generation time | {overall['baseline']['total']:.3f}s | {overall['mtp']['total']:.3f}s | **{overall['speedup']:.2f}x** |",
        f"| Mean per case | {overall['baseline']['mean']:.3f}s | {overall['mtp']['mean']:.3f}s |  |",
        f"| Median per case | {overall['baseline']['median']:.3f}s | {overall['mtp']['median']:.3f}s |  |",
        "",
        "## By Modality",
        "",
        "| Modality | Cases | Baseline total | MTP total | Speedup | Baseline mean | MTP mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for modality, data in summary.items():
        lines.append(
            f"| {modality} | {data['baseline']['count']} | {data['baseline']['total']:.3f}s | "
            f"{data['mtp']['total']:.3f}s | **{data['speedup']:.2f}x** | "
            f"{data['baseline']['mean']:.3f}s | {data['mtp']['mean']:.3f}s |"
        )

    lines.extend([
        "",
        "## Case-by-Case Timing",
        "",
        "| ID | Modality | Baseline | MTP | Speedup | Notes |",
        "|---|---|---:|---:|---:|---|",
    ])
    for row in comparison_rows:
        notes = []
        if row["baseline_error"]:
            notes.append(f"baseline error: {row['baseline_error']}")
        if row["mtp_error"]:
            notes.append(f"MTP error: {row['mtp_error']}")
        if row["baseline_expected"] is not None:
            notes.append(f"expected baseline/MTP: {row['baseline_expected']}/{row['mtp_expected']}")
        if row["baseline_audio"] or row["mtp_audio"]:
            notes.append(f"audio baseline/MTP: {row['baseline_audio']}/{row['mtp_audio']}")
        lines.append(
            f"| {row['id']} | {row['modality']} | {row['baseline_s']:.3f}s | "
            f"{row['mtp_s']:.3f}s | {row['speedup']:.2f}x | {'; '.join(notes)} |"
        )

    lines.extend([
        "",
        "## Response Samples",
        "",
    ])
    for row in comparison_rows:
        lines.append(f"### {row['id']} ({row['modality']})")
        lines.append("")
        lines.append(f"Prompt: {row['description']}")
        lines.append("")
        lines.append("| Variant | Preview |")
        lines.append("|---|---|")
        base_preview = row["baseline_preview"].replace("|", "\\|").replace("\n", " ")[:300]
        mtp_preview = row["mtp_preview"].replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(f"| Baseline | {base_preview} |")
        lines.append(f"| MTP | {mtp_preview} |")
        lines.append("")

    report_dir = bench_dir / "report"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def orchestrator_main(args: argparse.Namespace) -> int:
    bench_dir = Path(args.output_dir) if args.output_dir else ROOT / "benchmark_outputs" / f"gemma4_mtp_multimodal_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    bench_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = bench_dir / "assets"
    build_cases(asset_dir)

    print("=" * 72)
    print("  GEMMA 4 BASELINE VS MTP MULTIMODAL BENCHMARK")
    print("=" * 72)
    print(f"  Cases         : {args.limit or 50}")
    print(f"  Temperature   : {args.temperature}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"  Output        : {bench_dir}")
    print("=" * 72)

    return_codes = {}
    for variant in VARIANTS:
        out = bench_dir / variant["label"]
        cmd = [
            PYTHON,
            "-m",
            "benchmarks.gemma_mtp_multimodal",
            "--worker-profile",
            variant["profile"],
            "--worker-label",
            variant["label"],
            "--worker-mtp",
            variant["mtp"],
            "--asset-dir",
            str(asset_dir),
            "--output-dir",
            str(out),
            "--temperature",
            str(args.temperature),
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        print()
        print("=" * 72)
        print(f"  STARTING {variant['label']}")
        print("=" * 72)
        completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
        return_codes[variant["label"]] = completed.returncode

    if any(code != 0 for code in return_codes.values()):
        print(f"One or more variants failed: {return_codes}")
        return 1
    write_reports(bench_dir)
    print(f"\nBenchmark complete: {bench_dir}")
    print(f"Report: {bench_dir / 'report' / 'report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gemma 4 baseline vs MTP multimodal benchmark")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--worker-profile", default=None)
    parser.add_argument("--worker-label", default=None)
    parser.add_argument("--worker-mtp", choices=["on", "off"], default=None)
    parser.add_argument("--asset-dir", default=None)
    args = parser.parse_args()

    if args.worker_profile:
        return worker_main(args)
    return orchestrator_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
