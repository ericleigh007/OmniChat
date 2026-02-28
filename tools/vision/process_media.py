"""
process_media.py — Image and video processing via MiniCPM-o 4.5.

Provides high-level functions for:
- Image analysis (general description, OCR, table extraction)
- Document scanning (multi-slice OCR for dense text/PDFs)
- Video analysis (frame + audio interleaved understanding)

All functions delegate to model_manager for inference and return
structured results with auto-detected output format hints.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.model.model_manager import process_image, process_video, chat


# ── Format detection ─────────────────────────────────────────────────────────

def _detect_format(text: str) -> str:
    """
    Auto-detect the best output format based on content structure.

    Returns:
        'excel' — if the text looks like a table (rows of pipe-delimited or tab-delimited data)
        'markdown' — if the text has markdown headings, lists, or code blocks
        'text' — plain text fallback
    """
    lines = text.strip().splitlines()

    # Check for table patterns (pipe-delimited markdown tables)
    pipe_lines = sum(1 for line in lines if line.count("|") >= 2)
    if pipe_lines >= 3:
        return "excel"

    # Check for tab-delimited tables
    tab_lines = sum(1 for line in lines if line.count("\t") >= 2)
    if tab_lines >= 3:
        return "excel"

    # Check for markdown features
    md_indicators = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            md_indicators += 1
        elif stripped.startswith(("- ", "* ", "1. ", "```")):
            md_indicators += 1
        elif stripped.startswith("> "):
            md_indicators += 1

    if md_indicators >= 2:
        return "markdown"

    return "text"


def _parse_table(text: str) -> list[list[str]]:
    """
    Parse a markdown or tab-delimited table into rows of cells.

    Returns:
        List of rows, where each row is a list of cell strings.
    """
    lines = text.strip().splitlines()
    rows = []

    for line in lines:
        # Skip separator lines (e.g., |---|---|)
        stripped = line.strip()
        if stripped and all(c in "-| \t:" for c in stripped):
            continue

        if "|" in line:
            # Pipe-delimited
            cells = [c.strip() for c in line.split("|")]
            # Remove empty leading/trailing cells from outer pipes
            if cells and not cells[0]:
                cells = cells[1:]
            if cells and not cells[-1]:
                cells = cells[:-1]
            if cells:
                rows.append(cells)
        elif "\t" in line:
            # Tab-delimited
            cells = [c.strip() for c in line.split("\t")]
            if cells:
                rows.append(cells)

    return rows


# ── Image processing ─────────────────────────────────────────────────────────

def scan_image(
    image,
    prompt: str = "Describe this image in detail.",
    voice_ref: Optional[np.ndarray] = None,
    generate_audio: bool = False,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
) -> dict:
    """
    Analyze an image with a text prompt.

    Args:
        image: PIL.Image.Image object.
        prompt: What to do with the image (describe, OCR, extract table, etc.).
        voice_ref: Optional voice reference for audio response.
        generate_audio: Whether to generate spoken response.
        output_audio_path: Path to save audio response.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens to generate.

    Returns:
        dict with keys:
            text: str — the model's text response
            format: str — auto-detected format ('text', 'markdown', 'excel')
            table: list[list[str]] | None — parsed table if format is 'excel'
            audio: np.ndarray | None — audio if generated
            audio_path: str | None — path to saved audio
    """
    result = process_image(
        image=image,
        prompt=prompt,
        generate_audio=generate_audio,
        voice_ref=voice_ref,
        output_audio_path=output_audio_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_slice_nums=1,
    )

    fmt = _detect_format(result["text"])
    table = _parse_table(result["text"]) if fmt == "excel" else None

    return {
        "text": result["text"],
        "format": fmt,
        "table": table,
        "audio": result.get("audio"),
        "audio_path": result.get("audio_path"),
        "sample_rate": result.get("sample_rate"),
    }


def scan_document(
    image,
    prompt: str = "Extract all text from this document. Preserve formatting, tables, and structure.",
    voice_ref: Optional[np.ndarray] = None,
    generate_audio: bool = False,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
) -> dict:
    """
    OCR-focused document scanning with high slice count for dense text.

    Uses max_slice_nums=25 for better text extraction from documents and PDFs.
    Lower temperature (0.3) for more faithful extraction.

    Args:
        image: PIL.Image.Image of the document page.
        prompt: Extraction instructions.

    Returns:
        Same structure as scan_image().
    """
    result = process_image(
        image=image,
        prompt=prompt,
        generate_audio=generate_audio,
        voice_ref=voice_ref,
        output_audio_path=output_audio_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_slice_nums=25,
    )

    fmt = _detect_format(result["text"])
    table = _parse_table(result["text"]) if fmt == "excel" else None

    return {
        "text": result["text"],
        "format": fmt,
        "table": table,
        "audio": result.get("audio"),
        "audio_path": result.get("audio_path"),
        "sample_rate": result.get("sample_rate"),
    }


def analyze_video(
    video_path: str,
    prompt: str = "Describe what's happening in this video.",
    voice_ref: Optional[np.ndarray] = None,
    generate_audio: bool = False,
    output_audio_path: Optional[str] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
) -> dict:
    """
    Analyze a video file (frames + audio track).

    Args:
        video_path: Path to video file (.mp4, .avi, .mov, etc.).
        prompt: What to analyze in the video.

    Returns:
        Same structure as scan_image() (format is usually 'text' or 'markdown').
    """
    result = process_video(
        video_path=video_path,
        prompt=prompt,
        generate_audio=generate_audio,
        voice_ref=voice_ref,
        output_audio_path=output_audio_path,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    fmt = _detect_format(result["text"])
    table = _parse_table(result["text"]) if fmt == "excel" else None

    return {
        "text": result["text"],
        "format": fmt,
        "table": table,
        "audio": result.get("audio"),
        "audio_path": result.get("audio_path"),
        "sample_rate": result.get("sample_rate"),
    }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test media processing")
    parser.add_argument("--image", default=None, help="Path to image file")
    parser.add_argument("--video", default=None, help="Path to video file")
    parser.add_argument("--document", action="store_true", help="Use document/OCR mode for image")
    parser.add_argument("--prompt", default=None, help="Custom prompt")
    args = parser.parse_args()

    if args.image:
        from PIL import Image

        img = Image.open(args.image).convert("RGB")
        prompt = args.prompt or ("Extract all text from this document." if args.document else "Describe this image in detail.")

        if args.document:
            result = scan_document(img, prompt=prompt)
        else:
            result = scan_image(img, prompt=prompt)

        print(f"Format: {result['format']}")
        print(f"Text:\n{result['text']}")
        if result["table"]:
            print(f"\nParsed table ({len(result['table'])} rows):")
            for row in result["table"][:5]:
                print(f"  {row}")

    elif args.video:
        prompt = args.prompt or "Describe what's happening in this video."
        result = analyze_video(args.video, prompt=prompt)
        print(f"Format: {result['format']}")
        print(f"Text:\n{result['text']}")

    else:
        print("Provide --image or --video")
