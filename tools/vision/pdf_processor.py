"""PDF document scanning via per-page OCR.

Renders each PDF page to a PIL Image using PyMuPDF, sends each through
scan_document() for OCR, then aggregates results.  Designed for bank
statements and tabular financial documents.
"""

from pathlib import Path
from typing import Callable, Optional

from PIL import Image

# ── Default prompts ──────────────────────────────────────────────────────

BANK_STATEMENT_PROMPT = (
    "Extract all transactions from this bank statement page as a markdown table. "
    "Columns should be: Date | Description | Amount | Balance. "
    "Include the header row. If there is no transaction table on this page, "
    "extract whatever text is present and note that no transactions were found."
)

DEFAULT_DOCUMENT_PROMPT = (
    "Extract all text from this document page. "
    "Preserve formatting, tables, and structure. "
    "Output any tables as pipe-delimited markdown tables."
)


# ── PDF rendering ────────────────────────────────────────────────────────

def pdf_to_images(
    pdf_path: str,
    dpi: int = 300,
    first_page: int = 0,
    last_page: Optional[int] = None,
) -> list[Image.Image]:
    """Render PDF pages to PIL Images using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.
        dpi: Resolution for rendering (300 is standard for OCR).
        first_page: 0-indexed first page to render.
        last_page: 0-indexed last page (inclusive).  None = all pages.

    Returns:
        List of PIL Images, one per page.
    """
    import fitz

    doc = fitz.open(pdf_path)
    images = []

    end = last_page if last_page is not None else len(doc) - 1
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(first_page, min(end + 1, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images


def get_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    import fitz

    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ── Per-page OCR + aggregation ───────────────────────────────────────────

def scan_pdf(
    pdf_path: str,
    prompt: Optional[str] = None,
    dpi: int = 300,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    on_page_start: Optional[Callable[[int, int], None]] = None,
    on_page_done: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """Scan an entire PDF: render pages, OCR each, aggregate results.

    Args:
        pdf_path: Path to the PDF file.
        prompt: OCR prompt for each page.  Defaults to DEFAULT_DOCUMENT_PROMPT.
        dpi: Rendering resolution (300 recommended).
        temperature: Model temperature (lower = more faithful extraction).
        max_new_tokens: Max tokens per page.
        on_page_start: Callback(page_num, total_pages) before each page.
        on_page_done: Callback(page_num, total_pages, page_result) after each page.

    Returns:
        dict with keys:
            pages       -- list of per-page result dicts (from scan_document)
            combined_text   -- all page text joined with page markers
            combined_table  -- merged table rows (header from first page) or None
            page_count      -- number of pages scanned
    """
    from tools.vision.process_media import scan_document

    prompt = prompt or DEFAULT_DOCUMENT_PROMPT
    images = pdf_to_images(pdf_path, dpi=dpi)
    total = len(images)

    pages = []
    combined_table: list[list[str]] | None = None
    all_text_parts = []

    for i, img in enumerate(images):
        if on_page_start:
            on_page_start(i, total)

        result = scan_document(
            image=img,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        pages.append(result)

        all_text_parts.append(f"--- Page {i + 1} of {total} ---\n{result['text']}")
        combined_table = merge_tables(combined_table, result.get("table"))

        if on_page_done:
            on_page_done(i, total, result)

    combined_text = "\n\n".join(all_text_parts)

    return {
        "pages": pages,
        "combined_text": combined_text,
        "combined_table": combined_table,
        "page_count": total,
    }


def merge_tables(
    existing: list[list[str]] | None,
    new_rows: list[list[str]] | None,
) -> list[list[str]] | None:
    """Merge two tables, keeping one header row.

    If existing is None, returns new_rows as-is.
    If new_rows is None, returns existing as-is.
    Otherwise appends new_rows (skipping its header) to existing.
    """
    if not new_rows:
        return existing
    if not existing:
        return list(new_rows)
    # Skip header row of new_rows (row 0)
    return existing + new_rows[1:]


def scan_multiple_pdfs(
    pdf_paths: list[str],
    prompt: Optional[str] = None,
    bank_mode: bool = False,
    dpi: int = 300,
    temperature: float = 0.3,
    max_new_tokens: int = 4096,
    on_file_start: Optional[Callable[[int, int, str], None]] = None,
    on_page_start: Optional[Callable[[int, int], None]] = None,
    on_page_done: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """Scan multiple PDFs and merge all results into one combined table.

    Args:
        pdf_paths: List of paths to PDF files.
        prompt: OCR prompt (ignored in bank_mode).
        bank_mode: If True, use BANK_STATEMENT_PROMPT with temperature 0.2.
        dpi: Rendering resolution.
        temperature: Model temperature (ignored in bank_mode).
        max_new_tokens: Max tokens per page.
        on_file_start: Callback(file_idx, total_files, filename) before each file.
        on_page_start: Callback(page_num, total_pages) before each page.
        on_page_done: Callback(page_num, total_pages, result) after each page.

    Returns:
        dict with keys:
            combined_text   -- all text from all files with file/page markers
            combined_table  -- merged table rows across all files, or None
            total_pages     -- total pages across all files
            file_count      -- number of files processed
    """
    all_text_parts = []
    combined_table: list[list[str]] | None = None
    total_pages = 0

    for file_idx, pdf_path in enumerate(pdf_paths):
        filename = Path(pdf_path).name
        if on_file_start:
            on_file_start(file_idx, len(pdf_paths), filename)

        if bank_mode:
            result = scan_bank_statement(
                pdf_path, dpi=dpi,
                on_page_start=on_page_start, on_page_done=on_page_done,
            )
        else:
            result = scan_pdf(
                pdf_path, prompt=prompt, dpi=dpi,
                temperature=temperature, max_new_tokens=max_new_tokens,
                on_page_start=on_page_start, on_page_done=on_page_done,
            )

        all_text_parts.append(
            f"=== File: {filename} ({result['page_count']} pages) ===\n"
            f"{result['combined_text']}"
        )
        combined_table = merge_tables(combined_table, result["combined_table"])
        total_pages += result["page_count"]

    return {
        "combined_text": "\n\n".join(all_text_parts),
        "combined_table": combined_table,
        "total_pages": total_pages,
        "file_count": len(pdf_paths),
    }


def scan_bank_statement(
    pdf_path: str,
    dpi: int = 300,
    on_page_start: Optional[Callable[[int, int], None]] = None,
    on_page_done: Optional[Callable[[int, int, dict], None]] = None,
) -> dict:
    """Convenience wrapper for bank statement PDFs with optimized prompt.

    Returns same structure as scan_pdf().
    """
    return scan_pdf(
        pdf_path=pdf_path,
        prompt=BANK_STATEMENT_PROMPT,
        dpi=dpi,
        temperature=0.2,
        max_new_tokens=4096,
        on_page_start=on_page_start,
        on_page_done=on_page_done,
    )


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scan PDF documents")
    parser.add_argument("pdf", nargs="+", help="Path(s) to PDF file(s)")
    parser.add_argument("--prompt", default=None, help="Custom OCR prompt")
    parser.add_argument("--bank", action="store_true", help="Use bank statement mode")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI")
    parser.add_argument("--csv", default=None, help="Save combined table as CSV to this path")
    args = parser.parse_args()

    def _on_file(idx, total, name):
        print(f"\n[{idx + 1}/{total}] {name}")

    def _on_page(page, total):
        print(f"  Scanning page {page + 1}/{total}...")

    if len(args.pdf) == 1:
        if args.bank:
            res = scan_bank_statement(args.pdf[0], dpi=args.dpi, on_page_start=_on_page)
        else:
            res = scan_pdf(args.pdf[0], prompt=args.prompt, dpi=args.dpi, on_page_start=_on_page)
        print(f"\nPages scanned: {res['page_count']}")
        table = res["combined_table"]
    else:
        res = scan_multiple_pdfs(
            args.pdf, prompt=args.prompt, bank_mode=args.bank, dpi=args.dpi,
            on_file_start=_on_file, on_page_start=_on_page,
        )
        print(f"\nFiles: {res['file_count']}, Total pages: {res['total_pages']}")
        table = res["combined_table"]

    print(f"\n{res['combined_text'][:2000]}")

    if table:
        print(f"\nCombined table: {len(table)} rows")
        for row in table[:5]:
            print(f"  {row}")

        if args.csv:
            import csv
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(table)
            print(f"\nSaved CSV: {args.csv}")
