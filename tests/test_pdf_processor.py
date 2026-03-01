"""Tests for pdf_processor.py -- PDF rendering and aggregation logic.

PDF rendering tests use real tiny PDFs created with PyMuPDF.
OCR aggregation tests mock scan_document to avoid GPU dependency.

Run with: python -m pytest tests/test_pdf_processor.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from PIL import Image


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tiny_pdf(tmp_path):
    """Create a minimal 2-page PDF using PyMuPDF."""
    import fitz

    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=612, height=792)  # US Letter
        page.insert_text((72, 72), f"Page {i + 1} content", fontsize=12)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def single_page_pdf(tmp_path):
    """Create a 1-page PDF."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "Single page", fontsize=12)
    pdf_path = tmp_path / "single.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def _make_scan_result(text, table=None):
    """Helper to build a scan_document-shaped result dict."""
    return {
        "text": text,
        "format": "excel" if table else "text",
        "table": table,
        "audio": None,
        "audio_path": None,
        "sample_rate": None,
    }


# ── PDF Rendering ────────────────────────────────────────────────────────

class TestPdfToImages:
    """Test PDF-to-image rendering (no model needed)."""

    def test_renders_all_pages(self, tiny_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        images = pdf_to_images(tiny_pdf, dpi=72)
        assert len(images) == 2

    def test_images_are_pil(self, tiny_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        images = pdf_to_images(tiny_pdf, dpi=72)
        assert all(isinstance(img, Image.Image) for img in images)

    def test_images_are_rgb(self, tiny_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        images = pdf_to_images(tiny_pdf, dpi=72)
        assert all(img.mode == "RGB" for img in images)

    def test_dpi_affects_resolution(self, tiny_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        low = pdf_to_images(tiny_pdf, dpi=72)[0]
        high = pdf_to_images(tiny_pdf, dpi=150)[0]
        assert high.width > low.width
        assert high.height > low.height

    def test_page_range_first_only(self, tiny_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        images = pdf_to_images(tiny_pdf, dpi=72, first_page=0, last_page=0)
        assert len(images) == 1

    def test_page_range_second_only(self, tiny_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        images = pdf_to_images(tiny_pdf, dpi=72, first_page=1, last_page=1)
        assert len(images) == 1

    def test_single_page_pdf(self, single_page_pdf):
        from tools.vision.pdf_processor import pdf_to_images

        images = pdf_to_images(single_page_pdf, dpi=72)
        assert len(images) == 1


class TestGetPageCount:
    """Test page counting."""

    def test_two_page_pdf(self, tiny_pdf):
        from tools.vision.pdf_processor import get_page_count

        assert get_page_count(tiny_pdf) == 2

    def test_single_page_pdf(self, single_page_pdf):
        from tools.vision.pdf_processor import get_page_count

        assert get_page_count(single_page_pdf) == 1


# ── OCR Aggregation (mocked model) ──────────────────────────────────────

class TestScanPdfAggregation:
    """Test table merging and text aggregation with mocked model."""

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_tables_merged_across_pages(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))] * 2
        mock_doc.side_effect = [
            _make_scan_result(
                "| Date | Desc | Amount |\n|---|---|---|\n| 01/01 | Deposit | 100 |",
                table=[["Date", "Desc", "Amount"], ["01/01", "Deposit", "100"]],
            ),
            _make_scan_result(
                "| Date | Desc | Amount |\n|---|---|---|\n| 01/02 | Withdrawal | -50 |",
                table=[["Date", "Desc", "Amount"], ["01/02", "Withdrawal", "-50"]],
            ),
        ]

        from tools.vision.pdf_processor import scan_pdf

        result = scan_pdf("fake.pdf")
        table = result["combined_table"]

        assert table is not None
        assert len(table) == 3  # header + 2 data rows
        assert table[0] == ["Date", "Desc", "Amount"]
        assert table[1][0] == "01/01"
        assert table[2][0] == "01/02"

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_combined_text_has_page_markers(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))] * 2
        mock_doc.side_effect = [
            _make_scan_result("Page one text"),
            _make_scan_result("Page two text"),
        ]

        from tools.vision.pdf_processor import scan_pdf

        result = scan_pdf("fake.pdf")

        assert "--- Page 1 of 2 ---" in result["combined_text"]
        assert "--- Page 2 of 2 ---" in result["combined_text"]
        assert "Page one text" in result["combined_text"]
        assert "Page two text" in result["combined_text"]

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_no_table_pages_yield_none(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))]
        mock_doc.return_value = _make_scan_result("Summary page with no tables")

        from tools.vision.pdf_processor import scan_pdf

        result = scan_pdf("fake.pdf")
        assert result["combined_table"] is None

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_mixed_table_and_no_table_pages(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))] * 3
        mock_doc.side_effect = [
            _make_scan_result("Account summary page"),
            _make_scan_result(
                "| Date | Amount |\n| 01/01 | 100 |",
                table=[["Date", "Amount"], ["01/01", "100"]],
            ),
            _make_scan_result(
                "| Date | Amount |\n| 01/02 | 200 |",
                table=[["Date", "Amount"], ["01/02", "200"]],
            ),
        ]

        from tools.vision.pdf_processor import scan_pdf

        result = scan_pdf("fake.pdf")
        table = result["combined_table"]

        assert table is not None
        assert len(table) == 3  # header + 2 data rows
        assert result["page_count"] == 3

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_page_count(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))] * 5
        mock_doc.return_value = _make_scan_result("text")

        from tools.vision.pdf_processor import scan_pdf

        result = scan_pdf("fake.pdf")
        assert result["page_count"] == 5
        assert len(result["pages"]) == 5

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_callbacks_invoked(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))] * 2
        mock_doc.return_value = _make_scan_result("text")

        starts = []
        dones = []

        from tools.vision.pdf_processor import scan_pdf

        scan_pdf(
            "fake.pdf",
            on_page_start=lambda p, t: starts.append((p, t)),
            on_page_done=lambda p, t, r: dones.append((p, t)),
        )

        assert starts == [(0, 2), (1, 2)]
        assert dones == [(0, 2), (1, 2)]

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_custom_prompt_passed_through(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))]
        mock_doc.return_value = _make_scan_result("text")

        from tools.vision.pdf_processor import scan_pdf

        scan_pdf("fake.pdf", prompt="Extract all numbers")

        call_kwargs = mock_doc.call_args[1]
        assert call_kwargs["prompt"] == "Extract all numbers"

    @patch("tools.vision.pdf_processor.pdf_to_images")
    @patch("tools.vision.process_media.scan_document")
    def test_default_prompt_used(self, mock_doc, mock_render):
        mock_render.return_value = [Image.new("RGB", (100, 100))]
        mock_doc.return_value = _make_scan_result("text")

        from tools.vision.pdf_processor import scan_pdf, DEFAULT_DOCUMENT_PROMPT

        scan_pdf("fake.pdf")

        call_kwargs = mock_doc.call_args[1]
        assert call_kwargs["prompt"] == DEFAULT_DOCUMENT_PROMPT


class TestScanBankStatement:
    """Verify bank statement mode uses correct prompt and temperature."""

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_uses_bank_prompt(self, mock_scan):
        mock_scan.return_value = {
            "pages": [], "combined_text": "", "combined_table": None, "page_count": 0,
        }

        from tools.vision.pdf_processor import scan_bank_statement, BANK_STATEMENT_PROMPT

        scan_bank_statement("fake.pdf")

        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["prompt"] == BANK_STATEMENT_PROMPT

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_uses_low_temperature(self, mock_scan):
        mock_scan.return_value = {
            "pages": [], "combined_text": "", "combined_table": None, "page_count": 0,
        }

        from tools.vision.pdf_processor import scan_bank_statement

        scan_bank_statement("fake.pdf")

        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["temperature"] == 0.2

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_passes_callbacks(self, mock_scan):
        mock_scan.return_value = {
            "pages": [], "combined_text": "", "combined_table": None, "page_count": 0,
        }

        from tools.vision.pdf_processor import scan_bank_statement

        cb_start = lambda p, t: None
        cb_done = lambda p, t, r: None
        scan_bank_statement("fake.pdf", on_page_start=cb_start, on_page_done=cb_done)

        call_kwargs = mock_scan.call_args[1]
        assert call_kwargs["on_page_start"] is cb_start
        assert call_kwargs["on_page_done"] is cb_done


# ── merge_tables unit tests ──────────────────────────────────────────────

class TestMergeTables:
    """Test the merge_tables helper function directly."""

    def test_both_none(self):
        from tools.vision.pdf_processor import merge_tables

        assert merge_tables(None, None) is None

    def test_existing_none_returns_new(self):
        from tools.vision.pdf_processor import merge_tables

        new = [["A", "B"], ["1", "2"]]
        result = merge_tables(None, new)
        assert result == [["A", "B"], ["1", "2"]]

    def test_new_none_returns_existing(self):
        from tools.vision.pdf_processor import merge_tables

        existing = [["A", "B"], ["1", "2"]]
        result = merge_tables(existing, None)
        assert result is existing

    def test_merge_skips_header_of_new(self):
        from tools.vision.pdf_processor import merge_tables

        existing = [["Date", "Amount"], ["01/01", "100"]]
        new = [["Date", "Amount"], ["01/02", "200"]]
        result = merge_tables(existing, new)
        assert len(result) == 3
        assert result[0] == ["Date", "Amount"]
        assert result[1] == ["01/01", "100"]
        assert result[2] == ["01/02", "200"]

    def test_merge_three_tables_sequentially(self):
        from tools.vision.pdf_processor import merge_tables

        t1 = [["H"], ["a"]]
        t2 = [["H"], ["b"]]
        t3 = [["H"], ["c"]]
        result = merge_tables(merge_tables(t1, t2), t3)
        assert result == [["H"], ["a"], ["b"], ["c"]]

    def test_empty_list_treated_as_falsy(self):
        from tools.vision.pdf_processor import merge_tables

        existing = [["A"], ["1"]]
        result = merge_tables(existing, [])
        assert result is existing

    def test_returns_copy_not_reference(self):
        from tools.vision.pdf_processor import merge_tables

        new = [["A"], ["1"]]
        result = merge_tables(None, new)
        assert result == new
        assert result is not new  # should be a copy

    def test_single_row_new_table(self):
        """A new table with only a header and no data rows."""
        from tools.vision.pdf_processor import merge_tables

        existing = [["H"], ["data"]]
        new = [["H"]]  # header only, no data
        result = merge_tables(existing, new)
        # Should just keep existing since new only has a header to skip
        assert result == [["H"], ["data"]]


# ── scan_multiple_pdfs tests ─────────────────────────────────────────────

class TestScanMultiplePdfs:
    """Test multi-file scanning with mocked single-file scan functions."""

    def _fake_scan_result(self, page_count, text, table=None):
        return {
            "pages": [{}] * page_count,
            "combined_text": text,
            "combined_table": table,
            "page_count": page_count,
        }

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_combines_text_from_multiple_files(self, mock_scan):
        mock_scan.side_effect = [
            self._fake_scan_result(2, "File 1 text"),
            self._fake_scan_result(3, "File 2 text"),
        ]

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/a.pdf", "/tmp/b.pdf"])

        assert "=== File: a.pdf (2 pages) ===" in result["combined_text"]
        assert "=== File: b.pdf (3 pages) ===" in result["combined_text"]
        assert "File 1 text" in result["combined_text"]
        assert "File 2 text" in result["combined_text"]

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_total_pages_summed(self, mock_scan):
        mock_scan.side_effect = [
            self._fake_scan_result(2, "text"),
            self._fake_scan_result(5, "text"),
        ]

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/a.pdf", "/tmp/b.pdf"])
        assert result["total_pages"] == 7
        assert result["file_count"] == 2

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_tables_merged_across_files(self, mock_scan):
        mock_scan.side_effect = [
            self._fake_scan_result(
                1, "text",
                table=[["Date", "Amount"], ["01/01", "100"]],
            ),
            self._fake_scan_result(
                1, "text",
                table=[["Date", "Amount"], ["02/01", "200"]],
            ),
        ]

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/a.pdf", "/tmp/b.pdf"])
        table = result["combined_table"]
        assert table is not None
        assert len(table) == 3  # header + 2 data rows
        assert table[0] == ["Date", "Amount"]
        assert table[1] == ["01/01", "100"]
        assert table[2] == ["02/01", "200"]

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_no_tables_yields_none(self, mock_scan):
        mock_scan.side_effect = [
            self._fake_scan_result(1, "text"),
            self._fake_scan_result(1, "text"),
        ]

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/a.pdf", "/tmp/b.pdf"])
        assert result["combined_table"] is None

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_mixed_table_and_no_table_files(self, mock_scan):
        mock_scan.side_effect = [
            self._fake_scan_result(1, "text"),  # no table
            self._fake_scan_result(
                1, "text",
                table=[["Col"], ["val"]],
            ),
        ]

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/a.pdf", "/tmp/b.pdf"])
        assert result["combined_table"] == [["Col"], ["val"]]

    @patch("tools.vision.pdf_processor.scan_bank_statement")
    def test_bank_mode_dispatches_correctly(self, mock_bank):
        mock_bank.return_value = self._fake_scan_result(1, "bank text")

        from tools.vision.pdf_processor import scan_multiple_pdfs

        scan_multiple_pdfs(["/tmp/stmt.pdf"], bank_mode=True)

        mock_bank.assert_called_once()
        assert mock_bank.call_args[0][0] == "/tmp/stmt.pdf"

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_non_bank_mode_dispatches_correctly(self, mock_scan):
        mock_scan.return_value = self._fake_scan_result(1, "text")

        from tools.vision.pdf_processor import scan_multiple_pdfs

        scan_multiple_pdfs(["/tmp/doc.pdf"], bank_mode=False, prompt="Custom")

        mock_scan.assert_called_once()
        assert mock_scan.call_args[1]["prompt"] == "Custom"

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_file_start_callback(self, mock_scan):
        mock_scan.return_value = self._fake_scan_result(1, "text")

        from tools.vision.pdf_processor import scan_multiple_pdfs

        calls = []
        scan_multiple_pdfs(
            ["/tmp/a.pdf", "/tmp/b.pdf"],
            on_file_start=lambda idx, total, name: calls.append((idx, total, name)),
        )

        assert calls == [(0, 2, "a.pdf"), (1, 2, "b.pdf")]

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_single_file(self, mock_scan):
        mock_scan.return_value = self._fake_scan_result(
            3, "content",
            table=[["H"], ["r1"], ["r2"]],
        )

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/only.pdf"])
        assert result["file_count"] == 1
        assert result["total_pages"] == 3
        assert len(result["combined_table"]) == 3

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_empty_file_list(self, mock_scan):
        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs([])
        assert result["file_count"] == 0
        assert result["total_pages"] == 0
        assert result["combined_table"] is None
        assert result["combined_text"] == ""
        mock_scan.assert_not_called()

    @patch("tools.vision.pdf_processor.scan_pdf")
    def test_three_files_tables_accumulated(self, mock_scan):
        """Simulate scanning 3 monthly statements."""
        mock_scan.side_effect = [
            self._fake_scan_result(
                2, "Jan", table=[["Date", "Amt"], ["01/01", "10"], ["01/15", "20"]],
            ),
            self._fake_scan_result(
                2, "Feb", table=[["Date", "Amt"], ["02/01", "30"]],
            ),
            self._fake_scan_result(
                1, "Mar", table=[["Date", "Amt"], ["03/01", "40"], ["03/20", "50"]],
            ),
        ]

        from tools.vision.pdf_processor import scan_multiple_pdfs

        result = scan_multiple_pdfs(["/tmp/jan.pdf", "/tmp/feb.pdf", "/tmp/mar.pdf"])
        table = result["combined_table"]
        assert len(table) == 6  # header + 5 data rows
        assert table[0] == ["Date", "Amt"]
        assert result["total_pages"] == 5
        assert result["file_count"] == 3
