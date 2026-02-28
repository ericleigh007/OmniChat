"""Tests for save_output.py — file saving utilities."""

import re
from pathlib import Path

import pytest

import tools.output.save_output as so


@pytest.fixture(autouse=True)
def patch_outputs_dir(tmp_outputs_dir, monkeypatch):
    """Redirect OUTPUTS_DIR to temp directory for all tests."""
    monkeypatch.setattr(so, "OUTPUTS_DIR", tmp_outputs_dir)


class TestTimestampStem:
    """Test timestamp-based filename generation."""

    def test_default_prefix(self):
        stem = so._timestamp_stem()
        assert stem.startswith("output_")
        # Pattern: output_YYYY-MM-DD_HHMMSS
        assert re.match(r"output_\d{4}-\d{2}-\d{2}_\d{6}", stem)

    def test_custom_prefix(self):
        stem = so._timestamp_stem("scan")
        assert stem.startswith("scan_")


class TestUniquePath:
    """Test unique path generation with collision avoidance."""

    def test_no_collision(self, tmp_outputs_dir):
        path = so._unique_path(tmp_outputs_dir, "test", ".txt")
        assert path == tmp_outputs_dir / "test.txt"

    def test_collision_increments(self, tmp_outputs_dir):
        # Create the first file
        (tmp_outputs_dir / "test.txt").write_text("first")
        path = so._unique_path(tmp_outputs_dir, "test", ".txt")
        assert path == tmp_outputs_dir / "test_2.txt"

    def test_multiple_collisions(self, tmp_outputs_dir):
        (tmp_outputs_dir / "test.txt").write_text("first")
        (tmp_outputs_dir / "test_2.txt").write_text("second")
        path = so._unique_path(tmp_outputs_dir, "test", ".txt")
        assert path == tmp_outputs_dir / "test_3.txt"


class TestSaveAsMarkdown:
    """Test markdown file saving."""

    def test_creates_md_file(self, tmp_outputs_dir):
        path = so.save_as_markdown("# Hello World", filename="test_doc")
        assert Path(path).exists()
        assert path.endswith(".md")
        assert Path(path).read_text(encoding="utf-8") == "# Hello World"

    def test_auto_filename(self, tmp_outputs_dir):
        path = so.save_as_markdown("content")
        assert Path(path).exists()
        assert "scan_" in Path(path).stem


class TestSaveAsText:
    """Test plain text file saving."""

    def test_creates_txt_file(self, tmp_outputs_dir):
        path = so.save_as_text("Hello World", filename="test_text")
        assert Path(path).exists()
        assert path.endswith(".txt")
        assert Path(path).read_text(encoding="utf-8") == "Hello World"


class TestSaveAsExcel:
    """Test Excel file saving."""

    def test_creates_xlsx_file(self, tmp_outputs_dir):
        data = [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]]
        path = so.save_as_excel(data, filename="test_table")
        assert Path(path).exists()
        assert path.endswith(".xlsx")

    def test_header_row_bold(self, tmp_outputs_dir):
        from openpyxl import load_workbook

        data = [["Name", "Age"], ["Alice", "30"]]
        path = so.save_as_excel(data, filename="bold_test")
        wb = load_workbook(path)
        ws = wb.active
        assert ws.cell(row=1, column=1).font.bold is True
        assert ws.cell(row=2, column=1).font.bold is not True

    def test_correct_cell_values(self, tmp_outputs_dir):
        from openpyxl import load_workbook

        data = [["A", "B"], ["1", "2"]]
        path = so.save_as_excel(data, filename="values_test")
        wb = load_workbook(path)
        ws = wb.active
        assert ws.cell(row=1, column=1).value == "A"
        assert ws.cell(row=2, column=2).value == "2"


class TestSaveAuto:
    """Test auto-format selection and saving."""

    def test_auto_detects_markdown(self, tmp_outputs_dir):
        content = "# Title\n\n## Section\n\n- Item one\n- Item two"
        path = so.save_auto(content, fmt="auto")
        assert path.endswith(".md")

    def test_auto_detects_text(self, tmp_outputs_dir):
        content = "Just a plain sentence."
        path = so.save_auto(content, fmt="auto")
        assert path.endswith(".txt")

    def test_explicit_markdown(self, tmp_outputs_dir):
        path = so.save_auto("plain content", fmt="markdown")
        assert path.endswith(".md")

    def test_explicit_text(self, tmp_outputs_dir):
        path = so.save_auto("# looks like md", fmt="text")
        assert path.endswith(".txt")

    def test_explicit_excel_with_table(self, tmp_outputs_dir):
        table = [["A", "B"], ["1", "2"]]
        path = so.save_auto("| A | B |", fmt="excel", table=table)
        assert path.endswith(".xlsx")
