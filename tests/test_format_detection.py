"""Tests for format detection and table parsing from process_media.py."""

import pytest

from tools.vision.process_media import _detect_format, _parse_table


class TestDetectFormat:
    """Test auto-detection of output format from text content."""

    def test_pipe_table_detected_as_excel(self):
        text = "| Name | Age | City |\n|---|---|---|\n| Alice | 30 | NY |\n| Bob | 25 | LA |"
        assert _detect_format(text) == "excel"

    def test_tab_table_detected_as_excel(self):
        text = "Name\tAge\tCity\nAlice\t30\tNY\nBob\t25\tLA\nCharlie\t35\tTokyo"
        assert _detect_format(text) == "excel"

    def test_markdown_with_headings_and_lists(self):
        text = "# Title\n\n## Section\n\n- Item one\n- Item two\n\nSome text."
        assert _detect_format(text) == "markdown"

    def test_markdown_with_code_block(self):
        text = "# Code Example\n\n```python\nprint('hello')\n```\n\nMore text."
        assert _detect_format(text) == "markdown"

    def test_markdown_with_blockquotes(self):
        text = "# Quote\n\n> This is a quote\n> Continued"
        assert _detect_format(text) == "markdown"

    def test_plain_text(self):
        text = "This is just a paragraph of plain text.\nWith a second line."
        assert _detect_format(text) == "text"

    def test_single_pipe_not_table(self):
        text = "A | B\nSome text here\nAnother line"
        assert _detect_format(text) != "excel"

    def test_two_pipe_rows_not_enough(self):
        text = "| A | B |\n| C | D |"
        assert _detect_format(text) != "excel"

    def test_three_pipe_rows_enough(self):
        text = "| A | B |\n| C | D |\n| E | F |"
        assert _detect_format(text) == "excel"

    def test_empty_string(self):
        assert _detect_format("") == "text"

    def test_single_heading_not_markdown(self):
        text = "# Just a heading"
        assert _detect_format(text) == "text"

    def test_numbered_list_counts_as_markdown(self):
        text = "# Instructions\n\n1. First step\n2. Second step"
        assert _detect_format(text) == "markdown"


class TestParseTable:
    """Test table parsing into rows and cells."""

    def test_pipe_delimited_basic(self):
        text = "| Name | Age |\n|---|---|\n| Alice | 30 |\n| Bob | 25 |"
        rows = _parse_table(text)
        assert len(rows) == 3  # header + 2 data rows (separator stripped)
        assert rows[0] == ["Name", "Age"]
        assert rows[1] == ["Alice", "30"]

    def test_pipe_delimited_no_outer_pipes(self):
        text = "Name | Age\n---|---\nAlice | 30"
        rows = _parse_table(text)
        assert len(rows) >= 2
        assert "Name" in rows[0][0]

    def test_separator_line_stripped(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        rows = _parse_table(text)
        # Separator line should NOT appear in rows
        for row in rows:
            assert not all(c in "-|: " for cell in row for c in cell)

    def test_tab_delimited(self):
        text = "Name\tAge\tCity\nAlice\t30\tNY\nBob\t25\tLA"
        rows = _parse_table(text)
        assert len(rows) == 3
        assert rows[0] == ["Name", "Age", "City"]
        assert rows[1] == ["Alice", "30", "NY"]

    def test_empty_input(self):
        assert _parse_table("") == []

    def test_no_delimiters(self):
        text = "Just plain text\nWith no tables"
        assert _parse_table(text) == []

    def test_whitespace_stripped(self):
        text = "|  Name  |  Age  |\n| Alice  | 30    |"
        rows = _parse_table(text)
        assert rows[0] == ["Name", "Age"]
        assert rows[1] == ["Alice", "30"]

    def test_complex_separator(self):
        text = "| A | B |\n|:---:|---:|\n| 1 | 2 |"
        rows = _parse_table(text)
        assert len(rows) == 2  # header + 1 row, separator stripped
        assert rows[0] == ["A", "B"]

    def test_mixed_content_pipe_wins(self):
        text = "| A\tB | C |\n| D\tE | F |\n| G\tH | I |"
        rows = _parse_table(text)
        # Pipe-delimited takes precedence when both | and \t exist
        assert len(rows) == 3
