"""Tests for tools.shared.chat_render."""

import base64

from tools.shared.chat_render import _math_expr_to_data_uri, render_chat_history_html


def test_render_chat_history_renders_assistant_markdown():
    history = [
        ("assistant", "## Title\n\n- one\n- two\n\n**bold** text"),
    ]

    html = render_chat_history_html(history)

    assert "<b>OmniChat:</b>" in html
    assert "Title" in html
    assert "<ul" in html
    assert "<li" in html
    assert "font-weight:700" in html or "<strong>bold</strong>" in html


def test_render_chat_history_escapes_user_and_system_text():
    history = [
        ("user", "Hello <b>there</b>\nnext line"),
        ("system", "Switched to <default> voice"),
    ]

    html = render_chat_history_html(history)

    assert "Hello &lt;b&gt;there&lt;/b&gt;<br>next line" in html
    assert "Switched to &lt;default&gt; voice" in html
    assert "<b>there</b>" not in html


def test_render_chat_history_renders_spoken_markers_and_partial():
    history = [
        ("user", "[voice input]"),
        ("_assistant_hidden", "secret"),
        ("_partial", "Working on **it**"),
    ]

    html = render_chat_history_html(history)

    assert "[Speech Input]" in html
    assert "OmniChat [Spoken Text]:" in html
    assert "secret" in html
    assert "font-style:italic" in html
    assert "OmniChat" in html
    assert "..." in html
    assert "it" in html
    assert "**it**" in html
    assert "<strong>it</strong>" not in html


def test_render_chat_history_uses_custom_assistant_label():
    history = [
        ("assistant", "hello"),
        ("_assistant_hidden", "spoken"),
        ("_partial", "working"),
    ]

    html = render_chat_history_html(history, assistant_label="OmniChat [Gemma 4 llama.cpp]")

    assert "OmniChat [Gemma 4 llama.cpp]:" in html
    assert "OmniChat [Gemma 4 llama.cpp] [Spoken Text]:" in html


def test_render_chat_history_keeps_typed_assistant_text_non_italic():
    history = [
        ("assistant", "**Normal** response"),
    ]

    html = render_chat_history_html(history)

    assert "[Spoken Text]" not in html
    assert "font-style:italic" not in html
    assert "font-weight:700" in html or "<strong>Normal</strong>" in html


def test_render_chat_history_renders_inline_and_block_math():
    history = [
        (
            "assistant",
            "The intensity \\( I \\) scales as \\[ I \\propto \\frac{1}{\\lambda^4} \\]",
        ),
    ]

    html = render_chat_history_html(history)

    assert "data:image/svg+xml;base64," in html
    assert "MATHPLACEHOLDER" not in html


def test_render_chat_history_resets_alignment_per_message_block():
    history = [
        ("assistant", "Equation:\n\n\\[ I \\propto \\frac{1}{\\lambda^4} \\]"),
        ("user", "next question"),
    ]

    html = render_chat_history_html(history)

    assert html.startswith("<div style='text-align:left;'>")
    assert "display:block; width:100%; margin-bottom:12px; text-align:left;" in html
    assert "<b>You:</b>" in html


def test_math_svg_matches_dark_theme_colors():
    uri = _math_expr_to_data_uri(r"I \propto \frac{1}{\lambda^4}")

    assert uri is not None
    svg = base64.b64decode(uri.split(",", 1)[1]).decode("utf-8", "ignore")

    assert "#cdd6f4" in svg
    assert "fill: #ffffff" not in svg
    assert 'fill="#ffffff"' not in svg