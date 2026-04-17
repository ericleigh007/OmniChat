"""Helpers for rendering chat history in desktop rich-text views."""

import base64
from functools import lru_cache
from html import escape
from io import BytesIO
import re

from PySide6.QtGui import QTextDocumentFragment

try:
    from matplotlib.mathtext import math_to_image
except ImportError:  # pragma: no cover - dependency is declared for the app, but keep a safe fallback.
    math_to_image = None


_FRAGMENT_RE = re.compile(r"<!--StartFragment-->(.*)<!--EndFragment-->", re.DOTALL)
_BODY_RE = re.compile(r"<body[^>]*>(.*)</body>", re.DOTALL)
_BLOCK_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_SVG_LIGHT_BG_RE = re.compile(r"fill:\s*#ffffff")
_SVG_LIGHT_BG_ATTR_RE = re.compile(r'fill="#ffffff"')
_MATH_FOREGROUND = "#cdd6f4"


def _message_block(inner_html: str) -> str:
    return (
        "<div style='display:block; width:100%; margin-bottom:12px; text-align:left;'>"
        f"{inner_html}"
        "</div>"
    )


def _plain_text_to_html(text: str) -> str:
    return escape(text).replace("\n", "<br>")


@lru_cache(maxsize=256)
def _math_expr_to_data_uri(expr: str) -> str | None:
    if math_to_image is None:
        return None

    try:
        buffer = BytesIO()
        math_to_image(f"${expr}$", buffer, format="svg", color=_MATH_FOREGROUND)
    except Exception:
        return None

    svg = buffer.getvalue().decode("utf-8", "ignore")
    svg = _SVG_LIGHT_BG_RE.sub("fill: none", svg)
    svg = _SVG_LIGHT_BG_ATTR_RE.sub('fill="none"', svg)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _render_math_html(expr: str, display: bool) -> str:
    normalized = expr.strip()
    data_uri = _math_expr_to_data_uri(normalized)
    fallback = f"<code>{escape(normalized)}</code>"
    if not data_uri:
        return fallback if not display else f"<div style='margin:8px 0;'>{fallback}</div>"

    if display:
        return (
            "<div style='margin:8px 0; text-align:center;'>"
            f"<img src=\"{data_uri}\" style='vertical-align:middle;' />"
            "</div>"
        )
    return f"<img src=\"{data_uri}\" style='vertical-align:middle;' />"


def _extract_math_placeholders(text: str) -> tuple[str, dict[str, str]]:
    replacements: dict[str, str] = {}
    counter = 0

    def _make_replacer(display: bool):
        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            token = f"MATHPLACEHOLDER{counter}TOKEN"
            counter += 1
            replacements[token] = _render_math_html(match.group(1), display=display)
            if display:
                return f"\n\n{token}\n\n"
            return token

        return _replace

    text = _BLOCK_MATH_RE.sub(_make_replacer(True), text)
    text = _INLINE_MATH_RE.sub(_make_replacer(False), text)
    return text, replacements


def _markdown_to_html_fragment(text: str) -> str:
    text, placeholders = _extract_math_placeholders(text)
    fragment = QTextDocumentFragment.fromMarkdown(text or "")
    html = fragment.toHtml()
    match = _FRAGMENT_RE.search(html)
    if match:
        html = match.group(1)
    else:
        match = _BODY_RE.search(html)
        if match:
            html = match.group(1)
        else:
            html = _plain_text_to_html(text)

    for token, replacement in placeholders.items():
        html = html.replace(token, replacement)
    return html


def render_chat_history_html(
    history: list[tuple[str, str]],
    limit: int = 20,
    assistant_label: str = "OmniChat",
) -> str:
    """Render chat history to HTML with assistant markdown support."""
    blocks = []

    for role, text in history[-limit:]:
        if role == "user":
            if text == "[voice input]":
                blocks.append(_message_block(
                    "<p style='margin:0;'><b>[Speech Input]</b></p>"
                ))
            else:
                blocks.append(_message_block(
                    "<p style='margin:0 0 4px 0;'><b>You:</b></p>"
                    f"<p style='margin:0;'>{_plain_text_to_html(text)}</p>"
                ))
        elif role == "assistant":
            blocks.append(_message_block(
                f"<p style='margin:0 0 4px 0;'><b>{escape(assistant_label)}:</b></p>"
                f"{_markdown_to_html_fragment(text)}"
            ))
        elif role == "_assistant_hidden":
            blocks.append(_message_block(
                f"<p style='margin:0 0 4px 0;'><b>{escape(assistant_label)} [Spoken Text]:</b></p>"
                f"<div style='font-style:italic;'>{_markdown_to_html_fragment(text)}</div>"
            ))
        elif role == "_partial":
            blocks.append(_message_block(
                f"<p style='margin:0 0 4px 0;'><b>{escape(assistant_label)}:</b></p>"
                f"<p style='margin:0;'>{_plain_text_to_html(text + '...')}</p>"
            ))
        elif role == "system":
            blocks.append(_message_block(
                f"<p style='margin:0;'><i>{_plain_text_to_html(text)}</i></p>"
            ))

    return "<div style='text-align:left;'>" + "".join(blocks) + "</div>"