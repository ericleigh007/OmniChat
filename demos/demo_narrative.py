"""Terminal presentation helpers for the OmniChat demo.

All output is ASCII-safe to avoid Windows cp1252 encoding crashes.
"""

import sys
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ActResult:
    """Result of a single demo act."""
    act_num: int
    title: str
    passed: bool
    soft_fail: bool = False
    detail: str = ""


def _safe_print(text: str) -> None:
    """Print text safely on any terminal encoding."""
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(safe)


def banner(act_num: int, title: str) -> None:
    """Print a boxed act banner."""
    line = "=" * 52
    _safe_print("")
    _safe_print(line)
    _safe_print(f"  ACT {act_num}: {title.upper()}")
    _safe_print(line)
    _safe_print("")


def narrate(text: str) -> None:
    """Print indented narration."""
    _safe_print(f"  >> {text}")


def show_result(label: str, value: str) -> None:
    """Print a labeled result."""
    _safe_print(f"  [{label}] {value}")


def pass_fail(act_num: int, title: str, passed: bool, soft_fail: bool = False, detail: str = "") -> None:
    """Print a PASS/FAIL line for an act."""
    if passed:
        status = "PASS"
    elif soft_fail:
        status = "SOFT-FAIL"
    else:
        status = "FAIL"
    msg = f"  Act {act_num} ({title}): {status}"
    if detail:
        msg += f" -- {detail}"
    _safe_print(msg)


def summary_table(results: list[ActResult]) -> None:
    """Print final summary table of all act results."""
    _safe_print("")
    _safe_print("=" * 52)
    _safe_print("  DEMO SUMMARY")
    _safe_print("=" * 52)
    _safe_print("")
    _safe_print(f"  {'Act':<6} {'Title':<28} {'Result':<12}")
    _safe_print(f"  {'-'*6} {'-'*28} {'-'*12}")

    pass_count = 0
    fail_count = 0
    soft_count = 0

    for r in results:
        if r.passed:
            status = "PASS"
            pass_count += 1
        elif r.soft_fail:
            status = "SOFT-FAIL"
            soft_count += 1
        else:
            status = "FAIL"
            fail_count += 1
        _safe_print(f"  {r.act_num:<6} {r.title:<28} {status:<12}")

    _safe_print("")
    _safe_print(f"  Total: {pass_count} passed, {soft_count} soft-failed, {fail_count} failed")
    _safe_print(f"  Overall: {'PASS' if fail_count == 0 else 'FAIL'}")
    _safe_print("")
