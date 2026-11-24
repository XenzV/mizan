
from __future__ import annotations

import os
import sys

__all__ = ["gib", "tip", "badge", "table", "bar", "C", "heading"]

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


class C:
    RESET = "" if _NO_COLOR else "\033[0m"
    BOLD = "" if _NO_COLOR else "\033[1m"
    DIM = "" if _NO_COLOR else "\033[2m"
    RED = "" if _NO_COLOR else "\033[31m"
    GREEN = "" if _NO_COLOR else "\033[32m"
    YELLOW = "" if _NO_COLOR else "\033[33m"
    BLUE = "" if _NO_COLOR else "\033[34m"
    CYAN = "" if _NO_COLOR else "\033[36m"


def gib(mib: float) -> str:
    return f"{mib / 1024:6.2f} GiB"


def heading(text: str) -> str:
    return f"{C.BOLD}{C.CYAN}{text}{C.RESET}"


def tip(text: str) -> str:
    return f"{C.DIM}{text}{C.RESET}"


def badge(status: str) -> str:
    colors = {"OK": C.GREEN, "TIGHT": C.YELLOW, "OVER": C.RED, "UNKNOWN": C.DIM}
    return f"{colors.get(status, '')}{C.BOLD}{status}{C.RESET}"


def bar(fraction: float, width: int = 24) -> str:
    filled = max(0, min(width, int(round(fraction * width))))
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    full, empty = ("█", "░") if "utf" in encoding else ("#", ".")
    return full * filled + empty * (width - filled)


def table(rows: list[list[str]], headers: list[str] | None = None) -> str:
    all_rows = ([headers] if headers else []) + rows
    if not all_rows:
        return ""
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(all_rows[0]))]
    lines: list[str] = []
    for idx, row in enumerate(all_rows):
        lines.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if headers and idx == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)
