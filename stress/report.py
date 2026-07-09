"""
Rendering of stress-run results: a per-scenario console report plus
an optional machine-readable JSON file.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from .runner import MetricValue, ScenarioResult

_STATUS_MARK = {
    "OK": "✓",
    "FAIL": "✗",
    "ERROR": "!",
    "SKIP": "-",
}


def _format_value(value: MetricValue) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:_d}"
    if isinstance(value, float):
        return f"{value:_.3f}".rstrip("0").rstrip(".")
    return str(value)


def render(results: List[ScenarioResult], quick: bool) -> str:
    """Human-readable report for the whole run."""
    lines: List[str] = []
    mode = "quick" if quick else "full"
    lines.append(
        f"aiologging stress run ({mode}, "
        f"python {platform.python_version()})"
    )
    lines.append("")

    for result in results:
        mark = _STATUS_MARK[result.status]
        lines.append(
            f"{mark} {result.name} — {result.status} "
            f"({result.duration_s:.2f}s)"
        )
        if result.skipped is not None:
            lines.append(f"    skipped: {result.skipped}")
        for key, value in result.metrics.items():
            lines.append(f"    {key} = {_format_value(value)}")
        for check in result.checks:
            check_mark = "✓" if check.ok else "✗"
            detail = f"  [{check.detail}]" if check.detail else ""
            lines.append(f"    {check_mark} {check.name}{detail}")
        if result.error is not None:
            for error_line in result.error.rstrip().splitlines():
                lines.append(f"    ! {error_line}")
        lines.append("")

    counts = {"OK": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0}
    for result in results:
        counts[result.status] += 1
    total_s = sum(result.duration_s for result in results)
    lines.append(
        f"{len(results)} scenarios in {total_s:.1f}s: "
        f"{counts['OK']} ok, {counts['FAIL']} failed, "
        f"{counts['ERROR']} errored, {counts['SKIP']} skipped"
    )
    return "\n".join(lines)


def write_json(
    results: List[ScenarioResult], path: Path, quick: bool
) -> None:
    """Dump the run to a JSON file for tracking across revisions."""
    payload = {
        "mode": "quick" if quick else "full",
        "python": platform.python_version(),
        "platform": sys.platform,
        "results": [
            {**asdict(result), "status": result.status}
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
