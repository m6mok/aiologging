"""
Throughput baselines: per-scenario metric floors for full runs.

``baselines.json`` maps scenario names to ``{metric: floor}``. Quick
runs are never compared (their workloads are scaled down). By default
a miss is advisory — reported after the run without affecting the
exit code; ``run --enforce-baselines`` turns misses into failing
checks instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .runner import Check, ScenarioResult

_BASELINE_PATH = Path(__file__).parent / "baselines.json"


def load() -> Dict[str, Dict[str, float]]:
    """The baseline table; empty when the file is absent."""
    if not _BASELINE_PATH.exists():
        return {}
    payload = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    return {
        name: floors
        for name, floors in payload.items()
        if not name.startswith("_")
    }


def compare(
    results: List[ScenarioResult], enforce: bool
) -> List[str]:
    """
    Compare full-run metrics against the baseline floors.

    With ``enforce`` misses are appended as failing checks; otherwise
    the returned warning lines are the only trace.
    """
    warnings: List[str] = []
    baselines = load()
    for result in results:
        floors = baselines.get(result.name)
        if not floors or result.status not in ("OK", "FAIL"):
            continue
        for metric, floor in floors.items():
            value = result.metrics.get(metric)
            if not isinstance(value, (int, float)):
                continue
            ok = value >= floor
            if enforce:
                result.checks.append(
                    Check(
                        name=f"baseline: {metric} >= {floor:_.0f}",
                        ok=ok,
                        detail=f"measured={value:_}",
                    )
                )
            elif not ok:
                warnings.append(
                    f"baseline miss: {result.name} {metric}="
                    f"{value:_} < floor {floor:_.0f}"
                )
    return warnings
