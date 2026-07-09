"""Command-line entry point: ``python -m stress {list,run}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import scenarios  # noqa: F401 - fills the registry
from . import baseline, report, runner


def _cmd_list() -> int:
    for item in runner.SCENARIOS.values():
        print(f"{item.name:36s} {item.doc}")
    return 0


def _cmd_run(
    patterns: List[str],
    quick: bool,
    json_path: Optional[str],
    repeat: int = 1,
    enforce_baselines: bool = False,
) -> int:
    selected = runner.select(patterns)
    if not selected:
        print(f"no scenarios match {patterns!r}", file=sys.stderr)
        return 2

    def on_start(item: runner.Scenario) -> None:
        print(f"... {item.name}", flush=True)

    results: List[runner.ScenarioResult] = []
    for iteration in range(repeat):
        if repeat > 1:
            print(f"--- pass {iteration + 1}/{repeat} ---", flush=True)
        results += runner.run(patterns, quick=quick, on_start=on_start)

    # Quick workloads are scaled down; baselines only make sense for
    # full runs
    warnings: List[str] = []
    if not quick:
        warnings = baseline.compare(results, enforce=enforce_baselines)

    print()
    print(report.render(results, quick=quick))
    for line in warnings:
        print(f"⚠ {line}")
    if json_path is not None:
        report.write_json(results, Path(json_path), quick=quick)
        print(f"\nJSON report written to {json_path}")

    failed = any(
        result.status in ("FAIL", "ERROR") for result in results
    )
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stress",
        description="aiologging stress-test harness",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="list available scenarios")

    run_parser = subparsers.add_parser("run", help="run scenarios")
    run_parser.add_argument(
        "patterns",
        nargs="*",
        help="substring filters, e.g. 'overload' or 'chaos.loop'",
    )
    run_parser.add_argument(
        "--quick",
        action="store_true",
        help="scaled-down smoke run (seconds instead of minutes)",
    )
    run_parser.add_argument(
        "--json",
        metavar="PATH",
        help="also write a machine-readable JSON report",
    )
    run_parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="run the selected scenarios N times (flake hunting)",
    )
    run_parser.add_argument(
        "--enforce-baselines",
        action="store_true",
        help="turn baseline misses into failing checks (full runs)",
    )

    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    if args.repeat < 1:
        print("--repeat must be >= 1", file=sys.stderr)
        return 2
    return _cmd_run(
        args.patterns,
        args.quick,
        args.json,
        repeat=args.repeat,
        enforce_baselines=args.enforce_baselines,
    )


if __name__ == "__main__":
    sys.exit(main())
