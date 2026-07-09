"""Command-line entry point: ``python -m stress {list,run}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import scenarios  # noqa: F401 - fills the registry
from . import report, runner


def _cmd_list() -> int:
    for item in runner.SCENARIOS.values():
        print(f"{item.name:36s} {item.doc}")
    return 0


def _cmd_run(
    patterns: List[str], quick: bool, json_path: Optional[str]
) -> int:
    selected = runner.select(patterns)
    if not selected:
        print(f"no scenarios match {patterns!r}", file=sys.stderr)
        return 2

    def on_start(item: runner.Scenario) -> None:
        print(f"... {item.name}", flush=True)

    results = runner.run(patterns, quick=quick, on_start=on_start)
    print()
    print(report.render(results, quick=quick))
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

    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list()
    return _cmd_run(args.patterns, args.quick, args.json)


if __name__ == "__main__":
    sys.exit(main())
