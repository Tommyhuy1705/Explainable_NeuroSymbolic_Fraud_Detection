"""Command-line entry point for proposal-aligned thesis stress tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stress_testing.experiment import run_stress_test  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a leakage-safe TransXion v2 or AMLNet v1.0 stress test. "
            "Full mode requires the exact configured dataset checksum and exact model backends."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Stress-test YAML path")
    parser.add_argument(
        "--data-root",
        action="append",
        default=None,
        help="Dataset search root; repeat for multiple Kaggle/local roots",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Artifact directory override")
    parser.add_argument("--device", default=None, help="Torch device request, for example cuda or cpu")
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Run a reduced engineering check; artifacts are not thesis-claim eligible",
    )
    parser.add_argument(
        "--test-fixture",
        action="store_true",
        help="Generate an explicit schema fixture (requires --quick-run; never claim eligible)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = run_stress_test(
        arguments.config,
        data_roots=arguments.data_root,
        output_dir=arguments.output_dir,
        quick_run=arguments.quick_run,
        use_test_fixture=arguments.test_fixture,
        device=arguments.device,
    )
    print(
        json.dumps(
            {
                "output_dir": str(result["output_dir"]),
                "claim_eligible": bool(result["claim_eligible"]),
                "claim_blockers": result["claim_blockers"],
                "summary": result["summary"],
            },
            indent=2,
            ensure_ascii=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
