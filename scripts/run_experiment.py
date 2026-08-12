"""Command-line entry point for predictive thesis benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment import run_predictive_benchmarks, run_repeated_predictive_benchmarks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ieee_cis.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=["mlp", "tabular_resnet", "tree"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--synthetic-fallback", action="store_true")
    parser.add_argument("--repeated", action="store_true", help="Run all configured seeds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = run_repeated_predictive_benchmarks if args.repeated else run_predictive_benchmarks
    result = runner(
        PROJECT_ROOT / args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        model_names=args.models,
        quick_run=args.quick,
        max_rows=args.max_rows,
        synthetic_fallback=args.synthetic_fallback,
    )
    table = result.get("summary", result["metrics"])
    print(table.to_string(index=False))
    print(f"Artifacts: {result['output_dir']}")


if __name__ == "__main__":
    main()
