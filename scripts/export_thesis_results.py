"""Collect final metric CSV files into compact thesis-ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--output", default="results/tables/all_predictive_metrics.csv")
    args = parser.parse_args()

    run_root = Path(args.runs_root)
    files = sorted(run_root.glob("**/predictive_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No predictive_metrics.csv files found below {run_root}")
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame.insert(0, "run", str(path.parent.relative_to(run_root)))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"Wrote {len(combined)} rows to {output}")


if __name__ == "__main__":
    main()
