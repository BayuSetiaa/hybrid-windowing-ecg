"""Run the experiment groups reported in the article."""

import argparse
from pathlib import Path

from hybrid_windowing.dataset import load_dataset
from hybrid_windowing.experiments import GROUPS, ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mitbih_segments.npz"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--smoke", action="store_true", help="short check; never use for reported results")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset.signals)} records and {len(dataset.segments)} segments")
    runner = ExperimentRunner(dataset, args.output, smoke=args.smoke)
    runner.write_environment()
    for group in args.groups:
        getattr(runner, group)()
    completed = sum(1 for path in runner.runs_dir.glob("*/metrics.json"))
    print(f"Completed runs: {completed}")


if __name__ == "__main__":
    main()
