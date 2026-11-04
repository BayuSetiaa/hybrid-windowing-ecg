"""Download MIT-BIH if needed and build the segment dataset used by the experiments."""

import argparse
from pathlib import Path

from hybrid_windowing.config import ALL_RECORDS
from hybrid_windowing.dataset import build_dataset, download_mitdb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mitdb-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--output", type=Path, default=Path("data/mitbih_segments.npz"))
    args = parser.parse_args()

    missing = [r for r in ALL_RECORDS if not (args.mitdb_dir / f"{r}.dat").exists()]
    if missing:
        print(f"Downloading {len(missing)} records from PhysioNet into {args.mitdb_dir}")
        args.mitdb_dir.mkdir(parents=True, exist_ok=True)
        download_mitdb(args.mitdb_dir)

    counts = build_dataset(args.mitdb_dir, args.output)
    print(f"Records: {len(ALL_RECORDS)}")
    print(f"Segments: {sum(counts.values())} {counts}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
