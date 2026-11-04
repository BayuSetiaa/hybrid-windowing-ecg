"""Audit window labels and edge padding for each configuration (Table 7)."""

import argparse
import collections
import math
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_windowing.config import (
    AAMI_MAP,
    BASE_STRIDE,
    CLASSES,
    GROUP_ORDER_SEED,
    GROUP_SIZE,
    TEST_RECORDS,
    TRAIN_RECORDS,
    WINDOW_SIZE,
)
from hybrid_windowing.dataset import load_dataset, read_annotations
from hybrid_windowing.windowing import adaptive_strides


def other_class_beats(dataset, mitdb_dir):
    """Return, per segment, the offsets of annotated beats belonging to another class."""
    annotations = {}
    for record, offset in dataset.offsets.items():
        samples, symbols = read_annotations(mitdb_dir, record)
        annotations[record] = (samples - offset, symbols)
    offsets = []
    for segment in dataset.segments:
        samples, symbols = annotations[segment.record]
        first = np.searchsorted(samples, segment.start)
        last = np.searchsorted(samples, segment.start + len(segment.signal), side="right")
        offsets.append(
            [
                int(sample) - segment.start
                for sample, symbol in zip(samples[first:last], symbols[first:last])
                if str(symbol) in AAMI_MAP and AAMI_MAP[str(symbol)] != segment.label
            ]
        )
    return offsets


def describe(segment, others, start, left_padding=0, right_padding=0):
    position = segment.beat - segment.start
    return {
        "has": start <= position < start + WINDOW_SIZE,
        "other": any(start <= other < start + WINDOW_SIZE for other in others),
        "pad": left_padding + right_padding,
    }


def audit_static(items):
    return [
        {**describe(segment, others, 0), "cls": segment.label}
        for segment, others in items
        if len(segment.signal) >= WINDOW_SIZE
    ]


def audit_majority(items, majority):
    representatives = []
    for segment, others in items:
        if segment.label != majority:
            continue
        signal, length = segment.signal, len(segment.signal)
        candidates = [(i, signal[i : i + WINDOW_SIZE], 0) for i in range(0, length - WINDOW_SIZE + 1, WINDOW_SIZE)]
        if length >= WINDOW_SIZE:
            candidates.append((length - WINDOW_SIZE, signal[-WINDOW_SIZE:], 0))
        else:
            padding = WINDOW_SIZE - length
            candidates.append((-padding, np.pad(signal, (padding, 0), mode="edge"), padding))
        stacked = np.stack([candidate[1] for candidate in candidates])
        chosen = int(np.argmin(((stacked - stacked.max(axis=0)) ** 2).mean(axis=1)))
        representatives.append((segment, others, *candidates[chosen]))

    order = np.random.default_rng(GROUP_ORDER_SEED).permutation(len(representatives))
    shuffled = [representatives[i] for i in order]
    rows = []
    for index in range(math.ceil(len(shuffled) / GROUP_SIZE)):
        group = shuffled[index * GROUP_SIZE : (index + 1) * GROUP_SIZE]
        energy = np.array([np.abs(item[3]).sum() for item in group])
        segment, others, start, _, padding = group[int(np.argmin(np.abs(energy - energy.mean())))]
        rows.append({**describe(segment, others, start, left_padding=padding), "cls": majority})
    return rows


def audit_minority(items, name, stride):
    rows, duplicates, total = [], 0, 0
    half = int(0.5 * stride)
    for segment, others in items:
        if segment.label != name:
            continue
        length = len(segment.signal)
        forward = list(range(0, length - WINDOW_SIZE + 1, stride))
        rows += [{**describe(segment, others, start), "cls": name} for start in forward]
        total += len(forward)
        if length < WINDOW_SIZE:
            rows.append({**describe(segment, others, 0, right_padding=WINDOW_SIZE - length), "cls": name})
            total += 1
            continue
        start = length + half - WINDOW_SIZE
        while start >= 0:
            padding = max(0, start + WINDOW_SIZE - length)
            rows.append({**describe(segment, others, start, right_padding=padding), "cls": name})
            total += 1
            if padding == 0 and start in forward:
                duplicates += 1
            start -= stride
    return rows, duplicates, total


def audit_hybrid(items):
    counts = collections.Counter(segment.label for segment, _ in items)
    majority, minority, strides = adaptive_strides(dict(counts), BASE_STRIDE)
    rows = audit_majority(items, majority)
    duplicates, totals = {}, {}
    for name in minority:
        windows, duplicates[name], totals[name] = audit_minority(items, name, strides[name])
        rows += windows
    return rows, strides, duplicates, totals


def tabulate(rows, configuration):
    table = pd.DataFrame(rows)
    output = []
    for name in CLASSES:
        subset = table[table.cls == name]
        padded = subset.pad > 0
        output.append(
            {
                "config": configuration,
                "cls": name,
                "windows": len(subset),
                "beat_inside_pct": round(100 * subset.has.mean(), 1),
                "other_class_beat_pct": round(100 * subset.other.mean(), 1),
                "padded_pct": round(100 * padded.mean(), 1),
                "mean_pad_samples_when_padded": round(subset.pad[padded].mean(), 1) if padded.any() else 0,
                "pad_ge_10pct_of_window_pct": round(100 * (subset.pad >= 36).mean(), 1),
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/mitbih_segments.npz"))
    parser.add_argument("--mitdb-dir", type=Path, default=Path("data/mitdb"))
    parser.add_argument("--output", type=Path, default=Path("outputs/label_audit.csv"))
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    items = list(zip(dataset.segments, other_class_beats(dataset, args.mitdb_dir)))
    train = [item for item in items if item[0].record in TRAIN_RECORDS]
    test = [item for item in items if item[0].record in TEST_RECORDS]

    rows = tabulate(audit_static(train), "DS1train_static")
    hybrid, strides, duplicates, totals = audit_hybrid(train)
    rows += tabulate(hybrid, "DS1train_hybrid")
    print(f"DS1 training strides {strides}; exact duplicate windows {duplicates} of {totals}")
    rows += tabulate(audit_static(test), "DS2test_static")
    rows += tabulate(audit_static(items), "Intra_static")
    hybrid, strides, duplicates, totals = audit_hybrid(items)
    rows += tabulate(hybrid, "Intra_hybrid")
    print(f"All-record strides {strides}; exact duplicate windows {duplicates} of {totals}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
