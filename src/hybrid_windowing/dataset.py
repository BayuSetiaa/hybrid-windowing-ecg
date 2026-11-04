"""Construction and loading of the MIT-BIH segment dataset."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AAMI_MAP, ALL_RECORDS, CLASS_INDEX, CLASSES


@dataclass(frozen=True, slots=True)
class Segment:
    """A beat segment; start and beat are positions in the record signal."""

    signal: np.ndarray
    label: str
    record: str
    start: int
    beat: int


@dataclass(frozen=True)
class SegmentDataset:
    signals: dict[str, np.ndarray]
    offsets: dict[str, int]
    segments: list[Segment]

    def subset(self, records) -> list[Segment]:
        keep = set(records)
        return [segment for segment in self.segments if segment.record in keep]


def download_mitdb(directory) -> None:
    import wfdb

    wfdb.dl_database("mitdb", str(directory), records=list(ALL_RECORDS), annotators=["atr"])


def read_annotations(mitdb_dir, record: str) -> tuple[np.ndarray, np.ndarray]:
    import wfdb

    annotation = wfdb.rdann(str(Path(mitdb_dir) / record), "atr")
    return np.asarray(annotation.sample), np.array(annotation.symbol)


def read_mlii(mitdb_dir, record: str) -> np.ndarray:
    """Return the unfiltered, unnormalized MLII channel in millivolts."""
    import wfdb

    data = wfdb.rdrecord(str(Path(mitdb_dir) / record))
    return data.p_signal[:, data.sig_name.index("MLII")]


def segment_rows(record: str, samples: np.ndarray, symbols: np.ndarray) -> list[tuple]:
    """Span each beat from the preceding to the following annotation of any type."""
    rows = []
    for index in range(1, len(samples) - 1):
        label = AAMI_MAP.get(str(symbols[index]))
        if label is not None:
            rows.append(
                (
                    int(record),
                    CLASS_INDEX[label],
                    int(samples[index - 1]),
                    int(samples[index + 1]),
                    int(samples[index]),
                )
            )
    return rows


def build_dataset(mitdb_dir, output_path) -> dict[str, int]:
    """Write the segment dataset and return the number of segments per class."""
    arrays, rows = {}, []
    for record in ALL_RECORDS:
        samples, symbols = read_annotations(mitdb_dir, record)
        signal = read_mlii(mitdb_dir, record)
        first, last = int(samples[0]), int(samples[-1])
        arrays[f"signal_{record}"] = signal[first:last].astype(np.float32)
        arrays[f"offset_{record}"] = np.int64(first)
        rows += segment_rows(record, samples, symbols)

    # This order determines the grouping permutation, so it must remain stable.
    rows.sort(key=lambda row: (row[1], f"{row[0]}_{row[2]}_{row[3]}"))
    table = np.array(rows, dtype=np.int64)
    arrays.update(
        segment_record=table[:, 0].astype(np.int16),
        segment_label=table[:, 1].astype(np.int8),
        segment_start=table[:, 2].astype(np.int32),
        segment_end=table[:, 3].astype(np.int32),
        segment_beat=table[:, 4].astype(np.int32),
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return {name: int((table[:, 1] == index).sum()) for index, name in enumerate(CLASSES)}


def load_dataset(path) -> SegmentDataset:
    data = np.load(path)
    signals = {key[7:]: data[key] for key in data.files if key.startswith("signal_")}
    offsets = {key[7:]: int(data[key]) for key in data.files if key.startswith("offset_")}
    segments = []
    columns = (
        data["segment_record"],
        data["segment_label"],
        data["segment_start"],
        data["segment_end"],
        data["segment_beat"],
    )
    for record, label, start, end, beat in zip(*columns):
        name = str(int(record))
        offset = offsets[name]
        first = int(start) - offset
        segments.append(
            Segment(
                signal=signals[name][first : int(end) - offset],
                label=CLASSES[int(label)],
                record=name,
                start=first,
                beat=int(beat) - offset,
            )
        )
    return SegmentDataset(signals, offsets, segments)
