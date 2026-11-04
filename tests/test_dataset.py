from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hybrid_windowing.config import (
    ALL_RECORDS,
    CLASSES,
    EXPECTED_WINDOW_COUNTS,
    GRID_GROUPS,
    GRID_POOLS,
    TEST_RECORDS,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from hybrid_windowing.dataset import load_dataset
from hybrid_windowing.windowing import build_hybrid, build_static

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "mitbih_segments.npz"
COUNTS = ROOT / "results" / "first_environment" / "window_counts.csv"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run scripts/prepare_data.py first")


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA)


def counts(labels):
    return tuple(np.bincount(labels, minlength=len(CLASSES)).tolist())


def test_dataset_size(dataset):
    assert len(dataset.signals) == 44
    assert len(dataset.segments) == 100674


@pytest.mark.parametrize(
    "name, records, builder",
    [
        ("train_static", TRAIN_RECORDS, build_static),
        ("train_hybrid", TRAIN_RECORDS, build_hybrid),
        ("validation_static", VALIDATION_RECORDS, build_static),
        ("test_static", TEST_RECORDS, build_static),
        ("intra_static", ALL_RECORDS, build_static),
        ("intra_hybrid", ALL_RECORDS, build_hybrid),
    ],
)
def test_window_counts_match_the_article(dataset, name, records, builder):
    _, labels = builder(dataset.subset(records))
    assert counts(labels) == EXPECTED_WINDOW_COUNTS[name]


@pytest.mark.parametrize("pool_method", GRID_POOLS)
@pytest.mark.parametrize("group_size", GRID_GROUPS)
def test_grid_counts_match_the_recorded_values(dataset, pool_method, group_size):
    table = pd.read_csv(COUNTS)
    row = table[(table.protocol == "intra") & (table.set == f"all_hybrid_{pool_method}_g{group_size}")].iloc[0]
    _, labels = build_hybrid(dataset.subset(ALL_RECORDS), pool_method=pool_method, group_size=group_size)
    assert counts(labels) == tuple(int(row[name]) for name in CLASSES)
