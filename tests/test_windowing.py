import numpy as np
import pytest

from hybrid_windowing.dataset import Segment
from hybrid_windowing.windowing import (
    HYBRID_VARIANTS,
    adaptive_strides,
    backward_windows,
    build_hybrid,
    build_matched_random,
    build_static,
    forward_windows,
    overwindow,
    select_by_energy,
    static_windows,
    underwindow,
)


def ramp(length):
    return np.arange(length, dtype=np.float32)


def make_segments():
    generator = np.random.default_rng(0)
    segments = []
    for label, count in (("N", 300), ("V", 60), ("S", 30), ("F", 12)):
        for index in range(count):
            length = 300 if index == 0 else int(generator.integers(400, 800))
            signal = generator.normal(size=length).astype(np.float32)
            segments.append(Segment(signal, label, "100", 0, length // 2))
    return segments


def test_forward_windows_are_slices_of_the_signal():
    windows = forward_windows(ramp(10), 4, 3)
    assert [w[0] for w in windows] == [0, 3, 6]
    assert all(len(w) == 4 for w in windows)


def test_backward_windows_repeat_the_last_sample():
    windows = backward_windows(ramp(10), 4, 4, 2)
    assert windows[0].tolist() == [8, 9, 9, 9]
    assert windows[1].tolist() == [4, 5, 6, 7]


def test_short_segment_is_padded_or_dropped():
    assert backward_windows(ramp(3), 5, 2, 1)[0].tolist() == [0, 1, 2, 2, 2]
    assert backward_windows(ramp(3), 5, 2, 1, keep_padded=False) == []


def test_keep_padded_false_removes_edge_repeats():
    windows = backward_windows(ramp(10), 4, 2, 1, keep_padded=False)
    assert all(w.max() <= 9 and len(set(w.tolist())) == 4 for w in windows)


def test_static_window_uses_the_first_samples_only():
    assert static_windows(ramp(10), 4)[0].tolist() == [0, 1, 2, 3]
    assert static_windows(ramp(3), 4) == []


def test_adaptive_strides_scale_with_class_size():
    majority, minority, strides = adaptive_strides({"N": 1000, "V": 100, "S": 25, "F": 2}, 288)
    assert majority == "N"
    assert minority == ["V", "S", "F"]
    assert strides == {"V": 288, "S": 72, "F": 6}


def test_adaptive_stride_is_at_least_one():
    _, _, strides = adaptive_strides({"N": 100000, "V": 10000, "F": 1}, 288)
    assert strides["F"] == 1


def test_energy_selection_keeps_the_window_closest_to_the_group_mean():
    windows = [np.full(4, value, dtype=np.float32) for value in (1.0, 2.0, 9.0)]
    selected = select_by_energy(windows, 3)
    assert len(selected) == 1
    assert selected[0][0] == 2.0


def test_energy_selection_returns_one_window_per_group():
    windows = [np.full(4, float(i), dtype=np.float32) for i in range(10)]
    assert len(select_by_energy(windows, 3)) == 4


def test_underwindow_returns_windows_of_the_requested_length():
    selected = underwindow([ramp(500), ramp(100)], 360, "max", 1)
    assert len(selected) == 2
    assert all(len(window) == 360 for window in selected)


def test_overwindow_without_backward_pass_only_uses_forward_windows():
    assert len(overwindow([ramp(10)], 4, 3, backward=False)) == 3


def test_static_windows_keep_the_class_order():
    _, labels = build_static(make_segments())
    assert labels.tolist() == sorted(labels.tolist())


def test_hybrid_reduces_the_majority_and_expands_the_minorities():
    segments = make_segments()
    _, static_labels = build_static(segments)
    _, hybrid_labels = build_hybrid(segments)
    assert hybrid_labels.tolist().count(0) < static_labels.tolist().count(0)
    for index in (1, 2, 3):
        assert hybrid_labels.tolist().count(index) > static_labels.tolist().count(index)


@pytest.mark.parametrize("variant", HYBRID_VARIANTS)
def test_every_variant_produces_fixed_length_windows(variant):
    windows, labels = build_hybrid(make_segments(), variant)
    assert windows.shape[1:] == (360, 1)
    assert len(windows) == len(labels)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError):
        build_hybrid(make_segments(), "unknown")


def test_hybrid_windowing_is_deterministic():
    first, _ = build_hybrid(make_segments())
    second, _ = build_hybrid(make_segments())
    assert np.array_equal(first, second)


def test_matched_random_reproduces_the_hybrid_class_volumes():
    segments = make_segments()
    _, hybrid_labels = build_hybrid(segments)
    _, random_labels = build_matched_random(segments)
    assert np.bincount(random_labels).tolist() == np.bincount(hybrid_labels).tolist()
