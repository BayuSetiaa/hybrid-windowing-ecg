"""R-peak-centred windows used to test sensitivity to the labelling strategy."""

import numpy as np

from .config import (
    CLASS_INDEX,
    CLASSES,
    GROUP_ORDER_SEED,
    GROUP_SIZE,
    POOL_METHOD,
    RPEAK_BASE_STRIDE,
    RPEAK_SHIFT,
    WINDOW_SIZE,
)
from .windowing import adaptive_strides, reference_signal, select_by_energy, to_arrays


def is_valid(segment, signals):
    half = WINDOW_SIZE // 2 + RPEAK_SHIFT
    return segment.beat - half >= 0 and segment.beat + half <= len(signals[segment.record])


def beat_window(segment, signals, shift=0):
    start = segment.beat - WINDOW_SIZE // 2 + shift
    return signals[segment.record][start : start + WINDOW_SIZE]


def beats_by_class(segments, signals):
    grouped = {
        name: [s for s in segments if s.label == name and is_valid(s, signals)] for name in CLASSES
    }
    return {name: beats for name, beats in grouped.items() if beats}


def build_rpeak_static(segments, signals):
    grouped = beats_by_class(segments, signals)
    return to_arrays(
        [(name, [beat_window(s, signals) for s in beats]) for name, beats in grouped.items()]
    )


def static_membership_mask(segments, signals):
    """Mark the beats whose segment also yields a segment-based static window."""
    grouped = beats_by_class(segments, signals)
    return np.array([len(s.signal) >= WINDOW_SIZE for beats in grouped.values() for s in beats])


def shifts_for_stride(stride):
    forward = list(range(-RPEAK_SHIFT, RPEAK_SHIFT + 1, stride))
    half = stride // 2
    backward = [shift + half for shift in forward if shift + half <= RPEAK_SHIFT and half > 0]
    return sorted(set(forward) | set(backward))


def build_rpeak_hybrid(segments, signals, targets):
    """Apply the hybrid procedure with the labelled beat inside every window."""
    grouped = beats_by_class(segments, signals)
    counts = {name: len(beats) for name, beats in grouped.items()}
    majority, minority, strides = adaptive_strides(counts, RPEAK_BASE_STRIDE)

    representatives = []
    for segment in grouped[majority]:
        candidates = [beat_window(segment, signals, k) for k in (-RPEAK_SHIFT, 0, RPEAK_SHIFT)]
        reference = reference_signal(candidates, POOL_METHOD)
        distances = [np.mean((candidate - reference) ** 2) for candidate in candidates]
        representatives.append(candidates[int(np.argmin(distances))])

    class_windows = [(majority, select_by_energy(representatives, GROUP_SIZE))]
    generator = np.random.default_rng(GROUP_ORDER_SEED)
    for name in minority:
        shifts = shifts_for_stride(strides[name])
        windows = [beat_window(s, signals, k) for s in grouped[name] for k in shifts]
        target = int(targets[CLASS_INDEX[name]])
        if target < len(windows):
            keep = np.sort(generator.choice(len(windows), target, replace=False))
            windows = [windows[i] for i in keep]
        class_windows.append((name, windows))
    return to_arrays(class_windows)
