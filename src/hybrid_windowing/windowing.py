"""Static and hybrid window extraction."""

import math

import numpy as np

from .config import (
    BASE_STRIDE,
    CLASS_INDEX,
    CLASSES,
    GROUP_ORDER_SEED,
    GROUP_SIZE,
    POOL_METHOD,
    WINDOW_SIZE,
)

HYBRID_VARIANTS = (
    "full",
    "no_adaptive",
    "no_phase",
    "no_padding",
    "no_energy",
    "uw_only",
    "ow_only",
)


def forward_windows(signal, window, stride):
    return [signal[i : i + window] for i in range(0, len(signal) - window + 1, stride)]


def backward_windows(signal, window, stride, pad_right, keep_padded=True):
    """Slide from the right edge after repeating the last sample pad_right times."""
    if len(signal) < window:
        return [np.pad(signal, (0, window - len(signal)), mode="edge")] if keep_padded else []
    padded = np.pad(signal, (0, pad_right), mode="edge")
    windows = []
    start = len(padded) - window
    while start >= 0:
        if keep_padded or start + window <= len(signal):
            windows.append(padded[start : start + window])
        start -= stride
    return windows


def reference_signal(signals, method):
    stacked = np.stack(signals)
    if method == "mean":
        return np.mean(stacked, axis=0)
    if method == "max":
        return np.max(stacked, axis=0)
    if method == "min":
        return np.min(stacked, axis=0)
    return np.median(stacked, axis=0)


def static_windows(signal, window):
    return [signal[:window]] if len(signal) >= window else []


def select_by_energy(candidates, group_size, selector="energy"):
    """Keep one window per group, closest to the mean group energy or chosen at random."""
    generator = np.random.default_rng(GROUP_ORDER_SEED + 1)
    order = np.random.default_rng(GROUP_ORDER_SEED).permutation(len(candidates))
    shuffled = [candidates[i] for i in order]
    selected = []
    for index in range(math.ceil(len(shuffled) / group_size)):
        group = shuffled[index * group_size : (index + 1) * group_size]
        if selector == "energy":
            energy = np.array([np.sum(np.abs(window)) for window in group])
            selected.append(group[int(np.argmin(np.abs(energy - energy.mean())))])
        else:
            selected.append(group[int(generator.integers(len(group)))])
    return selected


def underwindow(signals, window, pool_method, group_size, selector="energy", keep_short=True):
    """Reduce the majority class: adaptive pooling per segment, then energy pooling per group."""
    representatives = []
    for signal in signals:
        if len(signal) < window and not keep_short:
            continue
        candidates = [signal[i : i + window] for i in range(0, len(signal) - window + 1, window)]
        if len(signal) >= window:
            candidates.append(signal[-window:])
        else:
            candidates.append(np.pad(signal, (window - len(signal), 0), mode="edge"))
        reference = reference_signal(candidates, pool_method)
        distances = [np.mean((candidate - reference) ** 2) for candidate in candidates]
        representatives.append(candidates[int(np.argmin(distances))])
    return select_by_energy(representatives, group_size, selector)


def overwindow(signals, window, stride, backward=True, keep_padded=True):
    """Expand a minority class with forward and half-stride backward passes."""
    windows = []
    for signal in signals:
        windows += forward_windows(signal, window, stride)
        if backward:
            windows += backward_windows(signal, window, stride, int(0.5 * stride), keep_padded)
    return windows


def adaptive_strides(counts, base_stride):
    """Return the majority class, the ranked minority classes, and each minority stride."""
    majority = max(counts, key=counts.get)
    minority = sorted((name for name in counts if name != majority), key=counts.get, reverse=True)
    reference = counts[minority[0]]
    strides = {name: max(1, int(round(base_stride * counts[name] / reference))) for name in minority}
    return majority, minority, strides


def signals_by_class(segments):
    grouped = {name: [s.signal for s in segments if s.label == name] for name in CLASSES}
    return {name: signals for name, signals in grouped.items() if signals}


def to_arrays(class_windows):
    windows, labels = [], []
    for name, values in class_windows:
        windows += values
        labels += [CLASS_INDEX[name]] * len(values)
    return np.asarray(windows, dtype=np.float32)[..., np.newaxis], np.asarray(labels, dtype=np.int64)


def build_static(segments):
    grouped = signals_by_class(segments)
    return to_arrays(
        [
            (name, [w for signal in signals for w in static_windows(signal, WINDOW_SIZE)])
            for name, signals in grouped.items()
        ]
    )


def build_hybrid(segments, variant="full", pool_method=POOL_METHOD, group_size=GROUP_SIZE):
    """Build the hybrid window set, or a variant with one component removed."""
    if variant not in HYBRID_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    grouped = signals_by_class(segments)
    counts = {name: len(signals) for name, signals in grouped.items()}
    majority, minority, strides = adaptive_strides(counts, BASE_STRIDE)
    if variant == "no_adaptive":
        strides = {name: BASE_STRIDE for name in minority}

    if variant == "ow_only":
        majority_windows = [w for s in grouped[majority] for w in static_windows(s, WINDOW_SIZE)]
    else:
        majority_windows = underwindow(
            grouped[majority],
            WINDOW_SIZE,
            pool_method,
            group_size,
            selector="random" if variant == "no_energy" else "energy",
            keep_short=variant != "no_padding",
        )

    class_windows = [(majority, majority_windows)]
    for name in minority:
        if variant == "uw_only":
            windows = [w for s in grouped[name] for w in static_windows(s, WINDOW_SIZE)]
        else:
            windows = overwindow(
                grouped[name],
                WINDOW_SIZE,
                strides[name],
                backward=variant != "no_phase",
                keep_padded=variant != "no_padding",
            )
        class_windows.append((name, windows))
    return to_arrays(class_windows)


def build_matched_random(segments):
    """Match the hybrid class volumes by random resampling of static windows."""
    _, hybrid_labels = build_hybrid(segments)
    static_x, static_y = build_static(segments)
    generator = np.random.default_rng(GROUP_ORDER_SEED)
    indices = []
    for index in range(len(CLASSES)):
        available = np.flatnonzero(static_y == index)
        target = int((hybrid_labels == index).sum())
        indices.append(generator.choice(available, target, replace=target > len(available)))
    indices = np.concatenate(indices)
    return static_x[indices], static_y[indices]
