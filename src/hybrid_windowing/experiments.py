"""Experiment groups reported in the article."""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .config import (
    ALL_RECORDS,
    CLASSES,
    CV_FOLDS,
    EPOCHS,
    EXPECTED_WINDOW_COUNTS,
    GRID_GROUPS,
    GRID_POOLS,
    GROUP_SIZE,
    INTRA_SEEDS,
    POOL_METHOD,
    SEEDS,
    TEST_RECORDS,
    TRAIN_RECORDS,
    VALIDATION_RECORDS,
)
from .metrics import compute_metrics
from .rpeak import build_rpeak_hybrid, build_rpeak_static, static_membership_mask
from .training import train_and_predict
from .windowing import build_hybrid, build_matched_random, build_static

GROUPS = ("reference", "rpeak", "components", "intra", "grid", "cross_validation")
ABLATION = {
    "baseline": (False, False),
    "cw_only": (False, True),
    "hw_only": (True, False),
    "hw_cw": (True, True),
}
COMPONENTS = {
    "no_adaptive": ("no_adaptive", False),
    "no_phase": ("no_phase", False),
    "no_padding": ("no_padding", False),
    "no_energy": ("no_energy", False),
    "uw_only": ("uw_only", False),
    "ow_only": ("ow_only", False),
    "matched_random": ("matched_random", False),
    "hw_cw_no_padding": ("no_padding", True),
}
SMOKE_LIMITS = {"train": 1500, "validation": 1000, "test": 2000}


def class_counts(labels):
    return np.bincount(labels, minlength=len(CLASSES)).tolist()


class ExperimentRunner:
    """Run experiment groups; a run that already has metrics.json is skipped."""

    def __init__(self, dataset, output_dir, smoke=False):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.smoke = smoke
        self.seeds = SEEDS[:1] if smoke else SEEDS
        self.intra_seeds = INTRA_SEEDS[:1] if smoke else INTRA_SEEDS
        self.folds = CV_FOLDS[:1] if smoke else CV_FOLDS
        self.epochs = 1 if smoke else EPOCHS
        self.window_log = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_environment(self):
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        try:
            device = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except OSError:
            device = "none"
        environment = {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "gpu": device,
            "tf_sees_gpu": len(gpus),
            "smoke": self.smoke,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (self.output_dir / "env.json").write_text(json.dumps(environment, indent=2))
        if not gpus:
            print("Warning: no GPU detected. Scores may differ from the reported values.")

    def is_done(self, name):
        return (self.runs_dir / name / "metrics.json").exists()

    def log_windows(self, group, name, labels):
        self.window_log.append({"group": group, "windows": name, **dict(zip(CLASSES, class_counts(labels)))})
        pd.DataFrame(self.window_log).drop_duplicates(["group", "windows"]).to_csv(
            self.output_dir / "window_counts.csv", index=False
        )

    def check_counts(self, name, labels):
        counts = class_counts(labels)
        if counts != list(EXPECTED_WINDOW_COUNTS[name]):
            raise RuntimeError(f"Window count mismatch for {name}: {counts}")
        print(f"Window counts verified for {name}: {counts}")

    def limit(self, windows, labels, part):
        size = SMOKE_LIMITS[part]
        if not self.smoke or len(labels) <= size:
            return windows, labels
        indices = np.sort(np.random.default_rng(0).choice(len(labels), size, replace=False))
        return windows[indices], labels[indices]

    def run(self, name, train, validation, test, use_class_weights, seed, metadata, directory=None):
        run_dir = directory or self.runs_dir / name
        if directory is None and self.is_done(name):
            print(f"Skipping {name}")
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        train = self.limit(*train, "train")
        validation = self.limit(*validation, "validation") if validation is not None else None
        test_x, test_y = self.limit(*test, "test")

        result = train_and_predict(train, validation, test_x, use_class_weights, seed, self.epochs)
        metrics, matrix, predictions = compute_metrics(test_y, result.probabilities)
        metrics.update(metadata)
        metrics.update(
            run=name,
            seed=seed,
            class_weight=use_class_weights,
            train_sec=round(result.seconds, 2),
            epochs_run=len(result.history["loss"]),
            n_train=int(len(train[1])),
            train_counts=class_counts(train[1]),
            test_hash=hashlib.md5(test_y.tobytes()).hexdigest(),
        )
        np.savez_compressed(
            run_dir / "predictions.npz",
            y_true=test_y,
            y_pred=predictions,
            prob=result.probabilities.astype(np.float32),
        )
        pd.DataFrame(result.history).to_csv(run_dir / "history.csv", index=False)
        pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(run_dir / "confusion_matrix.csv")
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(
            f"{name}: macro-F1 {metrics['macro_f1']:.4f}, weighted F1 {metrics['weighted_f1']:.4f}, "
            f"{metrics['epochs_run']} epochs, {result.seconds:.0f} s"
        )

    def _partitions(self):
        return (
            self.dataset.subset(TRAIN_RECORDS),
            self.dataset.subset(VALIDATION_RECORDS),
            self.dataset.subset(TEST_RECORDS),
        )

    def reference(self):
        print("Group: reference configurations, DS1/DS2")
        train, validation, test = self._partitions()
        validation_set = build_static(validation)
        test_set = build_static(test)
        static_set = build_static(train)
        hybrid_set = build_hybrid(train)
        self.check_counts("validation_static", validation_set[1])
        self.check_counts("test_static", test_set[1])
        self.check_counts("train_static", static_set[1])
        self.check_counts("train_hybrid", hybrid_set[1])
        self.log_windows("reference", "train_static", static_set[1])
        self.log_windows("reference", "train_hybrid", hybrid_set[1])
        for experiment, (hybrid, weighted) in ABLATION.items():
            for seed in self.seeds:
                self.run(
                    f"ds1ds2_{experiment}_seed{seed}",
                    hybrid_set if hybrid else static_set,
                    validation_set,
                    test_set,
                    weighted,
                    seed,
                    {"group": "reference", "protocol": "ds1ds2", "experiment": experiment},
                )
        self._check_determinism(hybrid_set, validation_set, test_set)

    def _check_determinism(self, hybrid_set, validation_set, test_set):
        directory = self.output_dir / "determinism_check"
        if not (directory / "metrics.json").exists():
            self.run("determinism_check", hybrid_set, validation_set, test_set, False, 0, {"group": "check"}, directory)
        first = np.load(self.runs_dir / "ds1ds2_hw_only_seed0" / "predictions.npz")["y_pred"]
        second = np.load(directory / "predictions.npz")["y_pred"]
        report = {
            "identical_predictions": bool(np.array_equal(first, second)),
            "n_different": int((first != second).sum()),
            "n": int(len(first)),
        }
        (self.output_dir / "determinism_check.json").write_text(json.dumps(report))
        print(f"Determinism check: {report}")

    def rpeak(self):
        print("Group: R-peak-centred windows, DS1/DS2")
        signals = self.dataset.signals
        train, validation, test = self._partitions()
        validation_set = build_rpeak_static(validation, signals)
        test_set = build_rpeak_static(test, signals)
        np.save(self.output_dir / "rp_test_in_static_mask.npy", static_membership_mask(test, signals))
        static_set = build_rpeak_static(train, signals)
        hybrid_set = build_rpeak_hybrid(train, signals, EXPECTED_WINDOW_COUNTS["train_hybrid"])
        self.log_windows("rpeak", "train_rp_static", static_set[1])
        self.log_windows("rpeak", "train_rp_hybrid", hybrid_set[1])
        self.log_windows("rpeak", "val_rp_static", validation_set[1])
        self.log_windows("rpeak", "test_rp_static", test_set[1])
        for experiment, (hybrid, weighted) in ABLATION.items():
            for seed in self.seeds:
                self.run(
                    f"ds1ds2_rp_{experiment}_seed{seed}",
                    hybrid_set if hybrid else static_set,
                    validation_set,
                    test_set,
                    weighted,
                    seed,
                    {"group": "rpeak", "protocol": "ds1ds2", "experiment": f"rp_{experiment}"},
                )

    def components(self):
        print("Group: component-level ablation, DS1/DS2")
        train, validation, test = self._partitions()
        validation_set = build_static(validation)
        test_set = build_static(test)
        window_sets = {}
        for experiment, (variant, weighted) in COMPONENTS.items():
            if variant not in window_sets:
                window_sets[variant] = (
                    build_matched_random(train) if variant == "matched_random" else build_hybrid(train, variant)
                )
                self.log_windows("components", variant, window_sets[variant][1])
            for seed in self.seeds:
                self.run(
                    f"ds1ds2_{experiment}_seed{seed}",
                    window_sets[variant],
                    validation_set,
                    test_set,
                    weighted,
                    seed,
                    {"group": "components", "protocol": "ds1ds2", "experiment": experiment},
                )

    def _intra_runs(self, experiment, group, window_set, seeds=None):
        windows, labels = window_set
        for seed in seeds if seeds is not None else self.intra_seeds:
            name = f"intra_{experiment}_seed{seed}"
            if self.is_done(name):
                print(f"Skipping {name}")
                continue
            train_x, test_x, train_y, test_y = train_test_split(
                windows, labels, test_size=0.2, stratify=labels, random_state=seed
            )
            self.run(
                name,
                (train_x, train_y),
                None,
                (test_x, test_y),
                True,
                seed,
                {"group": group, "protocol": "intra", "experiment": experiment},
            )

    def intra(self):
        print("Group: intra-patient split")
        signals = self.dataset.signals
        segments = self.dataset.subset(ALL_RECORDS)
        static_set = build_static(segments)
        hybrid_set = build_hybrid(segments)
        self.check_counts("intra_static", static_set[1])
        self.check_counts("intra_hybrid", hybrid_set[1])
        self.log_windows("intra", "intra_static", static_set[1])
        self.log_windows("intra", "intra_hybrid", hybrid_set[1])
        self._intra_runs("cw_only", "intra", static_set)
        self._intra_runs("hw_cw", "intra", hybrid_set)
        del static_set, hybrid_set

        rpeak_static = build_rpeak_static(segments, signals)
        self.log_windows("intra", "intra_rp_static", rpeak_static[1])
        self._intra_runs("rp_cw_only", "intra", rpeak_static)
        del rpeak_static

        rpeak_hybrid = build_rpeak_hybrid(segments, signals, EXPECTED_WINDOW_COUNTS["intra_hybrid"])
        self.log_windows("intra", "intra_rp_hybrid", rpeak_hybrid[1])
        self._intra_runs("rp_hw_cw", "intra", rpeak_hybrid)
        del rpeak_hybrid

        no_padding = build_hybrid(segments, "no_padding")
        self.log_windows("intra", "intra_hybrid_no_padding", no_padding[1])
        self._intra_runs("hw_cw_no_padding", "intra", no_padding)

    def grid(self):
        print("Group: pooling method and group size grid, intra-patient split")
        segments = self.dataset.subset(ALL_RECORDS)
        for pool_method in GRID_POOLS:
            for group_size in GRID_GROUPS:
                if (pool_method, group_size) == (POOL_METHOD, GROUP_SIZE):
                    continue
                experiment = f"grid_{pool_method}_g{group_size}"
                window_set = build_hybrid(segments, pool_method=pool_method, group_size=group_size)
                self.log_windows("grid", experiment, window_set[1])
                self._intra_runs(experiment, "grid", window_set, seeds=(0,))

    def cross_validation(self):
        print("Group: patient-level five-fold cross-validation")
        for index, fold in enumerate(self.folds):
            train = self.dataset.subset(fold["train"].split())
            validation_set = build_static(self.dataset.subset(fold["validation"].split()))
            test_set = build_static(self.dataset.subset(fold["test"].split()))
            static_set = build_static(train)
            hybrid_set = build_hybrid(train)
            self.log_windows("cross_validation", f"fold{index}_static", static_set[1])
            self.log_windows("cross_validation", f"fold{index}_hybrid", hybrid_set[1])
            for experiment, (hybrid, weighted) in ABLATION.items():
                self.run(
                    f"cv5_{experiment}_fold{index}",
                    hybrid_set if hybrid else static_set,
                    validation_set,
                    test_set,
                    weighted,
                    index,
                    {"group": "cross_validation", "protocol": "cv5", "experiment": experiment, "fold": index},
                )
