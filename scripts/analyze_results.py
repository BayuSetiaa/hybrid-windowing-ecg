"""Compute the statistics for the additional experiments (Tables 8 to 10b)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from hybrid_windowing.config import CLASSES
from hybrid_windowing.statistics import holm_adjust, load_results

SUMMARY_METRICS = [
    "macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "weighted_roc_auc", "macro_precision",
    "macro_recall", "N_sensitivity", "S_sensitivity", "V_sensitivity", "F_sensitivity", "N_precision",
    "S_precision", "V_precision", "F_precision",
]
PAIRED_METRICS = ("macro_f1", "weighted_f1", "balanced_accuracy")
ABLATION = ["baseline", "cw_only", "hw_only", "hw_cw"]
ABLATION_PAIRS = [("hw_only", "baseline"), ("hw_cw", "cw_only"), ("cw_only", "baseline"), ("hw_cw", "baseline")]
COMPONENTS = ["no_adaptive", "no_phase", "no_padding", "no_energy", "uw_only", "ow_only", "matched_random"]


def summarize(results):
    groups = results.groupby(["protocol", "experiment"])
    summary = groups[SUMMARY_METRICS].agg(
        lambda v: f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else float('nan'):.3f}"
    )
    summary.insert(0, "n", groups.size())
    return summary


def paired_comparisons(results, protocol, pairs):
    subset = results[results.protocol == protocol]
    rows = []
    for metric in PAIRED_METRICS:
        block = []
        for first, second in pairs:
            a = subset[subset.experiment == first].set_index("seed")[metric]
            b = subset[subset.experiment == second].set_index("seed")[metric]
            common = a.index.intersection(b.index)
            difference = (a.loc[common] - b.loc[common]).values
            n = len(difference)
            row = {"protocol": protocol, "metric": metric, "comparison": f"{first} - {second}", "n": n}
            if n < 2:
                row["mean_diff"] = difference.mean() if n else np.nan
                block.append(row)
                continue
            sd = difference.std(ddof=1)
            margin = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
            row.update(
                mean_diff=difference.mean(),
                ci_low=difference.mean() - margin,
                ci_high=difference.mean() + margin,
                cohens_dz=difference.mean() / sd if sd > 0 else 0.0,
                paired_t_p=stats.ttest_rel(a.loc[common], b.loc[common]).pvalue if sd > 0 else 1.0,
                wins_a=int((difference > 0).sum()),
            )
            block.append(row)
        p_values = [row.get("paired_t_p", np.nan) for row in block]
        if not np.isnan(p_values).any():
            for row, adjusted in zip(block, holm_adjust(p_values)):
                row["p_holm"] = adjusted
        rows += block
    return pd.DataFrame(rows)


def compare_with_first_environment(results, reference_table):
    first = pd.read_csv(reference_table)
    rows = []
    for protocol, experiments in (("ds1ds2", ABLATION), ("intra", ["cw_only", "hw_cw"])):
        for experiment in experiments:
            old = first[(first.protocol == protocol) & (first.experiment == experiment)]
            new = results[(results.protocol == protocol) & (results.experiment == experiment)]
            if len(old) and len(new):
                rows.append(
                    {
                        "protocol": protocol,
                        "experiment": experiment,
                        "first_environment_macro_f1": old.macro_f1.iloc[0],
                        "second_environment_macro_f1": f"{new.macro_f1.mean():.4f} ± {new.macro_f1.std(ddof=1):.4f}",
                        "first_environment_weighted_f1": old.weighted_f1.iloc[0],
                        "second_environment_weighted_f1": f"{new.weighted_f1.mean():.4f} ± {new.weighted_f1.std(ddof=1):.4f}",
                    }
                )
    return pd.DataFrame(rows)


def rpeak_common_beats(results_dir):
    mask_path = results_dir / "rp_test_in_static_mask.npy"
    if not mask_path.exists():
        return None
    mask = np.load(mask_path)
    rows = []
    for experiment in ABLATION:
        for path in sorted((results_dir / "runs").glob(f"ds1ds2_rp_{experiment}_seed*/predictions.npz")):
            data = np.load(path)
            if len(data["y_true"]) != len(mask):
                continue
            labels, predictions = data["y_true"][mask], data["y_pred"][mask]
            rows.append(
                {
                    "experiment": f"rp_{experiment}",
                    "run": path.parent.name,
                    "n": int(mask.sum()),
                    "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(labels, predictions),
                }
            )
    return pd.DataFrame(rows) if rows else None


def best_epochs(results_dir, results):
    rows = []
    for row in results.itertuples():
        path = results_dir / "runs" / row.run / "history.csv"
        if path.exists():
            history = pd.read_csv(path)
            rows.append(
                {
                    "run": row.run,
                    "protocol": row.protocol,
                    "experiment": row.experiment,
                    "epochs": len(history),
                    "best": int(history["val_loss"].values.argmin()) + 1,
                }
            )
    return pd.DataFrame(rows) if rows else None


def pooled_predictions(results_dir, experiment, folds):
    paths = [results_dir / "runs" / f"cv5_{experiment}_fold{fold}" / "predictions.npz" for fold in folds]
    if not all(path.exists() for path in paths):
        return None
    loaded = [np.load(path) for path in paths]
    return np.concatenate([d["y_true"] for d in loaded]), np.concatenate([d["y_pred"] for d in loaded])


def pooled_cross_validation(results_dir, folds, detailed):
    rows = []
    for experiment in ABLATION:
        pooled = pooled_predictions(results_dir, experiment, folds)
        if pooled is None:
            return None
        labels, predictions = pooled
        row = {
            "exp": experiment,
            "n": len(labels),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }
        if detailed:
            matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASSES))))
            sensitivity = matrix.diagonal() / matrix.sum(1)
            precision = matrix.diagonal() / np.maximum(matrix.sum(0), 1)
            row["weighted_f1"] = f1_score(labels, predictions, average="weighted")
            row["ba"] = balanced_accuracy_score(labels, predictions)
            row.update({f"Se_{c}": sensitivity[i] for i, c in enumerate(CLASSES)})
            row.update({f"Pr_{c}": precision[i] for i, c in enumerate(CLASSES)})
            row["support"] = matrix.sum(1).tolist()
        else:
            row["ba"] = balanced_accuracy_score(labels, predictions)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/statistics"))
    parser.add_argument("--reference-table", type=Path, help="table_for_manuscript.csv of the first environment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    results.to_csv(args.output / "results_all_runs.csv", index=False)
    summarize(results).to_csv(args.output / "summary_mean_sd.csv")
    if args.reference_table:
        compare_with_first_environment(results, args.reference_table).to_csv(
            args.output / "A_rerun_vs_first_environment.csv", index=False
        )

    tables = {
        "A_ablation_paired": ("ds1ds2", ABLATION_PAIRS),
        "B_rpeak_paired": (
            "ds1ds2",
            [(f"rp_{e}", e) for e in ABLATION]
            + [("rp_hw_only", "rp_baseline"), ("rp_hw_cw", "rp_cw_only"),
               ("rp_cw_only", "rp_baseline"), ("rp_hw_cw", "rp_baseline")],
        ),
        "C_components_paired": (
            "ds1ds2",
            [(c, "hw_only") for c in COMPONENTS]
            + [("matched_random", "baseline"), ("uw_only", "baseline"), ("ow_only", "baseline")],
        ),
        "C_no_padding_with_cw_paired": ("ds1ds2", [("hw_cw_no_padding", "hw_cw")]),
        "D_intra_paired": (
            "intra",
            [("rp_hw_cw", "hw_cw"), ("rp_cw_only", "cw_only"), ("hw_cw_no_padding", "hw_cw"),
             ("hw_cw", "cw_only"), ("rp_hw_cw", "rp_cw_only")],
        ),
        "E_cv5_paired": ("cv5", ABLATION_PAIRS),
    }
    for name, (protocol, pairs) in tables.items():
        paired_comparisons(results, protocol, pairs).to_csv(args.output / f"{name}.csv", index=False)

    if (results.protocol == "cv5").any():
        folds = results[results.protocol == "cv5"].pivot_table(index="fold", columns="experiment", values="macro_f1")
        folds.to_csv(args.output / "E_cv5_macro_f1_per_fold.csv")

    common = rpeak_common_beats(args.results_dir)
    if common is not None:
        common.to_csv(args.output / "B_rpeak_common_beats_runs.csv", index=False)
        common.groupby("experiment")[["macro_f1", "balanced_accuracy"]].agg(["mean", "std"]).to_csv(
            args.output / "B_rpeak_common_beats_summary.csv"
        )
    epochs = best_epochs(args.results_dir, results)
    if epochs is not None:
        epochs.to_csv(args.output / "best_epoch_per_run.csv", index=False)
    pooled = pooled_cross_validation(args.results_dir, range(5), detailed=True)
    if pooled is not None:
        pooled.to_csv(args.output / "E_cv5_pooled.csv", index=False)
    pooled = pooled_cross_validation(args.results_dir, range(1, 5), detailed=False)
    if pooled is not None:
        pooled.to_csv(args.output / "E_cv5_pooled_folds1to4.csv", index=False)
    print(f"Tables written to {args.output}")


if __name__ == "__main__":
    main()
