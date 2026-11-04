"""Compute the statistics for the reference configurations (Tables 4 and 6)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hybrid_windowing.statistics import confidence_interval, holm_adjust, load_results, mcnemar_exact

KEY_METRICS = [
    "accuracy", "balanced_accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "macro_precision",
    "macro_recall", "macro_f1", "macro_roc_auc", "weighted_roc_auc", "macro_pr_auc",
]
TEST_METRICS = ["macro_f1", "balanced_accuracy", "weighted_f1"]
COMPARISONS = [
    ("hw_only", "baseline"), ("hw_cw", "cw_only"), ("cw_only", "baseline"),
    ("hw_cw", "hw_only"), ("hw_cw", "baseline"),
]
EXPERIMENT_ORDER = ["baseline", "cw_only", "hw_only", "hw_cw"]
TABLE4_METRICS = [
    "accuracy", "weighted_precision", "weighted_recall", "weighted_f1", "weighted_roc_auc",
    "macro_f1", "balanced_accuracy", "train_sec",
]


def summarize(results):
    per_class = [
        column
        for column in results.columns
        if column.endswith(("_sensitivity", "_specificity", "_f1", "_pr_auc"))
        and column.split("_")[0] in ("N", "S", "V", "F", "Q")
    ]
    full, formatted = [], []
    for (protocol, experiment), group in results.groupby(["protocol", "experiment"]):
        base = {"protocol": protocol, "experiment": experiment, "n_seeds": len(group)}
        row, text = dict(base), dict(base)
        for metric in KEY_METRICS + per_class:
            if metric not in group:
                continue
            low, high = confidence_interval(group[metric])
            row[f"{metric}_mean"], row[f"{metric}_sd"] = group[metric].mean(), group[metric].std(ddof=1)
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = low, high
            text[metric] = f"{group[metric].mean():.4f} ± {group[metric].std(ddof=1):.4f}"
        full.append(row)
        formatted.append(text)
    order = {experiment: index for index, experiment in enumerate(EXPERIMENT_ORDER)}

    def arrange(table):
        ranked = table.assign(_order=table["experiment"].map(order))
        return ranked.sort_values(["protocol", "_order"]).drop(columns="_order")

    return arrange(pd.DataFrame(full)), arrange(pd.DataFrame(formatted))


def paired_tests(results):
    rows = []
    for protocol, group in results.groupby("protocol"):
        for metric in TEST_METRICS:
            block = []
            for first, second in COMPARISONS:
                a = group[group.experiment == first].set_index("seed")[metric]
                b = group[group.experiment == second].set_index("seed")[metric]
                seeds = sorted(set(a.index) & set(b.index))
                if len(seeds) < 2:
                    continue
                difference = a.loc[seeds].values - b.loc[seeds].values
                try:
                    wilcoxon = stats.wilcoxon(difference).pvalue
                except ValueError:
                    wilcoxon = np.nan
                low, high = confidence_interval(difference)
                sd = difference.std(ddof=1)
                block.append(
                    {
                        "protocol": protocol,
                        "metric": metric,
                        "comparison": f"{first} - {second}",
                        "n_seeds": len(seeds),
                        "mean_diff": difference.mean(),
                        "diff_ci_low": low,
                        "diff_ci_high": high,
                        "cohens_dz": difference.mean() / sd if sd > 0 else np.nan,
                        "paired_t_p": stats.ttest_rel(a.loc[seeds], b.loc[seeds]).pvalue,
                        "wilcoxon_p": wilcoxon,
                    }
                )
            if block:
                for row, adjusted in zip(block, holm_adjust([r["paired_t_p"] for r in block])):
                    row["paired_t_p_holm"] = adjusted
                rows += block
    return pd.DataFrame(rows)


def mcnemar_tests(results_dir, results):
    rows = []
    seeds = sorted(set(results[results.protocol == "ds1ds2"].seed))
    for first, second in COMPARISONS:
        for seed in seeds:
            path_a = results_dir / "runs" / f"ds1ds2_{first}_seed{seed}" / "predictions.npz"
            path_b = results_dir / "runs" / f"ds1ds2_{second}_seed{seed}" / "predictions.npz"
            if not (path_a.exists() and path_b.exists()):
                continue
            a, b = np.load(path_a), np.load(path_b)
            if not np.array_equal(a["y_true"], b["y_true"]):
                raise ValueError("McNemar requires identical test labels")
            only_a, only_b, p_value = mcnemar_exact(a["y_true"], a["y_pred"], b["y_pred"])
            rows.append(
                {
                    "comparison": f"{first} vs {second}",
                    "seed": seed,
                    "A_correct_B_wrong": only_a,
                    "A_wrong_B_correct": only_b,
                    "mcnemar_exact_p": p_value,
                }
            )
    return pd.DataFrame(rows) if rows else None


def scenario_name(experiment):
    if experiment == "cw_only":
        return "360 (Baseline)"
    if experiment == "hw_cw":
        return "360-288-max-3"
    _, pool_method, group_size = experiment.split("_")
    return f"360-288-{pool_method}-{group_size[1:]}"


def table4(results):
    rows = []
    for protocol, group in results.groupby("protocol"):
        selected = group[
            (group.experiment == "cw_only") | (group.experiment == "hw_cw") | group.experiment.str.startswith("grid_")
        ]
        for experiment, runs in selected.groupby("experiment"):
            row = {"protocol": protocol, "scenario": scenario_name(experiment), "n_seeds": len(runs)}
            for metric in TABLE4_METRICS:
                row[metric] = (
                    f"{runs[metric].mean():.4f} ± {runs[metric].std(ddof=1):.4f}"
                    if len(runs) > 1
                    else f"{runs[metric].mean():.4f}"
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["protocol", "scenario"])


def protocol_gap(results):
    rows = []
    for experiment in EXPERIMENT_ORDER:
        intra = results[(results.protocol == "intra") & (results.experiment == experiment)]
        ds1ds2 = results[(results.protocol == "ds1ds2") & (results.experiment == experiment)]
        if len(intra) and len(ds1ds2):
            for metric in ("weighted_f1", "macro_f1", "balanced_accuracy"):
                rows.append(
                    {
                        "experiment": experiment,
                        "metric": metric,
                        "intra_mean": intra[metric].mean(),
                        "ds1ds2_mean": ds1ds2[metric].mean(),
                        "gap": intra[metric].mean() - ds1ds2[metric].mean(),
                    }
                )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reference_statistics"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    results.to_csv(args.output / "results_all_runs.csv", index=False)
    full, formatted = summarize(results)
    full.to_csv(args.output / "summary_mean_sd_ci.csv", index=False)
    formatted.to_csv(args.output / "table_for_manuscript.csv", index=False)
    paired_tests(results).to_csv(args.output / "paired_tests.csv", index=False)
    table4(results).to_csv(args.output / "table4_corrected.csv", index=False)
    protocol_gap(results).to_csv(args.output / "gap_intra_vs_ds1ds2.csv", index=False)
    mcnemar = mcnemar_tests(args.results_dir, results)
    if mcnemar is not None:
        mcnemar.to_csv(args.output / "mcnemar_per_seed.csv", index=False)
    print(f"Tables written to {args.output}")


if __name__ == "__main__":
    main()
