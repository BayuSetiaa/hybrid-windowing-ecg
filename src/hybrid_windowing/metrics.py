"""Classification metrics computed from predicted probabilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import CLASSES


def compute_metrics(labels, probabilities):
    """Return a metrics dictionary, the confusion matrix, and the predicted labels."""
    predictions = probabilities.argmax(1)
    present = sorted(np.unique(labels).tolist())
    metrics = {
        "n_test": int(len(labels)),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
    }
    for average in ("weighted", "macro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, labels=present, average=average, zero_division=0
        )
        metrics[f"{average}_precision"] = precision
        metrics[f"{average}_recall"] = recall
        metrics[f"{average}_f1"] = f1

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=present, zero_division=0
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASSES))))
    roc_aucs, pr_aucs = [], []
    for position, index in enumerate(present):
        name = CLASSES[index]
        true_negatives = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        binary = (labels == index).astype(int)
        roc_auc = (
            roc_auc_score(binary, probabilities[:, index])
            if 0 < binary.sum() < len(binary)
            else float("nan")
        )
        pr_auc = average_precision_score(binary, probabilities[:, index])
        metrics[f"{name}_sensitivity"] = recall[position]
        metrics[f"{name}_specificity"] = true_negatives / (matrix.sum() - matrix[index, :].sum())
        metrics[f"{name}_precision"] = precision[position]
        metrics[f"{name}_f1"] = f1[position]
        metrics[f"{name}_support"] = int(support[position])
        metrics[f"{name}_roc_auc"] = roc_auc
        metrics[f"{name}_pr_auc"] = pr_auc
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

    weights = support / support.sum()
    metrics["macro_roc_auc"] = float(np.nanmean(roc_aucs))
    metrics["weighted_roc_auc"] = float(np.nansum(weights * np.array(roc_aucs)))
    metrics["macro_pr_auc"] = float(np.mean(pr_aucs))
    metrics = {
        key: float(value) if isinstance(value, (np.floating, float)) else value
        for key, value in metrics.items()
    }
    return metrics, matrix, predictions
