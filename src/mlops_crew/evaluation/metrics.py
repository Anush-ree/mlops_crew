"""Binary classification metrics for the phishing detection task.

Recall and the false-negative rate are what we care about most: missing a real
phishing email (false negative) is more costly than flagging a legit one. F2
is included so model selection biases toward recall over precision.
"""

from __future__ import annotations

from typing import Any

from sklearn import metrics as sk_metrics


def binary_classification_report(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    *,
    positive_label: int = 1,
) -> dict[str, float]:
    """Threshold metrics, confusion-matrix counts, and (optional) score metrics."""
    tn, fp, fn, tp = sk_metrics.confusion_matrix(
        y_true, y_pred, labels=[0, 1]).ravel()

    report = {
        "accuracy": float(sk_metrics.accuracy_score(y_true, y_pred)),
        "precision": float(
            sk_metrics.precision_score(
                y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            sk_metrics.recall_score(
                y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(sk_metrics.f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "f2": float(
            sk_metrics.fbeta_score(
                y_true, y_pred, beta=2, pos_label=positive_label, zero_division=0
            )
        ),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }

    if y_score is not None:
        report["roc_auc"] = float(sk_metrics.roc_auc_score(y_true, y_score))
        report["average_precision"] = float(
            sk_metrics.average_precision_score(y_true, y_score))
    return report
