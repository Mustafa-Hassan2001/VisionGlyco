"""Evaluation metrics. Every number here is computed on held-out patients."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .config import ISO_ABS_LIMIT_MG_DL, ISO_REL_LIMIT, ISO_SWITCH_MG_DL


def choose_threshold(y_true, y_prob):
    """Pick the decision threshold on the *validation* set (Youden's J).

    Choosing it on the test set would leak test information into the reported
    accuracy.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(np.clip(thresholds[best], 0.0, 1.0))


def classification_metrics(y_true, y_prob, threshold, n_bootstrap=1000, seed=0):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2

    def safe_div(a, b):
        return float(a / b) if b else None

    result = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both_classes else None,
        "auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "sensitivity": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "ppv": safe_div(tp, tp + fp),
        "npv": safe_div(tn, tn + fn),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # Accuracy you would get by always predicting the majority class.
        "majority_baseline_accuracy": float(max(y_true.mean(), 1 - y_true.mean())),
    }
    result["accuracy_ci95"] = _bootstrap_ci(
        lambda t, p: accuracy_score(t, (p >= threshold).astype(int)), y_true, y_prob, n_bootstrap, seed)
    if both_classes:
        result["auc_ci95"] = _bootstrap_ci(roc_auc_score, y_true, y_prob, n_bootstrap, seed,
                                           need_both_classes=True)
    return result


def regression_metrics(y_true, y_pred):
    """MAE plus the glucose-meter metrics regulators ask for (MARD, ISO 15197 zone)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_pred - y_true)
    within_iso = np.where(
        y_true < ISO_SWITCH_MG_DL,
        abs_err <= ISO_ABS_LIMIT_MG_DL,
        abs_err <= ISO_REL_LIMIT * y_true,
    )
    baseline = np.abs(y_true - y_true.mean())
    return {
        "n": int(len(y_true)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mard_percent": float(np.mean(abs_err / np.maximum(y_true, 1e-6)) * 100),
        "within_iso15197_percent": float(within_iso.mean() * 100),
        # ISO 15197:2013 requires >= 95% of readings inside the zone.
        "meets_iso15197": bool(within_iso.mean() >= 0.95),
        "pearson_r": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and y_pred.std() > 0 else None,
        # MAE of always predicting the test-set mean; a useful model must beat this.
        "mean_baseline_mae": float(baseline.mean()),
    }


def _bootstrap_ci(metric, y_true, y_score, n_bootstrap, seed, need_both_classes=False):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if need_both_classes and len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric(y_true[idx], y_score[idx]))
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]
