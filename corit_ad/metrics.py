"""Discrete empirical low-FPR anomaly detection and escalation metrics (no interpolation)."""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Computes AUROC, AP, TPR@{1%,2%,5%}FPR, pAUC@5%FPR, and FPR@90%TPR."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if not np.isfinite(s).all():
        raise ValueError("Scores contain non-finite values.")
    if len(np.unique(y)) != 2:
        raise ValueError("Both normal (0) and anomalous (1) classes are required.")

    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    out = {
        "auroc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "pauc5": float(roc_auc_score(y, s, max_fpr=0.05)),
        "fpr90": float(fpr[tpr >= 0.90].min()),
    }
    for f in (0.01, 0.02, 0.05):
        out[f"tpr{int(round(f * 100))}"] = float(tpr[fpr <= f].max())
    return out


def operating_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    tau: float,
    escalate_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes operating-point FPR/TPR and optional escalation statistics."""
    y = np.asarray(y_true, dtype=bool)
    pred = np.asarray(scores) >= float(tau)
    out = {
        "op_fpr": float(pred[~y].mean()),
        "op_tpr": float(pred[y].mean()),
    }
    if escalate_mask is not None:
        rev = np.asarray(escalate_mask, dtype=bool)
        out.update(
            review_fraction=float(rev.mean()),
            review_normal=float(rev[~y].mean()),
            review_anomaly_recall=float(rev[y].mean()),
            anomaly_auto_release=float((~rev[y]).mean()),
        )
    return out
