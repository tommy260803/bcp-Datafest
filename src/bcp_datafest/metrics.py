import numpy as np
from sklearn.metrics import roc_auc_score


def gini(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    if len(y) != len(p):
        raise ValueError("Labels and predictions must have the same length")
    active = np.ones(len(y), dtype=bool)
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != y.shape or not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Invalid metric weights")
        active = weights > 0
    if len(np.unique(y[active])) < 2:
        return None
    return float(2 * roc_auc_score(y, p, sample_weight=weights) - 1)


def metrics(y, p, no_history, test_fraction):
    y, p, no_history = np.asarray(y), np.asarray(p), np.asarray(no_history, dtype=bool)
    if not (y.shape == p.shape == no_history.shape) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError("Metrics require aligned labels, probabilities and history flags")
    score = gini(y, p)
    result = {"rows": len(y), "positives": int(y.sum()), "gini": score, "auc": None if score is None else (score + 1) / 2}
    for name, mask in [("no_history", no_history), ("history", ~no_history)]:
        result[name] = {"rows": int(mask.sum()), "positives": int(y[mask].sum()), "gini": gini(y[mask], p[mask])}
    r = no_history.mean()
    result["composition_weighted_gini"] = gini(y, p, np.where(no_history, test_fraction / r, (1-test_fraction)/(1-r))) if 0 < r < 1 else None
    k = int(np.ceil(len(y) * .2))
    result["top20_count"] = k
    result["top20_positives"] = int(y[np.argsort(-p, kind="stable")[:k]].sum())
    result["top20_recall"] = float(result["top20_positives"] / y.sum()) if y.sum() else None
    return result
