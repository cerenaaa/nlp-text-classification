"""
Multi-label classification evaluation: micro/macro F1, Hamming loss, coverage.
"""
import numpy as np
from sklearn.metrics import f1_score, hamming_loss, coverage_error, label_ranking_loss


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray = None) -> dict:
    results = {
        "micro_f1": round(f1_score(y_true, y_pred, average="micro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "hamming_loss": round(hamming_loss(y_true, y_pred), 4),
        "subset_accuracy": round(float(np.mean(np.all(y_true == y_pred, axis=1))), 4),
    }
    if y_score is not None:
        results["coverage_error"] = round(coverage_error(y_true, y_score), 4)
        results["ranking_loss"] = round(label_ranking_loss(y_true, y_score), 4)
    for k, v in results.items():
        print(f"  {k:25s}: {v:.4f}")
    return results
