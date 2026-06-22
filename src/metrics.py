"""src/metrics.py - shared ranking evaluation. Used by LightGBM, TF-Ranking, and TFRS."""
import numpy as np
import pandas as pd

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at k. relevances must be in predicted rank order."""
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))

def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k for a single query group."""
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    ideal_order = np.argsort(y_true)[::-1]
    ideal_sorted = y_true[ideal_order]
    idcg = dcg_at_k(ideal_sorted, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(y_true_sorted, k) / idcg

def mrr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Reciprocal Rank: reciprocal rank of the first relevant item."""
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] > 0:
            return 1.0 / rank
    return 0.0

def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall@k: fraction of relevant items in top-k."""
    n_relevant = np.sum(y_true > 0)
    if n_relevant == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    top_k = y_true[order[:k]]
    return float(np.sum(top_k > 0)) / n_relevant

def evaluate_groups(df: pd.DataFrame, score_col: str, label_col: str, group_col: str, k_values: list[int]) -> dict:
    """
    Compute NDCG@k MRR, Recall@k per group, return mean over all groups.
    Groups with no relevant items contribute 0 to all metrics.
    """

    results = {f"ndcg@{k}": [] for k in k_values}
    results["mrr"] = []
    for k in k_values:
        results[f"recall@{k}"] = []

    for _, group in df.groupby(group_col, sort=False):
        y_true = group[label_col].values.astype(float)
        y_score = group[score_col].values.astype(float)
        results["mrr"].append(mrr(y_true, y_score))
        for k in k_values: 
            results[f"ndcg@{k}"].append(ndcg_at_k(y_true, y_score, k))
            results[f"recall@{k}"].append(recall_at_k(y_true, y_score, k))

    return {metric: float(np.mean(vals)) for metric, vals in results.items()}
