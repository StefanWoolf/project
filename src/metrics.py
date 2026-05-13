from __future__ import annotations
from typing import Iterable

import numpy as np
import pandas as pd

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommendations: dict[int, list],
                   ground_truth: dict[int, set],
                   k: int) -> float:
    scores = []
    for user_id, recs in recommendations.items():
        gt = ground_truth.get(user_id, set())
        if not gt:
            continue
        top_k = recs[:k]
        if len(top_k) == 0:
            scores.append(0.0)
            continue
        hits = sum(1 for item in top_k if item in gt)
        scores.append(hits / k)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(recommendations: dict[int, list],
                ground_truth: dict[int, set],
                k: int) -> float:

    scores = []
    for user_id, recs in recommendations.items():
        gt = ground_truth.get(user_id, set())
        if not gt:
            continue
        top_k = recs[:k]
        hits = sum(1 for item in top_k if item in gt)
        scores.append(hits / len(gt))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(recommendations: dict[int, list],
              ground_truth: dict[int, set],
              k: int) -> float:
    scores = []
    for user_id, recs in recommendations.items():
        gt = ground_truth.get(user_id, set())
        if not gt:
            continue
        top_k = recs[:k]
        gains = np.array([1.0 if item in gt else 0.0 for item in top_k])
        if gains.size == 0:
            scores.append(0.0)
            continue
        discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
        dcg = float(np.sum(gains * discounts))
        ideal_hits = min(len(gt), k)
        ideal_gains = np.ones(ideal_hits)
        ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_hits + 2))
        idcg = float(np.sum(ideal_gains * ideal_discounts))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(recommendations: dict[int, list],
                  ground_truth: dict[int, set],
                  k: int) -> float:

    hits = 0
    total = 0
    for user_id, recs in recommendations.items():
        gt = ground_truth.get(user_id, set())
        if not gt:
            continue
        total += 1
        if any(item in gt for item in recs[:k]):
            hits += 1
    return hits / total if total > 0 else 0.0


def coverage(recommendations: dict[int, list],
             all_items: Iterable) -> float:

    all_items_set = set(all_items)
    if not all_items_set:
        return 0.0
    recommended = set()
    for recs in recommendations.values():
        recommended.update(recs)
    return len(recommended & all_items_set) / len(all_items_set)


def build_ground_truth(eval_df: pd.DataFrame,
                       relevance_threshold: float = 4.0,
                       user_col: str = 'userId',
                       item_col: str = 'movieId',
                       rating_col: str = 'rating') -> dict[int, set]:

    relevant = eval_df[eval_df[rating_col] >= relevance_threshold]
    return relevant.groupby(user_col)[item_col].apply(set).to_dict()


def evaluate_topn(recommendations: dict[int, list],
                  ground_truth: dict[int, set],
                  ks: tuple[int, ...] = (5, 10, 20),
                  all_items: Iterable | None = None) -> dict[str, float]:

    out: dict[str, float] = {}
    for k in ks:
        out[f'precision@{k}'] = precision_at_k(recommendations, ground_truth, k)
        out[f'recall@{k}'] = recall_at_k(recommendations, ground_truth, k)
        out[f'ndcg@{k}'] = ndcg_at_k(recommendations, ground_truth, k)
        out[f'hit_rate@{k}'] = hit_rate_at_k(recommendations, ground_truth, k)
    if all_items is not None:
        out[f'coverage@{max(ks)}'] = coverage(recommendations, all_items)
    return out


def evaluate_rating_prediction(y_true: np.ndarray,
                               y_pred: np.ndarray) -> dict[str, float]:
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
    }