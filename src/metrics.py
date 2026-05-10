"""Метрики качества для рекомендательной системы.

Все функции принимают и возвращают чистые numpy/pandas объекты.
Используются всеми ноутбуками обучения (04–10) и Streamlit-приложением.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Метрики предсказания рейтинга (regression-style)
# ─────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Корень среднеквадратичной ошибки.

    Параметры:
        y_true: истинные оценки.
        y_pred: предсказанные оценки.
    Возвращает:
        RMSE как float.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка.

    Параметры:
        y_true: истинные оценки.
        y_pred: предсказанные оценки.
    Возвращает:
        MAE как float.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


# ─────────────────────────────────────────────
# Метрики ранжирования (top-N)
# ─────────────────────────────────────────────
#
# Контракт top-N метрик:
# - recommendations: dict {user_id: list[item_id]} — упорядоченный список
#   рекомендаций для каждого пользователя (длина ≤ K).
# - ground_truth: dict {user_id: set[item_id]} — множество «релевантных»
#   фильмов для пользователя (по умолчанию: фильмы из test/val,
#   которые пользователь оценил >= relevance_threshold).
# - K: длина рекомендации.

def precision_at_k(recommendations: dict[int, list],
                   ground_truth: dict[int, set],
                   k: int) -> float:
    """Усреднённая по пользователям Precision@K.

    Precision@K(u) = |Rec_u(K) ∩ GT_u| / K.
    Пользователи без ground_truth исключаются из усреднения.

    Параметры:
        recommendations: словарь {userId: [movieId, ...]}.
        ground_truth: словарь {userId: {movieId, ...}}.
        k: длина рекомендации.
    Возвращает:
        Среднюю Precision@K по пользователям как float.
    """
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
    """Усреднённая по пользователям Recall@K.

    Recall@K(u) = |Rec_u(K) ∩ GT_u| / |GT_u|.
    Пользователи без ground_truth исключаются.

    Параметры:
        recommendations: словарь {userId: [movieId, ...]}.
        ground_truth: словарь {userId: {movieId, ...}}.
        k: длина рекомендации.
    Возвращает:
        Средний Recall@K по пользователям как float.
    """
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
    """Усреднённая по пользователям NDCG@K с бинарной релевантностью.

    DCG@K = Σ_{i=1..K} rel_i / log2(i + 1),  rel_i ∈ {0, 1}.
    IDCG@K = DCG идеального списка длины min(K, |GT|).
    NDCG@K = DCG@K / IDCG@K.

    Параметры:
        recommendations: словарь {userId: [movieId, ...]}.
        ground_truth: словарь {userId: {movieId, ...}}.
        k: длина рекомендации.
    Возвращает:
        Средний NDCG@K по пользователям как float.
    """
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
    """Доля пользователей, у которых хотя бы один релевантный фильм попал в топ-K.

    Параметры:
        recommendations: словарь {userId: [movieId, ...]}.
        ground_truth: словарь {userId: {movieId, ...}}.
        k: длина рекомендации.
    Возвращает:
        Hit Rate@K как float от 0 до 1.
    """
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
    """Доля уникальных фильмов из all_items, попавших хотя бы в одну рекомендацию.

    Параметры:
        recommendations: словарь {userId: [movieId, ...]}.
        all_items: полный каталог фильмов (итерируемое).
    Возвращает:
        Coverage как float от 0 до 1.
    """
    all_items_set = set(all_items)
    if not all_items_set:
        return 0.0
    recommended = set()
    for recs in recommendations.values():
        recommended.update(recs)
    return len(recommended & all_items_set) / len(all_items_set)


# ─────────────────────────────────────────────
# Хелперы построения ground_truth и оценочного бандла
# ─────────────────────────────────────────────

def build_ground_truth(eval_df: pd.DataFrame,
                       relevance_threshold: float = 4.0,
                       user_col: str = 'userId',
                       item_col: str = 'movieId',
                       rating_col: str = 'rating') -> dict[int, set]:
    """Собрать ground_truth: для каждого пользователя — множество фильмов с rating >= порога.

    Параметры:
        eval_df: DataFrame с оценками (val или test).
        relevance_threshold: фильмы с этой оценкой и выше считаются «релевантными».
        user_col: имя колонки с userId.
        item_col: имя колонки с movieId.
        rating_col: имя колонки с рейтингом.
    Возвращает:
        Словарь {userId: set(movieId)}.
    """
    relevant = eval_df[eval_df[rating_col] >= relevance_threshold]
    return relevant.groupby(user_col)[item_col].apply(set).to_dict()


def evaluate_topn(recommendations: dict[int, list],
                  ground_truth: dict[int, set],
                  ks: tuple[int, ...] = (5, 10, 20),
                  all_items: Iterable | None = None) -> dict[str, float]:
    """Свести все top-N метрики в один словарь {metric_name: value}.

    Возвращает плоский словарь:
        precision@5, recall@5, ndcg@5, hit_rate@5,
        precision@10, ..., и (если задан all_items) coverage@K_max.

    Параметры:
        recommendations: словарь {userId: [movieId, ...]}.
        ground_truth: словарь {userId: {movieId, ...}}.
        ks: кортеж значений K для вычисления метрик.
        all_items: полный каталог для расчёта coverage (опционально).
    Возвращает:
        Плоский словарь метрик.
    """
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
    """Свести регрессионные метрики в один словарь.

    Параметры:
        y_true: массив истинных рейтингов.
        y_pred: массив предсказанных рейтингов.
    Возвращает:
        Словарь {'rmse': float, 'mae': float}.
    """
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
    }