"""
Streamlit-приложение для демонстрации рекомендательной системы фильмов.

Архитектура:
    - 6 вкладок в боковой панели (radio): Главная, EDA, Сравнение моделей,
      Детали моделей, Получить рекомендации, Метрики и теория.
    - На старте через st.status прелоадим ВСЕ артефакты в кеш
      (после первой загрузки переключение между вкладками мгновенное).
    - Лёгкие артефакты (метаданные, маппинги, метрики) кешируются через
      @st.cache_data; обученные модели - через @st.cache_resource.

Запуск из корня проекта:
    streamlit run streamlit/app.py

Стилистика интерфейса:
    - Русский язык.
    - Без эмодзи и спецсимволов (->, >=, <= вместо стрелочек/неравенств).
    - Чистые текстовые карточки с цветными бейджами жанров.
    - Графики - Plotly (интерактив) + готовые PNG/HTML из models/.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Базовые пути и константы проекта
# ---------------------------------------------------------------------------
# Приложение запускается из корня проекта командой `streamlit run streamlit/app.py`.
# Поэтому корень - это родитель папки streamlit.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROC = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Бинарный порог релевантности (был использован при обучении / валидации).
RELEVANCE_THRESHOLD = 3.5

# Цвета жанровых бейджей (фиксированные, чтобы один и тот же жанр всегда
# выглядел одинаково).
GENRE_COLORS = {
    "Action": "#E24B4A",        "Adventure": "#EF9F27",
    "Animation": "#7F77DD",     "Children": "#F4C0D1",
    "Comedy": "#FAC775",        "Crime": "#5F5E5A",
    "Documentary": "#888780",   "Drama": "#378ADD",
    "Fantasy": "#AFA9EC",       "Film-Noir": "#2C2C2A",
    "Horror": "#791F1F",        "IMAX": "#0C447C",
    "Musical": "#ED93B1",       "Mystery": "#534AB7",
    "Romance": "#D4537E",       "Sci-Fi": "#1D9E75",
    "Thriller": "#993C1D",      "War": "#444441",
    "Western": "#854F0B",       "(no genres listed)": "#B4B2A9",
}

# Дружественные подписи моделей.
MODEL_LABELS = {
    "global_mean": "Global Mean (baseline)",
    "popularity":  "Popularity (Bayesian average)",
    "svd":         "SVD (Surprise)",
    "knn":         "KNN item-based (Surprise)",
    "lightgbm":    "LightGBM (feature-based)",
    "als":         "ALS (implicit)",
    "ncf":         "NCF NeuMF (Keras)",
    "ensemble":    "Ансамбль ALS + SVD + NCF",
}

# Финальная модель (по best_model_decision.json - если он есть; иначе fallback).
DEFAULT_BEST_MODEL = "ensemble"

# ---------------------------------------------------------------------------
# Конфигурация страницы и базовый CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon=None,                    # без иконки-эмодзи, как и просили
    layout="wide",
    initial_sidebar_state="expanded",
)

# Минимальный кастомный CSS: аккуратные карточки, мягкие тени, отступы.
# Сознательно не делаем тёмную тему - пусть применяется системная.
_CUSTOM_CSS = """
<style>
    /* Уменьшаем верхний паддинг главного контейнера */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* Карточка - универсальный контейнер */
    .recsys-card {
        border: 1px solid rgba(120, 120, 120, 0.20);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        background: rgba(250, 250, 250, 0.55);
        margin-bottom: 0.75rem;
    }
    .recsys-card h4 { margin: 0 0 .35rem 0; font-weight: 500; }
    .recsys-card .meta { color: #5F5E5A; font-size: .92rem; }

    /* Бейдж жанра */
    .genre-badge {
        display: inline-block;
        padding: 2px 9px;
        margin: 0 4px 4px 0;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 500;
        color: #ffffff;
        white-space: nowrap;
    }

    /* Бейдж "лучшая модель" */
    .best-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        background: #0F6E56;
        color: #ffffff;
        font-size: .78rem;
        font-weight: 500;
        margin-left: 6px;
    }

    /* Метрика-крупно для главной */
    .big-metric-value {
        font-size: 2rem;
        font-weight: 500;
        color: #185FA5;
        line-height: 1.0;
    }
    .big-metric-label {
        font-size: .9rem;
        color: #5F5E5A;
        margin-top: .25rem;
    }

    /* Звёздочный рейтинг текстом */
    .star-row { color: #BA7517; letter-spacing: 1px; font-size: 1rem; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Классы моделей, сериализованных через joblib
# ---------------------------------------------------------------------------
# joblib.load восстанавливает объект по имени класса, поэтому класс
# PopularityRecommender и GlobalMeanModel должны быть доступны
# в текущем модуле, иначе восстановление падает с AttributeError.
# Дублируем определения из ноутбука 04 - этого достаточно для unpickle.


class GlobalMeanModel:
    """Baseline: предсказывает глобальное среднее train-рейтинга."""

    def __init__(self):
        self.global_mean_ = None

    def fit(self, train_df, rating_col="rating"):
        self.global_mean_ = float(train_df[rating_col].mean())
        return self

    def predict(self, user_ids, movie_ids):
        return np.full(len(user_ids), self.global_mean_, dtype=np.float64)


class PopularityRecommender:
    """Top-N рекомендатель на основе сглаженной популярности (Bayesian
    average). Класс определён здесь для совместимости с joblib.load."""

    def __init__(self, m: float = 10.0):
        self.m = m
        self.global_mean_ = None
        self.scores_ = None         # pd.Series, index=movieId, value=score
        self.user_seen_ = None      # dict[int, set]

    def fit(self, train_df, user_col="userId", item_col="movieId",
            rating_col="rating"):
        self.global_mean_ = float(train_df[rating_col].mean())
        agg = train_df.groupby(item_col)[rating_col].agg(["count", "mean"])
        n = agg["count"]
        mu = agg["mean"]
        C = self.global_mean_
        self.scores_ = (n / (n + self.m)) * mu + (self.m / (n + self.m)) * C
        self.scores_ = self.scores_.sort_values(ascending=False)
        self.user_seen_ = (
            train_df.groupby(user_col)[item_col].apply(set).to_dict()
        )
        return self

    def recommend(self, user_ids, k: int = 10, exclude_seen: bool = True):
        if self.scores_ is None:
            raise RuntimeError("Модель не обучена")
        ranked = self.scores_.index.values
        result = {}
        for u in user_ids:
            if exclude_seen:
                seen = self.user_seen_.get(u, set())
                rec = [m for m in ranked if m not in seen][:k]
            else:
                rec = list(ranked[:k])
            result[u] = rec
        return result


# ---------------------------------------------------------------------------
# Жёсткий справочник времени обучения каждой модели (берётся из ноутбуков).
# В *_metrics.json эти значения не сохранялись, поэтому держим их рядом.
# ---------------------------------------------------------------------------
TRAIN_TIME_SECONDS = {
    "popularity": 240.0,    # 4 мин - grid search по m
    "svd":        300.0,    # 5 мин - Optuna 30 trials + финальный фит
    "knn":        1080.0,   # 18 мин - Optuna 20 trials (KNN дорогой)
    "lightgbm":   420.0,    # 7 мин  - Optuna 50 trials
    "als":        180.0,    # 3 мин  - Optuna 50 trials
    "ncf":        1754.1,   # из ноутбука 09: 30 trials на CPU
    "ensemble":   600.0,    # 10 мин - 20 trials по весам
}


# ---------------------------------------------------------------------------
# Канонический набор top-N метрик и алиасы.
# В JSON-файлах разных моделей ключи писались по-разному
# (hit_rate@10 vs hitrate@10, метрики иногда вложены в test).
# Нормализатор разворачивает эту структуру.
# ---------------------------------------------------------------------------
_METRIC_ALIASES = {
    "hit_rate@5":  "hitrate@5",
    "hit_rate@10": "hitrate@10",
    "hit_rate@20": "hitrate@20",
    "ndcg@10":     "ndcg@10",
    "precision@10": "precision@10",
    "recall@10":   "recall@10",
    "coverage@20": "coverage@20",
    "rmse":        "rmse",
    "mae":         "mae",
}


def _normalize_metric_keys(d: dict) -> dict:
    """Приводит ключи метрик к единому виду (hit_rate@10 -> hitrate@10).

    Дополнительно склеивает несколько распространённых вариантов написания.
    """
    out = {}
    for k, v in d.items():
        if not isinstance(v, (int, float)):
            continue
        nk = _METRIC_ALIASES.get(k, k.replace("hit_rate", "hitrate"))
        out[nk] = v
    return out


def _flatten_metrics_json(raw: dict, model_key: str) -> dict:
    """Разворачивает структуру *_metrics.json в плоский словарь
    с каноническими ключами метрик на test.

    Поддерживаем несколько схем:
      A) raw = {"test": {"ndcg@10": ..., ...}, "val": {...}} -> берём test
      B) raw = {"ndcg@10": ..., ...}                          -> берём как есть
      C) raw = {"popularity": {"test": {...}}, "global_mean": ...}
         (формат из ноутбука 04) -> берём popularity.test
      D) raw = {"final": {"test_topn": {...}, "test_rating": {...}}}
         (так пишут ноутбуки 5-10: SVD, KNN, LightGBM, ALS, NCF, ансамбль)
         -> склеиваем test_topn + test_rating в плоский test
    """
    if not raw:
        return {}

    # Вариант C: popularity_metrics.json содержит и global_mean, и popularity.
    if model_key == "popularity":
        pop_block = raw.get("popularity") or raw
        if isinstance(pop_block, dict):
            test_block = pop_block.get("test")
            if isinstance(test_block, dict):
                return _normalize_metric_keys(test_block)
            return _normalize_metric_keys(pop_block)
        return {}

    if model_key == "global_mean":
        gm_block = raw.get("global_mean") or raw
        if isinstance(gm_block, dict):
            test_block = gm_block.get("test")
            if isinstance(test_block, dict):
                return _normalize_metric_keys(test_block)
            return _normalize_metric_keys(gm_block)
        return {}

    # Варианты A / B / D - универсально.
    if "test" in raw and isinstance(raw["test"], dict):
        # A: уже плоский test на верхнем уровне.
        flat = _normalize_metric_keys(raw["test"])
    elif "final" in raw and isinstance(raw["final"], dict):
        # D: ноутбуки 5-10 кладут метрики в final.test_topn и final.test_rating.
        final_block = raw["final"]
        merged: dict = {}
        for sub_key in ("test_rating", "test_topn"):
            sub = final_block.get(sub_key)
            if isinstance(sub, dict):
                merged.update(sub)
        flat = _normalize_metric_keys(merged) if merged \
               else _normalize_metric_keys(raw)
    else:
        # B: плоский словарь с метриками на верхнем уровне.
        flat = _normalize_metric_keys(raw)

    # RMSE/MAE иногда лежат в отдельном rating_block, иногда сразу в test.
    # Если их нет в плоском результате - попробуем достать из верхнего уровня.
    for rkey in ("rmse", "mae"):
        if rkey not in flat and isinstance(raw.get(rkey), (int, float)):
            flat[rkey] = raw[rkey]
    return flat


# ---------------------------------------------------------------------------
# Утилиты безопасной загрузки и форматирования
# ---------------------------------------------------------------------------

def _safe_read_json(path: Path) -> dict:
    """Читает JSON; при отсутствии файла или ошибке - пустой dict."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _safe_read_pickle(path: Path):
    """Читает pickle через joblib. В случае ошибки - None (плюс показываем
    предупреждение в UI). joblib умеет совместимо читать numpy/scipy-объекты."""
    import joblib
    try:
        return joblib.load(path)
    except Exception as exc:
        st.warning(
            f"Не удалось загрузить {path.name}: {type(exc).__name__}. "
            "Соответствующая модель будет недоступна."
        )
        return None


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    """Читает parquet; при ошибке возвращает None."""
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _safe_read_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    """Читает csv; при ошибке возвращает None."""
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def fmt_int(x: Any) -> str:
    """Форматирует целое число с пробелами как разделителями тысяч."""
    try:
        return f"{int(x):,}".replace(",", " ")
    except Exception:
        return "-"


def fmt_float(x: Any, digits: int = 3) -> str:
    """Форматирует число с фиксированным числом знаков; '-' для NaN/None."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "-"


def parse_year_from_title(title: str) -> Optional[int]:
    """Извлекает год из строки 'Toy Story (1995)' -> 1995."""
    if not isinstance(title, str):
        return None
    s = title.strip()
    if len(s) >= 6 and s.endswith(")") and s[-6] == "(":
        chunk = s[-5:-1]
        if chunk.isdigit():
            return int(chunk)
    return None


def strip_year(title: str) -> str:
    """Убирает '(1995)' из хвоста."""
    if not isinstance(title, str):
        return ""
    s = title.strip()
    if len(s) >= 6 and s.endswith(")") and s[-6] == "(" and s[-5:-1].isdigit():
        return s[:-7].strip()
    return s


def genres_to_list(g: Any) -> list[str]:
    """Преобразует строку жанров 'Action|Drama' в список."""
    if isinstance(g, list):
        return [x for x in g if x]
    if not isinstance(g, str):
        return []
    return [x for x in g.split("|") if x]


def render_genre_badges(genres: list[str]) -> str:
    """Возвращает HTML с цветными бейджами жанров."""
    out = []
    for g in genres:
        color = GENRE_COLORS.get(g, "#5F5E5A")
        out.append(
            f'<span class="genre-badge" style="background:{color}">{g}</span>'
        )
    return "".join(out)


def render_stars(rating: float) -> str:
    """Текстовый звёздочный рейтинг по округлению до 0.5."""
    if rating is None or (isinstance(rating, float) and np.isnan(rating)):
        return ""
    rating = max(0.0, min(5.0, float(rating)))
    full = int(rating)
    half = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    # Используем простые символы, не эмодзи. Юникод-звёздочки допустимы.
    return "★" * full + ("½" if half else "") + "☆" * empty



# ---------------------------------------------------------------------------
# Кешируемые загрузчики данных и моделей
# ---------------------------------------------------------------------------
# Стратегия:
#   * @st.cache_data       - для всего, что можно безопасно копировать
#                            (DataFrame, dict, numpy-массивы).
#   * @st.cache_resource   - для тяжёлых объектов с состоянием
#                            (модели Surprise, ALS, Keras, LightGBM).
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_ratings() -> Optional[pd.DataFrame]:
    df = _safe_read_csv(DATA_RAW / "ratings.csv")
    return df


@st.cache_data(show_spinner=False)
def load_movies_raw() -> Optional[pd.DataFrame]:
    df = _safe_read_csv(DATA_RAW / "movies.csv")
    if df is not None:
        df["year"] = df["title"].apply(parse_year_from_title)
        df["clean_title"] = df["title"].apply(strip_year)
        df["genres_list"] = df["genres"].apply(genres_to_list)
    return df


@st.cache_data(show_spinner=False)
def load_movies_enriched() -> Optional[pd.DataFrame]:
    """Обогащённые метаданные. Если файла нет - fallback на movies.csv."""
    df = _safe_read_parquet(DATA_PROC / "movies_enriched.parquet")
    if df is None:
        df = load_movies_raw()
        if df is None:
            return None
    if df is not None:
        if "year" not in df.columns:
            df["year"] = df["title"].apply(parse_year_from_title)
        if "clean_title" not in df.columns:
            df["clean_title"] = df["title"].apply(strip_year)
        if "genres_list" not in df.columns and "genres" in df.columns:
            df["genres_list"] = df["genres"].apply(genres_to_list)
    return df


@st.cache_data(show_spinner=False)
def load_tags() -> Optional[pd.DataFrame]:
    return _safe_read_csv(DATA_RAW / "tags.csv")


@st.cache_data(show_spinner=False)
def load_links() -> Optional[pd.DataFrame]:
    return _safe_read_csv(DATA_RAW / "links.csv")


@st.cache_data(show_spinner=False)
def load_splits() -> dict[str, Optional[pd.DataFrame]]:
    return {
        "train":      _safe_read_parquet(DATA_PROC / "train.parquet"),
        "val":        _safe_read_parquet(DATA_PROC / "val.parquet"),
        "test":       _safe_read_parquet(DATA_PROC / "test.parquet"),
        "cold_start": _safe_read_parquet(DATA_PROC / "cold_start_eval.parquet"),
    }


@st.cache_data(show_spinner=False)
def load_movie_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Агрегирует средний рейтинг и число оценок по movieId."""
    stats = ratings.groupby("movieId", as_index=False).agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
    )
    return stats


@st.cache_data(show_spinner=False)
def load_id_maps() -> dict:
    """ID-маппинги нужны для перехода raw_id <-> internal_idx во всех моделях."""
    return {
        "user_id_map":     _safe_read_pickle(MODELS_DIR / "user_id_map.pkl"),
        "movie_id_map":    _safe_read_pickle(MODELS_DIR / "movie_id_map.pkl"),
        "inv_user_id_map": _safe_read_pickle(MODELS_DIR / "inv_user_id_map.pkl"),
        "inv_movie_id_map":_safe_read_pickle(MODELS_DIR / "inv_movie_id_map.pkl"),
    }


@st.cache_data(show_spinner=False)
def load_all_metrics() -> dict[str, dict]:
    """Читает *_metrics.json для каждой модели и нормализует ключи.

    После нормализации каждый внутренний словарь содержит метрики на test
    с каноническими именами (ndcg@10, precision@10, hitrate@10, ...).
    Дополнительно подмешивается train_time_seconds из справочника
    TRAIN_TIME_SECONDS (в JSON оно не сохранялось).
    """
    out = {}
    metric_files = {
        "popularity": "popularity_metrics.json",
        "svd":        "svd_metrics.json",
        "knn":        "knn_metrics.json",
        "lightgbm":   "lightgbm_metrics.json",
        "als":        "als_metrics.json",
        "ncf":        "ncf_metrics.json",
        "ensemble":   "ensemble_metrics.json",
    }
    for key, fname in metric_files.items():
        raw = _safe_read_json(MODELS_DIR / fname)
        flat = _flatten_metrics_json(raw, key)
        # Время обучения подкладываем из справочника.
        if key in TRAIN_TIME_SECONDS:
            flat.setdefault("train_time_seconds", TRAIN_TIME_SECONDS[key])
        out[key] = flat
    return out


@st.cache_data(show_spinner=False)
def load_all_params() -> dict[str, dict]:
    out = {}
    params_files = {
        "popularity": "popularity_params.json",
        "svd":        "svd_params.json",
        "knn":        "knn_params.json",
        "lightgbm":   "lightgbm_params.json",
        "als":        "als_params.json",
        "ncf":        "ncf_params.json",
        "ensemble":   "ensemble_params.json",
    }
    for key, fname in params_files.items():
        out[key] = _safe_read_json(MODELS_DIR / fname)
    return out


@st.cache_data(show_spinner=False)
def load_metrics_summary() -> Optional[pd.DataFrame]:
    return _safe_read_parquet(MODELS_DIR / "metrics_summary.parquet")


@st.cache_data(show_spinner=False)
def load_best_model_decision() -> dict:
    return _safe_read_json(MODELS_DIR / "best_model_decision.json")


# --- Сами модели (cache_resource - не сериализуем) -------------------------

@st.cache_resource(show_spinner=False)
def load_popularity_model():
    return _safe_read_pickle(MODELS_DIR / "popularity_model.pkl")


@st.cache_resource(show_spinner=False)
def load_global_mean_model():
    return _safe_read_pickle(MODELS_DIR / "global_mean_model.pkl")


@st.cache_resource(show_spinner=False)
def load_svd_model():
    return _safe_read_pickle(MODELS_DIR / "svd_model.pkl")


@st.cache_resource(show_spinner=False)
def load_knn_model():
    return _safe_read_pickle(MODELS_DIR / "knn_model.pkl")


@st.cache_resource(show_spinner=False)
def load_lightgbm_model():
    # Сначала пробуем общий lightgbm_model.pkl; если его нет - regressor.
    for fname in ("lightgbm_model.pkl", "lightgbm_regressor.pkl",
                  "lightgbm_ranker.pkl"):
        p = MODELS_DIR / fname
        if p.exists():
            return _safe_read_pickle(p)
    return None


@st.cache_resource(show_spinner=False)
def load_als_model():
    return _safe_read_pickle(MODELS_DIR / "als_model.pkl")


@st.cache_resource(show_spinner=False)
def load_ncf_model():
    """NCF - Keras-модель. Загружаем с compile=False, чтобы избежать
    ошибок из-за отсутствующих кастомных метрик."""
    try:
        import tensorflow as tf  # отложенный импорт - тяжёлая зависимость
        p = MODELS_DIR / "ncf_model.keras"
        if not p.exists():
            return None
        # safe_mode=False позволяет загружать модели с lambda-слоями.
        model = tf.keras.models.load_model(str(p), compile=False)
        return model
    except Exception as exc:
        st.warning(
            "Не удалось загрузить NCF (ncf_model.keras): "
            f"{type(exc).__name__}. Рекомендации NCF будут недоступны."
        )
        return None


@st.cache_resource(show_spinner=False)
def load_user_item_sparse():
    """user_item_train.npz - sparse CSR (n_users x n_items) с бинарной/
    вещественной матрицей предпочтений. Нужен для ALS-инференса."""
    try:
        from scipy import sparse
        p = DATA_PROC / "user_item_train.npz"
        if not p.exists():
            return None
        return sparse.load_npz(str(p))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_preprocessing_meta() -> dict:
    return _safe_read_json(DATA_PROC / "preprocessing_meta.json")



# ---------------------------------------------------------------------------
# Функции рекомендаций (инференс)
# ---------------------------------------------------------------------------
# Соглашения:
#   * raw_id - идентификатор как в CSV (userId, movieId).
#   * idx    - внутренний индекс (как в train sparse / в model.trainset).
#   * Все функции возвращают list[tuple[movieId, score]], отсортированный
#     по убыванию score, длиной top_n, БЕЗ фильмов из exclude_movie_ids.
# ---------------------------------------------------------------------------

def _topk_from_scores(
    scores: np.ndarray,
    inv_movie_id_map: dict,
    exclude_idx: set[int],
    top_n: int,
) -> list[tuple[int, float]]:
    """Берёт массив скоров (по внутренним idx фильмов), исключает уже
    оценённые и возвращает топ-N (raw_movieId, score)."""
    # Маскируем исключённые фильмы.
    if exclude_idx:
        ex = np.fromiter(exclude_idx, dtype=int)
        ex = ex[(ex >= 0) & (ex < scores.shape[0])]
        scores = scores.copy()
        scores[ex] = -np.inf
    # Берём top_n индексов.
    k = min(top_n, scores.shape[0])
    if k == 0:
        return []
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    out: list[tuple[int, float]] = []
    for idx in top_idx:
        movie_raw = inv_movie_id_map.get(int(idx))
        if movie_raw is None:
            continue
        out.append((int(movie_raw), float(scores[int(idx)])))
    return out


def recommend_popularity(
    movie_stats: pd.DataFrame,
    exclude_movie_ids: set[int],
    top_n: int,
    m: int = 200,
) -> list[tuple[int, float]]:
    """Bayesian-average popularity baseline. Не зависит от пользователя -
    подходит как fallback для cold-start."""
    if movie_stats is None or movie_stats.empty:
        return []
    C = float(movie_stats["avg_rating"].mean())
    v = movie_stats["rating_count"].values
    R = movie_stats["avg_rating"].values
    bayes = (v / (v + m)) * R + (m / (v + m)) * C
    df = movie_stats.assign(score=bayes)
    df = df[~df["movieId"].isin(exclude_movie_ids)]
    df = df.sort_values("score", ascending=False).head(top_n)
    return list(zip(df["movieId"].astype(int).tolist(),
                    df["score"].astype(float).tolist()))


def recommend_als_existing_user(
    als_model,
    user_item_sparse,
    id_maps: dict,
    raw_user_id: int,
    exclude_movie_ids: set[int],
    top_n: int,
) -> list[tuple[int, float]]:
    """ALS-рекомендации для существующего пользователя через user_factors."""
    if als_model is None:
        return []
    inv_movie_id_map = id_maps["inv_movie_id_map"]
    user_id_map = id_maps["user_id_map"]
    u_idx = user_id_map.get(raw_user_id)
    if u_idx is None:
        return []
    # user_factors / item_factors - это numpy-массивы у implicit.als.
    try:
        u_vec = als_model.user_factors[u_idx]
        scores = als_model.item_factors @ u_vec
    except Exception:
        return []
    # Исключаем уже оценённые.
    exclude_idx = set()
    movie_id_map = id_maps["movie_id_map"]
    for mid in exclude_movie_ids:
        if mid in movie_id_map:
            exclude_idx.add(movie_id_map[mid])
    return _topk_from_scores(scores, inv_movie_id_map, exclude_idx, top_n)


def recommend_als_new_user(
    als_model,
    id_maps: dict,
    user_ratings: dict[int, float],
    top_n: int,
) -> list[tuple[int, float]]:
    """ALS-fold-in для нового пользователя.

    Реализация:
        1. Берём item-факторы по фильмам из истории.
        2. Веса w_i = max(rating_i - 3.0, 0)  - только "понравившиеся".
           (для implicit ALS отрицательная обратная связь не имеет смысла -
           она моделируется отсутствием взаимодействия).
        3. user_vector = (Y_subset.T @ w) / (sum(w) + eps).
        4. scores = item_factors @ user_vector.
    """
    if als_model is None or not user_ratings:
        return []
    movie_id_map = id_maps["movie_id_map"]
    inv_movie_id_map = id_maps["inv_movie_id_map"]
    idxs, weights = [], []
    for mid, r in user_ratings.items():
        if mid in movie_id_map:
            w = max(float(r) - 3.0, 0.0)
            if w > 0:
                idxs.append(movie_id_map[mid])
                weights.append(w)
    if not idxs:
        # Все оценки <= 3.0 - не на чем строить позитивный user-вектор.
        return []
    try:
        Y_sub = als_model.item_factors[np.array(idxs)]
        w = np.array(weights, dtype=np.float32)
        u_vec = (Y_sub.T @ w) / (w.sum() + 1e-9)
        scores = als_model.item_factors @ u_vec
    except Exception:
        return []
    exclude_idx = {movie_id_map[m] for m in user_ratings if m in movie_id_map}
    return _topk_from_scores(scores, inv_movie_id_map, exclude_idx, top_n)


def recommend_svd_existing_user(
    svd_model,
    id_maps: dict,
    raw_user_id: int,
    movie_stats: pd.DataFrame,
    exclude_movie_ids: set[int],
    top_n: int,
) -> list[tuple[int, float]]:
    """SVD из surprise: векторизованный расчёт скоров по всем фильмам.

    Вместо медленного цикла с svd_model.predict(uid, iid) умножаем
    матрицу qi на вектор pu пользователя - это математически эквивалентно
    предсказанию SVD: `r_hat = mu + bu + bi + pu . qi`.
    """
    if svd_model is None:
        return []
    try:
        ts = svd_model.trainset
        u_inner = ts.to_inner_uid(raw_user_id)
    except (ValueError, AttributeError):
        # Пользователя нет в trainset - возвращаем пусто
        # (вызывающий код может сделать fallback на popularity).
        return []

    try:
        mu = ts.global_mean
        pu = svd_model.pu[u_inner]
        bu = svd_model.bu[u_inner]
        # Векторизованно: shape (n_items,)
        raw_scores = mu + bu + svd_model.bi + svd_model.qi @ pu
    except Exception:
        return []

    # inner_iid -> raw movieId
    pairs = []
    for inner_iid, sc in enumerate(raw_scores):
        try:
            raw_mid = int(ts.to_raw_iid(inner_iid))
        except Exception:
            continue
        if raw_mid in exclude_movie_ids:
            continue
        pairs.append((raw_mid, float(sc)))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def recommend_svd_new_user(
    svd_model,
    user_ratings: dict[int, float],
    movie_stats: pd.DataFrame,
    top_n: int,
) -> list[tuple[int, float]]:
    """Приближённый fold-in для SVD из surprise.

    Surprise SVD хранит:
        model.trainset.global_mean - mu
        model.bu[u_inner], model.bi[i_inner] - смещения
        model.pu[u_inner], model.qi[i_inner] - факторы

    Для нового пользователя мы не имеем bu и pu. Аппроксимация:
        - bu_new = mean(user_ratings) - mu       (личное смещение)
        - pu_new = sum_i (r_i - mu - bi_i) * qi_i / (||qi||^2 + lambda)
          - то есть OLS-фит факторов из имеющихся оценок.
    Затем pred(u, i) = mu + bu_new + bi_i + pu_new . qi_i.
    """
    if svd_model is None or not user_ratings:
        return []
    try:
        ts = svd_model.trainset
        mu = ts.global_mean
        # raw_iid -> inner_iid для каждого оценённого фильма.
        inner_pairs = []
        for mid, r in user_ratings.items():
            try:
                iid_inner = ts.to_inner_iid(mid)
                inner_pairs.append((iid_inner, float(r)))
            except ValueError:
                continue
        if not inner_pairs:
            return []
        qi = svd_model.qi
        bi = svd_model.bi
        # Личное смещение пользователя.
        bu_new = float(np.mean([r - mu for _, r in inner_pairs]))
        # Residuals для подгонки факторов.
        Q, y = [], []
        for iid_inner, r in inner_pairs:
            Q.append(qi[iid_inner])
            y.append(r - mu - bu_new - bi[iid_inner])
        Q = np.asarray(Q, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        # Ridge: pu = (Q.T Q + lam I)^-1 Q.T y, lam=0.05 (мягкая регуляризация).
        lam = 0.05
        A = Q.T @ Q + lam * np.eye(qi.shape[1], dtype=np.float32)
        b = Q.T @ y
        pu_new = np.linalg.solve(A, b)
    except Exception:
        return []

    # Скоры по всем фильмам в трейнсете.
    all_scores = mu + bu_new + bi + qi @ pu_new
    # Преобразуем inner_iid -> raw movieId и фильтруем оценённые.
    exclude_inner = {iid for iid, _ in inner_pairs}
    pairs = []
    for inner_iid, sc in enumerate(all_scores):
        if inner_iid in exclude_inner:
            continue
        try:
            raw_mid = svd_model.trainset.to_raw_iid(inner_iid)
            pairs.append((int(raw_mid), float(sc)))
        except Exception:
            continue
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def recommend_ncf_existing_user(
    ncf_model,
    id_maps: dict,
    raw_user_id: int,
    movie_stats: pd.DataFrame,
    exclude_movie_ids: set[int],
    top_n: int,
) -> list[tuple[int, float]]:
    """NCF (NeuMF) для существующего пользователя.

    Делаем один batch-predict по всем кандидатам. Кешируем результат
    в session_state по userId, чтобы не пересчитывать при изменении top_n.
    """
    if ncf_model is None:
        return []
    user_id_map = id_maps["user_id_map"]
    movie_id_map = id_maps["movie_id_map"]
    u_idx = user_id_map.get(raw_user_id)
    if u_idx is None:
        return []

    cache_key = f"_ncf_scores_user_{raw_user_id}"
    if cache_key in st.session_state:
        scores = st.session_state[cache_key]
    else:
        # Кандидаты - фильмы, известные модели (через movie_id_map).
        candidate_raw = list(movie_id_map.keys())
        candidate_idx = np.array([movie_id_map[m] for m in candidate_raw],
                                 dtype=np.int64)
        user_arr = np.full_like(candidate_idx, u_idx, dtype=np.int64)
        try:
            pred = ncf_model.predict(
                [user_arr, candidate_idx], batch_size=4096, verbose=0
            ).ravel()
        except Exception:
            # Некоторые NCF-архитектуры ожидают единый вход или другой порядок.
            try:
                pred = ncf_model.predict(
                    {"user_input": user_arr, "item_input": candidate_idx},
                    batch_size=4096, verbose=0
                ).ravel()
            except Exception:
                return []
        scores = dict(zip(candidate_raw, pred.tolist()))
        st.session_state[cache_key] = scores

    pairs = [(int(m), float(s)) for m, s in scores.items()
             if m not in exclude_movie_ids]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def recommend_ensemble_existing_user(
    raw_user_id: int,
    als_model, svd_model, ncf_model,
    user_item_sparse,
    id_maps: dict,
    movie_stats: pd.DataFrame,
    exclude_movie_ids: set[int],
    top_n: int,
    weights: dict[str, float],
) -> list[tuple[int, float]]:
    """Линейная комбинация рангов/скоров от ALS, SVD и NCF.

    Чтобы корректно складывать модели разной природы (implicit ALS даёт
    скоры в одной шкале, SVD - предсказанный рейтинг 0.5..5, NCF - сигмоид
    вероятность), переводим каждый набор скоров в RANK-нормализованные
    значения [0, 1] и взвешиваем."""
    movie_id_map = id_maps["movie_id_map"]
    inv_movie_id_map = id_maps["inv_movie_id_map"]
    user_id_map = id_maps["user_id_map"]
    u_idx = user_id_map.get(raw_user_id)
    if u_idx is None:
        return []

    n_items = len(inv_movie_id_map)
    agg = np.zeros(n_items, dtype=np.float32)
    total_w = 0.0

    # ALS
    if als_model is not None and weights.get("als", 0) > 0:
        try:
            u_vec = als_model.user_factors[u_idx]
            sc = als_model.item_factors @ u_vec
            agg += weights["als"] * _to_rank01(sc)
            total_w += weights["als"]
        except Exception:
            pass

    # SVD
    if svd_model is not None and weights.get("svd", 0) > 0:
        try:
            ts = svd_model.trainset
            try:
                u_inner = ts.to_inner_uid(raw_user_id)
                pu = svd_model.pu[u_inner]
                bu = svd_model.bu[u_inner]
            except ValueError:
                pu = np.zeros(svd_model.qi.shape[1], dtype=np.float32)
                bu = 0.0
            raw_scores = ts.global_mean + bu + svd_model.bi + svd_model.qi @ pu
            # Преобразуем inner_iid -> idx через inv_movie_id_map.
            sc_full = np.full(n_items, -np.inf, dtype=np.float32)
            for inner_iid, sc in enumerate(raw_scores):
                raw_mid = ts.to_raw_iid(inner_iid)
                idx = movie_id_map.get(int(raw_mid))
                if idx is not None:
                    sc_full[idx] = sc
            agg += weights["svd"] * _to_rank01(sc_full)
            total_w += weights["svd"]
        except Exception:
            pass

    # NCF
    if ncf_model is not None and weights.get("ncf", 0) > 0:
        ncf_pairs = recommend_ncf_existing_user(
            ncf_model, id_maps, raw_user_id, movie_stats,
            exclude_movie_ids=set(),
            top_n=n_items,
        )
        if ncf_pairs:
            sc_full = np.full(n_items, -np.inf, dtype=np.float32)
            for mid, sc in ncf_pairs:
                idx = movie_id_map.get(mid)
                if idx is not None:
                    sc_full[idx] = sc
            agg += weights["ncf"] * _to_rank01(sc_full)
            total_w += weights["ncf"]

    if total_w == 0:
        return []
    agg = agg / total_w

    exclude_idx = {movie_id_map[m] for m in exclude_movie_ids
                   if m in movie_id_map}
    return _topk_from_scores(agg, inv_movie_id_map, exclude_idx, top_n)


def recommend_ensemble_new_user(
    user_ratings: dict[int, float],
    als_model, svd_model,
    id_maps: dict,
    top_n: int,
    weights: dict[str, float],
) -> list[tuple[int, float]]:
    """Ансамбль для нового пользователя - только ALS + SVD (NCF без user
    эмбеддинга в cold-start не используется, как мы и решили)."""
    movie_id_map = id_maps["movie_id_map"]
    inv_movie_id_map = id_maps["inv_movie_id_map"]
    n_items = len(inv_movie_id_map)
    agg = np.zeros(n_items, dtype=np.float32)
    total_w = 0.0

    # ALS fold-in.
    if als_model is not None and weights.get("als", 0) > 0:
        idxs, w_arr = [], []
        for mid, r in user_ratings.items():
            if mid in movie_id_map:
                w = max(float(r) - 3.0, 0.0)
                if w > 0:
                    idxs.append(movie_id_map[mid])
                    w_arr.append(w)
        if idxs:
            try:
                Y_sub = als_model.item_factors[np.array(idxs)]
                w = np.array(w_arr, dtype=np.float32)
                u_vec = (Y_sub.T @ w) / (w.sum() + 1e-9)
                sc = als_model.item_factors @ u_vec
                agg += weights["als"] * _to_rank01(sc)
                total_w += weights["als"]
            except Exception:
                pass

    # SVD fold-in.
    if svd_model is not None and weights.get("svd", 0) > 0:
        try:
            ts = svd_model.trainset
            mu = ts.global_mean
            inner_pairs = []
            for mid, r in user_ratings.items():
                try:
                    iid_inner = ts.to_inner_iid(mid)
                    inner_pairs.append((iid_inner, float(r)))
                except ValueError:
                    continue
            if inner_pairs:
                qi = svd_model.qi
                bi = svd_model.bi
                bu_new = float(np.mean([r - mu for _, r in inner_pairs]))
                Q, y = [], []
                for iid_inner, r in inner_pairs:
                    Q.append(qi[iid_inner])
                    y.append(r - mu - bu_new - bi[iid_inner])
                Q = np.asarray(Q, dtype=np.float32)
                y = np.asarray(y, dtype=np.float32)
                A = Q.T @ Q + 0.05 * np.eye(qi.shape[1], dtype=np.float32)
                pu_new = np.linalg.solve(A, Q.T @ y)
                raw_scores = mu + bu_new + bi + qi @ pu_new
                sc_full = np.full(n_items, -np.inf, dtype=np.float32)
                for inner_iid, sc in enumerate(raw_scores):
                    raw_mid = ts.to_raw_iid(inner_iid)
                    idx = movie_id_map.get(int(raw_mid))
                    if idx is not None:
                        sc_full[idx] = sc
                agg += weights["svd"] * _to_rank01(sc_full)
                total_w += weights["svd"]
        except Exception:
            pass

    if total_w == 0:
        return []
    agg = agg / total_w
    exclude_idx = {movie_id_map[m] for m in user_ratings
                   if m in movie_id_map}
    return _topk_from_scores(agg, inv_movie_id_map, exclude_idx, top_n)


def _to_rank01(scores: np.ndarray) -> np.ndarray:
    """Преобразует массив скоров в нормированные ранги [0, 1].

    -inf -> 0 (фильмы, для которых модель ничего не предсказала).
    """
    n = scores.shape[0]
    out = np.zeros(n, dtype=np.float32)
    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return out
    # Ранг по убыванию: лучший -> 1.0, худший -> ~0.0.
    finite_scores = scores[finite_mask]
    order = np.argsort(-finite_scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(1.0, 0.0, num=order.shape[0], dtype=np.float32)
    out[finite_mask] = ranks
    return out



# ---------------------------------------------------------------------------
# Прелоад всех артефактов на старте (один раз через st.status)
# ---------------------------------------------------------------------------

def warm_up_all_artifacts() -> dict:
    """Прогревает кеш всех функций загрузки.

    Возвращает словарь со ссылками на основные DataFrame-ы и маппинги,
    плюс bool-флаги о том, какие модели реально доступны (нужно для UI -
    серый чекмарк в боковой панели).
    """
    state: dict = {}
    steps = [
        ("Загружаю ratings.csv",            lambda: load_ratings()),
        ("Загружаю movies.csv",             lambda: load_movies_raw()),
        ("Загружаю обогащённые метаданные", lambda: load_movies_enriched()),
        ("Загружаю tags.csv",               lambda: load_tags()),
        ("Считаю статистику по фильмам",    lambda: load_movie_stats(load_ratings())),
        ("Загружаю train/val/test сплиты",  lambda: load_splits()),
        ("Загружаю ID-маппинги",            lambda: load_id_maps()),
        ("Загружаю метрики всех моделей",   lambda: load_all_metrics()),
        ("Загружаю гиперпараметры",         lambda: load_all_params()),
        ("Загружаю сводку метрик",          lambda: load_metrics_summary()),
        ("Загружаю popularity-модель",      lambda: load_popularity_model()),
        ("Загружаю SVD (Surprise)",         lambda: load_svd_model()),
        ("Загружаю KNN (Surprise)",         lambda: load_knn_model()),
        ("Загружаю LightGBM",               lambda: load_lightgbm_model()),
        ("Загружаю ALS (implicit)",         lambda: load_als_model()),
        ("Загружаю sparse user_item",       lambda: load_user_item_sparse()),
        ("Загружаю NCF (Keras)",            lambda: load_ncf_model()),
    ]
    placeholder = st.empty()
    with placeholder.status(
        "Прогреваю кеш моделей и данных. Это разовая операция, "
        "при последующих переходах между вкладками всё будет мгновенно.",
        expanded=True,
    ) as status:
        progress = st.progress(0.0)
        for i, (label, fn) in enumerate(steps, start=1):
            t0 = time.time()
            st.write(f"  - {label}...")
            try:
                fn()
                dt = time.time() - t0
                st.write(f"    готово за {dt:.2f} с")
            except Exception as exc:
                st.write(f"    пропущено ({type(exc).__name__})")
            progress.progress(i / len(steps))
        status.update(
            label="Все артефакты загружены. Приложение готово к работе.",
            state="complete",
            expanded=False,
        )
    # Освобождаем placeholder, чтобы шапка статуса не висела.
    placeholder.empty()

    # Собираем сводное состояние для UI.
    state["ratings"]          = load_ratings()
    state["movies"]           = load_movies_enriched()
    state["movie_stats"]      = load_movie_stats(load_ratings())
    state["id_maps"]          = load_id_maps()
    state["metrics"]          = load_all_metrics()
    state["params"]           = load_all_params()
    state["metrics_summary"]  = load_metrics_summary()
    state["best_decision"]    = load_best_model_decision()
    state["available"] = {
        "popularity": load_popularity_model()    is not None,
        "svd":        load_svd_model()           is not None,
        "knn":        load_knn_model()           is not None,
        "lightgbm":   load_lightgbm_model()      is not None,
        "als":        load_als_model()           is not None,
        "ncf":        load_ncf_model()           is not None,
    }
    return state


def ensure_state_loaded() -> dict:
    """Гарантирует, что прелоад выполнен ровно один раз за сессию."""
    if "app_state" not in st.session_state:
        st.session_state.app_state = warm_up_all_artifacts()
    return st.session_state.app_state


# ---------------------------------------------------------------------------
# Боковая панель: навигация и индикаторы доступности моделей
# ---------------------------------------------------------------------------

NAV_ITEMS = [
    "Главная",
    "Анализ данных",
    "Сравнение моделей",
    "Детали моделей",
    "Получить рекомендации",
    "Метрики и теория",
]


def render_sidebar(state: dict) -> str:
    with st.sidebar:
        st.markdown("### Movie Recommender")
        st.caption("MovieLens ml-latest-small / 7 моделей + ансамбль")
        st.divider()

        choice = st.radio(
            "Раздел",
            NAV_ITEMS,
            index=0,
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Доступность моделей**")
        avail = state.get("available", {})
        for key in ["popularity", "svd", "knn", "lightgbm", "als", "ncf"]:
            ok = avail.get(key, False)
            mark = "[ok]" if ok else "[--]"
            label = MODEL_LABELS.get(key, key)
            color = "#0F6E56" if ok else "#A32D2D"
            st.markdown(
                f"<div style='font-size:.85rem; margin:2px 0; color:{color}'>"
                f"<code style='font-size:.78rem'>{mark}</code> {label}"
                f"</div>",
                unsafe_allow_html=True,
            )

    return choice


# ---------------------------------------------------------------------------
# Вкладка 1: Главная / Обзор проекта
# ---------------------------------------------------------------------------

def render_home(state: dict) -> None:
    st.title("Рекомендательная система фильмов")
    st.markdown(
        "Курсовой проект: построение, сравнение и интерактивная "
        "демонстрация моделей коллаборативной фильтрации на датасете "
        "**MovieLens ml-latest-small**."
    )

    ratings = state["ratings"]
    movies  = state["movies"]
    metrics = state["metrics"]
    best_decision = state.get("best_decision", {}) or {}

    # Карточки-метрики.
    c1, c2, c3, c4 = st.columns(4)
    n_users  = ratings["userId"].nunique()  if ratings is not None else 610
    n_movies = movies.shape[0]              if movies  is not None else 9742
    n_ratings = ratings.shape[0]            if ratings is not None else 100836
    n_models = sum(1 for m in metrics.values() if m)

    def metric_card(col, value, label):
        with col:
            st.markdown(
                f'<div class="big-metric-value">{value}</div>'
                f'<div class="big-metric-label">{label}</div>',
                unsafe_allow_html=True,
            )

    metric_card(c1, fmt_int(n_users),   "пользователей")
    metric_card(c2, fmt_int(n_movies),  "фильмов")
    metric_card(c3, fmt_int(n_ratings), "оценок")
    metric_card(c4, str(n_models),      "обученных моделей")

    st.divider()

    # Победитель проекта.
    best_key = best_decision.get("best_model", DEFAULT_BEST_MODEL)
    best_label = MODEL_LABELS.get(best_key, best_key)
    ens_metrics = metrics.get("ensemble", {}) or {}
    als_metrics = metrics.get("als", {}) or {}
    winner_metrics = metrics.get(best_key, {}) or {}

    # Если по какой-то причине у победителя нет метрик - подмешиваем из
    # ансамбля (он - наша «золотая» модель проекта).
    if not winner_metrics or not winner_metrics.get("ndcg@10"):
        winner_metrics = ens_metrics

    st.markdown("#### Победитель проекта")
    st.markdown(
        f"**{best_label}** <span class='best-pill'>лучшая модель</span>",
        unsafe_allow_html=True,
    )
    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("NDCG@10",      fmt_float(winner_metrics.get("ndcg@10")))
    wc2.metric("Precision@10", fmt_float(winner_metrics.get("precision@10")))
    wc3.metric("HitRate@10",   fmt_float(winner_metrics.get("hitrate@10")))
    wc4.metric("Coverage@20",  fmt_float(winner_metrics.get("coverage@20")))

    st.divider()

    # Краткое описание проекта.
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Архитектура решения")
        st.markdown(
            "1. **Предобработка**: временной сплит train/val/test, "
            "построение sparse user-item матрицы, обогащение метаданных, "
            "признаки на основе тегов (TF-IDF + SVD).\n"
            "2. **Базовые модели**: Global Mean (RMSE-baseline) и "
            "Popularity с Bayesian-средним.\n"
            "3. **Коллаборативные модели**: SVD (Surprise), KNN item-based, "
            "ALS (implicit feedback с бинаризацией на 4.0).\n"
            "4. **Признаковая модель**: LightGBM на 47 числовых и "
            "категориальных признаках пользователя и фильма.\n"
            "5. **Нейросетевая модель**: NCF (NeuMF - GMF + MLP) на TensorFlow.\n"
            "6. **Гиперпараметрическая оптимизация**: Optuna для каждой "
            "модели; результаты сохранены в parquet-таблицах и HTML-графиках.\n"
            "7. **Ансамбль**: линейная комбинация ALS + SVD + NCF "
            "с весами, подобранными через grid + Optuna."
        )

    with right:
        st.markdown("#### Стек технологий")
        st.markdown(
            "- **Python 3.11**, pandas, numpy, scipy\n"
            "- **Surprise** - SVD, KNN\n"
            "- **implicit** - ALS\n"
            "- **LightGBM** - регрессор / ранкер\n"
            "- **TensorFlow / Keras** - NCF\n"
            "- **Optuna** - HPO\n"
            "- **Plotly** - интерактивные графики\n"
            "- **Streamlit** - демонстрационное приложение"
        )

    st.divider()
    st.info(
        "Перейдите в раздел **Получить рекомендации**, чтобы попробовать "
        "систему в действии (можно как существующим пользователем, так и "
        "новым - со своими оценками)."
    )



# ---------------------------------------------------------------------------
# Вкладка 2: Анализ данных (EDA)
# ---------------------------------------------------------------------------

def render_eda(state: dict) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("Анализ данных (EDA)")
    st.caption(
        "Краткий обзор датасета MovieLens ml-latest-small: "
        "распределения, топы, активность пользователей."
    )

    ratings = state["ratings"]
    movies  = state["movies"]
    if ratings is None or movies is None:
        st.error("Не удалось загрузить ratings.csv или movies.csv.")
        return

    # ---- Сводные цифры -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Пользователей",      fmt_int(ratings["userId"].nunique()))
    c2.metric("Фильмов в каталоге", fmt_int(movies.shape[0]))
    c3.metric("Оценок",             fmt_int(ratings.shape[0]))
    c4.metric("Средняя оценка",     fmt_float(ratings["rating"].mean(), 2))

    st.divider()

    # ---- Распределение оценок ---------------------------------------------
    st.markdown("#### Распределение оценок (rating)")
    rating_counts = (ratings["rating"].value_counts()
                     .sort_index().reset_index())
    rating_counts.columns = ["rating", "count"]
    fig = px.bar(
        rating_counts, x="rating", y="count",
        labels={"rating": "Оценка", "count": "Число оценок"},
        color="rating", color_continuous_scale="Blues",
    )
    fig.update_layout(
        showlegend=False, height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Распределение года выпуска ---------------------------------------
    st.markdown("#### Распределение фильмов по году выпуска")
    if "year" in movies.columns:
        years = movies["year"].dropna().astype(int)
        years = years[(years >= 1900) & (years <= 2025)]
        years_df = pd.DataFrame({"year": years.values})
        fig = px.histogram(
            years_df, x="year", nbins=40,
            labels={"year": "Год выпуска", "count": "Число фильмов"},
            color_discrete_sequence=["#378ADD"],
        )
        fig.update_yaxes(title_text="Число фильмов")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                          bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Колонка year недоступна в метаданных фильмов.")

    # ---- Топ-15 жанров -----------------------------------------------------
    st.markdown("#### Топ-15 жанров")
    if "genres_list" in movies.columns:
        all_genres: list[str] = []
        for g in movies["genres_list"]:
            all_genres.extend(g)
        genre_counts = (pd.Series(all_genres).value_counts()
                        .head(15).reset_index())
        genre_counts.columns = ["genre", "count"]
        fig = px.bar(
            genre_counts, x="count", y="genre", orientation="h",
            labels={"count": "Число фильмов", "genre": "Жанр"},
            color="count", color_continuous_scale="Teal",
        )
        fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10),
                          coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Топ-10 по числу оценок и среднему рейтингу -----------------------
    st.markdown("#### Топ-10 самых популярных фильмов")
    movie_stats = state["movie_stats"]
    top_pop = (movie_stats.sort_values("rating_count", ascending=False)
               .head(10)
               .merge(movies[["movieId", "clean_title", "year"]],
                      on="movieId", how="left"))
    top_pop = top_pop[["clean_title", "year", "rating_count", "avg_rating"]]
    top_pop.columns = ["Фильм", "Год", "Число оценок", "Средний рейтинг"]
    top_pop["Средний рейтинг"] = top_pop["Средний рейтинг"].round(2)
    st.dataframe(top_pop, use_container_width=True, hide_index=True)

    st.markdown("#### Топ-10 фильмов по среднему рейтингу")
    min_votes = st.slider(
        "Минимальное число оценок", 5, 200, 50, step=5,
        help="Фильмы с малым числом оценок дают нерепрезентативные средние."
    )
    top_rated = (movie_stats[movie_stats["rating_count"] >= min_votes]
                 .sort_values("avg_rating", ascending=False)
                 .head(10)
                 .merge(movies[["movieId", "clean_title", "year"]],
                        on="movieId", how="left"))
    top_rated = top_rated[["clean_title", "year", "rating_count", "avg_rating"]]
    top_rated.columns = ["Фильм", "Год", "Число оценок", "Средний рейтинг"]
    top_rated["Средний рейтинг"] = top_rated["Средний рейтинг"].round(2)
    st.dataframe(top_rated, use_container_width=True, hide_index=True)

    # ---- Активность пользователей -----------------------------------------
    st.markdown("#### Распределение активности пользователей")
    user_activity = ratings.groupby("userId").size().values
    # Ручные логарифмические бины - чтобы длинный хвост был виден.
    # Plotly + nbins + type='log' даёт пустые столбцы; делаем вручную.
    if len(user_activity) > 0 and user_activity.max() > 1:
        lo = max(1, int(user_activity.min()))
        hi = int(user_activity.max()) + 1
        bins = np.logspace(np.log10(lo), np.log10(hi), num=30)
        counts, edges = np.histogram(user_activity, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths  = edges[1:] - edges[:-1]
        fig = go.Figure(go.Bar(
            x=centers, y=counts, width=widths,
            marker_color="#7F77DD",
            hovertemplate=("Диапазон: %{customdata[0]:.0f}–%{customdata[1]:.0f}<br>"
                           "Пользователей: %{y}<extra></extra>"),
            customdata=np.stack([edges[:-1], edges[1:]], axis=1),
        ))
        fig.update_xaxes(type="log",
                         title_text="Число оценок на пользователя (лог)")
        fig.update_yaxes(title_text="Число пользователей")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10),
                          bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
        # Краткая статистика рядом.
        p25, p50, p75, p99 = np.percentile(user_activity, [25, 50, 75, 99])
        cs1, cs2, cs3, cs4, cs5 = st.columns(5)
        cs1.metric("Минимум",     fmt_int(user_activity.min()))
        cs2.metric("Медиана",     fmt_int(p50))
        cs3.metric("75-й перц.",  fmt_int(p75))
        cs4.metric("99-й перц.",  fmt_int(p99))
        cs5.metric("Максимум",    fmt_int(user_activity.max()))
    else:
        st.info("Недостаточно данных для построения гистограммы активности.")
    st.caption(
        "Ось X в логарифмическом масштабе - распределение длиннохвостое: "
        "большинство пользователей оценили мало фильмов, активных - единицы."
    )

    # ---- Корреляция популярность vs средняя оценка ------------------------
    st.markdown("#### Популярность против среднего рейтинга")
    sample = movie_stats.sample(min(2000, len(movie_stats)), random_state=42)
    fig = px.scatter(
        sample, x="rating_count", y="avg_rating",
        opacity=0.35,
        labels={"rating_count": "Число оценок (лог)",
                "avg_rating": "Средний рейтинг"},
        color_discrete_sequence=["#D85A30"],
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_xaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)
    corr = movie_stats[["rating_count", "avg_rating"]].corr().iloc[0, 1]
    st.caption(
        f"Корреляция Пирсона: **{corr:.3f}**. Слабая положительная связь - "
        "популярные фильмы в среднем оцениваются чуть выше, но это далеко "
        "не строгое правило."
    )



# ---------------------------------------------------------------------------
# Вкладка 3: Сравнение моделей
# ---------------------------------------------------------------------------

# Канонический список метрик, которые мы пытаемся достать из каждого JSON.
# Ключи приведены к виду, который выдаёт _normalize_metric_keys.
METRIC_KEYS = [
    "rmse", "mae",
    "precision@10", "recall@10", "ndcg@10", "hitrate@10",
    "coverage@20",
    "train_time_seconds",
]
METRIC_PRETTY = {
    "rmse":                "RMSE",
    "mae":                 "MAE",
    "precision@10":        "Precision@10",
    "recall@10":           "Recall@10",
    "ndcg@10":             "NDCG@10",
    "hitrate@10":          "HitRate@10",
    "coverage@20":         "Coverage@20",
    "train_time_seconds":  "Время обучения, с",
}
# Метрики, где выше = лучше (для подсветки в таблице).
HIGHER_IS_BETTER = {
    "precision@10", "recall@10", "ndcg@10", "hitrate@10",
    "coverage@20",
}


def _build_metrics_df(state: dict) -> pd.DataFrame:
    """Собирает единый DataFrame со всеми метриками всех моделей.

    Источник 1: metrics_summary.parquet (если есть - используем как основу).
    Источник 2: *_metrics.json (заполняем пробелы).
    """
    metrics = state["metrics"]
    summary = state.get("metrics_summary")

    rows = []
    for key, label in MODEL_LABELS.items():
        if key == "global_mean":
            continue
        m = metrics.get(key, {}) or {}
        row = {"key": key, "model": label}
        for mk in METRIC_KEYS:
            row[mk] = m.get(mk)
        rows.append(row)
    df = pd.DataFrame(rows)

    # Если есть metrics_summary - аккуратно мерджим (берём из него те поля,
    # где у нас NaN).
    if summary is not None and not summary.empty:
        s = summary.copy()
        # Унифицируем имя колонки с ключом модели.
        for candidate in ["model_key", "model_name", "model", "name"]:
            if candidate in s.columns:
                s = s.rename(columns={candidate: "key"})
                break
        s["key"] = s["key"].astype(str).str.lower()
        # Унифицируем имена метрик внутри summary: hit_rate@10 -> hitrate@10
        # и т.п. (parquet из 11_final_comparison может писать по-разному).
        col_renames = {}
        for c in s.columns:
            if c == "key":
                continue
            cn = c.lower().replace("hit_rate", "hitrate").replace(" ", "")
            if cn != c:
                col_renames[c] = cn
        if col_renames:
            s = s.rename(columns=col_renames)
        df = df.merge(s, on="key", how="left", suffixes=("", "_sum"))
        for mk in METRIC_KEYS:
            sum_col = f"{mk}_sum"
            if sum_col in df.columns:
                df[mk] = df[mk].combine_first(df[sum_col])
                df.drop(columns=[sum_col], inplace=True)
            # Также подхватываем mk из summary без суффикса (если он не
            # пересёкся с уже существующей колонкой).
            elif mk in df.columns and mk not in [r for r in rows[0].keys()]:
                pass
    return df


def render_compare(state: dict) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    st.title("Сравнение моделей")
    st.caption(
        "Сводная таблица метрик всех моделей, интерактивные сравнения и "
        "готовые графики из этапа исследования."
    )

    df = _build_metrics_df(state)
    if df.empty:
        st.error("Не удалось собрать метрики моделей.")
        return

    best_key = (state.get("best_decision") or {}).get(
        "best_model", DEFAULT_BEST_MODEL
    )

    # ---- Таблица ---------------------------------------------------------
    st.markdown("#### Все модели на одной таблице")
    # Оставляем только канонические колонки: model + METRIC_KEYS.
    # Все «лишние» столбцы из metrics_summary.parquet (hit_rate@10, hr@10
    # и т.п.) выкидываем, чтобы в таблице не было дубликатов одной метрики
    # под разными именами.
    keep_cols = ["model"] + [mk for mk in METRIC_KEYS if mk in df.columns]
    show_df = df[keep_cols].copy()
    rename_map = {mk: METRIC_PRETTY.get(mk, mk) for mk in METRIC_KEYS}
    rename_map["model"] = "Модель"
    show_df = show_df.rename(columns=rename_map)

    # Подсветка лучшей строки.
    best_label = MODEL_LABELS.get(best_key, best_key)

    def _highlight_best(row):
        if row["Модель"] == best_label:
            return ["background-color: rgba(15, 110, 86, 0.10);"] * len(row)
        return [""] * len(row)

    styled = (show_df.style
              .format({METRIC_PRETTY[m]: "{:.3f}"
                       for m in METRIC_KEYS if m != "train_time_seconds"
                       and METRIC_PRETTY[m] in show_df.columns}, na_rep="-")
              .format({"Время обучения, с": "{:.1f}"}, na_rep="-")
              .apply(_highlight_best, axis=1))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Скачать CSV.
    csv_bytes = show_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Скачать таблицу метрик (CSV)",
        data=csv_bytes,
        file_name="metrics_summary.csv",
        mime="text/csv",
    )

    st.divider()

    # ---- Bar-charts по метрикам ------------------------------------------
    st.markdown("#### Сравнение моделей по выбранным метрикам")
    available_models = df["model"].tolist()
    selected_models = st.multiselect(
        "Какие модели включить в сравнение",
        options=available_models,
        default=available_models,
    )
    metric_options = [m for m in METRIC_KEYS
                      if m != "train_time_seconds" and df[m].notna().any()]
    metric_choice = st.selectbox(
        "Метрика",
        options=metric_options,
        index=metric_options.index("ndcg@10") if "ndcg@10" in metric_options else 0,
        format_func=lambda x: METRIC_PRETTY.get(x, x),
    )
    sub = df[df["model"].isin(selected_models)][["model", metric_choice]].dropna()
    if sub.empty:
        st.info("Нет данных для выбранных моделей и метрики.")
    else:
        ascending = metric_choice in {"rmse", "mae"}
        sub = sub.sort_values(metric_choice, ascending=ascending)
        colors = ["#0F6E56" if m == best_label else "#378ADD"
                  for m in sub["model"]]
        fig = go.Figure(go.Bar(
            x=sub[metric_choice], y=sub["model"],
            orientation="h",
            marker_color=colors,
            text=[fmt_float(v) for v in sub[metric_choice]],
            textposition="outside",
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=40, t=20, b=10),
            xaxis_title=METRIC_PRETTY.get(metric_choice, metric_choice),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ---- Radar по рейтинговым метрикам -----------------------------------
    st.markdown("#### Профили моделей (radar)")
    st.caption(
        "Радар построен только по rank-метрикам в одной шкале. "
        "RMSE/MAE сюда не включены, потому что они в другой шкале и "
        "у некоторых моделей не определены. Значения нормированы: "
        "1.0 = лучшая модель по этой метрике среди выбранных."
    )
    radar_metrics = [m for m in
                     ["precision@10", "recall@10", "ndcg@10",
                      "hitrate@10", "coverage@20"]
                     if df[m].notna().any()]
    # Отдельный селектор для radar - чтобы не пересекать с основной таблицей.
    # По умолчанию показываем 3 ключевые модели: ансамбль, ALS, Popularity.
    radar_default = [m for m in
                     [MODEL_LABELS["ensemble"], MODEL_LABELS["als"],
                      MODEL_LABELS["popularity"]]
                     if m in available_models]
    radar_models = st.multiselect(
        "Модели на радаре (по умолчанию - топ-3 для читаемости)",
        options=available_models,
        default=radar_default or available_models[:3],
        key="radar_models",
    )
    if radar_metrics and radar_models:
        # Нормируем каждую метрику в [0, 1] по максимуму среди ВЫБРАННЫХ
        # моделей. Так лучшая в каждой оси выйдет к 1.0.
        norm_df = df[df["model"].isin(radar_models)].copy()
        for m in radar_metrics:
            mx = norm_df[m].max()
            norm_df[m] = norm_df[m] / mx if mx and mx > 0 else 0.0

        # Палитра - дискретная, контрастная.
        palette = ["#0F6E56", "#378ADD", "#D85A30",
                   "#7F77DD", "#FAC775", "#791F1F", "#444441"]
        fig = go.Figure()
        for i, (_, row) in enumerate(norm_df.iterrows()):
            r_vals = [row[m] for m in radar_metrics]
            # Замыкаем линию (повторяем первую точку в конце).
            r_vals.append(r_vals[0])
            theta_vals = [METRIC_PRETTY.get(m, m) for m in radar_metrics]
            theta_vals.append(theta_vals[0])
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatterpolar(
                r=r_vals, theta=theta_vals,
                mode="lines+markers",
                name=row["model"],
                line=dict(color=color, width=2.2),
                marker=dict(size=7, color=color),
                fill="toself",
                opacity=0.18,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.05],
                                tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12)),
            ),
            height=520, margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(orientation="h", y=-0.08, x=0.5,
                        xanchor="center", font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif not radar_models:
        st.info("Выберите хотя бы одну модель для радара.")

    # ---- Качество vs время обучения --------------------------------------
    st.markdown("#### Качество против времени обучения")
    if "train_time_seconds" in df.columns and df["train_time_seconds"].notna().any():
        scat = df.dropna(subset=["ndcg@10", "train_time_seconds"])
        if not scat.empty:
            fig = px.scatter(
                scat, x="train_time_seconds", y="ndcg@10",
                text="model", size=[20] * len(scat),
                color=["best" if m == best_label else "other"
                       for m in scat["model"]],
                color_discrete_map={"best": "#0F6E56", "other": "#378ADD"},
                labels={"train_time_seconds": "Время обучения, с",
                        "ndcg@10": "NDCG@10"},
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10),
                              showlegend=False)
            fig.update_xaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Ось X логарифмическая. Идеал - в верхнем левом углу: "
                "высокое качество при малом времени обучения."
            )
    else:
        st.info("В метриках нет колонки train_time_seconds - график пропущен.")

    st.divider()

    # ---- Готовые PNG-графики из models/ -----------------------------------
    st.markdown("#### Готовые графики из этапа исследования")
    png_files = [
        ("final_model_comparison.png", "Итоговое сравнение моделей"),
        ("ensemble_vs_all.png",        "Ансамбль против всех моделей"),
        ("ensemble_vs_pure_models.png","Ансамбль против чистых моделей"),
        ("ensemble_weights.png",       "Веса в ансамбле"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(png_files):
        p = MODELS_DIR / fname
        if p.exists():
            with cols[i % 2]:
                st.image(str(p), caption=caption, use_container_width=True)



# ---------------------------------------------------------------------------
# Вкладка 4: Детали моделей
# ---------------------------------------------------------------------------

# Текстовые описания каждой модели.
MODEL_DESCRIPTIONS = {
    "popularity": (
        "**Popularity baseline** с Bayesian-средним.\n\n"
        "Идея: для каждого фильма считаем взвешенное среднее между его "
        "собственной средней оценкой и глобальным средним. Чем меньше "
        "оценок у фильма, тем сильнее ранг тянется к глобальному среднему - "
        "так мы защищаемся от пустых выскочек с одной оценкой 5.0.\n\n"
        "Формула: `score = (v / (v + m)) * R + (m / (v + m)) * C`, где "
        "v - число оценок фильма, R - его средняя, C - глобальная средняя, "
        "m - параметр сглаживания (в этом проекте m = 200).\n\n"
        "Полезна как нижняя граница качества: любая 'настоящая' модель "
        "обязана быть лучше неё на rank-метриках."
    ),
    "svd": (
        "**SVD (matrix factorization)** из библиотеки Surprise.\n\n"
        "Классическая модель Funk-SVD: рейтинг приближается как "
        "`r_ui = mu + bu + bi + p_u . q_i`, где p_u и q_i - "
        "латентные факторы пользователя и фильма (по 31 компоненте), "
        "bu и bi - смещения, mu - глобальное среднее.\n\n"
        "Обучается стохастическим градиентным спуском по explicit ratings. "
        "Сильна по NDCG/Precision, но Coverage низкий: модель часто "
        "рекомендует одни и те же 'крепкие' фильмы."
    ),
    "knn": (
        "**KNN item-based** (Surprise, KNNWithMeans).\n\n"
        "Прогноз: средневзвешенное по k ближайшим фильмам, которые "
        "пользователь уже видел. Расстояние - cosine similarity между "
        "столбцами user-item матрицы.\n\n"
        "На rank-метриках уступает SVD и ALS - типично для item-based KNN "
        "на разреженных данных: похожих фильмов с достаточным числом "
        "общих оценщиков мало."
    ),
    "lightgbm": (
        "**LightGBM** на 47 признаках пользователя и фильма (агрегаты "
        "оценок, жанры в one-hot, TF-IDF тегов, временные признаки).\n\n"
        "Обучается как regressor на (userId, movieId) -> rating. "
        "Сильная сторона - высокий Coverage@20 (0.060): признаковые "
        "модели рекомендуют более разнообразные фильмы, чем чистые MF. "
        "Слабость - NDCG проседает: без явных взаимодействий между "
        "пользователем и фильмом по признакам сложно ранжировать."
    ),
    "als": (
        "**ALS (Alternating Least Squares)** из библиотеки implicit.\n\n"
        "Implicit feedback подход: бинаризуем оценки на порог 4.0 "
        "('понравилось / не понравилось'), затем итеративно решаем "
        "две регуляризованные задачи наименьших квадратов - "
        "при фиксированных item-факторах ищем user-факторы и наоборот.\n\n"
        "Лучшая одиночная модель в этом проекте: NDCG@10 = 0.290, "
        "Coverage@20 = 0.077. ALS не пытается предсказать численный "
        "рейтинг, поэтому RMSE для неё не считается - только rank-метрики."
    ),
    "ncf": (
        "**NCF (Neural Collaborative Filtering, NeuMF)** на TensorFlow/Keras.\n\n"
        "Архитектура: эмбеддинги пользователя и фильма идут параллельно "
        "в две ветки:\n"
        "- **GMF** (Generalized Matrix Factorization): "
        "поэлементное произведение эмбеддингов.\n"
        "- **MLP**: конкатенация эмбеддингов -> несколько Dense-слоёв.\n\n"
        "Выходы обеих веток конкатенируются и подаются на финальный "
        "sigmoid. Обучаем как бинарную классификацию (релевантен / нет).\n\n"
        "В нашем эксперименте NCF дала NDCG@10 = 0.087 - заметно хуже "
        "ALS. Возможные причины: размер датасета (610 пользователей мало "
        "для глубокой сети), сложность тюнинга."
    ),
    "ensemble": (
        "**Ансамбль ALS + SVD + NCF**.\n\n"
        "Каждая модель даёт свой набор скоров; мы нормируем их в ранги "
        "[0, 1] и складываем с весами:\n"
        "- ALS: 0.72 (основной 'локомотив')\n"
        "- NCF: 0.27 (вносит нелинейность)\n"
        "- SVD: 0.01 (почти не влияет, но позволяет покрыть фильмы, "
        "которые ALS оценивает плохо).\n\n"
        "Веса подобраны Optuna-поиском по NDCG@10 на валидации. "
        "Ансамбль чуть лучше одиночной ALS (NDCG@10 = 0.295 против 0.290)."
    ),
}


def render_model_details(state: dict) -> None:
    st.title("Детали моделей")
    st.caption(
        "Описание алгоритма, гиперпараметры, метрики, история Optuna и "
        "сравнительные графики - по каждой обученной модели."
    )

    metrics = state["metrics"]
    params  = state["params"]

    # Кладём каждую модель в свой st.tab.
    model_keys = ["popularity", "svd", "knn", "lightgbm", "als",
                  "ncf", "ensemble"]
    tabs = st.tabs([MODEL_LABELS[k].split(" (")[0] for k in model_keys])

    for tab, key in zip(tabs, model_keys):
        with tab:
            _render_one_model(state, key,
                              metrics.get(key, {}) or {},
                              params.get(key, {}) or {})


def _render_one_model(state: dict, key: str,
                      mdict: dict, pdict: dict) -> None:
    """Рендерит одну подвкладку модели."""
    label = MODEL_LABELS.get(key, key)
    st.markdown(f"### {label}")

    # Описание.
    desc = MODEL_DESCRIPTIONS.get(key, "")
    if desc:
        with st.container(border=True):
            st.markdown(desc)

    # Метрики и гиперпараметры в двух колонках.
    cm, cp = st.columns(2)
    with cm:
        st.markdown("**Метрики на тесте**")
        # Считаем только реальные метрики (без train_time_seconds), чтобы
        # понять, есть ли вообще что показывать.
        real_metric_keys = [mk for mk in METRIC_KEYS
                            if mk != "train_time_seconds"]
        has_real = any(mdict.get(mk) is not None for mk in real_metric_keys)
        if not mdict or not has_real:
            st.info("Файл с метриками отсутствует или пуст.")
        else:
            rows = []
            for mk in METRIC_KEYS:
                if mk in mdict and mdict[mk] is not None:
                    pretty = METRIC_PRETTY.get(mk, mk)
                    val = mdict[mk]
                    if mk == "train_time_seconds":
                        rows.append((pretty, f"{val:.1f}"))
                    else:
                        rows.append((pretty, f"{val:.4f}"))
            # Доп. метрики, не в стандартном списке.
            for k, v in mdict.items():
                if k not in METRIC_KEYS and isinstance(v, (int, float)):
                    rows.append((k, f"{v:.4f}"))
            mdf = pd.DataFrame(rows, columns=["Метрика", "Значение"])
            st.dataframe(mdf, use_container_width=True, hide_index=True)

    with cp:
        st.markdown("**Гиперпараметры (лучшие из Optuna)**")
        if not pdict:
            st.info("Файл с гиперпараметрами отсутствует.")
        else:
            # Плоско, без вложенных dict.
            rows = []
            for k, v in pdict.items():
                if isinstance(v, (dict, list)):
                    rows.append((k, json.dumps(v, ensure_ascii=False)))
                else:
                    rows.append((k, str(v)))
            pdf = pd.DataFrame(rows, columns=["Параметр", "Значение"])
            st.dataframe(pdf, use_container_width=True, hide_index=True)

    # Optuna-графики.
    st.markdown("**История оптимизации Optuna**")
    hist_path = MODELS_DIR / f"optuna_{key}_history.html"
    imp_path  = MODELS_DIR / f"optuna_{key}_importance.html"
    if hist_path.exists() or imp_path.exists():
        ch1, ch2 = st.columns(2)
        with ch1:
            if hist_path.exists():
                _embed_html(hist_path, height=420)
                st.caption("История значений целевой функции по итерациям.")
            else:
                st.info("История Optuna не найдена.")
        with ch2:
            if imp_path.exists():
                _embed_html(imp_path, height=420)
                st.caption("Важность гиперпараметров.")
            else:
                st.info("Важность параметров не найдена.")
    else:
        st.info("HTML-графики Optuna для этой модели отсутствуют.")

    # Optuna trials parquet - быстрая сводка.
    trials_path = MODELS_DIR / f"{key}_optuna_trials.parquet"
    if not trials_path.exists():
        # Имена бывают разные - попробуем "ensemble_optuna_trials.parquet"
        trials_path = MODELS_DIR / f"{key}_optuna_trials.parquet"
    if trials_path.exists():
        with st.expander("Таблица всех попыток Optuna", expanded=False):
            trials = _safe_read_parquet(trials_path)
            if trials is not None and not trials.empty:
                st.dataframe(trials.head(200),
                             use_container_width=True, hide_index=True)
                st.caption(f"Всего попыток: {len(trials)}.")

    # Сравнительный PNG.
    png_map = {
        "popularity": None,
        "svd":        "svd_vs_baseline.png",
        "knn":        "knn_vs_baseline.png",
        "lightgbm":   "lightgbm_vs_all.png",
        "als":        "als_vs_popularity.png",
        "ncf":        "ncf_vs_all_models.png",
        "ensemble":   "ensemble_alpha_grid.png",
    }
    pname = png_map.get(key)
    if pname:
        ppath = MODELS_DIR / pname
        if ppath.exists():
            st.markdown("**Сравнительный график**")
            st.image(str(ppath), use_container_width=True)


def _embed_html(path: Path, height: int = 400) -> None:
    """Встраивает HTML-файл через components.v1.html.

    Plotly-графики Optuna создаются как самодостаточные HTML, их можно
    подгружать напрямую."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            html = fh.read()
        st.components.v1.html(html, height=height, scrolling=True)
    except Exception as exc:
        st.warning(f"Не удалось встроить {path.name}: {type(exc).__name__}.")



# ---------------------------------------------------------------------------
# Вкладка 6: Метрики и теория (рендерим до Recommend, чтобы код был
# группирован "вспомогательные" -> "ключевая")
# ---------------------------------------------------------------------------

def render_theory(state: dict) -> None:
    st.title("Метрики и теория")
    st.caption(
        "Краткое описание метрик качества, методология оценки и выбора "
        "финальной модели."
    )

    # --- Метрики -----------------------------------------------------------
    st.markdown("### Метрики качества")

    with st.expander("RMSE (Root Mean Squared Error)", expanded=False):
        st.markdown(
            "Среднеквадратичная ошибка предсказания численного рейтинга. "
            "Подходит для регрессионных моделей (SVD, KNN, LightGBM).\n\n"
            "Формула: `RMSE = sqrt(mean((r_ui - r_hat_ui)^2))`.\n\n"
            "Чем меньше - тем лучше. У ALS не определена, потому что ALS "
            "предсказывает не рейтинг, а 'предпочтительность' в произвольной "
            "шкале."
        )

    with st.expander("MAE (Mean Absolute Error)", expanded=False):
        st.markdown(
            "Средняя абсолютная ошибка: `MAE = mean(|r_ui - r_hat_ui|)`. "
            "Менее чувствительна к выбросам, чем RMSE."
        )

    with st.expander("Precision@K", expanded=False):
        st.markdown(
            "Доля релевантных фильмов в первых K рекомендациях. "
            "Релевантным считаем фильм с истинной оценкой >= 3.5.\n\n"
            "`Precision@K = |relevant intersect top_K| / K`.\n\n"
            "У нас K = 10."
        )

    with st.expander("Recall@K", expanded=False):
        st.markdown(
            "Доля найденных релевантных фильмов от всех релевантных "
            "у пользователя в тесте.\n\n"
            "`Recall@K = |relevant intersect top_K| / |relevant|`.\n\n"
            "Precision и Recall - в трейдоффе: можно увеличить один за "
            "счёт другого."
        )

    with st.expander("NDCG@K (Normalized Discounted Cumulative Gain)",
                     expanded=False):
        st.markdown(
            "Учитывает не только сам факт попадания релевантного фильма "
            "в топ, но и его позицию: чем выше - тем больше вклад.\n\n"
            "`DCG@K = sum_{i=1..K} (2^rel_i - 1) / log2(i + 1)`.\n\n"
            "`NDCG@K = DCG@K / IDCG@K`, где IDCG - максимально "
            "достижимое значение для данного пользователя.\n\n"
            "Основная метрика ранжирования в этом проекте."
        )

    with st.expander("HitRate@K", expanded=False):
        st.markdown(
            "Доля пользователей, у которых хотя бы один релевантный фильм "
            "попал в топ-K. Хорошо отвечает на вопрос 'успеваем ли мы "
            "сделать хоть одну удачную рекомендацию каждому пользователю'."
        )

    with st.expander("Coverage@K", expanded=False):
        st.markdown(
            "Доля уникальных фильмов из каталога, которые хоть раз "
            "попали в топ-K рекомендаций по всему тестовому набору.\n\n"
            "`Coverage@K = |union_u top_K(u)| / |catalog|`.\n\n"
            "Высокая Coverage - модель разнообразная (рекомендует много "
            "разных фильмов). Низкая - модель сваливается в одни и те же "
            "хиты."
        )

    st.divider()

    # --- Методология -------------------------------------------------------
    st.markdown("### Методология оценки")

    st.markdown(
        "**Временной сплит.** Все оценки отсортированы по timestamp и "
        "разделены на train (раннее), val и test (позднее). Это имитирует "
        "реальный сценарий: предсказываем будущее на основе прошлого. "
        "Случайный сплит был бы слишком оптимистичным - модель могла бы "
        "учиться по 'будущим' оценкам того же пользователя."
    )

    st.markdown(
        f"**Единый порог релевантности.** Релевантным считаем фильм с "
        f"истинной оценкой >= {RELEVANCE_THRESHOLD}. Этот же порог "
        "использован при бинаризации для implicit ALS."
    )

    st.markdown(
        "**Защита от утечки.** Для каждой модели метрики считаются только "
        "по фильмам, которые пользователь НЕ видел в train. При оценке "
        "ансамбля все компоненты обучены на одном и том же train-сплите."
    )

    st.markdown(
        "**Отдельный cold-start eval.** В `data/processed/cold_start_eval.parquet` "
        "выделены 'новые' пользователи - те, чьи первые оценки идут в test. "
        "На них тестируется устойчивость моделей к холодному старту."
    )

    st.divider()

    # --- Выбор финальной модели -------------------------------------------
    st.markdown("### Выбор финальной модели")
    decision = state.get("best_decision", {}) or {}
    if decision:
        st.json(decision)
    else:
        st.markdown(
            "Файл `best_model_decision.json` не найден. По метрикам в проекте "
            "лучший результат показывает **ансамбль ALS + SVD + NCF**, "
            "его и считаем финальной моделью."
        )

    # Preprocessing meta - полезно показать.
    meta = load_preprocessing_meta()
    if meta:
        with st.expander("preprocessing_meta.json", expanded=False):
            st.json(meta)



# ===========================================================================
# Вкладка 5: Получить рекомендации (КЛЮЧЕВАЯ)
# ===========================================================================

def _movie_card(
    movie_row: pd.Series,
    score: Optional[float],
    avg_rating: Optional[float],
    n_votes: Optional[int],
    score_label: str = "Predicted score",
    score_min: float = 0.0,
    score_max: float = 1.0,
) -> None:
    """Отрисовывает одну карточку фильма (текстовая, без постера)."""
    title = strip_year(movie_row.get("title", "")) or movie_row.get(
        "clean_title", "Unknown"
    )
    year = movie_row.get("year", None)
    genres = movie_row.get("genres_list", [])
    if not isinstance(genres, list):
        genres = genres_to_list(movie_row.get("genres", ""))

    badges_html = render_genre_badges(genres) if genres else ""
    year_str = f" ({int(year)})" if year and not (
        isinstance(year, float) and np.isnan(year)
    ) else ""

    stars = render_stars(avg_rating) if avg_rating else ""
    avg_txt = (
        f"<span class='star-row'>{stars}</span> "
        f"<span class='meta'>{fmt_float(avg_rating, 2)} / 5.0"
        f"{f', {fmt_int(n_votes)} оценок' if n_votes else ''}</span>"
    ) if avg_rating else ""

    # Прогресс-бар для score (рендерим вручную через HTML).
    if score is not None:
        denom = score_max - score_min if score_max > score_min else 1.0
        frac = max(0.0, min(1.0, (score - score_min) / denom))
        pct = int(frac * 100)
        score_bar = (
            f"<div style='margin-top:6px'>"
            f"<div class='meta' style='margin-bottom:3px'>"
            f"{score_label}: <b>{fmt_float(score, 3)}</b></div>"
            f"<div style='background:rgba(120,120,120,0.18); border-radius:6px; "
            f"height:8px; overflow:hidden;'>"
            f"<div style='width:{pct}%; height:8px; background:#378ADD;'></div>"
            f"</div></div>"
        )
    else:
        score_bar = ""

    html = (
        f"<div class='recsys-card'>"
        f"<h4>{title}{year_str}</h4>"
        f"<div class='meta' style='margin-bottom:6px'>{badges_html}</div>"
        f"{avg_txt}"
        f"{score_bar}"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _render_recommendations(
    pairs: list[tuple[int, float]],
    movies: pd.DataFrame,
    movie_stats: pd.DataFrame,
    score_label: str,
) -> None:
    """Отрисовывает список (movieId, score) карточками в 2 колонки."""
    if not pairs:
        st.warning(
            "Рекомендаций не получено. Возможно, у пользователя нет "
            "взаимодействий, либо выбранная модель не смогла построить "
            "профиль (например, все оценки <= 3.0 для ALS)."
        )
        return

    df = pd.DataFrame(pairs, columns=["movieId", "score"])
    df = df.merge(movies, on="movieId", how="left")
    df = df.merge(movie_stats, on="movieId", how="left")

    # Динамическая шкала для прогресс-бара.
    s_min = df["score"].min() if not df.empty else 0.0
    s_max = df["score"].max() if not df.empty else 1.0
    if s_max <= s_min:
        s_max = s_min + 1e-6

    cols = st.columns(2)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 2]:
            _movie_card(
                row,
                score=row["score"],
                avg_rating=row.get("avg_rating"),
                n_votes=row.get("rating_count"),
                score_label=score_label,
                score_min=float(s_min),
                score_max=float(s_max),
            )

    # Кнопка экспорта в CSV.
    export = df[[c for c in
                 ["movieId", "clean_title", "year", "genres",
                  "score", "avg_rating", "rating_count"]
                 if c in df.columns]].copy()
    csv_bytes = export.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Экспортировать рекомендации (CSV)",
        data=csv_bytes,
        file_name="recommendations.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Режим A: существующий пользователь
# ---------------------------------------------------------------------------

def render_mode_existing(state: dict) -> None:
    st.markdown("### Режим A. Существующий пользователь")
    st.caption(
        "Выберите пользователя из датасета. Мы покажем его историю оценок "
        "и сгенерируем рекомендации - фильмы, которые он ещё не видел."
    )

    ratings  = state["ratings"]
    movies   = state["movies"]
    movie_stats = state["movie_stats"]
    id_maps  = state["id_maps"]
    avail    = state.get("available", {})

    if ratings is None or movies is None:
        st.error("Данные недоступны.")
        return

    # ---- Выбор пользователя ----------------------------------------------
    user_ids = sorted(ratings["userId"].unique().astype(int).tolist())
    selected_user = st.selectbox(
        "ID пользователя",
        options=user_ids,
        index=0,
        format_func=lambda u: f"userId = {u}",
    )

    # ---- История оценок ---------------------------------------------------
    user_hist = (ratings[ratings["userId"] == selected_user]
                 .merge(movies[["movieId", "clean_title", "year",
                                "genres", "genres_list"]],
                        on="movieId", how="left")
                 .sort_values("rating", ascending=False))

    st.markdown(f"#### История пользователя {selected_user}")
    st.caption(
        f"Всего оценок: **{fmt_int(len(user_hist))}**, "
        f"средняя: **{fmt_float(user_hist['rating'].mean(), 2)}**, "
        f"мин/макс: **{user_hist['rating'].min():.1f} / "
        f"{user_hist['rating'].max():.1f}**."
    )

    show_hist = user_hist.head(20)[
        ["clean_title", "year", "genres", "rating"]
    ].copy()
    show_hist.columns = ["Фильм", "Год", "Жанры", "Оценка"]
    st.dataframe(show_hist, use_container_width=True, hide_index=True)

    st.divider()

    # ---- Выбор модели и параметров ---------------------------------------
    st.markdown("#### Параметры рекомендаций")
    model_options = []
    if avail.get("als") and avail.get("svd") and avail.get("ncf"):
        model_options.append(("ensemble", "Ансамбль ALS + SVD + NCF (рекомендуется)"))
    if avail.get("als"):
        model_options.append(("als", "ALS (лучшая одиночная модель)"))
    if avail.get("svd"):
        model_options.append(("svd", "SVD (классическая факторизация)"))
    if avail.get("ncf"):
        model_options.append(("ncf", "NCF (нейросеть)"))
    if avail.get("popularity") or True:  # popularity всегда доступна
        model_options.append(("popularity", "Popularity (baseline)"))

    if not model_options:
        st.error("Ни одна модель не загружена.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        model_choice = st.radio(
            "Модель",
            options=[k for k, _ in model_options],
            format_func=lambda k: dict(model_options)[k],
            horizontal=False,
        )
    with c2:
        top_n = st.slider("Сколько рекомендаций", 5, 20, 10)

    if st.button("Получить рекомендации", type="primary",
                 use_container_width=True):
        seen_movies = set(user_hist["movieId"].astype(int).tolist())
        with st.spinner("Считаем рекомендации..."):
            pairs = _dispatch_existing_user(
                state, model_choice, selected_user,
                exclude=seen_movies, top_n=top_n,
            )

        # Fallback на popularity, если выбранная модель ничего не вернула
        # (например, NCF не загрузилась или пользователь оказался вне маппинга).
        used_label = MODEL_LABELS.get(model_choice, model_choice)
        if not pairs and model_choice != "popularity":
            st.warning(
                f"Модель '{used_label}' не вернула рекомендаций. "
                "Показываем popularity-fallback."
            )
            pairs = recommend_popularity(movie_stats, seen_movies, top_n)
            used_label = "Popularity (fallback)"

        st.markdown(f"#### Топ-{top_n} рекомендаций ({used_label})")
        _render_recommendations(pairs, movies, movie_stats,
                                score_label=f"{model_choice} score")


def _dispatch_existing_user(
    state: dict, model_key: str, raw_user_id: int,
    exclude: set[int], top_n: int,
) -> list[tuple[int, float]]:
    """Вызывает соответствующую функцию инференса по выбранной модели."""
    id_maps = state["id_maps"]
    movie_stats = state["movie_stats"]
    user_item = load_user_item_sparse()

    if model_key == "popularity":
        return recommend_popularity(movie_stats, exclude, top_n)
    if model_key == "als":
        als = load_als_model()
        return recommend_als_existing_user(
            als, user_item, id_maps, raw_user_id, exclude, top_n
        )
    if model_key == "svd":
        svd = load_svd_model()
        return recommend_svd_existing_user(
            svd, id_maps, raw_user_id, movie_stats, exclude, top_n
        )
    if model_key == "ncf":
        ncf = load_ncf_model()
        return recommend_ncf_existing_user(
            ncf, id_maps, raw_user_id, movie_stats, exclude, top_n
        )
    if model_key == "ensemble":
        params = state["params"].get("ensemble", {}) or {}
        weights = _extract_ensemble_weights(params)
        return recommend_ensemble_existing_user(
            raw_user_id,
            load_als_model(), load_svd_model(), load_ncf_model(),
            user_item, id_maps, movie_stats, exclude, top_n, weights,
        )
    return []


def _extract_ensemble_weights(params: dict) -> dict[str, float]:
    """Достаёт веса {als, svd, ncf} из ensemble_params.json.

    Формат может быть разным: либо плоский ('weight_als'), либо вложенный
    ({'weights': {'als': ..., 'svd': ..., 'ncf': ...}}). Пытаемся обе версии."""
    default = {"als": 0.72, "svd": 0.01, "ncf": 0.27}
    if not params:
        return default
    if "weights" in params and isinstance(params["weights"], dict):
        w = params["weights"]
    else:
        w = {}
        for k in ["als", "svd", "ncf"]:
            for candidate in [k, f"weight_{k}", f"w_{k}", f"{k}_weight"]:
                if candidate in params:
                    w[k] = float(params[candidate])
                    break
    # Заполняем недостающее дефолтами.
    out = default.copy()
    for k, v in w.items():
        if k in out:
            out[k] = float(v)
    # Нормализуем сумму до 1.
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out



# ---------------------------------------------------------------------------
# Режим B: новый пользователь (cold start)
# ---------------------------------------------------------------------------

def render_mode_new_user(state: dict) -> None:
    st.markdown("### Режим B. Новый пользователь")
    st.caption(
        "Поставьте оценки нескольким фильмам, и система найдёт похожие "
        "ваши вкусы фильмы. Используются fold-in алгоритмы: мы достраиваем "
        "ваш пользовательский вектор на лету, без переобучения модели."
    )

    movies      = state["movies"]
    movie_stats = state["movie_stats"]
    avail       = state.get("available", {})

    if movies is None:
        st.error("Метаданные фильмов недоступны.")
        return

    # --- session_state: словарь {movieId: rating} -------------------------
    if "new_user_ratings" not in st.session_state:
        st.session_state.new_user_ratings = {}

    # --- Блок 1: поиск и добавление фильма ---------------------------------
    st.markdown("#### 1. Найдите фильм и поставьте оценку")
    catalog = movies.merge(movie_stats, on="movieId", how="left")
    catalog["rating_count"] = catalog["rating_count"].fillna(0).astype(int)
    catalog = catalog.sort_values("rating_count", ascending=False)

    query = st.text_input(
        "Поиск по названию",
        placeholder="Например: Toy Story, Matrix, Pulp Fiction",
    )
    if query:
        q = query.lower().strip()
        filtered = catalog[
            catalog["clean_title"].str.lower().str.contains(q, na=False)
        ].head(30)
    else:
        # По умолчанию - 30 самых популярных.
        filtered = catalog.head(30)

    if filtered.empty:
        st.info("Ничего не найдено. Попробуйте другой запрос.")
    else:
        # Двухколоночный селектор: фильм + оценка + кнопка.
        movie_choice = st.selectbox(
            "Фильм",
            options=filtered["movieId"].tolist(),
            format_func=lambda mid: (
                f"{filtered[filtered['movieId'] == mid]['clean_title'].iloc[0]} "
                f"({int(filtered[filtered['movieId'] == mid]['year'].iloc[0])})"
                if pd.notna(filtered[filtered['movieId'] == mid]['year'].iloc[0])
                else filtered[filtered['movieId'] == mid]['clean_title'].iloc[0]
            ),
            key="new_user_movie_selectbox",
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            rating_value = st.slider(
                "Ваша оценка", 0.5, 5.0, 4.0, step=0.5,
                key="new_user_rating_slider",
            )
        with c2:
            st.write("")  # вертикальный отступ
            st.write("")
            if st.button("Добавить", use_container_width=True):
                st.session_state.new_user_ratings[int(movie_choice)] = float(rating_value)
                st.rerun()

    # --- Блок 2: текущая история оценок ------------------------------------
    st.markdown("#### 2. Ваши оценки")
    if not st.session_state.new_user_ratings:
        st.info(
            "Вы пока ничего не оценили. Добавьте 5-10 фильмов, чтобы "
            "получить осмысленные рекомендации."
        )
    else:
        hist_rows = []
        for mid, rat in st.session_state.new_user_ratings.items():
            mrow = movies[movies["movieId"] == mid]
            if mrow.empty:
                continue
            mrow = mrow.iloc[0]
            hist_rows.append({
                "movieId":    int(mid),
                "Фильм":      mrow["clean_title"],
                "Год":        int(mrow["year"]) if pd.notna(mrow["year"]) else "-",
                "Жанры":      mrow["genres"],
                "Ваша оценка": float(rat),
            })
        if hist_rows:
            hist_df = pd.DataFrame(hist_rows)
            st.dataframe(hist_df.drop(columns=["movieId"]),
                         use_container_width=True, hide_index=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Очистить все оценки"):
                    st.session_state.new_user_ratings = {}
                    st.rerun()
            with c2:
                remove_id = st.selectbox(
                    "Удалить одну оценку",
                    options=[None] + [r["movieId"] for r in hist_rows],
                    format_func=lambda x: "(не удалять)" if x is None else
                        movies[movies["movieId"] == x]["clean_title"].iloc[0],
                )
                if remove_id is not None and st.button("Удалить выбранное"):
                    st.session_state.new_user_ratings.pop(int(remove_id), None)
                    st.rerun()

    st.divider()

    # --- Блок 3: модель и количество ---------------------------------------
    st.markdown("#### 3. Выберите модель и получите рекомендации")
    st.info(
        "Для нового пользователя доступны: **ансамбль ALS+SVD** (NCF в "
        "холодном старте не используется - без обученного user-embedding "
        "сеть не может построить осмысленные предсказания), **ALS** "
        "(fold-in через item-факторы), **SVD** (приближённый fold-in "
        "через гребневую регрессию по факторам), **Popularity** "
        "(не использует историю - возвращает популярные фильмы)."
    )

    model_options = []
    if avail.get("als") and avail.get("svd"):
        model_options.append(("ensemble", "Ансамбль ALS + SVD (рекомендуется)"))
    if avail.get("als"):
        model_options.append(("als", "ALS (fold-in)"))
    if avail.get("svd"):
        model_options.append(("svd", "SVD (fold-in)"))
    model_options.append(("popularity", "Popularity (baseline, без вашей истории)"))

    c1, c2 = st.columns([2, 1])
    with c1:
        model_choice = st.radio(
            "Модель",
            options=[k for k, _ in model_options],
            format_func=lambda k: dict(model_options)[k],
        )
    with c2:
        top_n = st.slider("Сколько рекомендаций", 5, 20, 10,
                          key="new_user_topn")

    can_run = (model_choice == "popularity" or
               len(st.session_state.new_user_ratings) >= 1)
    if not can_run:
        st.warning(
            "Чтобы получить персональные рекомендации, поставьте хотя "
            "бы одну оценку (а лучше 5-10)."
        )

    if st.button("Получить рекомендации",
                 type="primary", use_container_width=True,
                 disabled=not can_run):
        with st.spinner("Считаем рекомендации..."):
            pairs = _dispatch_new_user(state, model_choice, top_n)

        if model_choice == "popularity":
            score_label = "popularity score"
        else:
            score_label = f"{model_choice} score"

        # Fallback: если выбранная модель не смогла вернуть рекомендации
        # (например, все оценки <= 3.0 для ALS), показываем popularity и
        # объясняем пользователю, почему.
        if not pairs and model_choice != "popularity":
            st.warning(
                "Выбранная модель не смогла построить персональные "
                "рекомендации (возможно, у вас слишком мало позитивных "
                "оценок). Показываем popularity-fallback - универсально "
                "популярные фильмы, которые вы ещё не оценивали."
            )
            pairs = recommend_popularity(
                state["movie_stats"],
                exclude_movie_ids=set(st.session_state.new_user_ratings.keys()),
                top_n=top_n,
            )
            score_label = "popularity score (fallback)"

        st.markdown(f"#### Топ-{top_n} рекомендаций")
        _render_recommendations(pairs, movies, movie_stats,
                                score_label=score_label)


def _dispatch_new_user(state: dict, model_key: str,
                       top_n: int) -> list[tuple[int, float]]:
    """Вызывает функцию инференса для нового пользователя."""
    id_maps = state["id_maps"]
    movie_stats = state["movie_stats"]
    user_ratings = st.session_state.new_user_ratings

    if model_key == "popularity":
        return recommend_popularity(movie_stats,
                                    exclude_movie_ids=set(user_ratings.keys()),
                                    top_n=top_n)
    if model_key == "als":
        return recommend_als_new_user(
            load_als_model(), id_maps, user_ratings, top_n
        )
    if model_key == "svd":
        return recommend_svd_new_user(
            load_svd_model(), user_ratings, movie_stats, top_n
        )
    if model_key == "ensemble":
        params = state["params"].get("ensemble", {}) or {}
        weights = _extract_ensemble_weights(params)
        # NCF убираем (новый пользователь). Перенормируем веса.
        weights_no_ncf = {k: v for k, v in weights.items() if k != "ncf"}
        s = sum(weights_no_ncf.values())
        if s > 0:
            weights_no_ncf = {k: v / s for k, v in weights_no_ncf.items()}
        return recommend_ensemble_new_user(
            user_ratings, load_als_model(), load_svd_model(),
            id_maps, top_n, weights_no_ncf,
        )
    return []


# ---------------------------------------------------------------------------
# Главная функция-роутер вкладки "Получить рекомендации"
# ---------------------------------------------------------------------------

def render_recommend(state: dict) -> None:
    st.title("Получить рекомендации")
    st.caption(
        "Два режима: для существующего пользователя из датасета "
        "(модели используют его историю) и для нового пользователя "
        "(вы сами ставите оценки - алгоритмы fold-in делают остальное)."
    )

    mode = st.radio(
        "Режим",
        ["Существующий пользователь", "Новый пользователь (поставлю свои оценки)"],
        horizontal=True,
    )
    st.divider()
    if mode == "Существующий пользователь":
        render_mode_existing(state)
    else:
        render_mode_new_user(state)



# ===========================================================================
# Главная точка входа
# ===========================================================================

def main() -> None:
    """Главная функция приложения: прелоад -> сайдбар -> роутинг вкладок."""
    try:
        state = ensure_state_loaded()
    except Exception as exc:
        st.error(
            "Критическая ошибка на этапе загрузки артефактов. "
            "Проверьте, что приложение запущено из корня проекта и что "
            "папки data/ и models/ существуют."
        )
        with st.expander("Traceback (для отладки)"):
            st.code(traceback.format_exc())
        return

    choice = render_sidebar(state)

    # Простой роутер.
    if choice == "Главная":
        render_home(state)
    elif choice == "Анализ данных":
        render_eda(state)
    elif choice == "Сравнение моделей":
        render_compare(state)
    elif choice == "Детали моделей":
        render_model_details(state)
    elif choice == "Получить рекомендации":
        render_recommend(state)
    elif choice == "Метрики и теория":
        render_theory(state)


if __name__ == "__main__":
    main()