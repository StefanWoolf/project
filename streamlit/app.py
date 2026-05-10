"""
Streamlit-приложение — Рекомендательная система фильмов (MovieLens).
Запуск: streamlit run streamlit/app.py  (из корня проекта)
"""
import sys
from pathlib import Path
from typing import Optional

# Делаем src импортируемым
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import sparse

from src.utils import SEED

MODELS_DIR    = PROJECT_ROOT / 'models'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
RAW_DIR       = PROJECT_ROOT / 'data' / 'raw'

st.set_page_config(
    page_title='MovieLens Recommender',
    page_icon='🎬',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ──────────────────────────────────────────────────────────────────────────────
# Блок 1. Кэшируемые загрузчики
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner='Загрузка датасетов...')
def load_data():
    """Загрузить все DataFrame, нужные приложению."""
    train           = pd.read_parquet(PROCESSED_DIR / 'train.parquet')
    val             = pd.read_parquet(PROCESSED_DIR / 'val.parquet')
    test            = pd.read_parquet(PROCESSED_DIR / 'test.parquet')
    movies_enriched = pd.read_parquet(PROCESSED_DIR / 'movies_enriched.parquet')
    links           = pd.read_csv(RAW_DIR / 'links.csv')
    metrics_summary = pd.read_parquet(MODELS_DIR / 'metrics_summary.parquet')
    return {
        'train': train, 'val': val, 'test': test,
        'movies_enriched': movies_enriched,
        'links': links,
        'metrics_summary': metrics_summary,
    }


# ── Заглушки классов для корректной десериализации joblib ─────────────────
# Popularity и GlobalMean были обучены в ноутбуке (04_model_popularity.ipynb),
# где они определены как локальные классы в __main__. При загрузке через joblib
# в Streamlit Python ищет эти классы в текущем модуле — поэтому объявляем их здесь.

class GlobalMeanModel:
    """Stub-класс для десериализации GlobalMeanModel из joblib."""
    def __init__(self):
        self.global_mean_: float = 0.0

    def fit(self, train_df, rating_col='rating'):
        self.global_mean_ = float(train_df[rating_col].mean())
        return self

    def predict(self, user_ids, movie_ids):
        import numpy as _np
        return _np.full(len(user_ids), self.global_mean_, dtype=float)


class PopularityRecommender:
    """Stub-класс для десериализации PopularityRecommender из joblib."""
    def __init__(self, m: float = 10.0):
        self.m = m
        self.global_mean_: float = 0.0
        self.scores_ = None        # pd.Series index=movieId, value=score
        self.user_seen_: dict = {}

    def fit(self, train_df, user_col='userId', item_col='movieId',
            rating_col='rating'):
        import pandas as _pd
        import numpy as _np
        self.global_mean_ = float(train_df[rating_col].mean())
        agg = train_df.groupby(item_col)[rating_col].agg(['count', 'mean'])
        n, mu, C = agg['count'], agg['mean'], self.global_mean_
        self.scores_ = ((n / (n + self.m)) * mu
                        + (self.m / (n + self.m)) * C
                        ).sort_values(ascending=False)
        self.user_seen_ = (
            train_df.groupby(user_col)[item_col].apply(set).to_dict()
        )
        return self

    def recommend(self, user_ids, k=10, exclude_seen=True):
        ranked = self.scores_.index.values
        result = {}
        for u in user_ids:
            if exclude_seen:
                seen = self.user_seen_.get(u, set())
                recs = [m for m in ranked if m not in seen][:k]
            else:
                recs = list(ranked[:k])
            result[u] = recs
        return result


@st.cache_resource(show_spinner='Загрузка ID-мап...')
def load_id_maps():
    return {
        'user_id_map':      joblib.load(MODELS_DIR / 'user_id_map.pkl'),
        'movie_id_map':     joblib.load(MODELS_DIR / 'movie_id_map.pkl'),
        'inv_user_id_map':  joblib.load(MODELS_DIR / 'inv_user_id_map.pkl'),
        'inv_movie_id_map': joblib.load(MODELS_DIR / 'inv_movie_id_map.pkl'),
    }


@st.cache_resource(show_spinner='Загрузка моделей...')
def load_models():
    """Загрузить все модели. Если какая-то не подгружается — пропустить с warning."""
    models = {}
    errors = {}

    def _try(name, loader):
        try:
            models[name] = loader()
        except Exception as e:
            errors[name] = str(e)

    _try('Popularity', lambda: joblib.load(MODELS_DIR / 'popularity_model.pkl'))
    _try('GlobalMean', lambda: joblib.load(MODELS_DIR / 'global_mean_model.pkl'))
    _try('SVD',        lambda: joblib.load(MODELS_DIR / 'svd_model.pkl'))
    _try('KNN',        lambda: joblib.load(MODELS_DIR / 'knn_model.pkl'))
    _try('LightGBM',   lambda: joblib.load(MODELS_DIR / 'lightgbm_model.pkl'))

    # ALS (hybrid) — пробуем als_model.pkl, потом lightfm_model.pkl
    def _load_als():
        for p in [MODELS_DIR / 'als_model.pkl', MODELS_DIR / 'lightfm_model.pkl']:
            if p.exists():
                return joblib.load(p)
        raise FileNotFoundError('Файл ALS-модели не найден')
    _try('ALS (hybrid)', _load_als)

    # NCF — формат Keras
    def _load_ncf():
        import tensorflow as tf
        return tf.keras.models.load_model(MODELS_DIR / 'ncf_model.keras')
    _try('NCF', _load_ncf)

    return models, errors


@st.cache_resource(show_spinner='Загрузка артефактов...')
def load_artifacts():
    """Скейлеры, энкодеры, sparse-матрицы для feature-моделей и ALS."""
    out = {}
    out['genre_encoder'] = joblib.load(MODELS_DIR / 'genre_encoder.pkl')
    out['tfidf_tags']    = joblib.load(MODELS_DIR / 'tfidf_tags.pkl')
    out['movie_scaler']  = joblib.load(MODELS_DIR / 'movie_scaler.pkl')
    out['user_scaler']   = joblib.load(MODELS_DIR / 'user_scaler.pkl')

    if (MODELS_DIR / 'tag_svd.pkl').exists():
        out['tag_svd'] = joblib.load(MODELS_DIR / 'tag_svd.pkl')

    out['movie_features'] = pd.read_parquet(PROCESSED_DIR / 'movie_features.parquet')
    out['user_features']  = pd.read_parquet(PROCESSED_DIR / 'user_features.parquet')
    out['genre_features'] = pd.read_parquet(PROCESSED_DIR / 'genre_features.parquet')

    if (MODELS_DIR / 'lightfm_item_features.npz').exists():
        out['als_item_features'] = sparse.load_npz(
            MODELS_DIR / 'lightfm_item_features.npz'
        )

    out['tag_features'] = sparse.load_npz(PROCESSED_DIR / 'tag_features.npz')
    out['tag_order']    = pd.read_parquet(PROCESSED_DIR / 'tag_movie_order.parquet')

    return out


@st.cache_data
def load_metrics_jsons():
    """JSON-метрики и параметры всех моделей."""
    out = {}
    for name in ['popularity', 'svd', 'knn', 'lightgbm', 'lightfm', 'ncf']:
        for suffix in ['metrics', 'params']:
            p = MODELS_DIR / f'{name}_{suffix}.json'
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    out[f'{name}_{suffix}'] = json.load(f)
    if (MODELS_DIR / 'best_model_decision.json').exists():
        with open(MODELS_DIR / 'best_model_decision.json', 'r', encoding='utf-8') as f:
            out['best_decision'] = json.load(f)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Блок 2. Inference-хелперы
# ──────────────────────────────────────────────────────────────────────────────

def _genre_filtered_popularity(top_genres: set, popularity_model,
                               movies_df: pd.DataFrame, k: int = 10,
                               exclude_movie_ids: Optional[set] = None) -> list:
    """Топ-K популярных фильмов, отфильтрованных по жанрам пользователя."""
    exclude_movie_ids = exclude_movie_ids or set()
    ranked = popularity_model.scores_.sort_values(ascending=False).index
    out = []
    for mid in ranked:
        if mid in exclude_movie_ids:
            continue
        row = movies_df[movies_df['movieId'] == mid]
        if row.empty:
            continue
        genres = set(row.iloc[0].get('genres_list', []))
        if not top_genres or (genres & top_genres):
            out.append(int(mid))
        if len(out) >= k:
            break
    return out


def _als_recommend_new_user(als_model, n_users: int, n_movies: int,
                             user_ratings_dict: dict, movie_id_map: dict,
                             inv_movie_id_map: dict, k: int = 10) -> list:
    """Fold-in для нового пользователя через Implicit ALS (recalculate_user)."""
    cols, data = [], []
    for raw_mid, rating_val in user_ratings_dict.items():
        if raw_mid not in movie_id_map:
            continue
        cols.append(movie_id_map[raw_mid])
        data.append(float(rating_val))
    if not cols:
        return []
    user_row = sparse.csr_matrix(
        (data, ([0] * len(cols), cols)),
        shape=(1, n_movies),
    )
    try:
        item_ids, _ = als_model.recommend(
            userid=0, user_items=user_row, N=k,
            filter_already_liked_items=True, recalculate_user=True,
        )
    except TypeError:
        # Старая версия implicit без recalculate_user
        item_ids, _ = als_model.recommend(
            userid=0, user_items=user_row, N=k,
            filter_already_liked_items=True,
        )
    return [int(inv_movie_id_map[int(i)]) for i in item_ids]


def _lightgbm_recommend_new_user(lgbm_model, user_ratings_dict: dict,
                                  movies_enriched: pd.DataFrame,
                                  artifacts: dict, k: int = 10) -> list:
    """Inference LightGBM для нового пользователя."""
    if not user_ratings_dict:
        return []

    rated_ratings = np.array(list(user_ratings_dict.values()), dtype=float)
    user_n    = len(rated_ratings)
    user_mean = float(rated_ratings.mean())
    user_std  = float(rated_ratings.std()) if user_n > 1 else 0.0
    user_days = 0.0   # для нового юзера временной разброс = 0

    raw = np.array([[user_n, user_mean, user_std, user_days]])
    user_vec  = artifacts['user_scaler'].transform(raw).flatten()
    user_cols = ['user_n_ratings_train', 'user_mean_rating_train',
                 'user_std_rating_train', 'user_active_days_train']

    movie_feats  = artifacts['movie_features']
    genre_feats  = artifacts['genre_features']
    rated_set    = set(user_ratings_dict.keys())
    candidates   = np.array([m for m in movies_enriched['movieId'].values
                              if m not in rated_set])

    user_block  = pd.DataFrame(
        np.tile(user_vec, (len(candidates), 1)),
        columns=user_cols, index=candidates,
    )
    movie_block = movie_feats.set_index('movieId').reindex(candidates).fillna(0)
    genre_block = genre_feats.set_index('movieId').reindex(candidates).fillna(0)
    parts       = [user_block, movie_block, genre_block]

    if 'tag_svd' in artifacts:
        tag_svd   = artifacts['tag_svd']
        tag_matrix = artifacts['tag_features']
        tag_order  = artifacts['tag_order']
        tag_idx    = tag_order.set_index('movieId')['tag_row_idx'].reindex(candidates).fillna(0).astype(int).values
        tag_proj   = tag_svd.transform(tag_matrix[tag_idx])
        tag_block  = pd.DataFrame(
            tag_proj,
            columns=[f'tag_svd_{i}' for i in range(tag_proj.shape[1])],
            index=candidates,
        )
        parts.append(tag_block)

    X = pd.concat(parts, axis=1).fillna(0.0)

    # Согласовать порядок колонок с обученной моделью
    feat_names = None
    if hasattr(lgbm_model, 'feature_name_'):
        feat_names = lgbm_model.feature_name_
    elif hasattr(lgbm_model, 'booster_'):
        try:
            feat_names = lgbm_model.booster_.feature_name()
        except Exception:
            pass
    if feat_names is not None:
        cols_ok = [c for c in feat_names if c in X.columns]
        X = X.reindex(columns=feat_names, fill_value=0.0)

    scores    = lgbm_model.predict(X.values)
    top_k_idx = np.argsort(-scores)[:k]
    return [int(candidates[i]) for i in top_k_idx]


def compute_recommendations(model_name: str, user_ratings_dict: dict,
                            models: dict, artifacts: dict, maps: dict,
                            movies_enriched: pd.DataFrame, k: int = 10) -> list:
    """Главный диспетчер: для каждой модели выбирает путь inference."""
    movie_id_map     = maps['movie_id_map']
    inv_movie_id_map = maps['inv_movie_id_map']
    n_users          = len(maps['user_id_map'])
    n_movies_total   = len(movie_id_map)

    rated_ids     = set(user_ratings_dict.keys())
    high_rated    = {mid for mid, r in user_ratings_dict.items() if r >= 4.0}
    user_genres   = set()
    for mid in (high_rated or rated_ids):
        row = movies_enriched[movies_enriched['movieId'] == mid]
        if not row.empty and 'genres_list' in movies_enriched.columns:
            for g in row.iloc[0].get('genres_list', []):
                user_genres.add(g)

    pop_model = models.get('Popularity')

    if model_name == 'Popularity':
        if pop_model is None:
            return []
        return _genre_filtered_popularity(user_genres, pop_model,
                                          movies_enriched, k, rated_ids)

    if model_name == 'ALS (hybrid)':
        return _als_recommend_new_user(
            models['ALS (hybrid)'], n_users, n_movies_total,
            user_ratings_dict, movie_id_map, inv_movie_id_map, k,
        )

    if model_name == 'LightGBM':
        return _lightgbm_recommend_new_user(
            models['LightGBM'], user_ratings_dict,
            movies_enriched, artifacts, k,
        )

    # SVD / KNN / NCF → fallback: жанрово-отфильтрованный Popularity
    if pop_model is not None:
        return _genre_filtered_popularity(user_genres, pop_model,
                                          movies_enriched, k, rated_ids)
    return []


@st.cache_data(show_spinner=False)
def _get_tmdb_api_key() -> Optional[str]:
    """Читаем TMDB_API_KEY только если secrets.toml реально существует."""
    secrets_paths = [
        Path.home() / '.streamlit' / 'secrets.toml',
        PROJECT_ROOT / '.streamlit' / 'secrets.toml',
    ]
    if not any(p.exists() for p in secrets_paths):
        return None          # файла нет — не трогаем st.secrets вообще
    try:
        return st.secrets.get('TMDB_API_KEY')
    except Exception:
        return None


def _tmdb_poster_url(movie_id: int, _links_hash: str) -> Optional[str]:
    """Постер через TMDb API. Требует TMDB_API_KEY в st.secrets."""
    api_key = _get_tmdb_api_key()
    if not api_key:
        return None
    links = pd.read_csv(RAW_DIR / 'links.csv')
    row = links[links['movieId'] == movie_id]
    if row.empty or pd.isna(row.iloc[0]['tmdbId']):
        return None
    tmdb_id = int(row.iloc[0]['tmdbId'])
    try:
        import requests
        resp = requests.get(
            f'https://api.themoviedb.org/3/movie/{tmdb_id}',
            params={'api_key': api_key},
            timeout=3,
        )
        if resp.status_code != 200:
            return None
        poster_path = resp.json().get('poster_path')
        return f'https://image.tmdb.org/t/p/w300{poster_path}' if poster_path else None
    except Exception:
        return None


def render_recommendation_cards(rec_movie_ids: list, movies_enriched: pd.DataFrame,
                                 links: pd.DataFrame):
    """Отрисовка карточек рекомендованных фильмов."""
    cols_per_row = 5
    links_hash   = str(len(links))  # хэш для cache_data
    for r_idx in range(0, len(rec_movie_ids), cols_per_row):
        row_ids = rec_movie_ids[r_idx:r_idx + cols_per_row]
        cols    = st.columns(cols_per_row)
        for col, mid in zip(cols, row_ids):
            mrow = movies_enriched[movies_enriched['movieId'] == mid]
            if mrow.empty:
                continue
            mrow = mrow.iloc[0]
            with col:
                poster_url = _tmdb_poster_url(mid, links_hash)
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.markdown('🎞')
                st.markdown(f"**{mrow['title']}**")
                st.caption(f"_{mrow['genres']}_")


# ──────────────────────────────────────────────────────────────────────────────
# Блок 3. Страница «🏠 Главная»
# ──────────────────────────────────────────────────────────────────────────────

def render_home():
    st.title('🎬 Рекомендательная система фильмов')
    st.markdown(
        'Курсовой проект по теме **«Рекомендательные системы и библиотеки '
        'RePlay, TensorFlow, Sklearn»**. Обучено **7 моделей** разных семейств, '
        'сравнено их качество на тестовой выборке.'
    )

    data   = load_data()
    train  = data['train']
    movies = data['movies_enriched']
    links  = data['links']

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Пользователей',      f"{train['userId'].nunique():,}")
    col2.metric('Фильмов в каталоге', f"{movies.shape[0]:,}")
    col3.metric('Оценок (train)',      f"{len(train):,}")
    col4.metric('Связок с TMDb',       f"{links['tmdbId'].notna().sum():,}")

    st.divider()

    st.subheader('Состав моделей')
    families = pd.DataFrame([
        {'Модель': 'GlobalMean',   'Семейство': 'Baseline',      'Назначение': 'предсказание рейтинга'},
        {'Модель': 'Popularity',   'Семейство': 'Baseline',      'Назначение': 'top-N ранжирование'},
        {'Модель': 'SVD',          'Семейство': 'Collaborative', 'Назначение': 'и то, и другое'},
        {'Модель': 'KNN',          'Семейство': 'Collaborative', 'Назначение': 'и то, и другое'},
        {'Модель': 'LightGBM',     'Семейство': 'Feature-based', 'Назначение': 'и то, и другое'},
        {'Модель': 'ALS (hybrid)', 'Семейство': 'Hybrid',        'Назначение': 'top-N ранжирование'},
        {'Модель': 'NCF',          'Семейство': 'Neural',        'Назначение': 'и то, и другое'},
    ])
    st.dataframe(families, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader('Стек технологий')
    st.markdown(
        '- **Surprise** — SVD, KNN (коллаборативная фильтрация)\n'
        '- **LightGBM** — feature-based бустинг (жанры, теги, агрегаты из train)\n'
        '- **Implicit (ALS)** — гибридная факторизация с контентной инициализацией\n'
        '- **TensorFlow / Keras** — Neural Collaborative Filtering (NeuMF)\n'
        '- **Optuna** — автоматический подбор гиперпараметров (≥ 30 trials)\n'
        '- **Streamlit + Plotly** — интерактивный отчёт и инференс'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Блок 4. Страница «📊 EDA»
# ──────────────────────────────────────────────────────────────────────────────

def render_eda():
    st.title('📊 Разведочный анализ данных')
    data   = load_data()
    train  = data['train']
    movies = data['movies_enriched']

    tab1, tab2, tab3, tab4 = st.tabs([
        'Распределение оценок',
        'Активность пользователей',
        'Фильмы и годы',
        'Жанры и теги',
    ])

    with tab1:
        st.subheader('Распределение оценок (train)')
        fig = px.histogram(
            train, x='rating', nbins=10,
            title='Гистограмма рейтингов 0.5 – 5.0',
            color_discrete_sequence=['#1f77b4'],
        )
        fig.update_layout(bargap=0.05, height=420)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric('Средний рейтинг',   f"{train['rating'].mean():.3f}")
        col_b.metric('Медианный рейтинг', f"{train['rating'].median():.1f}")
        col_c.metric('Оценок ≥ 4.0',
                     f"{(train['rating'] >= 4.0).sum():,} "
                     f"({(train['rating'] >= 4.0).mean()*100:.1f}%)")

    with tab2:
        st.subheader('Активность пользователей')
        user_counts = train.groupby('userId').size().rename('n_ratings').reset_index()
        fig = px.histogram(
            user_counts, x='n_ratings', nbins=50, log_y=True,
            title='Число оценок на пользователя (log Y)',
            color_discrete_sequence=['#ff7f0e'],
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Минимум: {user_counts['n_ratings'].min()} | "
            f"Медиана: {int(user_counts['n_ratings'].median())} | "
            f"Максимум: {user_counts['n_ratings'].max()}"
        )

        st.subheader('Активность фильмов')
        movie_counts = train.groupby('movieId').size().rename('n_ratings').reset_index()
        fig2 = px.histogram(
            movie_counts, x='n_ratings', nbins=50, log_y=True,
            title='Число оценок на фильм (log Y)',
            color_discrete_sequence=['#2ca02c'],
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)
        one_rating = (movie_counts['n_ratings'] == 1).sum()
        st.caption(
            f"Фильмов с ровно 1 оценкой (long tail): {one_rating:,} | "
            f"Фильмов с ≥ 50 оценками: {(movie_counts['n_ratings'] >= 50).sum():,}"
        )

    with tab3:
        st.subheader('Распределение года выпуска фильмов')
        years = movies['year'].dropna().astype(int)
        fig = px.histogram(years, nbins=40, title='Год выпуска (по каталогу)',
                           color_discrete_sequence=['#9467bd'])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Число оценок по году выпуска фильма (train)')
        train_year = train.merge(movies[['movieId', 'year']], on='movieId', how='left')
        by_year    = (
            train_year.dropna(subset=['year'])
            .groupby(train_year['year'].dropna().astype(int))
            .size().reset_index(name='n')
        )
        fig2 = px.line(by_year, x='year', y='n',
                       title='Оценок на год выпуска фильма',
                       color_discrete_sequence=['#1f77b4'])
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.subheader('Топ-15 жанров')
        all_genres  = movies['genres_list'].explode()
        top_genres  = all_genres.value_counts().head(15).reset_index()
        top_genres.columns = ['genre', 'count']
        fig = px.bar(
            top_genres, x='count', y='genre', orientation='h',
            title='Самые популярные жанры (число фильмов)',
            color='count', color_continuous_scale='Blues',
        )
        fig.update_layout(height=520, yaxis={'categoryorder': 'total ascending'},
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Теги')
        with_tags = int((movies['n_tags'] > 0).sum())
        col_a, col_b = st.columns(2)
        col_a.metric('Фильмов с тегами', f'{with_tags:,} из {len(movies):,}')
        col_b.metric('Покрытие тегами',
                     f'{with_tags / len(movies) * 100:.1f}%')

        no_genre = int(movies['no_genres_flag'].sum())
        st.caption(f"Фильмов без жанра ('(no genres listed)'): {no_genre}")


# ──────────────────────────────────────────────────────────────────────────────
# Блок 5. Страница «📈 Метрики моделей»
# ──────────────────────────────────────────────────────────────────────────────

def render_metrics():
    st.title('📈 Сравнение моделей')
    data    = load_data()
    summary = data['metrics_summary']
    jsons   = load_metrics_jsons()

    st.subheader('Сводная таблица метрик (test)')
    show_cols = [c for c in [
        'model', 'family', 'rmse', 'mae',
        'ndcg@10', 'precision@10', 'recall@10', 'hit_rate@10',
        'coverage@20', 'final_train_time_sec', 'optuna_search_time_sec',
    ] if c in summary.columns]
    st.dataframe(summary[show_cols].round(4),
                 hide_index=True, use_container_width=True)

    if 'best_decision' in jsons:
        bd = jsons['best_decision']
        st.success(
            f"**Дефолтная модель для рекомендаций:** "
            f"`{bd['streamlit_default']['recommendation_model']}` — "
            f"{bd['streamlit_default']['rationale']}"
        )

    st.divider()

    tab1, tab2, tab3 = st.tabs(['Bar charts', 'Radar chart', 'Качество vs время'])

    with tab1:
        st.subheader('Сравнение по одной метрике')
        metric_choice = st.selectbox(
            'Метрика',
            ['ndcg@10', 'precision@10', 'recall@10', 'hit_rate@10',
             'coverage@20', 'rmse', 'mae'],
        )
        sub_df    = summary.dropna(subset=[metric_choice]).copy()
        ascending = metric_choice in ['rmse', 'mae']
        sub_df    = sub_df.sort_values(metric_choice, ascending=ascending)
        fig = px.bar(
            sub_df, x='model', y=metric_choice,
            color='family',
            text=sub_df[metric_choice].round(4),
            title=f'{metric_choice} (на test)',
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader('Radar chart (нормализованные top-N метрики)')
        radar_metrics = ['ndcg@10', 'precision@10', 'recall@10',
                         'hit_rate@10', 'coverage@20']
        radar_df = summary.dropna(subset=radar_metrics).copy().reset_index(drop=True)
        norm     = radar_df[radar_metrics].copy()
        for c in radar_metrics:
            mn, mx = norm[c].min(), norm[c].max()
            norm[c] = (norm[c] - mn) / (mx - mn) if mx > mn else 0.5
        fig = go.Figure()
        for idx, model_name in enumerate(radar_df['model'].values):
            vals = norm.iloc[idx].values.tolist()
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself', name=model_name, opacity=0.55,
            ))
        fig.update_layout(
            title='Профиль моделей (нормализованные top-N метрики)',
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=560,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader('NDCG@10 vs время обучения')
        sc_df = summary.dropna(subset=['ndcg@10']).copy()
        sc_df['total_time_sec'] = (
            sc_df['final_train_time_sec'].fillna(0) +
            sc_df['optuna_search_time_sec'].fillna(0)
        )
        sc_df = sc_df[sc_df['total_time_sec'] > 0]
        if not sc_df.empty:
            fig = px.scatter(
                sc_df, x='total_time_sec', y='ndcg@10',
                text='model', color='family',
                size='coverage@20', log_x=True,
                title='NDCG@10 vs полное время обучения (log scale)',
                labels={'total_time_sec': 'Время (Optuna + финального обучения), сек'},
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Нет данных о времени обучения (запустите ноутбуки 4–9 полностью).')

    st.divider()
    with st.expander('🔧 Лучшие гиперпараметры (Optuna)'):
        name_map = {
            'svd': 'SVD', 'knn': 'KNN', 'lightgbm': 'LightGBM',
            'lightfm': 'ALS (hybrid)', 'ncf': 'NCF',
        }
        for short, disp in name_map.items():
            key = f'{short}_params'
            if key in jsons and jsons[key].get('best_params'):
                st.markdown(f'**{disp}**')
                st.json(jsons[key]['best_params'])


# ──────────────────────────────────────────────────────────────────────────────
# Блок 6. Страница «🎬 Рекомендации»
# ──────────────────────────────────────────────────────────────────────────────

def render_recommendations():
    st.title('🎬 Получить рекомендации')
    st.markdown(
        'Выставите оценки нескольким известным вам фильмам (минимум 5 штук), '
        'выберите модель — и получите топ рекомендованных фильмов на основе вашей истории.'
    )

    data      = load_data()
    movies    = data['movies_enriched']
    links     = data['links']
    train     = data['train']

    models, errors = load_models()
    artifacts = load_artifacts()
    maps      = load_id_maps()

    if errors:
        with st.expander('⚠ Незагруженные модели'):
            for name, err in errors.items():
                st.warning(f'**{name}**: {err}')

    # ── 6.1 Поиск и добавление оценок ────────────────────────────────────────
    st.subheader('1. Ваши оценки')
    if 'user_ratings' not in st.session_state:
        st.session_state['user_ratings'] = {}

    col_search, col_rating = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            'Поиск фильма по названию',
            placeholder='Например: Toy Story',
        )
    with col_rating:
        rating_value = st.select_slider(
            'Оценка',
            options=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            value=4.0,
        )

    if search_query.strip():
        matches = movies[
            movies['title'].str.contains(search_query, case=False, na=False)
        ].head(20)
        if matches.empty:
            st.info('Ничего не найдено — попробуйте другой запрос.')
        else:
            for _, row in matches.iterrows():
                c1, c2, c3 = st.columns([5, 2, 1])
                c1.write(f"**{row['title']}** — _{row['genres']}_")
                c2.caption(
                    f"movieId={row['movieId']}, "
                    f"год={row.get('year') or '?'}"
                )
                if c3.button('+ Добавить', key=f'add_{row["movieId"]}'):
                    st.session_state['user_ratings'][int(row['movieId'])] = float(rating_value)
                    st.rerun()

    # ── Текущий список оценок ─────────────────────────────────────────────────
    st.markdown('**Ваш список оценённых фильмов:**')
    if not st.session_state['user_ratings']:
        st.caption('Список пуст. Найдите и добавьте хотя бы 5 фильмов.')
    else:
        for mid, rval in list(st.session_state['user_ratings'].items()):
            title_row = movies[movies['movieId'] == mid]
            title = title_row.iloc[0]['title'] if not title_row.empty else f'movieId={mid}'
            c1, c2 = st.columns([8, 1])
            c1.write(f"⭐ {rval:.1f} — {title}")
            if c2.button('✕', key=f'del_{mid}'):
                st.session_state['user_ratings'].pop(mid, None)
                st.rerun()

        if st.button('🗑 Очистить список', type='secondary'):
            st.session_state['user_ratings'] = {}
            st.rerun()

    st.divider()

    # ── 6.2 Выбор модели ──────────────────────────────────────────────────────
    st.subheader('2. Выбор модели')

    # GlobalMean не выдаёт топ-N → убираем из списка
    topn_models = [m for m in models.keys() if m != 'GlobalMean']
    if not topn_models:
        st.error('Ни одна модель для топ-N не загрузилась.')
        return

    foldin_capable = {'ALS (hybrid)', 'LightGBM', 'Popularity'}
    model_labels   = []
    for m in topn_models:
        if m in foldin_capable:
            model_labels.append(f'✅ {m}')
        else:
            model_labels.append(f'⚠ {m}  (fallback: популярность по жанрам)')

    chosen_label      = st.selectbox(
        'Выберите модель',
        model_labels,
        help='✅ — корректный inference для нового пользователя; '
             '⚠ — фолбэк на популярность, отфильтрованную по жанрам.',
    )
    chosen_model_name = (chosen_label
                         .replace('✅ ', '')
                         .replace('⚠ ', '')
                         .split('  (fallback')[0])

    n_recs = st.slider('Число рекомендаций', min_value=5, max_value=20, value=10)

    n_rated      = len(st.session_state['user_ratings'])
    can_recommend = n_rated >= 5

    if not can_recommend:
        st.info(f'Добавьте ещё {5 - n_rated} фильмов, чтобы разблокировать кнопку.')

    if st.button('🚀 Получить рекомендации', type='primary', disabled=not can_recommend):
        with st.spinner('Считаем рекомендации...'):
            recs = compute_recommendations(
                chosen_model_name,
                st.session_state['user_ratings'],
                models, artifacts, maps, movies,
                k=n_recs,
            )
        if not recs:
            st.warning('Не удалось получить рекомендации. Попробуйте другую модель.')
        else:
            st.session_state['last_recs']  = recs
            st.session_state['last_model'] = chosen_model_name

    # ── 6.3 Показ рекомендаций ────────────────────────────────────────────────
    if 'last_recs' in st.session_state and st.session_state['last_recs']:
        st.divider()
        model_shown = st.session_state['last_model']
        fold_in_ok  = model_shown in foldin_capable
        st.subheader(
            f"3. Топ-{len(st.session_state['last_recs'])} от «{model_shown}» "
            f"{'✅' if fold_in_ok else '⚠'}"
        )
        if not fold_in_ok:
            st.caption(
                f'Модель {model_shown} не поддерживает fold-in для нового пользователя — '
                'показаны популярные фильмы, отфильтрованные по вашим жанровым предпочтениям.'
            )
        render_recommendation_cards(st.session_state['last_recs'], movies, links)

        # Краткая сводка рекомендованных фильмов
        with st.expander('Список рекомендаций (таблица)'):
            rec_info = movies[movies['movieId'].isin(st.session_state['last_recs'])][
                ['movieId', 'title', 'genres', 'year']
            ].copy()
            rec_info['rank'] = rec_info['movieId'].map(
                {mid: i + 1 for i, mid in enumerate(st.session_state['last_recs'])}
            )
            st.dataframe(
                rec_info.sort_values('rank')[['rank', 'title', 'genres', 'year']],
                hide_index=True, use_container_width=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Блок 7. Навигация и точка входа
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.title('🎬 MovieLens Recommender')
    st.sidebar.markdown(
        'Курсовой проект по машинному обучению.\n\n'
        'Датасет: **MovieLens ml-latest-small** (2018)'
    )

    page = st.sidebar.radio(
        'Раздел',
        ['🏠 Главная', '📊 EDA', '📈 Метрики моделей', '🎬 Рекомендации'],
    )

    st.sidebar.divider()
    st.sidebar.caption(f'SEED = {SEED}')

    if page == '🏠 Главная':
        render_home()
    elif page == '📊 EDA':
        render_eda()
    elif page == '📈 Метрики моделей':
        render_metrics()
    elif page == '🎬 Рекомендации':
        render_recommendations()


if __name__ == '__main__':
    main()