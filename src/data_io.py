from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse

try:
    # Когда файл импортируется как модуль
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    # Когда код выполняется в Jupyter (__file__ не определён)
    PROJECT_ROOT = Path.cwd().parent

PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'


def load_splits():
    return {
        'train': pd.read_parquet(PROCESSED_DIR / 'train.parquet'),
        'val': pd.read_parquet(PROCESSED_DIR / 'val.parquet'),
        'test': pd.read_parquet(PROCESSED_DIR / 'test.parquet'),
        'cold_start_eval': pd.read_parquet(PROCESSED_DIR / 'cold_start_eval.parquet'),
    }


def load_features():
    return {
        'movies': pd.read_parquet(PROCESSED_DIR / 'movie_features.parquet'),
        'users': pd.read_parquet(PROCESSED_DIR / 'user_features.parquet'),
        'genres': pd.read_parquet(PROCESSED_DIR / 'genre_features.parquet'),
        'movies_enriched': pd.read_parquet(PROCESSED_DIR / 'movies_enriched.parquet'),
    }


def load_id_maps():
    return {
        'user_id_map': joblib.load(MODELS_DIR / 'user_id_map.pkl'),
        'movie_id_map': joblib.load(MODELS_DIR / 'movie_id_map.pkl'),
        'inv_user_id_map': joblib.load(MODELS_DIR / 'inv_user_id_map.pkl'),
        'inv_movie_id_map': joblib.load(MODELS_DIR / 'inv_movie_id_map.pkl'),
    }


def load_user_item_matrix():
    return sparse.load_npz(PROCESSED_DIR / 'user_item_train.npz')


def load_tag_features():
    return {
        'matrix': sparse.load_npz(PROCESSED_DIR / 'tag_features.npz'),
        'order': pd.read_parquet(PROCESSED_DIR / 'tag_movie_order.parquet'),
    }


def load_encoders():
    return {
        'genre_encoder': joblib.load(MODELS_DIR / 'genre_encoder.pkl'),
        'tfidf_tags': joblib.load(MODELS_DIR / 'tfidf_tags.pkl'),
        'movie_scaler': joblib.load(MODELS_DIR / 'movie_scaler.pkl'),
        'user_scaler': joblib.load(MODELS_DIR / 'user_scaler.pkl'),
    }