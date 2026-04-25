"""Offline-препроцессинг и сохранение артефактов.

Этот модуль нужен, чтобы часть работы можно было выполнять заранее:
подготовить признаки фильмов, статистики рейтингов и TF-IDF матрицу.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import MovieLensData, PROCESSED_DIR, PROJECT_ROOT, ensure_data_dirs
from src.recommend import ContentFeatureMatrix, build_content_feature_matrix, build_feature_text


MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ArtifactPaths:
    """Пути к файлам, которые создает pipeline подготовки артефактов."""

    movies_features: Path
    rating_stats: Path
    tfidf_vectorizer: Path
    tfidf_matrix: Path


def default_artifact_paths(
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> ArtifactPaths:
    """Возвращает стандартные пути к локальным артефактам проекта."""
    return ArtifactPaths(
        movies_features=processed_dir / "movies_features.parquet",
        rating_stats=processed_dir / "rating_stats.parquet",
        tfidf_vectorizer=models_dir / "tfidf_vectorizer.joblib",
        tfidf_matrix=models_dir / "tfidf_matrix.npz",
    )


def create_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    """Добавляет текстовые признаки, которые потом используются content-based моделью."""
    result = movies.copy()
    result["genres"] = result["genres"].fillna("(no genres listed)")
    result["genres_text"] = result["genres"].str.replace("|", " ", regex=False)
    result["title_clean"] = result["title"].fillna("").str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    result["feature_text"] = build_feature_text(result)
    return result


def create_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Считает базовую статистику оценок по каждому фильму."""
    return (
        ratings.groupby("movieId", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
        .sort_values(["rating_count", "mean_rating"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_content_artifacts(movies_features: pd.DataFrame) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    """Обучает TF-IDF vectorizer и строит разреженную матрицу признаков фильмов."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(movies_features["feature_text"])
    return vectorizer, matrix


def _unique_movie_ids(movies: pd.DataFrame) -> list[int] | None:
    """Возвращает movieId в порядке DataFrame или None, если есть дубликаты."""
    movie_ids = movies["movieId"].astype(int).tolist()
    if len(movie_ids) != len(set(movie_ids)):
        return None
    return movie_ids


def load_content_feature_matrix_artifact(
    movies: pd.DataFrame,
    paths: ArtifactPaths | None = None,
) -> ContentFeatureMatrix | None:
    """Загружает TF-IDF матрицу с диска и выравнивает ее под переданный каталог.

    Возвращает None, если артефактов нет, они неполные или не покрывают нужные movieId.
    Такой контракт удобен для Streamlit: быстрый путь используется автоматически,
    а fallback остается обычной сборкой признаков на лету.
    """
    paths = paths or default_artifact_paths()
    if not paths.movies_features.exists() or not paths.tfidf_matrix.exists():
        return None

    requested_movie_ids = _unique_movie_ids(movies)
    if requested_movie_ids is None:
        return None

    movies_features = pd.read_parquet(paths.movies_features)
    if "movieId" not in movies_features.columns:
        return None

    artifact_movie_ids = _unique_movie_ids(movies_features)
    if artifact_movie_ids is None:
        return None

    matrix = sparse.load_npz(paths.tfidf_matrix).tocsr()
    if matrix.shape[0] != len(artifact_movie_ids):
        return None

    movie_id_to_artifact_index = {
        movie_id: index for index, movie_id in enumerate(artifact_movie_ids)
    }
    if any(movie_id not in movie_id_to_artifact_index for movie_id in requested_movie_ids):
        return None

    row_indices = [movie_id_to_artifact_index[movie_id] for movie_id in requested_movie_ids]
    selected_matrix = matrix[row_indices].tocsr()
    return ContentFeatureMatrix(
        matrix=selected_matrix,
        movie_ids=requested_movie_ids,
        movie_id_to_index={movie_id: index for index, movie_id in enumerate(requested_movie_ids)},
    )


def load_or_build_content_feature_matrix(
    movies: pd.DataFrame,
    paths: ArtifactPaths | None = None,
) -> ContentFeatureMatrix:
    """Использует сохраненную TF-IDF матрицу или строит ее на лету."""
    artifact = load_content_feature_matrix_artifact(movies, paths=paths)
    return artifact if artifact is not None else build_content_feature_matrix(movies)


def load_rating_stats_artifact(paths: ArtifactPaths | None = None) -> pd.DataFrame | None:
    """Загружает сохраненную статистику оценок по фильмам, если файл корректен."""
    paths = paths or default_artifact_paths()
    if not paths.rating_stats.exists():
        return None

    rating_stats = pd.read_parquet(paths.rating_stats)
    required_columns = {"movieId", "rating_count", "mean_rating"}
    if not required_columns.issubset(rating_stats.columns):
        return None

    result = rating_stats[["movieId", "rating_count", "mean_rating"]].copy()
    result["movieId"] = result["movieId"].astype(int)
    result["rating_count"] = result["rating_count"].astype(int)
    result["mean_rating"] = result["mean_rating"].astype(float)
    return result


def load_or_create_rating_stats(
    ratings: pd.DataFrame,
    paths: ArtifactPaths | None = None,
) -> pd.DataFrame:
    """Использует сохраненную статистику рейтингов или считает ее из ratings."""
    artifact = load_rating_stats_artifact(paths=paths)
    return artifact if artifact is not None else create_rating_stats(ratings)


def build_and_save_artifacts(
    data: MovieLensData,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> ArtifactPaths:
    """Полный offline pipeline: признаки, статистики и модельные файлы на диск."""
    ensure_data_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    movies_features = create_movie_features(data.movies)
    rating_stats = create_rating_stats(data.ratings)
    vectorizer, matrix = build_content_artifacts(movies_features)

    paths = default_artifact_paths(processed_dir=processed_dir, models_dir=models_dir)

    # Таблицы сохраняем в parquet, а ML-объекты — в форматах, удобных для быстрой загрузки.
    movies_features.to_parquet(paths.movies_features, index=False)
    rating_stats.to_parquet(paths.rating_stats, index=False)
    joblib.dump(vectorizer, paths.tfidf_vectorizer)
    sparse.save_npz(paths.tfidf_matrix, matrix)

    return paths
