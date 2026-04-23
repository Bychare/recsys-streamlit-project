"""SVD-рекомендатель на user-item матрице.

Это еще один персональный подход поверх пользовательских оценок.
В отличие от item-item CF здесь ищутся скрытые факторы: например, любовь
к блокбастерам, динамичным боевикам или спокойным драмам.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.collaborative import get_item_rating_stats


@dataclass(frozen=True)
class SVDRecommender:
    """Латентные представления пользователей и фильмов после факторизации матрицы."""

    user_factors: np.ndarray
    item_factors: np.ndarray
    user_means: np.ndarray
    user_ids: list[int]
    movie_ids: list[int]
    user_id_to_index: dict[int, int]
    movie_id_to_index: dict[int, int]
    n_components: int


def build_svd_recommender(
    ratings: pd.DataFrame,
    n_components: int = 20,
    min_rating: float = 0.0,
    random_state: int = 42,
) -> SVDRecommender:
    """Строит SVD-модель по таблице оценок.

    Матрица сначала центрируется по среднему рейтингу каждого пользователя,
    а затем раскладывается через TruncatedSVD. Для MovieLens small этого
    хватает, чтобы быстро получить рабочий персональный baseline.
    """
    if n_components < 1:
        raise ValueError("n_components must be positive")

    interactions = ratings[ratings["rating"] >= min_rating].copy()
    interactions = interactions.groupby(["userId", "movieId"], as_index=False).agg(rating=("rating", "mean"))
    if interactions.empty:
        raise ValueError("ratings must contain at least one interaction")

    user_ids = sorted(interactions["userId"].astype(int).unique().tolist())
    movie_ids = sorted(interactions["movieId"].astype(int).unique().tolist())
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}

    rows = interactions["userId"].astype(int).map(user_id_to_index)
    columns = interactions["movieId"].astype(int).map(movie_id_to_index)
    values = interactions["rating"].astype(float)
    matrix = csr_matrix((values, (rows, columns)), shape=(len(user_ids), len(movie_ids)))

    user_sum = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    user_count = np.diff(matrix.indptr).astype(float)
    global_mean = float(interactions["rating"].mean())
    user_means = np.divide(
        user_sum,
        user_count,
        out=np.full(len(user_ids), global_mean, dtype=float),
        where=user_count > 0,
    )

    centered = matrix.astype(float).copy().tocsr()
    for user_index in range(centered.shape[0]):
        start = centered.indptr[user_index]
        end = centered.indptr[user_index + 1]
        if start != end:
            centered.data[start:end] -= user_means[user_index]

    max_components = min(centered.shape) - 1
    effective_components = min(n_components, max_components)

    if effective_components < 1 or centered.nnz == 0:
        # Для совсем маленьких train-сплитов оставляем bias-only fallback:
        # все фильмы получают примерно среднюю оценку пользователя.
        user_factors = np.zeros((len(user_ids), 1), dtype=float)
        item_factors = np.zeros((len(movie_ids), 1), dtype=float)
        effective_components = 0
    else:
        svd = TruncatedSVD(n_components=effective_components, random_state=random_state)
        user_factors = svd.fit_transform(centered)
        item_factors = svd.components_.T

    return SVDRecommender(
        user_factors=user_factors,
        item_factors=item_factors,
        user_means=user_means,
        user_ids=user_ids,
        movie_ids=movie_ids,
        user_id_to_index=user_id_to_index,
        movie_id_to_index=movie_id_to_index,
        n_components=effective_components,
    )


def _empty_recommendations() -> pd.DataFrame:
    """Возвращает пустую таблицу с той же схемой, что и обычный ответ модели."""
    return pd.DataFrame(
        columns=[
            "movieId",
            "title",
            "genres",
            "predicted_rating",
            "rating_count",
            "mean_rating",
        ]
    )


def get_svd_recommendations_from_model(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    recommender: SVDRecommender,
    rating_stats: pd.DataFrame | None = None,
    limit: int = 10,
) -> pd.DataFrame:
    """Возвращает персональные рекомендации для известного пользователя."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if int(user_id) not in recommender.user_id_to_index:
        raise ValueError(f"Unknown userId: {user_id}")

    user_history = ratings[ratings["userId"].astype(int) == int(user_id)].copy()
    seen_movie_ids = set(user_history["movieId"].astype(int))
    if len(seen_movie_ids) >= len(recommender.movie_ids):
        return _empty_recommendations()

    user_index = recommender.user_id_to_index[int(user_id)]
    predicted = recommender.user_factors[user_index].dot(recommender.item_factors.T)
    predicted = np.asarray(predicted).ravel() + recommender.user_means[user_index]
    predicted = np.clip(predicted, 0.5, 5.0)

    scored = pd.DataFrame(
        {
            "movieId": recommender.movie_ids,
            "predicted_rating": predicted,
        }
    )
    scored = scored[~scored["movieId"].isin(seen_movie_ids)]

    rating_stats = rating_stats if rating_stats is not None else get_item_rating_stats(ratings)
    result = scored.merge(movies, on="movieId", how="inner").merge(rating_stats, on="movieId", how="left")
    result = result.fillna({"rating_count": 0, "mean_rating": 0.0})

    return (
        result.sort_values(
            ["predicted_rating", "mean_rating", "rating_count", "title"],
            ascending=[False, False, False, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def get_svd_recommendations(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    limit: int = 10,
    n_components: int = 20,
) -> pd.DataFrame:
    """Удобная обертка для UI: строит SVD-модель и сразу возвращает рекомендации."""
    recommender = build_svd_recommender(ratings, n_components=n_components)
    return get_svd_recommendations_from_model(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        recommender=recommender,
        rating_stats=get_item_rating_stats(ratings),
        limit=limit,
    )
