"""Гибридный рекомендатель.

Здесь объединяются три сигнала:
- item-item collaborative filtering,
- SVD-предсказание рейтинга,
- глобальная популярность как fallback.

Такой подход полезен, когда один из сигналов слабый: например, у пользователя
слишком мало явно понравившихся фильмов для CF, но SVD все еще может дать
осмысленный скоринг, а популярность не оставит выдачу пустой.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from src.collaborative import (
    ItemUserMatrix,
    build_item_user_matrix,
    get_item_item_recommendations_from_matrix,
    get_item_rating_stats,
)
from src.matrix_factorization import (
    SVDRecommender,
    build_svd_recommender,
    get_svd_recommendations_from_model,
)
from src.recommend import get_popular_movies


@dataclass(frozen=True)
class HybridArtifacts:
    """Кешируемые артефакты, чтобы не пересобирать модели на каждый запрос."""

    item_user: ItemUserMatrix
    svd_recommender: SVDRecommender
    rating_stats: pd.DataFrame
    popularity_ranking: pd.DataFrame


@dataclass(frozen=True)
class HybridWeights:
    """Веса сигналов в гибридном ранжировщике."""

    cf_weight: float = 0.82
    svd_weight: float = 0.13
    popularity_weight: float = 0.05

    def normalized(self) -> "HybridWeights":
        """Нормализует веса до суммы 1, сохраняя их относительные пропорции."""
        total = self.cf_weight + self.svd_weight + self.popularity_weight
        if total <= 0:
            raise ValueError("at least one hybrid weight must be positive")
        return HybridWeights(
            cf_weight=self.cf_weight / total,
            svd_weight=self.svd_weight / total,
            popularity_weight=self.popularity_weight / total,
        )


def _build_popularity_ranking(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Готовит глобальный рейтинг фильмов для fallback и hybrid score."""
    ranking = get_popular_movies(movies, ratings, min_ratings=1, limit=len(movies)).copy()
    if ranking.empty:
        ranking["popularity_score"] = pd.Series(dtype=float)
        ranking["popularity_rank"] = pd.Series(dtype=float)
        return ranking

    if len(ranking) == 1:
        ranking["popularity_score"] = 1.0
        ranking["popularity_rank"] = 1
        return ranking

    ranking["popularity_score"] = 1 - (ranking.index / (len(ranking) - 1))
    ranking["popularity_rank"] = ranking.index + 1
    return ranking


def build_hybrid_artifacts(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    n_components: int = 20,
) -> HybridArtifacts:
    """Собирает CF, SVD и popularity-артефакты в один пакет."""
    return HybridArtifacts(
        item_user=build_item_user_matrix(ratings),
        svd_recommender=build_svd_recommender(ratings, n_components=n_components),
        rating_stats=get_item_rating_stats(ratings),
        popularity_ranking=_build_popularity_ranking(movies, ratings),
    )


def _empty_recommendations() -> pd.DataFrame:
    """Возвращает пустую таблицу со схемой гибридного ответа."""
    return pd.DataFrame(
        columns=[
            "movieId",
            "title",
            "genres",
            "hybrid_score",
            "cf_score",
            "predicted_rating",
            "popularity_score",
            "rating_count",
            "mean_rating",
        ]
    )


def _normalize_score(column: pd.Series) -> pd.Series:
    """Нормализует сигнал в диапазон 0..1, не ломаясь на константах и NaN."""
    values = column.astype(float)
    non_null = values.dropna()
    if non_null.empty:
        return pd.Series(0.0, index=column.index, dtype=float)

    min_value = float(non_null.min())
    max_value = float(non_null.max())
    if min_value == max_value:
        return values.notna().astype(float)

    normalized = (values - min_value) / (max_value - min_value)
    return normalized.fillna(0.0)


def _rank_score(rank: float | int | None, weight: float) -> float:
    """Преобразует место фильма в выдаче одной модели во вклад в hybrid score."""
    if pd.isna(rank) or weight <= 0:
        return 0.0
    return weight / math.log2(float(rank) + 1)


def get_hybrid_recommendations_from_artifacts(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    artifacts: HybridArtifacts,
    limit: int = 10,
    min_positive_rating: float = 4.0,
    candidate_pool_size: int | None = None,
    weights: HybridWeights = HybridWeights(),
) -> pd.DataFrame:
    """Собирает персональную выдачу из нескольких сигналов."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if int(user_id) not in artifacts.svd_recommender.user_id_to_index:
        raise ValueError(f"Unknown userId: {user_id}")

    user_history = ratings[ratings["userId"].astype(int) == int(user_id)].copy()
    seen_movie_ids = set(user_history["movieId"].astype(int))
    if len(seen_movie_ids) >= len(movies):
        return _empty_recommendations()

    pool_size = candidate_pool_size if candidate_pool_size is not None else max(limit * 10, 100)
    if pool_size < limit:
        raise ValueError("candidate_pool_size must be greater than or equal to limit")
    weights = weights.normalized()

    cf_recommendations = get_item_item_recommendations_from_matrix(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        item_user=artifacts.item_user,
        rating_stats=artifacts.rating_stats,
        limit=pool_size,
        min_positive_rating=min_positive_rating,
    )[["movieId", "cf_score"]].reset_index(drop=True)
    cf_recommendations["cf_rank"] = cf_recommendations.index + 1

    svd_recommendations = get_svd_recommendations_from_model(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        recommender=artifacts.svd_recommender,
        rating_stats=artifacts.rating_stats,
        limit=pool_size,
    )[["movieId", "predicted_rating"]].reset_index(drop=True)
    svd_recommendations["svd_rank"] = svd_recommendations.index + 1

    popularity_recommendations = artifacts.popularity_ranking.copy()
    if popularity_recommendations.empty:
        popularity_recommendations = pd.DataFrame(columns=["movieId", "popularity_score", "popularity_rank"])
    else:
        popularity_recommendations = popularity_recommendations.loc[
            ~popularity_recommendations["movieId"].astype(int).isin(seen_movie_ids),
            ["movieId", "popularity_score", "popularity_rank", "rating_count", "mean_rating"],
        ].head(pool_size)

    candidate_ids = sorted(
        set(cf_recommendations["movieId"].astype(int))
        | set(svd_recommendations["movieId"].astype(int))
        | set(popularity_recommendations["movieId"].astype(int))
    )
    if not candidate_ids:
        return _empty_recommendations()

    result = movies[movies["movieId"].astype(int).isin(candidate_ids)].copy()
    result = result.merge(cf_recommendations, on="movieId", how="left")
    result = result.merge(svd_recommendations, on="movieId", how="left")
    result = result.merge(popularity_recommendations, on="movieId", how="left")
    result = result.merge(artifacts.rating_stats, on="movieId", how="left", suffixes=("", "_stats"))

    if "rating_count_stats" in result.columns:
        result["rating_count"] = result["rating_count"].fillna(result["rating_count_stats"])
        result = result.drop(columns=["rating_count_stats"])
    if "mean_rating_stats" in result.columns:
        result["mean_rating"] = result["mean_rating"].fillna(result["mean_rating_stats"])
        result = result.drop(columns=["mean_rating_stats"])

    result = result.fillna({"rating_count": 0, "mean_rating": 0.0})
    result["cf_score_norm"] = _normalize_score(result["cf_score"])
    result["predicted_rating_norm"] = _normalize_score(result["predicted_rating"])
    result["popularity_score_norm"] = _normalize_score(result["popularity_score"])

    # Ранги оказались устойчивее сырого min-max: сильный CF-сигнал сохраняет верх
    # выдачи, а SVD и popularity помогают заполнить хвост и cold-ish случаи.
    result["hybrid_score"] = (
        result["cf_rank"].apply(lambda rank: _rank_score(rank, weights.cf_weight))
        + result["svd_rank"].apply(lambda rank: _rank_score(rank, weights.svd_weight))
        + result["popularity_rank"].apply(lambda rank: _rank_score(rank, weights.popularity_weight))
    )

    return (
        result.sort_values(
            ["hybrid_score", "cf_rank", "svd_rank", "popularity_rank", "mean_rating", "rating_count", "title"],
            ascending=[False, True, True, True, False, False, True],
        )
        .head(limit)
        .reset_index(drop=True)[
            [
                "movieId",
                "title",
                "genres",
                "hybrid_score",
                "cf_score",
                "predicted_rating",
                "popularity_score",
                "rating_count",
                "mean_rating",
            ]
        ]
    )


def get_hybrid_recommendations(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    limit: int = 10,
    min_positive_rating: float = 4.0,
    n_components: int = 20,
    weights: HybridWeights = HybridWeights(),
) -> pd.DataFrame:
    """Удобная обертка для UI и скриптов."""
    artifacts = build_hybrid_artifacts(movies, ratings, n_components=n_components)
    return get_hybrid_recommendations_from_artifacts(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        artifacts=artifacts,
        limit=limit,
        min_positive_rating=min_positive_rating,
        weights=weights,
    )
