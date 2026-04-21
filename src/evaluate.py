"""Offline-оценка рекомендательных моделей.

Здесь используются простые, но воспроизводимые метрики: hit_rate, precision,
coverage и novelty. Оценка сделана через leave-one-out split по пользователям.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.collaborative import (
    build_item_user_matrix,
    get_item_item_recommendations_from_matrix,
)
from src.recommend import get_popular_movies


@dataclass(frozen=True)
class LeaveOneOutSplit:
    """Train/test разбиение: последняя оценка пользователя уходит в test."""

    train: pd.DataFrame
    test: pd.DataFrame


def leave_one_out_split(ratings: pd.DataFrame, min_user_ratings: int = 2) -> LeaveOneOutSplit:
    """Оставляет по одной последней оценке каждого подходящего пользователя для test."""
    if min_user_ratings < 2:
        raise ValueError("min_user_ratings must be at least 2")

    ratings_ordered = ratings.copy()
    ratings_ordered["_row_order"] = range(len(ratings_ordered))
    # Если timestamp есть, считаем последней самую позднюю оценку; иначе сохраняем порядок строк.
    sort_columns = ["userId", "timestamp"] if "timestamp" in ratings_ordered.columns else ["userId"]
    sort_columns.append("_row_order")
    ratings_ordered = ratings_ordered.sort_values(sort_columns)

    user_counts = ratings_ordered.groupby("userId")["movieId"].transform("size")
    eligible = ratings_ordered[user_counts >= min_user_ratings]
    test_indices = eligible.groupby("userId").tail(1).index

    test = ratings_ordered.loc[test_indices].drop(columns=["_row_order"]).reset_index(drop=True)
    train = ratings_ordered.drop(index=test_indices).drop(columns=["_row_order"]).reset_index(drop=True)
    return LeaveOneOutSplit(train=train, test=test)


def _user_recommendations(
    ranked_movie_ids: list[int],
    seen_movie_ids: set[int],
    k: int,
) -> list[int]:
    """Берет первые K фильмов, которые пользователь еще не видел."""
    return [movie_id for movie_id in ranked_movie_ids if movie_id not in seen_movie_ids][:k]


def _metrics_result(
    k: int,
    evaluated_users: int,
    hits: int,
    precision_sum: float,
    recommended_movie_ids: set[int] | None = None,
    all_movie_count: int = 0,
    movie_popularity: dict[int, int] | None = None,
) -> dict[str, float | int]:
    """Собирает общую форму ответа для всех моделей."""
    recommended_movie_ids = recommended_movie_ids or set()
    coverage = len(recommended_movie_ids) / all_movie_count if all_movie_count else 0.0
    if recommended_movie_ids and movie_popularity:
        novelty = sum(1 / (1 + movie_popularity.get(movie_id, 0)) for movie_id in recommended_movie_ids) / len(
            recommended_movie_ids
        )
    else:
        novelty = 0.0

    if evaluated_users == 0:
        return {
            "k": k,
            "evaluated_users": 0,
            "hit_rate": 0.0,
            "precision": 0.0,
            "coverage": coverage,
            "novelty": novelty,
        }

    return {
        "k": k,
        "evaluated_users": evaluated_users,
        "hit_rate": hits / evaluated_users,
        "precision": precision_sum / evaluated_users,
        "coverage": coverage,
        "novelty": novelty,
    }


def evaluate_popularity_baseline(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    min_user_ratings: int = 2,
    min_movie_ratings: int = 1,
) -> dict[str, float | int]:
    """Оценивает baseline, который рекомендует глобально популярные фильмы."""
    if k < 1:
        raise ValueError("k must be positive")

    split = leave_one_out_split(ratings, min_user_ratings=min_user_ratings)
    popular = get_popular_movies(
        movies,
        split.train,
        min_ratings=min_movie_ratings,
        limit=len(movies),
    )
    ranked_movie_ids = popular["movieId"].astype(int).tolist()

    test_by_user = split.test.set_index("userId")["movieId"].astype(int).to_dict()
    train_by_user = split.train.groupby("userId")["movieId"].apply(lambda values: set(values.astype(int)))

    hits = 0
    precision_sum = 0.0
    evaluated_users = 0
    recommended_movie_ids: set[int] = set()
    movie_popularity = split.train.groupby("movieId").size().astype(int).to_dict()

    for user_id, relevant_movie_id in test_by_user.items():
        seen_movie_ids = train_by_user.get(user_id, set())
        recommendations = _user_recommendations(ranked_movie_ids, seen_movie_ids, k)
        if not recommendations:
            continue

        recommended_movie_ids.update(recommendations)
        hit = int(relevant_movie_id in recommendations)
        hits += hit
        precision_sum += hit / len(recommendations)
        evaluated_users += 1

    return _metrics_result(
        k,
        evaluated_users,
        hits,
        precision_sum,
        recommended_movie_ids=recommended_movie_ids,
        all_movie_count=int(movies["movieId"].nunique()),
        movie_popularity=movie_popularity,
    )


def evaluate_item_item_cf(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    min_user_ratings: int = 2,
    min_positive_rating: float = 4.0,
    max_users: int | None = 100,
) -> dict[str, float | int]:
    """Оценивает item-item collaborative filtering на leave-one-out split."""
    if k < 1:
        raise ValueError("k must be positive")
    if max_users is not None and max_users < 1:
        raise ValueError("max_users must be positive or None")

    split = leave_one_out_split(ratings, min_user_ratings=min_user_ratings)
    test = split.test.sort_values("userId").reset_index(drop=True)
    if max_users is not None:
        # CF дороже popularity baseline, поэтому в UI и CLI можно ограничить число пользователей.
        test = test.head(max_users)

    if test.empty:
        return _metrics_result(k, evaluated_users=0, hits=0, precision_sum=0.0)

    item_user = build_item_user_matrix(split.train)
    rating_stats = (
        split.train.groupby("movieId", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
    )

    hits = 0
    precision_sum = 0.0
    evaluated_users = 0
    recommended_movie_ids: set[int] = set()
    movie_popularity = split.train.groupby("movieId").size().astype(int).to_dict()

    for row in test.itertuples(index=False):
        recommendations = get_item_item_recommendations_from_matrix(
            user_id=int(row.userId),
            movies=movies,
            ratings=split.train,
            item_user=item_user,
            rating_stats=rating_stats,
            limit=k,
            min_positive_rating=min_positive_rating,
        )
        if recommendations.empty:
            continue

        user_recommended_movie_ids = set(recommendations["movieId"].astype(int).tolist())
        recommended_movie_ids.update(user_recommended_movie_ids)
        hit = int(int(row.movieId) in user_recommended_movie_ids)
        hits += hit
        precision_sum += hit / len(recommendations)
        evaluated_users += 1

    return _metrics_result(
        k,
        evaluated_users,
        hits,
        precision_sum,
        recommended_movie_ids=recommended_movie_ids,
        all_movie_count=int(movies["movieId"].nunique()),
        movie_popularity=movie_popularity,
    )


def compare_recommenders(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    min_user_ratings: int = 2,
    min_movie_ratings: int = 1,
    min_positive_rating: float = 4.0,
    max_cf_users: int | None = 100,
) -> dict[str, dict[str, float | int]]:
    """Считает метрики сразу для двух моделей, чтобы их удобно сравнивать."""
    return {
        "popularity_baseline": evaluate_popularity_baseline(
            movies,
            ratings,
            k=k,
            min_user_ratings=min_user_ratings,
            min_movie_ratings=min_movie_ratings,
        ),
        "item_item_cf": evaluate_item_item_cf(
            movies,
            ratings,
            k=k,
            min_user_ratings=min_user_ratings,
            min_positive_rating=min_positive_rating,
            max_users=max_cf_users,
        ),
    }


def evaluate_at_k_values(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k_values: list[int],
    min_user_ratings: int = 2,
    min_movie_ratings: int = 1,
    min_positive_rating: float = 4.0,
    max_cf_users: int | None = 100,
) -> pd.DataFrame:
    """Строит таблицу метрик для нескольких K, из нее рисуются кривые в UI."""
    rows = []
    for k in k_values:
        metrics_by_model = compare_recommenders(
            movies,
            ratings,
            k=k,
            min_user_ratings=min_user_ratings,
            min_movie_ratings=min_movie_ratings,
            min_positive_rating=min_positive_rating,
            max_cf_users=max_cf_users,
        )
        for model_name, metrics in metrics_by_model.items():
            rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows)
