from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.recommend import get_popular_movies


@dataclass(frozen=True)
class LeaveOneOutSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def leave_one_out_split(ratings: pd.DataFrame, min_user_ratings: int = 2) -> LeaveOneOutSplit:
    if min_user_ratings < 2:
        raise ValueError("min_user_ratings must be at least 2")

    ratings_ordered = ratings.copy()
    ratings_ordered["_row_order"] = range(len(ratings_ordered))
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
    return [movie_id for movie_id in ranked_movie_ids if movie_id not in seen_movie_ids][:k]


def evaluate_popularity_baseline(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    k: int = 10,
    min_user_ratings: int = 2,
    min_movie_ratings: int = 1,
) -> dict[str, float | int]:
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

    for user_id, relevant_movie_id in test_by_user.items():
        seen_movie_ids = train_by_user.get(user_id, set())
        recommendations = _user_recommendations(ranked_movie_ids, seen_movie_ids, k)
        if not recommendations:
            continue

        hit = int(relevant_movie_id in recommendations)
        hits += hit
        precision_sum += hit / len(recommendations)
        evaluated_users += 1

    if evaluated_users == 0:
        return {
            "k": k,
            "evaluated_users": 0,
            "hit_rate": 0.0,
            "precision": 0.0,
        }

    return {
        "k": k,
        "evaluated_users": evaluated_users,
        "hit_rate": hits / evaluated_users,
        "precision": precision_sum / evaluated_users,
    }
