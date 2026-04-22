"""Item-item collaborative filtering.

Модель смотрит не на жанры, а на совместные оценки пользователей:
если одни и те же пользователи высоко оценивали два фильма, фильмы считаются похожими.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ItemUserMatrix:
    """Разреженная матрица `фильм x пользователь` и индексы для обратного поиска."""

    matrix: csr_matrix
    movie_ids: list[int]
    user_ids: list[int]
    movie_id_to_index: dict[int, int]
    user_id_to_index: dict[int, int]


def build_item_user_matrix(ratings: pd.DataFrame, min_rating: float = 0.0) -> ItemUserMatrix:
    """Строит sparse-матрицу оценок, где строки — фильмы, колонки — пользователи."""
    interactions = ratings[ratings["rating"] >= min_rating].copy()
    interactions = interactions.groupby(["movieId", "userId"], as_index=False).agg(rating=("rating", "mean"))

    movie_ids = sorted(interactions["movieId"].astype(int).unique().tolist())
    user_ids = sorted(interactions["userId"].astype(int).unique().tolist())
    movie_id_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

    rows = interactions["movieId"].astype(int).map(movie_id_to_index)
    columns = interactions["userId"].astype(int).map(user_id_to_index)
    values = interactions["rating"].astype(float)

    matrix = csr_matrix(
        (values, (rows, columns)),
        shape=(len(movie_ids), len(user_ids)),
    )

    return ItemUserMatrix(
        matrix=matrix,
        movie_ids=movie_ids,
        user_ids=user_ids,
        movie_id_to_index=movie_id_to_index,
        user_id_to_index=user_id_to_index,
    )


def _empty_recommendations() -> pd.DataFrame:
    """Возвращает пустую таблицу с теми же колонками, что и обычный ответ модели."""
    return pd.DataFrame(
        columns=[
            "movieId",
            "title",
            "genres",
            "cf_score",
            "rating_count",
            "mean_rating",
        ]
    )


def get_item_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по фильмам нужны, чтобы сортировать и показывать рекомендации понятнее."""
    return (
        ratings.groupby("movieId", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
    )


def get_item_item_recommendations_from_matrix(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    item_user: ItemUserMatrix,
    rating_stats: pd.DataFrame | None = None,
    limit: int = 10,
    min_positive_rating: float = 4.0,
) -> pd.DataFrame:
    """Рекомендует фильмы по уже построенной item-user матрице.

    Эта функция полезна для offline-оценки: матрицу можно построить один раз
    и переиспользовать для многих пользователей.
    """
    if user_id not in set(ratings["userId"].astype(int)):
        raise ValueError(f"Unknown userId: {user_id}")
    if limit < 1:
        raise ValueError("limit must be positive")

    user_history = ratings[ratings["userId"].astype(int) == int(user_id)].copy()
    seen_movie_ids = set(user_history["movieId"].astype(int))
    liked = user_history[user_history["rating"] >= min_positive_rating]

    # Положительная история пользователя превращается в взвешенный профиль интересов.
    liked_scores = liked.groupby("movieId")["rating"].mean().sort_values(ascending=False)
    liked_movie_ids = [
        int(movie_id)
        for movie_id in liked_scores.index.tolist()
        if int(movie_id) in item_user.movie_id_to_index
    ]

    if not liked_movie_ids:
        return _empty_recommendations()

    liked_indices = [item_user.movie_id_to_index[movie_id] for movie_id in liked_movie_ids]
    liked_weights = liked_scores.loc[liked_movie_ids].astype(float).to_numpy()

    # Считаем похожесть любимых фильмов пользователя со всеми фильмами каталога.
    similarities = cosine_similarity(item_user.matrix[liked_indices], item_user.matrix)
    scores = similarities.T.dot(liked_weights) / liked_weights.sum()

    scored = pd.DataFrame(
        {
            "movieId": item_user.movie_ids,
            "cf_score": scores,
        }
    )
    scored = scored[~scored["movieId"].isin(seen_movie_ids)]

    rating_stats = rating_stats if rating_stats is not None else get_item_rating_stats(ratings)
    result = scored.merge(movies, on="movieId", how="inner").merge(rating_stats, on="movieId", how="left")
    result = result.fillna({"rating_count": 0, "mean_rating": 0.0})

    return (
        result.sort_values(
            ["cf_score", "mean_rating", "rating_count", "title"],
            ascending=[False, False, False, True],
        )
        .head(limit)
        .reset_index(drop=True)
    )


def get_item_item_recommendations(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    limit: int = 10,
    min_positive_rating: float = 4.0,
) -> pd.DataFrame:
    """Удобная обертка для UI: сама строит матрицу и возвращает рекомендации."""
    item_user = build_item_user_matrix(ratings)
    return get_item_item_recommendations_from_matrix(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        item_user=item_user,
        rating_stats=get_item_rating_stats(ratings),
        limit=limit,
        min_positive_rating=min_positive_rating,
    )
