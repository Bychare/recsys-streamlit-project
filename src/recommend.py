"""Простые рекомендатели для MVP.

Здесь лежат быстрые baseline-подходы: популярность, top-rated и content-based
похожесть по жанрам/названию. Они простые, но полезны как точка сравнения.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _movie_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    """Считает количество оценок и среднюю оценку по каждому фильму."""
    return (
        ratings.groupby("movieId", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
    )


def _with_rating_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к таблице фильмов агрегаты по оценкам."""
    stats = _movie_rating_stats(ratings)
    return movies.merge(stats, on="movieId", how="left").fillna(
        {"rating_count": 0, "mean_rating": 0.0}
    )


def get_popular_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings: int = 1,
    limit: int = 10,
) -> pd.DataFrame:
    """Возвращает фильмы, которые чаще всего оценивали пользователи."""
    scored = _with_rating_stats(movies, ratings)
    return (
        scored[scored["rating_count"] >= min_ratings]
        .sort_values(["rating_count", "mean_rating", "title"], ascending=[False, False, True])
        .head(limit)
        .reset_index(drop=True)
    )


def get_top_rated_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings: int = 50,
    limit: int = 10,
) -> pd.DataFrame:
    """Возвращает фильмы с лучшей средней оценкой после отсечения по числу оценок."""
    scored = _with_rating_stats(movies, ratings)
    return (
        scored[scored["rating_count"] >= min_ratings]
        .sort_values(["mean_rating", "rating_count", "title"], ascending=[False, False, True])
        .head(limit)
        .reset_index(drop=True)
    )


def build_feature_text(movies: pd.DataFrame) -> pd.Series:
    """Собирает простой текст признаков из жанров и названия фильма."""
    genres = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    titles = movies["title"].fillna("").str.replace(r"\(\d{4}\)", "", regex=True)
    return genres + " " + titles


def get_similar_movies(movie_id: int, movies: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Ищет похожие фильмы через TF-IDF и cosine similarity."""
    if movie_id not in set(movies["movieId"]):
        raise ValueError(f"Unknown movieId: {movie_id}")

    movie_index = movies.index[movies["movieId"] == movie_id][0]
    vectorizer = TfidfVectorizer(stop_words="english")
    # Матрица признаков строится на лету: для MovieLens latest-small это быстро и прозрачно.
    feature_matrix = vectorizer.fit_transform(build_feature_text(movies))

    similarities = cosine_similarity(feature_matrix[movie_index], feature_matrix).ravel()
    result = movies.copy()
    result["similarity"] = similarities

    return (
        result[result["movieId"] != movie_id]
        .sort_values(["similarity", "title"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )
