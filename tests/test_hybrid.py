"""Тесты гибридного рекомендателя."""

from __future__ import annotations

import pandas as pd
import pytest

from src.hybrid import (
    build_hybrid_artifacts,
    get_hybrid_recommendations,
    get_hybrid_recommendations_from_artifacts,
)


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["Matrix", "John Wick", "Slow Drama", "Speed", "Heat"],
            "genres": [
                "Action|Sci-Fi",
                "Action|Thriller",
                "Drama",
                "Action",
                "Crime|Drama",
            ],
        }
    )


@pytest.fixture
def ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "movieId": [1, 2, 3, 1, 2, 4, 5, 1, 4, 5, 2, 4, 5],
            "rating": [5.0, 4.0, 2.0, 5.0, 4.5, 5.0, 4.0, 4.0, 5.0, 3.5, 5.0, 4.0, 4.5],
        }
    )


def test_build_hybrid_artifacts_contains_all_components(movies, ratings):
    artifacts = build_hybrid_artifacts(movies, ratings, n_components=3)

    assert artifacts.item_user.matrix.shape[0] == ratings["movieId"].nunique()
    assert artifacts.svd_recommender.n_components >= 0
    assert {"movieId", "popularity_score"}.issubset(artifacts.popularity_ranking.columns)


def test_get_hybrid_recommendations_combines_signals_and_excludes_seen(movies, ratings):
    result = get_hybrid_recommendations(1, movies, ratings, limit=3, n_components=2)

    assert not result.empty
    assert set(result["movieId"]).isdisjoint({1, 2, 3})
    assert "hybrid_score" in result.columns
    assert result.loc[0, "hybrid_score"] >= result.loc[len(result) - 1, "hybrid_score"]


def test_get_hybrid_recommendations_supports_filtered_catalog(movies, ratings):
    artifacts = build_hybrid_artifacts(movies, ratings, n_components=2)
    filtered_movies = movies[movies["movieId"].isin([4, 5])]

    result = get_hybrid_recommendations_from_artifacts(
        user_id=1,
        movies=filtered_movies,
        ratings=ratings,
        artifacts=artifacts,
        limit=2,
    )

    assert not result.empty
    assert set(result["movieId"]).issubset({4, 5})
    assert set(result["movieId"]).isdisjoint({1, 2, 3})


def test_get_hybrid_recommendations_rejects_invalid_input(movies, ratings):
    with pytest.raises(ValueError, match="Unknown userId"):
        get_hybrid_recommendations(999, movies, ratings)

    with pytest.raises(ValueError, match="limit must be positive"):
        get_hybrid_recommendations(1, movies, ratings, limit=0)
