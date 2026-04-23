"""Тесты SVD-рекомендателя."""

from __future__ import annotations

import pandas as pd
import pytest

from src.matrix_factorization import build_svd_recommender, get_svd_recommendations


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": ["Matrix", "John Wick", "Slow Drama", "Speed"],
            "genres": ["Action|Sci-Fi", "Action|Thriller", "Drama", "Action"],
        }
    )


@pytest.fixture
def ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 4, 4],
            "movieId": [1, 2, 3, 1, 2, 4, 1, 4, 2, 4],
            "rating": [5.0, 4.0, 2.0, 5.0, 4.0, 5.0, 4.0, 5.0, 5.0, 4.0],
        }
    )


def test_build_svd_recommender_caps_components_and_creates_indexes(ratings):
    recommender = build_svd_recommender(ratings, n_components=10)

    assert recommender.user_id_to_index[1] == 0
    assert recommender.movie_id_to_index[1] == 0
    assert recommender.n_components <= min(len(recommender.user_ids), len(recommender.movie_ids)) - 1
    assert recommender.user_factors.shape[0] == len(recommender.user_ids)
    assert recommender.item_factors.shape[0] == len(recommender.movie_ids)


def test_get_svd_recommendations_excludes_seen_movies_and_returns_scores(movies, ratings):
    result = get_svd_recommendations(1, movies, ratings, limit=2, n_components=2)

    assert len(result) == 1
    assert result.loc[0, "movieId"] == 4
    assert set(result["movieId"]).isdisjoint({1, 2, 3})
    assert 0.5 <= result.loc[0, "predicted_rating"] <= 5.0


def test_get_svd_recommendations_rejects_invalid_input(movies, ratings):
    with pytest.raises(ValueError, match="Unknown userId"):
        get_svd_recommendations(999, movies, ratings)

    with pytest.raises(ValueError, match="limit must be positive"):
        get_svd_recommendations(1, movies, ratings, limit=0)


def test_build_svd_recommender_rejects_invalid_components(ratings):
    with pytest.raises(ValueError, match="n_components must be positive"):
        build_svd_recommender(ratings, n_components=0)
