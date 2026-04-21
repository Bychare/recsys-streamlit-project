"""Тесты item-item collaborative filtering."""

from __future__ import annotations

import pytest

import pandas as pd

from src.collaborative import build_item_user_matrix, get_item_item_recommendations


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


def test_build_item_user_matrix_creates_sparse_matrix(ratings):
    item_user = build_item_user_matrix(ratings)

    assert item_user.matrix.shape == (4, 4)
    assert item_user.movie_id_to_index[1] == 0
    assert item_user.user_id_to_index[1] == 0
    assert item_user.matrix.nnz == len(ratings)


def test_build_item_user_matrix_filters_by_min_rating(ratings):
    item_user = build_item_user_matrix(ratings, min_rating=5.0)

    assert item_user.matrix.nnz == 5
    assert 3 not in item_user.movie_id_to_index


def test_get_item_item_recommendations_excludes_seen_and_ranks_neighbor(movies, ratings):
    result = get_item_item_recommendations(1, movies, ratings, limit=2, min_positive_rating=4.0)

    assert result.loc[0, "movieId"] == 4
    assert set(result["movieId"]).isdisjoint({1, 2, 3})
    assert result.loc[0, "cf_score"] > 0


def test_get_item_item_recommendations_returns_empty_without_likes(movies, ratings):
    result = get_item_item_recommendations(1, movies, ratings, limit=2, min_positive_rating=5.5)

    assert result.empty
    assert result.columns.tolist() == [
        "movieId",
        "title",
        "genres",
        "cf_score",
        "rating_count",
        "mean_rating",
    ]


def test_get_item_item_recommendations_rejects_invalid_input(movies, ratings):
    with pytest.raises(ValueError, match="Unknown userId"):
        get_item_item_recommendations(999, movies, ratings)

    with pytest.raises(ValueError, match="limit must be positive"):
        get_item_item_recommendations(1, movies, ratings, limit=0)
