from __future__ import annotations

import pytest

import pandas as pd

from src.recommend import (
    get_popular_movies,
    get_similar_movies,
    get_top_rated_movies,
)


@pytest.fixture
def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "title": [
                "Toy Story (1995)",
                "Toy Story 2 (1999)",
                "Heat (1995)",
                "Casino (1995)",
            ],
            "genres": [
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Adventure|Animation|Children|Comedy|Fantasy",
                "Action|Crime|Thriller",
                "Crime|Drama",
            ],
        }
    )


@pytest.fixture
def ratings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 3, 4, 1, 2],
            "movieId": [1, 1, 1, 2, 2, 3, 3, 4, 4],
            "rating": [4.0, 5.0, 4.0, 5.0, 5.0, 3.0, 4.0, 4.0, 4.0],
        }
    )


def test_get_popular_movies_orders_by_rating_count(movies, ratings):
    result = get_popular_movies(movies, ratings, min_ratings=1, limit=2)

    assert result["title"].tolist() == ["Toy Story (1995)", "Toy Story 2 (1999)"]
    assert result["rating_count"].tolist() == [3, 2]


def test_get_top_rated_movies_respects_min_ratings(movies, ratings):
    result = get_top_rated_movies(movies, ratings, min_ratings=2, limit=2)

    assert result["title"].tolist() == ["Toy Story 2 (1999)", "Toy Story (1995)"]
    assert result["mean_rating"].tolist() == [5.0, pytest.approx(4.3333333333)]


def test_get_similar_movies_returns_content_neighbors(movies):
    result = get_similar_movies(1, movies, limit=2)

    assert result.loc[0, "title"] == "Toy Story 2 (1999)"
    assert result.loc[0, "similarity"] > result.loc[1, "similarity"]


def test_get_similar_movies_rejects_unknown_movie_id(movies):
    with pytest.raises(ValueError, match="Unknown movieId"):
        get_similar_movies(999, movies, limit=2)
