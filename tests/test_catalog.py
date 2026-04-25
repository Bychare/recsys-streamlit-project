"""Tests for catalog browsing helpers."""

from __future__ import annotations

import pandas as pd

from src.catalog import add_movie_metadata, build_movie_labels, filter_movie_catalog, get_genre_options


def test_add_movie_metadata_extracts_year_and_primary_genre():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Unknown Title"],
            "genres": ["Adventure|Animation", None],
        }
    )

    result = add_movie_metadata(movies)

    assert result["year"].tolist() == [1995, pd.NA]
    assert result["title_clean"].tolist() == ["Toy Story", "Unknown Title"]
    assert result["primary_genre"].tolist() == ["Adventure", "Unknown"]


def test_get_genre_options_returns_sorted_real_genres():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["A (2000)", "B (2001)"],
            "genres": ["Drama|Comedy", "(no genres listed)"],
        }
    )

    assert get_genre_options(movies) == ["Comedy", "Drama"]


def test_filter_movie_catalog_combines_genre_year_and_query():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Toy Story (1995)", "Heat (1995)", "Toy Soldiers (1991)"],
            "genres": ["Adventure|Animation", "Action|Crime", "Action|Drama"],
        }
    )

    result = filter_movie_catalog(
        movies,
        genres=["Action"],
        year_range=(1990, 1994),
        query="toy",
    )

    assert result["movieId"].tolist() == [3]


def test_build_movie_labels_disambiguates_duplicate_titles():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Hamlet", "Hamlet", "Heat"],
            "genres": ["Drama", "Drama", "Crime"],
        }
    )

    result = build_movie_labels(movies)

    assert result["label"].tolist() == ["Hamlet · ID 1", "Hamlet · ID 2", "Heat"]
