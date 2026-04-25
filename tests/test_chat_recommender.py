"""Tests for rule-based recommendation chat parsing."""

from __future__ import annotations

import pandas as pd

from src.chat_recommender import (
    describe_request,
    detect_genres,
    detect_user_id,
    detect_year_range,
    find_movie_mention,
    parse_chat_request,
)


def movies() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Matrix, The (1999)", "Toy Story (1995)", "Heat (1995)"],
            "genres": ["Action|Sci-Fi", "Adventure|Animation|Children|Comedy", "Action|Crime"],
        }
    )


def test_detect_genres_supports_russian_and_english_aliases():
    result = detect_genres("Хочу смешные sci-fi фильмы")

    assert result == ("Comedy", "Sci-Fi")


def test_detect_year_range_handles_decade_and_bounds():
    assert detect_year_range("фантастика 90-х") == (1990, 1999)
    assert detect_year_range("после 2000", max_year=2020) == (2000, 2020)
    assert detect_year_range("до 1985", min_year=1902) == (1902, 1985)
    assert detect_year_range("1995") == (1995, 1995)


def test_detect_user_id_extracts_known_phrases():
    assert detect_user_id("пользователь 42 хочет комедии") == 42
    assert detect_user_id("user #7 drama") == 7
    assert detect_user_id("без пользователя") is None


def test_find_movie_mention_matches_clean_title_fragment():
    match = find_movie_mention("посоветуй что-то похожее на Matrix", movies())

    assert match == (1, "Matrix, The (1999)")


def test_parse_chat_request_combines_filters_and_anchor_movie():
    request = parse_chat_request(
        "user 2 хочет фантастику 90-х похожую на Matrix",
        movies(),
        available_genres={"Action", "Sci-Fi", "Comedy"},
        min_year=1900,
        max_year=2020,
    )

    assert request.user_id == 2
    assert request.genres == ("Sci-Fi",)
    assert request.year_range == (1990, 1999)
    assert request.anchor_movie_id == 1
    assert request.is_similar_request


def test_describe_request_returns_compact_summary():
    request = parse_chat_request("комедии 90-х", movies(), available_genres={"Comedy"})

    assert describe_request(request) == "Comedy; 1990-1999"
