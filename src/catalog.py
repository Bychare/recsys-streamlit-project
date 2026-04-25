"""Helpers for movie catalog browsing in the Streamlit UI."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")
TITLE_YEAR_PATTERN = re.compile(r"\s*\(\d{4}\)\s*$")
NO_GENRE = "(no genres listed)"


def add_movie_metadata(movies: pd.DataFrame) -> pd.DataFrame:
    """Adds lightweight display metadata used by filters and recommendation tables."""
    result = movies.copy()
    result["genres"] = result["genres"].fillna(NO_GENRE)
    result["year"] = result["title"].fillna("").str.extract(YEAR_PATTERN.pattern)[0].astype("Int64")
    result["title_clean"] = (
        result["title"]
        .fillna("")
        .str.replace(TITLE_YEAR_PATTERN.pattern, "", regex=True)
        .str.strip()
    )
    result["primary_genre"] = result["genres"].str.split("|").str[0]
    result.loc[result["primary_genre"] == NO_GENRE, "primary_genre"] = "Unknown"
    return result


def get_genre_options(movies: pd.DataFrame) -> list[str]:
    """Returns sorted genre labels available in the catalog."""
    genres = (
        movies["genres"]
        .fillna(NO_GENRE)
        .str.split("|")
        .explode()
        .dropna()
        .astype(str)
    )
    return sorted(genre for genre in genres.unique().tolist() if genre and genre != NO_GENRE)


def filter_movie_catalog(
    movies: pd.DataFrame,
    genres: Iterable[str] | None = None,
    year_range: tuple[int, int] | None = None,
    query: str | None = None,
) -> pd.DataFrame:
    """Filters movies by genre, release year and a case-insensitive title query."""
    result = add_movie_metadata(movies)

    selected_genres = {genre for genre in (genres or []) if genre}
    if selected_genres:
        genre_sets = result["genres"].str.split("|").map(set)
        result = result[genre_sets.map(lambda movie_genres: bool(movie_genres & selected_genres))]

    if year_range is not None:
        start_year, end_year = year_range
        result = result[result["year"].between(int(start_year), int(end_year), inclusive="both")]

    normalized_query = (query or "").strip()
    if normalized_query:
        result = result[
            result["title"].fillna("").str.contains(normalized_query, case=False, regex=False)
        ]

    return result.reset_index(drop=True)


def build_movie_labels(movies: pd.DataFrame) -> pd.DataFrame:
    """Builds stable selectbox labels and disambiguates duplicate titles."""
    result = add_movie_metadata(movies)
    title_counts = result["title"].value_counts()

    def label_for(row: pd.Series) -> str:
        label = str(row["title"])
        if title_counts.get(row["title"], 0) > 1:
            label = f"{label} · ID {int(row['movieId'])}"
        return label

    result["label"] = result.apply(label_for, axis=1)
    return result
