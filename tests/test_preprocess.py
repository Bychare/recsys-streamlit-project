"""Тесты offline-препроцессинга и сохранения артефактов."""

from __future__ import annotations

import pandas as pd
from scipy import sparse

from src.data_loader import MovieLensData
from src.preprocess import (
    build_and_save_artifacts,
    create_movie_features,
    create_rating_stats,
)


def test_create_movie_features_adds_text_columns():
    movies = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Toy Story (1995)"],
            "genres": ["Adventure|Animation|Children"],
        }
    )

    result = create_movie_features(movies)

    assert result.loc[0, "title_clean"] == "Toy Story"
    assert result.loc[0, "genres_text"] == "Adventure Animation Children"
    assert "Toy Story" in result.loc[0, "feature_text"]


def test_create_rating_stats_aggregates_counts_and_means():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 1, 2],
            "rating": [4.0, 5.0, 3.0],
        }
    )

    result = create_rating_stats(ratings)

    assert result.loc[0, "movieId"] == 1
    assert result.loc[0, "rating_count"] == 2
    assert result.loc[0, "mean_rating"] == 4.5


def test_build_and_save_artifacts_writes_expected_files(tmp_path):
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Heat (1995)"],
            "genres": ["Adventure|Animation|Children", "Action|Crime|Thriller"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 2, 1],
            "rating": [4.0, 5.0, 3.0],
        }
    )

    paths = build_and_save_artifacts(
        MovieLensData(movies=movies, ratings=ratings),
        processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
    )

    assert paths.movies_features.exists()
    assert paths.rating_stats.exists()
    assert paths.tfidf_vectorizer.exists()
    assert paths.tfidf_matrix.exists()
    assert sparse.load_npz(paths.tfidf_matrix).shape[0] == 2
