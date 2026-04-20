from __future__ import annotations

import json

import pandas as pd

from src.analytics import (
    get_artifact_status,
    get_dataset_summary,
    get_genre_counts,
    get_rating_activity_by_year,
    get_rating_distribution,
    get_top_users,
    load_metrics,
)


def test_get_dataset_summary_calculates_counts_and_sparsity():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Comedy|Drama", "(no genres listed)"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 2, 1],
            "rating": [4.0, 5.0, 3.0],
        }
    )

    summary = get_dataset_summary(movies, ratings)

    assert summary.movie_count == 3
    assert summary.rating_count == 3
    assert summary.user_count == 2
    assert summary.genre_count == 2
    assert summary.mean_rating == 4.0
    assert summary.sparsity == 0.5


def test_get_rating_distribution_orders_by_rating():
    ratings = pd.DataFrame({"movieId": [1, 2, 3], "rating": [5.0, 4.0, 5.0]})

    result = get_rating_distribution(ratings)

    assert result.to_dict("records") == [
        {"rating": 4.0, "rating_count": 1},
        {"rating": 5.0, "rating_count": 2},
    ]


def test_get_genre_counts_explodes_genres_and_limits():
    movies = pd.DataFrame(
        {
            "genres": [
                "Drama|Comedy",
                "Drama",
                "Action",
                "(no genres listed)",
            ]
        }
    )

    result = get_genre_counts(movies, limit=2)

    assert result.to_dict("records") == [
        {"genre": "Drama", "movie_count": 2},
        {"genre": "Action", "movie_count": 1},
    ]


def test_get_rating_activity_by_year_handles_timestamp_and_missing_column():
    ratings = pd.DataFrame(
        {
            "movieId": [1, 2],
            "timestamp": [946684800, 978307200],
        }
    )

    result = get_rating_activity_by_year(ratings)
    empty = get_rating_activity_by_year(pd.DataFrame({"movieId": [1]}))

    assert result.to_dict("records") == [
        {"year": 2000, "rating_count": 1},
        {"year": 2001, "rating_count": 1},
    ]
    assert empty.empty
    assert empty.columns.tolist() == ["year", "rating_count"]


def test_get_top_users_orders_by_activity():
    ratings = pd.DataFrame(
        {
            "userId": [2, 1, 1, 2, 2],
            "movieId": [1, 1, 2, 2, 3],
            "rating": [3.0, 5.0, 4.0, 2.0, 4.0],
        }
    )

    result = get_top_users(ratings, limit=1)

    assert result.loc[0, "userId"] == 2
    assert result.loc[0, "rating_count"] == 3
    assert result.loc[0, "mean_rating"] == 3.0


def test_get_artifact_status_reports_existing_and_missing_files(tmp_path):
    existing = tmp_path / "artifact.txt"
    existing.write_text("content" * 512, encoding="utf-8")
    missing = tmp_path / "missing.txt"

    result = get_artifact_status({"existing": existing, "missing": missing})

    assert result["artifact"].tolist() == ["existing", "missing"]
    assert result["exists"].tolist() == [True, False]
    assert result.loc[0, "size_mb"] > 0


def test_load_metrics_reads_json_and_returns_none_for_missing(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics = {"k": 10, "hit_rate": 0.5}
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    assert load_metrics(metrics_path) == metrics
    assert load_metrics(tmp_path / "missing.json") is None
