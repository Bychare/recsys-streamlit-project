"""Тесты аналитических функций для страниц Streamlit."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.analytics import (
    get_activity_histogram,
    get_artifact_status,
    get_dataset_summary,
    get_genre_counts,
    get_genre_rating_stats,
    get_long_tail_summary,
    get_movie_activity_distribution,
    get_movie_popularity_curve,
    get_movie_rating_scatter,
    get_rating_activity_by_year,
    get_rating_distribution,
    get_top_users,
    get_user_activity_distribution,
    load_metrics,
    load_metrics_by_k,
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


def test_activity_distributions_and_histogram():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 3, 3, 3],
            "movieId": [10, 20, 10, 10, 20, 30],
            "rating": [4.0, 5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )

    user_activity = get_user_activity_distribution(ratings)
    movie_activity = get_movie_activity_distribution(ratings)
    histogram = get_activity_histogram(user_activity, bins=2)

    assert user_activity["rating_count"].tolist() == [3, 2, 1]
    assert movie_activity.loc[0, "movieId"] == 10
    assert histogram["entity_count"].sum() == 3
    assert histogram["bucket"].tolist() == ["1", "2-4"]
    assert histogram["bucket_order"].tolist() == [1, 2]


def test_get_movie_popularity_curve_builds_cumulative_shares():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 1],
            "movieId": [10, 10, 10, 20, 20, 30],
            "rating": [4.0, 5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )

    result = get_movie_popularity_curve(ratings)

    assert result.loc[0, "movie_share"] == pytest.approx(1 / 3)
    assert result.loc[0, "rating_share"] == pytest.approx(3 / 6)
    assert result.loc[len(result) - 1, "movie_share"] == 1.0
    assert result.loc[len(result) - 1, "rating_share"] == 1.0


def test_get_movie_rating_scatter_filters_and_adds_primary_genre():
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["A", "B", "C"],
            "genres": ["Drama|Comedy", "Action", "Sci-Fi"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 1],
            "movieId": [10, 10, 10, 20, 20, 30],
            "rating": [4.0, 5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )

    result = get_movie_rating_scatter(movies, ratings, min_ratings=2)

    assert result["movieId"].tolist() == [10, 20]
    assert result.loc[0, "primary_genre"] == "Drama"
    assert result.loc[0, "rating_count"] == 3


def test_get_movie_rating_scatter_rejects_invalid_min_ratings():
    with pytest.raises(ValueError, match="min_ratings must be positive"):
        get_movie_rating_scatter(pd.DataFrame(), pd.DataFrame(), min_ratings=0)


def test_get_activity_histogram_rejects_invalid_bins():
    with pytest.raises(ValueError, match="bins must be positive"):
        get_activity_histogram(pd.DataFrame({"rating_count": [1]}), bins=0)


def test_get_genre_rating_stats_aggregates_by_genre():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "genres": ["Drama|Comedy", "Drama"],
        }
    )
    ratings = pd.DataFrame(
        {
            "movieId": [1, 1, 2],
            "rating": [5.0, 4.0, 3.0],
        }
    )

    result = get_genre_rating_stats(movies, ratings, min_ratings=1)

    assert set(result["genre"]) == {"Drama", "Comedy"}
    assert result.loc[result["genre"] == "Comedy", "mean_rating"].iloc[0] == 4.5


def test_get_long_tail_summary_calculates_head_and_tail():
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 1],
            "movieId": [10, 10, 10, 20, 20, 30],
            "rating": [4.0, 5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )

    summary = get_long_tail_summary(ratings, head_quantile=0.5)

    assert summary.movie_count == 3
    assert summary.head_movie_count >= 1
    assert 0.0 <= summary.head_rating_share <= 1.0
    assert 0.0 <= summary.tail_movie_share <= 1.0


def test_get_long_tail_summary_rejects_invalid_quantile():
    with pytest.raises(ValueError, match="head_quantile must be between 0 and 1"):
        get_long_tail_summary(pd.DataFrame(), head_quantile=1.0)


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


def test_load_metrics_by_k_reads_csv_and_returns_empty_for_missing(tmp_path):
    metrics_path = tmp_path / "metrics_by_k.csv"
    pd.DataFrame([{"model": "a", "k": 5, "hit_rate": 0.1}]).to_csv(metrics_path, index=False)

    result = load_metrics_by_k(metrics_path)
    missing = load_metrics_by_k(tmp_path / "missing.csv")

    assert result.to_dict("records") == [{"model": "a", "k": 5, "hit_rate": 0.1}]
    assert missing.empty
