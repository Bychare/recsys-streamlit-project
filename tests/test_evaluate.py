"""Тесты offline-оценки рекомендательных моделей."""

from __future__ import annotations

import pytest

import pandas as pd

from src.evaluate import (
    compare_recommenders,
    evaluate_at_k_values,
    evaluate_item_item_cf,
    evaluate_popularity_baseline,
    evaluate_top_rated_baseline,
    leave_one_out_split,
)


def test_leave_one_out_split_uses_last_rating_per_user():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 10, 30, 40],
            "rating": [4.0, 5.0, 3.0, 4.0, 5.0],
            "timestamp": [100, 200, 100, 300, 100],
        }
    )

    split = leave_one_out_split(ratings)

    assert split.test[["userId", "movieId"]].to_dict("records") == [
        {"userId": 1, "movieId": 20},
        {"userId": 2, "movieId": 30},
    ]
    assert len(split.train) == 3


def test_leave_one_out_split_rejects_invalid_min_user_ratings():
    with pytest.raises(ValueError, match="min_user_ratings must be at least 2"):
        leave_one_out_split(pd.DataFrame(), min_user_ratings=1)


def test_evaluate_popularity_baseline_returns_metrics():
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Drama", "Comedy"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 10, 30, 20, 30],
            "rating": [5.0, 4.0, 5.0, 3.0, 4.0, 5.0],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )

    metrics = evaluate_popularity_baseline(movies, ratings, k=2)

    assert metrics["k"] == 2
    assert metrics["evaluated_users"] == 3
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.0 <= metrics["novelty"] <= 1.0


def test_evaluate_popularity_baseline_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_popularity_baseline(pd.DataFrame(), pd.DataFrame(), k=0)


def test_evaluate_popularity_baseline_handles_no_evaluated_users():
    movies = pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Drama"]})
    ratings = pd.DataFrame({"userId": [1], "movieId": [1], "rating": [5.0]})

    metrics = evaluate_popularity_baseline(movies, ratings, k=10)

    assert metrics == {
        "k": 10,
        "evaluated_users": 0,
        "hit_rate": 0.0,
        "precision": 0.0,
        "coverage": 0.0,
        "novelty": 0.0,
    }


def test_evaluate_top_rated_baseline_returns_metrics():
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Drama", "Comedy"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3, 3],
            "movieId": [10, 20, 10, 30, 20, 30],
            "rating": [5.0, 4.0, 5.0, 3.0, 4.0, 5.0],
            "timestamp": [1, 2, 1, 2, 1, 2],
        }
    )

    metrics = evaluate_top_rated_baseline(movies, ratings, k=2)

    assert metrics["k"] == 2
    assert metrics["evaluated_users"] == 3
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.0 <= metrics["novelty"] <= 1.0


def test_evaluate_top_rated_baseline_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_top_rated_baseline(pd.DataFrame(), pd.DataFrame(), k=0)


def test_evaluate_item_item_cf_returns_metrics():
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["A", "B", "C", "D"],
            "genres": ["Action", "Action", "Drama", "Action"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movieId": [10, 20, 30, 10, 20, 40, 10, 30, 40],
            "rating": [5.0, 4.0, 2.0, 5.0, 4.0, 5.0, 4.0, 3.0, 5.0],
            "timestamp": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )

    metrics = evaluate_item_item_cf(movies, ratings, k=2, max_users=2)

    assert metrics["k"] == 2
    assert 0 <= metrics["evaluated_users"] <= 2
    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["coverage"] <= 1.0
    assert 0.0 <= metrics["novelty"] <= 1.0


def test_evaluate_item_item_cf_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="k must be positive"):
        evaluate_item_item_cf(pd.DataFrame(), pd.DataFrame(), k=0)

    with pytest.raises(ValueError, match="max_users must be positive or None"):
        evaluate_item_item_cf(pd.DataFrame(), pd.DataFrame(), max_users=0)


def test_compare_recommenders_returns_named_models():
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Drama", "Comedy"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [5.0, 4.0, 5.0, 4.0],
            "timestamp": [1, 2, 1, 2],
        }
    )

    metrics = compare_recommenders(movies, ratings, k=2, max_cf_users=1)

    assert set(metrics) == {"popularity_baseline", "top_rated_baseline", "item_item_cf"}
    assert metrics["popularity_baseline"]["k"] == 2
    assert metrics["top_rated_baseline"]["k"] == 2
    assert metrics["item_item_cf"]["k"] == 2


def test_evaluate_at_k_values_returns_curve_rows():
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["A", "B", "C"],
            "genres": ["Drama", "Drama", "Comedy"],
        }
    )
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [5.0, 4.0, 5.0, 4.0],
            "timestamp": [1, 2, 1, 2],
        }
    )

    result = evaluate_at_k_values(movies, ratings, k_values=[1, 2], max_cf_users=1)

    assert set(result["model"]) == {"popularity_baseline", "top_rated_baseline", "item_item_cf"}
    assert result["k"].tolist() == [1, 1, 1, 2, 2, 2]
    assert {"coverage", "novelty"}.issubset(result.columns)
