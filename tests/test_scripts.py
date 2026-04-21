"""Тесты CLI-скриптов без реального скачивания данных."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts import build_artifacts, evaluate
from src.data_loader import MovieLensData
from src.preprocess import ArtifactPaths


def test_build_artifacts_script_prints_created_paths(monkeypatch, tmp_path, capsys):
    data = MovieLensData(
        movies=pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Drama"]}),
        ratings=pd.DataFrame({"userId": [1], "movieId": [1], "rating": [5.0]}),
    )
    paths = ArtifactPaths(
        movies_features=tmp_path / "movies_features.parquet",
        rating_stats=tmp_path / "rating_stats.parquet",
        tfidf_vectorizer=tmp_path / "tfidf_vectorizer.joblib",
        tfidf_matrix=tmp_path / "tfidf_matrix.npz",
    )

    monkeypatch.setattr(build_artifacts, "load_movielens", lambda: data)
    monkeypatch.setattr(build_artifacts, "build_and_save_artifacts", lambda loaded_data: paths)

    build_artifacts.main()

    output = capsys.readouterr().out
    assert "Artifacts built:" in output
    assert str(paths.movies_features) in output
    assert str(paths.tfidf_matrix) in output


def test_evaluate_script_writes_metrics_json(monkeypatch, tmp_path, capsys):
    data = SimpleNamespace(
        movies=pd.DataFrame({"movieId": [1], "title": ["A"], "genres": ["Drama"]}),
        ratings=pd.DataFrame({"userId": [1], "movieId": [1], "rating": [5.0]}),
    )
    metrics = {
        "popularity_baseline": {
            "k": 10,
            "evaluated_users": 1,
            "hit_rate": 1.0,
            "precision": 0.1,
            "coverage": 0.2,
            "novelty": 0.3,
        },
        "item_item_cf": {
            "k": 10,
            "evaluated_users": 1,
            "hit_rate": 0.0,
            "precision": 0.0,
            "coverage": 0.4,
            "novelty": 0.5,
        },
    }
    curves = pd.DataFrame(
        [
            {"model": "popularity_baseline", "k": 5, "hit_rate": 0.1},
            {"model": "item_item_cf", "k": 5, "hit_rate": 0.2},
        ]
    )

    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(evaluate, "load_movielens", lambda: data)
    monkeypatch.setattr(
        evaluate,
        "compare_recommenders",
        lambda movies, ratings, k, max_cf_users: metrics,
    )
    monkeypatch.setattr(
        evaluate,
        "evaluate_at_k_values",
        lambda movies, ratings, k_values, max_cf_users: curves,
    )

    evaluate.main()

    metrics_path = Path(tmp_path) / "metrics.json"
    curves_path = Path(tmp_path) / "metrics_by_k.csv"
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
    assert curves_path.exists()

    output = capsys.readouterr().out
    assert "Recommender comparison metrics:" in output
    assert f"Saved to: {metrics_path}" in output
    assert f"K curves saved to: {curves_path}" in output
