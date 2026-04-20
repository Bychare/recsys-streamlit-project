from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data_loader import PROJECT_ROOT


@dataclass(frozen=True)
class DatasetSummary:
    movie_count: int
    rating_count: int
    user_count: int
    genre_count: int
    mean_rating: float
    sparsity: float


DEFAULT_ARTIFACT_PATHS = {
    "movies_features": PROJECT_ROOT / "data" / "processed" / "movies_features.parquet",
    "rating_stats": PROJECT_ROOT / "data" / "processed" / "rating_stats.parquet",
    "tfidf_vectorizer": PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib",
    "tfidf_matrix": PROJECT_ROOT / "models" / "tfidf_matrix.npz",
    "metrics": PROJECT_ROOT / "models" / "metrics.json",
}


def _display_path(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def get_dataset_summary(movies: pd.DataFrame, ratings: pd.DataFrame) -> DatasetSummary:
    movie_count = int(movies["movieId"].nunique())
    rating_count = int(len(ratings))
    user_count = int(ratings["userId"].nunique())
    genre_count = int(get_genre_counts(movies)["genre"].nunique())
    possible_interactions = movie_count * user_count
    sparsity = 1 - rating_count / possible_interactions if possible_interactions else 0.0

    return DatasetSummary(
        movie_count=movie_count,
        rating_count=rating_count,
        user_count=user_count,
        genre_count=genre_count,
        mean_rating=float(ratings["rating"].mean()) if rating_count else 0.0,
        sparsity=float(sparsity),
    )


def get_rating_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby("rating", as_index=False)
        .agg(rating_count=("movieId", "size"))
        .sort_values("rating")
        .reset_index(drop=True)
    )


def get_genre_counts(movies: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    genres = (
        movies["genres"]
        .fillna("(no genres listed)")
        .str.split("|")
        .explode()
        .rename("genre")
        .to_frame()
    )
    genres = genres[genres["genre"] != "(no genres listed)"]
    result = (
        genres.groupby("genre", as_index=False)
        .size()
        .rename(columns={"size": "movie_count"})
        .sort_values(["movie_count", "genre"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return result.head(limit).reset_index(drop=True) if limit else result


def get_rating_activity_by_year(ratings: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in ratings.columns:
        return pd.DataFrame(columns=["year", "rating_count"])

    result = ratings.copy()
    result["year"] = pd.to_datetime(result["timestamp"], unit="s").dt.year
    return (
        result.groupby("year", as_index=False)
        .agg(rating_count=("movieId", "size"))
        .sort_values("year")
        .reset_index(drop=True)
    )


def get_top_users(ratings: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return (
        ratings.groupby("userId", as_index=False)
        .agg(rating_count=("movieId", "size"), mean_rating=("rating", "mean"))
        .sort_values(["rating_count", "userId"], ascending=[False, True])
        .head(limit)
        .reset_index(drop=True)
    )


def get_artifact_status(paths: dict[str, Path] | None = None) -> pd.DataFrame:
    paths = paths or DEFAULT_ARTIFACT_PATHS
    rows = []
    for name, path in paths.items():
        exists = path.exists()
        rows.append(
            {
                "artifact": name,
                "path": _display_path(path),
                "exists": exists,
                "size_mb": round(path.stat().st_size / 1024 / 1024, 3) if exists else 0.0,
            }
        )
    return pd.DataFrame(rows)


def load_metrics(path: Path | None = None) -> dict[str, float | int] | None:
    metrics_path = path or DEFAULT_ARTIFACT_PATHS["metrics"]
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))
