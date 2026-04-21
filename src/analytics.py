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


@dataclass(frozen=True)
class LongTailSummary:
    movie_count: int
    head_movie_count: int
    tail_movie_count: int
    head_rating_share: float
    tail_movie_share: float


DEFAULT_ARTIFACT_PATHS = {
    "movies_features": PROJECT_ROOT / "data" / "processed" / "movies_features.parquet",
    "rating_stats": PROJECT_ROOT / "data" / "processed" / "rating_stats.parquet",
    "tfidf_vectorizer": PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib",
    "tfidf_matrix": PROJECT_ROOT / "models" / "tfidf_matrix.npz",
    "metrics": PROJECT_ROOT / "models" / "metrics.json",
    "metrics_by_k": PROJECT_ROOT / "models" / "metrics_by_k.csv",
}

ACTIVITY_BUCKETS = [
    (0, 0, "0"),
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 9, "5-9"),
    (10, 19, "10-19"),
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 199, "100-199"),
    (200, 499, "200-499"),
    (500, 999, "500-999"),
    (1000, 1999, "1000-1999"),
    (2000, 4999, "2000-4999"),
]


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


def get_user_activity_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby("userId", as_index=False)
        .agg(rating_count=("movieId", "size"))
        .sort_values(["rating_count", "userId"], ascending=[False, True])
        .reset_index(drop=True)
    )


def get_movie_activity_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby("movieId", as_index=False)
        .agg(rating_count=("userId", "size"), mean_rating=("rating", "mean"))
        .sort_values(["rating_count", "movieId"], ascending=[False, True])
        .reset_index(drop=True)
    )


def get_movie_popularity_curve(ratings: pd.DataFrame) -> pd.DataFrame:
    movie_activity = get_movie_activity_distribution(ratings)
    if movie_activity.empty:
        return pd.DataFrame(columns=["movie_share", "rating_share"])

    total_movies = len(movie_activity)
    total_ratings = movie_activity["rating_count"].sum()
    result = movie_activity.reset_index(drop=True).copy()
    result["movie_share"] = (result.index + 1) / total_movies
    result["rating_share"] = result["rating_count"].cumsum() / total_ratings
    return result[["movie_share", "rating_share"]]


def get_movie_rating_scatter(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings: int = 5,
) -> pd.DataFrame:
    if min_ratings < 1:
        raise ValueError("min_ratings must be positive")

    movie_activity = get_movie_activity_distribution(ratings)
    result = movies.merge(movie_activity, on="movieId", how="inner")
    result["primary_genre"] = result["genres"].fillna("(no genres listed)").str.split("|").str[0]
    return (
        result[result["rating_count"] >= min_ratings]
        .sort_values(["rating_count", "mean_rating", "title"], ascending=[False, False, True])
        .reset_index(drop=True)
    )


def _activity_bucket(value: int) -> tuple[int, str]:
    for index, (left, right, label) in enumerate(ACTIVITY_BUCKETS):
        if left <= value <= right:
            return index, label
    return len(ACTIVITY_BUCKETS), f"{ACTIVITY_BUCKETS[-1][1] + 1}+"


def get_activity_histogram(
    activity: pd.DataFrame,
    value_column: str = "rating_count",
    bins: int = 10,
) -> pd.DataFrame:
    if bins < 1:
        raise ValueError("bins must be positive")
    if activity.empty:
        return pd.DataFrame(columns=["bucket", "bucket_order", "entity_count"])

    result = activity.copy()
    bucket_values = result[value_column].astype(int).map(_activity_bucket)
    result["bucket_order"] = bucket_values.map(lambda value: value[0])
    result["bucket"] = bucket_values.map(lambda value: value[1])

    grouped = (
        result.groupby("bucket", as_index=False, sort=False)
        .agg(bucket_order=("bucket_order", "min"), entity_count=(value_column, "size"))
        .sort_values("bucket_order")
        .reset_index(drop=True)
    )
    return grouped


def get_genre_rating_stats(movies: pd.DataFrame, ratings: pd.DataFrame, min_ratings: int = 20) -> pd.DataFrame:
    movie_genres = movies[["movieId", "genres"]].copy()
    movie_genres["genre"] = movie_genres["genres"].fillna("(no genres listed)").str.split("|")
    movie_genres = movie_genres.explode("genre")
    movie_genres = movie_genres[movie_genres["genre"] != "(no genres listed)"]

    result = ratings.merge(movie_genres[["movieId", "genre"]], on="movieId", how="inner")
    result = (
        result.groupby("genre", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
        .query("rating_count >= @min_ratings")
        .sort_values(["mean_rating", "rating_count", "genre"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return result


def get_long_tail_summary(ratings: pd.DataFrame, head_quantile: float = 0.8) -> LongTailSummary:
    if not 0 < head_quantile < 1:
        raise ValueError("head_quantile must be between 0 and 1")

    movie_activity = get_movie_activity_distribution(ratings)
    if movie_activity.empty:
        return LongTailSummary(0, 0, 0, 0.0, 0.0)

    threshold = movie_activity["rating_count"].quantile(head_quantile)
    head = movie_activity[movie_activity["rating_count"] >= threshold]
    tail = movie_activity[movie_activity["rating_count"] < threshold]
    total_ratings = movie_activity["rating_count"].sum()

    return LongTailSummary(
        movie_count=int(len(movie_activity)),
        head_movie_count=int(len(head)),
        tail_movie_count=int(len(tail)),
        head_rating_share=float(head["rating_count"].sum() / total_ratings) if total_ratings else 0.0,
        tail_movie_share=float(len(tail) / len(movie_activity)) if len(movie_activity) else 0.0,
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


def load_metrics_by_k(path: Path | None = None) -> pd.DataFrame:
    metrics_path = path or DEFAULT_ARTIFACT_PATHS["metrics_by_k"]
    if not metrics_path.exists():
        return pd.DataFrame()
    return pd.read_csv(metrics_path)
