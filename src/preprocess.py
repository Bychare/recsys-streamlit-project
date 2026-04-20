from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import MovieLensData, PROCESSED_DIR, PROJECT_ROOT, ensure_data_dirs
from src.recommend import build_feature_text


MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ArtifactPaths:
    movies_features: Path
    rating_stats: Path
    tfidf_vectorizer: Path
    tfidf_matrix: Path


def create_movie_features(movies: pd.DataFrame) -> pd.DataFrame:
    result = movies.copy()
    result["genres"] = result["genres"].fillna("(no genres listed)")
    result["genres_text"] = result["genres"].str.replace("|", " ", regex=False)
    result["title_clean"] = result["title"].fillna("").str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    result["feature_text"] = build_feature_text(result)
    return result


def create_rating_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby("movieId", as_index=False)
        .agg(rating_count=("rating", "size"), mean_rating=("rating", "mean"))
        .sort_values(["rating_count", "mean_rating"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_content_artifacts(movies_features: pd.DataFrame) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(movies_features["feature_text"])
    return vectorizer, matrix


def build_and_save_artifacts(
    data: MovieLensData,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> ArtifactPaths:
    ensure_data_dirs()
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    movies_features = create_movie_features(data.movies)
    rating_stats = create_rating_stats(data.ratings)
    vectorizer, matrix = build_content_artifacts(movies_features)

    paths = ArtifactPaths(
        movies_features=processed_dir / "movies_features.parquet",
        rating_stats=processed_dir / "rating_stats.parquet",
        tfidf_vectorizer=models_dir / "tfidf_vectorizer.joblib",
        tfidf_matrix=models_dir / "tfidf_matrix.npz",
    )

    movies_features.to_parquet(paths.movies_features, index=False)
    rating_stats.to_parquet(paths.rating_stats, index=False)
    joblib.dump(vectorizer, paths.tfidf_vectorizer)
    sparse.save_npz(paths.tfidf_matrix, matrix)

    return paths
