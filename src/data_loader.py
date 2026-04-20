from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_ZIP = RAW_DIR / "ml-latest-small.zip"
MOVIELENS_DIR = RAW_DIR / "ml-latest-small"


@dataclass(frozen=True)
class MovieLensData:
    movies: pd.DataFrame
    ratings: pd.DataFrame


def ensure_data_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _safe_extract(zip_file: ZipFile, target_dir: Path) -> None:
    target_dir = target_dir.resolve()
    for member in zip_file.infolist():
        member_path = (target_dir / member.filename).resolve()
        if target_dir not in member_path.parents and member_path != target_dir:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    zip_file.extractall(target_dir)


def download_movielens(force: bool = False) -> Path:
    ensure_data_dirs()
    if force or not MOVIELENS_ZIP.exists():
        urlretrieve(MOVIELENS_URL, MOVIELENS_ZIP)
    return MOVIELENS_ZIP


def prepare_movielens(force_download: bool = False) -> Path:
    ensure_data_dirs()
    if MOVIELENS_DIR.exists() and not force_download:
        return MOVIELENS_DIR

    archive_path = download_movielens(force=force_download)
    with ZipFile(archive_path) as zip_file:
        _safe_extract(zip_file, RAW_DIR)
    return MOVIELENS_DIR


def load_movielens(force_download: bool = False) -> MovieLensData:
    dataset_dir = prepare_movielens(force_download=force_download)
    movies = pd.read_csv(dataset_dir / "movies.csv")
    ratings = pd.read_csv(dataset_dir / "ratings.csv")

    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    ratings["rating"] = ratings["rating"].astype(float)

    return MovieLensData(movies=movies, ratings=ratings)
