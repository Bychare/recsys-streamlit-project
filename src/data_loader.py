"""Загрузка MovieLens и подготовка локальных папок с данными.

Модуль отвечает только за то, чтобы датасет оказался на диске и был прочитан
в pandas DataFrame. ML-логика и аналитика вынесены в другие файлы.
"""

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
REQUIRED_MOVIELENS_FILES = ("movies.csv", "ratings.csv")


@dataclass(frozen=True)
class MovieLensData:
    """Контейнер для двух основных таблиц MovieLens."""

    movies: pd.DataFrame
    ratings: pd.DataFrame


def ensure_data_dirs() -> None:
    """Создает папки для сырых и обработанных данных, если их еще нет."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _safe_extract(zip_file: ZipFile, target_dir: Path) -> None:
    """Распаковывает архив и защищает от path traversal внутри zip-файла."""
    target_dir = target_dir.resolve()
    for member in zip_file.infolist():
        member_path = (target_dir / member.filename).resolve()
        if target_dir not in member_path.parents and member_path != target_dir:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    zip_file.extractall(target_dir)


def _missing_movielens_files(dataset_dir: Path) -> list[str]:
    """Возвращает обязательные CSV, которых нет в распакованном датасете."""
    return [filename for filename in REQUIRED_MOVIELENS_FILES if not (dataset_dir / filename).is_file()]


def _validate_movielens_dir(dataset_dir: Path) -> None:
    """Проверяет, что распакованный MovieLens содержит нужные для приложения CSV."""
    missing_files = _missing_movielens_files(dataset_dir)
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"MovieLens dataset is incomplete. Missing files: {missing}")


def download_movielens(force: bool = False) -> Path:
    """Скачивает архив MovieLens, если его еще нет локально."""
    ensure_data_dirs()
    if force or not MOVIELENS_ZIP.exists():
        urlretrieve(MOVIELENS_URL, MOVIELENS_ZIP)
    return MOVIELENS_ZIP


def prepare_movielens(force_download: bool = False) -> Path:
    """Гарантирует, что архив скачан и распакован в `data/raw`."""
    ensure_data_dirs()
    if MOVIELENS_DIR.exists() and not force_download and not _missing_movielens_files(MOVIELENS_DIR):
        return MOVIELENS_DIR

    archive_path = download_movielens(force=force_download)
    with ZipFile(archive_path) as zip_file:
        _safe_extract(zip_file, RAW_DIR)
    _validate_movielens_dir(MOVIELENS_DIR)
    return MOVIELENS_DIR


def load_movielens(force_download: bool = False) -> MovieLensData:
    """Читает `movies.csv` и `ratings.csv` из подготовленного MovieLens."""
    dataset_dir = prepare_movielens(force_download=force_download)
    movies = pd.read_csv(dataset_dir / "movies.csv")
    ratings = pd.read_csv(dataset_dir / "ratings.csv")

    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    ratings["rating"] = ratings["rating"].astype(float)

    return MovieLensData(movies=movies, ratings=ratings)
