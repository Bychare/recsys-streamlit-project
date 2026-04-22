"""Тесты загрузки MovieLens и безопасной работы с архивом."""

from __future__ import annotations

import pandas as pd
import pytest
from zipfile import ZipFile

from src import data_loader


def test_load_movielens_reads_local_dataset(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    dataset_dir = raw_dir / "ml-latest-small"
    dataset_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Heat (1995)"],
            "genres": ["Adventure|Animation|Children", None],
        }
    ).to_csv(dataset_dir / "movies.csv", index=False)
    pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 2, 1],
            "rating": [4, 5, 3],
            "timestamp": [100, 101, 102],
        }
    ).to_csv(dataset_dir / "ratings.csv", index=False)

    monkeypatch.setattr(data_loader, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(data_loader, "MOVIELENS_DIR", dataset_dir)

    data = data_loader.load_movielens()

    assert data.movies.shape == (2, 3)
    assert data.ratings.shape == (3, 4)
    assert data.movies.loc[1, "genres"] == "(no genres listed)"
    assert data.ratings["rating"].dtype == float


def test_download_movielens_uses_existing_archive(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    archive_path = raw_dir / "ml-latest-small.zip"
    raw_dir.mkdir()
    archive_path.write_bytes(b"existing")

    monkeypatch.setattr(data_loader, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_loader, "MOVIELENS_ZIP", archive_path)
    monkeypatch.setattr(
        data_loader,
        "urlretrieve",
        lambda url, filename: pytest.fail("urlretrieve should not be called"),
    )

    assert data_loader.download_movielens() == archive_path


def test_prepare_movielens_extracts_archive(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    archive_path = raw_dir / "ml-latest-small.zip"
    dataset_dir = raw_dir / "ml-latest-small"
    raw_dir.mkdir()

    with ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n1,A,Drama\n")
        zip_file.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n1,1,5,100\n")

    monkeypatch.setattr(data_loader, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(data_loader, "MOVIELENS_ZIP", archive_path)
    monkeypatch.setattr(data_loader, "MOVIELENS_DIR", dataset_dir)

    assert data_loader.prepare_movielens() == dataset_dir
    assert (dataset_dir / "movies.csv").exists()


def test_prepare_movielens_reextracts_incomplete_dataset(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    archive_path = raw_dir / "ml-latest-small.zip"
    dataset_dir = raw_dir / "ml-latest-small"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "movies.csv").write_text("movieId,title,genres\n1,A,Drama\n", encoding="utf-8")

    with ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("ml-latest-small/movies.csv", "movieId,title,genres\n1,A,Drama\n")
        zip_file.writestr("ml-latest-small/ratings.csv", "userId,movieId,rating,timestamp\n1,1,5,100\n")

    monkeypatch.setattr(data_loader, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_loader, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(data_loader, "MOVIELENS_ZIP", archive_path)
    monkeypatch.setattr(data_loader, "MOVIELENS_DIR", dataset_dir)

    assert data_loader.prepare_movielens() == dataset_dir
    assert (dataset_dir / "ratings.csv").exists()


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as zip_file:
        zip_file.writestr("../escape.csv", "bad")

    with ZipFile(archive_path) as zip_file:
        with pytest.raises(ValueError, match="Unsafe archive path"):
            data_loader._safe_extract(zip_file, tmp_path / "target")
