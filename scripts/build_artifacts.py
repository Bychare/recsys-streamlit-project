from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_movielens
from src.preprocess import build_and_save_artifacts


def main() -> None:
    data = load_movielens()
    paths = build_and_save_artifacts(data)

    print("Artifacts built:")
    print(f"- {paths.movies_features}")
    print(f"- {paths.rating_stats}")
    print(f"- {paths.tfidf_vectorizer}")
    print(f"- {paths.tfidf_matrix}")


if __name__ == "__main__":
    main()
