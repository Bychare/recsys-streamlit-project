from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_movielens
from src.evaluate import evaluate_popularity_baseline
from src.preprocess import MODELS_DIR


def main() -> None:
    data = load_movielens()
    metrics = evaluate_popularity_baseline(data.movies, data.ratings, k=10)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Popularity baseline metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {metrics_path}")


if __name__ == "__main__":
    main()
