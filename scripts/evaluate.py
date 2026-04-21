"""CLI-команда для offline-оценки рекомендателей.

Запуск:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Нужен для импортов из `src`, если скрипт запускают напрямую.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_movielens
from src.evaluate import compare_recommenders, evaluate_at_k_values
from src.preprocess import MODELS_DIR


def main() -> None:
    """Считает метрики моделей и сохраняет JSON/CSV в `models/`."""
    data = load_movielens()
    metrics = compare_recommenders(data.movies, data.ratings, k=10, max_cf_users=100)
    curves = evaluate_at_k_values(
        data.movies,
        data.ratings,
        k_values=[5, 10, 20, 30, 50],
        max_cf_users=100,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "metrics.json"
    curves_path = MODELS_DIR / "metrics_by_k.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    curves.to_csv(curves_path, index=False)

    print("Recommender comparison metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {metrics_path}")
    print(f"K curves saved to: {curves_path}")


if __name__ == "__main__":
    main()
