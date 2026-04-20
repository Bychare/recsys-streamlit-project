from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import get_artifact_status, load_metrics
from src.data_loader import MovieLensData, load_movielens
from src.evaluate import evaluate_popularity_baseline
from src.preprocess import MODELS_DIR


st.set_page_config(page_title="Model Evaluation", layout="wide")


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    return load_movielens()


def show_metrics(metrics: dict[str, float | int]) -> None:
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Evaluated users", f"{int(metrics['evaluated_users']):,}".replace(",", " "))
    metric_2.metric(f"Hit rate@{int(metrics['k'])}", f"{float(metrics['hit_rate']):.3f}")
    metric_3.metric(f"Precision@{int(metrics['k'])}", f"{float(metrics['precision']):.4f}")


st.title("Model Evaluation")
st.caption("Offline-оценка popularity baseline и состояние подготовленных артефактов.")

stored_metrics = load_metrics()
if stored_metrics:
    st.subheader("Сохраненные метрики")
    show_metrics(stored_metrics)
else:
    st.warning("Файл `models/metrics.json` пока не найден.")

st.subheader("Пересчет baseline")
control_1, control_2, control_3 = st.columns(3)
with control_1:
    k = st.slider("K", min_value=5, max_value=50, value=10, step=5)
with control_2:
    min_user_ratings = st.slider("Мин. оценок пользователя", min_value=2, max_value=20, value=2)
with control_3:
    min_movie_ratings = st.slider("Мин. оценок фильма", min_value=1, max_value=100, value=1)

if st.button("Пересчитать", type="primary"):
    data = get_data()
    metrics = evaluate_popularity_baseline(
        data.movies,
        data.ratings,
        k=k,
        min_user_ratings=min_user_ratings,
        min_movie_ratings=min_movie_ratings,
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    st.success(f"Метрики сохранены: {metrics_path.relative_to(PROJECT_ROOT)}")
    show_metrics(metrics)

st.subheader("Артефакты")
artifact_status = get_artifact_status().rename(
    columns={
        "artifact": "Артефакт",
        "path": "Путь",
        "exists": "Есть",
        "size_mb": "Размер, MB",
    }
)
st.dataframe(artifact_status, use_container_width=True, hide_index=True)
