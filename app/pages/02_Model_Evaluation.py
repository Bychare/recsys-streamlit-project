from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import get_artifact_status, load_metrics, load_metrics_by_k
from src.data_loader import MovieLensData, load_movielens
from src.evaluate import compare_recommenders, evaluate_at_k_values
from src.preprocess import MODELS_DIR


st.set_page_config(page_title="Model Evaluation", layout="wide")


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    return load_movielens()


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    if "hit_rate" in metrics:
        rows = [{"model": "popularity_baseline", **metrics}]
    else:
        rows = [{"model": model_name, **model_metrics} for model_name, model_metrics in metrics.items()]

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    for column in ("coverage", "novelty"):
        if column not in result.columns:
            result[column] = 0.0

    result = result[["model", "k", "evaluated_users", "hit_rate", "precision", "coverage", "novelty"]]
    for column in ("hit_rate", "precision", "coverage", "novelty"):
        result[column] = result[column].astype(float).round(4)
    return result.rename(
        columns={
            "model": "Модель",
            "k": "K",
            "evaluated_users": "Пользователей",
            "hit_rate": "Hit rate",
            "precision": "Precision",
            "coverage": "Coverage",
            "novelty": "Novelty",
        }
    )


def show_metrics_table(metrics: dict) -> None:
    st.dataframe(metrics_to_dataframe(metrics), use_container_width=True, hide_index=True)


def show_k_curves(curves: pd.DataFrame) -> None:
    if curves.empty:
        st.warning("Недостаточно данных для построения кривых.")
        return

    chart_data = curves[["model", "k", "hit_rate", "precision", "coverage", "novelty"]].copy()
    chart_data = chart_data.rename(
        columns={
            "model": "Модель",
            "k": "K",
            "hit_rate": "Hit rate",
            "precision": "Precision",
            "coverage": "Coverage",
            "novelty": "Novelty",
        }
    )

    metric_tabs = st.tabs(["Hit rate", "Precision", "Coverage", "Novelty"])
    for tab, metric in zip(metric_tabs, ["Hit rate", "Precision", "Coverage", "Novelty"]):
        with tab:
            pivot = chart_data.pivot(index="K", columns="Модель", values=metric)
            st.line_chart(pivot, use_container_width=True)


st.title("Model Evaluation")
st.caption("Offline-сравнение popularity baseline и item-item collaborative filtering.")

stored_metrics = load_metrics()
if stored_metrics:
    st.subheader("Сохраненные метрики")
    show_metrics_table(stored_metrics)
else:
    st.warning("Файл `models/metrics.json` пока не найден.")

stored_curves = load_metrics_by_k()
if not stored_curves.empty:
    st.subheader("Сохраненные кривые по K")
    show_k_curves(stored_curves)

st.subheader("Пересчет сравнения")
control_1, control_2, control_3, control_4 = st.columns(4)
with control_1:
    k = st.slider("K", min_value=5, max_value=50, value=10, step=5)
with control_2:
    min_user_ratings = st.slider("Мин. оценок пользователя", min_value=2, max_value=20, value=2)
with control_3:
    min_movie_ratings = st.slider("Мин. оценок фильма", min_value=1, max_value=100, value=1)
with control_4:
    max_cf_users = st.slider("Пользователей для CF", min_value=25, max_value=500, value=100, step=25)

min_positive_rating = st.slider("Порог положительной оценки для CF", min_value=0.5, max_value=5.0, value=4.0, step=0.5)

if st.button("Пересчитать сравнение", type="primary"):
    data = get_data()
    with st.spinner("Считаем offline-метрики..."):
        metrics = compare_recommenders(
            data.movies,
            data.ratings,
            k=k,
            min_user_ratings=min_user_ratings,
            min_movie_ratings=min_movie_ratings,
            min_positive_rating=min_positive_rating,
            max_cf_users=max_cf_users,
        )
        curves = evaluate_at_k_values(
            data.movies,
            data.ratings,
            k_values=[5, 10, 20, 30, 50],
            min_user_ratings=min_user_ratings,
            min_movie_ratings=min_movie_ratings,
            min_positive_rating=min_positive_rating,
            max_cf_users=max_cf_users,
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    st.success(f"Метрики сохранены: {metrics_path.relative_to(PROJECT_ROOT)}")
    show_metrics_table(metrics)
    st.subheader("Кривые по K")
    show_k_curves(curves)

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
