from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import MovieLensData, load_movielens
from src.recommend import (
    get_popular_movies,
    get_similar_movies,
    get_top_rated_movies,
)


st.set_page_config(
    page_title="recsys-streamlit-project",
    layout="wide",
)


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    return load_movielens()


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    visible_columns = {
        "title": "Фильм",
        "genres": "Жанры",
        "rating_count": "Оценок",
        "mean_rating": "Средняя оценка",
        "similarity": "Сходство",
    }
    result = df[[column for column in visible_columns if column in df.columns]].rename(
        columns=visible_columns
    )
    for column in ("Средняя оценка", "Сходство"):
        if column in result.columns:
            result[column] = result[column].round(3)
    return result


st.title("recsys-streamlit-project")
st.caption("Демо рекомендательной системы на MovieLens latest-small.")

try:
    data = get_data()
except Exception as exc:
    st.error("Не удалось загрузить MovieLens. Проверьте подключение к интернету и повторите запуск.")
    st.exception(exc)
    st.stop()

movies = data.movies
ratings = data.ratings

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Фильмов", f"{movies['movieId'].nunique():,}".replace(",", " "))
metric_2.metric("Оценок", f"{len(ratings):,}".replace(",", " "))
metric_3.metric("Пользователей", f"{ratings['userId'].nunique():,}".replace(",", " "))

with st.sidebar:
    st.header("Настройки")
    mode = st.radio(
        "Сценарий",
        ["Популярные фильмы", "Top-rated", "Похожие фильмы"],
    )
    limit = st.slider("Количество рекомендаций", min_value=5, max_value=30, value=10, step=5)
    min_ratings = st.slider(
        "Минимум оценок",
        min_value=1,
        max_value=300,
        value=50,
        step=5,
        disabled=mode == "Похожие фильмы",
    )

if mode == "Популярные фильмы":
    st.subheader("Популярные фильмы")
    recommendations = get_popular_movies(movies, ratings, min_ratings=min_ratings, limit=limit)
    st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)

elif mode == "Top-rated":
    st.subheader("Top-rated фильмы")
    recommendations = get_top_rated_movies(movies, ratings, min_ratings=min_ratings, limit=limit)
    st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)

else:
    st.subheader("Похожие фильмы")
    movie_options = movies.sort_values("title")[["movieId", "title"]].reset_index(drop=True)
    selected_title = st.selectbox("Выберите фильм", movie_options["title"].tolist(), index=0)
    selected_movie_id = int(movie_options.loc[movie_options["title"] == selected_title, "movieId"].iloc[0])

    recommendations = get_similar_movies(selected_movie_id, movies, limit=limit)
    st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)
