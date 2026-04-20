from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (
    get_dataset_summary,
    get_genre_counts,
    get_rating_activity_by_year,
    get_rating_distribution,
    get_top_users,
)
from src.data_loader import MovieLensData, load_movielens
from src.recommend import get_popular_movies


st.set_page_config(page_title="Data Overview", layout="wide")


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    return load_movielens()


data = get_data()
movies = data.movies
ratings = data.ratings
summary = get_dataset_summary(movies, ratings)

st.title("Data Overview")
st.caption("MovieLens latest-small: структура датасета, плотность оценок и базовые распределения.")

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Фильмов", f"{summary.movie_count:,}".replace(",", " "))
metric_2.metric("Оценок", f"{summary.rating_count:,}".replace(",", " "))
metric_3.metric("Пользователей", f"{summary.user_count:,}".replace(",", " "))
metric_4.metric("Sparsity", f"{summary.sparsity:.2%}")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Распределение оценок")
    rating_distribution = get_rating_distribution(ratings).set_index("rating")
    st.bar_chart(rating_distribution, y="rating_count", use_container_width=True)

with chart_right:
    st.subheader("Топ жанров")
    genre_counts = get_genre_counts(movies, limit=12).set_index("genre")
    st.bar_chart(genre_counts, y="movie_count", use_container_width=True)

st.subheader("Активность по годам")
activity_by_year = get_rating_activity_by_year(ratings).set_index("year")
st.line_chart(activity_by_year, y="rating_count", use_container_width=True)

table_left, table_right = st.columns(2)

with table_left:
    st.subheader("Самые оцениваемые фильмы")
    popular_movies = get_popular_movies(movies, ratings, min_ratings=1, limit=10)
    st.dataframe(
        popular_movies[["title", "genres", "rating_count", "mean_rating"]].rename(
            columns={
                "title": "Фильм",
                "genres": "Жанры",
                "rating_count": "Оценок",
                "mean_rating": "Средняя оценка",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with table_right:
    st.subheader("Самые активные пользователи")
    top_users = get_top_users(ratings, limit=10).rename(
        columns={
            "userId": "Пользователь",
            "rating_count": "Оценок",
            "mean_rating": "Средняя оценка",
        }
    )
    top_users["Средняя оценка"] = top_users["Средняя оценка"].round(3)
    st.dataframe(top_users, use_container_width=True, hide_index=True)
