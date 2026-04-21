"""Страница с обзором MovieLens.

Здесь нет моделирования: только быстрый анализ датасета, который помогает понять,
насколько разрежены оценки, какие жанры встречаются чаще и почему популярность
фильма надо учитывать вместе со средней оценкой.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import streamlit as st

# У страниц Streamlit та же проблема с импортами, что и у Home.py:
# файл лежит глубже в `app/pages`, поэтому явно добавляем корень проекта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics import (
    get_activity_histogram,
    get_dataset_summary,
    get_genre_counts,
    get_genre_rating_stats,
    get_long_tail_summary,
    get_movie_rating_scatter,
    get_rating_activity_by_year,
    get_rating_distribution,
    get_top_users,
    get_user_activity_distribution,
)
from src.data_loader import MovieLensData, load_movielens
from src.recommend import get_popular_movies


st.set_page_config(page_title="Data Overview", layout="wide")


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    """Кешируем датасет, чтобы переключение страниц не перечитывало CSV заново."""
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

tail_summary = get_long_tail_summary(ratings)
# Head/tail здесь считается по числу оценок у фильма: head — наиболее заметные фильмы,
# tail — длинный хвост фильмов с меньшей пользовательской активностью.
tail_1, tail_2, tail_3, tail_4 = st.columns(4)
tail_1.metric("Head фильмов", f"{tail_summary.head_movie_count:,}".replace(",", " "))
tail_2.metric("Tail фильмов", f"{tail_summary.tail_movie_count:,}".replace(",", " "))
tail_3.metric("Доля tail", f"{tail_summary.tail_movie_share:.1%}")
tail_4.metric("Оценок в head", f"{tail_summary.head_rating_share:.1%}")

chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Распределение оценок")
    rating_distribution = get_rating_distribution(ratings).set_index("rating")
    st.bar_chart(rating_distribution, y="rating_count", use_container_width=True)

with chart_right:
    st.subheader("Доля топ-жанров")
    st.caption("Donut chart показывает долю жанровых меток среди 12 самых частых жанров.")
    genre_counts = get_genre_counts(movies, limit=12)
    # Donut хорошо подходит именно для долей: это не рейтинг жанров, а структура каталога.
    genre_pie = (
        alt.Chart(genre_counts)
        .mark_arc(innerRadius=55)
        .encode(
            theta=alt.Theta("movie_count:Q", title="Фильмов"),
            color=alt.Color("genre:N", sort=genre_counts["genre"].tolist(), title="Жанр"),
            order=alt.Order("movie_count:Q", sort="descending"),
            tooltip=[
                alt.Tooltip("genre:N", title="Жанр"),
                alt.Tooltip("movie_count:Q", title="Фильмов"),
            ],
        )
    )
    st.altair_chart(genre_pie, use_container_width=True)

activity_left, activity_right = st.columns(2)

with activity_left:
    st.subheader("Сколько оценок оставляют пользователи")
    st.caption("Ось X: диапазон числа оценок на одного пользователя. Ось Y: сколько пользователей в этом диапазоне.")
    user_activity = get_user_activity_distribution(ratings)
    user_activity_hist = get_activity_histogram(user_activity, bins=12)
    # Altair нужен здесь, чтобы явно зафиксировать порядок bucket'ов на оси X.
    user_activity_chart = (
        alt.Chart(user_activity_hist)
        .mark_bar()
        .encode(
            x=alt.X("bucket:N", sort=alt.SortField("bucket_order", order="ascending"), title="Оценок на пользователя"),
            y=alt.Y("entity_count:Q", title="Пользователей"),
            tooltip=[
                alt.Tooltip("bucket:N", title="Оценок"),
                alt.Tooltip("entity_count:Q", title="Пользователей"),
            ],
        )
    )
    st.altair_chart(user_activity_chart, use_container_width=True)

with activity_right:
    st.subheader("Рейтинг vs популярность")
    st.caption("Каждая точка — фильм. X: число оценок. Y: средняя оценка. Видно, какие оценки надежнее.")
    rating_scatter = get_movie_rating_scatter(movies, ratings, min_ratings=5)
    # Логарифмическая шкала по X сжимает очень популярные фильмы и оставляет видимым long-tail.
    rating_scatter_chart = (
        alt.Chart(rating_scatter)
        .mark_circle(opacity=0.55)
        .encode(
            x=alt.X("rating_count:Q", title="Оценок у фильма", scale=alt.Scale(type="log")),
            y=alt.Y("mean_rating:Q", title="Средняя оценка", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("primary_genre:N", title="Жанр"),
            size=alt.Size("rating_count:Q", title="Оценок", legend=None),
            tooltip=[
                alt.Tooltip("title:N", title="Фильм"),
                alt.Tooltip("primary_genre:N", title="Жанр"),
                alt.Tooltip("rating_count:Q", title="Оценок"),
                alt.Tooltip("mean_rating:Q", title="Средняя оценка", format=".3f"),
            ],
        )
    )
    st.altair_chart(rating_scatter_chart, use_container_width=True)

trend_left, trend_right = st.columns(2)

with trend_left:
    st.subheader("Активность по годам")
    activity_by_year = get_rating_activity_by_year(ratings).set_index("year")
    st.line_chart(activity_by_year, y="rating_count", use_container_width=True)

with trend_right:
    st.subheader("Средний рейтинг по жанрам")
    genre_rating_stats = get_genre_rating_stats(movies, ratings, min_ratings=100)
    # Горизонтальный bar chart читабельнее, когда категорий много и подписи длинные.
    genre_rating_chart = (
        alt.Chart(genre_rating_stats)
        .mark_bar()
        .encode(
            x=alt.X("mean_rating:Q", title="Средняя оценка", scale=alt.Scale(domain=[0, 5])),
            y=alt.Y("genre:N", sort="-x", title="Жанр"),
            tooltip=[
                alt.Tooltip("genre:N", title="Жанр"),
                alt.Tooltip("mean_rating:Q", title="Средняя оценка", format=".3f"),
                alt.Tooltip("rating_count:Q", title="Оценок"),
            ],
        )
    )
    st.altair_chart(genre_rating_chart, use_container_width=True)

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
