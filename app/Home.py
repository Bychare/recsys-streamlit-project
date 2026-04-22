"""Главная страница Streamlit-приложения.

Здесь собраны пользовательские сценарии рекомендаций:
популярные фильмы, top-rated фильмы, похожие фильмы и персональные рекомендации.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# При запуске через `streamlit run app/Home.py` Python видит папку `app/`,
# но не всегда видит корень проекта. Добавляем корень вручную, чтобы работали
# импорты из `src`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collaborative import (
    ItemUserMatrix,
    build_item_user_matrix,
    get_item_item_recommendations_from_matrix,
    get_item_rating_stats,
)
from src.data_loader import MovieLensData, load_movielens
from src.recommend import (
    ContentFeatureMatrix,
    build_content_feature_matrix,
    get_popular_movies,
    get_similar_movies_from_matrix,
    get_top_rated_movies,
)


st.set_page_config(
    page_title="recsys-streamlit-project",
    layout="wide",
)


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    """Загружает MovieLens один раз и кеширует результат для Streamlit-сессии."""
    return load_movielens()


@st.cache_resource(show_spinner="Готовим content-based матрицу...")
def get_content_features(movies: pd.DataFrame) -> ContentFeatureMatrix:
    """Кеширует TF-IDF матрицу, чтобы похожие фильмы искались без пересборки признаков."""
    return build_content_feature_matrix(movies)


@st.cache_resource(show_spinner="Готовим CF-матрицу...")
def get_cf_artifacts(ratings: pd.DataFrame) -> tuple[ItemUserMatrix, pd.DataFrame]:
    """Кеширует item-user матрицу и статистики фильмов для персональных рекомендаций."""
    return build_item_user_matrix(ratings), get_item_rating_stats(ratings)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет в таблице только пользовательские колонки и приводит названия к русским."""
    visible_columns = {
        "title": "Фильм",
        "genres": "Жанры",
        "rating_count": "Оценок",
        "mean_rating": "Средняя оценка",
        "similarity": "Сходство",
        "cf_score": "CF score",
    }
    result = df[[column for column in visible_columns if column in df.columns]].rename(
        columns=visible_columns
    )
    for column in ("Средняя оценка", "Сходство", "CF score"):
        if column in result.columns:
            result[column] = result[column].round(3)
    return result


st.title("recsys-streamlit-project")
st.caption("Демо рекомендательной системы на MovieLens latest-small.")

try:
    data = get_data()
except Exception as exc:
    # Чаще всего сюда попадаем, если MovieLens еще не скачан, нет интернета или данные неполные.
    st.error("Не удалось загрузить MovieLens. Проверьте локальные данные или подключение к интернету.")
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
    # Один экран, но несколько разных рекомендательных сценариев.
    mode = st.radio(
        "Сценарий",
        ["Популярные фильмы", "Top-rated", "Похожие фильмы", "Персональные рекомендации"],
    )
    limit = st.slider("Количество рекомендаций", min_value=5, max_value=30, value=10, step=5)
    min_ratings = st.slider(
        "Минимум оценок",
        min_value=1,
        max_value=300,
        value=50,
        step=5,
        disabled=mode in {"Похожие фильмы", "Персональные рекомендации"},
    )
    min_positive_rating = st.slider(
        "Порог понравившихся",
        min_value=0.5,
        max_value=5.0,
        value=4.0,
        step=0.5,
        disabled=mode != "Персональные рекомендации",
    )

if mode == "Популярные фильмы":
    # Простая baseline-логика: чем больше оценок у фильма, тем выше он в списке.
    st.subheader("Популярные фильмы")
    recommendations = get_popular_movies(movies, ratings, min_ratings=min_ratings, limit=limit)
    st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)

elif mode == "Top-rated":
    # Средняя оценка имеет смысл только после фильтра по минимальному числу оценок.
    st.subheader("Top-rated фильмы")
    recommendations = get_top_rated_movies(movies, ratings, min_ratings=min_ratings, limit=limit)
    st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)

else:
    if mode == "Персональные рекомендации":
        # Item-item collaborative filtering строит рекомендации из истории выбранного пользователя.
        st.subheader("Персональные рекомендации")
        user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
        selected_user_id = st.selectbox(
            "Выберите пользователя",
            user_ids,
            format_func=lambda value: f"User {value}",
        )

        item_user, rating_stats = get_cf_artifacts(ratings)
        recommendations = get_item_item_recommendations_from_matrix(
            user_id=selected_user_id,
            movies=movies,
            ratings=ratings,
            item_user=item_user,
            rating_stats=rating_stats,
            limit=limit,
            min_positive_rating=min_positive_rating,
        )

        if recommendations.empty:
            st.warning("Для пользователя недостаточно положительных оценок при выбранном пороге.")
        else:
            st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)

        with st.expander("История пользователя"):
            history = (
                ratings[ratings["userId"].astype(int) == int(selected_user_id)]
                .merge(movies, on="movieId", how="left")
                .sort_values(["rating", "title"], ascending=[False, True])
                .head(20)
            )
            st.dataframe(
                history[["title", "genres", "rating"]].rename(
                    columns={"title": "Фильм", "genres": "Жанры", "rating": "Оценка"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.stop()

    # Content-based сценарий: ищем похожие фильмы по названию и жанрам.
    st.subheader("Похожие фильмы")
    movie_options = movies.sort_values("title")[["movieId", "title"]].reset_index(drop=True)
    title_counts = movie_options["title"].value_counts()
    movie_options["label"] = movie_options.apply(
        lambda row: (
            f"{row['title']} · ID {int(row['movieId'])}"
            if title_counts[row["title"]] > 1
            else row["title"]
        ),
        axis=1,
    )
    selected_index = st.selectbox(
        "Выберите фильм",
        movie_options.index.tolist(),
        index=0,
        format_func=lambda index: movie_options.loc[index, "label"],
    )
    selected_movie_id = int(movie_options.loc[selected_index, "movieId"])

    recommendations = get_similar_movies_from_matrix(
        selected_movie_id,
        movies,
        get_content_features(movies),
        limit=limit,
    )
    st.dataframe(format_table(recommendations), use_container_width=True, hide_index=True)
