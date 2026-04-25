"""Chat-like recommendation page built on local rule-based parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog import add_movie_metadata, filter_movie_catalog, get_genre_options
from src.chat_recommender import describe_request, parse_chat_request
from src.collaborative import build_item_user_matrix, get_item_item_recommendations_from_matrix
from src.data_loader import MovieLensData, load_movielens
from src.hybrid import build_hybrid_artifacts, get_hybrid_recommendations_from_artifacts
from src.matrix_factorization import build_svd_recommender, get_svd_recommendations_from_model
from src.preprocess import load_or_build_content_feature_matrix, load_or_create_rating_stats
from src.recommend import get_similar_movies_from_matrix, get_top_rated_movies


st.set_page_config(page_title="Chat Recommender", layout="wide")


@st.cache_data(show_spinner="Загружаем MovieLens...")
def get_data() -> MovieLensData:
    """Loads MovieLens once per Streamlit session."""
    return load_movielens()


@st.cache_resource(show_spinner="Готовим content-based матрицу...")
def get_content_features(movies: pd.DataFrame):
    """Caches TF-IDF features for movie similarity search."""
    return load_or_build_content_feature_matrix(movies)


@st.cache_resource(show_spinner="Готовим CF-матрицу...")
def get_cf_artifacts(ratings: pd.DataFrame):
    """Caches item-user matrix and rating stats."""
    return build_item_user_matrix(ratings), load_or_create_rating_stats(ratings)


@st.cache_resource(show_spinner="Готовим SVD-модель...")
def get_svd_artifacts(ratings: pd.DataFrame):
    """Caches SVD model and rating stats."""
    return build_svd_recommender(ratings), load_or_create_rating_stats(ratings)


@st.cache_resource(show_spinner="Готовим hybrid-модель...")
def get_hybrid_artifacts(movies: pd.DataFrame, ratings: pd.DataFrame):
    """Caches hybrid recommender artifacts."""
    return build_hybrid_artifacts(
        movies,
        ratings,
        rating_stats=load_or_create_rating_stats(ratings),
    )


def format_recommendations(df: pd.DataFrame, catalog: pd.DataFrame, reason: str) -> pd.DataFrame:
    """Formats a recommendation table for chat output."""
    if df.empty:
        return df

    metadata = add_movie_metadata(catalog)[["movieId", "year", "primary_genre"]]
    result = df.copy()
    for column in ("year", "primary_genre"):
        if column not in result.columns:
            result = result.merge(metadata[["movieId", column]], on="movieId", how="left")

    result["reason"] = reason
    columns = {
        "title": "Фильм",
        "year": "Год",
        "primary_genre": "Основной жанр",
        "genres": "Жанры",
        "reason": "Почему",
        "rating_count": "Оценок",
        "mean_rating": "Средняя оценка",
        "similarity": "Сходство",
        "hybrid_score": "Hybrid score",
        "cf_score": "CF score",
        "predicted_rating": "Прогноз",
    }
    visible = result[[column for column in columns if column in result.columns]].rename(columns=columns)
    for column in ("Средняя оценка", "Сходство", "Hybrid score", "CF score", "Прогноз"):
        if column in visible.columns:
            visible[column] = visible[column].round(3)
    return visible


def render_message(message: dict) -> None:
    """Renders a stored chat message with an optional recommendation table."""
    with st.chat_message(message["role"]):
        st.write(message["content"])
        table = message.get("table")
        if isinstance(table, pd.DataFrame) and not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)


def build_personal_recommendations(
    model_name: str,
    user_id: int,
    movies: pd.DataFrame,
    filtered_movies: pd.DataFrame,
    ratings: pd.DataFrame,
    limit: int,
    min_positive_rating: float,
) -> pd.DataFrame:
    """Builds recommendations for a concrete user and a filtered catalog."""
    if model_name == "Item-item CF":
        item_user, rating_stats = get_cf_artifacts(ratings)
        return get_item_item_recommendations_from_matrix(
            user_id=user_id,
            movies=filtered_movies,
            ratings=ratings,
            item_user=item_user,
            rating_stats=rating_stats,
            limit=limit,
            min_positive_rating=min_positive_rating,
        )

    if model_name == "SVD":
        recommender, rating_stats = get_svd_artifacts(ratings)
        return get_svd_recommendations_from_model(
            user_id=user_id,
            movies=filtered_movies,
            ratings=ratings,
            recommender=recommender,
            rating_stats=rating_stats,
            limit=limit,
        )

    artifacts = get_hybrid_artifacts(movies, ratings)
    return get_hybrid_recommendations_from_artifacts(
        user_id=user_id,
        movies=filtered_movies,
        ratings=ratings,
        artifacts=artifacts,
        limit=limit,
        min_positive_rating=min_positive_rating,
        candidate_pool_size=max(limit * 50, 500),
    )


def handle_prompt(
    prompt: str,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    selected_user_id: int | None,
    personal_model: str,
    limit: int,
    min_ratings: int,
    min_positive_rating: float,
) -> dict:
    """Turns a chat prompt into an assistant response and optional table."""
    movies_with_metadata = add_movie_metadata(movies)
    year_values = movies_with_metadata["year"].dropna().astype(int)
    request = parse_chat_request(
        prompt,
        movies,
        available_genres=set(get_genre_options(movies)),
        min_year=int(year_values.min()) if not year_values.empty else 1900,
        max_year=int(year_values.max()) if not year_values.empty else 2100,
    )

    filtered_movies = filter_movie_catalog(
        movies,
        genres=request.genres,
        year_range=request.year_range,
    )
    if filtered_movies.empty:
        return {
            "role": "assistant",
            "content": f"Не нашел фильмов под фильтры: {describe_request(request, selected_user_id)}.",
        }

    if request.is_similar_request:
        similar = get_similar_movies_from_matrix(
            int(request.anchor_movie_id),
            movies,
            get_content_features(movies),
            limit=max(limit * 20, 200),
        )
        allowed_movie_ids = set(filtered_movies["movieId"].astype(int))
        recommendations = similar[similar["movieId"].astype(int).isin(allowed_movie_ids)].head(limit)
        table = format_recommendations(
            recommendations,
            filtered_movies,
            f"похоже на {request.anchor_title}",
        )
        return {
            "role": "assistant",
            "content": f"Собрал подборку: {describe_request(request, selected_user_id)}.",
            "table": table,
        }

    user_id = request.user_id if request.user_id is not None else selected_user_id
    known_user_ids = set(ratings["userId"].astype(int))
    if user_id is not None:
        if int(user_id) not in known_user_ids:
            return {"role": "assistant", "content": f"User {user_id} не найден в MovieLens latest-small."}
        recommendations = build_personal_recommendations(
            personal_model,
            int(user_id),
            movies,
            filtered_movies,
            ratings,
            limit,
            min_positive_rating,
        )
        table = format_recommendations(
            recommendations,
            filtered_movies,
            f"{personal_model} для User {user_id}",
        )
        return {
            "role": "assistant",
            "content": f"Собрал персональную подборку: {describe_request(request, int(user_id))}.",
            "table": table,
        }

    recommendations = get_top_rated_movies(
        filtered_movies,
        ratings,
        min_ratings=min_ratings,
        limit=limit,
    )
    table = format_recommendations(
        recommendations,
        filtered_movies,
        f"top-rated при минимуме {min_ratings} оценок",
    )
    return {
        "role": "assistant",
        "content": f"Собрал общую подборку: {describe_request(request)}.",
        "table": table,
    }


try:
    data = get_data()
except Exception as exc:
    st.error("Не удалось загрузить MovieLens. Проверьте локальные данные или подключение к интернету.")
    st.exception(exc)
    st.stop()

movies = data.movies
ratings = data.ratings

with st.sidebar:
    st.header("Контекст")
    use_personal_context = st.toggle("Персональный пользователь", value=True)
    selected_user_id = None
    if use_personal_context:
        user_counts = ratings.groupby("userId").size().astype(int).to_dict()
        user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
        selected_user_id = st.selectbox(
            "Пользователь",
            user_ids,
            format_func=lambda value: f"User {value} · {user_counts.get(value, 0)} оценок",
        )
    personal_model = st.radio("Модель", ["Hybrid", "Item-item CF", "SVD"], horizontal=True)
    limit = st.slider("Количество рекомендаций", min_value=5, max_value=30, value=10, step=5)
    min_ratings = st.slider("Минимум оценок для общей выдачи", min_value=1, max_value=300, value=50, step=5)
    min_positive_rating = st.slider("Порог понравившихся", min_value=0.5, max_value=5.0, value=4.0, step=0.5)
    if st.button("Очистить чат"):
        st.session_state.chat_recommender_messages = []
        st.rerun()

st.title("Recommendation Chat")

if "chat_recommender_messages" not in st.session_state:
    st.session_state.chat_recommender_messages = [
        {
            "role": "assistant",
            "content": "Напишите, что хочется посмотреть: жанр, годы, похожий фильм или user id.",
        }
    ]

for stored_message in st.session_state.chat_recommender_messages:
    render_message(stored_message)

prompt = st.chat_input("Например: посоветуй фантастику 90-х похожую на Matrix")
if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.chat_recommender_messages.append(user_message)
    render_message(user_message)

    with st.spinner("Подбираю фильмы..."):
        assistant_message = handle_prompt(
            prompt,
            movies,
            ratings,
            selected_user_id,
            personal_model,
            limit,
            min_ratings,
            min_positive_rating,
        )
    st.session_state.chat_recommender_messages.append(assistant_message)
    render_message(assistant_message)
