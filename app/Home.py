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
)
from src.catalog import (
    add_movie_metadata,
    build_movie_labels,
    filter_movie_catalog,
    get_genre_options,
)
from src.data_loader import MovieLensData, load_movielens
from src.hybrid import (
    HybridArtifacts,
    build_hybrid_artifacts,
    get_hybrid_recommendations_from_artifacts,
)
from src.matrix_factorization import (
    SVDRecommender,
    build_svd_recommender,
    get_svd_recommendations_from_model,
)
from src.preprocess import load_or_build_content_feature_matrix, load_or_create_rating_stats
from src.recommend import (
    ContentFeatureMatrix,
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
    return load_or_build_content_feature_matrix(movies)


@st.cache_resource(show_spinner="Готовим CF-матрицу...")
def get_cf_artifacts(ratings: pd.DataFrame) -> tuple[ItemUserMatrix, pd.DataFrame]:
    """Кеширует item-user матрицу и статистики фильмов для персональных рекомендаций."""
    return build_item_user_matrix(ratings), load_or_create_rating_stats(ratings)


@st.cache_resource(show_spinner="Готовим SVD-модель...")
def get_svd_artifacts(ratings: pd.DataFrame) -> tuple[SVDRecommender, pd.DataFrame]:
    """Кеширует SVD-модель и статистики фильмов для персональных рекомендаций."""
    return build_svd_recommender(ratings), load_or_create_rating_stats(ratings)


@st.cache_resource(show_spinner="Готовим hybrid-модель...")
def get_hybrid_artifacts(movies: pd.DataFrame, ratings: pd.DataFrame) -> HybridArtifacts:
    """Кеширует артефакты гибридного рекомендателя."""
    return build_hybrid_artifacts(
        movies,
        ratings,
        rating_stats=load_or_create_rating_stats(ratings),
    )


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет в таблице только пользовательские колонки и приводит названия к русским."""
    visible_columns = {
        "title": "Фильм",
        "year": "Год",
        "primary_genre": "Основной жанр",
        "genres": "Жанры",
        "reason": "Почему в списке",
        "rating_count": "Оценок",
        "mean_rating": "Средняя оценка",
        "similarity": "Сходство",
        "cf_score": "CF score",
        "predicted_rating": "Прогноз оценки",
        "hybrid_score": "Hybrid score",
        "popularity_score": "Популярность",
    }
    result = df[[column for column in visible_columns if column in df.columns]].rename(
        columns=visible_columns
    )
    for column in ("Средняя оценка", "Сходство", "CF score", "Прогноз оценки", "Hybrid score", "Популярность"):
        if column in result.columns:
            result[column] = result[column].round(3)
    return result


def attach_display_metadata(recommendations: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Добавляет год и основной жанр к выдаче, если модель вернула только базовые поля."""
    if recommendations.empty:
        return recommendations

    metadata_columns = ["movieId", "year", "primary_genre"]
    metadata = add_movie_metadata(catalog)[metadata_columns].drop_duplicates("movieId")
    result = recommendations.copy()
    missing_columns = [column for column in metadata_columns[1:] if column not in result.columns]
    if missing_columns:
        result = result.merge(metadata, on="movieId", how="left")
    return result


def add_reason(recommendations: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Добавляет короткое объяснение ранжирования для таблицы рекомендаций."""
    result = recommendations.copy()
    if result.empty:
        result["reason"] = pd.Series(dtype=str)
        return result

    def numeric(row: pd.Series, column: str, default: float = 0.0) -> float:
        value = row.get(column, default)
        return default if pd.isna(value) else float(value)

    def reason_for(row: pd.Series) -> str:
        rating_count = int(numeric(row, "rating_count"))
        mean_rating = numeric(row, "mean_rating")
        if scenario == "popular":
            return f"{rating_count} оценок, средняя {mean_rating:.2f}"
        if scenario == "top_rated":
            return f"Высокая средняя при {rating_count} оценках"
        if scenario == "similar":
            return f"Контентное сходство {numeric(row, 'similarity'):.2f}"
        if scenario == "cf":
            return "Похож на фильмы из положительной истории"
        if scenario == "svd":
            return f"Прогноз оценки {numeric(row, 'predicted_rating'):.2f}"
        if scenario == "hybrid":
            return "Смешанный сигнал CF, SVD и популярности"
        return ""

    result["reason"] = result.apply(reason_for, axis=1)
    return result


def prepare_recommendation_table(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    """Готовит выдачу к показу: метаданные, объяснение, локализация колонок."""
    return format_table(add_reason(attach_display_metadata(recommendations, catalog), scenario))


def get_user_history(user_id: int, movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Возвращает историю пользователя с метаданными фильма."""
    history = ratings[ratings["userId"].astype(int) == int(user_id)].copy()
    return history.merge(add_movie_metadata(movies), on="movieId", how="left")


def top_user_genres(history: pd.DataFrame, limit: int = 3) -> str:
    """Собирает короткую строку с жанрами, которые чаще всего встречаются в истории."""
    if history.empty:
        return "нет данных"
    genres = (
        history["genres"]
        .fillna("(no genres listed)")
        .str.split("|")
        .explode()
    )
    genres = genres[genres != "(no genres listed)"]
    if genres.empty:
        return "нет данных"
    return ", ".join(genres.value_counts().head(limit).index.tolist())


def show_recommendations(
    recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
    scenario: str,
    empty_message: str,
) -> None:
    """Рендерит таблицу рекомендаций и единообразно обрабатывает пустую выдачу."""
    if recommendations.empty:
        st.warning(empty_message)
        return
    st.dataframe(
        prepare_recommendation_table(recommendations, catalog, scenario),
        use_container_width=True,
        hide_index=True,
    )


def get_personal_recommendations(
    model_name: str,
    user_id: int,
    recommendation_catalog: pd.DataFrame,
    all_movies: pd.DataFrame,
    ratings: pd.DataFrame,
    limit: int,
    min_positive_rating: float,
) -> pd.DataFrame:
    """Возвращает персональную выдачу выбранной модели с учетом фильтра каталога."""
    if model_name == "Item-item CF":
        item_user, rating_stats = get_cf_artifacts(ratings)
        return get_item_item_recommendations_from_matrix(
            user_id=user_id,
            movies=recommendation_catalog,
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
            movies=recommendation_catalog,
            ratings=ratings,
            recommender=recommender,
            rating_stats=rating_stats,
            limit=limit,
        )

    hybrid_artifacts = get_hybrid_artifacts(all_movies, ratings)
    return get_hybrid_recommendations_from_artifacts(
        user_id=user_id,
        movies=recommendation_catalog,
        ratings=ratings,
        artifacts=hybrid_artifacts,
        limit=limit,
        min_positive_rating=min_positive_rating,
        candidate_pool_size=max(limit * 50, 500),
    )


st.title("MovieLens Recommender")
st.caption("Рекомендации, похожие фильмы и offline-сравнение моделей на MovieLens latest-small.")

try:
    data = get_data()
except Exception as exc:
    # Чаще всего сюда попадаем, если MovieLens еще не скачан, нет интернета или данные неполные.
    st.error("Не удалось загрузить MovieLens. Проверьте локальные данные или подключение к интернету.")
    st.exception(exc)
    st.stop()

movies = data.movies
ratings = data.ratings
movies_with_metadata = add_movie_metadata(movies)
year_values = movies_with_metadata["year"].dropna().astype(int)

with st.sidebar:
    st.header("Сценарий")
    mode = st.radio(
        "Тип выдачи",
        ["Популярные фильмы", "Top-rated", "Похожие фильмы", "Персональные рекомендации"],
    )
    personal_model = None
    if mode == "Персональные рекомендации":
        personal_model = st.radio(
            "Модель",
            ["Hybrid", "Item-item CF", "SVD"],
            horizontal=True,
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
        disabled=mode != "Персональные рекомендации" or personal_model not in {"Item-item CF", "Hybrid"},
    )

    st.divider()
    st.header("Каталог")
    selected_genres = st.multiselect("Жанры", get_genre_options(movies))
    year_filter = None
    if year_values.empty:
        selected_years = None
    else:
        min_year = int(year_values.min())
        max_year = int(year_values.max())
        selected_years = st.slider(
            "Годы выпуска",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
        )
        if selected_years != (min_year, max_year):
            year_filter = selected_years

    filtered_movies = filter_movie_catalog(
        movies,
        genres=selected_genres,
        year_range=year_filter,
    )
    st.metric("Фильмов в фильтре", f"{filtered_movies['movieId'].nunique():,}".replace(",", " "))

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Фильмов", f"{movies['movieId'].nunique():,}".replace(",", " "))
metric_2.metric("Оценок", f"{len(ratings):,}".replace(",", " "))
metric_3.metric("Пользователей", f"{ratings['userId'].nunique():,}".replace(",", " "))
metric_4.metric("В фильтре", f"{filtered_movies['movieId'].nunique():,}".replace(",", " "))

if filtered_movies.empty:
    st.warning("Под выбранные фильтры не попал ни один фильм.")
    st.stop()

if mode == "Популярные фильмы":
    # Простая baseline-логика: чем больше оценок у фильма, тем выше он в списке.
    st.subheader("Популярные фильмы")
    recommendations = get_popular_movies(filtered_movies, ratings, min_ratings=min_ratings, limit=limit)
    show_recommendations(
        recommendations,
        filtered_movies,
        "popular",
        "Нет фильмов, которые проходят выбранные фильтры и порог оценок.",
    )

elif mode == "Top-rated":
    # Средняя оценка имеет смысл только после фильтра по минимальному числу оценок.
    st.subheader("Top-rated фильмы")
    recommendations = get_top_rated_movies(filtered_movies, ratings, min_ratings=min_ratings, limit=limit)
    show_recommendations(
        recommendations,
        filtered_movies,
        "top_rated",
        "Нет фильмов, которые проходят выбранные фильтры и порог оценок.",
    )

else:
    if mode == "Персональные рекомендации":
        # Здесь доступны три персональных сценария: CF, SVD и их гибрид.
        st.subheader("Персональные рекомендации")
        user_rating_counts = ratings.groupby("userId").size().astype(int).to_dict()
        user_ids = sorted(ratings["userId"].astype(int).unique().tolist())
        selected_user_id = st.selectbox(
            "Выберите пользователя",
            user_ids,
            format_func=lambda value: f"User {value} · {user_rating_counts.get(value, 0)} оценок",
        )

        history = get_user_history(selected_user_id, movies, ratings)
        profile_1, profile_2, profile_3, profile_4 = st.columns(4)
        profile_1.metric("Оценок пользователя", f"{len(history):,}".replace(",", " "))
        profile_2.metric("Средняя оценка", f"{history['rating'].mean():.2f}" if not history.empty else "нет данных")
        profile_3.metric(
            f"Оценок ≥ {min_positive_rating:g}",
            f"{int((history['rating'] >= min_positive_rating).sum()):,}".replace(",", " "),
        )
        seen_in_filter = history["movieId"].astype(int).isin(filtered_movies["movieId"].astype(int)).sum()
        profile_4.metric("В фильтре уже видел", f"{int(seen_in_filter):,}".replace(",", " "))
        st.caption(f"Частые жанры пользователя: {top_user_genres(history)}")

        scenario_by_model = {
            "Hybrid": "hybrid",
            "Item-item CF": "cf",
            "SVD": "svd",
        }
        compare_models = st.toggle("Сравнить модели", value=False)
        empty_message = "Не удалось построить рекомендации для выбранного пользователя и фильтров."

        if compare_models:
            model_tabs = st.tabs(["Hybrid", "Item-item CF", "SVD"])
            for tab, model_name in zip(model_tabs, ["Hybrid", "Item-item CF", "SVD"]):
                with tab:
                    recommendations = get_personal_recommendations(
                        model_name,
                        selected_user_id,
                        filtered_movies,
                        movies,
                        ratings,
                        limit,
                        min_positive_rating,
                    )
                    show_recommendations(
                        recommendations,
                        filtered_movies,
                        scenario_by_model[model_name],
                        empty_message,
                    )
        else:
            recommendations = get_personal_recommendations(
                personal_model or "Hybrid",
                selected_user_id,
                filtered_movies,
                movies,
                ratings,
                limit,
                min_positive_rating,
            )
            show_recommendations(
                recommendations,
                filtered_movies,
                scenario_by_model[personal_model or "Hybrid"],
                empty_message,
            )

        with st.expander("История пользователя"):
            history = (
                history
                .sort_values(["rating", "title"], ascending=[False, True])
                .head(20)
            )
            st.dataframe(
                history[["title", "year", "primary_genre", "genres", "rating"]].rename(
                    columns={
                        "title": "Фильм",
                        "year": "Год",
                        "primary_genre": "Основной жанр",
                        "genres": "Жанры",
                        "rating": "Оценка",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.stop()

    # Content-based сценарий: ищем похожие фильмы по названию и жанрам.
    st.subheader("Похожие фильмы")
    movie_query = st.text_input("Поиск фильма", placeholder="Matrix, Toy Story, Heat...")
    movie_options = build_movie_labels(filtered_movies.sort_values("title"))
    if movie_query.strip():
        movie_options = filter_movie_catalog(movie_options, query=movie_query)

    if movie_options.empty:
        st.warning("По такому запросу в выбранном каталоге ничего не найдено.")
        st.stop()
    if len(filtered_movies) < 2:
        st.warning("В выбранном каталоге нужен хотя бы еще один фильм для поиска похожих.")
        st.stop()

    labels_by_movie_id = movie_options.set_index("movieId")["label"].to_dict()
    selected_index = st.selectbox(
        "Выберите фильм",
        movie_options["movieId"].astype(int).tolist(),
        index=0,
        format_func=lambda movie_id: labels_by_movie_id.get(movie_id, str(movie_id)),
    )
    selected_movie_id = int(selected_index)

    recommendations = get_similar_movies_from_matrix(
        selected_movie_id,
        filtered_movies,
        get_content_features(filtered_movies),
        limit=limit,
    )
    show_recommendations(
        recommendations,
        filtered_movies,
        "similar",
        "Не нашлось похожих фильмов внутри выбранных фильтров.",
    )
