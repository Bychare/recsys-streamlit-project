"""Rule-based parsing for the recommendation chat page."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.catalog import add_movie_metadata


GENRE_ALIASES: dict[str, tuple[str, ...]] = {
    "Action": ("action", "боевик", "боевики", "экшен", "экшн"),
    "Adventure": ("adventure", "adventures", "приключение", "приключения"),
    "Animation": ("animation", "animated", "мультфильм", "мультфильмы", "анимация"),
    "Children": ("children", "kids", "семейный", "семейные", "детский", "детские"),
    "Comedy": ("comedy", "comedies", "комедия", "комедии", "смешное", "смешные"),
    "Crime": ("crime", "criminal", "криминал", "криминальный", "криминальные"),
    "Documentary": ("documentary", "documentaries", "документальный", "документальные"),
    "Drama": ("drama", "dramas", "драма", "драмы", "драму"),
    "Fantasy": ("fantasy", "фэнтези", "фентези"),
    "Film-Noir": ("film-noir", "noir", "нуар"),
    "Horror": ("horror", "ужасы", "хоррор", "страшное", "страшные"),
    "Musical": ("musical", "musicals", "мюзикл", "мюзиклы"),
    "Mystery": ("mystery", "mysteries", "детектив", "детективы", "мистика"),
    "Romance": ("romance", "romantic", "романтика", "романтический", "мелодрама", "мелодрамы"),
    "Sci-Fi": (
        "sci-fi",
        "scifi",
        "science fiction",
        "фантастика",
        "фантастику",
        "фантастический",
        "научная фантастика",
    ),
    "Thriller": ("thriller", "thrillers", "триллер", "триллеры"),
    "War": ("war", "военный", "военные", "война", "про войну"),
    "Western": ("western", "вестерн", "вестерны"),
}

SIMILARITY_MARKERS = (
    "похоже на",
    "похожее на",
    "похожую на",
    "похожие на",
    "как",
    "в стиле",
    "similar to",
    "like",
)


@dataclass(frozen=True)
class ChatRecommendationRequest:
    """Structured intent extracted from a user chat message."""

    text: str
    genres: tuple[str, ...] = ()
    year_range: tuple[int, int] | None = None
    anchor_movie_id: int | None = None
    anchor_title: str | None = None
    user_id: int | None = None

    @property
    def is_similar_request(self) -> bool:
        """Whether the message refers to a concrete anchor movie."""
        return self.anchor_movie_id is not None


def normalize_text(text: str) -> str:
    """Normalizes text for lightweight rule matching."""
    return text.lower().replace("ё", "е").strip()


def detect_genres(text: str, available_genres: set[str] | None = None) -> tuple[str, ...]:
    """Detects MovieLens genres mentioned in Russian or English."""
    normalized = f" {normalize_text(text)} "
    result: list[str] = []
    for genre, aliases in GENRE_ALIASES.items():
        if available_genres is not None and genre not in available_genres:
            continue
        if any(f" {alias} " in normalized or alias in normalized for alias in aliases):
            result.append(genre)
    return tuple(result)


def detect_year_range(text: str, min_year: int = 1900, max_year: int = 2100) -> tuple[int, int] | None:
    """Extracts a useful release-year filter from a short chat message."""
    normalized = normalize_text(text)

    decade_match = re.search(r"\b(\d{2})(?:-?х|е|s)\b", normalized)
    if decade_match:
        decade = int(decade_match.group(1))
        base = 2000 if decade <= 30 else 1900
        start = base + decade
        return max(start, min_year), min(start + 9, max_year)

    years = [int(value) for value in re.findall(r"\b(19\d{2}|20\d{2})\b", normalized)]
    if len(years) >= 2:
        start, end = sorted(years[:2])
        return max(start, min_year), min(end, max_year)

    if len(years) == 1:
        year = years[0]
        if re.search(r"\b(после|с|from|after|since)\b", normalized):
            return max(year, min_year), max_year
        if re.search(r"\b(до|before|older than)\b", normalized):
            return min_year, min(year, max_year)
        return max(year, min_year), min(year, max_year)

    return None


def detect_user_id(text: str) -> int | None:
    """Extracts a user id from phrases like `user 42` or `пользователь 42`."""
    normalized = normalize_text(text)
    match = re.search(r"\b(?:user|пользователь|юзер)\s*#?\s*(\d+)\b", normalized)
    return int(match.group(1)) if match else None


def _candidate_phrase(text: str) -> str | None:
    normalized = normalize_text(text)
    for marker in SIMILARITY_MARKERS:
        marker_index = normalized.find(marker)
        if marker_index == -1:
            continue
        phrase = text[marker_index + len(marker) :].strip(" :,.!?\"'«»")
        return phrase or None
    return None


def find_movie_mention(text: str, movies: pd.DataFrame) -> tuple[int, str] | None:
    """Finds the most plausible movie title mentioned in a chat message."""
    if movies.empty:
        return None

    normalized_text = normalize_text(text)
    phrase = _candidate_phrase(text)
    search_space = normalize_text(phrase) if phrase else normalized_text
    catalog = add_movie_metadata(movies)

    matches: list[tuple[int, str, int]] = []
    for row in catalog.itertuples(index=False):
        title = str(row.title)
        title_clean = str(row.title_clean)
        normalized_title = normalize_text(title)
        normalized_clean = normalize_text(title_clean)
        if normalized_title and normalized_title in search_space:
            matches.append((int(row.movieId), title, len(normalized_title)))
        elif normalized_clean and normalized_clean in search_space:
            matches.append((int(row.movieId), title, len(normalized_clean)))

    if matches:
        movie_id, title, _ = max(matches, key=lambda value: value[2])
        return movie_id, title

    if phrase:
        phrase_normalized = normalize_text(phrase)
        contains = catalog[
            catalog["title_clean"]
            .fillna("")
            .str.lower()
            .str.replace("ё", "е", regex=False)
            .str.contains(phrase_normalized, regex=False)
        ]
        if not contains.empty:
            row = contains.sort_values("title").iloc[0]
            return int(row["movieId"]), str(row["title"])

    return None


def parse_chat_request(
    text: str,
    movies: pd.DataFrame,
    available_genres: set[str] | None = None,
    min_year: int = 1900,
    max_year: int = 2100,
) -> ChatRecommendationRequest:
    """Parses a short recommendation request without calling an LLM."""
    movie_match = find_movie_mention(text, movies)
    return ChatRecommendationRequest(
        text=text,
        genres=detect_genres(text, available_genres=available_genres),
        year_range=detect_year_range(text, min_year=min_year, max_year=max_year),
        anchor_movie_id=movie_match[0] if movie_match else None,
        anchor_title=movie_match[1] if movie_match else None,
        user_id=detect_user_id(text),
    )


def describe_request(request: ChatRecommendationRequest, fallback_user_id: int | None = None) -> str:
    """Builds a compact human-readable summary of the parsed request."""
    parts: list[str] = []
    if request.anchor_title:
        parts.append(f"похоже на «{request.anchor_title}»")
    if request.genres:
        parts.append(", ".join(request.genres))
    if request.year_range:
        start, end = request.year_range
        parts.append(str(start) if start == end else f"{start}-{end}")
    user_id = request.user_id or fallback_user_id
    if user_id is not None:
        parts.append(f"User {user_id}")
    return "; ".join(parts) if parts else "общая подборка"
