from __future__ import annotations

from apps.streamlit_app import filter_movie_labels, resolve_movie_label


def test_filter_movie_labels_prefers_relevant_matches():
    options = {
        "Jumanji (1995)": 1,
        "Jumanji: Welcome to the Jungle (2017)": 2,
        "Toy Story (1995)": 3,
        "The Matrix (1999)": 4,
    }
    matches = filter_movie_labels(options, "jumanji", "", limit=10)
    assert matches[:2] == ["Jumanji (1995)", "Jumanji: Welcome to the Jungle (2017)"]


def test_resolve_movie_label_returns_best_fuzzy_match():
    options = {
        "Jumanji (1995)": 1,
        "Jumanji: Welcome to the Jungle (2017)": 2,
        "Toy Story (1995)": 3,
    }
    resolved, matches = resolve_movie_label(options, "jumanj")
    assert resolved == "Jumanji (1995)"
    assert matches[:2] == ["Jumanji (1995)", "Jumanji: Welcome to the Jungle (2017)"]


def test_resolve_movie_label_uses_year_for_disambiguation():
    options = {
        "Jumanji (1995)": 1,
        "Jumanji: Welcome to the Jungle (2017)": 2,
        "Toy Story (1995)": 3,
    }
    resolved, matches = resolve_movie_label(options, "Jumanji 2017")
    assert resolved == "Jumanji: Welcome to the Jungle (2017)"
    assert matches[0] == "Jumanji: Welcome to the Jungle (2017)"


def test_resolve_movie_label_does_not_force_short_ambiguous_queries():
    options = {
        "The Matrix (1999)": 1,
        "The Prestige (2006)": 2,
        "The Lion King (1994)": 3,
    }
    resolved, matches = resolve_movie_label(options, "the")
    assert resolved == ""
    assert matches[:2] == ["The Matrix (1999)", "The Prestige (2006)"]
