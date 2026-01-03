from __future__ import annotations

from movie_recommender.llm.base import heuristic_parse_query


def test_query_parser_understands_casual_language():
    parsed = heuristic_parse_query("I want a funny movie for kids tonight")
    assert parsed["mood"] == "funny"
    assert parsed["time_of_day"] == "night"
    assert "children" in parsed["genres"]


def test_query_parser_extracts_unquoted_reference_movie():
    parsed = heuristic_parse_query("Recommend something like Interstellar for tonight")
    assert parsed["seed_movie_title"] == "Interstellar"


def test_query_parser_does_not_treat_movie_title_numbers_as_top_k():
    parsed = heuristic_parse_query('Recommend something like "Toy Story 4" for tonight')
    assert parsed["seed_movie_title"] == "Toy Story 4"
    assert parsed["top_k"] is None


def test_query_parser_extracts_explicit_top_k_requests():
    parsed = heuristic_parse_query('Recommend 6 movies like "Toy Story 4" for tonight')
    assert parsed["seed_movie_title"] == "Toy Story 4"
    assert parsed["top_k"] == 6
