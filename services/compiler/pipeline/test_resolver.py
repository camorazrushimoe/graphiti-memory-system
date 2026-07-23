"""Unit tests for the Entity Resolver's pure logic (no LLM/DB/Qdrant calls).

Covers `_find_via_fuzzy` and the curated `KNOWN_ALIAS_GROUPS` lookup
(`_find_via_known_alias_group`) — the two resolution tiers that don't need
network access. The embedding-similarity tier (`_find_via_embedding`) and
the full `resolve_entity`/`resolve_entities` async flow (which round-trip
through Postgres + Qdrant) are exercised only by the live e2e tests
recorded in the tech-spec implementation log, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.compiler.pipeline import resolver


def test_known_alias_group_matches_case_insensitively():
    match = resolver._find_via_known_alias_group("anthropic claude")
    assert match is not None
    canonical_name, variants = match
    assert canonical_name == "Claude"
    assert "Claude CLI" in variants


def test_known_alias_group_matches_preferred_canonical_first():
    # "Claude" itself is the group's first/preferred element.
    match = resolver._find_via_known_alias_group("Claude")
    assert match is not None
    assert match[0] == "Claude"


def test_known_alias_group_no_match_for_unrelated_surface_form():
    assert resolver._find_via_known_alias_group("MySQL") is None


def test_known_alias_group_strips_whitespace():
    match = resolver._find_via_known_alias_group("  Claude Cli  ")
    assert match is not None
    assert match[0] == "Claude"


def _registry_row(canonical_id: str, canonical_name: str, aliases=None) -> dict:
    return {
        "canonical_id": canonical_id,
        "canonical_name": canonical_name,
        "aliases": aliases or [],
    }


def test_fuzzy_match_finds_close_typo():
    registry = [_registry_row("ent_1", "Qdrant")]
    result = resolver._find_via_fuzzy("Qdrnat", registry)
    assert result == "ent_1"


def test_fuzzy_match_checks_aliases_too():
    registry = [_registry_row("ent_1", "Postgres", aliases=["PostgreSQL"])]
    result = resolver._find_via_fuzzy("Postgre SQL", registry)
    assert result == "ent_1"


def test_fuzzy_match_returns_none_below_threshold():
    registry = [_registry_row("ent_1", "Qdrant")]
    result = resolver._find_via_fuzzy("MySQL", registry)
    assert result is None


def test_fuzzy_match_returns_none_for_empty_registry():
    assert resolver._find_via_fuzzy("Qdrant", []) is None


def test_fuzzy_match_picks_best_scoring_candidate():
    registry = [
        _registry_row("ent_1", "Neo4j"),
        _registry_row("ent_2", "Neo4J"),  # near-identical, different casing
    ]
    result = resolver._find_via_fuzzy("neo4j", registry)
    # Either is a valid "best" match given identical scores after
    # lowercasing — what matters is *some* match above threshold is found.
    assert result in {"ent_1", "ent_2"}
