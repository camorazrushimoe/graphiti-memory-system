"""Unit tests for the Contradiction Checker's pure logic (no LLM/DB calls).

Covers `is_near_duplicate` (the cheap rapidfuzz rule tier) and
`action_for_score` (the spec's three-way threshold mapping) — both
extracted as standalone pure functions in session 7 specifically to make
this kind of testing possible without mocking Postgres/Instructor. The
full `check_contradiction` async flow (DB candidate lookup + LLM judgment)
is exercised only by the live e2e tests recorded in the tech-spec
implementation log, not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.compiler.pipeline import contradiction


def test_near_duplicate_identical_text():
    assert contradiction.is_near_duplicate(
        "Use Postgres for metadata storage", "Use Postgres for metadata storage"
    )


def test_near_duplicate_minor_wording_difference():
    assert contradiction.is_near_duplicate(
        "Use Postgres for metadata storage.", "use postgres for metadata storage"
    )


def test_not_near_duplicate_for_different_statements():
    assert not contradiction.is_near_duplicate(
        "Use Postgres for metadata storage", "Switch to MySQL instead of Postgres"
    )


def test_action_for_score_none_when_llm_says_no_contradiction():
    assert contradiction.action_for_score(contradicts=False, score=0.95) == "none"


def test_action_for_score_none_below_review_threshold():
    assert contradiction.action_for_score(contradicts=True, score=0.5) == "none"


def test_action_for_score_flag_for_review_in_middle_band():
    assert (
        contradiction.action_for_score(contradicts=True, score=0.6) == "flag_for_review"
    )
    assert (
        contradiction.action_for_score(contradicts=True, score=0.85)
        == "flag_for_review"
    )


def test_action_for_score_auto_update_above_threshold():
    assert contradiction.action_for_score(contradicts=True, score=0.86) == "auto_update"
    assert contradiction.action_for_score(contradicts=True, score=1.0) == "auto_update"
