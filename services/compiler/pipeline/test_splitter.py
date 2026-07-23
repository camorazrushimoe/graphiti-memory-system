"""Unit tests for the Episode Splitter's deterministic gap-filling logic.

No LLM calls here — these test `_fill_coverage_gaps` directly with
synthetic boundary inputs to guarantee full turn coverage regardless of
what the LLM returns.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from services.compiler.pipeline.splitter import _fill_coverage_gaps, _parse_boundaries


def test_fill_coverage_gaps_no_gaps():
    ordered_ids = ["u1", "a1", "u2", "a2"]
    boundaries = [("u1", "a1", "topic1"), ("u2", "a2", "topic2")]
    result = _fill_coverage_gaps(boundaries, ordered_ids)
    assert result == [("u1", "a1", "topic1"), ("u2", "a2", "topic2")]


def test_fill_coverage_gaps_middle_gap_merges_into_previous():
    ordered_ids = ["u1", "a1", "u2", "a2", "u3", "a3"]
    boundaries = [("u1", "u2", "topic1"), ("u3", "a3", "topic2")]
    result = _fill_coverage_gaps(boundaries, ordered_ids)
    # 'a2' is uncovered, should merge into the range ending right before it
    assert result[0] == ("u1", "a2", "topic1")
    assert result[1] == ("u3", "a3", "topic2")
    covered = set()
    for s, e, _ in result:
        si, ei = ordered_ids.index(s), ordered_ids.index(e)
        covered.update(ordered_ids[si : ei + 1])
    assert covered == set(ordered_ids)


def test_fill_coverage_gaps_leading_gap_creates_own_episode():
    ordered_ids = ["u1", "a1", "u2", "a2"]
    boundaries = [("u2", "a2", "topic2")]
    result = _fill_coverage_gaps(boundaries, ordered_ids)
    covered = set()
    for s, e, _ in result:
        si, ei = ordered_ids.index(s), ordered_ids.index(e)
        covered.update(ordered_ids[si : ei + 1])
    assert covered == set(ordered_ids)


def test_fill_coverage_gaps_empty_boundaries_creates_single_episode():
    ordered_ids = ["u1", "a1"]
    result = _fill_coverage_gaps([], ordered_ids)
    covered = set()
    for s, e, _ in result:
        si, ei = ordered_ids.index(s), ordered_ids.index(e)
        covered.update(ordered_ids[si : ei + 1])
    assert covered == set(ordered_ids)


def test_fill_coverage_gaps_overlapping_boundaries_first_wins():
    ordered_ids = ["u1", "a1", "u2", "a2"]
    boundaries = [("u1", "u2", "topic1"), ("a1", "a2", "topic2")]
    result = _fill_coverage_gaps(boundaries, ordered_ids)
    covered = set()
    for s, e, _ in result:
        si, ei = ordered_ids.index(s), ordered_ids.index(e)
        covered.update(ordered_ids[si : ei + 1])
    assert covered == set(ordered_ids)
    # first boundary (u1-u2) should have claimed a1 already
    assert ("u1", "u2", "topic1") in result


def test_parse_boundaries_ignores_malformed_lines():
    valid_ids = {"u1", "a1"}
    raw = "u1|a1|topic one\nnot a valid line\nu1|unknown_id|topic two"
    result = _parse_boundaries(raw, valid_ids)
    assert result == [("u1", "a1", "topic one")]
