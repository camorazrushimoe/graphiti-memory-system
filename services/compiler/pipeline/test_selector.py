"""Unit tests for the Memory Selector — pure Python, no LLM needed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from schema.memory_event import EpisodeType, MemoryItem
from services.compiler.pipeline import selector


def _item(
    text: str, confidence: float, type_: EpisodeType = EpisodeType.FACT
) -> MemoryItem:
    return MemoryItem(text=text, type=type_, confidence=confidence)


def test_discards_below_hard_confidence_floor():
    items = [_item("The sky is sometimes blue-ish today", 0.35)]
    assert selector.select_memory_items(items) == []


def test_discards_between_floor_and_keep_threshold():
    items = [_item("We might possibly consider Postgres", 0.5)]
    assert selector.select_memory_items(items) == []


def test_keeps_at_or_above_keep_threshold():
    items = [_item("Use Graphiti as the memory layer", 0.91, EpisodeType.DECISION)]
    kept = selector.select_memory_items(items)
    assert len(kept) == 1
    assert kept[0].text == "Use Graphiti as the memory layer"


def test_discards_pure_acknowledgment_even_with_high_confidence():
    items = [_item("ok", 0.95), _item("Got it!", 0.9), _item("sounds good.", 0.99)]
    assert selector.select_memory_items(items) == []


def test_keeps_substantive_text_containing_ack_word_as_substring():
    # "ok" appears inside a longer, substantive sentence — must not be
    # discarded as a pure acknowledgment.
    items = [
        _item(
            "Broke it down into smaller tasks so the scope feels manageable.",
            0.8,
            EpisodeType.TASK,
        )
    ]
    assert len(selector.select_memory_items(items)) == 1


def test_mixed_batch_keeps_only_qualifying_items():
    items = [
        _item("Use Neo4j for the graph core", 0.9, EpisodeType.DECISION),
        _item("thanks", 0.99),
        _item("maybe consider it later", 0.5),
        _item("no", 0.2),
    ]
    kept = selector.select_memory_items(items)
    assert len(kept) == 1
    assert kept[0].text == "Use Neo4j for the graph core"
