"""Step 7 · Memory Selector — Layer 3.

Final quality gate before a candidate `MemoryItem` is sent on to the
Entity Resolver / Contradiction Checker / Graphiti+Qdrant+Postgres writers,
per spec Step 7:

  - Discard rules: confidence < 0.4, trivial small-talk, pure
    acknowledgments ("ok", "got it").
  - Keeps: decisions, tasks, preferences, facts with confidence >= 0.6.

Implementation notes
---------------------
- Most small-talk is already filtered upstream at the *episode* level: the
  Episode Classifier's `meta` type is skipped entirely before extraction
  even runs (see `services/compiler/main.py`). This module additionally
  catches *item-level* acknowledgments that can still surface inside an
  otherwise substantive episode (e.g. one turn in a `decision` episode is
  just "ok, sounds good" but gets extracted as its own low-value item).
- The spec's keep-list only names four of the eight non-meta
  `EpisodeType`s explicitly (decision/task/preference/fact). The other
  types (question/idea/constraint/entity_update) aren't mentioned either
  way. To avoid silently dropping real signal that earlier pipeline
  sessions have already relied on (see the tech-spec implementation log:
  live e2e tests with `entity_update` items were accepted end-to-end
  before this module existed), the same confidence >= 0.6 threshold is
  applied uniformly across all non-meta types rather than restricting to
  just the four named ones. Revisit if `question`/`idea`/`constraint`
  items turn out to need different thresholds in practice.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from schema.memory_event import MemoryItem  # noqa: E402

logger = logging.getLogger("compiler.selector")

# Per spec: confidence < 0.4 is an unconditional discard.
DISCARD_CONFIDENCE_THRESHOLD = 0.4
# Per spec: keep decisions/tasks/preferences/facts (and, per the docstring
# above, all other non-meta types) with confidence >= 0.6.
KEEP_CONFIDENCE_THRESHOLD = 0.6

# Pure acknowledgment / trivial small-talk phrases per spec example
# ("ok", "got it"). Matched against the *whole* item text (after
# stripping punctuation/whitespace) so a longer sentence that happens to
# contain "ok" as a substring isn't discarded.
_ACK_PHRASES = {
    "ok",
    "okay",
    "got it",
    "sounds good",
    "sure",
    "thanks",
    "thank you",
    "yes",
    "no",
    "yep",
    "yeah",
    "nope",
    "alright",
    "cool",
    "understood",
    "noted",
    "great",
    "perfect",
}
_STRIP_PUNCT_RE = re.compile(r"[^\w\s]")


def _is_pure_acknowledgment(text: str) -> bool:
    normalized = _STRIP_PUNCT_RE.sub("", text).strip().lower()
    return normalized in _ACK_PHRASES


def select_memory_items(items: list[MemoryItem]) -> list[MemoryItem]:
    """Apply the Memory Selector quality gate to a batch of candidate
    items, returning only the ones that should proceed to Entity
    Resolver / Contradiction Checker / graph+vector+metadata writes.
    """
    kept: list[MemoryItem] = []
    for item in items:
        if item.confidence < DISCARD_CONFIDENCE_THRESHOLD:
            logger.info(
                "memory selector: discarded (confidence %.2f < %.2f): %r",
                item.confidence,
                DISCARD_CONFIDENCE_THRESHOLD,
                item.text[:80],
            )
            continue
        if _is_pure_acknowledgment(item.text):
            logger.info(
                "memory selector: discarded (pure acknowledgment): %r", item.text
            )
            continue
        if item.confidence < KEEP_CONFIDENCE_THRESHOLD:
            logger.info(
                "memory selector: discarded (confidence %.2f < %.2f): %r",
                item.confidence,
                KEEP_CONFIDENCE_THRESHOLD,
                item.text[:80],
            )
            continue
        kept.append(item)
    return kept
