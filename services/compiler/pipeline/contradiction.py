"""Step 6 · Contradiction Checker — Layer 3.

Compares a new (already entity-resolved) `MemoryItem` against existing
active facts that share at least one canonical entity, per spec Step 6.
Logic, per spec:

  - If new fact contradicts existing -> mark old as `status: outdated`,
    create a `supersedes` link. Both versions are kept (temporal graph
    supports this natively).
  - Contradiction score threshold: > 0.85 triggers automatic update;
    0.6-0.85 flags for review.

Implementation notes
---------------------
- "Same entity scope" is implemented as: any existing active fact of the
  *same type* that shares at least one `canonical_id` with the new item
  (see `common.db.get_active_facts_by_entity`). Restricting to the same
  type avoids comparing, say, a `task` against a `decision` that happen to
  mention the same entity — those aren't contradictions, they're just
  related facts.
- Per spec ("rules + Gemma 4B for ambiguous cases"): a cheap rule handles
  the unambiguous non-contradiction case (near-identical text -> skip the
  LLM call entirely, rapidfuzz similarity >= NEAR_DUPLICATE_THRESHOLD).
  Everything else goes to Gemma 4B via Instructor for a structured
  contradiction judgment, since "contradicts" is a semantic judgment a
  hard rule can't reliably make (e.g. "we use Postgres" vs "we use
  MySQL" contradicts; "we use Postgres" vs "we use Postgres for
  metadata" does not, despite low text similarity).
- Only the single most-recent matching active fact per candidate is sent
  to the LLM as the comparison target (spec doesn't say "compare against
  every existing fact" and doing so would multiply LLM calls per new
  item); if it doesn't contradict, older facts for the same entity are
  assumed non-contradictory too (weak assumption, acceptable for MVP —
  revisit if false negatives on older facts become a problem).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from common import db  # noqa: E402
from schema.memory_event import ContradictionResult, MemoryItem  # noqa: E402
from services.compiler.pipeline.instructor_config import (  # noqa: E402
    LLM_MODEL,
    get_instructor_client,
)

logger = logging.getLogger("compiler.contradiction")

# Per spec: >0.85 -> auto_update, 0.6-0.85 -> flag_for_review, else none.
AUTO_UPDATE_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.6

# rapidfuzz ratio above which two fact texts are treated as restating the
# same information (not a contradiction) without an LLM call — cheap rule
# tier per spec ("rules + Gemma 4B for ambiguous cases").
NEAR_DUPLICATE_THRESHOLD = 92


class _ContradictionJudgment(BaseModel):
    contradicts: bool = Field(
        description=(
            "True only if the new statement directly conflicts with the "
            "existing one (e.g. states the opposite, or a mutually "
            "exclusive choice). False if they're unrelated, complementary, "
            "or simply restate/refine each other."
        )
    )
    contradiction_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0.0 = no contradiction at all, 1.0 = directly and "
            "unambiguously contradicts the existing statement."
        ),
    )


_JUDGE_PROMPT = (
    "EXISTING STATEMENT:\n{existing}\n\n"
    "NEW STATEMENT:\n{new}\n\n"
    "Does the NEW statement contradict the EXISTING statement? Two "
    "statements about the same topic that are simply consistent, "
    "complementary, or one refines the other are NOT a contradiction — "
    "only judge `contradicts=true` if accepting the new statement as true "
    "means the existing one can no longer be true."
)


def _judge_with_llm(existing_text: str, new_text: str) -> tuple[bool, float]:
    client = get_instructor_client()
    try:
        result = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": _JUDGE_PROMPT.format(
                        existing=existing_text, new=new_text
                    ),
                }
            ],
            response_model=_ContradictionJudgment,
            max_retries=2,
        )
    except Exception:
        logger.exception("contradiction checker: LLM judgment call failed")
        return False, 0.0
    return result.contradicts, result.contradiction_score


async def check_contradiction(
    item: MemoryItem, fact_texts_by_id: dict[str, str]
) -> ContradictionResult:
    """Check a new MemoryItem against existing active facts sharing a
    canonical entity, returning update instructions for the caller.

    `fact_texts_by_id` must contain the display text for any candidate
    fact_id this function might compare against (Postgres `facts` doesn't
    store `text` itself — see services/ingest/retrieval.py's docstring for
    the same reasoning — so the caller, which already has Qdrant access,
    is expected to look these up).
    """
    canonical_ids = [e.canonical_id for e in item.entities if e.canonical_id]
    if not canonical_ids:
        return ContradictionResult()

    candidates: list[dict] = []
    seen_fact_ids: set[str] = set()
    for canonical_id in canonical_ids:
        rows = await db.get_active_facts_by_entity(canonical_id, type_=item.type.value)
        for row in rows:
            if row["fact_id"] not in seen_fact_ids:
                seen_fact_ids.add(row["fact_id"])
                candidates.append(row)

    if not candidates:
        return ContradictionResult()

    # Most recent candidate first — see module docstring on why only one
    # comparison target is used per candidate entity set.
    candidates.sort(key=lambda r: r["created_at"], reverse=True)

    for candidate in candidates:
        existing_text = fact_texts_by_id.get(candidate["fact_id"])
        if not existing_text:
            continue  # can't compare without the existing fact's text

        similarity = fuzz.ratio(item.text.lower(), existing_text.lower())
        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            # Cheap rule tier: near-identical restatement, not a
            # contradiction — skip the LLM call for this candidate.
            continue

        contradicts, score = _judge_with_llm(existing_text, item.text)
        if not contradicts or score < REVIEW_THRESHOLD:
            continue

        action = "auto_update" if score > AUTO_UPDATE_THRESHOLD else "flag_for_review"
        logger.info(
            "contradiction checker: new item contradicts fact %s (score=%.2f, action=%s)",
            candidate["fact_id"],
            score,
            action,
        )
        return ContradictionResult(
            contradicts_fact_id=candidate["fact_id"],
            contradiction_score=score,
            action=action,
        )

    return ContradictionResult()
