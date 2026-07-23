"""Step 2 · Episode Splitter — DSPy module (Layer 3).

Groups consecutive turns from a NormalizedSession into semantic episodes.
Splits on: topic change, new project mention, decision point, new entity
introduced. Does NOT split by token count — splits by meaning.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import dspy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from schema.memory_event import Episode  # noqa: E402
from schema.turn import NormalizedSession, Turn  # noqa: E402

DEFAULT_WINDOW_SIZE = 6  # spec: 4-6 turns, configurable


class SplitEpisodes(dspy.Signature):
    """Split a conversation into semantic episodes. Each episode covers one
    coherent topic or task. Split on topic change, new project mention,
    decision point, or new entity introduced. Do NOT split by turn count —
    split by meaning. Return episode boundaries as message_id pairs plus a
    short topic hint for each episode.
    """

    turns: str = dspy.InputField(
        desc="Numbered list of turns as 'role[message_id]: content', one per line."
    )
    episode_boundaries: str = dspy.OutputField(
        desc=(
            "One episode per line, format: "
            "'<start_message_id>|<end_message_id>|<topic_hint>'. "
            "Cover every turn exactly once, in order."
        )
    )


def _format_turns_for_prompt(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        content = t.content.replace("\n", " ")[:500]  # keep prompt compact
        lines.append(f"{t.role}[{t.message_id}]: {content}")
    return "\n".join(lines)


def _parse_boundaries(
    raw_output: str, valid_message_ids: set[str]
) -> list[tuple[str, str, str]]:
    """Parse the LLM's pipe-delimited boundary lines, discarding malformed
    or out-of-range lines defensively (LLM output is not always clean).
    """
    boundaries = []
    for line in raw_output.strip().splitlines():
        line = line.strip().strip("-* ")
        if not line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        start_id, end_id = parts[0], parts[1]
        topic_hint = parts[2] if len(parts) > 2 else "untitled"
        if start_id in valid_message_ids and end_id in valid_message_ids:
            boundaries.append((start_id, end_id, topic_hint))
    return boundaries


def _fallback_single_episode(turns: list[Turn]) -> list[tuple[str, str, str]]:
    """If the LLM output can't be parsed, fall back to treating the whole
    window as one episode rather than dropping data.
    """
    if not turns:
        return []
    return [(turns[0].message_id, turns[-1].message_id, "untitled")]


class EpisodeSplitter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(SplitEpisodes)

    def forward(self, turns: list[Turn]) -> list[tuple[str, str, str]]:
        if not turns:
            return []

        message_ids = {t.message_id for t in turns}
        turns_text = _format_turns_for_prompt(turns)

        try:
            result = self.predict(turns=turns_text)
            boundaries = _parse_boundaries(result.episode_boundaries, message_ids)
        except Exception:
            boundaries = []

        if not boundaries:
            boundaries = _fallback_single_episode(turns)

        return boundaries


def _fill_coverage_gaps(
    boundaries: list[tuple[str, str, str]], ordered_ids: list[str]
) -> list[tuple[str, str, str]]:
    """Ensure every turn in `ordered_ids` is covered by exactly one episode.

    The LLM's boundaries are index ranges over `ordered_ids`; overlaps are
    resolved by trusting the first boundary that claims a turn, and any
    uncovered indices are merged into the nearest adjacent episode (or, if
    none exists, promoted to their own single-turn episode) so no turn is
    ever silently dropped.
    """
    n = len(ordered_ids)
    if n == 0:
        return []

    id_to_idx = {mid: i for i, mid in enumerate(ordered_ids)}
    claimed = [False] * n
    ranges: list[list] = []  # [start_idx, end_idx, topic_hint]

    for start_id, end_id, topic_hint in boundaries:
        start_idx, end_idx = id_to_idx[start_id], id_to_idx[end_id]
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        # Skip indices already claimed by an earlier (higher-priority) range
        if all(claimed[start_idx : end_idx + 1]):
            continue
        for i in range(start_idx, end_idx + 1):
            claimed[i] = True
        ranges.append([start_idx, end_idx, topic_hint])

    ranges.sort(key=lambda r: r[0])

    # Fill any uncovered indices into the nearest neighboring range
    i = 0
    while i < n:
        if claimed[i]:
            i += 1
            continue
        # find contiguous uncovered block [i, j)
        j = i
        while j < n and not claimed[j]:
            j += 1
        # try to extend the range ending just before i, else the range starting at j
        extended = False
        for r in ranges:
            if r[1] == i - 1:
                r[1] = j - 1
                extended = True
                break
        if not extended:
            for r in ranges:
                if r[0] == j:
                    r[0] = i
                    extended = True
                    break
        if not extended:
            ranges.append([i, j - 1, "untitled"])
        for k in range(i, j):
            claimed[k] = True
        i = j

    ranges.sort(key=lambda r: r[0])
    return [(ordered_ids[r[0]], ordered_ids[r[1]], r[2]) for r in ranges]


def split_into_episodes(
    session: NormalizedSession, window_turns: list[Turn]
) -> list[Episode]:
    """Run the splitter over a window of turns and materialize `Episode`
    objects (grouped turn text + message ids) ready for classification.
    """
    splitter = EpisodeSplitter()
    boundaries = splitter(turns=window_turns)

    turns_by_id = {t.message_id: t for t in window_turns}
    ordered_ids = [t.message_id for t in window_turns]
    boundaries = _fill_coverage_gaps(boundaries, ordered_ids)

    episodes: list[Episode] = []
    for start_id, end_id, topic_hint in boundaries:
        start_idx = ordered_ids.index(start_id)
        end_idx = ordered_ids.index(end_id)
        episode_ids = ordered_ids[start_idx : end_idx + 1]
        episode_turns = [turns_by_id[mid] for mid in episode_ids]
        episode_text = "\n".join(f"{t.role}: {t.content}" for t in episode_turns)

        episodes.append(
            Episode(
                episode_id=f"ep_{uuid.uuid4().hex[:12]}",
                session_id=session.session_id,
                source_agent=session.source_agent,
                topic_hint=topic_hint,
                message_ids=episode_ids,
                text=episode_text,
            )
        )

    return episodes
