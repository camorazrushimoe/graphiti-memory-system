"""Step 3 · Episode Classifier — DSPy module (Layer 3).

Classifies what kind of memory an Episode contains, so the extractor knows
which fields to focus on. Keeps prompts focused per spec.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import dspy

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from schema.memory_event import Episode, EpisodeType  # noqa: E402

_EPISODE_TYPE_VALUES = [e.value for e in EpisodeType]
EpisodeTypeLiteral = Literal[
    "fact",
    "decision",
    "task",
    "preference",
    "question",
    "idea",
    "constraint",
    "entity_update",
    "meta",
]


class ClassifyEpisode(dspy.Signature):
    """Classify what kind of memory this episode contains. Choose exactly
    one type: fact (a stated piece of information), decision (a choice
    that was made), task (something to be done), preference (a like/
    dislike or working style), question (an open question, not yet
    answered), idea (a suggestion or brainstorm, not yet decided),
    constraint (a limitation or requirement), entity_update (new info
    about an existing entity), or meta (small talk / acknowledgments /
    not memory-worthy).
    """

    episode_text: str = dspy.InputField()
    episode_type: EpisodeTypeLiteral = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")


class EpisodeClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ClassifyEpisode)

    def forward(self, episode_text: str) -> tuple[EpisodeType, float]:
        try:
            result = self.predict(episode_text=episode_text)
            episode_type = EpisodeType(result.episode_type)
            confidence = float(result.confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            # Defensive fallback — treat unparseable output as low-confidence fact
            episode_type = EpisodeType.FACT
            confidence = 0.3
        return episode_type, confidence


def classify_episode(episode: Episode) -> Episode:
    """Classify an Episode in place (returns a new Episode with type/confidence set)."""
    classifier = EpisodeClassifier()
    episode_type, confidence = classifier(episode_text=episode.text)
    return episode.model_copy(
        update={"episode_type": episode_type, "confidence": confidence}
    )
