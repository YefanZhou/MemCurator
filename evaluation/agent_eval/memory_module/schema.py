"""Data schemas for the memory module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    """A single memory record.

    Covers three schema variants:
      - Raw trajectories: question + trajectory + reward
      - + insights/skills: populate the ``insights`` field
      - + tags: populate the ``tags`` field
    """

    question: str
    answer: str = ""
    trajectory: str = ""
    reward: float = 0.0
    tags: List[str] = field(default_factory=list)
    insights: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- serialisation helpers ------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"question": self.question}
        if self.answer:
            d["answer"] = self.answer
        if self.trajectory:
            d["trajectory"] = self.trajectory
        d["reward"] = self.reward
        if self.tags:
            d["tags"] = self.tags
        if self.insights:
            d["insights"] = self.insights
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryEntry":
        """Construct from a dict, with backward-compat for old schemas.

        Old format uses ``plan`` instead of ``trajectory`` and ``case``
        instead of ``question``.
        """
        question = d.get("question") or d.get("case", "")
        trajectory = d.get("trajectory") or d.get("plan", "")
        answer = d.get("answer", "")
        reward_raw = d.get("reward", 0)
        if isinstance(reward_raw, str):
            reward_raw = 1.0 if reward_raw == "positive" else 0.0
        reward = float(reward_raw)

        # Old schema: case_label -> reward
        if "case_label" in d and "reward" not in d:
            reward = 1.0 if d["case_label"] == "positive" else 0.0

        tags = d.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]

        return cls(
            question=question,
            answer=answer,
            trajectory=trajectory,
            reward=reward,
            tags=tags,
            insights=d.get("insights", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """A memory entry returned by a search, with ranking info."""

    entry: MemoryEntry
    entry_id: int
    rank: int
    score: float
