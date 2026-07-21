"""Abstract base classes for memory write/read strategies.

Every memory approach — the trained curator, Synapse, ExpeL, AWM,
ReasoningBank, or a future method — implements this interface.
The evaluation loop in ``evaluate.py`` uses ``WriteStrategy`` to decide
*when* and *how* to write memories after each episode, and
``ReadStrategy`` to produce the curated-memory text the downstream agent
sees at inference time.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..memory_module.schema import MemoryEntry
from ..memory_module.store import MemoryStore


@dataclass
class ReadResult:
    """Return value of :meth:`ReadStrategy.read`.

    Attributes
    ----------
    text : str
        The curated-memory string passed to the downstream agent.
    extra_info : dict
        Strategy-specific debug info logged in the result record.
        Typical keys: ``retrieved_text`` (raw retrieval before curation),
        ``num_retrieved``, ``rule_count``, etc.
    """
    text: str
    extra_info: Dict[str, Any] = field(default_factory=dict)


class WriteStrategy(ABC):
    """Transforms a completed episode record into a MemoryEntry for storage.

    Subclasses override :meth:`transform` and optionally :meth:`should_write`.
    """

    def should_write(self, record: dict) -> bool:
        """Whether to write this episode to memory.

        Default behaviour: write only on success.  Strategies that learn
        from failures (ExpeL, ReasoningBank) override this to return True
        unconditionally.
        """
        return bool(record.get("correct", False))

    @abstractmethod
    async def transform(
        self,
        record: dict,
        semaphore: asyncio.Semaphore,
        store: Optional[MemoryStore] = None,
    ) -> Optional[MemoryEntry]:
        """Return a :class:`MemoryEntry` to store, or *None* to skip.

        Parameters
        ----------
        record : dict
            The result dict from ``_process_one``.  Contains at least:
            ``question``, ``trajectory``, ``pred``, ``score``, ``correct``,
            ``ground_truth``, ``downstream_response``.
        semaphore : asyncio.Semaphore
            Shared rate-limiter for API calls.
        store : MemoryStore or None
            The current memory store.  Strategies that need to inspect
            existing entries (e.g. ExpeL contrastive extraction) use
            this to look up prior results for the same question.
        """
        ...


class ReadStrategy(ABC):
    """Produces the ``curated_memory`` string from the memory store."""

    @abstractmethod
    async def read(
        self,
        question: str,
        store: MemoryStore,
        top_k: int,
        semaphore: asyncio.Semaphore,
    ) -> ReadResult:
        """Return a :class:`ReadResult` with the curated-memory text and
        debug info.

        Parameters
        ----------
        question : str
            The current task question (used as retrieval query).
        store : MemoryStore
            The in-memory JSONL-backed store with dense retrieval.
        top_k : int
            Maximum number of entries to retrieve.
        semaphore : asyncio.Semaphore
            Shared rate-limiter for API calls (unused by non-LLM readers).
        """
        ...
