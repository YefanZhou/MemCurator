"""Synapse strategy — trajectory-as-exemplar prompting with memory.

Based on: Synapse (arXiv 2306.07863, ICLR 2024)
Reference code: https://github.com/ltzheng/Synapse

Write: store raw successful trajectories as-is.
Read:  retrieve top-k by SimCSE similarity, format as numbered exemplars.
LLM calls: 0

SIMPLIFICATION vs. original paper:
  - Uses SimCSE retrieval instead of text-embedding-ada-002 + FAISS.
  - Skips LLM-based state abstraction (our trajectories are already text,
    not raw HTML observations).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from ..memory_module.schema import MemoryEntry
from ..memory_module.store import MemoryStore
from .strategies import ReadResult, ReadStrategy, WriteStrategy


class SynapseWriter(WriteStrategy):
    """Store raw trajectory on success — identical to the default behaviour."""

    async def transform(
        self, record: dict, semaphore: asyncio.Semaphore,
        store: Optional["MemoryStore"] = None,
    ) -> Optional[MemoryEntry]:
        return MemoryEntry(
            question=record["question"],
            trajectory=record["trajectory"],
            answer=record.get("pred", ""),
            reward=1.0,
            metadata={"ground_truth": record.get("ground_truth", "")},
        )


class SynapseReader(ReadStrategy):
    """Retrieve top-k trajectories and format as numbered exemplars."""

    async def read(
        self,
        question: str,
        store: MemoryStore,
        top_k: int,
        semaphore: asyncio.Semaphore,
    ) -> ReadResult:
        if len(store) == 0:
            return ReadResult(text="")

        results = store.search(question, top_k=top_k)
        if not results:
            return ReadResult(text="")

        parts = [
            MemoryStore._format_case(j, r.entry)
            for j, r in enumerate(results, 1)
        ]
        text = "\n".join(parts)
        return ReadResult(
            text=text,
            extra_info={
                "num_retrieved": len(results),
                "retrieved_text": text,
            },
        )
