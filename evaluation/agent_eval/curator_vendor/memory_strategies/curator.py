"""Curator strategy — the trained Memory Curator as a read/write strategy.

This wraps the existing curator LLM pipeline (retrieve → curator inference →
curated memory) into the unified WriteStrategy / ReadStrategy interface so
it is treated identically to the baseline strategies.

Write: store raw successful trajectories (same as Synapse).
Read:  retrieve top-k from MemoryStore, format as "Retrieved Memories",
       call the curator LLM to produce a curated summary.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, InternalServerError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..memory_module.schema import MemoryEntry
from ..memory_module.store import MemoryStore
from ..prompt_loader import load_template
from .strategies import ReadResult, ReadStrategy, WriteStrategy

logger = logging.getLogger(__name__)


class CuratorWriter(WriteStrategy):
    """Store raw successful trajectories — identical to the default / Synapse."""

    async def transform(
        self, record: dict, semaphore: asyncio.Semaphore,
        store: Optional["MemoryStore"] = None,
    ) -> Optional[MemoryEntry]:
        return MemoryEntry(
            question=record["question"],
            trajectory=record.get("trajectory", ""),
            answer=record.get("pred", ""),
            reward=1.0,
            metadata={"ground_truth": record.get("ground_truth", "")},
        )


class CuratorReader(ReadStrategy):
    """Retrieve memories, call the trained curator LLM to curate them."""

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        *,
        max_tokens: int = 512,
        template_dir: str | None = None,
        sampling: dict | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        # Extra create() kwargs (temperature/top_p/extra_body{top_k,chat_template_kwargs}).
        # Empty => no change. Built by create_strategies from --memory_* args.
        self._sampling = sampling or {}
        self._system_prompt = load_template(
            "curator_system.txt", template_dir=template_dir,
        )

    async def read(
        self,
        question: str,
        store: MemoryStore,
        top_k: int,
        semaphore: asyncio.Semaphore,
    ) -> ReadResult:
        retrieved_text = ""
        num_retrieved = 0
        if len(store) > 0:
            search_results = store.search(question, top_k=top_k)
            if search_results:
                num_retrieved = len(search_results)
                parts = [
                    MemoryStore._format_case(j, r.entry)
                    for j, r in enumerate(search_results, 1)
                ]
                retrieved_text = "\n".join(parts)

        curator_output = await self._run_curator_inference(
            question, retrieved_text, semaphore,
        )
        print('curator_output df', curator_output)
        #sys.exit()
        return ReadResult(
            text=curator_output,
            extra_info={
                "retrieved_text": retrieved_text,
                "num_retrieved": num_retrieved,
            },
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _run_curator_inference(
        self,
        question: str,
        retrieved_text: str,
        semaphore: asyncio.Semaphore,
    ) -> str:
        if retrieved_text:
            user_content = f"Question: {question}\n\nRetrieved Memories:\n{retrieved_text}"
        else:
            user_content = f"Question: {question}\n\nNo past memories available yet."

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        async with semaphore:
            # [SAMPLING-DEBUG] verified 2026-07-04 (curator create pass-through). Uncomment to re-check.
            # print(f"[SAMPLING-DEBUG][curator create] model={self._model} "
            #       f"sampling_applied={self._sampling}", flush=True)
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=self._max_tokens,
                **self._sampling,
            )
        return resp.choices[0].message.content or ""
