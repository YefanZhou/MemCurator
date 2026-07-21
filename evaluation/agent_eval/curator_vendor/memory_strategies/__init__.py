"""Pluggable memory read/write strategies for the evaluation pipeline.

Every memory approach — the trained curator, Synapse, ExpeL, AWM,
ReasoningBank — is a (WriteStrategy, ReadStrategy) pair that plugs into
the evaluation loop.  The ``--memory_strategy`` CLI argument selects which
pair to use.

Registry
--------
- ``curator``      — trained curator LLM (retrieve → curate → inject)
- ``synapse``      — store raw trajectories, retrieve as exemplars (no LLM)
- ``expel``        — contrastive insight extraction with vote-counting rule pool
- ``awm``          — per-trajectory workflow induction via LLM
- ``reasoningbank``— strategy distillation for success & failure episodes
- ``none``         — no memory: empty write and empty read (downstream only)
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

from openai import AsyncOpenAI

from .strategies import ReadResult, ReadStrategy, WriteStrategy
from .synapse import SynapseReader, SynapseWriter


STRATEGY_CHOICES = ["curator", "synapse", "expel", "awm", "reasoningbank", "none"]

_NEEDS_LLM = frozenset({"curator", "expel", "awm", "reasoningbank"})


class _NoOpWriter(WriteStrategy):
    """Never writes — used with ``none`` strategy."""

    def should_write(self, record: dict) -> bool:
        return False

    async def transform(self, record, semaphore, store=None):
        return None


class _NoOpReader(ReadStrategy):
    """Always returns empty — used with ``none`` strategy."""

    async def read(self, question, store, top_k, semaphore):
        return ReadResult(text="")


def _build_client(
    api_base: str | None = None,
    api_key: str | None = None,
) -> AsyncOpenAI:
    """Build an AsyncOpenAI client from optional overrides."""
    kwargs: dict = {}
    if api_base:
        kwargs["base_url"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    gateway_key = os.getenv("X_API_KEY")
    if gateway_key and "api_key" not in kwargs and not kwargs.get("base_url"):
        kwargs["default_headers"] = {"X-Api-Key": gateway_key}
    return AsyncOpenAI(**kwargs)


def create_strategies(
    name: str,
    *,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 512,
    template_dir: Optional[str] = None,
    memory_path: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    enable_thinking: Optional[bool] = None,
) -> Tuple[WriteStrategy, ReadStrategy]:
    """Instantiate the write/read strategy pair for *name*.

    Parameters
    ----------
    name : str
        One of :data:`STRATEGY_CHOICES`.
    model : str or None
        Model name for LLM calls (required for LLM-based strategies).
    api_base : str or None
        Override base URL for the OpenAI client.
    api_key : str or None
        Override API key for the OpenAI client.
    max_tokens : int
        Max completion tokens for strategy LLM calls.
    template_dir : str or None
        Override for prompt template directory.
    memory_path : str or None
        Path to the memory store; used by strategies (e.g. ExpeL) that
        derive auxiliary file paths relative to the store location.
    """
    if name == "none":
        return _NoOpWriter(), _NoOpReader()

    if name == "synapse":
        return SynapseWriter(), SynapseReader()

    assert name in _NEEDS_LLM, (
        f"Unknown memory strategy {name!r}. Choose from: {STRATEGY_CHOICES}"
    )
    assert model is not None, (
        f"--memory_model is required for the {name!r} strategy"
    )
    client = _build_client(api_base=api_base, api_key=api_key)

    # Build the extra create() kwargs once (inline; omit any None). temperature/top_p
    # are top-level; top_k + enable_thinking ride extra_body (vLLM-only). Empty dict
    # => no behavior change. Shared by whichever strategy makes LLM calls.
    sampling: dict = {}
    if temperature is not None:
        sampling["temperature"] = temperature
    if top_p is not None:
        sampling["top_p"] = top_p
    _extra: dict = {}
    if top_k is not None:
        _extra["top_k"] = top_k
    if enable_thinking is not None:
        _extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if _extra:
        sampling["extra_body"] = _extra

    # [SAMPLING-DEBUG] verified 2026-07-04 (memory pass-through). Uncomment to re-check.
    # print(f"[SAMPLING-DEBUG][memory create_strategies] strategy={name} model={model} "
    #       f"sampling={sampling}", flush=True)

    if name == "curator":
        from .curator import CuratorReader, CuratorWriter
        return (
            CuratorWriter(),
            CuratorReader(
                client, model,
                max_tokens=max_tokens,
                template_dir=template_dir,
                sampling=sampling,
            ),
        )

    if name == "reasoningbank":
        from .reasoningbank import ReasoningBankReader, ReasoningBankWriter
        return (
            ReasoningBankWriter(client, model, max_tokens=max_tokens, template_dir=template_dir, sampling=sampling),
            ReasoningBankReader(template_dir=template_dir),
        )

    if name == "awm":
        from .awm import AWMReader, AWMWriter
        return (
            AWMWriter(client, model, max_tokens=max_tokens, template_dir=template_dir, sampling=sampling),
            AWMReader(),
        )

    if name == "expel":
        from .expel import ExpeLReader, ExpeLWriter
        return (
            ExpeLWriter(
                client, model,
                memory_path=memory_path,
                max_tokens=max_tokens,
                template_dir=template_dir,
                sampling=sampling,
            ),
            ExpeLReader(template_dir=template_dir),
        )

    raise ValueError(
        f"Unknown memory strategy {name!r}. Choose from: {STRATEGY_CHOICES}"
    )
