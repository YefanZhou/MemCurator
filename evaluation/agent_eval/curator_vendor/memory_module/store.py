"""MemoryStore — the main CRUD + search + prompt-building facade."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from .retriever import SimCSERetriever
from .schema import MemoryEntry, SearchResult

logger = logging.getLogger(__name__)

_DEFAULT_MAX_POS = 8
_DEFAULT_MAX_NEG = 8


class MemoryStore:
    """Persistent JSONL-backed memory with dense retrieval.

    Parameters
    ----------
    path : str
        Path to the JSONL file. Created on first :meth:`add` if it does
        not exist.
    retriever : SimCSERetriever or None
        Optionally inject a pre-built retriever.  When *None* (default),
        one is created lazily on the first :meth:`search` call.
    top_k : int
        Default number of results returned by :meth:`search`.
    max_pos_examples : int
        Max positive examples used when building prompts.
    max_neg_examples : int
        Max negative examples used when building prompts.
    """

    def __init__(
        self,
        path: str,
        *,
        retriever: SimCSERetriever | None = None,
        top_k: int = 8,
        max_pos_examples: int = _DEFAULT_MAX_POS,
        max_neg_examples: int = _DEFAULT_MAX_NEG,
        retrieve_enabled: bool = True,
    ) -> None:
        self.path = path
        self.top_k = top_k
        self.max_pos = max_pos_examples
        self.max_neg = max_neg_examples
        self.retrieve_enabled = retrieve_enabled

        self._retriever = retriever
        self._entries: List[MemoryEntry] = []
        self._deleted: set[int] = set()
        self._next_id = 0
        self._key_embeddings: Optional[torch.Tensor] = None
        self._dirty = False

        if os.path.exists(path):
            self._load()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the JSONL file into ``_entries``."""
        entries: List[MemoryEntry] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for ln, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(MemoryEntry.from_dict(json.loads(line)))
                except Exception as exc:
                    logger.warning("Skipping line %d in %s: %s", ln, self.path, exc)
        self._entries = entries
        self._deleted = set()
        self._next_id = len(entries)
        self._dirty = False
        logger.info("Loaded %d memory entries from %s", len(entries), self.path)

    def _flush(self) -> None:
        """Rewrite the JSONL file from the in-memory list (compacts deletes)."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        live = [e for i, e in enumerate(self._entries) if i not in self._deleted]
        with open(self.path, "w", encoding="utf-8") as fh:
            for entry in live:
                fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._entries = live
        self._deleted = set()
        self._next_id = len(live)
        self._dirty = False

    def _append_line(self, entry: MemoryEntry) -> None:
        """Append a single entry to the JSONL file."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------

    def _ensure_retriever(self) -> SimCSERetriever:
        if self._retriever is None:
            self._retriever = SimCSERetriever()
        return self._retriever

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry, *, persist: bool = True) -> int:
        """Append *entry* to memory.  Returns the entry id.

        When *persist* is False the entry is added to the in-memory list
        and embedding cache but **not** written to the JSONL file.  Call
        :meth:`_flush` later to persist all pending entries.  This is
        useful when the caller manages its own checkpoint / flush cycle.
        """
        entry_id = self._next_id
        self._entries.append(entry)
        self._next_id += 1
        if persist:
            self._append_line(entry)
        else:
            self._dirty = True
        self._embed_entry(entry)
        return entry_id

    def _embed_entry(self, entry: MemoryEntry) -> None:
        """Embed a single entry and append to the cache."""
        try:
            retriever = self._ensure_retriever()
            vec = retriever.embed([entry.question])
            if self._key_embeddings is None:
                self._key_embeddings = vec
            else:
                self._key_embeddings = torch.cat([self._key_embeddings, vec], dim=0)
        except Exception as exc:
            logger.warning("Failed to embed new entry, cache invalidated: %s", exc)
            self._key_embeddings = None

    def _rebuild_embeddings(self) -> None:
        """Rebuild the full embedding cache from live entries."""
        live = [e for i, e in enumerate(self._entries) if i not in self._deleted]
        if not live:
            self._key_embeddings = None
            return
        try:
            retriever = self._ensure_retriever()
            self._key_embeddings = retriever.embed([e.question for e in live])
        except Exception as exc:
            logger.warning("Failed to rebuild embedding cache: %s", exc)
            self._key_embeddings = None

    def get(self, entry_id: int) -> Optional[MemoryEntry]:
        if entry_id in self._deleted or entry_id < 0 or entry_id >= len(self._entries):
            return None
        return self._entries[entry_id]

    def update(self, entry_id: int, **kwargs: Any) -> None:
        """Update fields on an existing entry.

        Accepted keyword args match :class:`MemoryEntry` fields, e.g.
        ``store.update(eid, reward=0.5, tags=["math"])``.

        Triggers a full JSONL rewrite so the file stays in sync.
        """
        if entry_id in self._deleted or entry_id < 0 or entry_id >= len(self._entries):
            raise KeyError(f"No entry with id {entry_id}")
        if "question" in kwargs:
            raise ValueError(
                "Cannot update 'question' — it is the embedding key. "
                "Delete and re-add the entry instead."
            )
        entry = self._entries[entry_id]
        for k, v in kwargs.items():
            if not hasattr(entry, k):
                raise AttributeError(f"MemoryEntry has no field '{k}'")
            setattr(entry, k, v)
        self._flush()

    def delete(self, entry_id: int) -> None:
        """Soft-delete an entry.  Compacted on next :meth:`reload`."""
        if entry_id < 0 or entry_id >= len(self._entries):
            raise KeyError(f"No entry with id {entry_id}")

        if self._key_embeddings is not None:
            # Find the local row index: count live entries before entry_id
            local_idx = sum(
                1 for i in range(entry_id) if i not in self._deleted
            )
            keep = [j for j in range(self._key_embeddings.shape[0]) if j != local_idx]
            self._key_embeddings = self._key_embeddings[keep] if keep else None

        self._deleted.add(entry_id)

    @property
    def entries(self) -> List[MemoryEntry]:
        """Return a list of all live (non-deleted) entries."""
        return [e for i, e in enumerate(self._entries) if i not in self._deleted]

    def __len__(self) -> int:
        return len(self._entries) - len(self._deleted)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """Dense-retrieve the closest memory entries to *query*.

        Uses a cached embedding matrix so that existing entries are never
        re-embedded.  Only the *query* is embedded on each call.
        """
        top_k = top_k or self.top_k
        live_entries = [(i, e) for i, e in enumerate(self._entries) if i not in self._deleted]
        if not live_entries:
            return []

        retriever = self._ensure_retriever()

        # Rebuild cache if missing or stale (e.g. after delete/reload)
        expected_len = len(live_entries)
        if self._key_embeddings is None or self._key_embeddings.shape[0] != expected_len:
            self._rebuild_embeddings()
        if self._key_embeddings is None:
            return []

        try:
            hits = retriever.search_cached(query, self._key_embeddings, top_k=top_k)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            return []

        results: List[SearchResult] = []
        for rank, (local_idx, score) in enumerate(hits, 1):
            real_id, entry = live_entries[local_idx]
            results.append(
                SearchResult(entry=entry, entry_id=real_id, rank=rank, score=round(score, 6))
            )
        return results

    # ------------------------------------------------------------------
    # Prompt building (convenience)
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        *,
        top_k: int | None = None,
        closing: str = "",
        max_pos: int | None = None,
        max_neg: int | None = None,
    ) -> Optional[str]:
        """Search memory and format results as a prompt string.

        Positive examples (reward > 0) and negative examples (reward <= 0)
        are presented separately so the agent can learn from both.

        Side-effect: populates ``self.last_retrieval`` with structured
        metadata about each retrieved entry (for logging / result storage).
        """
        if not self.retrieve_enabled:
            self.last_retrieval: List[Dict[str, Any]] = []
            return None

        results = self.search(query, top_k=top_k)
        if not results:
            self.last_retrieval: List[Dict[str, Any]] = []
            return None

        max_pos = max_pos or self.max_pos
        max_neg = max_neg or self.max_neg

        positive = [r for r in results if r.entry.reward > 0]
        negative = [r for r in results if r.entry.reward <= 0]

        shown_pos = positive[:max_pos]
        shown_neg = negative[:max_neg]

        self.last_retrieval = [
            {
                "entry_id": r.entry_id,
                "rank": r.rank,
                "score": round(r.score, 4),
                **r.entry.to_dict(),
            }
            for r in shown_pos + shown_neg
        ]

        parts: List[str] = []

        if shown_pos:
            parts.append(
                f"Positive Examples (reward>0) — Showing {len(shown_pos)} of {len(positive)}:"
            )
            for i, r in enumerate(shown_pos, 1):
                parts.append(self._format_case(i, r.entry))

        if shown_neg:
            parts.append(
                f"Negative Examples (reward<=0) — Showing {len(shown_neg)} of {len(negative)}:"
            )
            for i, r in enumerate(shown_neg, 1):
                parts.append(self._format_case(i, r.entry))

        if closing:
            parts.append(closing)

        return "\n".join(parts) if parts else None

    @staticmethod
    def _format_case(idx: int, entry: MemoryEntry) -> str:
        """Format a single example for the prompt."""
        traj = entry.trajectory
        try:
            parsed = json.loads(traj)
            # Plan JSON: {"plan": [{"id": 1, "description": "..."}]}
            if isinstance(parsed, dict) and "plan" in parsed:
                steps = parsed["plan"]
                if isinstance(steps, list) and steps:
                    traj = "\n".join(f"{s['id']}. {s['description']}" for s in steps)
            # Message chain: [{"role": "assistant", "content": "..."}, ...]
            elif isinstance(parsed, list):
                parts = []
                for msg in parsed:
                    role = msg.get("role", "")
                    content = msg.get("content") or ""
                    if role == "assistant" and content:
                        parts.append(f"  [{role}] {content}")
                    elif role == "tool" and content:
                        name = msg.get("name", "tool")
                        preview = content[:200] + "..." if len(content) > 200 else content
                        parts.append(f"  [{name}] {preview}")
                traj = "\n".join(parts) if parts else traj
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

        lines = [f"Example {idx}:", f"  Question: {entry.question}"]
        if entry.insights:
            lines.append(f"  Insight: {entry.insights}")
        if entry.tags:
            lines.append(f"  Tags: {', '.join(entry.tags)}")
        if traj:
            lines.append(f"  Trajectory: {traj}")
        if entry.answer:
            lines.append(f"  Answer: {entry.answer}")
        gt = entry.metadata.get("ground_truth") if entry.metadata else None
        if gt:
            lines.append(f"  Ground Truth: {gt}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-read the JSONL file from disk, compacting any soft deletes."""
        if self._deleted:
            self._flush()
        self._load()
        self._key_embeddings = None  # rebuilt lazily on next search
