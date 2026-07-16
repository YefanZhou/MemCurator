"""curator_v1 — FAITHFUL vendored MemCurator for ALFWorld.

This is the accurate transfer of sea-mem-policy's `curator/` memory method (see
`curator_vendor/` for byte-identical copies of the original code + PROVENANCE.md).
It replaces the earlier hand-reimplemented `curator_alfworld.py` (kept unchanged as
memory_type='curator') with a thin adapter over the ORIGINAL classes, so:

  * the curator SYSTEM PROMPT is the verbatim `curator_vendor/templates/curator_system.txt`
    (loaded via the vendored prompt_loader),
  * the user turn is the vendored `CuratorReader._run_curator_inference` verbatim
    (`Question: {q}\n\nRetrieved Memories:\n{retrieved_text}` / "...No past memories..."),
  * each retrieved memory is rendered by the vendored `MemoryStore._format_case`
    (`Example {idx}:` + Question/Insight/Tags/Trajectory/Answer/Ground Truth),
  * the store is the vendored `MemoryEntry`/`MemoryStore` JSONL schema
    (question/answer/trajectory/reward/tags/insights/metadata) — richer than curator's
    4-field file, and ready to store failures/insights/tags later.

What this adapter changes (orchestration only; the vendored prompt/format/inference code is
untouched), all deliberate and documented:
  * RETRIEVAL = BM25, ALIGNED WITH reasoningbank_alfworld.py: langchain BM25Retriever built
    from Document(page_content=<NL task>, metadata={idx}), k=retrieve_num, rebuilt when k
    differs, queried via .invoke(query) — same library/key/handling reasoningbank uses (NOT
    the vendored dense SimCSE). Done via a `MemoryStore` subclass overriding only `search()`.
  * TASK_CONTEXT: the BM25 key is always the short NL task; `curator_question` (may be the
    obs0-enriched current task) is what the curator LLM sees as "Question:" — same decoupling
    as the current curator's --task_context.
  * CURATOR_ON_EMPTY: the vendored read() ALWAYS calls the curator LLM (even on empty store) —
    that equals curator_on_empty=True. With curator_on_empty=False we short-circuit to "" (Q2).
  * SAMPLING: the vendored CuratorReader `sampling` dict is built from the CURATION_* env knobs
    exactly as `curator/memory_strategies/__init__.py` does.
  * WRITE: success-only (matches CuratorWriter.should_write default). `answer` is left empty
    (ALFWorld has no separate prediction distinct from the trajectory; avoids _format_case
    printing the trajectory twice). Set STORE_FAILURES=1 later to also store failures.
  * LOGGING: every read-time call is logged to curator_calls.jsonl WITH per-retrieved-entry
    {store_index, score, question} (fixes the traceability gap from the analysis).

Backend: HTTP/OpenAI-compatible via --curation_base_url (AsyncOpenAI). vLLM's `vllm serve`
exposes an OpenAI-compatible /v1 endpoint, so AsyncOpenAI(base_url=<curation_base_url>) calls
the vLLM-served curator model directly (top_k/enable_thinking ride extra_body, which the vLLM
OpenAI server accepts) — same endpoint the current curator's HTTP path uses. No in-process
vLLM load, so curation_base_url is required.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from curator_vendor.memory_module.schema import MemoryEntry, SearchResult
from curator_vendor.memory_module.store import MemoryStore
from curator_vendor.memory_strategies.curator import CuratorReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_VENDOR_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "curator_vendor", "templates")


# ------------------------------------------------------------------ #
# Curation sampling knobs (env-overridable) — mirror the CURATION_* vars used by      #
# reasoningbank/curator so one env-var set controls curation across methods, mapped   #
# into the vendored CuratorReader `sampling` dict exactly like curator/…/__init__.py.  #
# ------------------------------------------------------------------ #
def _cur_float(name, default=None):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default

def _cur_int(name, default=None):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default

def _build_sampling() -> dict:
    """temperature/top_p top-level; top_k + enable_thinking under extra_body (vLLM conv)."""
    sampling: dict = {}
    temp = _cur_float("CURATION_TEMPERATURE")
    top_p = _cur_float("CURATION_TOP_P")
    top_k = _cur_int("CURATION_TOP_K")
    _cet = os.environ.get("CURATION_ENABLE_THINKING", "")
    enable_thinking = None if _cet == "" else (_cet.lower() in ("1", "true", "yes"))
    if temp is not None:
        sampling["temperature"] = temp
    if top_p is not None:
        sampling["top_p"] = top_p
    extra: dict = {}
    if top_k is not None:
        extra["top_k"] = top_k
    if enable_thinking is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if extra:
        sampling["extra_body"] = extra
    return sampling

_CUR_MAX_TOK = _cur_int("CURATION_MAX_TOKENS", 1024)
_LOG_CALLS = os.environ.get("CURATOR_LOG_CALLS", "1").lower() in ("1", "true", "yes")
_STORE_FAILURES = os.environ.get("STORE_FAILURES", "0").lower() in ("1", "true", "yes")


# ------------------------------------------------------------------ #
# BM25-backed store: vendored MemoryStore with search() replaced by BM25 and         #
# embedding disabled (so add()/search never load torch/SimCSE). Everything else       #
# — schema, JSONL persistence, _format_case — is the vendored code untouched.         #
# ------------------------------------------------------------------ #
class BM25MemoryStore(MemoryStore):
    def __init__(self, path: str, *, top_k: int = 3):
        super().__init__(path, top_k=top_k)
        self._bm25: Optional[BM25Retriever] = None
        self._bm25_n: int = -1  # k the current retriever was built with

    def _embed_entry(self, entry: MemoryEntry) -> None:  # override: BM25 needs no embeddings
        self._bm25 = None  # invalidate; rebuilt lazily on next search

    def _rebuild_embeddings(self) -> None:  # never used (search overridden), keep safe
        self._bm25 = None

    def _build_bm25(self, k: int) -> Optional[BM25Retriever]:
        live = self.entries  # vendored property: non-deleted entries in id order
        if not live:
            return None
        docs = [
            Document(page_content=e.question, metadata={"idx": i})
            for i, e in enumerate(live)
        ]
        return BM25Retriever.from_documents(docs, k=k)

    def search(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """BM25 retrieval returning vendored SearchResult objects (score = BM25 rank-desc proxy).

        Same return type/shape the vendored CuratorReader.read expects, so downstream
        _format_case usage is unchanged.
        """
        k = top_k or self.top_k
        live = self.entries
        if not live:
            return []
        # (re)build retriever if k changed or invalidated
        if self._bm25 is None or self._bm25_n != k:
            self._bm25 = self._build_bm25(k)
            self._bm25_n = k
        if self._bm25 is None:
            return []
        docs = self._bm25.invoke(query)
        results: List[SearchResult] = []
        for rank, doc in enumerate(docs, 1):
            idx = doc.metadata["idx"]
            entry = live[idx]
            # BM25Retriever doesn't expose scores; use a descending rank proxy so higher=better.
            results.append(SearchResult(entry=entry, entry_id=idx, rank=rank,
                                        score=float(len(docs) - rank + 1)))
        return results


class CuratorV1Alfworld:
    """Faithful curator adapter exposing the runner's sync retrieve()/add() seam."""

    def __init__(
        self,
        storage_path: str = "./memory/curator_v1_memory.jsonl",
        curation_model_hf=None,             # accepted for signature-parity; unused (HTTP only)
        curation_tokenizer=None,            # unused
        retrieve_num: int = 3,
        curation_base_url: str = None,
        curation_model_name: str = None,
        curator_on_empty: bool = False,
    ):
        if not curation_base_url:
            raise ValueError(
                "curator_v1 requires --curation_base_url (HTTP/OpenAI-compatible endpoint); "
                "the vendored curator is API-client based (no in-process vLLM path)."
            )
        self.storage_path = storage_path
        self.retrieve_num = retrieve_num
        self.curator_on_empty = curator_on_empty
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        self.calls_log_path = os.path.join(
            os.path.dirname(self.storage_path) or ".", "curator_calls.jsonl"
        )

        # vendored store (BM25 variant) — loads existing MemoryEntry JSONL if present
        self.store = BM25MemoryStore(self.storage_path, top_k=retrieve_num)

        # vendored CuratorReader: exact system prompt + user turn + create() call.
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            base_url=curation_base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
        self._reader = CuratorReader(
            client, curation_model_name,
            max_tokens=_CUR_MAX_TOK,
            template_dir=_VENDOR_TEMPLATE_DIR,
            sampling=_build_sampling(),
        )
        # private event loop + semaphore to drive the async vendored inference synchronously
        self._loop = asyncio.new_event_loop()
        self._sem = asyncio.Semaphore(1)
        logger.info(
            f"CuratorV1Alfworld initialised at {storage_path} "
            f"(store size {len(self.store)}, curator_on_empty={curator_on_empty})"
        )

    # ---- trajectory rendering: identical to curator_alfworld.py for consistency ---- #
    def _trajectory_to_text(self, messages: List[Dict]) -> str:
        parts, step, pending_obs = [], 0, None
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                pending_obs = m["content"]
            else:
                match = re.search(r"<action>(.*?)</action>", m["content"], re.DOTALL)
                action = match.group(1).strip() if match else m["content"]
                parts.append(f"[Step {step}]")
                parts.append(f"[Observation]: {pending_obs}")
                parts.append(f"[Action]: {action}")
                parts.append("")
                step += 1
                pending_obs = None
        return "\n".join(parts)

    def _log_call(self, query: str, retrieved_text: str, briefing: str,
                  retrieved: List[Dict]):
        if not _LOG_CALLS:
            return
        try:
            with open(self.calls_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": query,
                    "retrieved": retrieved,          # [{store_index, score, question}] — traceable
                    "retrieved_text": retrieved_text,
                    "briefing": briefing,
                }) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log curator_v1 call: {e}")

    # ------------------------------------------------------------------ #
    # Public API (matches CuratorAlfworld / the runner seam)              #
    # ------------------------------------------------------------------ #
    def add(self, task_id: str, task: str, messages: List[Dict], reward: bool):
        """Store a trajectory as a vendored MemoryEntry (success-only by default)."""
        if not reward and not _STORE_FAILURES:
            logger.info(f"curator_v1: skip non-success (not stored): {task_id}")
            return
        trajectory = self._trajectory_to_text(messages)
        entry = MemoryEntry(
            question=task,                 # NL task = retrieval key (NOT a file path)
            answer="",                     # ALFWorld has no pred distinct from trajectory
            trajectory=trajectory,
            reward=1.0 if reward else 0.0,
            metadata={"task_id": task_id},
        )
        self.store.add(entry, persist=True)
        logger.info(f"curator_v1: stored {'success' if reward else 'FAILURE'} for {task_id} "
                    f"(store size {len(self.store)})")

    def retrieve(self, query: str, n: int = None, curator_question: str = None) -> str:
        """BM25-retrieve → vendored _format_case → vendored curator inference → briefing.

        query           : BM25 retrieval key (short NL task).
        curator_question : the "Question:" shown to the curator LLM (may be obs0-enriched);
                           defaults to query.
        """
        n = n or self.retrieve_num
        curator_question = curator_question if curator_question is not None else query

        # --- retrieve (BM25) + render with the VENDORED _format_case (verbatim) ---
        retrieved_text = ""
        retrieved_meta: List[Dict] = []
        if len(self.store) > 0:
            results = self.store.search(query, top_k=n)
            if results:
                parts = [MemoryStore._format_case(j, r.entry)
                         for j, r in enumerate(results, 1)]
                retrieved_text = "\n".join(parts)   # exactly as vendored read() joins
                retrieved_meta = [
                    {"store_index": r.entry_id, "score": r.score, "question": r.entry.question}
                    for r in results
                ]

        # curator_on_empty=False => Q2 short-circuit (our extension); True => vendored behavior
        if not retrieved_text and not self.curator_on_empty:
            return ""

        # --- curator inference: the VENDORED _run_curator_inference (exact prompt + call) ---
        briefing = self._loop.run_until_complete(
            self._reader._run_curator_inference(curator_question, retrieved_text, self._sem)
        )
        briefing = briefing or ""
        self._log_call(curator_question, retrieved_text, briefing, retrieved_meta)
        return briefing
