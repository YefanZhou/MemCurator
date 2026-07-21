"""MemCurator memory method for ALFWorld — a read/write memory strategy.

This is the SkillCurator-repo port of the "curator" strategy originally implemented in
``sea-mem-policy/curator_react/memory_strategies/curator.py`` and adapted into verl-agent
via ``agent_system/memory_eval/backends/jsonl_dense.py``.

Design (the essential difference vs ReasoningBank):
  * WRITE is trivial — store *raw successful* trajectories only, append-only. No LLM call,
    no update/delete. (ReasoningBank runs an LLM on every write, success and failure, to
    distill <=3 items.)
  * READ is the smart part — BM25-retrieve the top-k most similar past trajectories, then
    call a "Memory Curator" LLM to synthesize them into one concise, actionable briefing,
    and return that briefing for injection. (ReasoningBank does NO LLM on read; it just
    concatenates stored items.)

Preserved gotchas from the original dev logs:
  * Q2 — empty store => return "" with NO LLM call (no evidence-free hallucinated briefing).
  * Q3 — the retrieval / curator key must be the natural-language task description, not a
    file path. The runner passes ``task_descriptions[i]`` so this holds by construction.

Training-awareness (mirrors how SkillOS's curator is made GRPO-trainable):
  * ``build_curator_messages`` is a module-level, isolated prompt-builder so a future RL
    rollout generator can import the *identical* prompt construction used here at eval time.
  * ``_llm_from_messages`` shares the exact tokenizer / apply_chat_template / backend path
    used by the other methods, and the curation checkpoint is swappable via
    ``--curation_model`` / ``--curation_base_url`` (see ``init_memory`` in the runner).
  * ``_strip_think`` is a named parser hook where a later format/reward check can attach.
  * Every read-time curation call is logged as ``{query, retrieved_text, briefing}`` to a
    sibling ``curator_calls.jsonl`` so the read-time policy is harvestable for training.

Backend selection and the ``CURATION_*`` sampling env knobs are identical to
``reasoningbank_alfworld.py`` on purpose, so curation behaves the same across memory types.
"""

import os
import re
import json
import logging
from typing import Dict, List, Optional
from vllm import SamplingParams
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Curation sampling knobs (env-overridable; defaults = prior behaviour) #
# Mirror the CURATION_* vars used by run_unified_dev.py / reasoningbank  #
# so one set of env vars controls curation sampling across memory types. #
# ------------------------------------------------------------------ #
def _cur_float(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default

def _cur_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default

_CUR_TEMP     = _cur_float("CURATION_TEMPERATURE", 0.7)
_CUR_TOP_P    = _cur_float("CURATION_TOP_P", None)
_CUR_TOP_K    = _cur_int("CURATION_TOP_K", None)
_CUR_MAX_TOK  = _cur_int("CURATION_MAX_TOKENS", 1024)   # prior default was 1024
_CET          = os.environ.get("CURATION_ENABLE_THINKING", "")
_CUR_THINKING = None if _CET == "" else (_CET.lower() in ("1", "true", "yes"))

# Whether to log every read-time curation call to curator_calls.jsonl (default on).
_LOG_CALLS = os.environ.get("CURATOR_LOG_CALLS", "1").lower() in ("1", "true", "yes")


# Ported verbatim from sea-mem-policy/curator_react/templates/curator_system.txt, with a
# one-line note that the memories here are successful ALFWorld household trajectories.
CURATOR_SYSTEM = """You are a Memory Curator. You will be given a task that an AI agent needs to solve in a household (ALFWorld) environment, along with retrieved past experiences from similar successful tasks.

Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

Each memory contains:
- A past task and the trajectory (observations and actions) that solved it

Your output should:
1. Identify which past experiences are most relevant
2. Extract strategies that worked on similar tasks (e.g., where to find objects, useful action orders)
3. Give specific guidance for THIS task

Be concise — the agent has limited context. Focus on generalizable strategies, not specific object names or locations."""


def build_curator_messages(query: str, retrieved_text: str) -> List[Dict[str, str]]:
    """Isolated, shared prompt-builder for the read-time curation call.

    Kept module-level (not a method) so a future RL rollout generator can import the
    *identical* prompt construction used at eval time — the training-awareness contract.

    Mirrors the original ``CuratorReader._run_curator_inference`` message layout.
    """
    if retrieved_text:
        user_content = f"Question: {query}\n\nRetrieved Memories:\n{retrieved_text}"
    else:
        user_content = f"Question: {query}\n\nNo past memories available yet."
    return [
        {"role": "system", "content": CURATOR_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _strip_think(text: str) -> str:
    """Named parser hook: drop <think>...</think> reasoning so it never pollutes memory.

    A later format / reward check can attach here (training-awareness contract #3).
    """
    if not text:
        return ""
    # If the model emitted a closed think block, keep only what follows it.
    text = text.split("</think>")[-1]
    # Drop any stray/unclosed <think> tags.
    text = re.sub(r"</?think>", "", text)
    return text.strip()


class CuratorAlfworld:
    def __init__(
        self,
        storage_path: str = "./memory/curator_memory.jsonl",
        curation_model_hf=None,
        curation_tokenizer=None,
        retrieve_num: int = 3,
        curation_base_url: str = None,
        curation_model_name: str = None,
        curator_on_empty: bool = False,
    ):
        self.storage_path = storage_path
        self.curation_model_hf = curation_model_hf
        self.curation_tokenizer = curation_tokenizer
        self.retrieve_num = retrieve_num
        self.curation_base_url = curation_base_url
        self.curation_model_name = curation_model_name
        # When True, call the curator LLM to synthesize a briefing even when retrieval
        # returns nothing (empty/cold store). Default False = Q2 behavior (no call, "").
        self.curator_on_empty = curator_on_empty

        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        # Read-time curation transcripts land next to the memory store.
        self.calls_log_path = os.path.join(
            os.path.dirname(self.storage_path), "curator_calls.jsonl"
        )

        self.memory_bank: List[Dict] = self._load_jsonl(self.storage_path)
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._rebuild_bm25()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _load_jsonl(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def _llm_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Run the curation LLM on a prebuilt chat-message list.

        Thin variant of reasoningbank_alfworld's ``_llm`` that takes prebuilt messages
        (from ``build_curator_messages``) instead of ``(system, user)`` strings, so the
        tokenizer / apply_chat_template / backend selection path is shared and unchanged.
        """
        if self.curation_base_url is not None:
            from litellm import completion
            kwargs = dict(
                model=self.curation_model_name,
                messages=messages,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.curation_base_url,
                temperature=_CUR_TEMP,
                max_tokens=_CUR_MAX_TOK,
                num_retries=10,
            )
            if _CUR_TOP_P is not None:
                kwargs["top_p"] = _CUR_TOP_P
            extra_body = {}
            if _CUR_TOP_K is not None:
                extra_body["top_k"] = _CUR_TOP_K
            if _CUR_THINKING is not None:
                extra_body["chat_template_kwargs"] = {"enable_thinking": _CUR_THINKING}
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = completion(**kwargs)
            return resp.choices[0].message.content or ""
        if self.curation_tokenizer is not None:
            enable_thinking = False if _CUR_THINKING is None else _CUR_THINKING
            prompt = self.curation_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            # No tokenizer/backend: flatten to plain text (used only in tests/stubs).
            prompt = "\n\n".join(m["content"] for m in messages)
        sp_kwargs = dict(temperature=_CUR_TEMP, max_tokens=_CUR_MAX_TOK)
        if _CUR_TOP_P is not None:
            sp_kwargs["top_p"] = _CUR_TOP_P
        if _CUR_TOP_K is not None:
            sp_kwargs["top_k"] = _CUR_TOP_K
        sampling_params = SamplingParams(**sp_kwargs)
        outputs = self.curation_model_hf.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _trajectory_to_text(self, messages: List[Dict]) -> str:
        """Convert ALFWorld message list to a readable trajectory string."""
        parts = []
        step = 0
        pending_obs = None
        for m in messages:
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                pending_obs = m["content"]
            else:  # assistant
                match = re.search(r"<action>(.*?)</action>", m["content"], re.DOTALL)
                action = match.group(1).strip() if match else m["content"]
                parts.append(f"[Step {step}]")
                parts.append(f"[Observation]: {pending_obs}")
                parts.append(f"[Action]: {action}")
                parts.append("")
                step += 1
                pending_obs = None
        return "\n".join(parts)

    def _format_case(self, idx: int, record: Dict) -> str:
        """Render a retrieved record as a numbered case for the curator's context.

        Adapted from ``MemoryStore._format_case`` in the original repo; our records only
        store successes, so no reward/insight fields are needed.
        """
        return (
            f"Memory {idx}:\n"
            f"Question: {record.get('query', '')}\n"
            f"Trajectory:\n{record.get('trajectory', '')}"
        )

    def _rebuild_bm25(self):
        if not self.memory_bank:
            self.bm25_retriever = None
            return
        docs = [
            Document(
                page_content=record["query"],
                metadata={"task_id": record.get("task_id", ""), "idx": i},
            )
            for i, record in enumerate(self.memory_bank)
        ]
        self.bm25_retriever = BM25Retriever.from_documents(docs, k=self.retrieve_num)

    def _save_record(self, record: Dict):
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _log_call(self, query: str, retrieved_text: str, briefing: str):
        if not _LOG_CALLS:
            return
        try:
            with open(self.calls_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": query,
                    "retrieved_text": retrieved_text,
                    "briefing": briefing,
                }) + "\n")
        except Exception as e:  # logging must never break a run
            logger.warning(f"Failed to log curator call: {e}")

    # ------------------------------------------------------------------ #
    # Public API (matches ReasoningBankAlfworld so the runner wiring is a copy) #
    # ------------------------------------------------------------------ #

    def add(self, task_id: str, task: str, messages: List[Dict], reward: bool):
        """Store a raw *successful* trajectory (curator write — no LLM, append-only).

        Failures are not stored (write-on-success only), matching the original curator's
        ``WriteStrategy.should_write``.

        Args:
            task_id: unique game identifier.
            task: the task description string (the retrieval key).
            messages: full conversation message list from the episode.
            reward: True if the agent won, False otherwise.
        """
        if not reward:
            logger.info(f"Curator: skipping failed trajectory (not stored): {task_id}")
            return

        trajectory = self._trajectory_to_text(messages)
        record = {
            "task_id": task_id,
            "query": task,
            "trajectory": trajectory,
            "status": "success",
        }
        self._save_record(record)
        self.memory_bank.append(record)
        self._rebuild_bm25()
        logger.info(f"Curator: stored success trajectory for: {task_id}")

    def retrieve(self, query: str, n: int = None, curator_question: str = None) -> str:
        """Retrieve similar past trajectories and curate them into a briefing (curator read).

        Args:
            query: the BM25 retrieval key (the short NL task; also what was stored).
            n: number of records to retrieve (defaults to self.retrieve_num).
            curator_question: the text shown to the curator LLM as ``Question: {..}`` in the
                user turn. Defaults to ``query``. Passing a richer string (e.g. task +
                step-0 room observation, via --task_context obs0) enriches ONLY what the
                curator sees for the CURRENT task — it does NOT change the BM25 key or the
                stored records (so retrieval and the past-task ``Question:`` stay short).

        Returns the synthesized briefing string, or "" if the store is empty (Q2: no LLM
        call in that case). The framing header is applied by the runner template's
        ``{context_header}`` slot, so this returns only the synthesized text.
        """
        curator_question = curator_question if curator_question is not None else query
        retrieved_text = ""
        if self.bm25_retriever is not None:
            if n is not None and n != self.retrieve_num:
                docs_all = [
                    Document(
                        page_content=record["query"],
                        metadata={"task_id": record.get("task_id", ""), "idx": i},
                    )
                    for i, record in enumerate(self.memory_bank)
                ]
                retriever = BM25Retriever.from_documents(docs_all, k=n)
            else:
                retriever = self.bm25_retriever

            docs = retriever.invoke(query)   # BM25 keys on the SHORT task, always
            parts = []
            for j, doc in enumerate(docs or [], 1):
                idx = doc.metadata["idx"]
                record = self.memory_bank[idx]
                parts.append(self._format_case(j, record))
            retrieved_text = "\n\n".join(parts)

        # Nothing retrieved: by default (Q2) do NOT call the curator LLM — return "" so no
        # evidence-free briefing is injected. With curator_on_empty, still call the LLM
        # (build_curator_messages emits a "No past memories available yet." user turn).
        if not retrieved_text and not self.curator_on_empty:
            return ""

        # curator_question (may be enriched) becomes "Question: {..}" for the CURRENT task;
        # retrieved_text still holds the short past-task Questions from _format_case.
        messages = build_curator_messages(curator_question, retrieved_text)
        briefing = _strip_think(self._llm_from_messages(messages))
        self._log_call(curator_question, retrieved_text, briefing)
        return briefing if briefing else ""
