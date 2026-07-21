"""MemCurator (curator_v1) for ALFWorld — API/multi-backend variant.

Same faithful curator_v1 as ``curator_alfworld_v1.py`` (two --curation_mode prompts,
reward-aware store, Result: Success/Failure marking, full-prompt + BM25-provenance logging),
but the read-time LLM call is the MULTI-BACKEND path from ``curator_alfworld_api.py``:
gemini/ -> Vertex AI (google-genai); external gateway -> litellm with max_completion_tokens +
gpt-5 temperature fallback (CURATION_LLM_BACKEND=openai, X-Api-Key); else local vLLM. This is
the curator_v1 twin of curator_alfworld_api.py (as curator_alfworld_v1.py is to curator_alfworld.py).

Port of the "curator" strategy originally in
``sea-mem-policy/curator_react/memory_strategies/curator.py``.

Design (the essential difference vs ReasoningBank):
  * WRITE is trivial — store raw trajectories append-only with their numeric ``reward`` and
    a derived ``status``. No LLM call, no update/delete. ``--curation_mode`` controls what is
    stored: success_only (default) keeps wins only; success_and_fail keeps both (and marks
    each retrieved memory Result: Success/Failure at read time). (ReasoningBank instead runs
    an LLM on every write, success and failure, to distill <=3 items.)
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
  * Every read-time curation call is logged to a sibling ``curator_calls.jsonl`` as
    ``{query, retrieved, retrieved_text, messages, curation_mode, model, briefing}``.
    ``messages`` is the FULL system+user prompt actually sent to the curator LLM (prompt-
    auditable); ``retrieved`` is per-entry provenance ``[{store_index, score, rank, question,
    status}]`` — store_index is the unambiguous row in curator_v1_memory.jsonl (resolves the
    duplicate-query ambiguity) and score is the real BM25 relevance, so retrieval quality is
    fully traceable (see analysis/curator_deep_analysis/CURATOR_DEEP_ANALYSIS.md §5).

Backend selection and the ``CURATION_*`` sampling env knobs are identical to
``reasoningbank_alfworld.py`` on purpose, so curation behaves the same across memory types.
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Optional
# vllm.SamplingParams is imported LAZILY inside the local-vLLM branch of _llm_from_messages,
# so a pure-HTTP / external-gateway / gemini curator can import this module without vllm.
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Vertex (google-genai) does NOT retry transient 5xx / 429 by default (unlike the litellm path's
# num_retries=10). A single 503 UNAVAILABLE would otherwise blank one curation. Retry
# 503/500/429/UNAVAILABLE/DEADLINE_EXCEEDED with exponential backoff, then raise.
_VERTEX_RETRYABLE = ("503", "500", "429", "unavailable", "deadline", "resource_exhausted",
                     "internal error", "overloaded")

def _vertex_generate_content_with_retry(client, _max_tries=10, _base_delay=1.0, _max_delay=30.0, **kwargs):
    """client.models.generate_content(**kwargs) with exponential-backoff retry on transient
    Vertex errors. Raises the last error after _max_tries (so a truly-down call still surfaces)."""
    last = None
    for attempt in range(_max_tries):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as e:
            msg = str(e).lower()
            if attempt == _max_tries - 1 or not any(tok in msg for tok in _VERTEX_RETRYABLE):
                raise
            delay = min(_base_delay * (2 ** attempt), _max_delay)   # 1,2,4,8,16,30,30,... capped
            logger.warning(f"[vertex retry] attempt {attempt+1}/{_max_tries} after transient "
                           f"error: {e} -> sleeping {delay:.0f}s")
            time.sleep(delay)
            last = e
    if last is not None:
        raise last


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

# _api: external-gateway curator switch. When CURATION_LLM_BACKEND=openai the vLLM-only extra_body
# fields (top_k, chat_template_kwargs) are NOT sent, and the X-Api-Key header is attached if set.
_CUR_EXTERNAL = (os.environ.get("CURATION_LLM_BACKEND", "vllm").lower() == "openai")
_X_API_KEY    = os.environ.get("X_API_KEY") or None
# Vertex (gemini/) creds — default to the Salesforce Vertex project if unset, so a gemini
# curator works without extra exports.
_GCLOUD_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")  or "salesforce-research-internal"
_GCLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"


def _completion_with_temp_fallback(**kwargs):
    """litellm.completion, but if the model rejects a non-1 temperature (gpt-5 / gpt-5.x reasoning
    models: 'temperature does not support X; only the default (1) value is supported'), retry once
    WITHOUT temperature (and top_p if it also complains). vLLM/other models never raise this, so
    they are unaffected. Without this a gpt-5.x curator crashes and the memory store stays EMPTY."""
    from litellm import completion
    try:
        return completion(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        retried = dict(kwargs); changed = False
        if "temperature" in msg and ("does not support" in msg or "only the default" in msg or "unsupported value" in msg):
            retried.pop("temperature", None); changed = True
        if "top_p" in msg and ("does not support" in msg or "unsupported value" in msg):
            retried.pop("top_p", None); changed = True
        if not changed:
            raise
        return completion(**retried)

# Whether to log every read-time curation call to curator_calls.jsonl (default on).
_LOG_CALLS = os.environ.get("CURATOR_LOG_CALLS", "1").lower() in ("1", "true", "yes")


# ------------------------------------------------------------------ #
# Two curator system prompts, selected by --curation_mode.                                   #
#                                                                                             #
# success_only (DEFAULT): the store holds ONLY successful trajectories, so the prompt must    #
#   NOT mention failures/pitfalls or an "insights" field we don't have.                       #
#                                                                                             #
# success_and_fail: the store holds both; each retrieved memory carries a "Result:            #
#   Success/Failure" line (see _format_case), so the prompt tells the curator to use both     #
#   (extract what worked, warn about pitfalls). Marking style mirrors skillos's               #
#   "## Result: Success|Failure" (skillos/reasoningbank curate ONE trajectory; curator sees   #
#   MULTIPLE retrieved ones, so the label is per-memory in _format_case).                     #
#                                                                                             #
# The two prompts use CONSISTENT vocabulary ("task", "solve the current task", "THIS task");  #
# they differ ONLY in the success/failure content (opening "successful", memory bullet, the   #
# extra "Result:" bullet, and the "Warn about pitfalls" output point).                        #
#                                                                                             #
# CURATOR_SYSTEM_SUCCESS_ONLY vs curator_alfworld.py's CURATOR_SYSTEM — 3 wording changes:    #
#   (1) dropped the ALFWorld-specific "in a household (ALFWorld) environment" from the opener;#
#   (2) memory bullet trimmed "(observations and actions)" -> "A past task and the trajectory #
#       that solved it";                                                                      #
#   (3) trimmed the parenthetical examples + the closing sentence, i.e. output point 2 is just#
#       "Extract strategies that worked on similar tasks" (no "e.g., where to find objects,   #
#       useful action orders") and the trailing "Focus on generalizable strategies, not       #
#       specific object names or locations." is removed.                                      #
#   Everything else (structure, "solve the current task", 3 output points, "Be concise…") is  #
#   identical to curator_alfworld.py.                                                         #
# ------------------------------------------------------------------ #
CURATOR_SYSTEM_SUCCESS_ONLY = """You are a Memory Curator. You will be given a task that an AI agent needs to solve, along with retrieved past experiences from similar successful tasks.

Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

Each memory contains:
- A past task and the trajectory that solved it

Your output should:
1. Identify which past experiences are most relevant
2. Extract strategies that worked on similar tasks
3. Give specific guidance for THIS task

Be concise — the agent has limited context."""

# ------------------------------------------------------------------ #
# success_only_v1: a REVISED success-only prompt (still success-only — store & curate ONLY   #
# successful trajectories, same write-gating and _format_case as success_only). It exists so  #
# a reworded briefing prompt can be A/B-compared against success_only WITHOUT touching the     #
# original. Selected via --curation_mode success_only_v1.                                      #
#                                                                                             #
# >>> EDIT THE PROMPT BELOW: change the few sentences you want to test. Keep the same          #
#     vocabulary ("task", "solve the current task", "THIS task") and DO NOT add failure/       #
#     pitfall wording (the store holds only wins in this mode). Starts as an exact copy of     #
#     CURATOR_SYSTEM_SUCCESS_ONLY so an unedited v1 behaves identically.                        #
# ------------------------------------------------------------------ #
CURATOR_SYSTEM_SUCCESS_ONLY_V1 = """You are a Memory Curator. You will be given a task that an AI agent needs to solve in a household (ALFWorld) environment, along with retrieved past experiences from similar successful tasks.

Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

Each memory contains:
- A past task and the trajectory that solved it

Your output should:
1. Identify which past experiences are most relevant
2. Extract strategies that worked on similar tasks (e.g., where to find objects, useful action orders)
3. Give specific guidance for THIS task

Be concise — the agent has limited context."""

# CURATOR_SYSTEM = """You are a Memory Curator. You will be given a task that an AI agent needs to solve in a household (ALFWorld) environment, along with retrieved past experiences from similar successful tasks.

# Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

# Each memory contains:
# - A past task and the trajectory (observations and actions) that solved it

# Your output should:
# 1. Identify which past experiences are most relevant
# 2. Extract strategies that worked on similar tasks (e.g., where to find objects, useful action orders)
# 3. Give specific guidance for THIS task

# Be concise — the agent has limited context. Focus on generalizable strategies, not specific object names or locations."""

CURATOR_SYSTEM_SUCCESS_AND_FAIL = """You are a Memory Curator. You will be given a task that an AI agent needs to solve, along with retrieved past experiences from similar tasks.

Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

Each memory contains:
- A past task and the trajectory taken
- Whether it succeeded or failed (shown as "Result: Success" or "Result: Failure")

Your output should:
1. Identify which past experiences are most relevant
2. Extract strategies that worked on similar tasks
3. Warn about pitfalls from failed attempts
4. Give specific guidance for THIS task

Be concise — the agent has limited context."""



CURATOR_SYSTEM_SUCCESS_AND_FAIL_V1 = """You are a Memory Curator. You will be given a task that an AI agent needs to solve in a household (ALFWorld) environment, along with retrieved past experiences from similar tasks.

Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

Each memory contains:
- A past task and the trajectory taken
- Whether it succeeded or failed (shown as "Result: Success" or "Result: Failure")

Your output should:
1. Identify which past experiences are most relevant
2. Extract strategies that worked on similar tasks (e.g., where to find objects, useful action orders)
3. Warn about pitfalls from failed attempts
4. Give specific guidance for THIS task

Be concise — the agent has limited context."""
# --- curation-mode registry -------------------------------------------------------------- #
# Each --curation_mode is defined by THREE facts, kept in one place so adding a mode is a
# single-row change and no scattered ``== "success_and_fail"`` string-test can drift:
#   * which system prompt build_curator_messages uses;
#   * whether add() STORES failed trajectories (the "_and_fail" family) or only wins;
#   * whether _format_case MARKS each retrieved memory with a Result: Success/Failure line
#     (same "_and_fail" family — a success-only store has nothing to mark).
# The ``*_v1`` modes are prompt-only A/B variants: SAME store/mark semantics as their base
# mode, different system prompt.
_MODE_SYSTEM_PROMPT = {
    "success_only":        CURATOR_SYSTEM_SUCCESS_ONLY,
    "success_only_v1":     CURATOR_SYSTEM_SUCCESS_ONLY_V1,
    "success_and_fail":    CURATOR_SYSTEM_SUCCESS_AND_FAIL,
    "success_and_fail_v1": CURATOR_SYSTEM_SUCCESS_AND_FAIL_V1,
}
# Modes whose store keeps failures AND whose _format_case emits the Result: line.
_STORE_FAILURES_MODES = {"success_and_fail", "success_and_fail_v1"}
_MARK_RESULTS_MODES   = _STORE_FAILURES_MODES

# valid --curation_mode values (derived from the registry — never drifts from it)
CURATION_MODES = tuple(_MODE_SYSTEM_PROMPT.keys())


def build_curator_messages(query: str, retrieved_text: str,
                           curation_mode: str = "success_only") -> List[Dict[str, str]]:
    """Isolated, shared prompt-builder for the read-time curation call.

    Kept module-level (not a method) so a future RL rollout generator can import the
    *identical* prompt construction used at eval time — the training-awareness contract.

    Mirrors the original ``CuratorReader._run_curator_inference`` message layout; the system
    prompt is chosen by ``curation_mode`` via the _MODE_SYSTEM_PROMPT registry (falls back to
    the success_only prompt for an unknown mode).
    """
    system = _MODE_SYSTEM_PROMPT.get(curation_mode, CURATOR_SYSTEM_SUCCESS_ONLY)
    if retrieved_text:
        user_content = f"Question: {query}\n\nRetrieved Memories:\n{retrieved_text}"
    else:
        user_content = f"Question: {query}\n\nNo past memories available yet."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def _record_success(record: Dict) -> bool:
    """Whether a stored memory record is a success. Source of truth is the numeric
    ``reward`` (>0) when present; falls back to the ``status`` string for older records
    written before the reward field existed."""
    if "reward" in record and record["reward"] is not None:
        return float(record["reward"]) > 0
    return record.get("status", "success") == "success"


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


# --- with_thinking trajectory support ---------------------------------------------------- #
# The executor's chain-of-thought (CoT) is NOT in the stored `messages` (those hold only the
# extracted <action>); it lives in the raw per-step model response. The `with_thinking`
# trajectory style pulls the CoT from those raw responses so the curator can see HOW a step was
# reasoned, not just WHAT was done. `action_only` (default) is byte-identical to the old behavior.
_TRAJECTORY_STYLES = ("action_only", "with_thinking")
_CHARS_PER_TOKEN = 4   # rough token->char factor for the per-step budget (tokenizer-free)

def _extract_thinking(response: str) -> str:
    """Pull the reasoning that preceded the action out of a raw executor response.

    Uniform rule across model types: thinking = everything BEFORE the last <action> tag, with
    <think>/<reason> tags stripped. Handles thinking-Qwen (<think>...</think>...<action>),
    reason_tag (<reason>...</reason>), and gpt-style (rationale\\n\\n<action>) identically."""
    if not response:
        return ""
    head = response.rsplit("<action>", 1)[0] if "<action>" in response else response
    head = re.sub(r"</?think>", "", head)
    head = re.sub(r"</?reason>", "", head)
    return head.strip()

def _truncate_head_tail(text: str, cap: int) -> str:
    """Keep the first cap/2 + last cap/2 chars (the gist is front-loaded; the tail often holds
    the concluded decision). cap<=0 or short-enough text => returned unchanged."""
    if cap <= 0 or len(text) <= cap:
        return text
    half = cap // 2
    return text[:half] + "\n…[thinking truncated]…\n" + text[-half:]


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
        curation_mode: str = "success_only",
        trajectory_style: str = "action_only",
        think_token_budget: int = 8000,
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
        # success_only (default): store & curate only wins; success_and_fail: store both and
        # tag each retrieved memory with Result: Success/Failure in the curator prompt.
        # (*_v1 = prompt-only A/B variant of the same store/mark semantics.)
        assert curation_mode in CURATION_MODES, f"curation_mode must be one of {CURATION_MODES}"
        self.curation_mode = curation_mode
        # Behavior flags derived ONCE from the registry, so add()/_format_case never string-test
        # the mode (adding a mode can't silently break failure-storage or Result: marking).
        self._store_failures = curation_mode in _STORE_FAILURES_MODES
        self._mark_results   = curation_mode in _MARK_RESULTS_MODES

        # Trajectory rendering. action_only (default) = the original [Observation]/[Action] text.
        # with_thinking additionally injects a per-step [Thinking] line, sourced from the raw
        # executor responses passed to add() (NOT from `messages`, which never held the CoT).
        assert trajectory_style in _TRAJECTORY_STYLES, \
            f"trajectory_style must be one of {_TRAJECTORY_STYLES}"
        self.trajectory_style = trajectory_style
        # Total per-trajectory thinking budget (in ~tokens). The per-STEP char cap is derived at
        # write time as (budget / num_steps) * chars_per_token, then head/tail-truncated. <=0 =
        # unbounded (keep full thinking per step).
        self.think_token_budget = think_token_budget

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
        """Run the curation LLM on a prebuilt chat-message list — MULTI-BACKEND (api):
        gemini/ -> Vertex AI; external gateway -> litellm (max_completion_tokens + gpt-5 temp
        fallback, X-Api-Key); else local vLLM. Identical backend path to curator_alfworld_api.py.
        """
        # gemini/ curator -> Vertex AI. Curator curation is plain text-in/text-out.
        if self.curation_model_name and self.curation_model_name.startswith("gemini/"):
            from google import genai
            from google.genai import types
            model_id = self.curation_model_name[len("gemini/"):]
            system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
            user_text  = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
            client = genai.Client(vertexai=True, project=_GCLOUD_PROJECT, location=_GCLOUD_LOCATION)
            resp = _vertex_generate_content_with_retry(
                client,
                model=model_id,
                contents=user_text,
                config=types.GenerateContentConfig(
                    temperature=_CUR_TEMP,
                    system_instruction=system_msg,
                    max_output_tokens=_CUR_MAX_TOK,
                ),
            )
            return resp.text or ""
        if self.curation_base_url is not None:
            kwargs = dict(
                model=self.curation_model_name,
                messages=messages,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.curation_base_url,
                temperature=_CUR_TEMP,
                num_retries=10,
            )
            # gpt-5 family rejects `max_tokens` -> use `max_completion_tokens` when external.
            kwargs["max_completion_tokens" if _CUR_EXTERNAL else "max_tokens"] = _CUR_MAX_TOK
            if _CUR_TOP_P is not None:
                kwargs["top_p"] = _CUR_TOP_P
            # extra_body (top_k, chat_template_kwargs) is vLLM-only; skip for external gateway.
            extra_body = {}
            if not _CUR_EXTERNAL:
                if _CUR_TOP_K is not None:
                    extra_body["top_k"] = _CUR_TOP_K
                if _CUR_THINKING is not None:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": _CUR_THINKING}
            if extra_body:
                kwargs["extra_body"] = extra_body
            if _CUR_EXTERNAL and _X_API_KEY:
                kwargs["extra_headers"] = {"X-Api-Key": _X_API_KEY}
            resp = _completion_with_temp_fallback(**kwargs)
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
        # Local vLLM path — lazy import so external/gemini curators don't need vllm installed.
        from vllm import SamplingParams
        sp_kwargs = dict(temperature=_CUR_TEMP, max_tokens=_CUR_MAX_TOK)
        if _CUR_TOP_P is not None:
            sp_kwargs["top_p"] = _CUR_TOP_P
        if _CUR_TOP_K is not None:
            sp_kwargs["top_k"] = _CUR_TOP_K
        sampling_params = SamplingParams(**sp_kwargs)
        outputs = self.curation_model_hf.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _trajectory_to_text(self, messages: List[Dict]) -> str:
        """Convert ALFWorld message list to a readable trajectory string (action_only)."""
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

    def _trajectory_to_text_with_thinking(self, messages: List[Dict],
                                          step_responses: List[str]) -> str:
        """Like _trajectory_to_text, but adds a per-step [Thinking] line sourced from the raw
        executor responses (step_responses[k] is the full model output for step k). The CoT is
        NOT in `messages` (those hold only the extracted action), so step_responses is required.

        Length control: the total think budget (think_token_budget ~tokens) is split evenly
        across the executor steps, giving a per-step char cap = (budget/steps)*_CHARS_PER_TOKEN;
        each step's thinking is head/tail-truncated to that cap (gist front-loaded, decision at
        the tail). Falls back to action_only formatting for any step with no captured response."""
        # steps = number of (obs, action) turns == number of assistant messages.
        n_steps = sum(1 for m in messages if m["role"] == "assistant") or 1
        if self.think_token_budget and self.think_token_budget > 0:
            per_step_cap = max(1, int(self.think_token_budget / n_steps) * _CHARS_PER_TOKEN)
        else:
            per_step_cap = 0   # unbounded

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
                raw = step_responses[step] if step < len(step_responses) else ""
                thinking = _truncate_head_tail(_extract_thinking(raw), per_step_cap)
                parts.append(f"[Step {step}]")
                parts.append(f"[Observation]: {pending_obs}")
                if thinking:
                    parts.append(f"[Thinking]: {thinking}")
                parts.append(f"[Action]: {action}")
                parts.append("")
                step += 1
                pending_obs = None
        return "\n".join(parts)

    def _format_case(self, idx: int, record: Dict) -> str:
        """Render a retrieved record as a numbered case for the curator's context.

        In success_and_fail mode each memory carries a ``Result: Success/Failure`` line
        (skillos-style) so the curator can tell wins from failures. In success_only mode
        (default) the store holds only wins, so the label is omitted.
        """
        lines = [f"Memory {idx}:"]
        if self._mark_results:
            lines.append(f"Result: {'Success' if _record_success(record) else 'Failure'}")
        lines.append(f"Question: {record.get('query', '')}")
        lines.append(f"Trajectory:\n{record.get('trajectory', '')}")
        return "\n".join(lines)

    @staticmethod
    def _bm25_scores(retriever, query: str):
        """Real BM25 relevance scores over ALL docs (index-aligned to store `idx`), via the
        rank_bm25 backend the langchain BM25Retriever wraps. Returns a list or None if the
        backend/API differs (logging must never break retrieval)."""
        try:
            pre = getattr(retriever, "preprocess_func", None)
            toks = pre(query) if pre is not None else query.split()
            return list(retriever.vectorizer.get_scores(toks))
        except Exception as e:
            logger.warning(f"Could not compute BM25 scores (logging only): {e}")
            return None

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

    def _log_call(self, query: str, retrieved_text: str, briefing: str,
                  messages: List[Dict[str, str]] = None, retrieved: List[Dict] = None,
                  briefing_raw: str = None):
        if not _LOG_CALLS:
            return
        try:
            rec = {
                "query": query,
                # Per-retrieved-entry provenance: store_index (row in curator_v1_memory.jsonl —
                # UNAMBIGUOUS even when query strings duplicate), BM25 score, rank, question,
                # status. Unblocks retrieval-quality/causal analysis (see CURATOR_DEEP_ANALYSIS).
                "retrieved": retrieved if retrieved is not None else [],
                "retrieved_text": retrieved_text,
                # The FULL prompt actually sent to the curator LLM (system + user), so each
                # run is self-documenting / prompt-auditable. Falls back to rebuilding it if
                # not passed in.
                "messages": messages if messages is not None else build_curator_messages(
                    query, retrieved_text, curation_mode=self.curation_mode),
                "curation_mode": self.curation_mode,
                "model": self.curation_model_name,
                # briefing = what's INJECTED (post _strip_think). briefing_raw = the RAW curator
                # LLM output BEFORE strip — lets you verify CURATION_ENABLE_THINKING (a <think>
                # block present in raw but not in briefing => the curator was thinking).
                "briefing": briefing,
                "briefing_raw": briefing_raw if briefing_raw is not None else briefing,
            }
            with open(self.calls_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # logging must never break a run
            logger.warning(f"Failed to log curator call: {e}")

    # ------------------------------------------------------------------ #
    # Public API (matches ReasoningBankAlfworld so the runner wiring is a copy) #
    # ------------------------------------------------------------------ #

    def add(self, task_id: str, task: str, messages: List[Dict], reward,
            step_responses: List[str] = None):
        """Store a raw trajectory (curator write — no LLM, append-only).

        success_only (default): store wins only (skip failures), matching the original
        curator's ``WriteStrategy.should_write``. success_and_fail: store both, tagging
        ``reward``/``status`` so the curator prompt can distinguish them at read time.

        Args:
            task_id: unique game identifier.
            task: the task description string (the retrieval key).
            messages: full conversation message list from the episode.
            reward: the episode reward. Accepts float (e.g. 1.0/0.0, or a graded score) or
                bool; stored as a float. success = reward > 0.
            step_responses: OPTIONAL list of the raw per-step executor responses (the CoT lives
                here, NOT in `messages`). Only used when trajectory_style == 'with_thinking';
                ignored otherwise. If the style is with_thinking but this is missing, we fall
                back to the action_only rendering (never crash a run).
        """
        # Normalize to float so we keep the real numeric reward (future-proof for graded
        # rewards); success is reward > 0 (matches sea-mem CuratorWriter / MemoryEntry).
        reward = float(reward)
        success = reward > 0

        # Store failures ONLY in the _and_fail family (self._store_failures); every other mode
        # (success_only, success_only_v1, …) is success-only, so skip failed episodes.
        if not success and not self._store_failures:
            logger.info(f"Curator: skipping failed trajectory (not stored): {task_id}")
            return

        if self.trajectory_style == "with_thinking" and step_responses:
            trajectory = self._trajectory_to_text_with_thinking(messages, step_responses)
        else:
            trajectory = self._trajectory_to_text(messages)
        record = {
            "task_id": task_id,
            "query": task,
            "trajectory": trajectory,
            "reward": reward,                              # numeric reward (source of truth)
            "status": "success" if success else "fail",    # derived, for the Result: line
        }
        self._save_record(record)
        self.memory_bank.append(record)
        self._rebuild_bm25()
        logger.info(f"Curator: stored {record['status']} (reward={reward}) trajectory for: {task_id}")

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
        retrieved_meta: List[Dict] = []   # per-entry {store_index, score, rank, question, status}
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
            # Surface the real BM25 scores for the retrieved docs. rank_bm25 (the backend)
            # scores ALL docs for the query; index by each retrieved doc's store `idx`.
            bm25_scores = self._bm25_scores(retriever, query)
            parts = []
            for j, doc in enumerate(docs or [], 1):
                idx = doc.metadata["idx"]
                record = self.memory_bank[idx]
                parts.append(self._format_case(j, record))
                retrieved_meta.append({
                    "store_index": idx,                       # row in curator_v1_memory.jsonl (unambiguous)
                    "score": (round(float(bm25_scores[idx]), 6)
                              if bm25_scores is not None and idx < len(bm25_scores) else None),
                    "rank": j,                                # 1-based BM25 rank in this retrieval
                    # `key` = the EXACT text BM25 indexed/matched on (Document.page_content).
                    # Today this equals `question` (the NL task), but logging page_content
                    # directly keeps the log correct if the retrieval key ever changes
                    # (e.g. a skill-description / summarized key) — no re-derivation.
                    "key": doc.page_content,
                    "question": record.get("query", ""),
                    "status": record.get("status", "success"),
                })
            retrieved_text = "\n\n".join(parts)

        # Nothing retrieved: by default (Q2) do NOT call the curator LLM — return "" so no
        # evidence-free briefing is injected. With curator_on_empty, still call the LLM
        # (build_curator_messages emits a "No past memories available yet." user turn).
        if not retrieved_text and not self.curator_on_empty:
            return ""

        # curator_question (may be enriched) becomes "Question: {..}" for the CURRENT task;
        # retrieved_text still holds the short past-task Questions from _format_case.
        messages = build_curator_messages(curator_question, retrieved_text,
                                          curation_mode=self.curation_mode)
        briefing_raw = self._llm_from_messages(messages)   # RAW LLM output (may contain <think>)
        briefing = _strip_think(briefing_raw)              # what actually gets injected
        self._log_call(curator_question, retrieved_text, briefing,
                       messages=messages, retrieved=retrieved_meta, briefing_raw=briefing_raw)
        return briefing if briefing else ""
