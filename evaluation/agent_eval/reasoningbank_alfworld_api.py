import os
import re
import json
import logging
from typing import Dict, List, Optional
# NOTE (_api): `from vllm import SamplingParams` is NOT imported at module top here — it is
# imported lazily inside the local-vLLM branch of _llm(). This lets a pure-HTTP / external-gateway
# curator import this module on a box without vllm installed.
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Curation sampling knobs (env-overridable; defaults = prior behaviour) #
# Mirror the CURATION_* vars used by run_unified_dev.py so one set of   #
# env vars controls curation sampling across memory types.              #
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

# Vertex (gemini/) creds — default to the Salesforce Vertex project (matches tests/test_api_model.py)
# if the env vars are unset, so a gemini reasoningbank curator works without extra exports.
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


SUCCESSFUL_SI = """You are an expert in household task planning. You will be given a task and a trajectory representing how an agent successfully completed the task in a household environment.

## Guidelines
Extract and summarize useful insights as memory items that would help an agent solve similar household tasks in the future.

## Important notes
  - Think about why the trajectory succeeded, then summarize the insights.
  - Extract *at most 3* memory items.
  - Do not repeat similar or overlapping items.
  - Focus on generalizable strategies (e.g., where to find objects, what order to do actions), not specific object names or locations.

## Output Format
Your output must strictly follow this Markdown format:

```
# Memory Item i
## Title <short title>
## Description <one sentence summary>
## Content <1-3 sentences of actionable insight>
```
"""

FAILED_SI = """You are an expert in household task planning. You will be given a task and a trajectory representing how an agent attempted but failed to complete a household task.

## Guidelines
Extract lessons learned as memory items to help avoid the same mistakes in future similar tasks.

## Important notes
  - Reflect on why the trajectory failed, then summarize what to do differently.
  - Extract *at most 3* memory items.
  - Do not repeat similar or overlapping items.
  - Focus on generalizable strategies, not specific object names or locations.

## Output Format
Your output must strictly follow this Markdown format:

```
# Memory Item i
## Title <short title>
## Description <one sentence summary>
## Content <1-3 sentences of actionable insight>
```
"""


class ReasoningBankAlfworld:
    def __init__(
        self,
        storage_path: str = "./memory/reasoning_bank.jsonl",
        curation_model_hf=None,
        curation_tokenizer=None,
        retrieve_num: int = 3,
        curation_base_url: str = None,
        curation_model_name: str = None,
    ):
        self.storage_path = storage_path
        self.curation_model_hf = curation_model_hf
        self.curation_tokenizer = curation_tokenizer
        self.retrieve_num = retrieve_num
        self.curation_base_url = curation_base_url
        self.curation_model_name = curation_model_name

        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

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

    def _llm(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # gemini/ curator -> Vertex AI. ReasoningBank curation is plain text-in/text-out (the
        # markdown memory items are parsed downstream), so no function-calling is needed here.
        if self.curation_model_name and self.curation_model_name.startswith("gemini/"):
            from google import genai
            from google.genai import types
            model_id = self.curation_model_name[len("gemini/"):]
            client = genai.Client(vertexai=True, project=_GCLOUD_PROJECT, location=_GCLOUD_LOCATION)
            resp = client.models.generate_content(
                model=model_id,
                contents=user,
                config=types.GenerateContentConfig(
                    temperature=_CUR_TEMP,
                    system_instruction=system,
                    max_output_tokens=_CUR_MAX_TOK,
                ),
            )
            return resp.text or ""
        if self.curation_base_url is not None:
            # Was hardcoded: temperature=0.7, max_tokens=1024 (no top_p/top_k/thinking).
            # Now env-driven via CURATION_* knobs (max_tokens default kept at 1024).
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
            # Local tokenizer path: honour CURATION_ENABLE_THINKING if set, else keep
            # the prior default (was hardcoded: enable_thinking=False).
            enable_thinking = False if _CUR_THINKING is None else _CUR_THINKING
            prompt = self.curation_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            prompt = f"{system}\n\n{user}"
        # Local vLLM path — import SamplingParams lazily here (not at module top) so a pure-HTTP
        # / external-gateway curator can import this module without vllm installed.
        from vllm import SamplingParams
        # Was hardcoded: SamplingParams(temperature=0.7, max_tokens=1024).
        # Now env-driven via CURATION_* knobs (max_tokens default kept at 1024).
        sp_kwargs = dict(temperature=_CUR_TEMP, max_tokens=_CUR_MAX_TOK)
        if _CUR_TOP_P is not None:
            sp_kwargs["top_p"] = _CUR_TOP_P
        if _CUR_TOP_K is not None:
            sp_kwargs["top_k"] = _CUR_TOP_K
        sampling_params = SamplingParams(**sp_kwargs)
        outputs = self.curation_model_hf.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _extract_memory_items(self, task: str, trajectory: str, success: bool) -> List[str]:
        si = SUCCESSFUL_SI if success else FAILED_SI
        prompt = f"**Task:** {task}\n\n**Trajectory:**\n{trajectory}"
        raw = self._llm(si, prompt)
        # Split by Memory Item blocks
        raw = raw.split("</think>")[-1]  # In case the model adds <think> tags, ignore them
        # The curator often wraps its output in a markdown code block (```), matching the
        # format example in its own system prompt. Strip fence-only lines so the opening
        # ``` doesn't become a phantom item and closing ``` don't trail into real items.
        raw = re.sub(r"^\s*```[a-zA-Z]*\s*$", "", raw, flags=re.MULTILINE)
        items = [
            block.strip().rstrip("`").strip()
            for block in raw.split("# Memory Item")
            if block.strip().strip("`").strip()
        ]
        return ["# Memory Item " + item for item in items]

    def _trajectory_to_text(self, messages: List[Dict]) -> str:
        """Convert ALFWorld message list to readable trajectory string."""
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

    def _rebuild_bm25(self):
        if not self.memory_bank:
            self.bm25_retriever = None
            return
        docs = [
            Document(
                page_content=record["query"],
                metadata={"task_id": record["task_id"], "idx": i},
            )
            for i, record in enumerate(self.memory_bank)
        ]
        self.bm25_retriever = BM25Retriever.from_documents(docs, k=self.retrieve_num)

    def _save_record(self, record: Dict):
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(self, task_id: str, task: str, messages: List[Dict], reward: bool):
        """
        Extract memory items from a completed ALFWorld trajectory and store them.

        Args:
            task_id: unique game identifier (e.g. 'pick_and_place-Apple-Fridge-42')
            task: the task description string
            messages: full conversation message list from alfworld_run_batch
            reward: True if the agent won, False otherwise
        """
        trajectory = self._trajectory_to_text(messages)
        memory_items = self._extract_memory_items(task, trajectory, success=reward)

        record = {
            "task_id": task_id,
            "query": task,
            "memory_items": memory_items,
            "status": "success" if reward else "fail",
        }
        self._save_record(record)
        self.memory_bank.append(record)
        self._rebuild_bm25()
        logger.info(f"Indexed {'success' if reward else 'fail'} memory for: {task_id}")

    def retrieve(self, query: str, n: int = None) -> str:
        """
        Retrieve relevant memory items for a given task query using BM25.

        Args:
            query (str): The task description to search for.
            n (int): Number of records to retrieve (defaults to self.retrieve_num).

        Returns:
            A formatted string of memory items to inject into the prompt,
            or empty string if no memories exist yet.
        """
        if self.bm25_retriever is None:
            return ""

        if n is not None and n != self.retrieve_num:
            docs_all = [
                Document(
                    page_content=record["query"],
                    metadata={"task_id": record["task_id"], "idx": i},
                )
                for i, record in enumerate(self.memory_bank)
            ]
            retriever = BM25Retriever.from_documents(docs_all, k=n)
        else:
            retriever = self.bm25_retriever

        docs = retriever.invoke(query)
        if not docs:
            return ""

        mem_parts = []
        for doc in docs:
            idx = doc.metadata["idx"]
            record = self.memory_bank[idx]
            mem_parts.append("\n\n".join(record["memory_items"]))

        return "\n\n---\n\n".join(mem_parts)
