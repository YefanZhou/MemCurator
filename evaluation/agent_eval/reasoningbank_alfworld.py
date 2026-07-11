import os
import json
import logging
from typing import Dict, List, Optional
from vllm import SamplingParams
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        if self.curation_base_url is not None:
            from litellm import completion
            resp = completion(
                model=self.curation_model_name,
                messages=messages,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.curation_base_url,
                temperature=0.7,
                max_tokens=1024,
                num_retries=10,
            )
            return resp.choices[0].message.content or ""
        if self.curation_tokenizer is not None:
            prompt = self.curation_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            prompt = f"{system}\n\n{user}"
        sampling_params = SamplingParams(temperature=0.7, max_tokens=1024)
        outputs = self.curation_model_hf.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _extract_memory_items(self, task: str, trajectory: str, success: bool) -> List[str]:
        si = SUCCESSFUL_SI if success else FAILED_SI
        prompt = f"**Task:** {task}\n\n**Trajectory:**\n{trajectory}"
        raw = self._llm(si, prompt)
        # Split by Memory Item blocks
        raw = raw.split("</think>")[-1]  # In case the model adds <think> tags, ignore them
        items = [block.strip() for block in raw.split("# Memory Item") if block.strip()]
        return ["# Memory Item " + item for item in items]

    def _trajectory_to_text(self, messages: List[Dict]) -> str:
        """Convert ALFWorld message list to readable trajectory string."""
        import re
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
