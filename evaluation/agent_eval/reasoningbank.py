import os
import json
import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from google import genai
from google.genai.types import HttpOptions, GenerateContentConfig
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# Config Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# System Instructions for Memory Extraction, depending on success or failure

SUCCESSFUL_SI = """
You are an expert in coding, specifically fixing a given issue in a code repository. You will be given an issue to be fixed, the corresponding trajectory that represents **how an agent successfully resolved the issue**. 

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first think why the trajectory is successful, and then summarize the insights.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolve the issue in the future>
```
"""

FAILED_SI = """
You are an expert in coding, specifically fixing a given issue in a code repository. You will be given a user query, the corresponding trajectory that represents **how an agent attempted to resolve the issue but failed**. 

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
  - You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
  - You can extract *at most 3* memory items from the trajectory.
  - You must not repeat similar or overlapping items.
  - Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:

```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully resolve the issue in the future>
```
"""


class ReasoningBank:
    def __init__(
        self,
        storage_path: str = "./memory/reasoning_bank.jsonl",
        embedding_path: str = "./memory/embeddings.jsonl",
        model_name: str = "gemini-2.5-flash",
        embedding_model: str = "gemini-embedding-001",
        embed_instruction: str = "Given the prior software engineering queries, your task is to analyze a current query's intent and select relevant prior queries that could help resolve it.",
        retrieve_num: int = 3,
        curation_model_hf=None,
        curation_tokenizer=None,
        curation_model_name: str = None,
        curation_base_url: str = None,
    ):
        self.storage_path = storage_path
        self.embedding_path = embedding_path
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.embed_instruction = embed_instruction
        self.retrieve_num = retrieve_num
        self.curation_model_hf = curation_model_hf
        self.curation_tokenizer = curation_tokenizer
        self.curation_model_name = curation_model_name
        self.curation_base_url = curation_base_url

        # Gemini client only needed when not using local vLLM or HTTP API
        self.client = (
            None if (curation_base_url is not None or curation_model_hf is not None)
            else genai.Client(http_options=HttpOptions(api_version="v1"))
        )
        
        # ensure memory directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.embedding_path), exist_ok=True)
        
        # load existing memory bank and embeddings
        self.memory_bank = self._load_jsonl(self.storage_path)
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._rebuild_bm25()


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

    def _load_jsonl(self, path: str) -> List[Dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
        

    def _l2_normalize(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return F.normalize(x, p=2, dim=dim)
    

    def _get_gemini_embedding(self, text: str, dimensionality: int = 3072) -> torch.Tensor:
        """Obtain text embedding using Gemini Text Embedding Model."""
        model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
        text_input = TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT")
        resp = model.get_embeddings([text_input], output_dimensionality=dimensionality)
        return torch.tensor([resp[0].values], dtype=torch.float32)


    def _get_qwen_embedding(self, query: str) -> Tuple[torch.Tensor, str, int]:
        """Returns (1, D) torch tensor (on CPU), model_name, dim."""
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        batch = tokenizer([query], max_length=1024, padding=True, truncation=True, return_tensors='pt')
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
            last_hidden = out.last_hidden_state  # (1, L, D)
            masked = last_hidden.masked_fill(~batch['attention_mask'][..., None].bool(), 0.0)
            pooled = masked.sum(dim=1) / batch['attention_mask'].sum(dim=1)[..., None]  # (1, D)
        pooled = pooled.to('cpu')
        pooled = self._l2_normalize(pooled, dim=1)
        return pooled
    

    def _load_cached_embeddings(self) -> Tuple[List[str], List[str], torch.Tensor]:
        ids, texts, vecs = [], [], []
        if not os.path.exists(self.embedding_path):
            return ids, texts, torch.empty(0)

        with open(self.embedding_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                ids.append(obj["id"])
                texts.append(obj.get("text", ""))
                vecs.append(obj["embedding"])

        if not vecs:
            return ids, texts, torch.empty(0)

        emb = torch.tensor(vecs, dtype=torch.float32)
        return ids, texts, self._l2_normalize(emb, dim=1)
    

    def _llm_judge_status(self, task: str, trajectory: str) -> bool:
        prompt = f"Task: {task}\n\nTrajectory:\n{trajectory}\n\nDid the agent successfully complete the task? Answer with 'success' or 'fail' only."
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=GenerateContentConfig(
                temperature=0.0,
                system_instruction="You are a helpful assistant that judges whether the agent successfully completed the task.",
            )
        )
        return "success" in response.text.strip().lower()


    def add(self, instance_id: str, task: str, trajectory: str,
            is_successful: bool = None):
        """
        Evaluates task performance, generates structured memory items using
        status-specific instructions, and persists data to both storage and vector cache.

        Args:
            instance_id (str): Unique identifier for the task instance.
            task (str): The original problem statement or user query.
            trajectory (str): The full sequence of actions and thoughts from the agent.
            is_successful (bool): If provided, skips the LLM judge and uses this value
                                  directly (e.g. from exact-match scoring).
        """
        # 1. Judge the outcome — use provided value or fall back to LLM judge
        if is_successful is None:
            is_successful = self._llm_judge_status(task, trajectory)
        status_str = "success" if is_successful else "fail"
        
        # 2. Prepare the prompt and select the appropriate System Instruction
        # The prompt combines the original task and the execution history
        generation_prompt = f"**Query:** {task}\n\n**Trajectory:**\n{trajectory}"
        system_instruction = SUCCESSFUL_SI if is_successful else FAILED_SI

        # 3. Generate structured memory items via the LLM
        # This extracts generalizable insights (titles, descriptions, contents)
        generated_memory_items = self._extract_memory_items(
            prompt=generation_prompt, 
            si=system_instruction
        )

        # 4. Persist the complete record to the primary JSONL storage
        record = {
            "task_id": instance_id, 
            "query": task, 
            "memory_items": generated_memory_items,
            "status": status_str,
        }
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.memory_bank.append(record)

        # Embedding-based retrieval is superseded by BM25; skip embedding generation.

        self._rebuild_bm25()
        logger.info(f"Successfully indexed {status_str} memory for task: {instance_id}")


    def _llm_vllm(self, system: str, user: str) -> str:
        """Call the curation model: HTTP API if curation_base_url is set, else local vLLM."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.curation_base_url is not None:
            from litellm import completion
            resp = completion(
                model=self.curation_model_name,
                messages=messages,
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=self.curation_base_url,
                temperature=0.7,
                max_tokens=2048,
                num_retries=10,
            )
            return resp.choices[0].message.content or ""
        if self.curation_tokenizer is not None:
            prompt = self.curation_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        else:
            prompt = f"{system}\n\n{user}"
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
        outputs = self.curation_model_hf.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

    def _extract_memory_items(self, prompt: str, si: str) -> List[str]:
        """
        Extract memory items via API/vLLM (if configured) or Gemini.

        Returns:
            List[str]: A list of memory item strings parsed from the LLM response.
        """
        if self.curation_base_url is not None or self.curation_model_hf is not None:
            raw = self._llm_vllm(system=si.strip(), user=prompt)
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=65536,
                    system_instruction=si.strip(),
                )
            )
            raw = response.text.strip()
        raw = raw.split("</think>")[-1].strip()
        return [b.strip() for b in raw.split("\n\n") if b.strip()]
    

    def retrieve(self, cur_query: str, n: int = None) -> str:
        """
        Retrieve the top-n most relevant past experiences using BM25.

        Args:
            cur_query (str): The current task or problem statement.
            n (int): Number of records to retrieve (defaults to self.retrieve_num).

        Returns:
            str: Formatted memory items ready for prompt injection,
                 or empty string if the bank is empty.
        """
        if self.bm25_retriever is None:
            logger.warning("ReasoningBank is empty. No records available for retrieval.")
            return ""

        if n is not None and n != self.retrieve_num:
            # Rebuild temporarily with the requested k
            docs = [
                Document(
                    page_content=record["query"],
                    metadata={"task_id": record["task_id"], "idx": i},
                )
                for i, record in enumerate(self.memory_bank)
            ]
            retriever = BM25Retriever.from_documents(docs, k=n)
        else:
            retriever = self.bm25_retriever

        matched_docs = retriever.invoke(cur_query)
        if not matched_docs:
            return ""

        mem_parts = []
        for doc in matched_docs:
            idx = doc.metadata["idx"]
            record = self.memory_bank[idx]
            mem_parts.append("\n\n".join(record["memory_items"]))

        return "\n\n---\n\n".join(mem_parts)



# Sample Use
    # res = memory.retrieve(task, n=1)
    # if not res:
    #     selected_memory = ""
    # else:
    #     mem_items = []
    #     for item in res:
    #         for i in item["memory_items"]:
    #             mem_items.append(i)
    #     selected_memory = "\n\n".join(mem_items)

    # progress_manager.on_instance_start(instance_id)
    # progress_manager.update_instance_status(instance_id, "Pulling/starting docker")

    # agent = None
    # extra_info = None

    # try:
    #     env = get_sb_environment(config, instance)
    #     agent = ProgressTrackingAgent(
    #         model,
    #         env,
    #         progress_manager=progress_manager,
    #         instance_id=instance_id,
    #         **config.get("agent", {}),
    #     )
    #     exit_status, result = agent.run(task, selected_memory=selected_memory)
    # except Exception as e:
    #     logger.error(f"Error processing instance {instance_id}: {e}", exc_info=True)
    #     exit_status, result = type(e).__name__, str(e)
    #     extra_info = {"traceback": traceback.format_exc()}
    # finally:
    #     save_traj(
    #         agent,
    #         instance_dir / f"{instance_id}.traj.json",
    #         exit_status=exit_status,
    #         result=result,
    #         extra_info=extra_info,
    #         instance_id=instance_id,
    #         print_fct=logger.info,
    #     )
    #     update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, result)
    #     progress_manager.on_instance_end(instance_id, exit_status)

    #     # read trajectory and extract memory
    #     with open(instance_dir / f"{instance_id}.traj.json", "r") as f:
    #         messages = json.load(f)["messages"]
    #     trajectory = "\n".join([m["content"] for m in messages if m["role"] != "system"])

    #     memory.add(trajectory, task, instance_id)