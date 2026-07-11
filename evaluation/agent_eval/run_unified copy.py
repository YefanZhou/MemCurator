"""
run_unified.py  —  Single entry-point for all MemP experiments.

Supported combinations
----------------------
  --env         : alfworld | webshop | amc23 | aime24 | aime25 | gpqa
  --memory_type : none | skillos | reasoningbank

  Note: reasoningbank only supports alfworld.
        Reasoning envs (amc23/aime24/aime25/gpqa) ignore --memory_type.

Example commands
----------------
  # No memory — ALFWorld
  python run_unified.py --env alfworld --memory_type none --model openai/Qwen/Qwen3-8B --exp_name baseline

  # SkillOS — ALFWorld
  python run_unified.py --env alfworld --memory_type skillos --curation_model Qwen/Qwen3-8B \
      --model openai/Qwen/Qwen3-8B --exp_name skillos-qwen3-8b

  # ReasoningBank — ALFWorld
  python run_unified.py --env alfworld --memory_type reasoningbank --curation_model Qwen/Qwen3-8B \
      --model openai/Qwen/Qwen3-8B --exp_name rb-qwen3-8b

  # No memory — WebShop
  python run_unified.py --env webshop --memory_type none --model openai/Qwen/Qwen3-8B --exp_name baseline

  # SkillOS — WebShop
  python run_unified.py --env webshop --memory_type skillos --curation_model Qwen/Qwen3-8B \
      --model openai/Qwen/Qwen3-8B --exp_name skillos-qwen3-8b

  # Reasoning benchmark (memory_type ignored)
  python run_unified.py --env aime24 --model openai/Qwen/Qwen3-8B --exp_name run1
"""

import os
import sys
import json
import math
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import yaml
from litellm import completion

# Heavy imports deferred: loaded only when needed
LLM = None
SamplingParams = None
AutoTokenizer = None
QwenFnCallPrompt = None
Message = None
ContentItem = None

# SkillOS path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SkillOS'))
from skills_memory import SkillMemory

# Qwen function-call special tokens
FN_NAME   = '✿FUNCTION✿'
FN_ARGS   = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT   = '✿RETURN✿'

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
openai.api_key = os.environ["OPENAI_API_KEY"]

HISTORY_LENGTH = 5

REASONING_ENVS = {'amc23', 'aime24', 'aime25', 'gpqa'}

# ------------------------------------------------------------------ #
# Prompt templates                                                    #
# ------------------------------------------------------------------ #

ALFWORLD_TEMPLATE_NO_HIS = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\
"""

ALFWORLD_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.\
"""

ALFWORLD_TEMPLATE_WITH_CONTEXT = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Past Relevant {context_label}

{retrieved_context}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant {context_label_lower}. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.\
"""

WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE_WITH_CONTEXT = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. Your task is to: {task_description}.

## Past Relevant {context_label}

{retrieved_context}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are:
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

REASONING_TEMPLATE = "{question}\n\nPlease reason step by step and put final answer within \\boxed{{}}."

REASONING_TEMPLATE_WITH_CONTEXT = """\
## Past Relevant {context_label}

{retrieved_context}

## Problem

{question}

Please reason step by step, using the past relevant {context_label_lower} where helpful, and put your final answer within \\boxed{{}}.\
"""

GPQA_TEMPLATE = "{question}\n\nChoices:\n(A) {choice_a}\n(B) {choice_b}\n(C) {choice_c}\n(D) {choice_d}\n\nPlease reason step by step and put final answer within \\boxed{{}}."

GPQA_TEMPLATE_WITH_CONTEXT = """\
## Past Relevant {context_label}

{retrieved_context}

## Problem

{question}

Choices:
(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}

Please reason step by step, using the past relevant {context_label_lower} where helpful, and put your final answer within \\boxed{{}}.\
"""

# ------------------------------------------------------------------ #
# SkillOS tool schemas                                                #
# ------------------------------------------------------------------ #

MEMORY_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "new_skill_insert",
            "description": "If there is no existing relevant skill, create new skill with desired skill name and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "The name of the new skill to create."},
                    "content":    {"type": "string", "description": "The markdown content for the new skill."},
                },
                "required": ["skill_name", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_update",
            "description": "If the existing skill can be improved, update the specific skill by its skill_name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name":   {"type": "string", "description": "The name of the skill to update. Must exactly match an existing skill title."},
                    "new_name":     {"type": "string", "description": "The new skill name (optional)."},
                    "new_content":  {"type": "string", "description": "The new full content for the skill (optional)."},
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill_delete",
            "description": "Delete an existing skill by its title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "The name of the skill to delete."},
                },
                "required": ["skill_name"],
            },
        },
    },
]

# ------------------------------------------------------------------ #
# LLM helpers                                                         #
# ------------------------------------------------------------------ #

def llm_vertexai(prompt, model="gemini-2.5-pro"):
    from google import genai
    from google.genai import types
    if isinstance(prompt, list):
        text = "\n".join(m["content"] for m in prompt if m.get("role") != "system")
    elif isinstance(prompt, str):
        text = prompt
    else:
        raise ValueError(f'prompt must be a list or a string, got {type(prompt)}')
    client = genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
    )
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(temperature=0.7),
    )
    return response.text or "Output Error"


def llm(prompt, stop=None, model="openai/Qwen/Qwen2.5-7B-Instruct"):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, got {type(prompt)}')
    if model.startswith("gemini/"):
        return llm_vertexai(prompt, model=model[len("gemini/"):])
    response = completion(
        model=model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['OPENAI_API_BASE'],
        num_retries=10,
        temperature=0.7,
        stop=stop,
    )
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    return "Output Error"


# ------------------------------------------------------------------ #
# Shared utilities                                                    #
# ------------------------------------------------------------------ #

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ') + 2:]
    return ob


def format_action_history(history, history_length):
    recent = history[-history_length:]
    if not recent:
        return "None"
    lines = []
    for i, (obs, action) in enumerate(recent, 1):
        lines.append(f"Observation {i}: {obs}")
        lines.append(f"Action {i}: {action}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Memory retrieval helpers                                            #
# ------------------------------------------------------------------ #

def get_skillos_text(skill_memory: SkillMemory, query: str, retrieve_num: int) -> str:
    """Retrieve relevant skills from SkillMemory and return formatted text."""
    if not skill_memory.skills:
        return ""
    results = skill_memory.memory_search(query=query, top_k=retrieve_num, search_method="bm25")
    if not results:
        return ""
    parts = []
    for idx, (skill, _) in enumerate(results):
        parts.append(f"**Skill {idx + 1}: {skill['title']}**\n{skill['content']}")
    return "\n\n---\n\n".join(parts)


def get_reasoningbank_text(bank, query: str, retrieve_num: int) -> str:
    """Retrieve from ReasoningBankAlfworld or ReasoningBank — both return str via BM25."""
    result = bank.retrieve(query, retrieve_num)
    return result if result else ""


def retrieve_context(memory_type, memory_obj, query: str, retrieve_num: int) -> str:
    """Unified retrieval: returns context string (empty if none)."""
    if memory_type == 'skillos':
        return get_skillos_text(memory_obj, query, retrieve_num)
    elif memory_type == 'reasoningbank':
        return get_reasoningbank_text(memory_obj, query, retrieve_num)
    return ""


# ------------------------------------------------------------------ #
# SkillOS persistence                                                 #
# ------------------------------------------------------------------ #

def save_skills(skill_memory: SkillMemory, storage_path: str):
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    with open(storage_path, "w", encoding="utf-8") as f:
        json.dump(skill_memory.skills, f, indent=2, ensure_ascii=False)


def load_skills(skill_memory: SkillMemory, storage_path: str):
    if not os.path.exists(storage_path):
        return
    with open(storage_path, "r", encoding="utf-8") as f:
        skill_memory.skills = json.load(f)


# ------------------------------------------------------------------ #
# ALFWorld batch runner                                               #
# ------------------------------------------------------------------ #

def alfworld_run_batch(env, obs, names, task_descriptions, admissible_commands,
                       max_steps=30, model="openai/Qwen/Qwen2.5-7B-Instruct",
                       skills_context=None, context_label="Skills"):
    """
    Run a batch of ALFWorld tasks.

    skills_context : dict mapping game index -> retrieved context text (empty = no injection)
    context_label  : section heading used in the with-context template ("Skills" or "Memories")
    """
    n = len(obs)
    histories = [[] for _ in range(n)]
    current_obs = list(obs)
    current_admissible = list(admissible_commands)
    task_rewards = [0] * n
    active_tasks = list(range(n))

    for _ in range(max_steps):
        if not active_tasks:
            break
        print(f'\033[91mActive tasks: {active_tasks}\033[0m')

        prompts = {}
        for idx in active_tasks:
            history_str = format_action_history(histories[idx], HISTORY_LENGTH)
            admissible_str = "\n ".join(f"'{s}'" for s in current_admissible[idx] if s != 'help')
            step_count = len(histories[idx])
            ctx_text = skills_context.get(idx, "") if skills_context else ""

            if step_count == 0:
                prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=current_obs[idx],
                    admissible_actions=admissible_str,
                )
            elif ctx_text:
                prompt_text = ALFWORLD_TEMPLATE_WITH_CONTEXT.format(
                    task_description=task_descriptions[idx],
                    retrieved_context=ctx_text,
                    context_label=context_label,
                    context_label_lower=context_label.lower(),
                    step_count=step_count,
                    history_length=min(HISTORY_LENGTH, step_count),
                    action_history=history_str,
                    current_step=step_count + 1,
                    current_observation=current_obs[idx],
                    admissible_actions=admissible_str,
                )
            else:
                prompt_text = ALFWORLD_TEMPLATE.format(
                    task_description=task_descriptions[idx],
                    step_count=step_count,
                    history_length=min(HISTORY_LENGTH, step_count),
                    action_history=history_str,
                    current_step=step_count + 1,
                    current_observation=current_obs[idx],
                    admissible_actions=admissible_str,
                )

            prompts[idx] = [{"role": "user", "content": prompt_text}]

        responses = {}
        with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
            futures = {executor.submit(llm, prompts[idx], None, model): idx for idx in active_tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response = future.result()
                    print(f'\033[92mAgent {idx}: \n{response}\033[0m')
                    responses[idx] = response
                except Exception as e:
                    print(f'Error {idx}: {e}')

        responses = dict(sorted(responses.items()))

        action_list = [""] * n
        for idx in active_tasks:
            if idx in responses:
                response = responses[idx]
                if '<action>' in response and '</action>' in response:
                    action_list[idx] = response.split('<action>')[-1].split('</action>')[0].strip()

        observation, _, done, info = env.step(action_list)
        observation = [process_ob(ob) for ob in observation]
        print(f'\033[93mObservation: \n{observation}\033[0m')
        new_admissible = info.get('admissible_commands', current_admissible)
        won = info['won']

        new_active_tasks = []
        for idx in active_tasks:
            histories[idx].append((current_obs[idx], action_list[idx]))
            current_obs[idx] = observation[idx]
            current_admissible[idx] = new_admissible[idx]
            if done[idx]:
                task_rewards[idx] = won[idx]
            else:
                new_active_tasks.append(idx)
        active_tasks = new_active_tasks

    results = []
    for idx in range(n):
        messages = []
        for obs_h, act_h in histories[idx]:
            messages.append({"role": "user",      "content": obs_h})
            messages.append({"role": "assistant",  "content": act_h})
        messages.append({"role": "user", "content": current_obs[idx]})
        results.append({"messages": messages, "reward": task_rewards[idx], "name": names[idx]})

    return results


# ------------------------------------------------------------------ #
# WebShop batch runner                                                #
# ------------------------------------------------------------------ #

def _make_webshop_manager(raw_env):
    from types import SimpleNamespace
    sys.path.insert(0, os.path.dirname(__file__))
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection
    from functools import partial

    cfg = SimpleNamespace(env=SimpleNamespace(history_length=HISTORY_LENGTH))

    class _Manager(WebshopEnvironmentManager):
        def reset(self, indices=None):
            if indices is not None:
                obs, infos = self.envs.reset(indices=indices)
            else:
                obs, infos = self.envs.reset()
            self.tasks = self.extract_task(obs)
            obs = self.format_obs(obs)
            self.pre_text_obs = obs
            self.memory.reset(batch_size=len(infos))
            observations = {
                'text':   self.build_text_obs(obs, infos, init=True),
                'image':  None,
                'anchor': obs.copy(),
            }
            return observations, infos

    return _Manager(raw_env, partial(webshop_projection), cfg)


def webshop_run_batch(manager, obs_dict, max_steps=30,
                      model="openai/Qwen/Qwen2.5-7B-Instruct",
                      skills_context=None):
    """
    Run a batch of WebShop tasks with optional skill injection.

    skills_context : dict mapping batch index -> retrieved context text.
                     When non-empty, skills are prepended to the prompt at
                     each step after the first (mirrors ALFWorld behaviour).
    """
    n = len(obs_dict['text'])
    prompts = list(obs_dict['text'])
    task_rewards = [0.0] * n
    dones_arr = [False] * n
    histories = [[] for _ in range(n)]
    step_counts = [0] * n

    for _ in range(max_steps):
        active = [i for i in range(n) if not dones_arr[i]]
        if not active:
            break
        print(f'\033[91mActive tasks: {active}\033[0m')

        responses = [""] * n
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {}
            for i in active:
                ctx_text = (skills_context.get(i, "") if skills_context else "")
                if ctx_text and step_counts[i] > 0:
                    content = f"## Past Relevant Skills\n\n{ctx_text}\n\n---\n\n{prompts[i]}"
                else:
                    content = prompts[i]
                futures[executor.submit(llm, [{"role": "user", "content": content}], None, model)] = i
            for future in as_completed(futures):
                i = futures[future]
                try:
                    responses[i] = future.result()
                    print(f'\033[92mAgent {i}: \n{responses[i]}\033[0m')
                except Exception as e:
                    print(f'Error {i}: {e}')

        for i in active:
            histories[i].append((prompts[i], responses[i]))
            step_counts[i] += 1

        next_obs_dict, rewards, dones, infos = manager.step(responses)
        prompts = list(next_obs_dict['text'])
        print(f'\033[93mObservation: \n{prompts}\033[0m')

        for i in range(n):
            if dones[i] and not dones_arr[i]:
                dones_arr[i] = True
                task_rewards[i] = infos[i].get('task_score', rewards[i])

    results = []
    for i in range(n):
        messages = []
        for obs_h, act_h in histories[i]:
            messages.append({"role": "user",      "content": obs_h})
            messages.append({"role": "assistant",  "content": act_h})
        messages.append({"role": "user", "content": prompts[i]})
        results.append({"messages": messages, "reward": task_rewards[i]})
    return results


# ------------------------------------------------------------------ #
# Reasoning benchmarks                                                #
# ------------------------------------------------------------------ #

REASONING_DATASETS = {
    'amc23':  ('math-ai/amc23',   None),
    'aime24': ('math-ai/aime24',  None),
    'aime25': ('math-ai/aime25',  None),
    'gpqa':   ('Idavidrein/gpqa', 'gpqa_diamond'),
}


def load_reasoning_dataset(env_name):
    """
    Returns a list of dicts with at minimum:
      'question' : the fully-formatted prompt (no-memory version)
      'answer'   : ground-truth answer string
    For GPQA problems an extra 'raw' key is included with the unformatted
    fields so that the with-context template can be built at runtime.
    """
    problems = []
    if env_name == 'gpqa':
        import random, pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), 'gpqa_diamond.csv')
        data = pd.read_csv(csv_path).to_dict('records')
        for row in data:
            choices = [row['Correct Answer'], row['Incorrect Answer 1'],
                       row['Incorrect Answer 2'], row['Incorrect Answer 3']]
            rng = random.Random(hash(row['Question']))
            rng.shuffle(choices)
            labels = ['A', 'B', 'C', 'D']
            correct_label = labels[choices.index(row['Correct Answer'])]
            prompt = GPQA_TEMPLATE.format(
                question=row['Question'],
                choice_a=choices[0], choice_b=choices[1],
                choice_c=choices[2], choice_d=choices[3],
            )
            problems.append({
                'question': prompt,
                'answer':   correct_label,
                # kept for context-injected template
                'raw': {
                    'question': row['Question'],
                    'choice_a': choices[0], 'choice_b': choices[1],
                    'choice_c': choices[2], 'choice_d': choices[3],
                },
            })
    else:
        from datasets import load_dataset
        hf_name, subset = REASONING_DATASETS[env_name]
        ds = load_dataset(hf_name, subset, trust_remote_code=True)
        split = 'test' if 'test' in ds else list(ds.keys())[0]
        col_q = {'aime24': 'problem', 'aime25': 'problem', 'amc23': 'question'}[env_name]
        col_a = {'aime24': 'solution', 'aime25': 'answer',  'amc23': 'answer'}[env_name]
        for row in ds[split]:
            problems.append({
                'question': REASONING_TEMPLATE.format(question=row[col_q]),
                'answer':   str(row[col_a]),
                'raw': {'question': row[col_q]},
            })
    return problems


def format_reasoningbank_results(results) -> str:
    """
    Convert the List[Dict] returned by ReasoningBank.retrieve() into a
    human-readable string for prompt injection.
    Each dict has keys: task_id, query, memory_items (list of str), status.
    """
    if not results:
        return ""
    parts = []
    for item in results:
        for mem_item in item.get("memory_items", []):
            if mem_item.strip():
                parts.append(mem_item.strip())
    return "\n\n".join(parts)


def reasoning_trajectory_to_text(messages: list) -> str:
    """Format a single-turn reasoning trajectory (question + answer) as plain text."""
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"[Question]: {m['content']}")
        else:
            parts.append(f"[Answer]: {m['content']}")
    return "\n\n".join(parts)


def build_reasoning_prompt_with_context(prob: dict, ctx_text: str,
                                        context_label: str, env_name: str) -> str:
    """Build the context-injected prompt for a reasoning problem."""
    lbl_lower = context_label.lower()
    raw = prob.get('raw', {})
    if env_name == 'gpqa':
        return GPQA_TEMPLATE_WITH_CONTEXT.format(
            context_label=context_label,
            context_label_lower=lbl_lower,
            retrieved_context=ctx_text,
            question=raw.get('question', ''),
            choice_a=raw.get('choice_a', ''), choice_b=raw.get('choice_b', ''),
            choice_c=raw.get('choice_c', ''), choice_d=raw.get('choice_d', ''),
        )
    else:
        return REASONING_TEMPLATE_WITH_CONTEXT.format(
            context_label=context_label,
            context_label_lower=lbl_lower,
            retrieved_context=ctx_text,
            question=raw.get('question', prob['question']),
        )


def score_reasoning(pred_text, gold, env_name):
    if env_name == 'gpqa':
        # Handle \boxed{A} and nested forms like \boxed{\text{A}}, \boxed{\textbf{A}}
        matches = re.findall(r'\\boxed\{([^{}]*)\}', pred_text)
        if not matches:
            matches = re.findall(r'\\boxed\{\\[a-z]+\{([^{}]*)\}\}', pred_text)
        if matches:
            raw = matches[-1].strip()
            letter = re.search(r'\b([A-D])\b', raw, re.IGNORECASE)
            pred_letter = letter.group(1).upper() if letter else raw.upper()
        else:
            # Fallback for models that don't use \boxed{} (e.g. Gemini natural language)
            nl_patterns = [
                r'[Ff]inal [Aa]nswer[^\n]*\b([A-D])\b',
                r'[Aa]nswer is[^\n]*\b([A-D])\b',
                r'choice \(([A-D])\)',
                r'option \(([A-D])\)',
                r'\(([A-D])\) is correct',
                r'corresponds to \(([A-D])\)',
                r'corresponds to choice ([A-D])\b',
                r'is therefore ([A-D])\b',
                r'answer: \*\*([A-D])\*\*',
                r'\*\*([A-D])\*\* is (?:correct|the answer)',
            ]
            pred_letter = ''
            for pat in nl_patterns:
                m = re.search(pat, pred_text, re.IGNORECASE)
                if m:
                    pred_letter = m.group(1).upper()
                    break
        return 1.0 if pred_letter == gold.strip().upper() else 0.0
    else:
        from math_verify import parse, verify
        try:
            return 1.0 if verify(parse(gold), parse(pred_text)) else 0.0
        except Exception:
            return 0.0


def run_reasoning(problems, model, batch_size, output_path, finished, env_name='',
                  memory_type='none', memory_obj=None, retrieve_num=3,
                  curation_tokenizer=None, curation_model_hf=None,
                  skills_storage_path=None,
                  curation_model=None, curation_base_url=None):
    """
    Single-turn inference over all problems, with optional memory augmentation.

    memory_type : 'none' | 'skillos' | 'reasoningbank'
    memory_obj  : SkillMemory | ReasoningBank | None
    """
    use_memory = memory_type != 'none' and memory_obj is not None
    context_label = "Skills" if memory_type == 'skillos' else "Memories"
    all_reward = 0.0

    for i in tqdm(range(0, len(problems), batch_size)):
        batch = problems[i:i + batch_size]

        # ---- Skip already-finished problems ----
        if i + len(batch) <= finished:
            for j in range(len(batch)):
                fpath = f'{output_path}/idx_{i+j}.json'
                if os.path.exists(fpath):
                    all_reward += json.load(open(fpath))['reward']
            continue

        # ---- Retrieve context for each problem in batch ----
        batch_contexts = [''] * len(batch)
        if use_memory:
            for j, prob in enumerate(batch):
                query = prob['raw'].get('question', prob['question'])
                batch_contexts[j] = retrieve_context(memory_type, memory_obj, query, retrieve_num)

        # ---- Build prompts ----
        prompts = []
        for j, prob in enumerate(batch):
            ctx = batch_contexts[j]
            if ctx:
                prompts.append(build_reasoning_prompt_with_context(
                    prob, ctx, context_label, env_name
                ))
            else:
                prompts.append(prob['question'])

        # ---- Parallel inference ----
        responses = [''] * len(batch)
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    llm,
                    [{"role": "user", "content": prompts[j]}],
                    None, model,
                ): j
                for j in range(len(batch))
            }
            for future in as_completed(futures):
                j = futures[future]
                try:
                    responses[j] = future.result()
                    print(f'\033[92mProblem {i+j}: {responses[j][:120]}\033[0m')
                except Exception as e:
                    print(f'Error problem {i+j}: {e}')

        # ---- Score, save, update memory ----
        for j, (prob, resp, ctx) in enumerate(zip(batch, responses, batch_contexts)):
            if i + j < finished:
                continue
            reward = score_reasoning(resp, prob['answer'], env_name)
            result = {
                'messages': [
                    {"role": "user",      "content": prompts[j]},
                    {"role": "assistant", "content": resp},
                ],
                'answer':   prob['answer'],
                'reward':   reward,
            }
            with open(f'{output_path}/idx_{i+j}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            all_reward += reward

            # Update memory from this result
            if use_memory:
                task_text  = prob['raw'].get('question', prob['question'])
                instance_id = f'problem_{i+j}'
                if memory_type == 'skillos':
                    # Batch curation handled below after the inner loop
                    pass
                elif memory_type == 'reasoningbank':
                    trajectory = reasoning_trajectory_to_text(result['messages'])
                    memory_obj.add(
                        instance_id=instance_id,
                        task=task_text,
                        trajectory=trajectory,
                        is_successful=bool(reward),
                    )

        # ---- SkillOS: batch curation after each mini-batch ----
        if use_memory and memory_type == 'skillos':
            batch_data = []
            for j, (prob, resp, ctx) in enumerate(zip(batch, responses, batch_contexts)):
                if i + j < finished:
                    continue
                task_text = prob['raw'].get('question', prob['question'])
                messages  = [
                    {"role": "user",      "content": prompts[j]},
                    {"role": "assistant", "content": resp},
                ]
                reward = score_reasoning(resp, prob['answer'], env_name)
                batch_data.append((task_text, messages, bool(reward), ctx))
            if batch_data:
                batch_update_skills_from_trajectories(
                    skill_memory=memory_obj,
                    curation_tokenizer=curation_tokenizer,
                    curation_model_hf=curation_model_hf,
                    batch_data=batch_data,
                    curation_model=curation_model,
                    curation_base_url=curation_base_url,
                )
                save_skills(memory_obj, skills_storage_path)
                print(f"Skills saved: {len(memory_obj.skills)} total skills.")

        done = min(i + len(batch), len(problems))
        tqdm.write(f'Avg accuracy: {all_reward / done * 100:.2f}%  [{done}/{len(problems)}]')

    print(f'\nFinal accuracy: {all_reward / len(problems) * 100:.2f}%  ({int(all_reward)}/{len(problems)})')


# ------------------------------------------------------------------ #
# SkillOS curation                                                    #
# ------------------------------------------------------------------ #

def execute_tool(skill_memory: SkillMemory, tool_name: str, arguments: dict) -> dict:
    if tool_name == "new_skill_insert":
        try:
            title = skill_memory.new_memory_insert(arguments["skill_name"], arguments["content"])
            return {"status": "ok", "message": "Skill created.", "skill_name": title}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "skill_update":
        try:
            updated = skill_memory.memory_update(
                title=arguments["skill_name"],
                new_name=arguments.get("new_name"),
                new_content=arguments.get("new_content"),
            )
            return {"status": "ok", "message": "Skill updated.", "updated_skill": updated}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "skill_delete":
        try:
            skill_memory.memory_delete(arguments["skill_name"])
            return {"status": "ok", "message": "Skill deleted."}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "error", "message": f"Unknown tool: {tool_name}"}


def _remove_incomplete_special_tokens(text: str) -> str:
    special_tokens = (FN_NAME, FN_ARGS, FN_RESULT, FN_EXIT)
    text = text.rstrip()
    if text.endswith(special_tokens):
        for s in special_tokens:
            if text.endswith(s):
                text = text[:-len(s)]
                break
    else:
        trail_start = text.rfind('✿')
        if trail_start >= 0:
            trail_token = text[trail_start:]
            for s in special_tokens:
                if s.startswith(trail_token):
                    text = text[:trail_start]
                    break
    return text.lstrip('\n').rstrip()


def _remove_trailing_comment_of_fn_args(fn_args: str) -> str:
    fn_args = fn_args.strip()
    if fn_args.startswith('{'):
        k = fn_args.rfind('}')
        if k > 0:
            fn_args = fn_args[:k + 1]
    if fn_args.startswith('```'):
        k = fn_args.rfind('\n```')
        if k > 0:
            fn_args = fn_args[:k + 4]
    return fn_args


def _parse_function_calls_from_text(text: str):
    function_calls = []
    pattern = (f'{re.escape(FN_NAME)}:\\s*([^\\n]+)\\s*'
               f'{re.escape(FN_ARGS)}:\\s*([^✿]+?)'
               f'(?={re.escape(FN_RESULT)}|{re.escape(FN_EXIT)}|{re.escape(FN_NAME)}|$)')
    for match in re.finditer(pattern, text, re.DOTALL):
        func_name = match.group(1).strip()
        func_args = _remove_trailing_comment_of_fn_args(match.group(2).strip())
        function_calls.append({'name': func_name, 'arguments': func_args})
    first_fn_pos = text.find(FN_NAME)
    remaining_text = _remove_incomplete_special_tokens(
        text[:first_fn_pos].strip() if first_fn_pos >= 0 else text.strip()
    )
    return function_calls, remaining_text


def _messages_to_qwen_format(messages: list) -> list:
    return [
        Message(role=m["role"], content=[ContentItem(text=m.get("content") or "")])
        for m in messages
    ]


def trajectory_to_text(messages):
    parts = []
    step = 0
    pending_obs = None
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


def _build_curation_messages(skill_memory: SkillMemory,
                              task: str, messages: list, reward: bool,
                              retrieved_skills_text: str) -> list:
    """Build the preprocessed message list for SkillOS curation (shared by HTTP and vLLM)."""
    functions = [tool["function"] for tool in MEMORY_TOOL_SCHEMAS]
    trajectory_text = trajectory_to_text(messages)
    result_str = "Success" if reward else "Failure"

    system_messages = skill_memory.render_system_prompt(status='memorie')
    user_content = f"""# Task Context
## Task Description:
```
{task}
```

## Past Skills:
```
{retrieved_skills_text if retrieved_skills_text else "(none)"}
```

## Agent Trajectory:
```
{trajectory_text}
```

## Result:
```
{result_str}
```

# Output Format:
Your output must contain the following sections:
- Analysis: Analyze the trajectory, associated skills, and the final result. Identify what went well and what didn't.
- Tool Calls: Based on your analysis, determine whether to insert a new skill, update an existing skill, or delete an existing skill.
"""
    base_messages = system_messages + [{"role": "user", "content": user_content}]

    processed = QwenFnCallPrompt.preprocess_fncall_messages(
        messages=_messages_to_qwen_format(base_messages),
        functions=functions,
        lang='en',
        parallel_function_calls=True,
        function_choice='auto',
    )
    dict_messages = []
    for m in processed:
        content_text = "".join(ci.text for ci in m.content)
        if m.role == 'system':
            content_text = content_text.split("✿RESULT✿")[0].strip()
        dict_messages.append({"role": m.role, "content": content_text})
    return dict_messages


def _build_curation_prompt(skill_memory: SkillMemory, curation_tokenizer,
                            task: str, messages: list, reward: bool,
                            retrieved_skills_text: str) -> str:
    """Tokenize curation messages into a raw string for local vLLM inference."""
    dict_messages = _build_curation_messages(
        skill_memory, task, messages, reward, retrieved_skills_text
    )
    return curation_tokenizer.apply_chat_template(
        dict_messages, tokenize=False, add_generation_prompt=True
    )


def _apply_curation_output(skill_memory: SkillMemory, raw: str, label: str = ""):
    function_calls, _ = _parse_function_calls_from_text(raw)
    print(function_calls)
    for fc in function_calls:
        try:
            arguments = json.loads(fc["arguments"]) or {}
        except json.JSONDecodeError:
            try:
                arguments = json.loads(fc["arguments"].replace('\n', '\\n').replace('\r', '\\r')) or {}
            except json.JSONDecodeError:
                arguments = {}
        result = execute_tool(skill_memory, fc["name"], arguments)
        print(f"[SkillCuration{label}] {fc['name']}({arguments.get('skill_name', '')}) "
              f"-> {result['status']}: {result.get('message', '')}")


def curation_llm(messages: list, curation_model: str, curation_base_url: str) -> str:
    """LLM call for curation — Vertex AI for gemini/ models, HTTP otherwise."""
    if curation_model.startswith("gemini/"):
        from google import genai
        from google.genai import types
        model_id = curation_model[len("gemini/"):]
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_text  = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        client = genai.Client(
            vertexai=True,
            project=os.environ["GOOGLE_CLOUD_PROJECT"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
        )
        response = client.models.generate_content(
            model=model_id,
            contents=user_text,
            config=types.GenerateContentConfig(
                temperature=0.7,
                system_instruction=system_msg,
            ),
        )
        return response.text or ""
    response = completion(
        model=curation_model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=curation_base_url,
        num_retries=10,
        temperature=0.7,
    )
    return response.choices[0].message.content or ""


def batch_update_skills_from_trajectories(
    skill_memory: SkillMemory,
    curation_tokenizer,
    curation_model_hf,
    batch_data: list,  # list of (task, messages, reward, retrieved_skills_text)
    curation_model: str = None,
    curation_base_url: str = None,
):
    use_http = curation_base_url is not None or (
        curation_model is not None and curation_model.startswith("gemini/")
    )

    if use_http:
        # Build preprocessed message lists for each item
        all_messages = []
        for task, messages, reward, retrieved_skills_text in batch_data:
            try:
                msgs = _build_curation_messages(
                    skill_memory, task, messages, reward, retrieved_skills_text
                )
                all_messages.append(msgs)
            except Exception as e:
                print(f"[SkillCuration] Prompt build failed: {e}")
                all_messages.append(None)

        valid_indices = [i for i, m in enumerate(all_messages) if m is not None]
        if not valid_indices:
            return

        # Parallel HTTP calls
        raw_outputs = [None] * len(all_messages)
        with ThreadPoolExecutor(max_workers=len(valid_indices)) as executor:
            futures = {
                executor.submit(curation_llm, all_messages[i], curation_model, curation_base_url): i
                for i in valid_indices
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    raw_outputs[i] = future.result()
                except Exception as e:
                    print(f"[SkillCuration] HTTP call failed for item {i}: {e}")

        for i in valid_indices:
            if raw_outputs[i]:
                _apply_curation_output(skill_memory, raw_outputs[i], label=f" game={i}")

    else:
        # Local vLLM path
        texts = []
        for task, messages, reward, retrieved_skills_text in batch_data:
            try:
                text = _build_curation_prompt(
                    skill_memory, curation_tokenizer,
                    task, messages, reward, retrieved_skills_text,
                )
                texts.append(text)
            except Exception as e:
                print(f"[SkillCuration] Prompt build failed: {e}")
                texts.append(None)

        valid_indices = [i for i, t in enumerate(texts) if t is not None]
        valid_texts   = [texts[i] for i in valid_indices]

        if not valid_texts:
            return

        try:
            sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)
            outputs = curation_model_hf.generate(valid_texts, sampling_params)
        except Exception as e:
            print(f"[SkillCuration] vLLM batch inference failed: {e}")
            return

        for orig_idx, output in zip(valid_indices, outputs):
            raw = output.outputs[0].text
            _apply_curation_output(skill_memory, raw, label=f" game={orig_idx}")


# ------------------------------------------------------------------ #
# Memory initialisation                                               #
# ------------------------------------------------------------------ #

def init_memory(args, storage_path, env_name='alfworld'):
    """
    Initialise the memory object for the chosen memory_type.
    Returns (memory_obj, curation_tokenizer, curation_model_hf).

    When --curation_base_url is provided, curation runs via HTTP (no vLLM loaded);
    curation_model_hf will be None. The tokenizer is still loaded for SkillOS
    prompt preprocessing (QwenFnCallPrompt requires it).
    """
    if args.memory_type == 'none':
        return None, None, None

    is_reasoning_env = env_name in REASONING_ENVS
    use_gemini_curator = args.curation_model.startswith("gemini/")
    use_http = bool(getattr(args, 'curation_base_url', None)) or use_gemini_curator

    # SkillOS needs QwenFnCallPrompt + tokenizer for prompt preprocessing regardless of backend
    if args.memory_type == 'skillos':
        global AutoTokenizer, QwenFnCallPrompt, Message, ContentItem
        from transformers import AutoTokenizer as _AutoTokenizer
        from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt as _QFP
        from qwen_agent.llm.schema import Message as _Message, ContentItem as _ContentItem
        AutoTokenizer = _AutoTokenizer
        QwenFnCallPrompt, Message, ContentItem = _QFP, _Message, _ContentItem
        curation_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    else:
        curation_tokenizer = None

    if use_http:
        curation_model_hf = None
        if use_gemini_curator:
            print(f"Curation via Vertex AI: {args.curation_model}")
        else:
            print(f"Curation via HTTP: {args.curation_model} @ {args.curation_base_url}")
    else:
        global LLM, SamplingParams
        from vllm import LLM as _LLM, SamplingParams as _SP
        LLM, SamplingParams = _LLM, _SP
        if curation_tokenizer is None:
            from transformers import AutoTokenizer as _AutoTokenizer
            curation_tokenizer = _AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        print(f"Loading curation model: {args.curation_model}")
        curation_model_hf = LLM(model=args.curation_model, tokenizer="Qwen/Qwen3-8B", dtype="bfloat16")
        print("Curation model loaded.")

    curation_base_url = getattr(args, 'curation_base_url', None)

    if args.memory_type == 'skillos':
        skill_memory = SkillMemory()
        load_skills(skill_memory, storage_path)
        print(f"Loaded {len(skill_memory.skills)} existing skills from {storage_path}")
        return skill_memory, curation_tokenizer, curation_model_hf

    elif args.memory_type == 'reasoningbank':
        if is_reasoning_env:
            from reasoningbank import ReasoningBank
            bank = ReasoningBank(
                storage_path=storage_path,
                embedding_path=storage_path.replace('.jsonl', '_embeddings.jsonl'),
                retrieve_num=args.retrieve_num,
                curation_model_hf=curation_model_hf,
                curation_tokenizer=curation_tokenizer,
                curation_model_name=args.curation_model,
                curation_base_url=curation_base_url,
            )
            print(f"ReasoningBank ({'HTTP' if use_http else 'vLLM'}) initialised at {storage_path}")
        else:
            from reasoningbank_alfworld import ReasoningBankAlfworld
            bank = ReasoningBankAlfworld(
                storage_path=storage_path,
                curation_model_hf=curation_model_hf,
                curation_tokenizer=curation_tokenizer,
                retrieve_num=args.retrieve_num,
                curation_model_name=args.curation_model,
                curation_base_url=curation_base_url,
            )
            print(f"ReasoningBankAlfworld ({'HTTP' if use_http else 'vLLM'}) initialised at {storage_path}")
        return bank, curation_tokenizer, curation_model_hf

    raise ValueError(f"Unknown memory_type: {args.memory_type}")


# ------------------------------------------------------------------ #
# Memory update helpers                                               #
# ------------------------------------------------------------------ #

def update_memory_after_batch(
    memory_type, memory_obj,
    curation_tokenizer, curation_model_hf,
    batch_results, task_descriptions, skills_context,
    skills_storage_path, env_name,
    curation_model=None, curation_base_url=None,
):
    """Update memory after a game batch and persist to disk."""
    if memory_type == 'none' or memory_obj is None:
        return

    if memory_type == 'skillos':
        batch_data = [
            (task_desc, result['messages'], bool(result['reward']), skills_context.get(i, ""))
            for i, (result, task_desc) in enumerate(zip(batch_results, task_descriptions))
        ]
        batch_update_skills_from_trajectories(
            skill_memory=memory_obj,
            curation_tokenizer=curation_tokenizer,
            curation_model_hf=curation_model_hf,
            batch_data=batch_data,
            curation_model=curation_model,
            curation_base_url=curation_base_url,
        )
        save_skills(memory_obj, skills_storage_path)
        print(f"Skills saved: {len(memory_obj.skills)} total skills.")

    elif memory_type == 'reasoningbank':
        for result, task_desc in zip(batch_results, task_descriptions):
            task_id = result.get('name', task_desc[:40]).replace('/', '_')
            memory_obj.add(
                task_id=task_id,
                task=task_desc,
                messages=result['messages'],
                reward=bool(result['reward']),
            )
        # ReasoningBankAlfworld persists internally via its storage_path


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main(args):
    model_name  = args.model
    memory_type = args.memory_type

    # ---- Path setup ----
    if args.env == 'alfworld':
        output_path         = (f'Alfworld/results/{model_name}/'
                               f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_{memory_type}')
        skills_storage_path = (
            f'Alfworld/memory/skillos_{args.exp_name}/skills.json'
            if memory_type == 'skillos' else
            f'Alfworld/memory/reasoningbank_{args.exp_name}/reasoning_bank.jsonl'
        )
    elif args.env in REASONING_ENVS:
        output_path         = (f'Reasoning/results/{args.env}/{model_name}/'
                               f'{args.exp_name}_{memory_type}')
        skills_storage_path = (
            f'Reasoning/memory/{args.env}/skillos_{args.exp_name}/skills.json'
            if memory_type == 'skillos' else
            f'Reasoning/memory/{args.env}/reasoningbank_{args.exp_name}/reasoning_bank.jsonl'
        )
    else:  # webshop
        output_path         = (f'Webshop/results/{model_name}/'
                               f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_{memory_type}')
        skills_storage_path = f'Webshop/memory/skillos_{args.exp_name}/skills.json'

    os.makedirs(output_path, exist_ok=True)

    # ---- Reasoning: single-turn inference (with optional memory) ----
    if args.env in REASONING_ENVS:
        problems = load_reasoning_dataset(args.env)
        if args.num_games > 0:
            problems = problems[:args.num_games]
        finished = sum(1 for f in os.listdir(output_path) if f.endswith('.json'))
        print(f'Total problems: {len(problems)}, already finished: {finished}')

        memory_obj, curation_tokenizer, curation_model_hf = init_memory(
            args, skills_storage_path, env_name=args.env
        )

        run_reasoning(
            problems=problems,
            model=model_name,
            batch_size=args.batch_size,
            output_path=output_path,
            finished=finished,
            env_name=args.env,
            memory_type=memory_type,
            memory_obj=memory_obj,
            retrieve_num=args.retrieve_num,
            curation_tokenizer=curation_tokenizer,
            curation_model_hf=curation_model_hf,
            skills_storage_path=skills_storage_path,
            curation_model=args.curation_model,
            curation_base_url=getattr(args, 'curation_base_url', None),
        )
        return

    # ---- Validate memory_type / env combination (alfworld / webshop) ----
    if memory_type == 'reasoningbank' and args.env != 'alfworld':
        print(f"[WARNING] ReasoningBank only supports ALFWorld for interactive tasks; "
              f"env='{args.env}' will run WITHOUT memory.")
        memory_type = 'none'

    # ---- Initialise memory ----
    memory_obj, curation_tokenizer, curation_model_hf = init_memory(
        args, skills_storage_path, env_name=args.env
    )

    # ---- Context label for prompt templates ----
    context_label = "Skills" if memory_type == 'skillos' else "Memories"

    # ---- Resume accounting ----
    finished_games = 0
    all_reward     = 0.0
    for file in os.listdir(output_path):
        if file.endswith('.json'):
            finished_games += 1
            with open(f'{output_path}/{file}', 'r') as f:
                all_reward += json.load(f)['reward']

    # ==================================================================
    # ALFWorld loop
    # ==================================================================
    if args.env == 'alfworld':
        num_games_to_run = num_games  # set in __main__ block

        for idx in tqdm(range(math.ceil(num_games_to_run / env.batch_size))):
            ob_list, info = env.reset()
            if idx * env.batch_size + env.batch_size <= finished_games:
                continue

            task_descriptions = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]
            ob_list           = ['\n'.join(ob.split('\n\n')[1:2]) for ob in ob_list]
            admissible_commands = info['admissible_commands']

            # Retrieve context for each game
            skills_context = {}
            if memory_obj is not None:
                for i, query in enumerate(task_descriptions):
                    ctx = retrieve_context(memory_type, memory_obj, query, args.retrieve_num)
                    if ctx:
                        skills_context[i] = ctx

            name_list = [
                '/'.join(info['extra.gamefile'][i].split('/')[-3:-1])
                for i in range(len(ob_list))
            ]

            batch_results = alfworld_run_batch(
                env=env,
                obs=ob_list,
                names=name_list,
                task_descriptions=task_descriptions,
                admissible_commands=admissible_commands,
                max_steps=args.max_steps,
                model=model_name,
                skills_context=skills_context,
                context_label=context_label,
            )

            for result in batch_results:
                all_reward     += result['reward']
                finished_games += 1
            tqdm.write(f'Avg reward: {all_reward / finished_games:.3f}')

            for i, result in enumerate(batch_results):
                with open(f'{output_path}/idx_{idx * env.batch_size + i}.json', 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

            print(f'Finished {idx * env.batch_size + len(batch_results)} games')

            update_memory_after_batch(
                memory_type, memory_obj,
                curation_tokenizer, curation_model_hf,
                batch_results, task_descriptions, skills_context,
                skills_storage_path, args.env,
                curation_model=args.curation_model,
                curation_base_url=getattr(args, 'curation_base_url', None),
            )

    # ==================================================================
    # WebShop loop
    # ==================================================================
    else:
        import importlib
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webshop'))
        WebshopMultiProcessEnv = importlib.import_module('envs').WebshopMultiProcessEnv

        is_train = (args.split == 'train')
        webshop_env = WebshopMultiProcessEnv(
            seed=42,
            env_num=args.batch_size,
            group_n=1,
            resources_per_worker={'num_cpus': 1},
            is_train=is_train,
            env_kwargs={'observation_mode': 'text'},
        )
        webshop_manager = _make_webshop_manager(webshop_env)

        num_webshop_games = len(webshop_env.goal_idxs)
        if args.num_games > 0:
            num_webshop_games = min(args.num_games, num_webshop_games)
        print(f"Total WebShop games: {num_webshop_games}")

        for batch_idx in tqdm(range(math.ceil(num_webshop_games / args.batch_size))):
            start  = batch_idx * args.batch_size
            end    = min(start + args.batch_size, num_webshop_games)
            real_n = end - start

            if end <= finished_games:
                continue

            indices = list(range(start, end))
            if real_n < args.batch_size:
                indices += [indices[-1]] * (args.batch_size - real_n)

            obs_dict, _ = webshop_manager.reset(indices=indices)
            task_descriptions = webshop_manager.tasks

            skills_context = {}
            if memory_obj is not None:
                for i in range(real_n):
                    ctx = retrieve_context(memory_type, memory_obj, task_descriptions[i], args.retrieve_num)
                    if ctx:
                        skills_context[i] = ctx

            batch_results = webshop_run_batch(
                manager=webshop_manager,
                obs_dict=obs_dict,
                max_steps=args.max_steps,
                model=model_name,
                skills_context=skills_context,
            )

            for i in range(real_n):
                result          = batch_results[i]
                all_reward     += result['reward']
                finished_games += 1
                with open(f'{output_path}/idx_{start + i}.json', 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

            tqdm.write(f'Avg reward: {all_reward / finished_games:.3f}')
            print(f'Finished {finished_games} games')

            update_memory_after_batch(
                memory_type, memory_obj,
                curation_tokenizer, curation_model_hf,
                batch_results[:real_n], task_descriptions[:real_n], skills_context,
                skills_storage_path, args.env,
                curation_model=args.curation_model,
                curation_base_url=getattr(args, 'curation_base_url', None),
            )


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified MemP runner (all envs × all memory types)')
    parser.add_argument('--model',          type=str,  default='openai/Qwen/Qwen2.5-7B-Instruct',
                        help='LLM model for gameplay (litellm format)')
    parser.add_argument('--env',            type=str,  default='alfworld',
                        choices=['alfworld', 'webshop', 'amc23', 'aime24', 'aime25', 'gpqa'],
                        help='Task environment')
    parser.add_argument('--memory_type',    type=str,  default='none',
                        choices=['none', 'skillos', 'reasoningbank'],
                        help='Memory mechanism to use')
    parser.add_argument('--split',          type=str,  default='dev',
                        choices=['dev', 'test', 'train'],
                        help='Dataset split')
    parser.add_argument('--batch_size',     type=int,  default=10,
                        help='Number of parallel tasks per batch')
    parser.add_argument('--max_steps',      type=int,  default=30,
                        help='Maximum steps per task episode')
    parser.add_argument('--exp_name',       type=str,  default='exp',
                        help='Experiment name (used in output/memory paths)')
    parser.add_argument('--few_shot',       action='store_true',
                        help='Enable few-shot examples (name tag only; inject manually in templates if needed)')
    parser.add_argument('--retrieve_num',   type=int,  default=3,
                        help='Number of memory items to retrieve per query')
    parser.add_argument('--curation_model', type=str,  default='Qwen/Qwen3-8B',
                        help='Model name for memory curation (skillos / reasoningbank)')
    parser.add_argument('--curation_base_url', type=str, default=None,
                        help='OpenAI-compatible API base URL for curation model. '
                             'When set, uses the API instead of loading vLLM locally.')
    parser.add_argument('--num_games',      type=int,  default=0,
                        help='Limit number of games to run (0 = all)')
    parser.add_argument('--overwrite',      action='store_true',
                        help='Delete existing results before running')
    args = parser.parse_args()

    # ---- Overwrite existing results ----
    if args.overwrite and args.env not in REASONING_ENVS:
        result_dir = (
            f'Alfworld/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_{args.memory_type}'
            if args.env == 'alfworld' else
            f'Webshop/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_{args.memory_type}'
        )
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                os.remove(f'{result_dir}/{file}')
            print(f'Cleared existing results in {result_dir}')

    # ---- Set up ALFWorld environment (only when needed) ----
    if args.env == 'alfworld':
        import alfworld
        import alfworld.agents.environment
        from alfworld.agents.environment import get_environment

        with open('Alfworld/base_config.yaml') as reader:
            config = yaml.safe_load(reader)

        split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"
        env       = get_environment(config["env"]["type"])(config, train_eval=split)
        env       = env.init_env(batch_size=args.batch_size)
        num_games = len(env.gamefiles)
        if args.num_games > 0:
            num_games = min(args.num_games, num_games)
        print(f"Total ALFWorld games: {num_games}")

    main(args)
