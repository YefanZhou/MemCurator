import os
import sys
import json
import math
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import re
import openai
import yaml
from litellm import completion

# Heavy imports deferred: only loaded when needed
LLM = None
SamplingParams = None
AutoTokenizer = None
QwenFnCallPrompt = None
Message = None
ContentItem = None

# Add SkillOS directory to path so we can import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'SkillOS'))
from skills_memory import SkillMemory

# Qwen function-call special tokens (mirrors skills_agent.py)
FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'


os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
openai.api_key = os.environ["OPENAI_API_KEY"]

HISTORY_LENGTH = 5

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

ALFWORLD_TEMPLATE_WITH_SKILLS = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Past Relevant Skills

{retrieved_skills}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant skills. This reasoning process MUST be enclosed within <think> </think> tags.
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

WEBSHOP_TEMPLATE_WITH_SKILLS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. Your task is to: {task_description}.

## Past Relevant Skills

{retrieved_skills}

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

# ------------------------------------------------------------------ #
# Inline tool schemas (avoids skills.skills_functions import issues)  #
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
                    "content": {"type": "string", "description": "The markdown content for the new skill."}
                },
                "required": ["skill_name", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "skill_update",
            "description": "If the existing skill can be improved, update the specific skill by its skill_name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "The name of the skill to update. Must exactly match an existing skill title."},
                    "new_name": {"type": "string", "description": "The new skill name (optional)."},
                    "new_content": {"type": "string", "description": "The new full content for the skill (optional)."}
                },
                "required": ["skill_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "skill_delete",
            "description": "Delete an existing skill by its title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "The name of the skill to delete."}
                },
                "required": ["skill_name"]
            }
        }
    }
]


def execute_tool(skill_memory: SkillMemory, tool_name: str, arguments: dict) -> dict:
    """Execute a skill memory tool by name."""
    if tool_name == "new_skill_insert":
        try:
            title = skill_memory.new_memory_insert(arguments["skill_name"], arguments["content"])
            return {"status": "ok", "message": f"Skill created successfully.", "skill_name": title}
        except (ValueError, Exception) as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "skill_update":
        try:
            updated = skill_memory.memory_update(
                title=arguments["skill_name"],
                new_name=arguments.get("new_name"),
                new_content=arguments.get("new_content")
            )
            return {"status": "ok", "message": f"Skill updated.", "updated_skill": updated}
        except (ValueError, Exception) as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "skill_delete":
        try:
            skill_memory.memory_delete(arguments["skill_name"])
            return {"status": "ok", "message": f"Skill deleted."}
        except (ValueError, Exception) as e:
            return {"status": "error", "message": str(e)}
    else:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}


# ------------------------------------------------------------------ #
# Skill persistence                                                   #
# ------------------------------------------------------------------ #

def save_skills(skill_memory: SkillMemory, storage_path: str):
    """Persist in-memory skills to a JSON file."""
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    with open(storage_path, "w", encoding="utf-8") as f:
        json.dump(skill_memory.skills, f, indent=2, ensure_ascii=False)


def load_skills(skill_memory: SkillMemory, storage_path: str):
    """Load skills from a JSON file into the SkillMemory instance."""
    if not os.path.exists(storage_path):
        return
    with open(storage_path, "r", encoding="utf-8") as f:
        skill_memory.skills = json.load(f)


# ------------------------------------------------------------------ #
# LLM helpers                                                         #
# ------------------------------------------------------------------ #

def llm_vertexai(prompt, model="gemini-3.1-pro-preview"):
    """Call Gemini via Vertex AI SDK directly."""
    from google import genai
    from google.genai import types
    if isinstance(prompt, list):
        # Convert message list to a single user turn (Gemini expects Contents)
        text = "\n".join(
            m["content"] for m in prompt if m.get("role") != "system"
        )
    elif isinstance(prompt, str):
        text = prompt
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
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
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
    if model.startswith("gemini/"):
        return llm_vertexai(prompt, model=model[len("gemini/"):])
    response = completion(
        model=model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['OPENAI_API_BASE'],
        num_retries=10,
        temperature=0.7,
        stop=stop
    )
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    return "Output Error"


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


def get_example(name, examples_list):
    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    for k, v in prefixes.items():
        if name.startswith(k):
            for example in examples_list:
                if example['task'] == v:
                    return example['example']
    assert False, f'{name} not found'


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
# Skill retrieval and injection                                       #
# ------------------------------------------------------------------ #

def get_skills_text(skill_memory: SkillMemory, query: str, retrieve_num: int) -> str:
    """Retrieve relevant skills for a query and return as formatted text."""
    if not skill_memory.skills:
        return ""
    results = skill_memory.memory_search(query=query, top_k=retrieve_num, search_method="bm25")
    if not results:
        return ""
    parts = []
    for idx, (skill, _) in enumerate(results):
        parts.append(f"**Skill {idx + 1}: {skill['title']}**\n{skill['content']}")
    return "\n\n---\n\n".join(parts)


# ------------------------------------------------------------------ #
# ALFWorld batch runner                                               #
# ------------------------------------------------------------------ #

def alfworld_run_batch(obs, names, task_descriptions, admissible_commands,
                       max_steps=30, model="openai/Qwen/Qwen2.5-7B-Instruct",
                       skills_context=None):
    """
    Run a batch of ALFWorld tasks using structured per-step prompts with a sliding
    history window.

    skills_context: dict mapping game index -> skills text to inject via template
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
            skills_text = skills_context.get(idx, "") if skills_context else ""

            if step_count == 0:
                prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=current_obs[idx],
                    admissible_actions=admissible_str,
                )
            elif skills_text:
                prompt_text = ALFWORLD_TEMPLATE_WITH_SKILLS.format(
                    task_description=task_descriptions[idx],
                    retrieved_skills=skills_text,
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

    # Reconstruct flat message list for curation and saving
    results = []
    for idx in range(n):
        messages = []
        for obs_h, act_h in histories[idx]:
            messages.append({"role": "user", "content": obs_h})
            messages.append({"role": "assistant", "content": act_h})
        messages.append({"role": "user", "content": current_obs[idx]})
        results.append({"messages": messages, "reward": task_rewards[idx], "name": names[idx]})

    return results


# ------------------------------------------------------------------ #
# WebShop batch runner (uses WebshopEnvironmentManager from          #
# agent_system/environments/env_manager.py for obs formatting,       #
# action extraction, and history management)                         #
# ------------------------------------------------------------------ #

def _make_webshop_manager(raw_env):
    """Wrap a WebshopMultiProcessEnv in WebshopEnvironmentManager."""
    from types import SimpleNamespace
    sys.path.insert(0, os.path.dirname(__file__))
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection
    from functools import partial

    cfg = SimpleNamespace(env=SimpleNamespace(history_length=HISTORY_LENGTH))

    class _Manager(WebshopEnvironmentManager):
        """Subclass that supports reset(indices=...) for sequential batching."""
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
                'text': self.build_text_obs(obs, infos, init=True),
                'image': None,
                'anchor': obs.copy(),
            }
            return observations, infos

    return _Manager(raw_env, partial(webshop_projection), cfg)


def webshop_run_batch(manager, obs_dict, max_steps=30,
                      model="openai/Qwen/Qwen2.5-7B-Instruct",
                      skills_context=None):
    """
    Run a batch of WebShop tasks.

    manager : _Manager instance (already reset; obs_dict from manager.reset())
    obs_dict: {'text': [prompt, ...], ...} returned by manager.reset()
    """
    n = len(obs_dict['text'])
    prompts = list(obs_dict['text'])   # formatted prompts from manager
    task_rewards = [0.0] * n
    dones_arr = [False] * n
    histories = [[] for _ in range(n)]  # (prompt, raw_response) per step

    for _ in range(max_steps):
        active = [i for i in range(n) if not dones_arr[i]]
        if not active:
            break
        print(f'\033[91mActive tasks: {active}\033[0m')

        responses = [""] * n
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {
                executor.submit(llm, [{"role": "user", "content": prompts[i]}], None, model): i
                for i in active
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    responses[i] = future.result()
                    print(f'\033[92mAgent {i}: \n{responses[i]}\033[0m')
                except Exception as e:
                    print(f'Error {i}: {e}')

        for i in active:
            histories[i].append((prompts[i], responses[i]))

        # manager handles action extraction (<action>...</action>) + env.step
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
            messages.append({"role": "user", "content": obs_h})
            messages.append({"role": "assistant", "content": act_h})
        messages.append({"role": "user", "content": prompts[i]})
        results.append({"messages": messages, "reward": task_rewards[i]})
    return results


# ------------------------------------------------------------------ #
# Reasoning benchmarks (AMC23, AIME24, AIME25, GPQA) — single turn   #
# ------------------------------------------------------------------ #

REASONING_TEMPLATE = "{question}\n\nPlease reason step by step and put final answer within \\boxed{{}}."

GPQA_TEMPLATE = "{question}\n\nChoices:\n(A) {choice_a}\n(B) {choice_b}\n(C) {choice_c}\n(D) {choice_d}\n\nPlease reason step by step and put final answer within \\boxed{{}}."

REASONING_DATASETS = {
    'amc23':  ('math-ai/amc23',  None),
    'aime24': ('math-ai/aime24', None),
    'aime25': ('math-ai/aime25', None),
    'gpqa':   ('Idavidrein/gpqa', 'gpqa_diamond'),
}


def load_reasoning_dataset(env_name):
    """Load problem list as [{'question': ..., 'answer': ...}, ...]."""
    problems = []
    if env_name == 'gpqa':
        import random, pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), 'gpqa_diamond.csv')
        data = pd.read_csv(csv_path).to_dict('records')
        # GPQA has: 'Question', 'Correct Answer', 'Incorrect Answer 1/2/3'
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
            problems.append({'question': prompt, 'answer': correct_label})
    else:
        from datasets import load_dataset
        hf_name, subset = REASONING_DATASETS[env_name]
        ds = load_dataset(hf_name, subset, trust_remote_code=True)
        split = 'test' if 'test' in ds else list(ds.keys())[0]
        # Per-dataset column mapping
        # aime24: problem / solution (answer is \boxed{N})
        # aime25: problem / answer
        # amc23:  question / answer
        col_q = {'aime24': 'problem', 'aime25': 'problem', 'amc23': 'question'}[env_name]
        col_a = {'aime24': 'solution', 'aime25': 'answer',  'amc23': 'answer'}[env_name]
        for row in ds[split]:
            problems.append({
                'question': REASONING_TEMPLATE.format(question=row[col_q]),
                'answer': str(row[col_a]),
            })

    return problems


def score_reasoning(pred_text, gold, env_name):
    """Return 1.0 if correct, else 0.0."""
    if env_name == 'gpqa':
        import re as _re
        matches = _re.findall(r'\\boxed\{([^{}]*)\}', pred_text)
        if not matches:
            matches = _re.findall(r'\\boxed\{\\[a-z]+\{([^{}]*)\}\}', pred_text)
        raw = matches[-1].strip() if matches else ''
        letter = _re.search(r'\b([A-D])\b', raw, _re.IGNORECASE)
        pred_letter = letter.group(1).upper() if letter else raw.upper()
        return 1.0 if pred_letter == gold.strip().upper() else 0.0
    else:
        # Use math-verify for AIME/AMC
        from math_verify import parse, verify
        try:
            return 1.0 if verify(parse(gold), parse(pred_text)) else 0.0
        except Exception:
            return 0.0


def run_reasoning(problems, model, batch_size, output_path, finished, env_name=''):
    """Run single-turn inference on all problems, saving one JSON per problem."""
    all_reward = 0.0

    for i in tqdm(range(0, len(problems), batch_size)):
        batch = problems[i:i + batch_size]
        # Skip already-finished problems
        if i + len(batch) <= finished:
            for j in range(len(batch)):
                fpath = f'{output_path}/idx_{i+j}.json'
                if os.path.exists(fpath):
                    all_reward += json.load(open(fpath))['reward']
            continue

        responses = [''] * len(batch)
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    llm,
                    [{"role": "user", "content": batch[j]['question']}],
                    None, model
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

        for j, (prob, resp) in enumerate(zip(batch, responses)):
            if i + j < finished:
                continue
            reward = score_reasoning(resp, prob['answer'], env_name)
            result = {
                'messages': [
                    {"role": "user",      "content": prob['question']},
                    {"role": "assistant", "content": resp},
                ],
                'answer': prob['answer'],
                'reward': reward,
            }
            with open(f'{output_path}/idx_{i+j}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            all_reward += reward

        done = min(i + len(batch), len(problems))
        tqdm.write(f'Avg accuracy: {all_reward / done * 100:.2f}%  [{done}/{len(problems)}]')

    print(f'\nFinal accuracy: {all_reward / len(problems) * 100:.2f}%  ({int(all_reward)}/{len(problems)})')


# ------------------------------------------------------------------ #
# Skill curation via tool calling                                     #
# ------------------------------------------------------------------ #

def trajectory_to_text(messages):
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


# ------------------------------------------------------------------ #
# Qwen function-call parsing (mirrors skills_agent.py)               #
# ------------------------------------------------------------------ #

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
    """Parse ✿FUNCTION✿/✿ARGS✿ tokens from raw Qwen output. Returns (function_calls, remaining_text)."""
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
    """Convert plain dict messages to QwenFnCallPrompt-processed dict messages."""
    qwen_msgs = [
        Message(role=m["role"], content=[ContentItem(text=m.get("content") or "")])
        for m in messages
    ]
    return qwen_msgs


def _build_curation_prompt(skill_memory: SkillMemory, curation_tokenizer,
                           task: str, messages: list, reward: bool,
                           retrieved_skills_text: str) -> str:
    """Build the chat-template string for one curation request."""
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

    return curation_tokenizer.apply_chat_template(
        dict_messages, tokenize=False, add_generation_prompt=True
    )


def _apply_curation_output(skill_memory: SkillMemory, raw: str, label: str = ""):
    """Parse ✿FUNCTION✿/✿ARGS✿ calls from raw output and execute them."""
    function_calls, _ = _parse_function_calls_from_text(raw)

    print(function_calls)
    for fc in function_calls:
        try:
            arguments = json.loads(fc["arguments"]) or {}
        except json.JSONDecodeError:
            try:
                # LLM sometimes emits literal newlines inside JSON string values
                arguments = json.loads(fc["arguments"].replace('\n', '\\n').replace('\r', '\\r')) or {}
            except json.JSONDecodeError:
                arguments = {}
        result = execute_tool(skill_memory, fc["name"], arguments)
        print(f"[SkillCuration{label}] {fc['name']}({arguments.get('skill_name', '')}) "
              f"-> {result['status']}: {result.get('message', '')}")


def batch_update_skills_from_trajectories(
    skill_memory: SkillMemory,
    curation_tokenizer,
    curation_model_hf,
    batch_data: list,  # list of (task, messages, reward, retrieved_skills_text)
):
    """
    Batch skill curation for an entire game batch via a single vLLM call.
    Updates skill_memory in place sequentially after generation.
    """
    # Build all prompts
    texts = []
    for task, messages, reward, retrieved_skills_text in batch_data:
        try:
            text = _build_curation_prompt(
                skill_memory, curation_tokenizer,
                task, messages, reward, retrieved_skills_text
            )
            texts.append(text)
        except Exception as e:
            print(f"[SkillCuration] Prompt build failed: {e}")
            texts.append(None)

    valid_indices = [i for i, t in enumerate(texts) if t is not None]
    valid_texts = [texts[i] for i in valid_indices]

    if not valid_texts:
        return

    # Single batched vLLM call
    try:
        sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)
        outputs = curation_model_hf.generate(valid_texts, sampling_params)
    except Exception as e:
        print(f"[SkillCuration] vLLM batch inference failed: {e}")
        return

    # Apply results sequentially (skill memory must be updated in order)
    for orig_idx, output in zip(valid_indices, outputs):
        raw = output.outputs[0].text
        _apply_curation_output(skill_memory, raw, label=f" game={orig_idx}")


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main(args):
    model_name = args.model

    REASONING_ENVS = {'amc23', 'aime24', 'aime25', 'gpqa'}

    if args.env == 'alfworld':
        output_path = f'Alfworld/results/{model_name}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_skillos_{args.use_memory}'
        skills_storage_path = f'Alfworld/memory/skillos_{args.exp_name}/skills.json'
    elif args.env in REASONING_ENVS:
        output_path = f'Reasoning/results/{args.env}/{model_name}/{args.exp_name}'
        skills_storage_path = None
    else:  # webshop
        output_path = f'Webshop/results/{model_name}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_skillos_{args.use_memory}'
        skills_storage_path = f'Webshop/memory/skillos_{args.exp_name}/skills.json'
    os.makedirs(output_path, exist_ok=True)

    # Reasoning benchmarks: single-turn, no memory/env setup needed
    if args.env in REASONING_ENVS:
        problems = load_reasoning_dataset(args.env)
        if args.num_games > 0:
            problems = problems[:args.num_games]
        finished = sum(1 for f in os.listdir(output_path) if f.endswith('.json'))
        print(f'Total problems: {len(problems)}, already finished: {finished}')
        run_reasoning(problems, model_name, args.batch_size, output_path, finished, args.env)
        return

    # Initialize SkillMemory
    skill_memory = SkillMemory()

    # Load local curation model (only needed when using memory)
    curation_tokenizer = None
    curation_model_hf = None
    if args.use_memory:
        global LLM, SamplingParams, AutoTokenizer, QwenFnCallPrompt, Message, ContentItem
        from transformers import AutoTokenizer as _AutoTokenizer
        from vllm import LLM as _LLM, SamplingParams as _SamplingParams
        from qwen_agent.llm.fncall_prompts.qwen_fncall_prompt import QwenFnCallPrompt as _QFP
        from qwen_agent.llm.schema import Message as _Message, ContentItem as _ContentItem
        AutoTokenizer = _AutoTokenizer
        LLM, SamplingParams = _LLM, _SamplingParams
        QwenFnCallPrompt, Message, ContentItem = _QFP, _Message, _ContentItem

        curation_model_name = args.curation_model
        print(f"Loading curation model: {curation_model_name}")
        curation_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
        curation_model_hf = LLM(model=curation_model_name, tokenizer="Qwen/Qwen3-8B", dtype="bfloat16")
        print(f"Curation model loaded.")

    # Load existing skills if available
    if args.use_memory:
        load_skills(skill_memory, skills_storage_path)
        print(f"Loaded {len(skill_memory.skills)} existing skills.")

    finished_games = 0
    all_reward = 0

    # Resume from existing results
    for file in os.listdir(output_path):
        if file.endswith('.json'):
            finished_games += 1
            with open(f'{output_path}/{file}', 'r') as f:
                result = json.load(f)
                all_reward += result['reward']

    if args.env == 'alfworld':
        # ------------------------------------------------------------------ #
        # ALFWorld loop                                                        #
        # ------------------------------------------------------------------ #
        for idx in tqdm(range(math.ceil(num_games / env.batch_size))):
            ob_list, info = env.reset()
            if idx * env.batch_size + env.batch_size <= finished_games:
                continue

            task_descriptions = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]
            ob_list = ['\n'.join(ob.split('\n\n')[1:2]) for ob in ob_list]

            admissible_commands = info['admissible_commands']

            # Retrieve skills for each game
            skills_context = {}
            if args.use_memory and skill_memory.skills:
                for i, query in enumerate(task_descriptions):
                    skills_text = get_skills_text(skill_memory, query, args.retrieve_num)
                    if skills_text:
                        skills_context[i] = skills_text

            name_list = ['/'.join(info['extra.gamefile'][i].split('/')[-3:-1]) for i in range(len(ob_list))]

            batch_results = alfworld_run_batch(
                obs=ob_list,
                names=name_list,
                task_descriptions=task_descriptions,
                admissible_commands=admissible_commands,
                max_steps=args.max_steps,
                model=model_name,
                skills_context=skills_context,
            )

            for result in batch_results:
                all_reward += result['reward']
                finished_games += 1
            tqdm.write(f'Avg reward: {all_reward / finished_games:.3f}')

            for i, result in enumerate(batch_results):
                with open(f'{output_path}/idx_{idx * env.batch_size + i}.json', 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

            print(f'Finished {idx * env.batch_size + i + 1} games')

            if args.use_memory:
                batch_data = [
                    (task_desc, result['messages'], bool(result['reward']), skills_context.get(i, ""))
                    for i, (result, task_desc) in enumerate(zip(batch_results, task_descriptions))
                ]
                batch_update_skills_from_trajectories(
                    skill_memory=skill_memory,
                    curation_tokenizer=curation_tokenizer,
                    curation_model_hf=curation_model_hf,
                    batch_data=batch_data,
                )
                save_skills(skill_memory, skills_storage_path)
                print(f"Skills saved: {len(skill_memory.skills)} total skills.")

    else:
        # ------------------------------------------------------------------ #
        # WebShop loop                                                         #
        # ------------------------------------------------------------------ #
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
            start = batch_idx * args.batch_size
            end = min(start + args.batch_size, num_webshop_games)
            real_n = end - start

            if end <= finished_games:
                continue

            # Pad last batch to match env_num
            indices = list(range(start, end))
            if real_n < args.batch_size:
                indices += [indices[-1]] * (args.batch_size - real_n)

            obs_dict, _ = webshop_manager.reset(indices=indices)
            task_descriptions = webshop_manager.tasks  # set by manager during reset

            # Retrieve skills for real games only
            skills_context = {}
            if args.use_memory and skill_memory.skills:
                for i in range(real_n):
                    skills_text = get_skills_text(skill_memory, task_descriptions[i], args.retrieve_num)
                    if skills_text:
                        skills_context[i] = skills_text

            batch_results = webshop_run_batch(
                manager=webshop_manager,
                obs_dict=obs_dict,
                max_steps=args.max_steps,
                model=model_name,
                skills_context=skills_context,
            )

            for i in range(real_n):
                result = batch_results[i]
                all_reward += result['reward']
                finished_games += 1
                with open(f'{output_path}/idx_{start + i}.json', 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

            tqdm.write(f'Avg reward: {all_reward / finished_games:.3f}')
            print(f'Finished {finished_games} games')

            if args.use_memory:
                batch_data = [
                    (task_descriptions[i], batch_results[i]['messages'],
                     bool(batch_results[i]['reward']), skills_context.get(i, ""))
                    for i in range(real_n)
                ]
                batch_update_skills_from_trajectories(
                    skill_memory=skill_memory,
                    curation_tokenizer=curation_tokenizer,
                    curation_model_hf=curation_model_hf,
                    batch_data=batch_data,
                )
                save_skills(skill_memory, skills_storage_path)
                print(f"Skills saved: {len(skill_memory.skills)} total skills.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='openai/Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='rb')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--use_memory', action='store_true')
    parser.add_argument('--retrieve_num', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--curation_model', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--env', type=str, default='alfworld',
                        choices=['alfworld', 'webshop', 'amc23', 'aime24', 'aime25', 'gpqa'])
    parser.add_argument('--num_games', type=int, default=0, help='Limit number of games (0 = all)')
    args = parser.parse_args()

    REASONING_ENVS = {'amc23', 'aime24', 'aime25', 'gpqa'}

    if args.overwrite and args.env not in REASONING_ENVS:
        result_dir = (
            f'Alfworld/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_skillos_{args.use_memory}'
            if args.env == 'alfworld' else
            f'Webshop/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_skillos_{args.use_memory}'
        )
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                os.remove(f'{result_dir}/{file}')

    if args.env == 'alfworld':
        import alfworld
        import alfworld.agents.environment
        from alfworld.agents.environment import get_environment

        with open('Alfworld/base_config.yaml') as reader:
            config = yaml.safe_load(reader)

        split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        env = env.init_env(batch_size=args.batch_size)
        num_games = len(env.gamefiles)
        print(f"Total games: {num_games}")

    main(args)
