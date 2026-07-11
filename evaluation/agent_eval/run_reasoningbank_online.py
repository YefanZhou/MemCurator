import os
import openai
from litellm import completion

import yaml
import alfworld
import alfworld.agents.environment
from alfworld.agents.environment import get_environment
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import argparse
import json
import math
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from reasoningbank_alfworld import ReasoningBankAlfworld

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
openai.api_key = os.environ["OPENAI_API_KEY"]

HISTORY_LENGTH = 5

ALFWORLD_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}.

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.\
"""

ALFWORLD_TEMPLATE_WITH_SKILLS = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}

## Past Relevant Memories

{retrieved_skills}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: {admissible_actions}

Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant memory items. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.\
"""


def llm_vertexai(prompt, model="gemini-2.5-pro"):
    """Call Gemini via Vertex AI SDK directly."""
    from google import genai
    from google.genai import types
    if isinstance(prompt, list):
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
        temperature=1,
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


def alfworld_run_batch(obs, names, task_descriptions, admissible_commands,
                       max_steps=30, model="openai/Qwen/Qwen2.5-7B-Instruct",
                       retrieved_skills_list=None):
    """
    Run a batch of ALFWorld tasks using structured per-step prompts with a sliding
    history window.

    History is tracked as (observation, action) pairs per task rather than growing
    the full message list — only the most recent HISTORY_LENGTH steps are shown
    in each prompt.
    """
    n = len(obs)
    # Per-task state
    histories = [[] for _ in range(n)]          # list of (obs_str, action_str) per task
    current_obs = list(obs)
    current_admissible = list(admissible_commands)
    task_rewards = [0] * n
    active_tasks = list(range(n))

    for step in range(max_steps):
        if not active_tasks:
            break
        print(f'\033[91mActive tasks: {active_tasks}\033[0m')

        # Build a fresh single-turn prompt for each active task
        prompts = {}
        for idx in active_tasks:
            history_str = format_action_history(histories[idx], HISTORY_LENGTH)
            admissible_str = ", ".join(current_admissible[idx])
            step_count = len(histories[idx])

            if retrieved_skills_list and retrieved_skills_list[idx]:
                prompt_text = ALFWORLD_TEMPLATE_WITH_SKILLS.format(
                    task_description=task_descriptions[idx],
                    retrieved_skills=retrieved_skills_list[idx],
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

        # for item in prompts.items():
        #     print(f'\033[94mPrompt for task {item[0]}: \n{item[1][0]["content"]}\033[0m')

        # Query LLM for all active tasks in parallel
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

        # Parse <action>...</action> tags
        action_list = [""] * n
        for idx in active_tasks:
            if idx in responses:
                response = responses[idx]
                if '<action>' in response and '</action>' in response:
                    action_list[idx] = response.split('<action>')[-1].split('</action>')[0].strip()

        observation, reward, done, info = env.step(action_list)
        observation = [process_ob(ob) for ob in observation]
        print(f'\033[93mObservation: \n{observation}\033[0m')
        new_admissible = info.get('admissible_commands', current_admissible)
        won = info['won']

        new_active_tasks = []
        for idx in active_tasks:
            # Append (prev_obs, action_taken) to history before moving to new obs
            histories[idx].append((current_obs[idx], action_list[idx]))
            current_obs[idx] = observation[idx]
            current_admissible[idx] = new_admissible[idx]
            if done[idx]:
                task_rewards[idx] = won[idx]
            else:
                new_active_tasks.append(idx)
        active_tasks = new_active_tasks

    # Reconstruct a flat message list for downstream saving (system + history pairs)
    results = []
    for idx in range(n):
        messages = []
        for obs_h, act_h in histories[idx]:
            messages.append({"role": "user", "content": obs_h})
            messages.append({"role": "assistant", "content": act_h})
        messages.append({"role": "user", "content": current_obs[idx]})
        results.append({"messages": messages, "reward": task_rewards[idx], "name": names[idx]})

    return results


def main(args):
    model_name = args.model

    with open('Alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"

    output_path = f'Alfworld/results/{model_name}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_reasoningbank_{args.use_memory}'
    os.makedirs(output_path, exist_ok=True)

    # Load local curation model
    print(f"Loading curation model: {args.curation_model}")
    curation_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    curation_model_hf = LLM(model=args.curation_model, tokenizer="Qwen/Qwen3-8B", dtype="bfloat16")
    print("Curation model loaded.")

    # Memory init
    if args.use_memory:
        bank = ReasoningBankAlfworld(
            storage_path=f'Alfworld/memory/reasoningbank_{args.exp_name}/reasoning_bank.jsonl',
            curation_model_hf=curation_model_hf,
            curation_tokenizer=curation_tokenizer,
            retrieve_num=args.retrieve_num,
        )

    finished_games = 0
    all_reward = 0

    # Resume from existing results
    for file in os.listdir(output_path):
        if file.endswith('.json'):
            finished_games += 1
            with open(f'{output_path}/{file}', 'r') as f:
                result = json.load(f)
                all_reward += result['reward']

    for idx in tqdm(range(math.ceil(num_games / env.batch_size))):
        ob_list, info = env.reset()
        if idx * env.batch_size + env.batch_size <= finished_games:
            continue


        task_descriptions = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]
        ob_list = ['\n'.join(ob.split('\n\n')[1:2]) for ob in ob_list]
        admissible_commands = info['admissible_commands']

        # Build retrieved skills per task if memory is enabled
        retrieved_skills_list = None
        if args.use_memory and len(bank.memory_bank) > 0:
            retrieved_skills_list = []
            for task_desc in task_descriptions:
                mem_text = bank.retrieve(task_desc)
                retrieved_skills_list.append(mem_text if mem_text else "")

        name_list = ['/'.join(info['extra.gamefile'][i].split('/')[-3:-1]) for i in range(len(ob_list))]

        batch_results = alfworld_run_batch(
            obs=ob_list,
            names=name_list,
            task_descriptions=task_descriptions,
            admissible_commands=admissible_commands,
            max_steps=args.max_steps,
            model=model_name,
            retrieved_skills_list=retrieved_skills_list,
        )

        for result in batch_results:
            all_reward += result['reward']
            finished_games += 1
        tqdm.write(f'Avg reward: {all_reward / finished_games:.3f}')

        for i, result in enumerate(batch_results):
            with open(f'{output_path}/idx_{idx * env.batch_size + i}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        print(f'Finished {idx * env.batch_size + i + 1} games')

        # Update memory bank from this batch
        if args.use_memory:
            for result, task_desc in zip(batch_results, task_descriptions):
                task_id = result['name'].replace('/', '_')
                bank.add(
                    task_id=task_id,
                    task=task_desc,
                    messages=result['messages'],
                    reward=bool(result['reward']),
                )


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
    parser.add_argument('--curation_model', type=str, default='Qwen/Qwen3-8B')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    if args.overwrite:
        result_dir = f'Alfworld/results/{args.model}/{args.split}_{args.exp_name}_few_shot_{args.few_shot}_reasoningbank_{args.use_memory}'
        if os.path.exists(result_dir):
            for file in os.listdir(result_dir):
                os.remove(f'{result_dir}/{file}')

    with open('Alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)

    split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=args.batch_size)
    num_games = len(env.gamefiles)
    print(f"Total games: {num_games}")
    main(args)
