"""
MemP on WebShop.

Mirrors run_memp_online.py but for WebShop:
  - Uses the same `Memory` class from memory.py
  - Curator LLM is controlled via --mem_model / --mem_base_url
  - Retrieval is top-1 workflow; injected as "Past Relevant Guidelines" in the
    per-step prompt (at each step after step 0, matching SkillOS injection point).
  - Memory is updated after each batch.
"""
import os
import sys
import json
import math
import argparse
import yaml
import importlib
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/temurin-21-jdk-amd64")

from litellm import completion


def llm(prompt, model):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    response = completion(
        model=model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['OPENAI_API_BASE'],
        num_retries=10,
        temperature=0.7,
    )
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    return "Output Error"


def trajectory_text(messages):
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append(f"[Observation]: {m['content']}")
        else:
            parts.append(f"[Action]: {m['content']}")
    return "\n\n".join(parts[:30])  # cap length to keep curation prompt manageable


def run_memp_webshop_batch(manager, obs_dict, max_steps, model, guidelines_context):
    """
    guidelines_context : dict mapping batch index -> guidelines text (top-1 workflow).
                         Injected as "Past Relevant Guidelines" at each step after step 0.
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
                ctx_text = guidelines_context.get(i, "") if guidelines_context else ""
                if ctx_text and step_counts[i] > 0:
                    content = f"## Past Relevant Guidelines\n\n{ctx_text}\n\n---\n\n{prompts[i]}"
                else:
                    content = prompts[i]
                futures[executor.submit(llm, [{"role": "user", "content": content}], model)] = i
            for future in as_completed(futures):
                i = futures[future]
                try:
                    responses[i] = future.result()
                    print(f'\033[92mAgent {i}: \n{responses[i][:200]}\033[0m')
                except Exception as e:
                    print(f'Error {i}: {e}')

        for i in active:
            histories[i].append((prompts[i], responses[i]))
            step_counts[i] += 1

        next_obs_dict, rewards, dones, infos = manager.step(responses)
        prompts = list(next_obs_dict['text'])

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


def make_webshop_manager(raw_env, batch_size):
    from types import SimpleNamespace
    from agent_system.environments.env_manager import WebshopEnvironmentManager
    from agent_system.environments.env_package.webshop import webshop_projection

    cfg = SimpleNamespace(env=SimpleNamespace(history_length=5))

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


def main(args):
    output_path = f'Webshop/results/{args.model}/dev_{args.exp_name}_memp'
    if args.overwrite and os.path.exists(output_path):
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
    os.makedirs(output_path, exist_ok=True)

    # Memory setup (after setting env vars)
    if args.mem_model:
        os.environ["MODEL_NAME"] = args.mem_model
    if args.mem_base_url:
        os.environ["API_BASE_URL"] = args.mem_base_url
    from memory import Memory
    Memory_config = yaml.safe_load(open('ProcedureMem/config.yaml'))
    Memory_config["memory_dir"] = f"Webshop/memory/memp_{args.exp_name}"
    Memory_config["retrieve_num"] = args.retrieve_num
    Pro_Mem = Memory(**Memory_config)

    # Webshop env
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'webshop'))
    WebshopMultiProcessEnv = importlib.import_module('envs').WebshopMultiProcessEnv
    webshop_env = WebshopMultiProcessEnv(
        seed=42,
        env_num=args.batch_size,
        group_n=1,
        resources_per_worker={'num_cpus': 1},
        is_train=False,
        env_kwargs={'observation_mode': 'text'},
    )
    manager = make_webshop_manager(webshop_env, args.batch_size)

    num_games = len(webshop_env.goal_idxs)
    if args.num_games > 0:
        num_games = min(args.num_games, num_games)
    print(f"Total WebShop games: {num_games}")

    finished_games = 0
    all_reward = 0.0
    for f in os.listdir(output_path):
        if f.endswith('.json'):
            finished_games += 1
            all_reward += json.load(open(f'{output_path}/{f}')).get('reward', 0)

    for batch_idx in tqdm(range(math.ceil(num_games / args.batch_size))):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, num_games)
        real_n = end - start
        if end <= finished_games:
            continue

        indices = list(range(start, end))
        if real_n < args.batch_size:
            indices += [indices[-1]] * (args.batch_size - real_n)

        obs_dict, _ = manager.reset(indices=indices)
        task_descriptions = manager.tasks

        # Top-1 retrieval per task
        guidelines_context = {}
        workflow_list = []
        memory_list = []
        if len(Pro_Mem.documents) > 0:
            for i in range(real_n):
                hits = Pro_Mem.retrieve(task_descriptions[i])
                if hits:
                    top = hits[0][0] if isinstance(hits[0], tuple) else hits[0]
                    q = top.metadata.get('query')
                    w = top.metadata.get('workflow')
                    memory_list.append(q)
                    workflow_list.append(w)
                    guidelines_context[i] = json.dumps(
                        [{"task_name": q, "guidelines": w}], indent=4, ensure_ascii=False)
                else:
                    memory_list.append(None)
                    workflow_list.append(None)
        else:
            memory_list = [None] * real_n
            workflow_list = [None] * real_n

        batch_results = run_memp_webshop_batch(
            manager=manager,
            obs_dict=obs_dict,
            max_steps=args.max_steps,
            model=args.model,
            guidelines_context=guidelines_context,
        )

        for i in range(real_n):
            result = batch_results[i]
            all_reward += result['reward']
            finished_games += 1
            with open(f'{output_path}/idx_{start + i}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        tqdm.write(f'Avg reward: {all_reward / finished_games:.3f}  [{finished_games}/{num_games}]')

        # Update memory
        if Pro_Mem.is_cold_start == False:
            query_list = task_descriptions[:real_n]
            trajectory_list = [trajectory_text(batch_results[i]['messages']) for i in range(real_n)]
            reward_list = [1 if batch_results[i]['reward'] >= 1.0 else 0 for i in range(real_n)]
            Pro_Mem.update(query_list, trajectory_list, reward_list, workflow_list[:real_n], memory_list[:real_n])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--retrieve_num', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='memp')
    parser.add_argument('--num_games', type=int, default=500)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--mem_model', type=str, default=None)
    parser.add_argument('--mem_base_url', type=str, default=None)
    args = parser.parse_args()
    main(args)
