import os
import json
import re
import requests
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from verl import DataProto
from skills.skills_memory import SkillMemory
from skillos.utils import count_tokens

from skillos.llm_agent.generation_skills import MemoryGenerationManager, MemoryGenerationConfig
from agent_system.environments.env_package.alfworld.projection import alfworld_projection

ALFWORLD_CHUNK_TEMPLATE = """
# Task Context

## Task Description
```
{task_description}
```

## Past Skills
```
{past_skills}
```

## Agent Trajectory
```
{trajectory}
```

## Result
```
{result}
```

# Output Format:
Your output must contain the following sections:
- Analysis: Analyze the trajectory, associated skills, and the final result. Identify what went well and what didn't.
- Tool Calls: Based on your analysis, determine whether to insert a new skill, update an existing skill, or delete an existing skill.
"""


@dataclass
class AlfWorldGenerationConfig(MemoryGenerationConfig):
    alfworld_max_steps: int = 50
    # respond_url inherited from MemoryGenerationConfig — memory server handles
    # BM25 skill retrieval + frozen LLM call (no direct vLLM config needed)


class AlfWorldGenerationManager(MemoryGenerationManager):
    """Generation manager for memory agent training on AlfWorld tasks.

    Replaces math Q&A with multi-turn AlfWorld episodes:
    - Frozen LLM (vLLM server) executes tasks with memory as system prompt
    - Memory curator (actor_rollout_wg) learns to manage skills from trajectories

    Follows the standard verl-GRPO pattern: the trainer repeats gen_batch externally
    (gen_batch.repeat(num_rollouts=env.rollout.n, interleave=True)) before calling here.
    env.rollout.n is ALSO passed to make_envs as group_n, so the env has
    train_batch_size * env.rollout.n = total_batch_size slots.

    For each task:
      1. env_manager.reset() returns total_batch_size (e.g. 512) DIFFERENT observations.
      2. _execute_alfworld_episodes runs all 512 episodes with batched vLLM per step.
      3. Memory curator processes all 512 chunks at once (independent per slot).
    """

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: AlfWorldGenerationConfig,
        is_validation: bool = False,
    ):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation)
        # No vllm_client — frozen LLM is called via self.config.respond_url
        # (memory server handles BM25 retrieval + LLM inference)

    def _query_actions_from_server(
        self,
        active_memories: List[SkillMemory],
        active_observations: List[str],
    ) -> List[str]:
        """Query the remote memory server for AlfWorld actions.

        Mirrors _process_question_with_memory_using_server in generation_skills.py.
        Server does BM25 retrieval of relevant skills from memory, then calls the
        frozen LLM with those skills as context to produce an action.

        Args:
            active_memories: SkillMemory for each active env.
            active_observations: Current observation text for each active env.

        Returns:
            List of action strings, one per active env.
        """
        payload = {
            "memories": [{'skills': memory.skills} for memory in active_memories],
            "questions": [[obs] for obs in active_observations],  # one obs per env
        }
        max_retries = 3
        for attempt in range(max_retries):
            response = requests.post(self.config.respond_url, json=payload)
            results = response.json().get('result', [])
            retrieved_skills = response.json().get('retrieved_skills', [])
            
            if len(results) > 0:
                break

            if attempt < max_retries - 1:
                print(f"[AlfWorld] Empty server results on attempt {attempt + 1}, retrying...")
            else:
                raise AssertionError(f"Got empty results from server after {max_retries} attempts")
        
        return results, retrieved_skills

    def _execute_alfworld_episodes(
        self,
        env_manager,
        batch_memories: List[SkillMemory],
    ) -> Tuple[List[str], List[float], List[bool]]:
        """Run one round of AlfWorld episodes with the frozen LLM + memory as system prompt.

        Calls env_manager.reset() ONCE to draw total_batch_size unique scenarios
        (env has train_batch_size * env.rollout.n slots).
        All active envs query the remote memory server in one call per step:
        server does BM25 retrieval of relevant skills, then calls the frozen LLM.

        Args:
            env_manager: AlfWorldEnvironmentManager with total_batch_size parallel env slots.
            batch_memories: Current skills for each of the total_batch_size slots.
                            Used as system prompt for the frozen AlfWorld LLM.

        Returns:
            trajectories_text: Formatted trajectory string per env (len = total_batch_size).
            rewards: Total reward per env.
            successes: Whether each env succeeded.
        """
        obs, infos = env_manager.reset({})

        print(f"[DEBUG] Initial observations and infos from env_manager.reset(): (total {len(infos)})") 
        print(infos[0]['observation_text'].split("Your task is to:")[1].strip())  # print task instruction of the first env
        print(infos[1]['observation_text'].split("Your task is to:")[1].strip())
        print(infos[8]['observation_text'].split("Your task is to:")[1].strip())  # print task instruction of the 9th env
        # print(infos[16]['observation_text'].split("Your task is to:")[1].strip())  # print task instruction of the 17th env

        num_envs = len(obs['text'])
        env_dones = [False] * num_envs
        trajectories = [[] for _ in range(num_envs)]
        total_rewards = [0.0] * num_envs
        steps_per_env = [self.config.alfworld_max_steps] * num_envs
        task_descriptions = [infos[i]['observation_text'].split("Your task is to:")[1].strip() for i in range(len(infos))]

        for step_idx in range(self.config.alfworld_max_steps):
            active_indices = [i for i in range(num_envs) if not env_dones[i]]
            if not active_indices:
                break

            active_memories = [batch_memories[i] for i in active_indices]
            active_observations = [obs['text'][i] for i in active_indices]

            # ONE server call for all active envs: BM25 retrieval → frozen LLM → actions
            active_actions, retrieved_skills = self._query_actions_from_server(active_memories, active_observations)
            if step_idx == 0:
                final_retrieved_skills = retrieved_skills  # log retrieved skills from the first step

            actions = ["look"] * num_envs
            for j, i in enumerate(active_indices):
                actions[i] = active_actions[j][0]
                think_match = re.search(r'think[>\]](.*?)(?=[<\[]\/think)', active_actions[j][0], re.DOTALL)
                match = re.search(r'action[>\]](.*?)(?=[<\[]\/action)', active_actions[j][0], re.DOTALL)
                trajectories[i].append({
                    "step": step_idx + 1,
                    "observation": obs['anchor'][i].strip(),
                    # "action": active_actions[j][0].strip(),
                    # "think": think_match.group(1).strip() if think_match else "",
                    "action": match.group(1).strip() if match else active_actions[j][0].strip(),
                })


            obs, rewards, dones, infos = env_manager.step(actions)

            for i in range(num_envs):
                if not env_dones[i]:
                    total_rewards[i] += float(rewards[i]/10)
                    if dones[i]:
                        env_dones[i] = True
                        steps_per_env[i] = step_idx + 1

            if all(env_dones):
                break

        # Determine success from last infos
        successes = []
        for i in range(num_envs):
            won = bool(infos[i].get("won", False)) if env_dones[i] else False
            successes.append(won)

        # Format trajectories as text
        trajectories_text = []
        for i in range(num_envs):
            lines = []
            for step in trajectories[i]:
                lines.append(f"[Step {step['step']}]")
                lines.append(f"[Observation]: {step['observation']}")
                # lines.append(f"[Think]: {step['think']}")
                lines.append(f"[Action]: {step['action']}")
                lines.append("")
            trajectories_text.append("\n".join(lines))

        return trajectories_text, total_rewards, successes, final_retrieved_skills, task_descriptions, steps_per_env


    def run_memory_loop_alfworld(
        self,
        gen_batch: DataProto,
        env_manager,
        num_tasks: int,
        num_gpus: int = 1,
        global_steps: int = 0,
        is_validation: bool = False,
    ) -> DataProto:
        """Run memory agent loop on AlfWorld episodes.

        Mirrors generation_skills.py run_memory_loop exactly:
        - gen_batch is already repeated by the trainer (size = train_batch_size * env.rollout.n)
        - env has total_batch_size slots (env.rollout.n passed as group_n to make_envs)
        - env_manager.reset() returns total_batch_size DIFFERENT observations per task
        - Each slot i gets its own trajectory traj[i] — no broadcast

        For each of num_tasks sequential tasks:
          1. env_manager.reset() → total_batch_size unique scenarios
          2. _execute_alfworld_episodes runs all total_batch_size episodes (batched vLLM per step)
          3. Memory curator processes all total_batch_size chunks; batch_memories[i] updated independently

        Args:
            gen_batch: Already-repeated batch (size = train_batch_size * env.rollout.n).
            env_manager: AlfWorldEnvironmentManager with total_batch_size env slots.
            num_tasks: Fixed number of sequential AlfWorld episodes per training step.
            num_gpus: Number of GPUs for distributed processing.
            global_steps: Current training step (for logging).
            is_validation: Whether this is a validation run.

        Returns:
            DataProto with memory curator's prompts, responses, and metadata for RL training.
        """
        total_batch_size = gen_batch.batch['input_ids'].shape[0]

        # One independent memory per batch slot
        batch_memories = [SkillMemory() for _ in range(total_batch_size)]

        last_chunk_meta_info = {}

        chunk_input_ids_list = []
        chunk_responses_ids_list = []
        chunk_response_masks_list = []
        chunk_function_call_rewards_list = []
        chunk_function_calls_list = []

        total_chunk_length = [0] * total_batch_size

        all_trajectories_list = []
        all_rewards_list = []
        all_successes_list = []
        all_steps_list = []
        all_memories_list = []
        all_retrieved_skills_list = []

        for task_idx in range(num_tasks):
            print(f"[DEBUG] Processing AlfWorld task sequences {task_idx + 1}/{num_tasks}")

            # ONE call: all total_batch_size envs in parallel → total_batch_size trajectories
            # env_manager.reset() (inside) returns one unique scenario per slot
            traj, rew, succ, retrieved_skills, task_descriptions, steps = self._execute_alfworld_episodes(env_manager, batch_memories)

            print(f"  success_rate={sum(succ)}/{len(succ)}, "
                  f"avg_reward={sum(rew)/len(rew):.2f}")

            # Each slot i gets its OWN trajectory — no broadcast (mirrors generation_skills.py)
            current_chunks = []
            for i in range(total_batch_size):
                chunk_content = ALFWORLD_CHUNK_TEMPLATE.format(
                    trajectory=traj[i],
                    task_description=task_descriptions[i],
                    past_skills=retrieved_skills[i][0],
                    result="Success" if succ[i] else "Failure",
                )
                current_chunks.append(chunk_content)
                total_chunk_length[i] += count_tokens(chunk_content)

            # Process all chunks with memory curator (updates batch_memories in place)
            device = gen_batch.batch['input_ids'].device
            empty_ids = torch.zeros((total_batch_size, 0), dtype=torch.long, device=device)
            rollings = DataProto.from_dict({
                'input_ids': empty_ids,
                'attention_mask': torch.ones_like(empty_ids),
                'position_ids': torch.zeros_like(empty_ids),
            })
            rollings.meta_info = gen_batch.meta_info.copy()

            active_chunk_input_ids, active_chunk_responses_ids, active_chunk_response_mask, chunk_meta_info = \
                self._process_chunk_with_memory_operations(rollings, current_chunks, batch_memories)
            last_chunk_meta_info = chunk_meta_info

            task_chunk_input_ids = [None] * total_batch_size
            task_chunk_responses_ids = [None] * total_batch_size
            task_chunk_response_masks = [None] * total_batch_size
            task_chunk_fc_rewards = [None] * total_batch_size
            task_chunk_fc_calls = [None] * total_batch_size

            for i in range(total_batch_size):
                task_chunk_input_ids[i] = active_chunk_input_ids[i]
                task_chunk_responses_ids[i] = active_chunk_responses_ids[i]
                task_chunk_response_masks[i] = active_chunk_response_mask[i]
                task_chunk_fc_rewards[i] = chunk_meta_info['function_call_rewards'][i] \
                    if 'function_call_rewards' in chunk_meta_info else 0.0
                task_chunk_fc_calls[i] = chunk_meta_info['all_function_calls'][i] \
                    if 'all_function_calls' in chunk_meta_info else []

            # Log for debugging — each slot has its own trajectory
            all_traj_log = list(traj)
            all_rew_log  = list(rew)
            all_succ_log = list(succ)

            dump_path = os.getenv("ROLLOUT_DATA_DIR", ".") + (
                "/generation/validation" if is_validation else "/generation/training"
            )
            os.makedirs(dump_path, exist_ok=True)
            with open(os.path.join(dump_path, f"{global_steps}.jsonl"), "a") as f:
                f.write(json.dumps({
                    "task_idx": task_idx,
                    "memories": [{'skills': m.skills} for m in batch_memories],
                    "trajectories": all_traj_log,
                    "rewards": all_rew_log,
                    "successes": all_succ_log,
                }, ensure_ascii=False) + "\n")

            all_trajectories_list.append(all_traj_log)
            all_rewards_list.append(all_rew_log)
            all_successes_list.append(all_succ_log)
            all_steps_list.append(list(steps))
            all_memories_list.append([{'skills': m.skills} for m in batch_memories])

            chunk_input_ids_list.append(task_chunk_input_ids)
            chunk_responses_ids_list.append(task_chunk_responses_ids)
            chunk_response_masks_list.append(task_chunk_response_masks)
            chunk_function_call_rewards_list.append(task_chunk_fc_rewards)
            chunk_function_calls_list.append(task_chunk_fc_calls)

        # Assemble final output — same pattern as run_memory_loop (math)
        total_memory_length = [memory.total_length() for memory in batch_memories]

        all_input_ids = []
        all_response_ids = []
        all_response_masks = []
        indices_in_batch = []
        task_position_in_batch = []
        all_function_call_rewards = []
        all_function_calls = []
        per_chunk_rewards = []
        per_chunk_successes = []

        for idx in range(total_batch_size):
            current_input_ids = [step[idx] for step in chunk_input_ids_list]
            current_response_ids = [step[idx] for step in chunk_responses_ids_list]
            current_response_masks = [step[idx] for step in chunk_response_masks_list]
            current_fc_rewards = [step[idx] for step in chunk_function_call_rewards_list]
            current_fc_calls = [step[idx] for step in chunk_function_calls_list]

            indices_in_batch.extend([idx] * len(current_input_ids))
            task_position_in_batch.extend(range(len(current_input_ids)))
            all_input_ids.extend(current_input_ids)
            all_response_ids.extend(current_response_ids)
            all_response_masks.extend(current_response_masks)
            all_function_call_rewards.extend(current_fc_rewards)
            all_function_calls.extend(current_fc_calls)
            # Shifted credit assignment: chunk k gets reward from task k+1 (not task k),
            # because the curation at task k affects future tasks, not the current one.
            # The last chunk gets 0 since there is no task k+1 to observe.
            task_rews_for_slot  = [float(task_rews[idx]) for task_rews  in all_rewards_list]
            task_succ_for_slot  = [float(task_succ[idx]) for task_succ in all_successes_list]
            per_chunk_rewards.extend( task_rews_for_slot[1:]  + [0.0])
            per_chunk_successes.extend(task_succ_for_slot[1:] + [0.0])

        # Pad input_ids to the left
        max_input_length = max(len(x) for x in all_input_ids)
        new_all_input_ids = []
        for input_ids in all_input_ids:
            if len(input_ids) < max_input_length:
                new_all_input_ids.append(torch.cat([
                    torch.tensor([self.tokenizer.pad_token_id] * (max_input_length - len(input_ids))),
                    input_ids
                ]))
            else:
                new_all_input_ids.append(input_ids)
        all_input_ids = new_all_input_ids

        # Pad response_ids to the right
        max_response_length = max(len(x) for x in all_response_ids)
        new_all_response_ids = []
        new_all_response_masks = []
        for response_ids, response_mask in zip(all_response_ids, all_response_masks):
            if len(response_ids) < max_response_length:
                new_all_response_ids.append(torch.cat([
                    response_ids,
                    torch.tensor([self.tokenizer.pad_token_id] * (max_response_length - len(response_ids)))
                ]))
                new_all_response_masks.append(torch.cat([
                    response_mask,
                    torch.tensor([False] * (max_response_length - len(response_mask)))
                ]))
            else:
                new_all_response_ids.append(response_ids)
                new_all_response_masks.append(response_mask)

        all_response_ids = new_all_response_ids
        all_response_masks = new_all_response_masks

        final_output = {
            'prompts': torch.stack(all_input_ids),
            'responses': torch.stack(all_response_ids),
            'response_mask': torch.stack(all_response_masks),
        }

        final_output['input_ids'] = torch.cat([final_output['prompts'], final_output['responses']], dim=1)
        final_output['attention_mask'] = torch.where(
            final_output['input_ids'] != self.tokenizer.pad_token_id, 1, 0
        )
        final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])
        final_output['attention_mask'][:, -final_output['response_mask'].shape[1]:] = final_output['response_mask']

        if self.config.analyze_function_url is not None:
            all_function_call_content_rewards = self._analyze_function_call_content(all_function_calls)

        # GPU padding if needed
        if num_gpus > 1:
            current_batch_size = final_output['input_ids'].shape[0]
            padding_needed = ((current_batch_size + num_gpus - 1) // num_gpus) * num_gpus - current_batch_size

            if padding_needed > 0:
                repeat_indices = torch.arange(current_batch_size)[:padding_needed]

                for key in ['input_ids', 'attention_mask', 'position_ids', 'prompts', 'responses', 'response_mask']:
                    final_output[key] = torch.cat([
                        final_output[key],
                        final_output[key][repeat_indices]
                    ], dim=0)

                all_function_call_rewards.extend([all_function_call_rewards[i] for i in repeat_indices.tolist()])
                all_function_calls.extend([all_function_calls[i] for i in repeat_indices.tolist()])
                per_chunk_rewards.extend(  [per_chunk_rewards[i]   for i in repeat_indices.tolist()])
                per_chunk_successes.extend([per_chunk_successes[i] for i in repeat_indices.tolist()])

                if self.config.analyze_function_url is not None:
                    all_function_call_content_rewards.extend(
                        [all_function_call_content_rewards[i] for i in repeat_indices.tolist()]
                    )

                indices_in_batch.extend([indices_in_batch[i] for i in repeat_indices.tolist()])
                task_position_in_batch.extend([task_position_in_batch[i] for i in repeat_indices.tolist()])

        every_chunk_length = [rm.sum().item() for rm in final_output['response_mask']]

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update({
            'trajectories_list': all_trajectories_list,
            'steps_list': all_steps_list,
            'memories_list': all_memories_list,
            'indices_in_batch': indices_in_batch,
            'task_position_in_batch': task_position_in_batch,
            'total_chunk_length': total_chunk_length,
            'total_memory_length': total_memory_length,
            'every_chunk_length': every_chunk_length,
            'batch_memories': [{'skills': memory.skills} for memory in batch_memories],
            'all_function_call_rewards': all_function_call_rewards,
            'all_function_calls': all_function_calls,
            'per_chunk_rewards': per_chunk_rewards,
            'per_chunk_successes': per_chunk_successes,
        })
        if self.config.analyze_function_url is not None:
            final_output.meta_info['all_function_call_content_rewards'] = all_function_call_content_rewards
        final_output.meta_info.update(last_chunk_meta_info)

        return final_output
