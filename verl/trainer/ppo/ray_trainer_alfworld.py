"""
AlfWorld variant of the RayPPO Trainer for memory curator training.

Key differences from ray_trainer.py (math reasoning):
- No dataloader for task content — tasks come from env.reset()
- Dummy gen_batch carries only batch_size, device, meta_info
- num_tasks per training step is fixed (config: alfworld.num_tasks, default 10)
- GRPO rollouts handled by gen_batch.repeat() in the trainer (same as math)
- Uses AlfWorldGenerationManager instead of MemoryGenerationManager
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, Role, ResourcePoolManager, compute_advantage, apply_kl_penalty, compute_response_mask
from skillos.llm_agent.generation_alfworld import AlfWorldGenerationManager, AlfWorldGenerationConfig


class RayPPOTrainerAlfWorld(RayPPOTrainer):
    """AlfWorld variant of RayPPOTrainer for memory curator training.

    Instead of iterating over a dataloader with preprocessed QA pairs,
    this trainer runs AlfWorld episodes with a frozen LLM (vLLM server)
    and trains the memory curator to manage skills from trajectories.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls=None,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        envs=None,
        val_envs=None,
        device_name="cuda",
    ):
        if ray_worker_group_cls is None:
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=None,
            val_dataset=None,
            collate_fn=None,
            train_sampler=None,
            device_name=device_name,
        )

        # Environments are created in main_ppo.py via make_envs(config) and passed here.
        self.env_manager = envs
        self.val_env_manager = val_envs if val_envs is not None else envs

        # Derive parallel-env batch sizes from config.
        # make_envs creates train envs with env_num=train_batch_size, group_n=env.rollout.n
        # and val envs with env_num=val_batch_size, group_n=1.
        group_n = OmegaConf.select(config, "env.rollout.n", default=1)
        if not isinstance(group_n, int) or group_n < 1:
            group_n = 1
        self.alfworld_batch_size = config.data.train_batch_size * group_n
        self.val_alfworld_batch_size = config.data.val_batch_size  # val always uses group_n=1

        print(f"AlfWorld trainer: train_batch_size={self.alfworld_batch_size} "
              f"(train={config.data.train_batch_size}, group_n={group_n}), "
              f"val_batch_size={self.val_alfworld_batch_size}")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """Override: no dataloader needed for AlfWorld. Tasks come from env.reset()."""
        self.total_training_steps = self.config.trainer.total_training_steps
        if self.total_training_steps is None:
            raise ValueError(
                "For AlfWorld training, config.trainer.total_training_steps must be set "
                "(there is no dataloader to derive it from)."
            )
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config: {e}")

    def _build_dummy_gen_batch(self, do_sample=True, batch_size=None):
        """Construct a minimal dummy gen_batch for the generation manager.

        The gen_batch only carries batch_size, device, and meta_info.
        Actual task content comes from env.reset().
        """
        if batch_size is None:
            batch_size = self.alfworld_batch_size
        # The TaskRunner (where fit/validate run) is a CPU-only Ray task on the head node.
        # The dummy batch only carries metadata; actual GPU work is done by worker groups.
        dummy_ids = torch.zeros((batch_size, 1), dtype=torch.long, device="cpu")
        gen_batch = DataProto.from_dict({
            "input_ids": dummy_ids,
            "attention_mask": torch.ones_like(dummy_ids),
            "position_ids": torch.zeros_like(dummy_ids),
        })
        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": do_sample,
        }
        return gen_batch

    def _build_generation_manager(self, is_validation=False):
        """Create the AlfWorld generation manager."""
        gen_config = AlfWorldGenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            respond_url=self.config.get("respond_url", None),
            analyze_function_url=self.config.get("analyze_function_url", None),
            enable_thinking=self.config.get("enable_thinking", True),
            alfworld_max_steps=self.config.alfworld.get("max_steps", 30),
        )
        return AlfWorldGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=is_validation,
        )

    def _validate(self):
        """Validation loop for AlfWorld. Mirrors ray_trainer.py _validate() lines 600-834."""
        reward_tensor_lst = []
        compression_ratio_reward_scores_lst = []
        all_function_call_rewards_lst = []
        all_function_call_content_rewards_lst = []
        data_source_lst = []
        memory_indicator_scores_lst = []
        acc_tensor_lst = []

        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_memories = []
        reward_extra_infos_dict = {
            "reward": [],
            "compression_ratio_reward_score": [],
            "all_function_call_rewards": [],
            "all_function_call_content_rewards": [],
            "memory_indicator_scores": [],
            "acc_scores": [],
        }

        generation_manager = self._build_generation_manager(is_validation=True)
        gen_batch = self._build_dummy_gen_batch(do_sample=False, batch_size=self.val_alfworld_batch_size)
        num_tasks = self.config.alfworld.get("val_tasks", self.config.alfworld.get("num_tasks", 5))

        timing_raw = {}
        with marked_timer("step", timing_raw):
            with marked_timer("gen", timing_raw):
                final_gen_batch_output = generation_manager.run_memory_loop_alfworld(
                    gen_batch=gen_batch,
                    env_manager=self.val_env_manager,
                    num_tasks=num_tasks,
                    global_steps=self.global_steps,
                    is_validation=True,
                )

            batch = final_gen_batch_output
            data_sources = np.array(["alfworld"] * len(batch.batch["input_ids"]), dtype=object)
            batch.non_tensor_batch["data_source"] = data_sources

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)

            # Direct call — same as ray_trainer.py line 704 (avoids return_dict=True scalar issue)
            reward_tensor, acc_scores, compression_ratio_reward_scores, all_function_call_rewards, all_function_call_content_rewards, memory_indicator_scores = self.val_reward_fn(batch, data_sources)

            print(f"reward_tensor: {reward_tensor.sum(-1)}")
            print(f"acc_scores: {acc_scores}")
            print(f"compression_ratio_reward_scores: {compression_ratio_reward_scores}")
            print(f"all_function_call_rewards: {all_function_call_rewards}")
            print(f"all_function_call_content_rewards: {all_function_call_content_rewards}")
            print(f"memory_indicator_scores: {memory_indicator_scores}")

            output_ids = batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()

            input_ids = batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_scores.extend(scores)
            sample_memories.extend(final_gen_batch_output.meta_info['batch_memories'])

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos_dict["acc_scores"].extend(acc_scores)
            reward_extra_infos_dict["compression_ratio_reward_score"].extend(compression_ratio_reward_scores)
            reward_extra_infos_dict["all_function_call_rewards"].extend(all_function_call_rewards)
            reward_extra_infos_dict["all_function_call_content_rewards"].extend(all_function_call_content_rewards)
            reward_extra_infos_dict["memory_indicator_scores"].extend(memory_indicator_scores)

            reward_tensor_lst.append(reward_tensor)
            compression_ratio_reward_scores_lst.append(torch.tensor(compression_ratio_reward_scores))
            all_function_call_rewards_lst.append(torch.tensor(all_function_call_rewards))
            all_function_call_content_rewards_lst.append(torch.tensor(all_function_call_content_rewards))
            memory_indicator_scores_lst.append(torch.tensor(memory_indicator_scores))
            acc_tensor_lst.append(torch.tensor(acc_scores))
            data_source_lst.append(data_sources)

        # AlfWorld-specific raw metrics (not in math trainer)
        metric_dict = {}
        if "successes_list" in final_gen_batch_output.meta_info:
            all_successes = final_gen_batch_output.meta_info["successes_list"]
            flat_successes = [s for step_successes in all_successes for s in step_successes]
            metric_dict["val/alfworld_success_rate"] = float(np.mean(flat_successes)) if flat_successes else 0.0

        if "rewards_list" in final_gen_batch_output.meta_info:
            all_rewards_meta = final_gen_batch_output.meta_info["rewards_list"]
            flat_rewards = [r for step_rewards in all_rewards_meta for r in step_rewards]
            metric_dict["val/alfworld_avg_reward"] = float(np.mean(flat_rewards)) if flat_rewards else 0.0

        if "steps_list" in final_gen_batch_output.meta_info:
            all_steps = final_gen_batch_output.meta_info["steps_list"]
            flat_steps = [s for task_steps in all_steps for s in task_steps]
            metric_dict["val/avg_steps_per_task"] = float(np.mean(flat_steps)) if flat_steps else 0.0

        # Aggregation — identical to ray_trainer.py lines 752-834
        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()
        data_sources = np.concatenate(data_source_lst, axis=0)
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            ds = data_sources[i]
            if ds not in data_source_reward:
                data_source_reward[ds] = []
            data_source_reward[ds].append(reward_tensor[i].item())

        compression_ratio_reward_scores = torch.cat(compression_ratio_reward_scores_lst, dim=0).cpu()
        data_source_compression_ratio_reward_scores = {}
        for i in range(compression_ratio_reward_scores.shape[0]):
            ds = data_sources[i]
            if ds not in data_source_compression_ratio_reward_scores:
                data_source_compression_ratio_reward_scores[ds] = []
            data_source_compression_ratio_reward_scores[ds].append(compression_ratio_reward_scores[i].item())

        all_function_call_rewards = torch.cat(all_function_call_rewards_lst, dim=0).cpu()
        data_source_all_function_call_rewards = {}
        for i in range(all_function_call_rewards.shape[0]):
            ds = data_sources[i]
            if ds not in data_source_all_function_call_rewards:
                data_source_all_function_call_rewards[ds] = []
            data_source_all_function_call_rewards[ds].append(all_function_call_rewards[i].item())

        all_function_call_content_rewards = torch.cat(all_function_call_content_rewards_lst, dim=0).cpu()
        data_source_all_function_call_content_rewards = {}
        for i in range(all_function_call_content_rewards.shape[0]):
            ds = data_sources[i]
            if ds not in data_source_all_function_call_content_rewards:
                data_source_all_function_call_content_rewards[ds] = []
            data_source_all_function_call_content_rewards[ds].append(all_function_call_content_rewards[i].item())

        memory_indicator_scores = torch.cat(memory_indicator_scores_lst, dim=0).cpu()
        data_source_memory_indicator_scores = {}
        for i in range(memory_indicator_scores.shape[0]):
            ds = data_sources[i]
            if ds not in data_source_memory_indicator_scores:
                data_source_memory_indicator_scores[ds] = []
            data_source_memory_indicator_scores[ds].append(memory_indicator_scores[i].item())

        acc_tensor = torch.cat(acc_tensor_lst, dim=0).cpu()
        data_source_acc = {}
        for i in range(acc_tensor.shape[0]):
            ds = data_sources[i]
            if ds not in data_source_acc:
                data_source_acc[ds] = []
            data_source_acc[ds].append(acc_tensor[i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/reward/{data_source}'] = np.mean(rewards)
        for data_source, rewards in data_source_compression_ratio_reward_scores.items():
            metric_dict[f'val/compression_ratio_reward_score/{data_source}'] = np.mean(rewards)
        for data_source, rewards in data_source_all_function_call_rewards.items():
            metric_dict[f'val/all_function_call_rewards/{data_source}'] = np.mean(rewards)
        for data_source, rewards in data_source_all_function_call_content_rewards.items():
            metric_dict[f'val/all_function_call_content_rewards/{data_source}'] = np.mean(rewards)
        for data_source, rewards in data_source_memory_indicator_scores.items():
            metric_dict[f'val/memory_indicator_scores/{data_source}'] = np.mean(rewards)
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/acc/{data_source}'] = np.mean(accs)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        return metric_dict

    def fit(self):
        """Training loop for AlfWorld memory curator.

        Key differences from math training:
        - No dataloader iteration — each step runs env.reset() for fresh tasks
        - num_tasks per step is fixed (config: alfworld.num_tasks, default 10)
        - GRPO rollouts via gen_batch.repeat() before calling generation manager
        - Dummy gen_batch instead of dataset-provided batch
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # Initial validation
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        generation_manager = self._build_generation_manager()

        # One-time: lock each env slot to a single task type for consistent memory chains.
        if self.config.alfworld.get("same_task_type_per_chain", False):
            print("[DEBUG] same_task_type_per_chain=True: locking slot task types...")
            self.env_manager.lock_slot_task_types()

        num_tasks = self.config.alfworld.get("num_tasks", 10)
        num_rollouts = self.config.customized_grpo_rollout_n
        env_batch_size = self.config.data.train_batch_size

        for step in range(self.total_training_steps):
            is_last_step = self.global_steps >= self.total_training_steps

            metrics = {}
            timing_raw = {}
            reward_extra_infos_dict = {}

            with marked_timer("step", timing_raw):
                # Generate trajectories and process with memory curator.
                # Mirror the math GRPO pattern: repeat gen_batch by num_rollouts
                # (interleaved) before passing to the generation manager.
                with marked_timer("gen", timing_raw, color="red"):
                    # alfworld_batch_size = train_batch_size * env.rollout.n already
                    # (set in __init__), so no repeat() needed — unlike math where
                    # the dataloader batch starts at train_batch_size.
                    gen_batch = self._build_dummy_gen_batch(do_sample=True)

                    print(f"[Step {self.global_steps}] Running {num_tasks} AlfWorld tasks "
                          f"x {env_batch_size} envs x {num_rollouts} rollouts "
                          f"(total_batch={env_batch_size * num_rollouts})")

                    final_gen_batch_output = generation_manager.run_memory_loop_alfworld(
                        gen_batch=gen_batch,
                        env_manager=self.env_manager,
                        num_tasks=num_tasks,
                        num_gpus=self.actor_rollout_wg.world_size,
                        global_steps=self.global_steps,
                    )

                batch = final_gen_batch_output

                # Assign UIDs for GRPO grouping.
                # After interleave repeat the layout is:
                #   [env0_r0, env0_r1, ..., env0_r(N-1), env1_r0, ..., env(B-1)_r(N-1)]
                # so positions 0..N-1 all belong to env0, N..2N-1 to env1, etc.
                # env slot for position p = p // num_rollouts → uid[p // num_rollouts]
                total_batch_size = env_batch_size * num_rollouts
                base_uids = [str(uuid.uuid4()) for _ in range(env_batch_size)]
                uid_mapping = np.array(
                    [base_uids[i // num_rollouts] for i in range(total_batch_size)],
                    dtype=object,
                )

                # Use indices_in_batch + task_position_in_batch to assign UIDs.
                # UID is per (env_group, task_position) so that GRPO compares the
                # num_rollouts different curation strategies at the same task position
                # rather than mixing rewards from different task positions in one group.
                indices_in_batch = final_gen_batch_output.meta_info['indices_in_batch']
                task_positions   = final_gen_batch_output.meta_info['task_position_in_batch']
                output_uids = np.array(
                    [f"{uid_mapping[indices_in_batch[j]]}_{task_positions[j]}" for j in range(len(indices_in_batch))],
                    dtype=object,
                )

                data_sources = np.array(["alfworld"] * len(batch.batch["input_ids"]), dtype=object)
                batch.non_tensor_batch["data_source"] = data_sources
                batch.non_tensor_batch["uid"] = output_uids

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)

                # Balance batch across DP ranks
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # Compute rewards
                with marked_timer("reward", timing_raw, color="yellow"):
                    if self.use_rm:
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(
                            batch, self.reward_fn, data_sources=data_sources
                        )

                # Log AlfWorld-specific metrics from raw generation meta_info
                if "successes_list" in final_gen_batch_output.meta_info:
                    all_successes = final_gen_batch_output.meta_info["successes_list"]
                    flat_successes = [s for step_s in all_successes for s in step_s]
                    metrics["train/alfworld_success_rate"] = float(np.mean(flat_successes)) if flat_successes else 0.0

                if "rewards_list" in final_gen_batch_output.meta_info:
                    all_rewards = final_gen_batch_output.meta_info["rewards_list"]
                    flat_rewards = [r for step_r in all_rewards for r in step_r]
                    metrics["train/alfworld_avg_reward"] = float(np.mean(flat_rewards)) if flat_rewards else 0.0

                if "steps_list" in final_gen_batch_output.meta_info:
                    all_steps = final_gen_batch_output.meta_info["steps_list"]
                    flat_steps = [s for task_steps in all_steps for s in task_steps]
                    metrics["train/avg_steps_per_task"] = float(np.mean(flat_steps)) if flat_steps else 0.0

                if reward_extra_infos_dict is not None:

                    all_acc_reward_scores = np.array(reward_extra_infos_dict.get("acc_scores", []))
                    all_compression_ratio_reward_scores = np.array(reward_extra_infos_dict.get("compression_ratio_reward_scores", []))
                    all_function_call_rewards = np.array(reward_extra_infos_dict.get("all_function_call_rewards", []))
                    all_function_call_content_rewards = np.array(reward_extra_infos_dict.get("all_function_call_content_rewards", []))
                    all_training_rewards = np.array(reward_extra_infos_dict.get("total_reward_scores", []))
                    all_memory_indicator_scores = np.array(reward_extra_infos_dict.get("memory_indicator_scores", []))

                    # indices_in_batch may include repeated indices to fit into all gpus (to make sure the total length is a multiple of eight)
                    tmp_indices_in_batch = final_gen_batch_output.meta_info['indices_in_batch'][:len(all_function_call_rewards)]
                    function_call_rewards = []
                    function_call_content_rewards = []
                    compression_ratio_reward_scores = []
                    acc_reward_scores = []
                    memory_indicator_scores = []
                    training_rewards = []

                    for batch_idx in range(len(np.unique(tmp_indices_in_batch))):
                        function_call_rewards.append(np.mean(np.array(all_function_call_rewards)[np.where(np.array(tmp_indices_in_batch) == batch_idx)[0]]))
                        function_call_content_rewards.append(np.mean(np.array(all_function_call_content_rewards)[np.where(np.array(tmp_indices_in_batch) == batch_idx)[0]]))
                        compression_ratio_reward_scores.append(np.mean(np.array(all_compression_ratio_reward_scores)[np.where(np.array(tmp_indices_in_batch) == batch_idx)[0]]))
                        acc_reward_scores.append(np.mean(np.array(all_acc_reward_scores)[np.where(np.array(tmp_indices_in_batch) == batch_idx)[0]]))
                        memory_indicator_scores.append(np.mean(np.array(all_memory_indicator_scores)[np.where(np.array(tmp_indices_in_batch) == batch_idx)[0]]))
                        training_rewards.append(np.mean(np.array(all_training_rewards)[np.where(np.array(tmp_indices_in_batch) == batch_idx)[0]]))

                    function_call_rewards = np.array(function_call_rewards)
                    function_call_content_rewards = np.array(function_call_content_rewards)
                    compression_ratio_reward_scores = np.array(compression_ratio_reward_scores)
                    acc_reward_scores = np.array(acc_reward_scores)
                    memory_indicator_scores = np.array(memory_indicator_scores)
                    training_rewards = np.array(training_rewards)

                    assert len(acc_reward_scores) == len(compression_ratio_reward_scores) == len(function_call_rewards) == len(function_call_content_rewards) == len(training_rewards) == len(memory_indicator_scores)

                    # slot_data_sources: one entry per GRPO slot — same length as acc_reward_scores
                    # (chunk-level data_sources is longer: slots × tasks, so can't be used directly for ds_indices)
                    slot_data_sources = np.array(["alfworld"] * len(acc_reward_scores))
                    unique_data_sources = np.unique(slot_data_sources)
                    for ds in unique_data_sources:
                        ds_indices = np.where(slot_data_sources == ds)[0]
                        metrics[f'train/acc/{ds}'] = float(np.mean(acc_reward_scores[ds_indices]))
                        metrics[f'train/total_reward/{ds}'] = float(np.mean(training_rewards[ds_indices]))
                        metrics[f'train/compression_ratio_reward/{ds}'] = float(np.mean(compression_ratio_reward_scores[ds_indices]))
                        metrics[f'train/function_call_rewards/{ds}'] = float(np.mean(function_call_rewards[ds_indices]))
                        metrics[f'train/function_call_content_rewards/{ds}'] = float(np.mean(function_call_content_rewards[ds_indices]))
                        metrics[f'train/memory_indicator_scores/{ds}'] = float(np.mean(memory_indicator_scores[ds_indices]))

                    reward_extra_infos_dict = {}

                # Recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, color="blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    metrics["actor/entropy"] = entropy_agg.detach().item()
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)

                if self.use_reference_policy:
                    with marked_timer("ref", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=num_rollouts,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        multi_turn=False,
                        config=self.config.algorithm,
                    )

                # Update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # Update actor
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        batch.meta_info["multi_turn"] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Dump rollout generations
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                        self._dump_generations(
                            inputs=inputs, outputs=outputs, scores=scores,
                            reward_extra_infos_dict=reward_extra_infos_dict,
                            dump_path=rollout_data_dir,
                        )

                # Validate
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Save checkpoint
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            metrics.update({
                "training/global_step": self.global_steps,
                "training/num_tasks": num_tasks,
            })
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
