"""MemCuratorGenerationManager — rollout for the read-time briefing curator (ALFWorld).

One GRPO training step (per the OFFLINE-frozen-global-pool, DIRECT-credit design):
  1. env_manager.reset() → total_batch_size = train_batch_size * n unique ALFWorld tasks
     (n = customized_grpo_rollout_n; slots are laid out interleaved so slot p belongs to
     env-group p // n — matching the trainer's UID scheme).
  2. For each slot, RETRIEVE top-k from the FROZEN GLOBAL POOL (BM25 on the NL task,
     self-exclude the slot's own gamefile) and render retrieved_text (eval parity).
  3. CURATOR (trainable actor policy) generates ONE briefing per slot from
     build_curator_messages(task, retrieved_text) — this is the RL action we train on.
  4. EXECUTOR (frozen, litellm-served) runs the ALFWorld episode with the _strip_think'd
     briefing injected via eval's ALFWORLD_TEMPLATE*_WITH_CONTEXT; env stepping via the
     Ray-parallel env_manager. Repeated K times per briefing; reward = mean success.
  5. Assemble a DataProto with DIRECT per-chunk reward (= that slot's task success; NO
     [1:] shift, NO chains) and the meta_info keys the trainer's fit() expects.

Reuses (by subclassing MemoryGenerationManager): _generate_with_gpu_padding,
_update_rolling_state, _postprocess_responses, _batch_tokenize, tensor_fn, and the
two-pass thinking + loss-mask logic (adapted here for briefings, minus tool execution).

Executor prompt/parse parity is provided by memcurator.alfworld_executor (byte-identical
copy of the eval runner's templates; guarded by test_executor_parity).

Env note: env_manager.build_text_obs uses DIFFERENT templates than eval, so we DO NOT use
it — we drive env_manager only for reset/step/won and use the raw obs (the 'anchor' field)
to build our own eval-parity prompt.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from verl import DataProto

from skillos.llm_agent.generation_skills import MemoryGenerationManager, MemoryGenerationConfig
from skillos.utils import count_tokens

from memcurator.alfworld_executor import (
    CURATOR_CONTEXT_HEADER,
    CURATOR_CONTEXT_LABEL,
    build_executor_prompt,
    process_ob,
    parse_action,
)

# Curator prompt/trajectory backend — selected by --curator_variant (curator_alfworld [default] |
# curator_alfworld_v1 | curator_alfworld_v1_api) so TRAINING builds the curator prompt with the
# SAME eval module + curation_mode being reproduced. The backend handles the eval-dir sys.path
# APPEND (avoids shadowing repo-root agent_system) and the 2-arg vs 3-arg signature differences.
from memcurator.curator_backend import CuratorBackend  # noqa: E402


_SENTENCE_TO_MASK = (
    "\n\nConsidering the limited time by the user, I have to give the solution based on "
    "the thinking directly now.\n</think>\n\n"
)


@dataclass
class MemCuratorGenerationConfig(MemoryGenerationConfig):
    """Adds MemCurator-specific knobs on top of MemoryGenerationConfig."""
    # DATASET-DRIVEN mode (Stage B output): each row = (target game_file, per-target frozen S_T).
    # When set, training iterates this dataset and pins each slot to its target game (controllable).
    dataset_path: Optional[str] = None
    # Optional train-time target filter by measured p-hat (default None = all targets).
    target_phat_lo: Optional[float] = None
    target_phat_hi: Optional[float] = None
    # GRPO group size (= customized_grpo_rollout_n); the n slots of a group share a pinned target.
    n_rollouts: int = 1
    # Curator backend variant + mode (must match the eval variant being reproduced).
    curator_variant: str = "curator_alfworld"
    curation_mode: str = "success_only"
    # FALLBACK global-pool mode (superseded; used only if dataset_path is unset): one frozen pool,
    # random env sampling. Kept for the pre-pivot path.
    pool_path: Optional[str] = None
    retrieve_num: int = 3
    # Executor (frozen) — litellm-served, eval parity.
    executor_model: str = "openai/Qwen/Qwen3-8B"
    executor_api_base: Optional[str] = None
    executor_api_key: str = "EMPTY"
    executor_temperature: float = 0.7
    executor_top_p: Optional[float] = None
    executor_top_k: Optional[int] = None
    executor_max_tokens: Optional[int] = None
    executor_enable_thinking: Optional[bool] = None
    executor_max_steps: int = 30
    history_length: int = 3
    # Curator briefing generation cap (actor rollout uses config.max_response_length).
    curator_on_empty: bool = False
    # NOTE on reward denoising: we do NOT do K same-task executor repeats in v1. Within a GRPO
    # group the n=group_n slots reset to the SAME game (workers share seed = seed + i//group_n,
    # see envs.py:118 + AlfworldWorker.__init__), so the n rollouts already differ ONLY in the
    # briefing — that IS the GRPO comparison. Denoising binary reward is handled by larger n, not
    # K (matches the pressure-test's "spend budget on n" and the SkillOS trainer, which has no K).
    # K-sampling on a fixed task is a v2 refinement (needs a reset-to-specific-game hook).


class MemCuratorGenerationManager(MemoryGenerationManager):
    """Read-time briefing curator rollout (subclasses MemoryGenerationManager for helpers)."""

    def __init__(self, tokenizer, actor_rollout_wg, config: MemCuratorGenerationConfig,
                 is_validation: bool = False):
        super().__init__(tokenizer, actor_rollout_wg, config, is_validation=is_validation)
        self.mc_config = config
        self._n_rollouts = getattr(config, "n_rollouts", 1) or 1
        # Curator backend (variant + mode) — same module the eval side uses. All curator-prompt /
        # store / _strip_think calls go through this so training matches eval byte-for-byte.
        self._backend = CuratorBackend(
            variant=getattr(config, "curator_variant", "curator_alfworld"),
            curation_mode=getattr(config, "curation_mode", "success_only"),
        )
        print(f"[MemCurator] curator backend: {self._backend.variant} "
              f"(curation_mode={self._backend.curation_mode})")
        self._store_cache: Dict[str, Any] = {}  # store_path -> loaded per-target store

        if config.dataset_path:
            # DATASET-DRIVEN mode (Stage B): load target rows, optional p-hat filter, set a cursor.
            self._dataset = self._load_dataset(config)
            self._cursor = 0
            self._pool = None
            print(f"[MemCurator] DATASET mode: {len(self._dataset)} targets from {config.dataset_path}")
        else:
            # FALLBACK global-pool mode (superseded).
            self._dataset = None
            self._pool = self._load_frozen_pool(config.pool_path, config.retrieve_num)

    def _load_dataset(self, config) -> List[Dict]:
        rows = [json.loads(l) for l in open(config.dataset_path, encoding="utf-8") if l.strip()]
        lo, hi = config.target_phat_lo, config.target_phat_hi
        if lo is not None and hi is not None:
            before = len(rows)
            rows = [r for r in rows if r.get("p_hat") is not None and lo <= r["p_hat"] <= hi]
            print(f"[MemCurator] target p-hat filter [{lo},{hi}]: {before} -> {len(rows)} targets")
        if not rows:
            raise ValueError(f"dataset {config.dataset_path} has 0 targets after filtering.")
        return rows

    def _store_for(self, store_path: str):
        """Load (and cache) a per-target frozen S_T via the selected backend (BM25 retrieval)."""
        if not store_path:
            return None
        if store_path not in self._store_cache:
            self._store_cache[store_path] = self._backend.make_store(
                storage_path=store_path, retrieve_num=self.mc_config.retrieve_num,
                curator_on_empty=True,
            )
        return self._store_cache[store_path]

    def _next_target_batch(self, n_slots: int, n_rollouts: int) -> List[Dict]:
        """Take the next (n_slots/n_rollouts) distinct targets, each repeated n_rollouts times.

        Layout matches the trainer's UID scheme (slot p -> target p//n_rollouts): interleaved so
        the n_rollouts group-mates share a target (→ same pinned game → GRPO compares briefings).
        Wraps around the dataset (sampling without a fixed epoch boundary; alfworld has no epochs).
        """
        n_targets = n_slots // n_rollouts
        picked = []
        for _ in range(n_targets):
            picked.append(self._dataset[self._cursor % len(self._dataset)])
            self._cursor += 1
        # expand interleaved: [t0]*n, [t1]*n, ...
        expanded = []
        for t in picked:
            expanded.extend([t] * n_rollouts)
        return expanded

    # ------------------------------------------------------------------ #
    # Frozen global pool (read-only)                                      #
    # ------------------------------------------------------------------ #
    def _load_frozen_pool(self, pool_path: Optional[str], retrieve_num: int):
        if not pool_path:
            print("[MemCurator] no pool_path set — cold pool (empty briefings unless curator_on_empty).")
            return None
        # backend loads its JSONL store on init and builds BM25.
        pool = self._backend.make_store(storage_path=pool_path, retrieve_num=retrieve_num,
                                        curator_on_empty=True)  # retrieval only; no LLM call here
        print(f"[MemCurator] frozen pool loaded from {pool_path}: {len(pool.memory_bank)} records.")
        return pool

    def _retrieve_from_store(self, store, task_desc: str, exclude_gamefile: str) -> str:
        """Top-k retrieved_text from a SPECIFIC per-target frozen S_T (dataset mode).

        S_T is already self-excluded at build time (Stage B), so no further exclusion needed;
        we still pass exclude_gamefile defensively. Same BM25 + _format_case as _retrieve_text.
        """
        if store is None or store.bm25_retriever is None:
            return ""
        docs = store.bm25_retriever.invoke(task_desc)
        parts: List[str] = []
        for doc in docs:
            rec = store.memory_bank[doc.metadata["idx"]]
            if exclude_gamefile and rec.get("task_id", "") and rec["task_id"] in exclude_gamefile:
                continue
            parts.append(store._format_case(len(parts) + 1, rec))
            if len(parts) >= self.mc_config.retrieve_num:
                break
        return "\n\n".join(parts)

    def _retrieve_text(self, task_desc: str, exclude_gamefile: str) -> str:
        """Top-k retrieved_text from the frozen pool with self-exclusion of the target gamefile.

        Mirrors CuratorAlfworld.retrieve's BM25 + _format_case rendering, but returns the
        retrieved MEMORY TEXT (not a curated briefing) because the curator LLM here is the
        trainable actor, not an external call. Self-excludes any record whose task_id matches
        the current target's gamefile-derived id (prevents answer-key leakage).
        """
        if self._pool is None or self._pool.bm25_retriever is None:
            return ""
        # Retrieve a few extra then drop self-matches, keep top retrieve_num.
        docs = self._pool.bm25_retriever.invoke(task_desc)
        parts: List[str] = []
        for doc in docs:
            idx = doc.metadata["idx"]
            rec = self._pool.memory_bank[idx]
            if exclude_gamefile and rec.get("task_id", "") and rec["task_id"] in exclude_gamefile:
                continue
            parts.append(self._pool._format_case(len(parts) + 1, rec))
            if len(parts) >= self.mc_config.retrieve_num:
                break
        return "\n\n".join(parts)

    # ------------------------------------------------------------------ #
    # Curator briefing generation (trainable actor; no tool execution)   #
    # ------------------------------------------------------------------ #
    def _generate_briefings(self, gen_batch_meta: dict, retrieved_texts: List[str],
                            task_descs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """Generate one briefing per slot with the actor policy.

        Reuses the two-pass thinking + loss-mask pattern from
        MemoryGenerationManager._process_chunk_with_memory_operations, but the "chunk" is the
        curator prompt built by build_curator_messages, and there is NO tool execution —
        the response IS the briefing (we _strip_think it only for the executor injection).

        Returns (chunk_input_ids, response_ids, response_mask, briefings_str).
        """
        device = self.actor_rollout_wg.device if hasattr(self.actor_rollout_wg, "device") else "cpu"

        # Build curator prompts as tokenized chunk ids via the shared Qwen pipeline.
        prompt_texts = []
        for task, rtext in zip(task_descs, retrieved_texts):
            messages = self._backend.build_curator_messages(task, rtext)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )
            prompt_texts.append(text)

        chunk_ids = self.tokenizer(prompt_texts, add_special_tokens=False,
                                   return_tensors="pt", padding="longest")["input_ids"]
        chunk_ids = chunk_ids.to(chunk_ids.device)

        empty_ids = torch.zeros((chunk_ids.shape[0], 0), dtype=torch.long, device=chunk_ids.device)
        rollings = DataProto.from_dict({
            "input_ids": empty_ids,
            "attention_mask": torch.ones_like(empty_ids),
            "position_ids": torch.zeros_like(empty_ids),
        })
        rollings.meta_info = dict(gen_batch_meta)
        empty_response = chunk_ids[:, :0]
        rollings = self._update_rolling_state(rollings, empty_response, chunk_ids)
        chunk_input_ids = rollings.batch["input_ids"]

        gen_output = self._generate_with_gpu_padding(rollings)
        responses_ids, responses_str = self._postprocess_responses(gen_output.batch["responses"])

        original_response_ids = [r for r in responses_ids]
        needs_masking = [False] * len(responses_str)

        if self.config.enable_thinking:
            sentence_ids = self.tokenizer.encode(_SENTENCE_TO_MASK, add_special_tokens=False)
            new_ids, new_str, original_response_ids, needs_masking = [], [], [], []
            for rid, rstr in zip(responses_ids, responses_str):
                if "</think>" not in rstr:
                    original_response_ids.append(rid)
                    rstr2 = rstr + _SENTENCE_TO_MASK
                    new_ids.append(self.tokenizer.encode(rstr2, return_tensors="pt")[0])
                    new_str.append(rstr2)
                    needs_masking.append(True)
                else:
                    original_response_ids.append(rid)
                    new_ids.append(rid)
                    new_str.append(rstr)
                    needs_masking.append(False)
            max_len = max(len(x) for x in new_ids)
            new_ids = [torch.cat([x, torch.tensor([self.tokenizer.pad_token_id] * (max_len - len(x)))])
                       for x in new_ids]
            new_ids = torch.stack(new_ids).long()
            rollings = self._update_rolling_state(rollings, empty_response, new_ids)
            gen_output_2 = self._generate_with_gpu_padding(rollings)
            _, cont_str = self._postprocess_responses(gen_output_2.batch["responses"])
            responses_str = [a + b for a, b in zip(new_str, cont_str)]
            sentence_ids_len = len(sentence_ids)
        else:
            sentence_ids_len = 0

        responses_ids = self._batch_tokenize(responses_str)
        responses_ids, responses_str = self.tensor_fn._example_level_pad(
            responses_ids, responses_str, torch.ones(responses_ids.shape[0], dtype=torch.bool)
        )
        response_mask = self.tensor_fn.create_attention_mask(responses_ids)
        if self.config.enable_thinking:
            for i, (orig, need) in enumerate(zip(original_response_ids, needs_masking)):
                if need:
                    ol = len(orig)
                    response_mask[i, ol: ol + sentence_ids_len] = 0

        briefings = [self._backend.strip_think(s) for s in responses_str]
        # dump-only passthroughs (Edit A): raw pre-strip curator output + the exact chat-template
        # prompt per slot (== rollout/'s `input`). Not used by training; threaded to _dump_step_log.
        curator_raw = list(responses_str)
        curator_prompts = list(prompt_texts)
        return chunk_input_ids, responses_ids, response_mask, briefings, curator_raw, curator_prompts

    # ------------------------------------------------------------------ #
    # Frozen executor episode (env_manager stepping + eval-parity prompt)#
    # ------------------------------------------------------------------ #
    def _run_executor_episodes(self, env_manager, briefings: List[str],
                               init_obs_dict: dict, init_infos: list,
                               task_descs: List[str]) -> Tuple[List[float], List[int], List[str], List[List[dict]]]:
        """Run one ALFWorld episode per slot with the briefing injected, on the ALREADY-RESET env.

        IMPORTANT: does NOT reset — the caller resets ONCE, generates the briefing for that exact
        task, then runs the executor on the SAME env state (otherwise the briefing/task mismatch).
        Uses env_manager only for parallel step(); builds each step's prompt with eval templates
        (memcurator.alfworld_executor) + our own per-slot history. Executor LLM via litellm (frozen).
        Returns (successes, steps, trajectories_text, exec_turns).

        ``exec_turns[i]`` is a per-slot list of per-step dicts
        {step, executor_prompt, executor_raw, action} — the FULL executor prompt (briefing injected)
        + the raw LLM response (with <think>), for the #2 dump. This is the missing 0/8 diagnostic:
        the parsed-action `trajectories` alone can't tell a format/parse failure from genuine
        incompetence, but the raw prompt+response can. Dump-only; unused by training.
        """
        from litellm import completion

        raw_obs = [process_ob(o) for o in init_obs_dict["anchor"]]
        admissible = list(env_manager.envs.get_admissible_commands)

        n = len(raw_obs)
        histories: List[List[tuple]] = [[] for _ in range(n)]
        traj_lines: List[List[str]] = [[] for _ in range(n)]
        exec_turns: List[List[dict]] = [[] for _ in range(n)]   # #2 dump: raw prompt+response per step
        successes = [0.0] * n
        steps_per = [self.mc_config.executor_max_steps] * n
        env_done = [False] * n

        for step_idx in range(self.mc_config.executor_max_steps):
            active = [i for i in range(n) if not env_done[i]]
            if not active:
                break
            prompts = {}
            for i in active:
                prompts[i] = build_executor_prompt(
                    step_count=len(histories[i]),
                    current_observation=raw_obs[i],
                    admissible_commands=admissible[i],
                    task_description=task_descs[i],
                    history=histories[i],
                    history_length=self.mc_config.history_length,
                    ctx_text=briefings[i] if briefings[i].strip() else "",
                    context_header=CURATOR_CONTEXT_HEADER,
                    context_label=CURATOR_CONTEXT_LABEL,
                )
            responses = self._call_executor_batch(completion, [prompts[i] for i in active])
            resp_by_slot = {i: responses[j] for j, i in enumerate(active)}

            # Build one action string per slot; inactive slots get a no-op the env ignores.
            actions = ["look"] * n
            for i in active:
                actions[i] = resp_by_slot[i]  # raw model output; env_manager's projection parses <action>
                act_parsed = parse_action(resp_by_slot[i]) or ""
                traj_lines[i].append(f"[Step {len(histories[i])}]")
                traj_lines[i].append(f"[Observation]: {raw_obs[i]}")
                traj_lines[i].append(f"[Action]: {act_parsed}")
                traj_lines[i].append("")
                exec_turns[i].append({
                    "step": step_idx,
                    "executor_prompt": prompts[i],       # full prompt incl. injected briefing
                    "executor_raw": resp_by_slot[i],     # raw LLM output (has <think>/<action>)
                    "action": act_parsed,
                })

            next_obs, rewards, dones, infos = env_manager.step(actions)
            next_raw = [process_ob(o) for o in next_obs["anchor"]]
            admissible = list(env_manager.envs.get_admissible_commands)

            for i in active:
                act_parsed = parse_action(resp_by_slot[i]) or ""
                histories[i].append((raw_obs[i], act_parsed))
                raw_obs[i] = next_raw[i]
                if bool(dones[i]):
                    env_done[i] = True
                    steps_per[i] = step_idx + 1
                    successes[i] = float(infos[i].get("won", False))

        trajectories = ["\n".join(lines) for lines in traj_lines]
        return successes, steps_per, trajectories, exec_turns

    def _call_executor_batch(self, completion_fn, prompts: List[str]) -> List[str]:
        """Call the frozen executor (litellm) on a batch of prompts. Mirrors eval's llm()."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        c = self.mc_config

        def _one(prompt: str) -> str:
            kwargs = dict(
                model=c.executor_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=c.executor_api_key,
                base_url=c.executor_api_base,
                num_retries=10,
                temperature=c.executor_temperature,
            )
            if c.executor_top_p is not None:
                kwargs["top_p"] = c.executor_top_p
            if c.executor_max_tokens is not None:
                kwargs["max_tokens"] = c.executor_max_tokens
            extra_body = {}
            if c.executor_top_k is not None:
                extra_body["top_k"] = c.executor_top_k
            if c.executor_enable_thinking is not None:
                extra_body["chat_template_kwargs"] = {"enable_thinking": c.executor_enable_thinking}
            if extra_body:
                kwargs["extra_body"] = extra_body
            try:
                resp = completion_fn(**kwargs)
                return resp.choices[0].message.content or "Output Error"
            except Exception as e:  # noqa: BLE001
                return f"Output Error: {e}"

        out = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=max(1, len(prompts))) as pool:
            futs = {pool.submit(_one, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out

    # ------------------------------------------------------------------ #
    # Main entry — one training step                                     #
    @contextlib.contextmanager
    def _phase(self, name: str, global_steps: int, is_validation: bool):
        """Time + print a generation sub-phase (env pin/reset, curator rollout, executor rollout,
        assemble). Streams live (PYTHONUNBUFFERED=1) so progress is visible and slow phases are
        obvious rather than looking like hangs. fit() already times gen/reward/adv/update separately.
        """
        tag = "VAL" if is_validation else "TRAIN"
        print(f"[MemCurator][{tag} step {global_steps}] ▶ {name} ...", flush=True)
        t0 = time.time()
        try:
            yield
        finally:
            print(f"[MemCurator][{tag} step {global_steps}] ✔ {name} done in {time.time() - t0:.1f}s",
                  flush=True)

    # ------------------------------------------------------------------ #
    def run_memory_loop_alfworld(
        self,
        gen_batch: DataProto,
        env_manager,
        num_tasks: int = 1,
        num_gpus: int = 1,
        global_steps: int = 0,
        is_validation: bool = False,
    ) -> DataProto:
        """One MemCurator GRPO step. DIRECT credit (reward = current-task success), single task
        per slot (no chains). Signature matches AlfWorldGenerationManager.run_memory_loop_alfworld
        so the trainer's fit()/_validate() call sites work UNCHANGED; ``num_tasks`` is accepted for
        signature-compat but ignored (MemCurator is single-task-per-step, not a chain).
        """
        if num_tasks not in (None, 1):
            print(f"[MemCurator] note: num_tasks={num_tasks} ignored (single-task-per-step design).")
        total_batch_size = gen_batch.batch["input_ids"].shape[0]
        n_rollouts = getattr(self, "_n_rollouts", None) or 1
        _tag = "VAL" if is_validation else "TRAIN"
        print(f"\n[MemCurator][{_tag} step {global_steps}] ===== rollout: {total_batch_size} slots "
              f"= {total_batch_size // n_rollouts} targets x {n_rollouts} rollouts =====", flush=True)
        _t_step = time.time()

        # --- Select targets + retrieval stores for this step (env pin/reset + BM25 retrieve) ---
        with self._phase("env pin+reset+retrieve", global_steps, is_validation):
            if self._dataset is not None:
                # DATASET-DRIVEN + PINNED: take next targets (each repeated n_rollouts, interleaved),
                # pin each slot to its target game, then reset → slot i serves its assigned target.
                targets = self._next_target_batch(total_batch_size, n_rollouts)
                assert len(targets) == total_batch_size, \
                    f"target batch {len(targets)} != batch {total_batch_size} (n_rollouts={n_rollouts})"
                env_manager.set_slot_game_files([t["game_file"] for t in targets])
                obs_dict, infos = env_manager.reset({})
                task_descs = list(env_manager.tasks)
                # retrieval text comes from EACH target's own frozen S_T (per-target pool).
                retrieved_texts = []
                for i, t in enumerate(targets):
                    store = self._store_for(t.get("store_path"))
                    retrieved_texts.append(self._retrieve_from_store(store, task_descs[i], t.get("game_file", "")))
            else:
                # FALLBACK global-pool + random reset (superseded path).
                targets = None
                obs_dict, infos = env_manager.reset({})
                task_descs = list(env_manager.tasks)
                gamefiles = [str(gf) for gf in env_manager.gamefile]
                retrieved_texts = [self._retrieve_text(task_descs[i], gamefiles[i]) for i in range(total_batch_size)]

        # --- generate one briefing per slot (trainable actor) ---
        with self._phase("curator rollout (briefing gen)", global_steps, is_validation):
            (chunk_input_ids, response_ids, response_mask, briefings,
             curator_raw, curator_prompts) = self._generate_briefings(
                gen_batch.meta_info, retrieved_texts, task_descs
            )

        # --- run the executor on the SAME reset state; reward = task success (DIRECT credit) ---
        with self._phase("executor rollout (episodes)", global_steps, is_validation):
            successes, steps, trajectories, exec_turns = self._run_executor_episodes(
                env_manager, briefings, obs_dict, infos, task_descs
            )
        n_ok = sum(1 for s in successes if s)
        print(f"[MemCurator][{'VAL' if is_validation else 'TRAIN'} step {global_steps}] "
              f"executor success: {n_ok}/{len(successes)} "
              f"(mean_steps={sum(steps)/max(1,len(steps)):.1f})", flush=True)

        with self._phase("assemble DataProto", global_steps, is_validation):
            _out = self._assemble_output(
                chunk_input_ids=chunk_input_ids,
                response_ids=response_ids,
                response_mask=response_mask,
                briefings=briefings,
                retrieved_texts=retrieved_texts,
                task_descs=task_descs,
                mean_success=[float(s) for s in successes],
                steps=list(steps),
                trajectories=trajectories,
                exec_turns=exec_turns,
                curator_raw=curator_raw,
                curator_prompts=curator_prompts,
                gen_batch=gen_batch,
                num_gpus=num_gpus,
                global_steps=global_steps,
                is_validation=is_validation,
            )
        print(f"[MemCurator][{_tag} step {global_steps}] ===== generation total "
              f"{time.time() - _t_step:.1f}s (reward+advantage+PPO update timed separately by fit) =====",
              flush=True)
        return _out

    def _assemble_output(self, *, chunk_input_ids, response_ids, response_mask, briefings,
                         retrieved_texts, task_descs, mean_success, steps, trajectories,
                         gen_batch, num_gpus, global_steps, is_validation,
                         exec_turns=None, curator_raw=None, curator_prompts=None) -> DataProto:
        total_batch_size = len(mean_success)
        pad = self.tokenizer.pad_token_id

        all_input_ids = [chunk_input_ids[i] for i in range(total_batch_size)]
        all_response_ids = [response_ids[i] for i in range(total_batch_size)]
        all_response_masks = [response_mask[i] for i in range(total_batch_size)]
        indices_in_batch = list(range(total_batch_size))
        task_position_in_batch = [0] * total_batch_size  # single task per slot (no chains)

        # DIRECT credit: reward = this slot's own mean task success (NO [1:] shift).
        per_chunk_rewards = [float(s) for s in mean_success]
        per_chunk_successes = [float(s) for s in mean_success]
        # Briefing format-validity as the "function_call" analogue (1.0 if non-empty briefing).
        all_function_call_rewards = [1.0 if b.strip() else 0.0 for b in briefings]
        all_function_calls = [[] for _ in range(total_batch_size)]
        # SkillOS's generation emits per-slot skill-library snapshots as "batch_memories"; the shared
        # _validate() reads meta_info['batch_memories'] UNGUARDED (ray_trainer_alfworld.py:292) and
        # would KeyError on the full run's validation. MemCurator has no skill library, so emit an
        # empty list per slot (only extend()'d into a logging list that's never read downstream).
        batch_memories = [[] for _ in range(total_batch_size)]
        # MemCurator has NO content reward (no judge; function_content_reward_weight=0). But the
        # shared naive.py alfworld branch ALWAYS returns "all_function_call_content_rewards" in
        # reward_extra_info, and if we don't provide it, naive.py passes None → fit()'s per-slot
        # metrics loop does np.array(None) (0-d) and crashes with "too many indices for array".
        # Emit it as zeros (weight 0 → no effect on reward) so the shared metrics aggregation works.
        all_function_call_content_rewards = [0.0 for _ in range(total_batch_size)]

        # Left-pad prompts, right-pad responses (identical to generation_alfworld).
        max_in = max(len(x) for x in all_input_ids)
        all_input_ids = [
            torch.cat([torch.tensor([pad] * (max_in - len(x))), x]) if len(x) < max_in else x
            for x in all_input_ids
        ]
        max_out = max(len(x) for x in all_response_ids)
        new_resp, new_mask = [], []
        for rid, rm in zip(all_response_ids, all_response_masks):
            if len(rid) < max_out:
                new_resp.append(torch.cat([rid, torch.tensor([pad] * (max_out - len(rid)))]))
                new_mask.append(torch.cat([rm, torch.tensor([False] * (max_out - len(rm)))]))
            else:
                new_resp.append(rid)
                new_mask.append(rm)
        all_response_ids, all_response_masks = new_resp, new_mask

        final = {
            "prompts": torch.stack(all_input_ids),
            "responses": torch.stack(all_response_ids),
            "response_mask": torch.stack(all_response_masks),
        }
        final["input_ids"] = torch.cat([final["prompts"], final["responses"]], dim=1)
        final["attention_mask"] = torch.where(final["input_ids"] != pad, 1, 0)
        final["position_ids"] = self.tensor_fn.create_position_ids(final["attention_mask"])
        final["attention_mask"][:, -final["response_mask"].shape[1]:] = final["response_mask"]

        # GPU padding (repeat rows) to a multiple of num_gpus — mirror generation_alfworld.
        if num_gpus > 1:
            cur = final["input_ids"].shape[0]
            need = ((cur + num_gpus - 1) // num_gpus) * num_gpus - cur
            if need > 0:
                rep = torch.arange(cur)[:need]
                for key in ["input_ids", "attention_mask", "position_ids", "prompts", "responses", "response_mask"]:
                    final[key] = torch.cat([final[key], final[key][rep]], dim=0)
                for lst in (all_function_call_rewards, all_function_call_content_rewards,
                            per_chunk_rewards, per_chunk_successes):
                    lst.extend([lst[i] for i in rep.tolist()])
                all_function_calls.extend([all_function_calls[i] for i in rep.tolist()])
                batch_memories.extend([batch_memories[i] for i in rep.tolist()])
                indices_in_batch.extend([indices_in_batch[i] for i in rep.tolist()])
                task_position_in_batch.extend([task_position_in_batch[i] for i in rep.tolist()])

        # non-degenerate-group fraction is computed in the trainer/reward (needs uids); here we
        # just log per-step success + briefing stats to the rollout dir.
        self._dump_step_log(global_steps, is_validation, task_descs, retrieved_texts,
                            briefings, mean_success, trajectories,
                            exec_turns=exec_turns, curator_raw=curator_raw,
                            curator_prompts=curator_prompts)

        total_memory_length = [count_tokens(b) for b in briefings]  # briefing length (compression term, v2)
        total_chunk_length = [count_tokens(t + r) for t, r in zip(task_descs, retrieved_texts)]

        out = DataProto.from_dict(final)
        out.meta_info.update({
            "indices_in_batch": indices_in_batch,
            "task_position_in_batch": task_position_in_batch,
            "per_chunk_rewards": per_chunk_rewards,
            "per_chunk_successes": per_chunk_successes,
            "all_function_call_rewards": all_function_call_rewards,
            "all_function_call_content_rewards": all_function_call_content_rewards,
            "all_function_calls": all_function_calls,
            "batch_memories": batch_memories,   # empty per slot; _validate() reads this unguarded
            "total_chunk_length": total_chunk_length,
            "total_memory_length": total_memory_length,
            # trainer logs these directly:
            "successes_list": [list(mean_success)],
            "rewards_list": [list(mean_success)],
            "steps_list": [list(steps)],
            "briefings": briefings,
        })
        return out

    def _dump_step_log(self, global_steps, is_validation, task_descs, retrieved_texts,
                       briefings, mean_success, trajectories,
                       exec_turns=None, curator_raw=None, curator_prompts=None) -> None:
        """One-stop per-slot debug dump (#2). Additive fields default to None so any caller that
        doesn't pass them still works. Size-gated by MEMCURATOR_DUMP_FULL (default on): full =
        untruncated curator prompt/raw + full executor prompt+raw per step (the 0/8 diagnostic);
        MEMCURATOR_DUMP_FULL=0 falls back to the lean 800-char clips for the full training run.
        """
        dump_path = os.getenv("ROLLOUT_DATA_DIR", ".") + (
            "/generation/validation" if is_validation else "/generation/training"
        )
        os.makedirs(dump_path, exist_ok=True)
        full = os.getenv("MEMCURATOR_DUMP_FULL", "1") != "0"
        cap = None if full else 800

        def _clip(s):
            return s if (cap is None or s is None) else s[:cap]

        # TRUNCATE ("w"), not append ("a"): the dump is keyed by global_steps, so one step = one
        # file. Append DUPLICATED a step's records whenever the same step ran twice (a re-run or a
        # crash-and-restart that didn't clean the dir) — e.g. step-1 showed 16 rows for an 8-slot
        # step. The verl training tensors were unaffected (this dump is log-only), but the analysis
        # dumps we debug from were silently corrupted. Rewriting per step is correct + idempotent.
        with open(os.path.join(dump_path, f"{global_steps}.jsonl"), "w") as f:
            for i in range(len(task_descs)):
                rec = {
                    "task": task_descs[i],
                    "retrieved_text": _clip(retrieved_texts[i]) if full else retrieved_texts[i][:800],
                    # curator side: exact prompt the actor saw (== rollout/'s input) + raw pre-strip
                    # response (has <think>) + the stripped briefing actually injected into executor.
                    "curator_prompt": (curator_prompts[i] if curator_prompts else None),
                    "curator_raw": (_clip(curator_raw[i]) if curator_raw else None),
                    "briefing": briefings[i],
                    # executor side: parsed per-step [Obs]/[Action] text + FULL per-step turns
                    # (prompt incl. injected briefing + raw LLM response) — the 0/8 diagnostic.
                    "trajectory": trajectories[i] if i < len(trajectories) else None,
                    "executor_turns": (exec_turns[i] if exec_turns else None),
                    "mean_success": mean_success[i],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
