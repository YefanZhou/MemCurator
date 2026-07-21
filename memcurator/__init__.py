"""memcurator — RL training for the read-time MemCurator (briefing) curator.

This is a NEW top-level package (sibling to ``skillos/``). It trains the read-time
briefing curator that ``evaluation/agent_eval/run_unified_dev_async_curator.py
--memory_type curator`` runs at eval time (``curator_alfworld.CuratorAlfworld``).

Why a separate package (not inside ``skillos/``): "skillos" names the *write-time
skill-editing* curator, a different method. Keeping MemCurator code separate avoids
mislabeling. Shared helpers are reused by import from ``skillos`` (e.g.
``skillos.llm_agent.tensor_helper``, ``skillos.utils``).

Design (see .claude/plans plan file for the full rationale):
  * Curator = trainable policy; emits ONE briefing per task from BM25-retrieved
    successful trajectories (prompt = ``curator_alfworld.build_curator_messages``).
  * Executor = FROZEN; reuses the eval executor prompt/parse (ReAct <think>/<action>,
    ALFWORLD_TEMPLATE*_WITH_CONTEXT) so train == eval by construction.
  * Reward = DIRECT (executor success on the current task; no shift), mean over K
    executor rollouts to denoise binary success.
  * Store = OFFLINE frozen per-task snapshots S_T (read-only during training).

Existing files are reused via import / surgical config-gated edits (default = skillos,
so the SkillOS reproduction path stays byte-identical).
"""
