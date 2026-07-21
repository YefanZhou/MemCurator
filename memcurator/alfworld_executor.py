"""ALFWorld executor prompt + action parsing for MemCurator training — EVAL-PARITY copy.

These templates and helpers are copied VERBATIM from
``evaluation/agent_eval/run_unified_dev_async_curator.py`` so that the frozen executor
sees a byte-identical prompt at train time and eval time. We copy (not import) because
the eval runner is a script under ``evaluation/agent_eval/`` (not on the package path)
and importing it pulls heavy optional deps; ``memcurator/tests/test_executor_parity.py``
asserts these strings stay byte-identical to the eval source so drift is caught.

Parity contract (verified against the eval runner):
  * ``ALFWORLD_TEMPLATE_NO_HIS`` / ``..._NO_HIS_WITH_CONTEXT`` — step 0 (no history).
  * ``ALFWORLD_TEMPLATE`` / ``..._WITH_CONTEXT`` — steps >= 1 (rolling history).
  * ``_apply_prompt_style`` rewrites the <think> mandate by ``PROMPT_STYLE``.
  * ``format_action_history`` / ``process_ob`` — identical helpers.
  * The curator briefing is injected via the ``{context_header}`` slot; for
    ``--memory_type curator`` the eval default header is
    "Here are past experiences and trajectories that might be helpful for your decision:"
    and ``context_label`` is "experiences".
  * Action is parsed with ``response.split('<action>')[-1].split('</action>')[0].strip()``
    (matches the eval runner). Note: ``env_manager`` uses ``alfworld_projection`` which
    does the equivalent extraction, so passing the raw model output to ``env.step`` also works.
"""

from __future__ import annotations

import os
from typing import List, Optional

# ---- prompt style (env-driven) ----
# NOTE: default is 'revise_react' (NOT the eval runner's module default of 'think'). MemCurator
# training must match the frac0.5 harvest that built the pool, which ran PROMPT_STYLE=revise_react
# + ENABLE_THINKING=false. Using 'think' with a non-thinking executor collides (think-mandate +
# no-think prefill -> degenerate output; the eval runner even warns about this). The launcher sets
# PROMPT_STYLE explicitly (shell + runtime_env env_vars); this default is only a safety net so a
# forgotten flag still yields the correct nonthink-parity prompt. The _apply_prompt_style transform
# itself is byte-identical to eval (guarded by tests/test_executor_parity.py).
PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "revise_react").lower()
_THINK_MANDATE = "This reasoning process MUST be enclosed within <think> </think> tags."


def _apply_prompt_style(tmpl: str) -> str:
    """Rewrite the reasoning-tag mandate in a template per PROMPT_STYLE (verbatim from eval)."""
    if PROMPT_STYLE == "think":
        return tmpl
    if PROMPT_STYLE == "reason_tag":
        return tmpl.replace("<think> </think>", "<reason> </reason>")
    # revise_react: drop the mandate sentence (and the space that precedes it)
    return tmpl.replace(" " + _THINK_MANDATE, "")


# ---- default context framing for --memory_type curator (from the eval runner) ----
CURATOR_CONTEXT_HEADER = (
    "Here are past experiences and trajectories that might be helpful for your decision:"
)
CURATOR_CONTEXT_LABEL = "experiences"


# ==================================================================== #
# Templates — copied verbatim from run_unified_dev_async_curator.py    #
# ==================================================================== #

ALFWORLD_TEMPLATE_NO_HIS = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\
"""

# Step-0 with retrieved context (no {task_description}: task rides inside {current_observation}).
ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT = """\
You are an expert agent operating in the ALFRED Embodied Environment.

{context_header}

{retrieved_context}

## Current Progress

Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant {context_label_lower}. This reasoning process MUST be enclosed within <think> </think> tags.
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

{context_header}

{retrieved_context}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant {context_label_lower}. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.\
"""

# Apply PROMPT_STYLE (identical to the eval runner's post-definition rewrite).
ALFWORLD_TEMPLATE_NO_HIS              = _apply_prompt_style(ALFWORLD_TEMPLATE_NO_HIS)
ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT = _apply_prompt_style(ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT)
ALFWORLD_TEMPLATE                     = _apply_prompt_style(ALFWORLD_TEMPLATE)
ALFWORLD_TEMPLATE_WITH_CONTEXT        = _apply_prompt_style(ALFWORLD_TEMPLATE_WITH_CONTEXT)


# ==================================================================== #
# Helpers — copied verbatim from run_unified_dev_async_curator.py      #
# ==================================================================== #

def process_ob(ob: str) -> str:
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2:]
    return ob


def format_action_history(history: List[tuple], history_length: int) -> str:
    recent = history[-history_length:]
    if not recent:
        return "None"
    lines = []
    for i, (obs, action) in enumerate(recent, 1):
        lines.append(f"Observation {i}: {obs}")
        lines.append(f"Action {i}: {action}")
    return "\n".join(lines)


def build_executor_prompt(
    *,
    step_count: int,
    current_observation: str,
    admissible_commands: List[str],
    task_description: str,
    history: List[tuple],
    history_length: int,
    ctx_text: str = "",
    context_header: str = CURATOR_CONTEXT_HEADER,
    context_label: str = CURATOR_CONTEXT_LABEL,
) -> str:
    """Build ONE executor prompt string, reproducing alfworld_run_batch's branching exactly.

    Branching (identical to the eval runner):
      * step_count == 0 and ctx_text  -> NO_HIS_WITH_CONTEXT (task rides in current_observation)
      * step_count == 0 and no ctx    -> NO_HIS
      * step_count >= 1 and ctx_text  -> WITH_CONTEXT
      * step_count >= 1 and no ctx    -> TEMPLATE
    ``admissible_commands`` is formatted with the eval's exact join (excluding 'help').
    ``history`` is a list of (obs, action) tuples; only the last ``history_length`` are shown.
    """
    admissible_str = "\n ".join(f"'{s}'" for s in admissible_commands if s != "help")
    label_lower = context_label.lower()

    if step_count == 0:
        if ctx_text:
            return ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT.format(
                context_header=context_header,
                retrieved_context=ctx_text,
                context_label=context_label,
                context_label_lower=label_lower,
                current_observation=current_observation,
                admissible_actions=admissible_str,
            )
        return ALFWORLD_TEMPLATE_NO_HIS.format(
            current_observation=current_observation,
            admissible_actions=admissible_str,
        )

    if ctx_text:
        return ALFWORLD_TEMPLATE_WITH_CONTEXT.format(
            task_description=task_description,
            context_header=context_header,
            retrieved_context=ctx_text,
            context_label=context_label,
            context_label_lower=label_lower,
            step_count=step_count,
            history_length=min(history_length, step_count),
            action_history=format_action_history(history, history_length),
            current_step=step_count + 1,
            current_observation=current_observation,
            admissible_actions=admissible_str,
        )
    return ALFWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=step_count,
        history_length=min(history_length, step_count),
        action_history=format_action_history(history, history_length),
        current_step=step_count + 1,
        current_observation=current_observation,
        admissible_actions=admissible_str,
    )


def parse_action(response: str) -> Optional[str]:
    """Extract the action from an executor response (eval-identical parse).

    Returns None if no ``<action>...</action>`` block is present (eval leaves the action
    empty in that case).
    """
    if "<action>" in response and "</action>" in response:
        return response.split("<action>")[-1].split("</action>")[0].strip()
    return None
