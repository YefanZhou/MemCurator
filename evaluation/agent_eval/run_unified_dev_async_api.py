"""
run_unified_dev_async.py  —  group episode-async variant of run_unified_dev.py.

Same results as run_unified_dev.py with ANY memory method, but the ALFWorld loop runs
each game in a group of --batch_size CONCURRENTLY to completion (LLM call outside a
PDDL lock), with a barrier at the group boundary for memory retrieve/update. This
removes run_unified_dev.py's per-step barrier (the speed bottleneck) WITHOUT changing
memory semantics: memory is frozen within a group (retrieved single-threaded before the
group, updated only at the barrier) and both game order and update order are driven by the
SAME batched env.reset() stream the batch runner uses — identical to the batch runner's
retrieve-once / update-after-group contract by construction.

Concurrency is PINNED to --batch_size (= memory-update granularity); there is no separate
--concurrency knob, because decoupling it would change when memory updates fire.

Only the ALFWorld path is async. WebShop (batch-synchronous env) and reasoning (single-
turn) paths are byte-identical to run_unified_dev.py.

Carries over all run_unified_dev.py features: step-0 fix ([1:] slice), step-0 context
injection, PROMPT_STYLE, EXECUTOR_*/CURATION_* knobs, SAVE_RAW=N, timing.
See bug_fix_markdown/DEV_HISTORY_RUNNER_LINEAGE.md for details.

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
import time
import copy
import threading
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

HISTORY_LENGTH = int(os.environ.get("HISTORY_LENGTH", "5"))

REASONING_ENVS = {'amc23', 'aime24', 'aime25', 'gpqa'}

# ------------------------------------------------------------------ #
# Executor sampling hyperparameters (overridable via env vars)        #
# Mirrors run_unified_hyper_async_step0bug_fix.py so the same knobs    #
# work here. Only the EXECUTOR gameplay path reads these; memory       #
# curation is unchanged (curator thinking is decided by the curation   #
# server's chat template).                                             #
# ------------------------------------------------------------------ #
def _env_float(name):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None

def _env_int(name):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else None

EXECUTOR_TEMPERATURE = _env_float("EXECUTOR_TEMPERATURE")
if EXECUTOR_TEMPERATURE is None:
    EXECUTOR_TEMPERATURE = 0.7
EXECUTOR_TOP_P      = _env_float("EXECUTOR_TOP_P")
EXECUTOR_TOP_K      = _env_int("EXECUTOR_TOP_K")
EXECUTOR_MAX_TOKENS = _env_int("EXECUTOR_MAX_TOKENS")

# ENABLE_THINKING: unset -> server default / not sent; "true"/"false" forces it.
_ET = os.environ.get("ENABLE_THINKING", "")
ENABLE_THINKING = None if _ET == "" else (_ET.lower() in ("1", "true", "yes"))

# ------------------------------------------------------------------ #
# Prompt reasoning-instruction style (unifies the two async variants) #
#   PROMPT_STYLE = think        : "...MUST be enclosed within <think>  #
#                                 </think> tags." (default; pairs with #
#                                 ENABLE_THINKING=true — the mandate    #
#                                 matches Qwen3's thinking mode).       #
#                  reason_tag   : same mandate but with <reason></reason>#
#                                 (no collision with a no-think prefill)#
#                  revise_react : reasoning-tag mandate removed         #
#                                 (plain-text reasoning; for no-think). #
# Mirrors run_unified_hyper_async_step0bug_fix.py (think) and           #
# run_unified_hyper_async_revise_react_step0bug_fix.py (revise_react).  #
# ------------------------------------------------------------------ #
PROMPT_STYLE = os.environ.get("PROMPT_STYLE", "think").lower()
if PROMPT_STYLE not in ("think", "reason_tag", "revise_react"):
    raise ValueError(
        f"Invalid PROMPT_STYLE={PROMPT_STYLE!r}; "
        f"use 'think', 'reason_tag', or 'revise_react'."
    )

_THINK_MANDATE = "This reasoning process MUST be enclosed within <think> </think> tags."

def _apply_prompt_style(tmpl: str) -> str:
    """Rewrite the reasoning-tag mandate in a template per PROMPT_STYLE."""
    if PROMPT_STYLE == "think":
        return tmpl
    if PROMPT_STYLE == "reason_tag":
        return tmpl.replace("<think> </think>", "<reason> </reason>")
    # revise_react: drop the mandate sentence (and the space that precedes it)
    return tmpl.replace(" " + _THINK_MANDATE, "")

print(f"[executor hyperparams] temperature={EXECUTOR_TEMPERATURE} top_p={EXECUTOR_TOP_P} "
      f"top_k={EXECUTOR_TOP_K} max_tokens={EXECUTOR_MAX_TOKENS} "
      f"enable_thinking={ENABLE_THINKING} history_length={HISTORY_LENGTH} "
      f"prompt_style={PROMPT_STYLE}")

# Pairing sanity check (see bug_fix_markdown/SESSION_FINDINGS_AND_GOTCHAS.md):
# the <think> mandate collides with a no-think prefill -> degenerate output.
if PROMPT_STYLE == "think" and ENABLE_THINKING is False:
    print("\033[93m[WARNING] PROMPT_STYLE=think with ENABLE_THINKING=false collides "
          "(Qwen3 no-think prefill + <think> mandate -> garbage). Use "
          "PROMPT_STYLE=reason_tag or revise_react for no-think.\033[0m")

# ------------------------------------------------------------------ #
# Curation sampling hyperparameters (overridable via env vars)        #
# Symmetric to the EXECUTOR_* knobs, but for the memory-curation LLM   #
# (skillos + reasoningbank). Defaults preserve prior behaviour:        #
# temperature 0.7, top_p/top_k unset, max_tokens unset (per-path       #
# default kept), thinking = curation server's chat-template default.   #
# ------------------------------------------------------------------ #
CURATION_TEMPERATURE = _env_float("CURATION_TEMPERATURE")
if CURATION_TEMPERATURE is None:
    CURATION_TEMPERATURE = 0.7
CURATION_TOP_P      = _env_float("CURATION_TOP_P")
CURATION_TOP_K      = _env_int("CURATION_TOP_K")
CURATION_MAX_TOKENS = _env_int("CURATION_MAX_TOKENS")

# CURATION_ENABLE_THINKING: unset -> server default / not sent; "true"/"false" forces it.
_CT = os.environ.get("CURATION_ENABLE_THINKING", "")
CURATION_ENABLE_THINKING = None if _CT == "" else (_CT.lower() in ("1", "true", "yes"))

print(f"[curation hyperparams] temperature={CURATION_TEMPERATURE} top_p={CURATION_TOP_P} "
      f"top_k={CURATION_TOP_K} max_tokens={CURATION_MAX_TOKENS} "
      f"enable_thinking={CURATION_ENABLE_THINKING}")

# ------------------------------------------------------------------ #
# LLM backend switch (external API gateway vs local vLLM).            #
# _api runner only. Two INDEPENDENT switches so executor and curator  #
# can use different backends (e.g. Qwen executor on vLLM + GPT        #
# curator on the Salesforce gateway):                                 #
#   LLM_BACKEND          = vllm (default) | openai  -> _EXEC_EXTERNAL  #
#   CURATION_LLM_BACKEND = vllm (default) | openai  -> _CUR_EXTERNAL   #
# When external, the vLLM-only extra_body fields (top_k,              #
# chat_template_kwargs/enable_thinking) are NEVER sent — the OpenAI-   #
# compatible gateway rejects them. If X_API_KEY is set it is attached  #
# as the X-Api-Key header (Salesforce gateway); harmless otherwise.    #
# ------------------------------------------------------------------ #
LLM_BACKEND          = os.environ.get("LLM_BACKEND", "vllm").lower()
CURATION_LLM_BACKEND = os.environ.get("CURATION_LLM_BACKEND", "vllm").lower()
_EXEC_EXTERNAL = (LLM_BACKEND == "openai")
_CUR_EXTERNAL  = (CURATION_LLM_BACKEND == "openai")
_X_API_KEY = os.environ.get("X_API_KEY") or None

def _gateway_extra_headers():
    """X-Api-Key header for the Salesforce gateway, if X_API_KEY is set (else None)."""
    return {"X-Api-Key": _X_API_KEY} if _X_API_KEY else None

def _set_max_tokens(kwargs, value, external):
    """Set the token-limit kwarg. The gpt-5 family (reasoning models) rejects `max_tokens` and
    requires `max_completion_tokens`; vLLM/OpenAI-chat models use `max_tokens`. Pick per backend."""
    if value is None:
        return
    kwargs["max_completion_tokens" if external else "max_tokens"] = value

# ------------------------------------------------------------------ #
# Vertex AI (gemini/) helpers — for gemini/<model> executor + curator.  #
# GOOGLE_CLOUD_PROJECT/LOCATION default to the Salesforce Vertex project #
# (matches tests/test_api_model.py) if the env vars are unset, so a       #
# gemini run works without extra exports.                                 #
# ------------------------------------------------------------------ #
GCLOUD_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT")  or "salesforce-research-internal"
GCLOUD_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"

def _vertex_client():
    from google import genai
    return genai.Client(vertexai=True, project=GCLOUD_PROJECT, location=GCLOUD_LOCATION)

def _openai_tools_to_vertex(tool_schemas):
    """Convert OpenAI tool schemas (MEMORY_TOOL_SCHEMAS) to a Vertex types.Tool.
    JSON-schema types are upper-cased (STRING/OBJECT/...) as the Vertex SDK expects."""
    from google.genai import types
    _MAP = {"string": "STRING", "object": "OBJECT", "array": "ARRAY",
            "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}
    def _conv(node):
        if not isinstance(node, dict):
            return node
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = _MAP.get(v.lower(), v.upper())
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: _conv(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                out[k] = _conv(v)
            else:
                out[k] = v
        return out
    decls = []
    for tool in tool_schemas:
        fn = tool["function"]
        decls.append(types.FunctionDeclaration(
            name=fn["name"], description=fn.get("description", ""),
            parameters=_conv(fn.get("parameters", {"type": "object", "properties": {}})),
        ))
    return types.Tool(function_declarations=decls)

# MAX_CONCURRENCY: optional cap on concurrent executor threads per group. Defaults to
# batch_size (unchanged behaviour). Must be <= batch_size to preserve memory-group semantics.
MAX_CONCURRENCY = _env_int("MAX_CONCURRENCY")

print(f"[backend] executor={LLM_BACKEND} (external={_EXEC_EXTERNAL})  "
      f"curation={CURATION_LLM_BACKEND} (external={_CUR_EXTERNAL})  "
      f"x_api_key={'set' if _X_API_KEY else 'unset'}  max_concurrency={MAX_CONCURRENCY}")

# ------------------------------------------------------------------ #
# Timing + periodic-print knobs (mirrors                              #
# run_unified_hyper_async_step0bug_fix.py)                            #
#   PROMPT_SHOW_EVERY=N : every N steps, print the probe task's full  #
#                         PROMPT + RESPONSE. 0 (default) -> only the  #
#                         probe's step-0 response is printed.         #
#   PRINT_CHARS=N       : truncate printed RESPONSES to N chars       #
#                         (0 = no truncation). Prompts print in full   #
#                         so injected task/context is verifiable.      #
# The probe is the FIRST active task in each batch (batch runners) /  #
# the first problem in each mini-batch (reasoning).                   #
# ------------------------------------------------------------------ #
PROMPT_SHOW_EVERY = _env_int("PROMPT_SHOW_EVERY") or 0
PRINT_CHARS       = _env_int("PRINT_CHARS") or 0

# SAVE_RAW=N : persist the FULL injected prompt + raw executor response per step
# (`raw_trace` key) for N sample games EVENLY SPREAD across the run — e.g. SAVE_RAW=10
# over 140 games keeps 10 fat traces and lean files for the rest. 0/unset = off.
# The basic messages/reward/name are saved for EVERY game regardless. Adds no LLM calls;
# just keeps strings already in memory and writes a larger json for the sampled games.
SAVE_RAW_N = _env_int("SAVE_RAW") or 0
SAVE_RAW = SAVE_RAW_N > 0   # gates in-memory collection inside the batch runners


def _raw_save_indices(total, n):
    """Return a set of `n` game indices evenly spread across [0, total-1] (inclusive
    endpoints). n<=0 -> none; n>=total -> all. Used to pick which games persist raw_trace."""
    if n <= 0 or total <= 0:
        return set()
    if n >= total:
        return set(range(total))
    if n == 1:
        return {0}
    return {round(k * (total - 1) / (n - 1)) for k in range(n)}


# ------------------------------------------------------------------ #
# Per-run config dump + self-logging (into the results folder)        #
# ------------------------------------------------------------------ #
# Env vars whose values define an experiment. Captured verbatim into run_config.json so a
# result folder is self-describing (no need to reconstruct the launch command from history).
_TRACKED_ENV_VARS = [
    "OPENAI_API_BASE", "OPENAI_API_KEY", "API_BASE_URL", "MODEL_NAME",
    "ALFWORLD_DATA", "TMPDIR", "HISTORY_LENGTH",
    "EXECUTOR_TEMPERATURE", "EXECUTOR_TOP_P", "EXECUTOR_TOP_K", "EXECUTOR_MAX_TOKENS",
    "ENABLE_THINKING", "PROMPT_STYLE",
    "CURATION_TEMPERATURE", "CURATION_TOP_P", "CURATION_TOP_K", "CURATION_MAX_TOKENS",
    "CURATION_ENABLE_THINKING",
    "PROMPT_SHOW_EVERY", "PRINT_CHARS", "SAVE_RAW",
    "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
    "LLM_BACKEND", "CURATION_LLM_BACKEND", "X_API_KEY", "MAX_CONCURRENCY",
]


def dump_run_config(output_path, args, extra=None):
    """Write run_config.json into output_path: CLI args + RESOLVED hyperparams + tracked env
    vars, so every result folder records exactly how it was produced. Masks the API key."""
    def _mask(k, v):
        return ("****" if v else v) if (v is not None and "KEY" in k) else v
    cfg = {
        "runner": os.path.basename(__file__),
        "args": vars(args),
        "resolved_hyperparams": {
            "executor": {"temperature": EXECUTOR_TEMPERATURE, "top_p": EXECUTOR_TOP_P,
                         "top_k": EXECUTOR_TOP_K, "max_tokens": EXECUTOR_MAX_TOKENS,
                         "enable_thinking": ENABLE_THINKING, "prompt_style": PROMPT_STYLE,
                         "history_length": HISTORY_LENGTH},
            "curation": {"temperature": CURATION_TEMPERATURE, "top_p": CURATION_TOP_P,
                         "top_k": CURATION_TOP_K, "max_tokens": CURATION_MAX_TOKENS,
                         "enable_thinking": CURATION_ENABLE_THINKING},
            "print": {"prompt_show_every": PROMPT_SHOW_EVERY, "print_chars": PRINT_CHARS,
                      "save_raw_n": SAVE_RAW_N},
            "backend": {"llm_backend": LLM_BACKEND, "curation_llm_backend": CURATION_LLM_BACKEND,
                        "exec_external": _EXEC_EXTERNAL, "cur_external": _CUR_EXTERNAL,
                        "x_api_key_set": bool(_X_API_KEY), "max_concurrency": MAX_CONCURRENCY},
        },
        "env_vars": {k: _mask(k, os.environ.get(k)) for k in _TRACKED_ENV_VARS},
    }
    if extra:
        cfg.update(extra)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"[run_config] wrote {output_path}/run_config.json")


class _Tee:
    """Duplicate a stream to a file, so all stdout/stderr also lands in the results folder
    (replaces the need for an external `2>&1 | tee ...`). Line-buffered, flushes eagerly."""
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh
    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()
    def flush(self):
        self.stream.flush()
        self.fh.flush()
    # Delegate everything else (isatty, fileno, encoding, ...) to the real stream, so
    # libraries that probe stdout (e.g. textworld calls sys.stdout.isatty() at import) work.
    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()
    def fileno(self):
        return self.stream.fileno()
    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_self_logging(output_path, filename="run.log"):
    """Tee stdout+stderr into output_path/<filename>. Returns the open file handle (kept
    alive for the process lifetime). Safe to combine with an external `| tee` (harmless dup)."""
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, filename)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[self-logging] stdout+stderr -> {log_path}")
    return fh


_PRINT_LOCK = threading.Lock()

# --- ALFWorld group-async globals ---
# tatsu PDDL parser is NOT thread-safe -> serialize ALL env init/reset/step under PDDL_LOCK.
# Memory retrieval happens single-threaded in main() BEFORE each group (frozen snapshot) and
# updates happen after the group barrier, so no memory lock is needed inside the worker threads.
PDDL_LOCK = threading.Lock()
TEMPLATE_ENV = None   # base env holding the full game pool; set in main for alfworld


def _short(text):
    """Truncate long text for readable periodic prints (PRINT_CHARS=0 -> full)."""
    if PRINT_CHARS and text is not None and len(text) > PRINT_CHARS:
        return text[:PRINT_CHARS] + f" …[+{len(text) - PRINT_CHARS} chars]"
    return text


def _fmt_eta(seconds):
    """Human-readable H:MM:SS for ETA / elapsed."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


print(f"[print knobs] prompt_show_every={PROMPT_SHOW_EVERY} print_chars={PRINT_CHARS} "
      f"save_raw_n={SAVE_RAW_N}")

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

# Step-0 with retrieved context. No {task_description} field: at step 0 the task
# already rides inside {current_observation} (the [1:] slice keeps room + task),
# so adding a task field would duplicate it. This mirrors run_memp_ori.py, which
# appends the retrieved guidelines to the task-bearing initial observation.
ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT = """\
You are an expert agent operating in the ALFRED Embodied Environment.

## Past Relevant {context_label}

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

# Apply PROMPT_STYLE to the interactive templates (reasoning templates have no
# <think> mandate, so they are left unchanged).
ALFWORLD_TEMPLATE_NO_HIS              = _apply_prompt_style(ALFWORLD_TEMPLATE_NO_HIS)
ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT = _apply_prompt_style(ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT)
ALFWORLD_TEMPLATE                     = _apply_prompt_style(ALFWORLD_TEMPLATE)
ALFWORLD_TEMPLATE_WITH_CONTEXT        = _apply_prompt_style(ALFWORLD_TEMPLATE_WITH_CONTEXT)
WEBSHOP_TEMPLATE_NO_HIS        = _apply_prompt_style(WEBSHOP_TEMPLATE_NO_HIS)
WEBSHOP_TEMPLATE               = _apply_prompt_style(WEBSHOP_TEMPLATE)
WEBSHOP_TEMPLATE_WITH_CONTEXT  = _apply_prompt_style(WEBSHOP_TEMPLATE_WITH_CONTEXT)

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
    from google.genai import types
    if isinstance(prompt, list):
        text = "\n".join(m["content"] for m in prompt if m.get("role") != "system")
    elif isinstance(prompt, str):
        text = prompt
    else:
        raise ValueError(f'prompt must be a list or a string, got {type(prompt)}')
    client = _vertex_client()   # project/location from env or Salesforce defaults
    response = client.models.generate_content(
        model=model,
        contents=text,
        config=types.GenerateContentConfig(temperature=EXECUTOR_TEMPERATURE),
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

    # Was hardcoded: temperature=0.7 (no top_p/top_k/max_tokens/enable_thinking sent;
    # vLLM filled top_p/top_k from the model's generation_config, tokens uncapped).
    # Now env-driven via EXECUTOR_* knobs.
    kwargs = dict(
        model=model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['OPENAI_API_BASE'],
        num_retries=10,
        temperature=EXECUTOR_TEMPERATURE,
        stop=stop,
    )
    if EXECUTOR_TOP_P is not None:
        kwargs["top_p"] = EXECUTOR_TOP_P
    _set_max_tokens(kwargs, EXECUTOR_MAX_TOKENS, _EXEC_EXTERNAL)

    # extra_body fields (top_k, chat_template_kwargs) are vLLM-only. Skip them for an
    # external OpenAI-compatible gateway (it 400s on top_k / ignores chat_template_kwargs).
    extra_body = {}
    if not _EXEC_EXTERNAL:
        if EXECUTOR_TOP_K is not None:
            extra_body["top_k"] = EXECUTOR_TOP_K
        if ENABLE_THINKING is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if _EXEC_EXTERNAL:
        _hdrs = _gateway_extra_headers()
        if _hdrs:
            kwargs["extra_headers"] = _hdrs

    response = completion(**kwargs)
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
    raw_traces = [[] for _ in range(n)]   # SAVE_RAW: per-step {step, prompt, response}
    current_obs = list(obs)
    current_admissible = list(admissible_commands)
    task_rewards = [0.0] * n   # always float; won games overwrite with float(won) below
    active_tasks = list(range(n))

    for step_idx in range(max_steps):
        if not active_tasks:
            break
        print(f'\033[91mActive tasks: {active_tasks}\033[0m')

        # probe = first active task; used for periodic prompt/response display
        probe_idx = active_tasks[0]
        show_this_step = (
            PROMPT_SHOW_EVERY and step_idx % PROMPT_SHOW_EVERY == 0
        )

        prompts = {}
        for idx in active_tasks:
            history_str = format_action_history(histories[idx], HISTORY_LENGTH)
            admissible_str = "\n ".join(f"'{s}'" for s in current_admissible[idx] if s != 'help')
            step_count = len(histories[idx])
            ctx_text = skills_context.get(idx, "") if skills_context else ""

            if step_count == 0:
                if ctx_text:
                    # Step-0 context injection (mirrors run_memp_ori.py): memory guidance
                    # is available for the FIRST action, not only from step 1+.
                    prompt_text = ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT.format(
                        retrieved_context=ctx_text,
                        context_label=context_label,
                        context_label_lower=context_label.lower(),
                        current_observation=current_obs[idx],
                        admissible_actions=admissible_str,
                    )
                else:
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

            # periodic prompt display for the probe task, to spot-check correctness.
            # Prompt is printed IN FULL (PRINT_CHARS truncates responses only) so the
            # injected task/context can be verified — matches the async runners.
            if idx == probe_idx and show_this_step:
                with _PRINT_LOCK:
                    print(f'\033[96m[PROMPT task {idx} step {step_idx}]\n'
                          f'{prompt_text}\033[0m')

        responses = {}
        with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
            futures = {executor.submit(llm, prompts[idx], None, model): idx for idx in active_tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response = future.result()
                    # Only print the probe task's response (step 0 always, plus every
                    # PROMPT_SHOW_EVERY steps) — cuts I/O vs printing all ~N agents.
                    if idx == probe_idx and (step_idx == 0 or show_this_step):
                        with _PRINT_LOCK:
                            print(f'\033[92m[task {idx} step {step_idx}] response:\n'
                                  f'{_short(response)}\033[0m')
                    responses[idx] = response
                except Exception as e:
                    print(f'Error {idx}: {e}')

        responses = dict(sorted(responses.items()))

        # SAVE_RAW: keep the FULL injected prompt + raw response for this step (strings
        # already in memory; no extra LLM call). Persisted to idx_*.json at the end.
        if SAVE_RAW:
            for idx in active_tasks:
                raw_traces[idx].append({
                    "step": len(histories[idx]),
                    "prompt": prompts[idx][0]["content"],
                    "response": responses.get(idx, ""),
                })

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
                task_rewards[idx] = float(won[idx])   # normalize bool -> 1.0/0.0
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
        result = {"messages": messages, "reward": task_rewards[idx], "name": names[idx]}
        if SAVE_RAW:
            result["raw_trace"] = raw_traces[idx]
        results.append(result)

    return results


# ------------------------------------------------------------------ #
# ALFWorld per-game runner (group episode-async)                      #
# ------------------------------------------------------------------ #

def run_one_game(game_file, game_idx, model, max_steps,
                 task_description, ctx_text="", context_label="Skills"):
    """Run ONE pinned ALFWorld game to completion, concurrently with others in its group.

    Order/grouping/task/ctx come from the SHARED batched env.reset() stream in main() (the
    same stream the batch runner consumes), so game order + memory-update order are IDENTICAL
    to run_unified_dev.py by construction. Here we only parallelize execution: each game gets
    its OWN single-game env via copy.deepcopy(TEMPLATE_ENV)+init_env(batch_size=1), reset under
    PDDL_LOCK (tatsu not thread-safe); the LLM call is OUTSIDE the lock so games overlap on LLM
    latency. Prompt/parse logic is identical to run_unified_dev.py's alfworld_run_batch.

    task_description + ctx_text are passed in (retrieved single-threaded in main against the
    frozen pre-group snapshot) so this fn does no memory access at all.

    Returns {game_idx, messages, reward, name, raw_trace?}.
    """
    # --- pin + reset the env (locked). The reset here re-derives current_ob/admissible/name
    # for THIS game's own env; task_description is passed in from the shared-stream reset. ---
    with PDDL_LOCK:
        pinned = copy.deepcopy(TEMPLATE_ENV)
        pinned.game_files = [game_file]
        tw = pinned.init_env(batch_size=1)
        ob_raw, info = tw.reset()
        raw = ob_raw[0]
        # Step-0 fix: [1:] keeps room + task (NOT [1:2]); matches run_unified_dev.py.
        current_ob         = '\n'.join(raw.split('\n\n')[1:])
        current_admissible = info['admissible_commands'][0]
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

    history = []
    raw_trace = []
    reward = 0.0
    _probe = (game_idx == 0)

    for step_count in range(max_steps):
        admissible_str = "\n ".join(f"'{s}'" for s in current_admissible if s != 'help')

        if step_count == 0:
            if ctx_text:
                prompt_text = ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT.format(
                    retrieved_context=ctx_text,
                    context_label=context_label,
                    context_label_lower=context_label.lower(),
                    current_observation=current_ob,
                    admissible_actions=admissible_str,
                )
            else:
                prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=current_ob,
                    admissible_actions=admissible_str,
                )
        elif ctx_text:
            prompt_text = ALFWORLD_TEMPLATE_WITH_CONTEXT.format(
                task_description=task_description,
                retrieved_context=ctx_text,
                context_label=context_label,
                context_label_lower=context_label.lower(),
                step_count=step_count,
                history_length=min(HISTORY_LENGTH, step_count),
                action_history=format_action_history(history, HISTORY_LENGTH),
                current_step=step_count + 1,
                current_observation=current_ob,
                admissible_actions=admissible_str,
            )
        else:
            prompt_text = ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=step_count,
                history_length=min(HISTORY_LENGTH, step_count),
                action_history=format_action_history(history, HISTORY_LENGTH),
                current_step=step_count + 1,
                current_observation=current_ob,
                admissible_actions=admissible_str,
            )

        # periodic prompt display for the probe game (game 0); prompt IN FULL
        show_this_step = PROMPT_SHOW_EVERY and step_count % PROMPT_SHOW_EVERY == 0
        if _probe and show_this_step:
            with _PRINT_LOCK:
                print(f'\033[96m[PROMPT game {game_idx} step {step_count}]\n{prompt_text}\033[0m')

        # LLM call — OUTSIDE the lock (this is where episodes overlap)
        response = llm([{"role": "user", "content": prompt_text}], None, model)

        if SAVE_RAW:
            raw_trace.append({"step": step_count, "prompt": prompt_text, "response": response})

        action = ""
        if '<action>' in response and '</action>' in response:
            action = response.split('<action>')[-1].split('</action>')[0].strip()

        # env step — locked
        with PDDL_LOCK:
            obs, _, done, info = tw.step([action])
            next_ob = process_ob(obs[0])
            current_admissible = info.get('admissible_commands', [current_admissible])[0]
            won_val = info['won'][0]
            is_done = done[0]

        history.append((current_ob, action))
        current_ob = next_ob

        # response display for probe game: step 0 always, plus every PROMPT_SHOW_EVERY steps
        if _probe and (step_count == 0 or show_this_step):
            with _PRINT_LOCK:
                print(f'\033[92m[game {game_idx} step {step_count}] response:\n'
                      f'{_short(response)}\033[0m')

        if is_done:
            reward = 1.0 if won_val else 0.0
            break

    # build messages exactly like alfworld_run_batch (obs/action pairs + final obs)
    messages = []
    for obs_h, act_h in history:
        messages.append({"role": "user",      "content": obs_h})
        messages.append({"role": "assistant", "content": act_h})
    messages.append({"role": "user", "content": current_ob})

    out = {"game_idx": game_idx, "messages": messages, "reward": reward,
           "name": name, "task_description": task_description, "ctx_text": ctx_text}
    if SAVE_RAW:
        out["raw_trace"] = raw_trace
    return out


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
    raw_traces = [[] for _ in range(n)]   # SAVE_RAW: per-step {step, prompt, response}
    step_counts = [0] * n

    for step_idx in range(max_steps):
        active = [i for i in range(n) if not dones_arr[i]]
        if not active:
            break
        print(f'\033[91mActive tasks: {active}\033[0m')

        # probe = first active task; used for periodic prompt/response display
        probe_idx = active[0]
        show_this_step = (
            PROMPT_SHOW_EVERY and step_idx % PROMPT_SHOW_EVERY == 0
        )

        responses = [""] * n
        content_by_i = {}   # SAVE_RAW: the exact injected prompt sent per task this step
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {}
            for i in active:
                ctx_text = (skills_context.get(i, "") if skills_context else "")
                # Inject context from step 0 onward (was: step_counts[i] > 0, which
                # skipped the first step) so memory guidance is present for the FIRST
                # action too — mirrors run_memp_ori.py.
                if ctx_text:
                    content = f"## Past Relevant Skills\n\n{ctx_text}\n\n---\n\n{prompts[i]}"
                else:
                    content = prompts[i]
                content_by_i[i] = content
                # Prompt printed IN FULL (PRINT_CHARS truncates responses only).
                if i == probe_idx and show_this_step:
                    with _PRINT_LOCK:
                        print(f'\033[96m[PROMPT task {i} step {step_idx}]\n'
                              f'{content}\033[0m')
                futures[executor.submit(llm, [{"role": "user", "content": content}], None, model)] = i
            for future in as_completed(futures):
                i = futures[future]
                try:
                    responses[i] = future.result()
                    # Only print the probe task's response (step 0 always, plus every
                    # PROMPT_SHOW_EVERY steps) — cuts I/O vs printing all ~N agents.
                    if i == probe_idx and (step_idx == 0 or show_this_step):
                        with _PRINT_LOCK:
                            print(f'\033[92m[task {i} step {step_idx}] response:\n'
                                  f'{_short(responses[i])}\033[0m')
                except Exception as e:
                    print(f'Error {i}: {e}')

        for i in active:
            histories[i].append((prompts[i], responses[i]))
            if SAVE_RAW:
                raw_traces[i].append({
                    "step": step_counts[i],
                    "prompt": content_by_i.get(i, prompts[i]),
                    "response": responses[i],
                })
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
        result = {"messages": messages, "reward": task_rewards[i]}
        if SAVE_RAW:
            result["raw_trace"] = raw_traces[i]
        results.append(result)
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

    t_start = time.time()
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
                    # PRINT_CHARS controls truncation (default 120 when unset).
                    snippet = _short(responses[j]) if PRINT_CHARS else responses[j][:120]
                    print(f'\033[92mProblem {i+j}: {snippet}\033[0m')
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
        processed_this_run = max(done - finished, 1)
        elapsed = time.time() - t_start
        rate = processed_this_run / elapsed if elapsed > 0 else 0.0
        eta = (len(problems) - done) / rate if rate > 0 else 0.0
        tqdm.write(
            f'Avg accuracy: {all_reward / done * 100:.2f}%  [{done}/{len(problems)}]  '
            f'| elapsed {_fmt_eta(elapsed)}  {rate * 60:.1f} probs/min  ETA {_fmt_eta(eta)}'
        )

    total_elapsed = time.time() - t_start
    print(f'\nFinal accuracy: {all_reward / len(problems) * 100:.2f}%  ({int(all_reward)}/{len(problems)})')
    print(f'Total wall-clock: {_fmt_eta(total_elapsed)} ({total_elapsed / 60:.2f} min).')


# ------------------------------------------------------------------ #
# SkillOS curation                                                    #
# ------------------------------------------------------------------ #

def execute_tool(skill_memory: SkillMemory, tool_name: str, arguments: dict, task: str = None) -> dict:
    # `task` = the CURRENT task description that produced this curation op; passed to
    # insert/update so the skill records its originating task(s) as the BM25 retrieval
    # key (Option B: RB/Curator-aligned task<->task retrieval). None => legacy behavior.
    if tool_name == "new_skill_insert":
        try:
            title = skill_memory.new_memory_insert(arguments["skill_name"], arguments["content"], task=task)
            return {"status": "ok", "message": "Skill created.", "skill_name": title}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    elif tool_name == "skill_update":
        try:
            updated = skill_memory.memory_update(
                title=arguments["skill_name"],
                new_name=arguments.get("new_name"),
                new_content=arguments.get("new_content"),
                task=task,
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


def _build_curation_messages_plain(skill_memory: SkillMemory,
                                    task: str, messages: list, reward: bool,
                                    retrieved_skills_text: str) -> list:
    """Build PLAIN chat messages for native OpenAI tool-calling (no Qwen ✿-token preprocessing).

    Same system prompt + user content as _build_curation_messages, but the tool schemas are passed
    separately as the `tools=` arg (see curation_llm_native), not encoded into the text. Mirrors the
    native branch of SkillOS/skills_agent.py (is_qwen==False)."""
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
    return system_messages + [{"role": "user", "content": user_content}]


def curation_llm_native(messages: list, curation_model: str, curation_base_url: str):
    """Native OpenAI tool-calling curation for external (non-Qwen) models via the gateway.

    Passes MEMORY_TOOL_SCHEMAS as `tools=` and returns the parsed function calls as a list of
    {name, arguments(JSON str)} — the shape _apply_parsed_function_calls expects. No ✿-token
    encoding/parsing (GPT/Gemini never emit those). Mirrors SkillOS/skills_agent.py:366-402."""
    kwargs = dict(
        model=curation_model,
        messages=messages,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=curation_base_url or os.environ.get("OPENAI_API_BASE"),
        num_retries=10,
        temperature=CURATION_TEMPERATURE,
        tools=MEMORY_TOOL_SCHEMAS,
        tool_choice="auto",
    )
    if CURATION_TOP_P is not None:
        kwargs["top_p"] = CURATION_TOP_P
    _set_max_tokens(kwargs, CURATION_MAX_TOKENS, True)   # native path is always external
    _hdrs = _gateway_extra_headers()
    if _hdrs:
        kwargs["extra_headers"] = _hdrs

    response = completion(**kwargs)
    msg = response.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    parsed = []
    for tc in tool_calls:
        fn = tc.function
        parsed.append({"name": fn.name, "arguments": fn.arguments or "{}"})
    return parsed


def curation_vertex_native(messages: list, curation_model: str):
    """Native Vertex AI (gemini/) function-calling curation for skillos.

    Vertex equivalent of curation_llm_native: passes MEMORY_TOOL_SCHEMAS as a Vertex Tool and
    reads response.function_calls (Gemini emits real function calls, never Qwen ✿-tokens).
    Returns [{name, arguments(JSON str)}] — the shape _apply_parsed_function_calls expects."""
    from google.genai import types
    model_id = curation_model[len("gemini/"):] if curation_model.startswith("gemini/") else curation_model
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    user_text  = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
    client = _vertex_client()
    response = client.models.generate_content(
        model=model_id,
        contents=user_text,
        config=types.GenerateContentConfig(
            temperature=CURATION_TEMPERATURE,
            system_instruction=system_msg,
            tools=[_openai_tools_to_vertex(MEMORY_TOOL_SCHEMAS)],
        ),
    )
    parsed = []
    for fc in (getattr(response, "function_calls", None) or []):
        args = dict(fc.args) if fc.args else {}
        parsed.append({"name": fc.name, "arguments": json.dumps(args)})
    return parsed


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


def _apply_parsed_function_calls(skill_memory: SkillMemory, function_calls: list, label: str = "", task: str = None):
    """Execute already-parsed function calls (list of {name, arguments-as-JSON-string}).
    Shared by the Qwen-token path (_apply_curation_output) and the native tool-calls path.
    `task` = current task description; forwarded to execute_tool so inserted/updated
    skills record their originating task as the BM25 retrieval key (Option B)."""
    print(function_calls)
    for fc in function_calls:
        try:
            arguments = json.loads(fc["arguments"]) or {}
        except json.JSONDecodeError:
            try:
                arguments = json.loads(fc["arguments"].replace('\n', '\\n').replace('\r', '\\r')) or {}
            except json.JSONDecodeError:
                arguments = {}
        result = execute_tool(skill_memory, fc["name"], arguments, task=task)
        print(f"[SkillCuration{label}] {fc['name']}({arguments.get('skill_name', '')}) "
              f"-> {result['status']}: {result.get('message', '')}")


def _apply_curation_output(skill_memory: SkillMemory, raw: str, label: str = "", task: str = None):
    """Qwen-token path: parse ✿FUNCTION✿ tokens from raw text, then apply."""
    function_calls, _ = _parse_function_calls_from_text(raw)
    _apply_parsed_function_calls(skill_memory, function_calls, label, task=task)


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
                temperature=CURATION_TEMPERATURE,   # was hardcoded: temperature=0.7
                system_instruction=system_msg,
            ),
        )
        return response.text or ""

    # Was hardcoded: temperature=0.7 (no top_p/top_k/max_tokens/enable_thinking sent).
    # Now env-driven via CURATION_* knobs.
    kwargs = dict(
        model=curation_model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=curation_base_url,
        num_retries=10,
        temperature=CURATION_TEMPERATURE,
    )
    if CURATION_TOP_P is not None:
        kwargs["top_p"] = CURATION_TOP_P
    _set_max_tokens(kwargs, CURATION_MAX_TOKENS, _CUR_EXTERNAL)

    # extra_body (top_k, chat_template_kwargs) is vLLM-only; skip for external gateway.
    extra_body = {}
    if not _CUR_EXTERNAL:
        if CURATION_TOP_K is not None:
            extra_body["top_k"] = CURATION_TOP_K
        if CURATION_ENABLE_THINKING is not None:
            extra_body["chat_template_kwargs"] = {"enable_thinking": CURATION_ENABLE_THINKING}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if _CUR_EXTERNAL:
        _hdrs = _gateway_extra_headers()
        if _hdrs:
            kwargs["extra_headers"] = _hdrs

    response = completion(**kwargs)
    return response.choices[0].message.content or ""


def batch_update_skills_from_trajectories(
    skill_memory: SkillMemory,
    curation_tokenizer,
    curation_model_hf,
    batch_data: list,  # list of (task, messages, reward, retrieved_skills_text)
    curation_model: str = None,
    curation_base_url: str = None,
):
    is_gemini = curation_model is not None and curation_model.startswith("gemini/")
    use_http = curation_base_url is not None or _CUR_EXTERNAL or is_gemini
    # Native tool-calling path: external OpenAI gateway (GPT) OR Vertex (gemini/). Both emit real
    # function calls (never Qwen ✿-tokens), so both use PLAIN messages + MEMORY_TOOL_SCHEMAS and
    # return the {name, arguments} shape _apply_parsed_function_calls expects. Only the per-item
    # call fn differs: curation_vertex_native (Vertex) vs curation_llm_native (litellm gateway).
    use_native_fc = _CUR_EXTERNAL or is_gemini

    if use_http and use_native_fc:
        _native_call = (curation_vertex_native if is_gemini
                        else curation_llm_native)
        all_messages = []
        for task, messages, reward, retrieved_skills_text in batch_data:
            try:
                msgs = _build_curation_messages_plain(
                    skill_memory, task, messages, reward, retrieved_skills_text
                )
                all_messages.append(msgs)
            except Exception as e:
                print(f"[SkillCuration] Prompt build failed: {e}")
                all_messages.append(None)

        valid_indices = [i for i, m in enumerate(all_messages) if m is not None]
        if not valid_indices:
            return

        def _call(i):
            # Vertex fn takes (messages, model); gateway fn takes (messages, model, base_url).
            if is_gemini:
                return _native_call(all_messages[i], curation_model)
            return _native_call(all_messages[i], curation_model, curation_base_url)

        parsed_calls = [None] * len(all_messages)
        with ThreadPoolExecutor(max_workers=len(valid_indices)) as executor:
            futures = {executor.submit(_call, i): i for i in valid_indices}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    parsed_calls[i] = future.result()
                except Exception as e:
                    print(f"[SkillCuration] Native tool-call failed for item {i}: {e}")

        for i in valid_indices:
            if parsed_calls[i]:
                # batch_data[i] = (task, messages, reward, retrieved_skills_text); pass task for Option-B keying.
                _apply_parsed_function_calls(skill_memory, parsed_calls[i], label=f" game={i}", task=batch_data[i][0])

    elif use_http:
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
                # batch_data[i] = (task, messages, reward, retrieved_skills_text); pass task for Option-B keying.
                _apply_curation_output(skill_memory, raw_outputs[i], label=f" game={i}", task=batch_data[i][0])

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
            # Was hardcoded: SamplingParams(temperature=0.7, max_tokens=4096).
            # Now env-driven via CURATION_* knobs (max_tokens default kept at 4096).
            sp_kwargs = dict(
                temperature=CURATION_TEMPERATURE,
                max_tokens=CURATION_MAX_TOKENS if CURATION_MAX_TOKENS is not None else 4096,
            )
            if CURATION_TOP_P is not None:
                sp_kwargs["top_p"] = CURATION_TOP_P
            if CURATION_TOP_K is not None:
                sp_kwargs["top_k"] = CURATION_TOP_K
            sampling_params = SamplingParams(**sp_kwargs)
            outputs = curation_model_hf.generate(valid_texts, sampling_params)
        except Exception as e:
            print(f"[SkillCuration] vLLM batch inference failed: {e}")
            return

        for orig_idx, output in zip(valid_indices, outputs):
            raw = output.outputs[0].text
            # batch_data[orig_idx] = (task, messages, reward, retrieved_skills_text); pass task for Option-B keying.
            _apply_curation_output(skill_memory, raw, label=f" game={orig_idx}", task=batch_data[orig_idx][0])


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
    # _CUR_EXTERNAL (CURATION_LLM_BACKEND=openai) forces the HTTP/native path even if no
    # --curation_base_url is given (it falls back to OPENAI_API_BASE = the gateway).
    use_http = bool(getattr(args, 'curation_base_url', None)) or _CUR_EXTERNAL or use_gemini_curator

    # SkillOS needs QwenFnCallPrompt + Qwen tokenizer ONLY for the ✿-token curation path.
    # The native tool-calling paths — external gateway (_CUR_EXTERNAL) AND Vertex gemini/ — use
    # plain messages + tools=, so they need neither. Skip the Qwen load for both, so a GPT/Gemini
    # curator has no Qwen dependency.
    if args.memory_type == 'skillos' and not (_CUR_EXTERNAL or use_gemini_curator):
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
            print(f"Curation via Vertex AI (native tool-calls for skillos): {args.curation_model} "
                  f"@ project={GCLOUD_PROJECT} location={GCLOUD_LOCATION}")
        elif _CUR_EXTERNAL:
            _cb = getattr(args, 'curation_base_url', None) or os.environ.get("OPENAI_API_BASE")
            print(f"Curation via external gateway (native tool-calls): {args.curation_model} @ {_cb}")
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
    # External curator with no explicit --curation_base_url: fall back to the gateway (OPENAI_API_BASE).
    if _CUR_EXTERNAL and not curation_base_url:
        curation_base_url = os.environ.get("OPENAI_API_BASE")

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
            from reasoningbank_alfworld_api import ReasoningBankAlfworld
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
    elif args.env in REASONING_ENVS:
        output_path         = (f'Reasoning/results/{args.env}/{model_name}/'
                               f'{args.exp_name}_{memory_type}')
    else:  # webshop
        output_path         = (f'Webshop/results/{model_name}/'
                               f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_{memory_type}')

    # Memory now lives INSIDE the result folder (self-contained run: idx_*.json,
    # run_config.json, run.log, and the memory store all in one place).
    skills_storage_path = os.path.join(
        output_path,
        'skills.json' if memory_type == 'skillos' else 'reasoning_bank.jsonl'
    )

    os.makedirs(output_path, exist_ok=True)

    # Self-describing result folder: tee all output into <output_path>/run.log and dump the
    # full hyperparameter/env config into <output_path>/run_config.json.
    start_self_logging(output_path)
    dump_run_config(output_path, args, extra={"skills_storage_path": skills_storage_path})

    # ---- Reasoning: single-turn inference (with optional memory) ----
    if args.env in REASONING_ENVS:
        problems = load_reasoning_dataset(args.env)
        if args.num_games > 0:
            problems = problems[:args.num_games]
        finished = sum(1 for f in os.listdir(output_path)
                       if f.startswith('idx_') and f.endswith('.json'))
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
        # Only per-game result files (idx_*.json) — NOT run_config.json / other sidecars.
        if file.startswith('idx_') and file.endswith('.json'):
            finished_games += 1
            with open(f'{output_path}/{file}', 'r') as f:
                all_reward += json.load(f)['reward']

    # ==================================================================
    # ALFWorld loop
    # ==================================================================
    if args.env == 'alfworld':
        # GROUP EPISODE-ASYNC. EXACTNESS-CRITICAL: the group structure, game order, per-group
        # membership, game_idx, task strings, and memory retrieve/update points are ALL driven
        # by the SAME batched env.reset() stream that run_unified_dev.py consumes — so they are
        # identical to the batch runner by construction (textworld's serving order is
        # deterministic but is NOT game_files order, so we must NOT pin by game_files index).
        # We only parallelize EXECUTION: each game in a group runs in its own thread on a private
        # single-game env (run_one_game). Memory is retrieved single-threaded BEFORE the group
        # (frozen snapshot) and updated once AFTER the group barrier, exactly as the batch runner.
        num_games_to_run = num_games  # set in __main__ block
        group_size = env.batch_size

        # SAVE_RAW=N -> keep raw_trace only for these N game indices (evenly spread).
        raw_keep_idxs = _raw_save_indices(num_games_to_run, SAVE_RAW_N)
        if SAVE_RAW_N:
            print(f"[SAVE_RAW] keeping raw_trace for {len(raw_keep_idxs)} games: "
                  f"{sorted(raw_keep_idxs)}")

        t_start = time.time()
        games_this_run = 0
        for idx in tqdm(range(math.ceil(num_games_to_run / group_size))):
            ob_list, info = env.reset()   # SHARED stream: defines order/idx exactly like batch
            if idx * group_size + group_size <= finished_games:
                continue

            # Same parsing as run_unified_dev.py's batch loop (step-0 fix [1:] slice).
            task_descriptions = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]
            game_file_list = list(info['extra.gamefile'])
            name_list = ['/'.join(gf.split('/')[-3:-1]) for gf in game_file_list]
            real_n = len(ob_list)

            # --- retrieve context for each game (single-threaded, frozen pre-group snapshot) ---
            skills_context = {}
            if memory_obj is not None:
                for i, query in enumerate(task_descriptions):
                    ctx = retrieve_context(memory_type, memory_obj, query, args.retrieve_num)
                    if ctx:
                        skills_context[i] = ctx

            # --- run the group's games CONCURRENTLY to completion (barrier at with-block exit) ---
            # keyed by position i in the group (0..real_n-1), matching batch semantics.
            # MAX_CONCURRENCY (optional) caps in-flight threads for gateway rate limits; it never
            # exceeds group_size, so the memory-group barrier is preserved (games still all finish
            # before the post-group update).
            _workers = min(group_size, MAX_CONCURRENCY) if MAX_CONCURRENCY else group_size
            results_by_pos = {}
            with ThreadPoolExecutor(max_workers=_workers) as pool:
                futures = {
                    pool.submit(run_one_game, game_file_list[i], idx * group_size + i,
                                model_name, args.max_steps,
                                task_descriptions[i], skills_context.get(i, ""), context_label): i
                    for i in range(real_n)
                }
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        results_by_pos[i] = fut.result()
                    except Exception as e:
                        print(f"[game {idx * group_size + i}] ERROR: {e}")

            # --- order by group position (== batch runner order) for update + persist ---
            batch_results = [results_by_pos[i] for i in range(real_n) if i in results_by_pos]

            for res in batch_results:
                all_reward     += res['reward']
                finished_games += 1
                games_this_run += 1
            elapsed = time.time() - t_start
            rate = games_this_run / elapsed if elapsed > 0 else 0.0
            eta = (num_games_to_run - finished_games) / rate if rate > 0 else 0.0
            tqdm.write(
                f'Avg reward: {all_reward / max(finished_games,1):.3f}  '
                f'| elapsed {_fmt_eta(elapsed)}  {rate * 60:.1f} games/min  ETA {_fmt_eta(eta)}'
            )

            # --- persist per-game (idx = idx*group_size+i, identical to batch runner) ---
            for i in range(real_n):
                if i not in results_by_pos:
                    continue
                res = results_by_pos[i]
                game_idx = idx * group_size + i
                out = {"messages": res["messages"], "reward": res["reward"], "name": res["name"]}
                if SAVE_RAW and game_idx in raw_keep_idxs and "raw_trace" in res:
                    out["raw_trace"] = res["raw_trace"]
                with open(f'{output_path}/idx_{game_idx}.json', 'w') as f:
                    json.dump(out, f, indent=4, ensure_ascii=False)

            print(f'Finished {idx * group_size + real_n} games')

            # --- memory update at the group barrier (in group-position order) ---
            update_memory_after_batch(
                memory_type, memory_obj,
                curation_tokenizer, curation_model_hf,
                batch_results, task_descriptions, skills_context,
                skills_storage_path, args.env,
                curation_model=args.curation_model,
                curation_base_url=getattr(args, 'curation_base_url', None),
            )

        total_elapsed = time.time() - t_start
        print(f'\nFinal avg reward: {all_reward / max(finished_games, 1):.3f}  '
              f'({all_reward:.1f}/{finished_games})')
        print(f'Total wall-clock: {_fmt_eta(total_elapsed)} ({total_elapsed / 60:.2f} min) '
              f'for {games_this_run} games this run '
              f'({games_this_run / total_elapsed * 60:.1f} games/min).'
              if total_elapsed > 0 else 'Total wall-clock: 0s')

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

        # SAVE_RAW=N -> keep raw_trace only for these N game indices (evenly spread).
        raw_keep_idxs = _raw_save_indices(num_webshop_games, SAVE_RAW_N)
        if SAVE_RAW_N:
            print(f"[SAVE_RAW] keeping raw_trace for {len(raw_keep_idxs)} games: "
                  f"{sorted(raw_keep_idxs)}")

        t_start = time.time()
        games_this_run = 0
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
                games_this_run += 1
                game_idx = start + i
                # Drop raw_trace for non-sampled games (collected cheaply, kept for N only).
                if 'raw_trace' in result and game_idx not in raw_keep_idxs:
                    result = {k: v for k, v in result.items() if k != 'raw_trace'}
                with open(f'{output_path}/idx_{game_idx}.json', 'w') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

            elapsed = time.time() - t_start
            rate = games_this_run / elapsed if elapsed > 0 else 0.0
            eta = (num_webshop_games - finished_games) / rate if rate > 0 else 0.0
            tqdm.write(
                f'Avg reward: {all_reward / finished_games:.3f}  '
                f'| elapsed {_fmt_eta(elapsed)}  {rate * 60:.1f} games/min  ETA {_fmt_eta(eta)}'
            )
            print(f'Finished {finished_games} games')

            update_memory_after_batch(
                memory_type, memory_obj,
                curation_tokenizer, curation_model_hf,
                batch_results[:real_n], task_descriptions[:real_n], skills_context,
                skills_storage_path, args.env,
                curation_model=args.curation_model,
                curation_base_url=getattr(args, 'curation_base_url', None),
            )

        total_elapsed = time.time() - t_start
        print(f'\nFinal avg reward: {all_reward / max(finished_games, 1):.3f}  '
              f'({all_reward:.1f}/{finished_games})')
        print(f'Total wall-clock: {_fmt_eta(total_elapsed)} ({total_elapsed / 60:.2f} min) '
              f'for {games_this_run} games this run '
              f'({games_this_run / total_elapsed * 60:.1f} games/min).'
              if total_elapsed > 0 else 'Total wall-clock: 0s')


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
        # TWO envs:
        #  - `env`: the BATCHED env (batch_size groups). Its env.reset() stream in main() defines
        #    the exact game order / grouping / idx — identical to run_unified_dev.py.
        #  - `TEMPLATE_ENV`: the same base env holding the full game pool; each worker thread
        #    copy.deepcopy()s it + init_env(batch_size=1) to run ONE pinned game concurrently.
        env = get_environment(config["env"]["type"])(config, train_eval=split)
        TEMPLATE_ENV = env   # share the pool; run_one_game deepcopies before mutating game_files
        env = env.init_env(batch_size=args.batch_size)
        num_games = len(TEMPLATE_ENV.game_files)
        if args.num_games > 0:
            num_games = min(args.num_games, num_games)
        print(f"Total ALFWorld games: {num_games}  (group size / concurrency = {args.batch_size})")

    main(args)
