"""
run_unified_hyper_async_revise_react.py — rolling-pool async baseline, no reasoning-tag mandate.
(Same engine as run_unified_hyper_async.py; the "MUST use <think>" instruction is removed.)

Speed model (vs run_unified_hyper.py and run_unified_hyper_concurrent.py)
-------------------------------------------------------------------------
run_unified_hyper.py is STEP-synchronous (per-step barrier -> ~40% GPU util).
run_unified_hyper_concurrent.py is GROUP episode-async (barrier per group of
--batch_size -> high util, but the group tail wastes idle slots).

This file is FULLY async — a ROLLING POOL with NO barrier:
  - ALL games are submitted to one ThreadPoolExecutor(max_workers=--concurrency).
  - Each game runs to COMPLETION in its own thread (LLM call OUTSIDE the PDDL lock).
  - The moment ANY game finishes, its freed slot immediately pulls the next queued
    game — so ~--concurrency games are always in flight, no group boundary, no tail
    idle. This keeps the vLLM request queue continuously full -> highest GPU util.
  - Trade-off vs group version: NO clean barrier for per-group memory updates
    (fine here — this file is no-memory only).

Scope: ALFWorld + --memory_type none only. Same env (pip alfworld + base_config.yaml),
same prompts, same <action> parsing, same idx_*.json output as run_unified_hyper.py,
so results are directly comparable to the other SkillCurator runs — just faster.

Env vars (executor hyperparameters), same as run_unified_hyper.py:
  EXECUTOR_TEMPERATURE (default 0.7), EXECUTOR_TOP_P, EXECUTOR_TOP_K, EXECUTOR_MAX_TOKENS
  ENABLE_THINKING       ("true"/"false"; unset -> server default / not sent)
  PRINT_CHARS           (truncate printed responses; 0 = full)

Example
-------
  ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
  EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=false \
  python run_unified_hyper_concurrent.py --env alfworld --memory_type none \
      --model openai/Qwen/Qwen3-8B --exp_name baseline_conc --batch_size 30 \
      2>&1 | tee logs/baseline_conc.log
"""

import os
import sys
import copy
import json
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
import yaml
from litellm import completion

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")
openai.api_key = os.environ["OPENAI_API_KEY"]

HISTORY_LENGTH = int(os.environ.get("HISTORY_LENGTH", "5"))

# ------------------------------------------------------------------ #
# Executor sampling hyperparameters (overridable via env vars)        #
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

_ET = os.environ.get("ENABLE_THINKING", "")
ENABLE_THINKING = None if _ET == "" else (_ET.lower() in ("1", "true", "yes"))

PRINT_CHARS = _env_int("PRINT_CHARS") or 0

# Every N steps, print the probe game's (game_idx 0) full PROMPT and RESPONSE, so we can
# spot-check correctness (task present, formatting) without flooding across concurrent games.
# 0 (default) -> only the step-0 response of the probe game is printed.
PROMPT_SHOW_EVERY = _env_int("PROMPT_SHOW_EVERY") or 0

print(f"[executor hyperparams] temperature={EXECUTOR_TEMPERATURE} top_p={EXECUTOR_TOP_P} "
      f"top_k={EXECUTOR_TOP_K} max_tokens={EXECUTOR_MAX_TOKENS} "
      f"enable_thinking={ENABLE_THINKING} history_length={HISTORY_LENGTH}")

# ------------------------------------------------------------------ #
# Per-run config dump + self-logging (into the results folder)        #
# This is a no-memory runner, so only executor/print knobs are tracked #
# (no CURATION_*/PROMPT_STYLE/SAVE_RAW).                                #
# ------------------------------------------------------------------ #
_TRACKED_ENV_VARS = [
    "OPENAI_API_BASE", "OPENAI_API_KEY", "ALFWORLD_DATA", "TMPDIR", "HISTORY_LENGTH",
    "EXECUTOR_TEMPERATURE", "EXECUTOR_TOP_P", "EXECUTOR_TOP_K", "EXECUTOR_MAX_TOKENS",
    "ENABLE_THINKING", "PROMPT_SHOW_EVERY", "PRINT_CHARS",
]


def dump_run_config(output_path, args, extra=None):
    """Write run_config.json into output_path: CLI args + resolved executor hyperparams +
    tracked env vars, so every result folder records exactly how it was produced. Masks key."""
    def _mask(k, v):
        return ("****" if v else v) if (v is not None and "KEY" in k) else v
    cfg = {
        "runner": os.path.basename(__file__),
        "args": vars(args),
        "resolved_hyperparams": {
            "executor": {"temperature": EXECUTOR_TEMPERATURE, "top_p": EXECUTOR_TOP_P,
                         "top_k": EXECUTOR_TOP_K, "max_tokens": EXECUTOR_MAX_TOKENS,
                         "enable_thinking": ENABLE_THINKING, "history_length": HISTORY_LENGTH},
            "print": {"prompt_show_every": PROMPT_SHOW_EVERY, "print_chars": PRINT_CHARS},
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
    (replaces the need for an external `2>&1 | tee ...`). Flushes eagerly."""
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
    """Tee stdout+stderr into output_path/<filename>. Returns the open file handle (kept alive
    for the process lifetime). Safe to combine with an external `| tee` (harmless dup)."""
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, filename)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[self-logging] stdout+stderr -> {log_path}")
    return fh

# ------------------------------------------------------------------ #
# Prompt templates (identical to run_unified_hyper.py, ALFWorld)      #
# ------------------------------------------------------------------ #

ALFWORLD_TEMPLATE_NO_HIS = """\
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation.
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\
"""

ALFWORLD_TEMPLATE = """\
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags.\
"""

# ------------------------------------------------------------------ #
# LLM helper (same sampling wiring as run_unified_hyper.py)           #
# ------------------------------------------------------------------ #

def llm(prompt, stop=None, model="openai/Qwen/Qwen2.5-7B-Instruct"):
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

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
    if EXECUTOR_MAX_TOKENS is not None:
        kwargs["max_tokens"] = EXECUTOR_MAX_TOKENS

    extra_body = {}
    if EXECUTOR_TOP_K is not None:
        extra_body["top_k"] = EXECUTOR_TOP_K
    if ENABLE_THINKING is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": ENABLE_THINKING}
    if extra_body:
        kwargs["extra_body"] = extra_body

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


def _short(text):
    if PRINT_CHARS and text is not None and len(text) > PRINT_CHARS:
        return text[:PRINT_CHARS] + f" …[+{len(text) - PRINT_CHARS} chars]"
    return text


# ------------------------------------------------------------------ #
# Per-game episode runner (concurrent)                                #
# ------------------------------------------------------------------ #

# tatsu PDDL parser is NOT thread-safe -> serialize ALL env init/reset/step.
PDDL_LOCK = threading.Lock()

TEMPLATE_ENV = None   # base env holding the full game pool; set in main
_PRINT_LOCK = threading.Lock()


def run_one_game(game_file, game_idx, model, max_steps):
    """Run ONE pinned ALFWorld game to completion. Env ops under PDDL_LOCK;
    the LLM call is OUTSIDE the lock so games overlap on LLM latency."""
    # --- pin + reset the env (locked) ---
    with PDDL_LOCK:
        pinned = copy.deepcopy(TEMPLATE_ENV)
        pinned.game_files = [game_file]
        tw = pinned.init_env(batch_size=1)
        ob_raw, info = tw.reset()
        raw = ob_raw[0]
        task_description   = raw.split("\nYour task is to: ")[-1]
        # STEP-0 TASK BUG FIX: was [1:2], which kept only the room description and dropped
        # the "Your task is to: ..." line -> the step-0 prompt had no goal. [1:] keeps room +
        # task. Only affects step 0; step>0 overwrites current_ob with env.step()'s observation.
        current_ob         = '\n'.join(raw.split('\n\n')[1:])
        current_admissible = info['admissible_commands'][0]
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])

    history = []
    reward = 0.0

    for step_count in range(max_steps):
        admissible_str = "\n ".join(f"'{s}'" for s in current_admissible if s != 'help')
        if step_count == 0:
            prompt_text = ALFWORLD_TEMPLATE_NO_HIS.format(
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

        # periodic prompt display for the probe game (game 0), to spot-check correctness
        _probe = (game_idx == 0)
        if _probe and PROMPT_SHOW_EVERY and (step_count % PROMPT_SHOW_EVERY == 0):
            with _PRINT_LOCK:
                print(f'\033[96m[PROMPT game 0 step {step_count}]\n{prompt_text}\033[0m')

        # LLM call — OUTSIDE the lock (this is where episodes overlap)
        response = llm([{"role": "user", "content": prompt_text}], None, model)

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

        # response display for probe game: always step 0, plus every PROMPT_SHOW_EVERY steps
        if _probe and (step_count == 0 or (PROMPT_SHOW_EVERY and step_count % PROMPT_SHOW_EVERY == 0)):
            with _PRINT_LOCK:
                print(f'\033[92m[game 0 step {step_count}] response:\n{_short(response)}\033[0m')

        if is_done:
            reward = 1.0 if won_val else 0.0
            break

    # build messages exactly like run_unified_hyper.py's batch runner
    messages = []
    for obs_h, act_h in history:
        messages.append({"role": "user",      "content": obs_h})
        messages.append({"role": "assistant", "content": act_h})
    messages.append({"role": "user", "content": current_ob})

    return {"game_idx": game_idx, "messages": messages, "reward": reward, "name": name}


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main(args):
    global TEMPLATE_ENV
    model = args.model

    output_path = (f'Alfworld/results/{model}/'
                   f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_none')
    os.makedirs(output_path, exist_ok=True)

    if args.overwrite:
        for f in os.listdir(output_path):
            if f.endswith('.json'):
                os.remove(os.path.join(output_path, f))
        print(f'Cleared existing results in {output_path}')

    # Self-describing result folder: tee output into <output_path>/run.log and dump config.
    # (After the overwrite-clear so run_config.json isn't deleted by the .json sweep above.)
    start_self_logging(output_path)
    dump_run_config(output_path, args)

    # --- build the base env (full game pool) once ---
    from alfworld.agents.environment import get_environment
    with open('Alfworld/base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_in_distribution" if args.split == 'dev' else "eval_out_of_distribution"
    TEMPLATE_ENV = get_environment(config["env"]["type"])(config, train_eval=split)
    game_files = sorted(TEMPLATE_ENV.game_files)   # deterministic game_idx across reruns
    n = len(game_files)
    if args.num_games > 0:
        n = min(args.num_games, n)
    print(f"Total ALFWorld games: {n}  (group size / concurrency = {args.batch_size})")

    # --- resume: skip games whose idx_*.json already exists ---
    done_idx = set()
    for f in os.listdir(output_path):
        if f.startswith('idx_') and f.endswith('.json'):
            try:
                done_idx.add(int(f[len('idx_'):-len('.json')]))
            except ValueError:
                pass
    if done_idx:
        print(f"Resuming: {len(done_idx)} games already finished, will skip them.")

    concurrency = args.concurrency if args.concurrency > 0 else args.batch_size

    # --- rolling pool: submit ALL remaining games at once, bounded by concurrency ---
    todo = [(gi, game_files[gi]) for gi in range(n) if gi not in done_idx]
    print(f"Submitting {len(todo)} games to a rolling pool of {concurrency} workers "
          f"(no group barrier).")

    finished = 0
    rew_running = 0.0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_one_game, gf, gi, model, args.max_steps): gi
                   for gi, gf in todo}
        for fut in as_completed(futures):
            gi = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"[game {gi}] ERROR: {e}")
                continue
            with open(f'{output_path}/idx_{res["game_idx"]}.json', 'w') as f:
                json.dump({"messages": res["messages"],
                           "reward":   res["reward"],
                           "name":     res["name"]},
                          f, indent=4, ensure_ascii=False)
            finished += 1
            rew_running += res["reward"]
            elapsed = time.time() - t_start
            rate = finished / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - finished) / rate if rate > 0 else 0.0
            print(f'[{finished}/{len(todo)}] game {gi} done reward={res["reward"]}  '
                  f'running acc (this run): {rew_running / finished * 100:.2f}%  '
                  f'| elapsed {elapsed:.0f}s  {rate * 60:.1f} games/min  ETA {eta:.0f}s')

    total_elapsed = time.time() - t_start

    # --- final accuracy + wall-clock ---
    total = 0
    rew = 0.0
    for f in os.listdir(output_path):
        # Only per-game result files (idx_*.json) — NOT run_config.json / other sidecars.
        if f.startswith('idx_') and f.endswith('.json'):
            total += 1
            rew += json.load(open(os.path.join(output_path, f)))['reward']
    print(f'\nFinal accuracy: {rew / max(total, 1) * 100:.2f}%  ({int(rew)}/{total})')
    print(f'Total wall-clock: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min) '
          f'for {finished} games this run '
          f'({finished / total_elapsed * 60:.1f} games/min).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fastest no-memory ALFWorld baseline (rolling-pool, no barrier)')
    parser.add_argument('--model',       type=str, default='openai/Qwen/Qwen3-8B')
    parser.add_argument('--env',         type=str, default='alfworld', choices=['alfworld'])
    parser.add_argument('--memory_type', type=str, default='none', choices=['none'])
    parser.add_argument('--split',       type=str, default='dev', choices=['dev', 'test'])
    parser.add_argument('--concurrency', type=int, default=0,
                        help='Max games in flight in the rolling pool '
                             '(0 -> fall back to --batch_size)')
    parser.add_argument('--batch_size',  type=int, default=30,
                        help='Fallback concurrency if --concurrency is 0')
    parser.add_argument('--max_steps',   type=int, default=30)
    parser.add_argument('--exp_name',    type=str, default='exp')
    parser.add_argument('--few_shot',    action='store_true')
    parser.add_argument('--num_games',   type=int, default=0, help='0 = all games')
    parser.add_argument('--overwrite',   action='store_true')
    args = parser.parse_args()

    main(args)
