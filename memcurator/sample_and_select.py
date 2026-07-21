"""Offline: sample bare-executor trajectories on ALFWorld TRAIN, estimate p-hat, build the pool.

This is the FIRST step before training the MemCurator (run it before the training smoke). It
does three jobs in one pass over the train split:

  1. For each train game, run the FROZEN executor with NO briefing (empty context) K times,
     using the EVAL-parity prompt (memcurator.alfworld_executor) + litellm. Non-thinking by
     default (PROMPT_STYLE=revise_react, ENABLE_THINKING=false, temp 0.7) — matches the strong
     hist5-nonthink baseline; the TRAINING executor must use the same config for p-hat to transfer.
  2. Record per-game pass rate p-hat = (#successes / K). Write a histogram + a selected list of
     games with p-hat in [--lo, --hi] (the "learnable band" where a briefing has headroom AND
     GRPO gets signal). This measures how saturated the train split is → decides the
     global-pool-vs-per-task-pinning fork.
  3. Save every successful trajectory in the CuratorAlfworld JSONL schema
     ({task_id, query, trajectory, status}) → the FROZEN GLOBAL POOL the trainer retrieves from
     (and which build_curator_stores.py can slice into per-task S_T for v2).

Standalone: no verl/Ray. Builds its own ALFWorld env (train_eval='train'), deepcopies it per
game (eval's run_one_game pattern; tatsu PDDL parser is not thread-safe so init/reset/step are
serialized under a lock), and calls the executor via litellm. Concurrency overlaps LLM latency.

Usage (on the box, conda env `memory`, cwd = evaluation/agent_eval so Alfworld/base_config.yaml
resolves; OPENAI_API_BASE -> served frozen executor):
    cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval
    OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \
    ENABLE_THINKING=false PROMPT_STYLE=revise_react EXECUTOR_TEMPERATURE=0.7 \
    python -m memcurator.sample_and_select \
        --num_games 80 --k 8 --max_steps 30 --concurrency 16 \
        --out_dir /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/memcurator/pilot

DISK NOTE: textworld copies a ~35MB libdownward.so into a temp dir on EVERY episode (auto-removed
right after, so it does NOT leak — but at high --concurrency many copies coexist for a moment). By
default those land in <out_dir>/_tmp (same big disk as the outputs); override with --tmpdir or a
caller $TMPDIR. This is what prevents the [Errno 28] No space left crash the frac=0.2 run hit when
--concurrency 128 flooded the 31GB root /tmp. See _route_tmpdir.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

# Executor prompt/parse — eval-parity (byte-identical templates, guarded by test_executor_parity).
from memcurator.alfworld_executor import (
    build_executor_prompt,
    parse_action,
    process_ob,
)

# tatsu PDDL parser is NOT thread-safe → serialize env init/reset/step (eval's PDDL_LOCK pattern).
_PDDL_LOCK = threading.Lock()


def _route_tmpdir(out_dir: str, tmpdir: Optional[str] = None) -> str:
    """Point per-episode temp copies at a BIG filesystem so high --concurrency can't fill /tmp.

    ROOT CAUSE of the frac=0.2 crash ([Errno 28] No space left on /tmp/tmpXXXX/libdownward.so):
    textworld/fast_downward's ``load_lib()`` copies a ~35MB ``libdownward.so`` into a fresh
    ``tempfile.TemporaryDirectory()`` on EVERY episode's ``init_env()``. Each copy is auto-removed
    the instant ``load_lib()`` returns (verified by a leak probe: there is NO accumulation, and
    ``tw.close()`` frees nothing extra), BUT at --concurrency 128 up to 128 copies (~4.4GB) coexist
    for a moment; on the 31GB root disk that momentary peak overflows. Every eval command in
    RUN_COMMAND_Log.sh already sets ``TMPDIR=$HOME/tmp`` (on the 12T /fsx) for exactly this reason;
    the sampler simply forgot to. This routes the copies onto the big disk unconditionally.

    Precedence: explicit ``tmpdir`` arg > caller's ``$TMPDIR`` > ``<out_dir>/_tmp`` (guaranteed on
    the same big filesystem as the outputs the user chose). Sets BOTH ``os.environ['TMPDIR']`` and
    ``tempfile.tempdir`` (the latter overrides ``tempfile.gettempdir()``'s process-cached value, so
    it takes effect even if something already called gettempdir()).
    """
    import tempfile
    if tmpdir:
        tmp = tmpdir
    elif os.environ.get("TMPDIR"):
        tmp = os.environ["TMPDIR"]
    else:
        tmp = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp, exist_ok=True)
    os.environ["TMPDIR"] = tmp
    tempfile.tempdir = tmp  # defeat any cached gettempdir() so textworld's copies land here
    print(f"[sample_and_select] TMPDIR -> {tmp} (per-episode ~35MB libdownward.so copies land here)")
    return tmp


class _Tee:
    """Mirror everything written to stdout/stderr into <out_dir>/run.log AS WELL as the console.

    So a `run.log` always exists next to the outputs even if the launch command has no shell
    `| tee` (the script itself does not otherwise write a log). Line-buffered + flushed per write
    so `tail -f run.log` works live. Installed by _install_run_log(); restored is unnecessary
    (process exits after the run).
    """
    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return getattr(self._stream, "isatty", lambda: False)()


def _install_run_log(out_dir: str) -> str:
    """Redirect stdout+stderr through a _Tee into <out_dir>/run.log (append). Returns the path."""
    import sys
    log_path = os.path.join(out_dir, "run.log")
    fh = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print(f"[sample_and_select] logging console output to {log_path}")
    return log_path


def _messages_to_trajectory(messages: List[Dict], backend) -> str:
    """Build the stored 'trajectory' via the SELECTED curator backend's _trajectory_to_text,
    so it is byte-identical to what THAT eval variant stores (--curator_variant)."""
    return backend.trajectory_to_text(messages)


def _raw_to_trajectory_w_thinking(raw_responses: List[Dict]) -> str:
    """[w/thinking ablation] Build an obs + FULL-response (think+action) trajectory from raw
    per-step responses. Eval has NOT finalized the w/thinking format yet, so this is a provisional
    rendering (obs + raw response, which contains <think>…</think> + <action>…</action>). Kept
    alongside the current obs/action 'trajectory' so the format can be reorganized later w/o re-rolling.
    """
    parts = []
    for i, r in enumerate(raw_responses):
        parts.append(f"[Step {i}]")
        parts.append(f"[Observation]: {r['observation']}")
        parts.append(f"[Response]: {r['response']}")  # full response incl. <think> and <action>
        parts.append("")
    return "\n".join(parts)

# The 6 ALFWorld task-type categories (kept in sync with agent_system env_manager).
ALFWORLD_TASK_TYPE_NAMES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


def _category_of(game_file: str) -> Optional[str]:
    """The task-type category a game_file belongs to (first matching name), or None."""
    for t in ALFWORLD_TASK_TYPE_NAMES:
        if t in game_file:
            return t
    return None


def _select_games(all_games: List[str], num_games: int, stratify: bool,
                  frac: float, seed: int) -> List[str]:
    """Pick target games.

    stratify=False: shuffle and take the first ``num_games`` (flat random).
    stratify=True : take ``frac`` of EACH of the 6 categories (category-balanced), so every
                    task type is represented proportionally. ``num_games`` is ignored.
    """
    import random
    rng = random.Random(seed)
    if not stratify:
        g = list(all_games)
        rng.shuffle(g)
        return g[:num_games]

    by_cat: Dict[str, List[str]] = defaultdict(list)
    for g in all_games:
        c = _category_of(g)
        if c is not None:
            by_cat[c].append(g)
    picked: List[str] = []
    for cat in ALFWORLD_TASK_TYPE_NAMES:
        games_c = by_cat.get(cat, [])
        rng.shuffle(games_c)
        n_c = max(1, int(round(len(games_c) * frac))) if games_c else 0
        picked.extend(games_c[:n_c])
        print(f"  [stratify] {cat}: {len(games_c)} total -> {n_c} sampled ({frac:.0%})")
    rng.shuffle(picked)
    return picked

# Executor sampling knobs (read from env, mirroring the eval runner's EXECUTOR_* / ENABLE_THINKING).
_TEMP = float(os.environ.get("EXECUTOR_TEMPERATURE", "0.7"))
_TOP_P = os.environ.get("EXECUTOR_TOP_P")
_TOP_K = os.environ.get("EXECUTOR_TOP_K")
_MAX_TOKENS = os.environ.get("EXECUTOR_MAX_TOKENS")
_ET = os.environ.get("ENABLE_THINKING", "")
_ENABLE_THINKING = None if _ET == "" else (_ET.lower() in ("1", "true", "yes"))


def _executor_call(completion_fn, prompt: str, model: str, api_base: str, api_key: str) -> str:
    """One frozen-executor call via litellm — identical kwargs to the eval runner's llm()."""
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        base_url=api_base,
        num_retries=10,
        temperature=_TEMP,
    )
    if _TOP_P is not None:
        kwargs["top_p"] = float(_TOP_P)
    if _MAX_TOKENS is not None:
        kwargs["max_tokens"] = int(_MAX_TOKENS)
    extra_body = {}
    if _TOP_K is not None:
        extra_body["top_k"] = int(_TOP_K)
    if _ENABLE_THINKING is not None:
        extra_body["chat_template_kwargs"] = {"enable_thinking": _ENABLE_THINKING}
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        resp = completion_fn(**kwargs)
        return resp.choices[0].message.content or "Output Error"
    except Exception as e:  # noqa: BLE001
        return f"Output Error: {e}"


def _run_one_episode(template_env, game_file: str, model: str, api_base: str, api_key: str,
                     max_steps: int, history_length: int, first_prompts_out: Optional[list] = None):
    """Run ONE bare-executor episode on a pinned game.

    Returns a dict: {won, task, task_id, messages, raw_responses}.
      * messages       — eval-parity (obs,action) message list, EXACTLY as run_one_game builds it
                         (user=obs, assistant=raw response). Fed to CuratorAlfworld.add ->
                         _trajectory_to_text, so the stored "trajectory" is byte-identical to eval.
      * raw_responses  — the FULL executor response per step (incl. any <think>), so a future
                         w/thinking trajectory format can be rebuilt WITHOUT re-rolling.
    Mirrors eval's run_one_game: deepcopy the template env, pin game_files, reset under the PDDL
    lock, step with NO briefing (ctx_text=""), eval-parity prompt, PARSED action to the env.
    ``first_prompts_out``: if provided, appends the per-step (step, prompt) for sample dumping.
    """
    from litellm import completion

    with _PDDL_LOCK:
        pinned = copy.deepcopy(template_env)
        pinned.game_files = [game_file]
        tw = pinned.init_env(batch_size=1)
        ob_raw, info = tw.reset()
        raw = ob_raw[0]
        current_ob = "\n".join(raw.split("\n\n")[1:])  # step-0 [1:] slice (room + task)
        current_admissible = info["admissible_commands"][0]
        name = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])
        task_id = name.replace("/", "_")
    task = raw.split("\nYour task is to: ")[-1] if "Your task is to:" in raw else current_ob

    # eval-parity message list: run_one_game appends (user=obs, assistant=RAW response) per step,
    # then a trailing user=current_ob. CuratorAlfworld._trajectory_to_text consumes this.
    history: List[tuple] = []            # (obs, parsed_action) — for building the executor prompt
    messages: List[Dict] = []            # eval-parity messages (raw assistant content)
    raw_responses: List[Dict] = []       # {step, observation, response, action} — for w/thinking rebuild
    won = False

    for step_count in range(max_steps):
        prompt = build_executor_prompt(
            step_count=step_count,
            current_observation=current_ob,
            admissible_commands=current_admissible,
            task_description=task,
            history=history,
            history_length=history_length,
            ctx_text="",  # BARE executor: no briefing
        )
        if first_prompts_out is not None:
            first_prompts_out.append({"step": step_count, "prompt": prompt})
        response = _executor_call(completion, prompt, model, api_base, api_key)
        action = parse_action(response) or ""  # standalone alfworld env needs the PARSED action

        # eval-parity: messages carry (obs, RAW response); _trajectory_to_text re-extracts <action>.
        messages.append({"role": "user", "content": current_ob})
        messages.append({"role": "assistant", "content": response})
        raw_responses.append({"step": step_count, "observation": current_ob,
                              "response": response, "action": action})

        with _PDDL_LOCK:
            obs_raw, _scores, done, info = tw.step([action])  # step with parsed action (eval parity)
        next_ob = process_ob(obs_raw[0])
        current_admissible = info.get("admissible_commands", [current_admissible])[0]
        history.append((current_ob, action))
        current_ob = next_ob

        if done[0]:
            won = bool(info["won"][0])
            break

    messages.append({"role": "user", "content": current_ob})  # trailing final obs (eval parity)
    return {"won": won, "task": task, "task_id": task_id,
            "messages": messages, "raw_responses": raw_responses}


def sample_and_select(
    num_games: int,
    k: int,
    max_steps: int,
    concurrency: int,
    out_dir: str,
    model: str,
    api_base: str,
    api_key: str,
    history_length: int,
    lo: float,
    hi: float,
    stratify: bool = False,
    frac: float = 0.1,
    curator_variant: str = "curator_alfworld",
    curation_mode: str = "success_only",
    keep: str = "success",
    seed: int = 42,
    tmpdir: Optional[str] = None,
) -> None:
    import yaml
    from alfworld.agents.environment import get_environment
    from memcurator.curator_backend import CuratorBackend

    os.makedirs(out_dir, exist_ok=True)
    # Always write <out_dir>/run.log (mirrors console), so a log exists regardless of how the
    # launch command pipes output — install FIRST so all subsequent prints are captured.
    _install_run_log(out_dir)
    # Route per-episode textworld temp copies onto a big disk BEFORE any env is built (else the
    # first init_env() copies libdownward.so to the default /tmp). See _route_tmpdir for the why.
    _route_tmpdir(out_dir, tmpdir)
    # NOTE: Stage A does NOT build any curator prompt — it only runs the EXECUTOR and stores
    # trajectories. The backend is used ONLY for _trajectory_to_text (messages -> stored text),
    # which is byte-IDENTICAL across all curator variants. So curator_variant/curation_mode have
    # NO effect on Stage A output; they are kept for provenance + interface symmetry with Stage B /
    # training, where the curator prompt + _format_case Result: line actually depend on them.
    backend = CuratorBackend(variant=curator_variant, curation_mode=curation_mode)
    print(f"[sample_and_select] curator backend: {curator_variant} (curation_mode={curation_mode}) "
          f"— used only for trajectory formatting (identical across variants)")

    with open("Alfworld/base_config.yaml") as f:
        config = yaml.safe_load(f)
    template_env = get_environment(config["env"]["type"])(config, train_eval="train")
    all_games = list(template_env.game_files)
    games = _select_games(all_games, num_games, stratify, frac, seed)
    mode = f"stratified {frac:.0%}/category" if stratify else f"flat {num_games}"
    print(f"[sample_and_select] train games total={len(all_games)}, sampling {len(games)} "
          f"({mode}) x K={k} (thinking={_ENABLE_THINKING}, temp={_TEMP})")

    # ------------------------------------------------------------------ #
    # CRASH-SAFE INCREMENTAL DUMP + RESUME                                #
    # Every finished episode is appended to rollouts_raw.jsonl IMMEDIATELY (under a lock), so a
    # crash mid-run loses at most the in-flight episodes. On restart we read that file, skip the
    # (game_file, rollout) pairs already done, and re-run only the rest. rollouts_raw.jsonl always
    # holds ALL rollouts (success + fail) regardless of --keep; --keep is applied only when
    # deriving the final pool_source.jsonl. The final targets/selected/pool are computed from the
    # combined (resumed + new) raw records at the end.
    # ------------------------------------------------------------------ #
    try:
        from tqdm import tqdm as _tqdm
    except Exception:  # noqa: BLE001
        def _tqdm(x, **kw):  # minimal fallback
            return x

    raw_path = os.path.join(out_dir, "rollouts_raw.jsonl")
    _raw_lock = threading.Lock()
    raw_records: List[Dict] = []
    done_keys = set()  # (game_file, rollout_idx) already completed
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                raw_records.append(rec)
                done_keys.add((rec["game_file"], rec["rollout"]))
        print(f"[sample_and_select] RESUME: {len(done_keys)} rollouts already done in {raw_path}")

    def _dump_raw(rec: Dict) -> None:
        with _raw_lock:
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Fan out only the NOT-yet-done (game, rollout) jobs. First new job captures its per-step
    # executor prompts for the eval-parity sample file.
    all_jobs = [(g, r) for g in games for r in range(k)]
    jobs = [(g, r) for (g, r) in all_jobs if (g, r) not in done_keys]
    _sample_prompts: List[Dict] = []
    print(f"[sample_and_select] {len(all_jobs)} total rollouts, {len(jobs)} to run "
          f"({len(all_jobs) - len(jobs)} resumed) at concurrency {concurrency}")

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = {}
        for j, (g, r) in enumerate(jobs):
            capture = _sample_prompts if j == 0 else None
            futs[pool.submit(_run_one_episode, template_env, g, model, api_base, api_key,
                             max_steps, history_length, capture)] = (g, r)
        for fut in _tqdm(as_completed(futs), total=len(jobs), desc="rollouts"):
            g, r = futs[fut]
            try:
                e = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[sample_and_select] episode error on {g} r{r}: {exc}")
                continue
            # Build the crash-safe raw record (final fields precomputed, so no re-derivation on resume).
            rec = {
                "game_file": g, "rollout": r,
                "task_id": e["task_id"], "query": e["task"], "category": _category_of(g),
                "won": bool(e["won"]),
                "reward": 1.0 if e["won"] else 0.0,
                "status": "success" if e["won"] else "fail",
                "trajectory": _messages_to_trajectory(e["messages"], backend),
                "trajectory_w_thinking": _raw_to_trajectory_w_thinking(e["raw_responses"]),
                "raw_responses": e["raw_responses"],
            }
            _dump_raw(rec)
            raw_records.append(rec)

    # ------------------------------------------------------------------ #
    # Aggregate the combined (resumed + new) raw records into the final outputs.
    # ------------------------------------------------------------------ #
    by_game: Dict[str, List[Dict]] = defaultdict(list)
    for rec in raw_records:
        by_game[rec["game_file"]].append(rec)

    target_rows: List[Dict] = []
    pool_records: List[Dict] = []
    hist = Counter()
    for g, rolls in by_game.items():
        if not rolls:
            continue
        wins = sum(1 for e in rolls if e["won"])
        phat = wins / len(rolls)
        r0 = rolls[0]
        target_rows.append({"game_file": g, "task_id": r0["task_id"], "query": r0["query"],
                            "category": r0["category"], "p_hat": phat,
                            "n_rollouts": len(rolls), "n_success": wins})
        hist[round(phat, 3)] += 1
        # --keep filter: 'success' -> only wins into pool_source; 'all' -> keep failures too.
        # (rollouts_raw.jsonl always has ALL; this is just the downstream view for Stage B.)
        for e in rolls:
            if keep == "success" and not e["won"]:
                continue
            pool_records.append({
                "task_id": e["task_id"], "game_file": g, "query": e["query"],
                "category": e["category"],
                "trajectory": e["trajectory"],
                "trajectory_w_thinking": e["trajectory_w_thinking"],
                "raw_responses": e["raw_responses"],
                "reward": e["reward"], "status": e["status"],
            })

    # write outputs
    targets_path = os.path.join(out_dir, "targets.jsonl")
    with open(targets_path, "w", encoding="utf-8") as f:
        for row in target_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected = [r for r in target_rows if lo <= r["p_hat"] <= hi]
    sel_path = os.path.join(out_dir, "selected.jsonl")
    with open(sel_path, "w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # pool_source.jsonl: all harvested successes (CuratorAlfworld schema + game_file/category for
    # Stage B self-exclusion + category-controlled pools + trajectory_w_thinking/raw for later reorg).
    # Loadable by build_curator_stores.load_pool (extra keys ignored).
    pool_path = os.path.join(out_dir, "pool_source.jsonl")
    with open(pool_path, "w", encoding="utf-8") as f:
        for rec in pool_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # sample_executor_prompts.txt: per-step executor-input prompts from the first NEW episode, for
    # eval-parity comparison. Only (re)write when we actually captured prompts this run — on a full
    # resume (0 new jobs) _sample_prompts is empty, so we DON'T clobber a prior good sample file.
    sample_path = os.path.join(out_dir, "sample_executor_prompts.txt")
    if _sample_prompts:
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(f"# Executor-input prompts (first episode). PROMPT_STYLE={os.environ.get('PROMPT_STYLE','think')} "
                    f"ENABLE_THINKING={_ENABLE_THINKING} HISTORY_LENGTH={history_length}\n")
            f.write("# Compare vs eval run_one_game prompts (bare executor, ctx_text='').\n\n")
            for p in _sample_prompts:
                f.write(f"===== STEP {p['step']} =====\n{p['prompt']}\n\n")
    # sample_pool_records.jsonl: first few pool records (trajectory + trajectory_w_thinking + raw),
    # for the user to diff against the real curator_calls.jsonl trajectory format.
    sample_rec_path = os.path.join(out_dir, "sample_pool_records.jsonl")
    with open(sample_rec_path, "w", encoding="utf-8") as f:
        for rec in pool_records[:3]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _print_report(target_rows, selected, pool_records, hist, lo, hi)
    print(f"[sample_and_select] raw (crash-safe, all rollouts): {raw_path}  ({len(raw_records)} records)")
    print(f"[sample_and_select] wrote: {targets_path} | {sel_path} | {pool_path}")
    print(f"[sample_and_select] samples for parity check: {sample_path} | {sample_rec_path}")


def _print_report(target_rows, selected, pool_records, hist, lo, hi) -> None:
    n = len(target_rows)
    if n == 0:
        print("[sample_and_select] WARNING: 0 games produced results.")
        return
    mean_phat = sum(r["p_hat"] for r in target_rows) / n
    sat0 = sum(1 for r in target_rows if r["p_hat"] == 0.0)
    sat1 = sum(1 for r in target_rows if r["p_hat"] == 1.0)
    # per-category p-hat means (diagnostic: which task types have headroom)
    cat_phats: Dict[str, List[float]] = defaultdict(list)
    for r in target_rows:
        cat_phats[r.get("category") or "unknown"].append(r["p_hat"])
    print("\n================ p-hat report ================")
    print("per-category mean p-hat:")
    for c in sorted(cat_phats):
        v = cat_phats[c]
        print(f"  {c:<32}: {sum(v)/len(v):.3f}  (n={len(v)})")
    print(f"games measured : {n}")
    print(f"mean p-hat     : {mean_phat:.3f}")
    print(f"p-hat == 0     : {sat0} ({100*sat0/n:.1f}%)   [hopeless: no gradient]")
    print(f"p-hat == 1     : {sat1} ({100*sat1/n:.1f}%)   [saturated: no gradient]")
    print(f"selected [{lo},{hi}]: {len(selected)} ({100*len(selected)/n:.1f}%)   [learnable band]")
    print(f"pool successes : {len(pool_records)} trajectories")
    print("p-hat histogram:")
    for ph in sorted(hist):
        bar = "#" * hist[ph]
        print(f"  {ph:>4}: {hist[ph]:>3} {bar}")
    print("==============================================")
    # Fork guidance:
    mid_frac = len(selected) / n
    if mid_frac >= 0.4:
        print(">> Most games are in the learnable band → global-pool + random sampling (Resolution A) is fine.")
    else:
        print(f">> Only {100*mid_frac:.0f}% in the learnable band → consider per-task pinning to the selected "
              f"games (saturation would waste many GRPO groups under random sampling).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample bare-executor ALFWorld trajectories, estimate p-hat, build pool.")
    ap.add_argument("--num_games", type=int, default=80, help="Flat mode: total games (ignored if --stratify).")
    ap.add_argument("--stratify", action="store_true",
                    help="Sample --frac of EACH of the 6 task-type categories (category-balanced).")
    ap.add_argument("--frac", type=float, default=0.1, help="Fraction per category when --stratify.")
    ap.add_argument("--k", type=int, default=8, help="Rollouts per game for p-hat.")
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model", default=os.environ.get("EXECUTOR_MODEL", "openai/Qwen/Qwen3-8B"))
    ap.add_argument("--api_base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--history_length", type=int, default=int(os.environ.get("HISTORY_LENGTH", "3")))
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.75)
    ap.add_argument("--curator_variant", default="curator_alfworld",
                    choices=["curator_alfworld", "curator_alfworld_v1", "curator_alfworld_v1_api"],
                    help="Which eval curator module to match for trajectory rendering (default: curator_alfworld).")
    ap.add_argument("--curation_mode", default="success_only",
                    help="Curation mode for v1/v1_api variants (e.g. success_only, success_and_fail, "
                         "success_only_v1, success_and_fail_v1). Ignored by the default variant.")
    ap.add_argument("--keep", default="success", choices=["success", "all"],
                    help="Which rollouts to keep in pool_source.jsonl: 'success' (default, every "
                         "successful rollout) or 'all' (incl. failures, for success_and_fail pools). "
                         "All successes/failures are kept (not one) so Stage B can pick 1..N per task.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tmpdir", default=None,
                    help="Dir for textworld's per-episode ~35MB libdownward.so copies. Default: "
                         "$TMPDIR if set, else <out_dir>/_tmp. Put this on a BIG disk (e.g. /fsx) so "
                         "high --concurrency can't fill the small root /tmp (the frac=0.2 crash cause).")
    args = ap.parse_args()

    sample_and_select(
        num_games=args.num_games, k=args.k, max_steps=args.max_steps,
        concurrency=args.concurrency, out_dir=args.out_dir, model=args.model,
        api_base=args.api_base, api_key=args.api_key, history_length=args.history_length,
        lo=args.lo, hi=args.hi, stratify=args.stratify, frac=args.frac,
        curator_variant=args.curator_variant, curation_mode=args.curation_mode,
        keep=args.keep, seed=args.seed, tmpdir=args.tmpdir,
    )


if __name__ == "__main__":
    main()
