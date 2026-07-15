# SkillCurator ALFWorld Eval — Bug Fixes & Progress Log

Progress log for the ALFWorld downstream-eval work: environment setup, two bugs found
and fixed, the thinking-mode issue, the speed rewrite, and the resulting file set.

Servers (as of this log):
- `ip-10-0-225-57` — Qwen3-8B, data-parallel=8, port 8001, STOCK chat template, flash-attn confirmed.
- `ip-10-0-240-113` — Qwen3-32B.
- Root disk `/` is 100% full on 225-57 (system CUDA/container files, not user data). Use
  `TMPDIR=$HOME/tmp`; all user data/caches live on `/fsx` (plenty of space).

---

## Bug 1 — Step-0 missing task (correctness)

**Severity:** correctness. Affects every ALFWorld runner descended from `run_unified.py`.

### Symptom
On the FIRST step of every episode, the executor prompt does not contain the task goal.
Real log:
```
[game 104] step0 response:
<think> ... The user hasn't given a specific task yet, so maybe I should start by exploring ...
```

### Root cause
`env.reset()` returns a 3-chunk observation split by `\n\n`:
```
[0] "-= Welcome to TextWorld, ALFRED! =-"
[1] "You are in the middle of a room. ... you see a cabinet 1, ..."   (room)
[2] "Your task is to: put a clean soapbar in toilet."                 (task)
```
The code kept only chunk `[1]`, dropping the task:
```python
ob_list = ['\n'.join(ob.split('\n\n')[1:2]) for ob in ob_list]   # [1:2] -> task ([2]) dropped
```
AND the step-0 template `ALFWORLD_TEMPLATE_NO_HIS` has no `{task_description}` field (steps 1+
use `ALFWORLD_TEMPLATE`, which does). So the task appears nowhere at step 0.

### Fix
```python
ob_list = ['\n'.join(ob.split('\n\n')[1:]) for ob in ob_list]   # [1:] keeps room + task
```
In the async runners the same slice is in `run_one_game`:
```python
current_ob = '\n'.join(raw.split('\n\n')[1:])   # was [1:2]
```
**Only affects step 0.** At step>0 `current_obs`/`current_ob` is overwritten with
`env.step()`'s observation (never sliced), and the task is injected via the template's
`{task_description}` field — so steps 1+ are unchanged.

### Verification (probe)
Two probe files (verbatim copies of `run_unified.py` + a one-line print of
`task_descriptions[idx] in prompt_text`):
- `run_unified_step0_bugtest.py`  -> `task_in_prompt=False` (bug present)
- `run_unified_step0_fixtest.py`  -> `task_in_prompt=True` at steps 0,1,2 (fix works)

Command:
```bash
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
EXECUTOR_MAX_TOKENS=64 ENABLE_THINKING=false \
python -u run_unified_step0_fixtest.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name step0_fix_probe \
    --batch_size 1 --num_games 1 --max_steps 3 --overwrite 2>&1 | grep PROBE
```

### Known cosmetic side effect (not a bug)
Because the fixed step-0 observation now contains the task, when replayed in history at
steps 1+ the task appears twice (template field + `Observation 1`). Harmless redundancy.

### Full write-up
See `STEP0_TASK_BUG.md`.

---

## Bug 2 — Duplicate games when batch_size does not divide 140

**Severity:** metric correctness (inflated file count / double-counted games).

### Symptom
Result dirs with **150** `idx_*.json` files instead of 140; 10 game names appear twice
(e.g. `idx_6` and `idx_143` are the same game).

### Root cause
The batch loop iterates `range(math.ceil(num_games / batch_size))` and each `env.reset()`
serves a full batch. With `batch_size=30`: `ceil(140/30)=5` batches × 30 = 150 slots, and the
5th batch wraps around, re-running 10 already-seen games (written as `idx_140..149`).

### Fix / guidance
Use a `--batch_size` that DIVIDES 140: **20, 28, 35, 10** (not 30). Verified: batch 30 ->
"150 slots, 140 distinct, 10 served twice"; a divisor -> exactly 140.
- The **async (rolling-pool) runners are immune** — they submit each game once by index.
- For existing 150-file dirs, score only the 140 distinct games (dedup by `name`, or drop
  `idx_140..149`).

---

## Thinking-mode issue (not a bug, but a real interaction)

### Findings
- The **executor** sends no thinking flag in `run_unified.py`; thinking is decided by the
  vLLM server's chat template. Memory-curation paths DO set `enable_thinking=False`.
- Author confirmed: executor should be NON-thinking; memory generation IS thinking.
- vLLM 0.8.5 has NO `--chat-template-kwargs` launch flag. Two ways to force no-think:
  1. Custom template `qwen3_nothink.jinja` (flips the default) — served-side.
  2. Per-request `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` (client-side).

### The `<think>` collision
The ALFWorld prompt mandates reasoning inside `<think></think>`. Under no-think, the model
is handed a pre-closed empty `<think></think>` AND told to produce `<think>` — it emits
stray `</think></think>...` and often no `<action>` (degenerate output).

### Resolution — prompt variants
- **`_reason_tag`**: swap the mandate tag to `<reason></reason>` (no collision with the
  `<think>` no-think prefill). Verified 0/10 leak, valid actions.
- **`_revise_react`**: remove the reasoning-tag mandate entirely (plain-text reasoning).
- Plain `<think>` prompt is only correct with **thinking ON**.

Pairing rule:
| Prompt file variant | Use with |
|---|---|
| `<think>` mandate | thinking ON (stock server / `ENABLE_THINKING=true`) |
| `_reason_tag` (`<reason>`) | no-think |
| `_revise_react` (no mandate) | no-think |

---

## Speed rewrite (step-sync -> async)

- `run_unified.py` is STEP-synchronous: fires all N agents' step, waits for the slowest,
  env-steps together, repeats. Per-step barrier + inter-step queue drain -> ~40% GPU util.
- **Group episode-async** (`run_unified_hyper_concurrent.py`): games run to completion
  concurrently within a group of `--batch_size`, barrier between groups. Higher util; clean
  hook for per-group memory updates.
- **Rolling-pool async** (`run_unified_hyper_async*.py`): all games submitted to one pool of
  `--concurrency`; a finished game's slot is refilled immediately (no barrier). Highest util.
  The tatsu PDDL parser is not thread-safe, so env ops are under a global lock; the LLM call
  is OUTSIDE the lock (that's what overlaps). Env pinning verified: `game_idx i` -> `game_files[i]`,
  each of 140 games exactly once.

---

## Configurable knobs added to the runners

Env vars (executor path only; curation unchanged):
- `EXECUTOR_TEMPERATURE` (default 0.7), `EXECUTOR_TOP_P`, `EXECUTOR_TOP_K` (via extra_body),
  `EXECUTOR_MAX_TOKENS` (unset -> uncapped).
- `ENABLE_THINKING` (`hyper*` files only; the `nonthink*` files force it False).
- `PROMPT_SHOW_EVERY=N` — every N steps print the probe agent's full PROMPT + RESPONSE, to
  spot-check correctness. Batch files: probe = first active agent. Async files: probe = game 0.
- Per-step response printing reduced to the ONE probe agent (was all ~N agents) to cut I/O.
- Async files also print timing: per-game `elapsed / games/min / ETA` and a final
  `Total wall-clock`.

---

## File inventory

### Bug-probe / verification
- `run_unified_step0_bugtest.py` — shows Bug 1 present (`task_in_prompt=False`).
- `run_unified_step0_fixtest.py` — shows Bug 1 fixed (probe steps 0-2).

### FIXED runner set (Bug 1 fixed + knobs + reduced/periodic print)
**Batch (clean per-index; use batch_size dividing 140, e.g. 20/28):**
- `run_unified_nonthink_step0bug_fix.py`            (prompt: `<think>` — no-think forced -> COLLISION; prefer the two below)
- `run_unified_nonthink_reason_tag_step0bug_fix.py`  (prompt: `<reason>`) ✅ no-think
- `run_unified_nonthink_revise_react_step0bug_fix.py` (prompt: no mandate) ✅ no-think

**Async / rolling-pool (fastest; also have timing):**
- `run_unified_hyper_async_step0bug_fix.py`             (prompt: `<think>`; thinking env-controlled)
- `run_unified_hyper_async_reason_tag_step0bug_fix.py`   (prompt: `<reason>`)
- `run_unified_hyper_async_revise_react_step0bug_fix.py` (prompt: no mandate)

### Pre-fix runners (still carry Bug 1 unless noted)
- `run_unified_hyper.py`, `run_unified_hyper_reason_tag.py`, `run_unified_hyper_revise_react.py`
- `run_unified_hyper_concurrent.py`, `run_unified_hyper_async*.py`
- `run_unified_nonthink*.py`
- `run_unified.py` (source; memory runs) — STILL HAS Bug 1; fix before memory experiments.

### Edited existing files
- `evaluation/agent_eval/Alfworld/base_config.yaml` — hardcoded paths -> `$ALFWORLD_DATA`.
- `requirements.txt` — `flash-attn` commented out (installed from prebuilt wheel instead).

### Docs
- `STEP0_TASK_BUG.md`, `BUG_FIX_PROGRESS.md` (this file), `installment_caveat.md`,
  `INSTALL_TWO_ENVS.md`.
- `qwen3_nothink.jinja` (server-side custom template — superseded by per-request flag).

---

## Recommended run (fastest, clean no-think, reason-tag)

```bash
cd $HOME/mem-evolve/SkillCurator-main/evaluation/agent_eval
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
ENABLE_THINKING=false PROMPT_SHOW_EVERY=5 \
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_hyper_async_reason_tag_step0bug_fix.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_reasontag_async --concurrency 32 \
    2>&1 | tee logs/baseline_reasontag_async.log
```

## Open items
- `run_unified.py` (memory runs) still has Bug 1 — fix before running skillos/reasoningbank/memp.
- `run_unified_nonthink_step0bug_fix.py` keeps the `<think>` mandate under forced no-think
  (collision) — swap to `<reason>` if it will actually be used.
- Root disk on 225-57 full (system files) — sysadmin item; workaround `TMPDIR=$HOME/tmp`.
