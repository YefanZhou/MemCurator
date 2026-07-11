# Bug: ALFWorld executor is not told the task on the first step (step 0)

**Severity:** correctness — affects every ALFWorld run in this eval harness.
**Scope:** ALFWorld only (WebShop / reasoning envs unaffected).
**Status:** root-caused, reproduced, and fixed (see "Fix" and "Verification").

---

## Summary

On the **first step** of every ALFWorld episode, the prompt sent to the executor LLM
**does not contain the task goal** (`"Your task is to: ..."`). The model is asked to
choose an action having never been told what it is trying to accomplish. From step 1
onward the task is present. So every episode's first decision is made blind.

Observed symptom (real run log):

```
[game 104] step0 response:
<think>
Okay, let's see. I need to figure out what action to take next ... The user hasn't
given a specific task yet, so maybe I should start by exploring the area ...
```

The model is correct — on step 0 it genuinely was not given a task.

---

## Root cause

Two things combine, both in the ALFWorld setup path of `run_unified.py` (and every
file copied from it):

1. **The observation slice drops the task.** The raw observation returned by
   `env.reset()` has three `\n\n`-separated chunks:

   ```
   [0] "-= Welcome to TextWorld, ALFRED! =-"
   [1] "You are in the middle of a room. ... you see a cabinet 1, ..."   (room)
   [2] "Your task is to: put a clean soapbar in toilet."                 (task)
   ```

   The code keeps only chunk `[1]`:

   ```python
   # run_unified.py (ALFWorld loop, ~line 1388-1389)
   task_descriptions = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]
   ob_list           = ['\n'.join(ob.split('\n\n')[1:2]) for ob in ob_list]
   #                                              ^^^^ keeps [1] only -> task ([2]) dropped
   ```

   `task_descriptions` is parsed correctly, but the per-step `current_observation`
   (`ob_list`) loses the task.

2. **The step-0 template has no task field.** Steps 1+ use `ALFWORLD_TEMPLATE`, which
   includes `Your task is to: {task_description}`. But step 0 uses
   `ALFWORLD_TEMPLATE_NO_HIS`, which only has `{current_observation}` and
   `{admissible_actions}` — **no `{task_description}`**:

   ```python
   ALFWORLD_TEMPLATE_NO_HIS = """\
   You are an expert agent operating in the ALFRED Embodied Environment.
   Your current observation is: {current_observation}
   Your admissible actions of the current situation are: [{admissible_actions}].
   ...
   """
   ```

   So the only place the task could have appeared at step 0 (inside the observation)
   is exactly the part that got sliced away in (1).

Net effect: step 0 prompt = room + admissible actions, **no goal**.

---

## Reproduction

`run_unified_step0_bugtest.py` is a verbatim copy of `run_unified.py` with a one-line
probe added right after the step-0 prompt is built:

```python
print(f"[STEP0 PROBE game {idx}] task_in_prompt="
      f"{task_descriptions[idx].strip() in prompt_text} | prompt={prompt_text!r}")
```

Run (1 game, 1 step):

```bash
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
EXECUTOR_MAX_TOKENS=64 ENABLE_THINKING=false \
python -u run_unified_step0_bugtest.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name step0_probe \
    --batch_size 1 --num_games 1 --max_steps 1 --overwrite
```

Result (bug present):

```
[STEP0 PROBE game 0] task_in_prompt=False | prompt="You are an expert agent ...
Your current observation is: You are in the middle of a room. ... you see a cabinet 1 ...
Your admissible actions ... ['go to cabinet 1' ...]. Now it's your turn to take an action. ..."
```

`task_in_prompt=False` — no "Your task is to:" anywhere in the step-0 prompt.

---

## Fix

Change the slice from `[1:2]` to `[1:]` so the step-0 observation keeps the task line:

```python
# before
ob_list = ['\n'.join(ob.split('\n\n')[1:2]) for ob in ob_list]
# after
ob_list = ['\n'.join(ob.split('\n\n')[1:]) for ob in ob_list]
```

- **Only affects step 0.** At step 0 the prompt's `{current_observation}` now includes
  both the room and the task. Steps 1+ overwrite `current_obs` with the env's next
  observation and inject the task via the template's `{task_description}` field, so they
  are unchanged.
- Alternative fix: add a `{task_description}` field to `ALFWORLD_TEMPLATE_NO_HIS`. The
  slice fix is smaller and is what is applied here.

### Async runners note

In the concurrent/async runners (`run_unified_hyper_async*.py`) the same slice lives in
the per-game function `run_one_game` (not the batch loop):

```python
current_ob = '\n'.join(raw.split('\n\n')[1:2])   # -> change to [1:]
```

Same one-line fix.

---

## Verification of the fix

`run_unified_step0_fixtest.py` = the bugtest copy with `[1:]` applied and the probe
widened to steps 0-2. Run with `--max_steps 3`:

```bash
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
EXECUTOR_MAX_TOKENS=64 ENABLE_THINKING=false \
python -u run_unified_step0_fixtest.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-32B --exp_name step0_fix_probe \
    --batch_size 1 --num_games 1 --max_steps 3 --overwrite 2>&1 | grep PROBE
```

Result (all three steps now contain the task):

```
[PROBE game 0 step 0] task_in_prompt=True | prompt="... you see a ... towelholder 1.
    Your task is to: put a clean soapbar in toilet. Your admissible actions ..."
[PROBE game 0 step 1] task_in_prompt=True | prompt="... Your task is to: put a clean soapbar in toilet.
    Prior to this step, you have already taken 1 step(s). ..."
[PROBE game 0 step 2] task_in_prompt=True | prompt="... Your task is to: put a clean soapbar in toilet.
    Prior to this step, you have already taken 2 step(s). ..."
```

### Known cosmetic side effect (not a bug)

Because the fixed step-0 observation now contains the task, when it is stored into the
action history and replayed at steps 1+, the task text appears **twice** at those steps:
once in the template's `Your task is to:` field and once embedded in `Observation 1` of
the history. This is harmless (redundant task signal) and does not affect correctness.
It can be avoided by storing the task-free observation in history while showing the
task-included one only at step 0, but that was deemed not worth the added complexity.

---

## Affected files

Every ALFWorld runner descended from `run_unified.py` inherits the bug:

- `run_unified.py`  (source; memory runs)
- `run_unified_nonthink.py`, `run_unified_nonthink_reason_tag.py`, `run_unified_nonthink_revise_react.py`
- `run_unified_hyper.py`, `run_unified_hyper_reason_tag.py`, `run_unified_hyper_revise_react.py`
- `run_unified_hyper_async.py`, `run_unified_hyper_async_reason_tag.py`, `run_unified_hyper_async_revise_react.py`
- `run_unified_hyper_concurrent.py`

**Fixed reference file:** `run_unified_nonthink_step0bug_fix.py` (this bug fixed in the
`run_unified_nonthink.py` code path).

## Impact on results

Any ALFWorld accuracy measured before this fix used a first step made without the goal.
Short episodes (few steps) are affected proportionally more. Numbers produced pre-fix
are not directly comparable to post-fix numbers; re-run affected configurations.
