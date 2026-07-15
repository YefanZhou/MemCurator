# Runner Development History & Lineage

A living record of the ALFWorld eval runner files: where each came from, what it changes,
and its verification status. Purpose: let any collaborator / section **cross-check and trace**
which file to trust, how it differs from the pristine origin, and what has been verified.

**Update rule:** whenever a runner file is created or changed, add/adjust its row here and note
the change in the changelog at the bottom (date + what + why + verification).

Origin of truth: `SkillCurator-main-origin/evaluation/agent_eval/run_unified.py` (pristine upstream).

---

## Legend
- **step0 fix**: observation slice `[1:2]` → `[1:]` so the task goal is in the step-0 prompt
  (see `STEP0_TASK_BUG.md`). `bug` = still `[1:2]`; `fix` = `[1:]`.
- **engine**: `batch` = step-synchronous (per-step barrier); `async` = rolling-pool episode-async.
- **thinking**: how executor thinking is controlled.
- **prompt**: reasoning-tag instruction in the ALFWorld template. `PROMPT_STYLE` =
  selectable at runtime (think / reason_tag / revise_react) in `run_unified_dev.py`.
- **knobs**: env-var hyperparameters supported.

---

## Runner inventory (verified from disk)

| File | lines | engine | prompt | thinking | step0 | knobs | status |
|---|---|---|---|---|---|---|---|
| `run_unified.py` | 1577 | batch | `<think>` | server default | **bug** | temp hardcoded 0.7 | PRISTINE (== origin) |
| `run_unified_dev.py` | ~1920 | batch | `PROMPT_STYLE` | ENABLE_THINKING | **fix** | EXECUTOR_* + CURATION_* + PROMPT_STYLE + PROMPT_SHOW_EVERY/PRINT_CHARS + SAVE_RAW=N + timing | ✅ MEMORY DEV RUNNER (see dedicated section) |
| `run_unified_dev_async.py` | ~2125 | group-async | `PROMPT_STYLE` | ENABLE_THINKING | **fix** | same knobs as run_unified_dev.py | ✅ FAST MEMORY RUNNER — ALFWorld group episode-async, exact vs run_unified_dev.py (see dedicated section) |
| `run_unified_step0_bugtest.py` | 1578 | batch | `<think>` | — | bug | + probe print | shows bug (`task_in_prompt=False`) |
| `run_unified_step0_fixtest.py` | 1581 | batch | `<think>` | — | fix | + probe print | shows fix (True at steps 0-2) |
| `run_unified_hyper.py` | 1619 | batch | `<think>` | server | bug | EXECUTOR_* | superseded by *_step0bug_fix |
| `run_unified_hyper_reason_tag.py` | 1619 | batch | `<reason>` | server | bug | EXECUTOR_* | superseded |
| `run_unified_hyper_revise_react.py` | 1618 | batch | none | server | bug | EXECUTOR_* | superseded |
| `run_unified_nonthink.py` | 1580 | batch | `<think>` | forced OFF | bug | — | superseded; `<think>`+no-think COLLIDES |
| `run_unified_nonthink_reason_tag.py` | 1580 | batch | `<reason>` | forced OFF | bug | — | superseded |
| `run_unified_nonthink_revise_react.py` | 1580 | batch | none | forced OFF | bug | — | superseded |
| `run_unified_nonthink_step0bug_fix.py` | 1629 | batch | `<think>` | forced OFF | **fix** | EXECUTOR_* + PROMPT_SHOW_EVERY | ⚠ `<think>`+no-think collides |
| `run_unified_nonthink_reason_tag_step0bug_fix.py` | 1628 | batch | `<reason>` | forced OFF | **fix** | EXECUTOR_* + PROMPT_SHOW_EVERY | ✅ FIXED SET (no-think, batch) |
| `run_unified_nonthink_revise_react_step0bug_fix.py` | 1629 | batch | none | forced OFF | **fix** | EXECUTOR_* + PROMPT_SHOW_EVERY | ✅ FIXED SET (no-think, batch) |
| `run_unified_hyper_concurrent.py` | 358 | group-async | `<think>` | ENABLE_THINKING | bug | EXECUTOR_* | group-barrier variant |
| `run_unified_hyper_async.py` | 365 | async | `<think>` | ENABLE_THINKING | bug | EXECUTOR_* + timing | superseded by *_step0bug_fix |
| `run_unified_hyper_async_reason_tag.py` | 366 | async | `<reason>` | ENABLE_THINKING | bug | EXECUTOR_* + timing | superseded |
| `run_unified_hyper_async_revise_react.py` | 366 | async | none | ENABLE_THINKING | bug | EXECUTOR_* + timing | superseded |
| `run_unified_hyper_async_step0bug_fix.py` | 380 | async | `<think>` | ENABLE_THINKING | **fix** | EXECUTOR_* + timing + PROMPT_SHOW_EVERY | ✅ FIXED SET (fastest) |
| `run_unified_hyper_async_reason_tag_step0bug_fix.py` | 381 | async | `<reason>` | ENABLE_THINKING | **fix** | EXECUTOR_* + timing + PROMPT_SHOW_EVERY | ✅ FIXED SET (fastest) |
| `run_unified_hyper_async_revise_react_step0bug_fix.py` | 381 | async | none | ENABLE_THINKING | **fix** | EXECUTOR_* + timing + PROMPT_SHOW_EVERY | ✅ FIXED SET (fastest) |

(Line counts are a fingerprint — if a file's count changes unexpectedly, it was edited; re-check this table.)

---

## Lineage tree

```
run_unified.py  (PRISTINE origin — batch, <think>, step0 BUG)
├── run_unified_dev.py                    (MEMORY dev runner: step0 fix + step0 CONTEXT +
│                                          EXECUTOR_*/CURATION_* knobs + PROMPT_STYLE +
│                                          timing + PROMPT_SHOW_EVERY/PRINT_CHARS)
├── run_unified_step0_bugtest.py          (+ probe print; demonstrates bug)
├── run_unified_step0_fixtest.py          (+ probe print + step0 fix; demonstrates fix)
├── run_unified_hyper.py                  (+ EXECUTOR_* env knobs)
│   ├── run_unified_hyper_reason_tag.py       (prompt -> <reason>)
│   ├── run_unified_hyper_revise_react.py     (prompt -> no mandate)
│   ├── run_unified_hyper_concurrent.py       (engine -> group episode-async)
│   └── run_unified_hyper_async.py            (engine -> rolling-pool async + timing)
│       ├── *_reason_tag.py                       (prompt -> <reason>)
│       ├── *_revise_react.py                     (prompt -> no mandate)
│       └── *_step0bug_fix.py  (+ step0 fix + PROMPT_SHOW_EVERY)   ✅
│           ├── *_reason_tag_step0bug_fix.py                        ✅
│           └── *_revise_react_step0bug_fix.py                      ✅
└── run_unified_nonthink.py               (executor forced enable_thinking=False)
    ├── *_reason_tag.py / *_revise_react.py   (prompt variants)
    └── *_step0bug_fix.py  (+ step0 fix + EXECUTOR_* + PROMPT_SHOW_EVERY)  ✅
        ├── *_reason_tag_step0bug_fix.py                                    ✅
        └── *_revise_react_step0bug_fix.py                                  ✅
```

---

## Which file to use (decision guide)

- **Fastest baseline, no-think** → `run_unified_hyper_async_reason_tag_step0bug_fix.py`
  (async + `<reason>` + `ENABLE_THINKING=false`). `_revise_react` variant also fine.
- **Fastest baseline, thinking-on** → `run_unified_hyper_async_step0bug_fix.py`
  (`<think>` prompt + `ENABLE_THINKING=true`). This pairing is what `sweep_32b.sh`/`sweep_8b*.sh` use.
- **Clean per-index batch (comparability), no-think** → `run_unified_nonthink_reason_tag_step0bug_fix.py`
  (use `--batch_size` that divides 140: 20/28/35/10).
- **Memory runs (skillos/reasoningbank)** → **`run_unified_dev.py`** (the maintained memory
  path: step0 fix, step0 context, tunable executor+curator sampling, PROMPT_STYLE, timing).
  Only the `run_unified*` batch files carry memory logic; the hyper/async files are
  `--memory_type none` only. ⚠ `run_unified.py` still has the step0 bug AND lacks step0
  context — do NOT use it for memory runs; use `run_unified_dev.py`.
  (MemP is not in run_unified*; it runs via the legacy `run_memp_*.py`.)

### Hard pairing rule (do not violate)
`<think>` prompt ↔ thinking ON. `<reason>` / no-mandate ↔ thinking OFF.
`<think>` + no-think collides → `</think>` garbage. See `SESSION_FINDINGS_AND_GOTCHAS.md`.

---

## `run_unified_dev.py` — the memory dev runner (feature reference)

Started as a step0-fix copy of `run_unified.py`; now the maintained runner for **memory**
experiments (skillos / reasoningbank) across alfworld / webshop / reasoning. All additions
are env-var driven and **backward compatible** (unset → original behaviour). Diff vs
`run_unified.py` is additive except the one step0 slice line.

### Executor sampling knobs (env)
Mirror `run_unified_hyper_async_step0bug_fix.py`. Wired into the executor `llm()`:
`EXECUTOR_TEMPERATURE` (default 0.7), `EXECUTOR_TOP_P`, `EXECUTOR_TOP_K`, `EXECUTOR_MAX_TOKENS`
(top_k + thinking go via `extra_body.chat_template_kwargs`), `ENABLE_THINKING`
(unset → server default; true/false forces it), `HISTORY_LENGTH` (default 5).

### Curator sampling knobs (env) — symmetric to executor
`CURATION_TEMPERATURE` (default 0.7), `CURATION_TOP_P`, `CURATION_TOP_K`, `CURATION_MAX_TOKENS`,
`CURATION_ENABLE_THINKING`. Wired into **all** curation paths:
- SkillOS HTTP + Gemini + local-vLLM curators (in `run_unified_dev.py`).
- ReasoningBank HTTP + local-vLLM curators (in `reasoningbank_alfworld.py`, reading the same
  `CURATION_*` env vars, defaults preserving prior behaviour).
Defaults kept per-path: `CURATION_MAX_TOKENS` → 1024 (reasoningbank) / 4096 (skillos) when unset.
Curator thinking on the **HTTP path** is decided by the curation server's chat template unless
`CURATION_ENABLE_THINKING` is set (the HTTP call now forwards it via `chat_template_kwargs`).
Each edited call site carries a `# Was hardcoded: …` comment recording the original value.

### `PROMPT_STYLE` (env) — unifies the 3 prompt variants in ONE file
Rewrites the reasoning-tag mandate in the 6 interactive templates (ALFWorld ×3, WebShop ×3;
reasoning templates have no mandate → untouched):
| `PROMPT_STYLE` | reasoning instruction | mirrors | pair with |
|---|---|---|---|
| `think` (default) | `MUST … <think> </think>` | `run_unified_hyper_async_step0bug_fix.py` | thinking ON |
| `reason_tag` | `MUST … <reason> </reason>` | `…_reason_tag_step0bug_fix.py` | no-think |
| `revise_react` | (mandate removed) | `…_revise_react_step0bug_fix.py` | no-think |
Prints a startup WARNING if `PROMPT_STYLE=think` + `ENABLE_THINKING=false` (collision).

### Step-0 context injection (memory) — follows `run_memp_ori.py`
Retrieved skills/memories are now injected at **step 0**, not only step 1+.
- ALFWorld: new `ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT` (no `{task_description}` field — the
  task already rides in `{current_observation}` via the `[1:]` slice, so no duplication).
  Step-0 branch: `NO_HIS_WITH_CONTEXT` if ctx else `NO_HIS`.
- WebShop: context gate changed from `ctx and step_counts[i] > 0` → `ctx` (was skipping step 0).
- ⚠ **Changes memory-run numbers** (should improve; first action is where "where to find X"
  guidance matters most). Baseline `--memory_type none` unaffected. Pre-fix `run_unified.py`
  memory numbers are not directly comparable — re-run memory configs on `run_unified_dev.py`.

### `SAVE_RAW=N` (env) — persist full trajectories for N SAMPLE games
By default `idx_*.json` stores only the reconstructed obs→**parsed action** skeleton + reward
(the executor's `<think>` reasoning and the injected prompt/context are dropped — they live only
in the stdout log). `SAVE_RAW=N` adds a `"raw_trace"` key — a per-step list of
`{step, prompt, response}` with the **full injected prompt** (task + retrieved memory context +
reasoning mandate) and the **raw executor response** (incl `<think>…</think>`) — to **N sample
games evenly spread across the run** (`_raw_save_indices`, e.g. N=10 over 140 →
`[0,15,31,46,62,77,93,108,124,139]`). The basic `messages`/`reward`/`name` are still saved for
**every** game; only the fat `raw_trace` is limited to the samples.
- Why a count, not a boolean: `SAVE_RAW=1`-for-all made a 140-game thinking run's `results/` dir
  ~several hundred MB–1 GB (thinking traces dominate; a smoke game was 16–69 KB). N samples answer
  "did injected memory change behaviour?" without the bloat. `0`/unset = off.
- Clamps: N≥total → all; N=1 → `{0}`; N≥1 with total games → evenly spread incl. both endpoints.
- ALFWorld + WebShop batch runners (main loops strip `raw_trace` from non-sampled games at write
  time; collection itself is cheap/in-memory). Reasoning runner already saves full prompt+response
  in `messages` (single-turn, tiny) — unaffected.
- **No LLM calls added**; speed impact unmeasurable (eval is LLM-latency-bound). Startup prints
  `[SAVE_RAW] keeping raw_trace for N games: [...]`.
- Use when analyzing *why* the agent acted or whether injected memory changed behaviour;
  the default files can't answer that (thinking + injected context are stripped).

### Timing + periodic print (all 3 runners: alfworld/webshop/reasoning)
- `PROMPT_SHOW_EVERY=N`: every N steps, print the **probe** task's full PROMPT + response.
  Probe = FIRST ACTIVE task in the batch (shifts as tasks finish; batch runner can't pin one
  game like the async file's `game_idx 0`). Replaces the old "print every agent" (I/O cut).
- `PRINT_CHARS=N`: truncate printed **responses** to N chars (0 = full). **Prompts print in
  full** so injected task/context is verifiable — matches the async `_revise_react` file.
- Per-batch progress line appends `| elapsed H:MM:SS  X games/min  ETA H:MM:SS`; each runner
  prints a final `Total wall-clock`. Rate/ETA computed over games processed THIS run
  (resume-aware). Reasoning loop reports `probs/min`.

### Self-describing result folders (config dump + self-logging)
Every run writes into its `output_path` (the results dir):
- `run_config.json` — `dump_run_config()`: CLI args + RESOLVED hyperparams (executor + curation
  + print for dev/dev_async; executor + print only for the no-memory hyper runners) + tracked env
  vars (`_TRACKED_ENV_VARS`), API key masked.
- `run.log` — `start_self_logging()` tees stdout+stderr via a `_Tee` wrapper; makes the external
  `2>&1 | tee logs_debug_memory/...` redundant (harmless if kept). Append-mode (resume-safe);
  dumped AFTER the `--overwrite` `.json` sweep so it survives.
Applied to all 4 runners (dev, dev_async, hyper_async_step0bug_fix, hyper_async_revise_react).

### Verified (this dev cycle)
- PROMPT_STYLE transforms the REAL templates correctly for all 3 styles (AST-extracted from the
  file, executed, asserted): think keeps `<think>`, reason_tag → `<reason>` and no `<think>`,
  revise_react drops the mandate (clean `situation.\nOnce…`, no dangling space).
- Step-0 context template renders with no unfilled `{}` and task appears exactly once.
- `py_compile` clean for `run_unified_dev.py` and `reasoningbank_alfworld.py`.
- **Smoke-tested end-to-end** (jul-11, box1 `ip-10-0-240-113`, 3 games, reasoningbank, executor
  8002 / curator 8001): startup knobs resolved as set; `revise_react` prompt confirmed
  (plain reasoning, no `<think>` mandate); **step-0 context injection confirmed** (game 2's
  `[PROMPT task 0 step 0]` contained `## Past Relevant Memories`); executor still parsed
  `<action>` despite thinking-on. Log:
  `logs_debug_memory/rb_smoke_qwen3-8b_curator_0.6_nonthinking.log`.

### Companion file edited
`reasoningbank_alfworld.py`:
- `_llm` now reads `CURATION_*` env vars (HTTP + vLLM branches), defaults = prior behaviour
  (temp 0.7, max_tokens 1024, HTTP thinking = server default). Shared file, so non-dev callers
  are unaffected unless the env vars are set.
- **Fence-leak fix in `_extract_memory_items`** (pre-existing bug, surfaced by the smoke run):
  the curator wraps its output in a markdown ``` block (matching its own prompt's format
  example). The old naive `raw.split("# Memory Item")` produced a **phantom item[0] =
  `# Memory Item ``` `** and left trailing ``` fences inside every real item → **4 stored
  items instead of 3**, with ``` noise injected into future prompts. Fix: strip fence-only
  lines (`^\s*```[a-zA-Z]*\s*$`) and drop empty/backtick-only blocks before/after the split.
  `import re` moved to module top (was local). Verified on all 3 real records from the smoke
  run: OLD 4 items (fenced) → NEW 3 clean items, zero backticks. Advice text unchanged; BM25
  retrieval unaffected (keys on the task `query`, not `memory_items`). NOTE: prior runs' stored
  `reasoning_bank.jsonl` still contain the polluted items — rebuild memory (fresh `--exp_name`
  or delete the store) to benefit. The sibling `reasoningbank.py` (reasoning envs) has a similar
  naive split (`split("\n\n")`) — not yet fixed; check if you run reasoning-env reasoningbank.

---

## `run_unified_dev_async.py` — fast (group episode-async) memory runner

Goal: run ALFWorld faster than `run_unified_dev.py` (which is step-synchronous: per-step barrier
+ wait-for-slowest caps executor GPU util ~40–48%) while producing **equivalent results with any
memory method**. Copy of `run_unified_dev.py`; ONLY the ALFWorld path changed. WebShop
(batch-synchronous `WebshopMultiProcessEnv`) and reasoning (single-turn) paths are byte-identical
(verified by diff). Carries over all knobs: EXECUTOR_*/CURATION_*, PROMPT_STYLE, SAVE_RAW=N, timing.

### Execution model
- **GROUP episode-async** (modeled on `run_unified_hyper_concurrent.py`): games in a group of
  `--batch_size` run CONCURRENTLY to completion, each in its own thread on a private single-game
  env (`copy.deepcopy(TEMPLATE_ENV)` + `init_env(batch_size=1)`). The LLM call is OUTSIDE a global
  `PDDL_LOCK` (tatsu is not thread-safe) so episodes overlap on LLM latency. Barrier at the group
  boundary (ThreadPoolExecutor `with`-exit).
- **Concurrency is PINNED to batch_size** (= memory-update granularity). No `--concurrency` knob:
  raising it would change when memory updates fire → break exactness. More parallelism = raise
  `--batch_size` (coarsens memory updates identically to run_unified_dev.py at that batch_size).

### Why it's memory-EXACT vs run_unified_dev.py (by construction)
- **Order/grouping/idx are driven by the SAME batched `env.reset()` stream** the batch runner
  consumes. CRITICAL FINDING (verified on box): textworld's serving order is deterministic but is
  **NOT** `game_files` order — an early attempt pinning `game_files[k]` gave 0/12 order match and a
  different per-group game SET. Fix: keep the batched `env` as the order oracle, `env.reset()` per
  group exactly as the batch runner, and only parallelize each group's execution on per-game envs.
  So game order, per-group membership, `idx=idx*batch_size+i`, and task strings are identical.
- **Memory frozen per group:** retrieval is single-threaded BEFORE the group (same frozen snapshot
  for all games), update is once AFTER the group barrier, in group-position order — identical to
  `update_memory_after_batch`'s `enumerate(zip(batch_results, task_descriptions))` contract. No
  in-thread memory access (no lock needed).
- **"Exact" caveat:** executor samples at temp>0, so neither runner is bit-reproducible; what
  matches is the memory SEMANTICS + game order + update granularity/order. With temp=0 the
  end-to-end diff should be identical (see exactness_smoke.sh).

### Two envs in __main__
`env` = batched env (order oracle, `env.reset()` per group). `TEMPLATE_ENV` = same base pool;
each worker thread deepcopies it and pins one game. (`TEMPLATE_ENV = env` before `env.init_env`.)

### Tests (`tests/`)
- `test_async_orchestration.py` — pure-logic (no heavy deps): group chunking, out-of-order thread
  completion → in-order memory update, skills_context positional keying, resume skip, SAVE_RAW
  sampling. **PASS** (local + box).
- `probe_alfworld_order.py` — real-env: compares `game_files[k]` vs `env.reset()` order; this is
  what surfaced the ordering bug (DIFF) and confirmed each order is deterministic across runs.
- `exactness_smoke.sh` — end-to-end: runs batch + async over the same N games with temp=0 curator+
  executor, diffs every `idx_*.json` (name/reward/messages) and `reasoning_bank.jsonl`.

### Expected speedup
~2–3× at batch_size=10 thinking-on (removes per-step straggler wait + lets short episodes exit
early). Capped because concurrency=batch_size=10 under-feeds a DP=4 server (~2.5 req/replica) and
curation is still a serial burst per group. Raise batch_size (20/28) for more. MEASURE via the
`games/min` / `Total wall-clock` lines vs the step-sync run at the same batch_size.
- First real datapoint (6 games, bs3, thinking-off smoke): batch 4.1min/1.5 g/min vs async
  3.1min/1.9 g/min ≈ **1.3×** — but tiny sample; the per-step-barrier win grows with longer
  episodes (thinking-on, max_steps 30) and larger batch_size. Re-measure at bs10/28 thinking-on.

### EXACTNESS VERDICT (measured, not assumed)
The end-to-end temp-0 smoke showed per-step trajectory diffs — INVESTIGATED and resolved: the
batch runner run twice against ITSELF (temp0, top_k1) also diverges (idx0@msg23, idx1@msg19,
idx2@msg37), i.e. **vLLM continuous batching is nondeterministic** (FP reduction order depends on
dynamic batch composition). What IS exact and verified: game order, per-idx game identity, and the
memory-bank update sequence — the things that define "same memory semantics." So the async runner
is as reproducible as the batch runner; neither is bit-exact against the shared server. To get a
byte-identical A/B you'd need a deterministic/mock LLM (or `--enforce-eager` + single-stream vLLM),
which is out of scope for a speed change.

---

## Verification status (what's been proven)

| Claim | How verified | Result |
|---|---|---|
| step0 fix works (task in step-0 prompt) | `run_unified_step0_fixtest.py` probe, steps 0-2 | `task_in_prompt=True` all steps |
| step0 bug present in origin path | `run_unified_step0_bugtest.py` probe | `task_in_prompt=False` |
| async covers exactly 140 games once | mock-LLM full run, concurrency 32 | 140 files, 0 missing, 0 dup (see ASYNC_COVERAGE_VERIFICATION.md) |
| batch dup at bad batch_size | count of idx_*.json | batch 30 → 150 files, 10 dups; use divisors of 140 |
| game pinning correct under concurrency | pin probe idx {0,1,5,50,139} | game_idx i ↔ game_files[i] |
| step0 context injected on memory runs | 3-game reasoningbank smoke, `PROMPT_SHOW_EVERY=1` | game 2 step-0 prompt had `## Past Relevant Memories` |
| reasoningbank fence-leak fix | old vs new parser on 3 real smoke records | 4 fenced items → 3 clean; 0 backticks left |
| async orchestration matches batch | `tests/test_async_orchestration.py` (pure-logic) | PASS: update order/keying/resume/SAVE_RAW identical to batch, bs=1/10/20/28/35 |
| textworld order ≠ game_files order | `tests/probe_alfworld_order.py` (real env) | game_files[k] vs env.reset() = 0/12 match; each order deterministic across runs → async MUST drive from env.reset() |
| async == batch: game order + per-idx identity | `exactness_smoke.sh` 6 games bs3 | ✅ 6/6 name match; memory-bank task_id order identical |
| async == batch: per-step trajectory bytes | `exactness_smoke.sh` (temp0) | ❌ diverge — BUT proven to be SERVER nondeterminism, not async logic (next row) |
| vLLM is nondeterministic run-to-run | batch runner run TWICE, temp0/top_k1, no memory | ❌ batch disagrees with ITSELF (idx0@msg23, idx1@msg19, idx2@msg37) — continuous-batching FP non-associativity. So async-vs-batch trajectory diffs are expected & unavoidable; async is as exact as batch can be. |
| batch composition identical at bs=10 | `tests/probe_batch_order_bs10.py`, 140 games ×3 env builds | ✅ 0/140 (batch,pos)→game mismatches; env.reset() stream deterministic. Async uses SAME stream → same 10 games per batch, same positions, same cross-batch order. (End-to-end LLM diff only run at bs3 so far.) |
| config-dump + self-logging (4 runners) | REAL module import + `dump_run_config`/`start_self_logging` on box | ✅ all 4: correct schema (curation block present for dev/dev_async, absent for hyper), API key masked, tee writes `run.log`; all 9 files hash-identical local⇔/fsx |
| flash-attn used by vLLM | server log | `Using Flash Attention backend on V1 engine` |

---

## Related docs (cross-reference)
- `STEP0_TASK_BUG.md` — the step-0 bug in depth + fix + probe.
- `ASYNC_COVERAGE_VERIFICATION.md` — proof async runs 140 once, concurrency-independent.
- `BUG_FIX_PROGRESS.md` — both bugs (step0 + batch-dup) and overall progress.
- `SESSION_FINDINGS_AND_GOTCHAS.md` — thinking/prompt pairing, sampling defaults, speed,
  history_length 5-vs-3, MemP endpoints, infra (disk/flash-attn/env).

---

## Changelog
(Append newest at top. Format: date — file(s) — change — why — verification.)

- **jul-12 (config-dump + self-logging)** — `run_unified_dev.py`, `run_unified_dev_async.py`,
  `run_unified_hyper_async_step0bug_fix.py`, `run_unified_hyper_async_revise_react_step0bug_fix.py`
  — each run now writes into its results folder: (1) `run_config.json` = CLI args + RESOLVED
  hyperparams (executor + curation + print for the dev/dev_async memory runners; executor + print
  only for the two no-memory hyper runners) + tracked env vars, with the API key masked; (2)
  `run.log` = stdout+stderr tee'd in via a `_Tee` wrapper (`start_self_logging`), so the external
  `2>&1 | tee logs_debug_memory/...` is now redundant (harmless if kept). Config dump runs AFTER
  the `--overwrite` `.json` sweep so it isn't deleted; `run.log` is append-mode (preserves resume
  history). Why: make every result folder self-describing (no reconstructing the launch command).
  Verified on box (ip-10-0-140-50): all 9 files hash-identical local⇔/fsx; orchestration unit test
  PASS; all 4 runners py_compile; REAL module import + `dump_run_config`/`start_self_logging` call
  for each of the 4 → correct schema (curation block present for dev/dev_async, absent for hyper),
  API key masked, tee writes confirmed. NOTE: user asked to also capture an "external <?>" endpoint
  — clarification pending; `_TRACKED_ENV_VARS` currently covers OPENAI_API_BASE/API_BASE_URL/
  MODEL_NAME/GOOGLE_CLOUD_* — extend if the external service uses a different var.
- **jul-12** — `run_unified_dev_async.py` (NEW) + `tests/{test_async_orchestration.py,
  probe_alfworld_order.py,exactness_smoke.sh}` — group episode-async ALFWorld runner, memory-exact
  vs run_unified_dev.py. Why: step-sync per-step barrier caps GPU util / makes 140-game thinking
  runs slow. Design: drive grouping from the shared batched `env.reset()` stream (order oracle),
  run each group's games concurrently on per-game deepcopy'd single envs (LLM outside PDDL_LOCK),
  retrieve memory single-threaded pre-group + update post-barrier. KEY BUG CAUGHT during dev: first
  version pinned games by `sorted(game_files)[k]`, but the real env's `env.reset()` order is NOT
  game_files order (probe: 0/12 match) — would have played a different game sequence → wrong memory
  accumulation. Fixed by using env.reset() as the oracle. Verified: orchestration unit test PASS
  (box+local); ordering probe confirmed the bug+determinism; end-to-end exactness_smoke (temp0
  batch-vs-async diff) — see verification table. run_unified_dev.py NOT modified.
- **jul-11 (save-raw v2)** — `run_unified_dev.py` — `SAVE_RAW` changed from boolean to **count**
  `SAVE_RAW=N`: keep the fat `raw_trace` for only N sample games evenly spread across the run
  (basic messages/reward/name still saved for all). Why: `=1`-for-all made a 140-game thinking run
  ~hundreds of MB–1 GB; N=10 samples suffice to inspect behaviour. Verified: `_raw_save_indices`
  spread test (140/10→10 spread w/ endpoints, 0→none, ≥total→all, 1→{0}); main loops strip
  raw_trace from non-sampled games at write; py_compile clean.
- **jul-11 (save-raw)** — `run_unified_dev.py` — added `SAVE_RAW=1` env flag (persist full
  injected prompt + raw executor response per step into `idx_*.json` `raw_trace`; ALFWorld +
  WebShop). Confirmed live on the round-2 smoke (`raw_trace` w/ 5 steps, prompt had `## Past
  Relevant Memories`, response had `<think>`). SUPERSEDED same day by the count version above.
- **jul-11 (smoke + fix)** — `reasoningbank_alfworld.py` — fixed markdown fence leakage in
  `_extract_memory_items` (phantom `# Memory Item ``` ` item[0] + trailing ``` in every item →
  4 stored items instead of 3); `import re` moved to top. Found via a 3-game reasoningbank smoke
  on box1 that ALSO confirmed the dev runner's step-0 context injection and `revise_react` prompt
  work end-to-end. Why: polluted memory items were being injected into every future prompt.
  Verified: OLD vs NEW parser on all 3 real stored records → 4→3 clean items, 0 backticks.
  Not auto-migrated: existing `reasoning_bank.jsonl` stores keep old junk (rebuild to benefit);
  `reasoningbank.py` reasoning-env variant has a similar split, not yet fixed.
- **jul-11 (later)** — `run_unified_dev.py` (+ `reasoningbank_alfworld.py`) — grew from a
  step0-only copy into the **memory dev runner**: (1) EXECUTOR_* sampling knobs; (2) symmetric
  CURATION_* knobs across all skillos+reasoningbank curator paths (with `# Was hardcoded:`
  comments); (3) `PROMPT_STYLE` env (think/reason_tag/revise_react) unifying the 3 prompt
  variants in one file + collision warning; (4) step-0 context injection following
  `run_memp_ori.py` (new `ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT`; WebShop gate `>0`→`ctx`);
  (5) timing (elapsed/rate/ETA/wall-clock) + `PROMPT_SHOW_EVERY`/`PRINT_CHARS` probe printing
  (prompt full, response truncated) across alfworld/webshop/reasoning. Why: tunable, faithful
  (step-0 context like MemP), observable memory experiments from one file. Verified: AST-level
  PROMPT_STYLE transform on real templates (3 styles), step-0 template render (no unfilled
  fields, task once), py_compile clean; NOT run end-to-end — see the dedicated section above.
- **jul-11** — `DEV_HISTORY_RUNNER_LINEAGE.md` created — inventory of all `run_unified*` runners
  with lineage, decision guide, and verification table — for cross-section tracing.
- **jul-11** — `run_unified_dev.py` — created as batch source with step0 fix only (clean fixed
  batch path keeping run_unified.py pristine; diff vs origin = header + the one slice line).
  NOTE: superseded same day — see the "jul-11 (later)" entry; it has grown well beyond step0.
- **jul-10/11** — `*_step0bug_fix.py` (6 files: 3 batch nonthink, 3 async hyper) — step0 fix +
  EXECUTOR_* knobs + PROMPT_SHOW_EVERY + reduced/periodic printing (+timing for async) —
  correct, tunable, observable eval — probe + mock-coverage verified.
- **jul-10** — async runners (`run_unified_hyper_async*`) — rolling-pool engine — remove
  step-sync barrier for high GPU util — coverage + speed verified.
- **jul-10** — `base_config.yaml`, `run_memp_ori.py` — paths→$ALFWORLD_DATA, ALFWORLD_DATA
  setdefault — run on our boxes — only 2 existing-file edits vs origin.
