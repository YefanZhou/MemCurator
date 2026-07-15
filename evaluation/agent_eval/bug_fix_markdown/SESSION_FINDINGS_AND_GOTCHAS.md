# Session Findings & Gotchas — ALFWorld Eval

Consolidated notes from the debugging/setup session. Companion to `BUG_FIX_PROGRESS.md`
(the two bugs), `STEP0_TASK_BUG.md`, and `ASYNC_COVERAGE_VERIFICATION.md`. This file
captures the *other* important, non-obvious things learned — the ones easy to trip over.

---

## 1. Thinking mode ↔ prompt must be matched (or you get garbage)

The single most error-prone interaction in this codebase.

- **Executor thinking is decided at the SERVER** (chat template), not by `run_unified.py`
  itself — the original executor sends no `enable_thinking`. Memory-curation paths *do* set
  `enable_thinking=False`. Author confirmed: **executor = non-thinking, memory generation = thinking.**
- Qwen3 no-think works by pre-filling an empty `<think>\n\n</think>` into the prompt.
- The ALFWorld prompt originally MANDATES reasoning inside `<think></think>`.
- **Collision:** no-think prefill + a prompt demanding `<think>` → model emits stray
  `</think></think>...`, often no `<action>` → degenerate/looping output, wasted steps.

### The rule (do not violate)
| Prompt variant | Reasoning instruction | Use with |
|---|---|---|
| plain (`run_unified*`, `*_hyper*`, `*_async*` without suffix) | `MUST ... <think></think>` | **thinking ON** |
| `_reason_tag` | `MUST ... <reason></reason>` | **no-think** (no collision) |
| `_revise_react` | (mandate removed) | **no-think** |

- `ENABLE_THINKING=true` + `<think>` prompt  → correct (what `sweep_32b.sh` uses).
- `ENABLE_THINKING=false` + `<think>` prompt → GARBAGE. Use `_reason_tag`/`_revise_react` instead.
- Empirical: `<reason>`/`<thought>` tags gave 0/10 leak; `<think>` under no-think gave 10/10 leak.

### Two ways to disable thinking (identical effect)
1. Custom server template `qwen3_nothink.jinja` (flips default) — but vLLM 0.8.5 has **no**
   `--chat-template-kwargs` launch flag, so this needs a template file.
2. Per-request `extra_body={"chat_template_kwargs":{"enable_thinking":False}}` — client-side,
   what the `nonthink*`/`hyper*` files use. **Requires the server on the STOCK template**
   (do NOT combine with qwen3_nothink.jinja, or you double-inject).

---

## 2. Sampling defaults — what actually gets used

`run_unified.py` sends only `temperature=0.7` (no top_p/top_k/max_tokens). vLLM fills the
rest from Qwen3's `generation_config.json`: **top_p=0.95, top_k=20** (and temp=0.6 default,
but the request's 0.7 wins). Confirmed by the server log line:
```
Using default chat sampling params from model: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}
WARNING ... Default sampling parameters have been overridden by the model's HF generation config ...
```
So effective executor sampling (run_unified.py) = **temp 0.7, top_p 0.95, top_k 20, uncapped tokens.**
- If launched with `--generation-config vllm`, the model config is ignored (top_p=1.0, top_k=-1).
  Our server was NOT launched that way, so the Qwen3 values apply.
- The `*_hyper*` and `*_step0bug_fix` files expose `EXECUTOR_TEMPERATURE/TOP_P/TOP_K/MAX_TOKENS`.
- Higher temperature (→1.0) generally HURTS ALFWorld accuracy (compounding wrong-action risk
  over ~15-30 steps, worse format adherence). ~0.6-0.7 is the model-recommended range.
  Temp 0 risks getting stuck in loops. Sweep temps if you want the real curve.

---

## 3. Speed: step-sync vs async (why GPU util looked "low")

- `run_unified.py` is STEP-synchronous: fire all N agents' step → wait for slowest → env-step
  together → repeat. Per-step barrier + inter-step queue drain caps GPU util (~40%). Not a bug.
- Raising `--batch_size` helps a little; it never saturates because of the barrier.
- **Rolling-pool async** (`*_hyper_async*`) removes the barrier → high util. Verified faster.
- **Concurrency starvation:** util = concurrent_requests / (replicas × per-replica capacity).
  With DP=8 and only ~30 requests, that's ~4/GPU → low util. Use `--concurrency 64-128` to
  feed 8 replicas. GPU % is NOT a correctness signal for a 140-game eval; wall-clock is.
- **Startup convoy:** each async game does `copy.deepcopy(TEMPLATE_ENV)` (~0.6-1s) under a
  global PDDL lock (tatsu parser is not thread-safe). At high concurrency the first ~40-60s
  looks "stuck" at 0% GPU — it's the serialized setup ramp, not a hang. Wait it out.

---

## 4. `history_length` — code default is 5, paper says 3

Every runner defaults `HISTORY_LENGTH = 5` (batch) or `int(os.environ.get("HISTORY_LENGTH","5"))`
(async). The paper reports **3**. The released code/checkpoints use 5. This is a real
code-vs-paper discrepancy — set `HISTORY_LENGTH=3` explicitly if matching the paper
(sweep_32b.sh does). The window is `history[-N:]`, re-indexed 1..N in the prompt (local, not
global step numbers) — verified correct.

---

## 5. Two `llm_api` endpoint families (MemP runners hang if you miss one)

`run_memp_ori.py` (and the MemP path) use TWO separate endpoint configs:
- **Executor** (litellm): `OPENAI_API_BASE` / `OPENAI_API_KEY` / `--model`.
- **Memory builder** (`llm_api.py`, openai SDK): `API_BASE_URL` / `API_KEY` / `MODEL_NAME`
  (note: raw name, NO `openai/` prefix).
If you set only the first set, memory-building hangs on `llm_api.py`'s retry loop at
"Building memory from trajectory". Set BOTH families. Embeddings use yet another pair
(`EMBEDDING_MODEL_BASE_URL/KEY`) — only hit if the retrieve/build path needs embeddings
(BM25 retrieval avoids it).

Also: `--use_memory` gates BOTH retrieval (needs non-empty store) and building
(`is_cold_start==False`). First run builds memory from scratch (empty at start); it's used on
later games. `is_cold_start=true` + `traj_file_path` pre-loads memory but then STOPS updating.

---

## 6. Environment / infra gotchas

- **flash-attn install** from a prebuilt wheel is correct (matched cu12/torch2.6/cp310/abiFALSE).
  Test `import flash_attn_2_cuda` FAILS if torch isn't imported first (`libc10.so`) — that's a
  bad test, not a broken install. The real test (`flash_attn_func` forward pass) passes, and
  vLLM logs `Using Flash Attention backend on V1 engine` across all replicas. It works.
- **Root disk `/` is 100% full** on the H200 box (system CUDA toolkits + container rootfs, NOT
  user data). `/tmp` lives on root → writes fail. Workaround: `export TMPDIR=$HOME/tmp`
  (on /fsx, 21T free). Real fix (sudo): remove unused CUDA toolkits. All user data is on /fsx.
- **Single `memory` conda env** is what the authors used (training + serving + eval). The
  eval `evaluation/requirements.txt` (langchain/chroma) vs training `requirements.txt` have
  pin conflicts (openai 1.75 vs 1.78, opentelemetry <1.27 for vllm vs newer for chromadb) —
  they coexist "by luck" (imports pass despite pip-check warnings). After installing eval reqs,
  re-pin `openai==1.78.1` + otel 1.26 so vllm still imports. See `installment_caveat.md`.
- **vLLM DP=8** = 8 full replicas (one per GPU), each loads the full model. Confirmed working
  on 8x H200. Startup ~4 min (weights + torch.compile + flashinfer JIT).

---

## 7. Running the sweeps / scripts

- **cwd matters:** `sweep_32b.sh` and the runners use relative paths (`run_*.py`, `logs_debug/`).
  Fixed sweep_32b.sh to `cd "$(dirname "$0")/../agent_eval"` so it runs from anywhere.
- **Use tmux** for long runs — foreground `| tee` dies on SSH drop (SSH dropped repeatedly this
  session). `tmux new -s sweep` then run; detach `Ctrl-b d`, reattach `tmux attach -t sweep`.
- **Buffering:** pipe to `tee`/`grep` buffers stdout → looks frozen. Use `python -u` and/or
  `grep --line-buffered`. "No output" ≠ "stuck" — check result-file count / GPU / process.
- **Resume:** async + batch runners skip existing `idx_*.json`. Re-run same `--exp_name`
  WITHOUT `--overwrite` to continue; WITH `--overwrite` to start fresh. A run that "does
  nothing fast" is usually resume skipping already-done games.
- **partition:** `srun --account sfr-rl --gres=gpu:8 --partition=ml.p5en.48xlarge --time=4-00:00:00 --pty bash`

---

## 8. Files reference (quick)
- Correctness-fixed set: `*_step0bug_fix.py` (batch nonthink×3, async hyper×3).
- Fastest: `run_unified_hyper_async_*_step0bug_fix.py` (rolling pool + timing + hyperparams + periodic prompt).
- Probes: `run_unified_step0_bugtest.py` (bug), `run_unified_step0_fixtest.py` (fix).
- **`run_unified.py` is UNMODIFIED from origin** — still has the step-0 bug; fix before memory runs.
- Only 2 existing files were edited vs origin: `Alfworld/base_config.yaml` (paths → `$ALFWORLD_DATA`),
  `run_memp_ori.py` (ALFWORLD_DATA setdefault). Everything else is new/additive.
