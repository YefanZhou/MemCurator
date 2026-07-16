#!/bin/bash
# ==============================================================================
# EXTERNAL API executors, memory=NONE baseline — GPT-5.5 (gateway) + Gemini-3.1-flash-lite (Vertex).
#
# Two executors reached via DIFFERENT backends (no memory => no curator):
#   gpt-5.5               -> Salesforce gateway  (LLM_BACKEND=openai, --model openai/gpt-5.5)
#   gemini-3.1-flash-lite -> Vertex AI           (--model gemini/gemini-3.1-flash-lite; gemini/
#                            prefix routes to llm_vertexai in the _api runner — no gateway env)
# The per-model env is set inside run_exp so the two backends don't cross-contaminate.
# The two models run in PARALLEL lanes (different sources -> no contention).
#
# Uses run_unified_dev_async_api.py (the _api copy) so the vLLM originals are untouched.
#
# RESUME: SKIP_DONE=1 (default) skips any model/hist/run that already has a 140-idx result folder
# (stamp-agnostic) — so re-launching resumes instead of redoing finished runs.
#
# STAMP: defaults to the current timestamp ($(date +%Y%m%d_%H%M)). Do NOT pass STAMP unless you
# want a fixed label; passing it (e.g. STAMP=jul15) is what replaces the auto-timestamp.
#
# DEBUGGABILITY: SAVE_RAW defaults high (=20) so a fat raw_trace (every step's prompt+response) is
# saved for many games; set a small NUM_GAMES for a fast debug pass (NUM_GAMES=0 => all 140).
#
# Usage:
#   bash server_api_none_gpt55_gemini25pro.sh --dry-run          # print the plan, run nothing
#   NUM_GAMES=6 bash server_api_none_gpt55_gemini25pro.sh        # tiny debug pass (6 games, full traces)
#   tmux new -s api_none 'bash .../server_api_none_gpt55_gemini25pro.sh'   # full run (parallel lanes)
#   MODELS="openai/gpt-5.5" bash ...                             # just one model
#   HISTS="3 5" RUNS="1 2 3" bash ...                            # override grid
#   SKIP_DONE=0 bash ...                                         # force re-run of finished configs
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Ensure the conda `memory` env is active so `python` resolves — needed when launched from a
# non-interactive shell (nohup/ssh) where `conda activate` alone doesn't put python on PATH.
# Skipped in dry-run (no python needed). Override CONDA_SH/CONDA_ENV if your paths differ.
if [ "$DRY_RUN" = 0 ] && ! command -v python >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  CONDA_ENV="${CONDA_ENV:-memory}"
  if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH" && conda activate "$CONDA_ENV"
  fi
  command -v python >/dev/null 2>&1 || { echo "ERROR: 'python' not found (activate the '${CONDA_ENV}' conda env first)"; exit 1; }
fi

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
SWEEP_LOG="logs_debug_memory/_sweep_api_none_gpt55_gem25_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# ---- gateway config (for the gpt-5.5 runs; gemini ignores these) ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
# X_API_KEY optional (gateway header); export before running if needed.

# ---- Vertex creds (for the gemini runs); the _api runner also defaults these internally ----
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-salesforce-research-internal}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# ---- sampling / env (shared) ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
# Debug knobs: keep a fat raw_trace for many games + print probe prompt/response every few steps.
export SAVE_RAW="${SAVE_RAW:-140}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-10}" PRINT_CHARS="${PRINT_CHARS:-3000}"
# Concurrency guard (<= batch_size — can't exceed it). "As large as possible" => batch_size.
# NOTE: gemini-3.1-flash-lite tolerated concurrency 4 cleanly; 10 (=batch) is untested and the
# heavier gemini-2.5-pro hung under high thread concurrency (gRPC), so watch the gemini lane.
BATCH_SIZE="${BATCH_SIZE:-10}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$BATCH_SIZE}"

# MODELS: space-separated executor models. openai/ -> gateway ; gemini/ -> Vertex.
# Each model gets its OWN parallel lane (gpt-5.5, gpt-5.4 both hit the gateway; gemini hits Vertex).
MODELS="${MODELS:-openai/gpt-5.5 openai/gpt-5.4 gemini/gemini-3.1-flash-lite}"
HISTS="${HISTS:-5 3}"
RUNS="${RUNS:-1 2 3}"
# NUM_GAMES: 0 => all 140. A small value (e.g. 6) is a fast debug pass with full raw traces.
NUM_GAMES="${NUM_GAMES:-0}"
# SKIP_DONE=1 (default): skip any model/hist/run that ALREADY has a 140-idx result folder
# (stamp-agnostic), so re-launches resume instead of redoing finished runs. Set 0 to force all.
SKIP_DONE="${SKIP_DONE:-1}"

# Is this model/hist/run already finished (140 idx files in ANY stamp's folder)? Stamp-agnostic
# so a re-launch skips completed runs regardless of the timestamp they were produced under.
already_done() {
  local model="$1" hist="$2" run="$3"
  local base="Alfworld/results/${model}"   # e.g. Alfworld/results/openai/gpt-5.5
  local d
  for d in ${base}/dev_api-none_*_hist${hist}_run${run}_*_few_shot_False_none; do
    [ -d "$d" ] || continue
    local n; n=$(ls "$d"/idx_*.json 2>/dev/null | wc -l)
    [ "$n" -ge 140 ] && return 0
  done
  return 1
}

run_exp() {
  local model="$1" hist="$2" run="$3"
  local tag; tag=$(echo "$model" | tr '/.' '__')          # openai/gpt-5.5 -> openai_gpt-5_5
  local exp="api-none_${tag}_temp${EXECUTOR_TEMPERATURE}_max${EXECUTOR_MAX_TOKENS}_hist${hist}_run${run}_${STAMP}"

  if [ "${SKIP_DONE}" = 1 ] && already_done "$model" "$hist" "$run"; then
    echo "[$(date +%H:%M:%S)] SKIP  ${model} hist=${hist} run=${run} (already has a 140-idx folder)"
    return 0
  fi

  # Per-model backend selection. gemini/ -> Vertex (unset LLM_BACKEND); else -> gateway (openai).
  local backend_env
  if [[ "$model" == gemini/* ]]; then
    backend_env="LLM_BACKEND=vllm"      # gemini path is chosen by the model prefix, not LLM_BACKEND
  else
    backend_env="LLM_BACKEND=openai"    # gateway (suppresses top_k/thinking, uses max_completion_tokens)
  fi

  if [ "$DRY_RUN" = 1 ]; then
    local route; [[ "$model" == gemini/* ]] && route="Vertex AI" || route="gateway (${OPENAI_API_BASE})"
    echo "  exec=${model} [${route}]  temp=${EXECUTOR_TEMPERATURE} max=${EXECUTOR_MAX_TOKENS} hist=${hist} run=${run} num_games=${NUM_GAMES} save_raw=${SAVE_RAW} -> ${exp}"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  (${backend_env})"
  env $backend_env HISTORY_LENGTH="$hist" \
  python -u run_unified_dev_async_api.py --env alfworld --memory_type none \
      --model "$model" \
      --batch_size "$BATCH_SIZE" --retrieve_num 5 --max_steps 30 --num_games "$NUM_GAMES" \
      --exp_name "${exp}" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp} (exit $?)  -> logs_debug_memory/${exp}.log"
}

# One serial lane per model: iterate its hist x run in order. Lanes run in PARALLEL across models
# (different sources — gpt-5.5=gateway, gemini=Vertex — so no contention).
model_lane() {
  local m="$1"
  for h in $HISTS; do for r in $RUNS; do run_exp "$m" "$h" "$r"; done; done
  echo "[$(date +%H:%M:%S)] lane ${m} — done"
}

echo "[api-none] models='${MODELS}'  hists='${HISTS}'  runs='${RUNS}'  num_games=${NUM_GAMES}  batch=${BATCH_SIZE}"
echo "           temp=${EXECUTOR_TEMPERATURE} max_tok=${EXECUTOR_MAX_TOKENS} save_raw=${SAVE_RAW} max_conc=${MAX_CONCURRENCY} skip_done=${SKIP_DONE} stamp=${STAMP} (dry_run=${DRY_RUN})"
echo "           per-model lanes run IN PARALLEL: openai/*=gateway ${OPENAI_API_BASE} ; gemini/*=Vertex project=${GOOGLE_CLOUD_PROJECT} loc=${GOOGLE_CLOUD_LOCATION}"

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN (lanes shown sequentially; real run executes them in parallel) ====="
  n=0
  for m in $MODELS; do for h in $HISTS; do for r in $RUNS; do run_exp "$m" "$h" "$r"; n=$((n+1)); done; done; done
  echo "===== DRY RUN complete — ${n} run-slots planned (finished ones will SKIP), nothing executed ====="
  exit 0
fi

# Preflight the gateway model(s) — EXECUTOR-ONLY (memory=none never curates, so skip the skillos
# tool-call probe). Skip gemini entirely (Vertex, no gateway smoke).
for m in $MODELS; do
  if [[ "$m" != gemini/* ]]; then
    echo "[preflight] test_gateway_api.py ${m#openai/} (executor-only) ..."
    EXECUTOR_ONLY=1 python test_gateway_api.py "${m#openai/}" || { echo "ERROR: gateway smoke test failed for ${m}"; exit 1; }
  fi
done

# Launch one lane per model in parallel; wait for all.
PIDS=()
for m in $MODELS; do model_lane "$m" & PIDS+=($!); done
wait "${PIDS[@]}"
echo "[$(date +%H:%M:%S)] ALL RUNS COMPLETE."
