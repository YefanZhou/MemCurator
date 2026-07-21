#!/bin/bash
# ==============================================================================
# EXTERNAL API MEMORY sweep — CURATOR_V1 method, GEMINI-ONLY (jul-16).
# Gemini-3.1-flash-lite via Vertex AI. (GPT variant: server_api_curator_v1_gpt_jul16th.sh.)
# Split out from server_api_curator_v1_gpt_gemini.sh so the Vertex (gemini) lane runs
# independently of the gateway (gpt) lane. Only the MODELS default differs.
#
# curator_v1 twin of server_api_curator_gpt_gemini.sh — SAME executor/curator hypers, but:
#   - memory_type = curator_v1 (run_unified_dev_async_curator_api.py + curator_alfworld_v1_api.py):
#       faithful-prompt curator, --curation_mode {success_only, success_and_fail}, reward-aware
#       store, full-prompt + BM25-provenance logging. THIS SWEEP: --curation_mode success_only
#       (override via CURATION_MODE=success_and_fail).
#   - hist = 3 ONLY (per request)
#   - curator-specific axes:
#       --task_context {short, obs0}   (obs0 enriches ONLY the curator CURRENT-task Question)
#       --curator_on_empty             (ON — curator writes a briefing even when nothing retrieved)
#   - exp-name prefix api-cur_v1_ so results never collide with the original-curator API sweep.
#
# For each model both executor and curator ARE that same API model:
#   openai/*  -> gateway (LLM_BACKEND=openai, CURATION_LLM_BACKEND=openai)
#   gemini/*  -> Vertex  (model prefix routes it; no gateway env needed)
# Per-model lanes run IN PARALLEL (different sources). Within a lane, tc x run run serially.
#
# HYPERS (identical to server_api_memory_gpt_gemini.sh):
#   revise_react, thinking executor, exec temp 1.0, EXECUTOR_MAX_TOKENS=4096,
#   curator temp 1.0, CURATION_MAX_TOKENS=4096, retrieve_num 5, seeds {1,2,3}.
#   (cur_think is inert for API models; enable_thinking is a Qwen chat-template flag.)
#
# RESUME: SKIP_DONE=1 (default) skips any model/tc/run with a 140-idx folder (stamp-agnostic).
# STAMP: defaults to current timestamp; pass STAMP=... only to force a fixed label.
#
# DEFAULT: gemini-3.1-flash-lite only, task_context=short, seeds {1,2,3}, rn=5, hist=3 -> 3 jobs.
# Parallelism is at the JOB level (each seed is its own process); JOB_PARALLEL caps how many run
# at once (default = all). Each job also fans out MAX_CONCURRENCY concurrent Vertex calls.
#
# Usage:
#   bash server_api_curator_v1_gemini_jul16th.sh --dry-run
#   tmux new -s api_curv1_gem 'bash .../server_api_curator_v1_gemini_jul16th.sh'  # 3 seeds
#   JOB_PARALLEL=2 bash ...                                       # cap to 2 concurrent seeds
#   NUM_GAMES=6 bash ...                                          # fast debug pass
#   CURATION_MODE=success_and_fail bash ...                       # store failures too
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Ensure conda `memory` env is active so `python` resolves under nohup/ssh.
if [ "$DRY_RUN" = 0 ] && ! command -v python >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; conda activate "${CONDA_ENV:-memory}"; }
  command -v python >/dev/null 2>&1 || { echo "ERROR: 'python' not found (activate the 'memory' env)"; exit 1; }
fi

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
SWEEP_LOG="logs_debug_memory/_sweep_api_curator_v1_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# ---- gateway config (openai/* models) ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
# ---- Vertex creds (gemini/* models); the _api runner also defaults these ----
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-salesforce-research-internal}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# ---- shared executor env (identical to the memory sweep) ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export SAVE_RAW="${SAVE_RAW:-140}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-15}" PRINT_CHARS="${PRINT_CHARS:-2000}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Pydantic serializer warnings:UserWarning}"
export CURATOR_LOG_CALLS="${CURATOR_LOG_CALLS:-1}"

BATCH_SIZE="${BATCH_SIZE:-10}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$BATCH_SIZE}"

# What to sweep. GEMINI-only default (Vertex); task_context=short, seeds 1/2/3, rn=5, hist=3.
MODELS="${MODELS:-gemini/gemini-3.1-flash-lite}"
RETRIEVE_NUM="${RETRIEVE_NUM:-5}"
NUM_GAMES="${NUM_GAMES:-0}"        # 0 => all 140
RUNS="${RUNS:-1 2 3}"
HIST="${HIST:-3}"                  # hist=3 ONLY (per request)
TC_LIST="${TC_LIST:-short}"        # curator task_context axis (short only)
ON_EMPTY="${ON_EMPTY:-1}"          # curator_on_empty ON (matches jul13th)
CURATION_MODE="${CURATION_MODE:-success_only}"   # curator_v1: success_only (default) | success_and_fail
SKIP_DONE="${SKIP_DONE:-1}"
# Parallelism: run this many (model,tc,run) jobs at once. Default = all jobs (max parallel).
# Each running job itself fires up to MAX_CONCURRENCY concurrent gateway calls, so total in-flight
# = JOB_PARALLEL * MAX_CONCURRENCY. Lower JOB_PARALLEL if the gateway rate-limits.
JOB_PARALLEL="${JOB_PARALLEL:-0}"  # 0 => auto (= number of jobs)

RUNNER=run_unified_dev_async_curator_api.py

# Already finished? (140-idx folder for THIS EXACT config, any stamp). The glob must pin every
# config axis in the exp-name (temp/hist/rn/tc/oe/mode/run) — otherwise a folder from a
# DIFFERENT config (e.g. hist3) would satisfy a hist5 job and wrongly skip it. Only the
# trailing STAMP is wildcarded.
already_done() {
  local model="$1" tc="$2" run="$3" d n
  local pat="dev_api-cur_v1_*_temp${EXECUTOR_TEMPERATURE}_hist${HIST}_rn${RETRIEVE_NUM}_tc${tc}_oe${ON_EMPTY}_${CURATION_MODE}_run${run}_*_few_shot_False_curator_v1"
  for d in Alfworld/results/${model}/${pat}; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/idx_*.json 2>/dev/null | wc -l)
    [ "$n" -ge 140 ] && return 0
  done
  return 1
}

# args: model task_context run
run_exp() {
  local model="$1" tc="$2" run="$3"
  local tag; tag=$(echo "$model" | tr '/.' '__')
  local exp="api-cur_v1_${tag}_temp${EXECUTOR_TEMPERATURE}_hist${HIST}_rn${RETRIEVE_NUM}_tc${tc}_oe${ON_EMPTY}_${CURATION_MODE}_run${run}_${STAMP}"

  if [ "${SKIP_DONE}" = 1 ] && already_done "$model" "$tc" "$run"; then
    echo "[$(date +%H:%M:%S)] SKIP  curator ${model} tc=${tc} run=${run} (already 140-idx)"
    return 0
  fi

  local exec_backend cur_backend
  if [[ "$model" == gemini/* ]]; then
    exec_backend="vllm"; cur_backend="vllm"        # gemini chosen by model prefix, not the flags
  else
    exec_backend="openai"; cur_backend="openai"
  fi
  local oe_flag=""; [ "$ON_EMPTY" = 1 ] && oe_flag="--curator_on_empty"

  if [ "$DRY_RUN" = 1 ]; then
    local route; [[ "$model" == gemini/* ]] && route="Vertex" || route="gateway"
    echo "  [${route}] curator exec=${model} curator=${model} hist=${HIST} rn=${RETRIEVE_NUM} tc=${tc} on_empty=${ON_EMPTY} mode=${CURATION_MODE} run=${run} cur(max=4096) -> ${exp}"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  (exec=${exec_backend} cur=${cur_backend})"
  env LLM_BACKEND="$exec_backend" CURATION_LLM_BACKEND="$cur_backend" \
      HISTORY_LENGTH="$HIST" CURATION_ENABLE_THINKING=true CURATION_MAX_TOKENS=4096 \
  python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode "$CURATION_MODE" \
      --model          "$model" \
      --curation_model "$model" \
      --task_context "$tc" $oe_flag \
      --batch_size "$BATCH_SIZE" --retrieve_num "$RETRIEVE_NUM" --max_steps 30 --num_games "$NUM_GAMES" \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp} (exit $?)  -> logs_debug_memory/${exp}.log"
}

# Build the flat job list of (model, tc, run) triples.
JOBS=()
for m in $MODELS; do for tc in $TC_LIST; do for r in $RUNS; do JOBS+=("${m}|${tc}|${r}"); done; done; done
NJOBS=${#JOBS[@]}
[ "$JOB_PARALLEL" -le 0 ] 2>/dev/null && JOB_PARALLEL=$NJOBS   # 0/auto => all jobs at once

echo "[api-curator] models='${MODELS}'  hist=${HIST}  tc_list='${TC_LIST}'  on_empty=${ON_EMPTY}  mode=${CURATION_MODE}  runs='${RUNS}'"
echo "              exec temp=${EXECUTOR_TEMPERATURE} think=${ENABLE_THINKING} style=${PROMPT_STYLE} rn=${RETRIEVE_NUM} num_games=${NUM_GAMES}"
echo "              curator max_tokens=4096  max_conc=${MAX_CONCURRENCY} skip_done=${SKIP_DONE} stamp=${STAMP} (dry_run=${DRY_RUN})"
echo "              ${NJOBS} jobs, ${JOB_PARALLEL} in parallel (each job also fans out MAX_CONCURRENCY gateway calls)."

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN (lanes shown sequentially; real run executes them in parallel) ====="
  n=0
  for m in $MODELS; do for tc in $TC_LIST; do for r in $RUNS; do run_exp "$m" "$tc" "$r"; n=$((n+1)); done; done; done
  echo "===== DRY RUN complete — ${n} run-slots (finished ones SKIP), nothing executed ====="
  exit 0
fi

# Preflight gateway models (skip gemini — Vertex).
for m in $MODELS; do
  if [[ "$m" != gemini/* ]]; then
    echo "[preflight] test_gateway_api.py ${m#openai/} ..."
    python test_gateway_api.py "${m#openai/}" || { echo "ERROR: gateway smoke test failed for ${m}"; exit 1; }
  fi
done

# Run JOBS with at most JOB_PARALLEL in flight at once (bounded pool).
running=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r m tc r <<< "$job"
  run_exp "$m" "$tc" "$r" &
  running=$((running+1))
  if [ "$running" -ge "$JOB_PARALLEL" ]; then
    wait -n 2>/dev/null || wait   # free a slot as soon as any job finishes (fallback: wait all)
    running=$((running-1))
  fi
done
wait
echo "[$(date +%H:%M:%S)] ALL ${NJOBS} RUNS COMPLETE."
