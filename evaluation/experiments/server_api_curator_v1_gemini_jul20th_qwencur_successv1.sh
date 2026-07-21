#!/bin/bash
# ==============================================================================
# CROSS-BACKEND CURATOR_V1 sweep, jul-20 — GEMINI executor + LOCAL Qwen3-8B curator.
# Executor = gemini-3.1-flash-lite via Vertex AI; CURATOR = Qwen3-8B (thinking) served
# LOCALLY on :8001-:8004 (dp=2 each). --curation_mode success_only_v1. Curator temp axis
# CT_LIST=(1.0 0.6).
#
# Derived from server_api_curator_v1_gemini_jul16th_success_v1.sh, with the CURATOR
# swapped from the API model to a local Qwen3-8B endpoint:
#   - --curation_model openai/Qwen/Qwen3-8B  (was the API model)
#   - --curation_base_url http://localhost:<PORT>/v1  (round-robin over :8001-:8004)
#   - CURATION_LLM_BACKEND=vllm  (curator goes to local vLLM, NOT the gateway; so it uses
#     max_tokens + extra_body.chat_template_kwargs.enable_thinking — the Qwen thinking path)
#   - CURATION_ENABLE_THINKING=true  (curator is a THINKING Qwen)
#   - NEW axis CT_LIST=(1.0 0.6): curator sampling temperature, per job
# The EXECUTOR hits Vertex (gemini/ prefix routes it; no gateway needed for exec); executor and
# curator base_urls are independent in the runner, so this cross-backend split works.
#
# This node serves 4x Qwen3-8B DP=2 on :8001-:8004 (launched SEPARATELY, e.g.
# launch_vllm_4port_dp2.sh). This script does NOT launch/teardown vLLM.
#
# exp-name prefix api-cur_v1_qwencur_ (+ _ct{temp}) so results never collide with either the
# original API-curator sweep (api-cur_v1_, API curator) or the 8b/8b sweep (cur_v1_).
#
# DEFAULT: gemini-3.1-flash-lite executor, tc=short, seeds {1,2,3}, rn=5, hist{3,5}, CT{1.0,0.6}
#          => 1 model x 1 tc x 2 hist x 3 seeds x 2 ct = 12 jobs.
#
# Usage:
#   bash server_api_curator_v1_gemini_jul20th_qwencur_successv1.sh --dry-run
#   tmux new -s apiqc_gem 'bash .../server_api_curator_v1_gemini_jul20th_qwencur_successv1.sh'
#   CT_LIST="1.0" bash ...                                     # single curator temp
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
SWEEP_LOG="logs_debug_memory/_sweep_api_curator_v1_qwencur_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# ---- gateway config (kept in case MODELS includes an openai/* executor) ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
# ---- Vertex creds (EXECUTOR = gemini/* via Vertex); the _api runner also defaults these ----
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-salesforce-research-internal}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# ---- shared executor env ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export SAVE_RAW="${SAVE_RAW:-140}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-15}" PRINT_CHARS="${PRINT_CHARS:-2000}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Pydantic serializer warnings:UserWarning}"
export CURATOR_LOG_CALLS="${CURATOR_LOG_CALLS:-1}"
# ---- curator = LOCAL Qwen3-8B (thinking) — fixed knobs; CURATION_TEMPERATURE is per-job ----
export CURATION_TOP_P="${CURATION_TOP_P:-0.95}" CURATION_TOP_K="${CURATION_TOP_K:-20}"
export CURATION_MAX_TOKENS="${CURATION_MAX_TOKENS:-4096}" CURATION_ENABLE_THINKING="${CURATION_ENABLE_THINKING:-true}"

BATCH_SIZE="${BATCH_SIZE:-10}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$BATCH_SIZE}"

# What to sweep. Executor = gemini-3.1-flash-lite (Vertex); curator = local Qwen3-8B on ports below.
MODELS="${MODELS:-gemini/gemini-3.1-flash-lite}"
CURATION_MODEL="${CURATION_MODEL:-openai/Qwen/Qwen3-8B}"     # LOCAL Qwen curator
read -r -a PORTS <<< "${PORTS:-8001 8002 8003 8004}"          # local Qwen curator endpoints (round-robin)
RETRIEVE_NUM="${RETRIEVE_NUM:-5}"
NUM_GAMES="${NUM_GAMES:-0}"        # 0 => all 140
RUNS="${RUNS:-1 2 3}"
HIST_LIST="${HIST_LIST:-3 5}"
TC_LIST="${TC_LIST:-short}"
ON_EMPTY="${ON_EMPTY:-1}"
CURATION_MODE="${CURATION_MODE:-success_only_v1}"
read -r -a CT_LIST <<< "${CT_LIST:-1.0 0.6}"                  # curator temperature axis
SKIP_DONE="${SKIP_DONE:-1}"
JOB_PARALLEL="${JOB_PARALLEL:-0}"  # 0 => auto (= number of jobs)

RUNNER=run_unified_dev_async_curator_api.py

# Already finished? (140-idx folder for THIS EXACT config, any stamp). Pin every config axis
# in the exp-name (temp/hist/rn/tc/oe/mode/ct/run); only STAMP is wildcarded.
already_done() {
  local model="$1" tc="$2" run="$3" hist="$4" ct="$5" d n
  local tag; tag=$(echo "$model" | tr '/.' '__')
  local pat="dev_api-cur_v1_qwencur_${tag}_temp${EXECUTOR_TEMPERATURE}_hist${hist}_rn${RETRIEVE_NUM}_tc${tc}_oe${ON_EMPTY}_${CURATION_MODE}_ct${ct}_run${run}_*_few_shot_False_curator_v1"
  for d in Alfworld/results/${model}/${pat}; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/idx_*.json 2>/dev/null | wc -l)
    [ "$n" -ge 140 ] && return 0
  done
  return 1
}

# args: model task_context run hist cur_temp port
run_exp() {
  local model="$1" tc="$2" run="$3" hist="$4" ct="$5" port="$6"
  local tag; tag=$(echo "$model" | tr '/.' '__')
  local exp="api-cur_v1_qwencur_${tag}_temp${EXECUTOR_TEMPERATURE}_hist${hist}_rn${RETRIEVE_NUM}_tc${tc}_oe${ON_EMPTY}_${CURATION_MODE}_ct${ct}_run${run}_${STAMP}"

  if [ "${SKIP_DONE}" = 1 ] && already_done "$model" "$tc" "$run" "$hist" "$ct"; then
    echo "[$(date +%H:%M:%S)] SKIP  ${model} tc=${tc} hist=${hist} ct=${ct} run=${run} (already 140-idx)"
    return 0
  fi

  # executor backend: gpt -> gateway (openai); gemini -> Vertex (vllm sentinel). Curator ALWAYS
  # local vLLM (CURATION_LLM_BACKEND=vllm) via --curation_base_url to the given port.
  local exec_backend; [[ "$model" == gemini/* ]] && exec_backend="vllm" || exec_backend="openai"
  local oe_flag=""; [ "$ON_EMPTY" = 1 ] && oe_flag="--curator_on_empty"

  if [ "$DRY_RUN" = 1 ]; then
    echo "  exec=${model}(${exec_backend}) curator=${CURATION_MODEL}@:${port}(vllm) hist=${hist} rn=${RETRIEVE_NUM} tc=${tc} oe=${ON_EMPTY} mode=${CURATION_MODE} ct=${ct} run=${run} -> ${exp}"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  (exec=${exec_backend} curator=Qwen@:${port} ct=${ct})"
  env LLM_BACKEND="$exec_backend" CURATION_LLM_BACKEND="vllm" \
      HISTORY_LENGTH="$hist" CURATION_TEMPERATURE="$ct" \
  python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode "$CURATION_MODE" \
      --model          "$model" \
      --curation_model "$CURATION_MODEL" \
      --curation_base_url "http://localhost:${port}/v1" \
      --task_context "$tc" $oe_flag \
      --batch_size "$BATCH_SIZE" --retrieve_num "$RETRIEVE_NUM" --max_steps 30 --num_games "$NUM_GAMES" \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp} (exit $?)  -> logs_debug_memory/${exp}.log"
}

# Build the flat job list of (model, tc, run, hist, ct) tuples.
JOBS=()
for m in $MODELS; do for tc in $TC_LIST; do for h in $HIST_LIST; do for r in $RUNS; do for ct in "${CT_LIST[@]}"; do
  JOBS+=("${m}|${tc}|${r}|${h}|${ct}")
done; done; done; done; done
NJOBS=${#JOBS[@]}
[ "$JOB_PARALLEL" -le 0 ] 2>/dev/null && JOB_PARALLEL=$NJOBS

echo "[api-curator/qwencur] executors='${MODELS}'  curator='${CURATION_MODEL}'@'${PORTS[*]}'  hist_list='${HIST_LIST}'  ct_list='${CT_LIST[*]}'  mode=${CURATION_MODE}  runs='${RUNS}'"
echo "              ${NJOBS} jobs, ${JOB_PARALLEL} in parallel  stamp=${STAMP} (dry_run=${DRY_RUN})"

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN ====="
  n=0
  for m in $MODELS; do for tc in $TC_LIST; do for h in $HIST_LIST; do for r in $RUNS; do for ct in "${CT_LIST[@]}"; do
    run_exp "$m" "$tc" "$r" "$h" "$ct" "${PORTS[$(( n % ${#PORTS[@]} ))]}"; n=$((n+1))
  done; done; done; done; done
  echo "===== DRY RUN complete — ${n} run-slots, nothing executed ====="
  exit 0
fi

# Preflight the gateway executor(s) (skip gemini — Vertex) and the local curator ports.
for m in $MODELS; do
  if [[ "$m" != gemini/* ]]; then
    echo "[preflight] test_gateway_api.py ${m#openai/} ..."
    python test_gateway_api.py "${m#openai/}" || { echo "ERROR: gateway smoke test failed for ${m}"; exit 1; }
  fi
done
for port in "${PORTS[@]}"; do
  curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1 \
    && echo "[preflight] curator :${port} READY" \
    || { echo "ERROR: local Qwen curator :${port} not reachable"; exit 1; }
done

# Run JOBS with at most JOB_PARALLEL in flight; round-robin the curator port by job index.
# Track job PIDs and wait ONLY on them (a bare `wait` also waits on the tee proc-sub -> deadlock).
job_pids=()
running=0
idx=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r m tc r h ct <<< "$job"
  port="${PORTS[$(( idx % ${#PORTS[@]} ))]}"
  run_exp "$m" "$tc" "$r" "$h" "$ct" "$port" &
  job_pids+=("$!")
  idx=$((idx+1)); running=$((running+1))
  if [ "$running" -ge "$JOB_PARALLEL" ]; then
    wait -n "${job_pids[@]}" 2>/dev/null || wait -n 2>/dev/null
    running=$((running-1))
  fi
done
wait "${job_pids[@]}"
echo "[$(date +%H:%M:%S)] ALL ${NJOBS} RUNS COMPLETE."
