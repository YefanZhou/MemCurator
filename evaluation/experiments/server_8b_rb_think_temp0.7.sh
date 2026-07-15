#!/bin/bash
# ==============================================================================
# reasoningbank THINKING sweep at TEMPERATURE 0.7 — the temp=0.7 counterpart of the
# two temp=1.0 configs that scored best on ALFWorld:
#     rb  thinking  wo<think>(revise_react)  hist3  temp1.0 -> 56.90 ± 3.21
#     rb  thinking  wo<think>(revise_react)  hist5  temp1.0 -> 64.05 ± 0.89
# This sweep re-runs BOTH at temp 0.7 to see if lowering executor temperature helps.
#
# 6 runs total: reasoningbank, Qwen3-8B exec + Qwen3-8B curator, revise_react,
# ENABLE_THINKING=true, EXECUTOR_TEMPERATURE=0.7, hist{3,5} x seed{1,2,3}.
# All async (run_unified_dev_async.py). Two 8B servers (:8001,:8002) pull from a
# shared flock-guarded queue (self-balancing), 2 runs in flight.
#
# CURATION_TEMPERATURE stays 1.0 (matches the temp=1.0 runs' curator; only the
# EXECUTOR temperature is the variable under test). Override with CUR_TEMP=... if
# you want to lower the curator too.
#
# Usage:
#   bash server_8b_rb_think_temp0.7.sh --dry-run
#   tmux new -s rb07 'bash .../server_8b_rb_think_temp0.7.sh'
#   STAMP=mylabel bash ...        # override the exp_name timestamp suffix
# Requires two Qwen3-8B vLLM servers on :8001 and :8002 (see the slurm script).
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

SWEEP_LOG="logs_debug_memory/_sweep_rb_think_temp0.7_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN})"

# ---- shared env (per-run overrides: OPENAI_API_BASE / HISTORY_LENGTH) ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE="${EXEC_TEMP:-0.7}" EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE="${CUR_TEMP:-1.0}" CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false
export PROMPT_STYLE=revise_react
export ENABLE_THINKING=true
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[sweep] run stamp: ${STAMP}  exec_temp=${EXECUTOR_TEMPERATURE}  cur_temp=${CURATION_TEMPERATURE}"

# ---- preflight: both servers up (skipped in dry-run) ----
if [ "$DRY_RUN" = 0 ]; then
  for p in 8001 8002; do
    curl -sf "http://localhost:${p}/v1/models" >/dev/null || { echo "ERROR: no vLLM server on :${p}"; exit 1; }
  done
  echo "[preflight] servers on 8001 + 8002 OK"
fi

# args: port hist exp_name
run_exp() {
  local port="$1" hist="$2" exp="$3"
  exp="${exp}_${STAMP}"
  local memdir="Alfworld/memory/reasoningbank_${exp}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [:${port}] hist=${hist} exec_temp=${EXECUTOR_TEMPERATURE} -> ${exp}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  hist=${hist}"
  rm -rf "$memdir"
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" \
  python -u run_unified_dev_async.py --env alfworld --memory_type reasoningbank \
      --model          openai/Qwen/Qwen3-8B \
      --curation_model openai/Qwen/Qwen3-8B \
      --curation_base_url "http://localhost:${port}/v1" \
      --batch_size 10 --retrieve_num 5 --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> logs_debug_memory/${exp}.log"
}

# 6 async jobs: hist{3,5} x seed{1,2,3}. hist3 first (cheaper), then hist5.
JOBS=(
  "3|rb-async_think_temp0.7_hist3_run1"
  "3|rb-async_think_temp0.7_hist3_run2"
  "3|rb-async_think_temp0.7_hist3_run3"
  "5|rb-async_think_temp0.7_hist5_run1"
  "5|rb-async_think_temp0.7_hist5_run2"
  "5|rb-async_think_temp0.7_hist5_run3"
)

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN: ${#JOBS[@]} runs (shared queue over :8001,:8002) ====="
  for j in "${JOBS[@]}"; do IFS='|' read -r h e <<< "$j"; run_exp 8001 "$h" "$e"; done
  echo "===================== DRY RUN complete — nothing executed ====================="
  exit 0
fi

# ---- shared flock-guarded queue, one worker pinned per server ----
QUEUE="logs_debug_memory/_rb07_queue.txt"
QLOCK="logs_debug_memory/_rb07_queue.lock"
printf '%s\n' "${JOBS[@]}" > "$QUEUE"; : > "$QLOCK"

pop_job() {
  local line=""
  exec 9>"$QLOCK"; flock 9
  if [ -s "$QUEUE" ]; then
    line=$(head -n1 "$QUEUE"); tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
  fi
  flock -u 9
  printf '%s' "$line"
}

worker() {
  local port="$1" job
  while :; do
    job=$(pop_job); [ -z "$job" ] && break
    IFS='|' read -r hist exp <<< "$job"
    run_exp "$port" "$hist" "$exp"
  done
  echo "[$(date +%H:%M:%S)] worker :${port} — queue drained"
}

echo "[$(date +%H:%M:%S)] launching ${#JOBS[@]} runs: worker :8001 + worker :8002 (shared queue)"
worker 8001 & W1=$!
worker 8002 & W2=$!
wait "$W1" "$W2"
echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS COMPLETE."
