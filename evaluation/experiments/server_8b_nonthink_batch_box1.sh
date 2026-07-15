#!/bin/bash
# ==============================================================================
# BATCH (step-synchronous) MEMORY-vs-NONE ablation, NON-THINKING.
# 24 runs: memory{reasoningbank,none} x temp{0.7,1.0} x hist{3,5} x seed{1,2,3}, run with run_unified_dev.py
# (step-sync batch runner) instead of run_unified_dev_async.py. Lets you A/B the
# batch vs async engines on identical configs (they should agree up to vLLM
# nondeterminism; memory semantics are identical by design).
#
# Box1 (ip-10-0-240-113): both :8001 and :8002 serve Qwen3-8B (DP=4 each).
# All 6 are 8B memory runs (executor + curator both Qwen3-8B).
# Runs 2-at-a-time, one per server, ORDER hist3 run1/2/3 then hist5 run1/2/3.
#
# batch_size=10 DIVIDES 140 -> no duplicate-games bug (the step-sync runner
# over-counts only when batch_size does not divide 140; 10 is safe).
#
# exp_names use rb-BATCH_nonthink_* (distinct from the async rb-async_nonthink_*),
# plus a _<STAMP> suffix, so nothing collides on shared /fsx.
#
# Env identical to the async nonthink sweep: temp 1.0, revise_react, curator
# nonthinking max1024, SAVE_RAW=10, ENABLE_THINKING=false.
#
# Usage:
#   bash server_8b_nonthink_batch_box1.sh --dry-run
#   tmux new -s nonthink_batch 'bash .../server_8b_nonthink_batch_box1.sh'
#   STAMP=<label> bash ...
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

SWEEP_LOG="logs_debug_memory/_sweep_nonthink_batch_box1_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[nonthink-batch] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN})  stamp=${STAMP}"

# ---- env: identical to the async nonthink sweep, ENABLE_THINKING=false ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false
export PROMPT_STYLE=revise_react
export ENABLE_THINKING=false
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000

PORTS=(8001 8002)   # both serve Qwen3-8B on box1

# ---- preflight: both 8B servers up ----
if [ "$DRY_RUN" = 0 ]; then
  for P in "${PORTS[@]}"; do
    curl -sf "http://localhost:${P}/v1/models" >/dev/null || { echo "ERROR: no server on :${P}"; exit 1; }
    curl -s "http://localhost:${P}/v1/models" | grep -q "Qwen3-8B" || { echo "ERROR: :${P} not serving Qwen3-8B"; exit 1; }
  done
  echo "[preflight] :8001 + :8002 serving Qwen3-8B OK"
fi

# Matrix: memory{reasoningbank,none} x temp{0.7,1.0} x hist{3,5} x seed{1,2,3} = 24 runs.
# Fields: mem|temp|hist|seed. exp_name is built in run_exp:
#   reasoningbank -> rb-batch_nonthink_temp<T>_hist<H>_run<S>
#   none          -> none-batch_nonthink_temp<T>_hist<H>_run<S>
# Result folders differ by suffix (_reasoningbank vs _none), so they never collide.
JOBS=()
for mem in reasoningbank none; do
  for temp in 0.7 1.0; do
    for hist in 3 5; do
      for seed in 1 2 3; do
        JOBS+=("${mem}|${temp}|${hist}|${seed}")
      done
    done
  done
done

run_exp() {
  local port="$1" mem="$2" temp="$3" hist="$4" seed="$5"
  local pfx; [ "$mem" = none ] && pfx=none || pfx=rb
  local exp="${pfx}-batch_nonthink_temp${temp}_hist${hist}_run${seed}_${STAMP}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [:${port}] mem=${mem} temp=${temp} hist=${hist} run=${seed}  -> ${exp}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  mem=${mem} temp=${temp} hist=${hist}"
  # reasoningbank: wipe its memory store to match --overwrite. none: no memory store.
  [ "$mem" = reasoningbank ] && rm -rf "Alfworld/memory/reasoningbank_${exp}"

  if [ "$mem" = none ]; then
    OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" \
    EXECUTOR_TEMPERATURE="$temp" \
    python -u run_unified_dev.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B \
        --batch_size 10 --retrieve_num 5 --max_steps 30 \
        --exp_name "$exp" --overwrite \
        > "logs_debug_memory/${exp}.log" 2>&1
  else
    OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" \
    EXECUTOR_TEMPERATURE="$temp" CURATION_TEMPERATURE="$temp" \
    python -u run_unified_dev.py --env alfworld --memory_type reasoningbank \
        --model          openai/Qwen/Qwen3-8B \
        --curation_model openai/Qwen/Qwen3-8B \
        --curation_base_url "http://localhost:${port}/v1" \
        --batch_size 10 --retrieve_num 5 --max_steps 30 \
        --exp_name "$exp" --overwrite \
        > "logs_debug_memory/${exp}.log" 2>&1
  fi
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN: ${#JOBS[@]} nonthink BATCH runs (mem{rb,none} x temp{0.7,1.0} x hist{3,5} x seed{1,2,3}), 2-at-a-time ====="
  i=0; n=${#JOBS[@]}
  while [ $i -lt $n ]; do
    IFS='|' read -r m t h s <<< "${JOBS[$i]}"; run_exp 8001 "$m" "$t" "$h" "$s"; i=$((i+1))
    [ $i -lt $n ] && { IFS='|' read -r m t h s <<< "${JOBS[$i]}"; run_exp 8002 "$m" "$t" "$h" "$s"; i=$((i+1)); }
  done
  echo "===== DRY RUN complete — nothing executed ====="
  exit 0
fi

# Run 2 at a time — one run on :8001, one on :8002.
echo "[$(date +%H:%M:%S)] launching ${#JOBS[@]} nonthink BATCH runs across :8001 + :8002"
i=0; n=${#JOBS[@]}
while [ $i -lt $n ]; do
  IFS='|' read -r m1 t1 h1 s1 <<< "${JOBS[$i]}"; run_exp 8001 "$m1" "$t1" "$h1" "$s1" & p1=$!; i=$((i+1))
  p2=""
  if [ $i -lt $n ]; then IFS='|' read -r m2 t2 h2 s2 <<< "${JOBS[$i]}"; run_exp 8002 "$m2" "$t2" "$h2" "$s2" & p2=$!; i=$((i+1)); fi
  wait "$p1"; [ -n "$p2" ] && wait "$p2"
  echo "[$(date +%H:%M:%S)] pair done ($i/$n launched)"
done
echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} NONTHINK BATCH RUNS COMPLETE."
