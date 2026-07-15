#!/bin/bash
# ==============================================================================
# The 6 NON-THINKING reasoningbank runs, offloaded to box1 (ip-10-0-240-113).
# These are the runs that were removed from box2's (140-50) async queue so they
# don't collide on shared /fsx. Box1's 8b_32b baseline sweep is finished; its
# 8001 (Qwen3-8B DP=4, GPUs 0-3) is free.
#
# All 6 are 8B memory runs (executor + curator both Qwen3-8B) -> they MUST use
# :8001. Box1's :8002 serves Qwen3-32B and cannot run them, so it stays idle here.
#
# Runs 2-at-a-time on :8001 (2 x batch_size 10 = 20 concurrent on DP=4 -> good util),
# ORDER: hist3 run1/2/3 FIRST, then hist5 run1/2/3 (per request).
#
# Env EXACTLY matches the box2 memory sweep (server_8b_8b_8b_jul12th.sh) so results
# are comparable: temp 1.0, revise_react, curator nonthinking max1024, SAVE_RAW=10,
# ENABLE_THINKING=false. exp_names REUSE the original names (rb-async_nonthink_*) so
# they slot into the same experiment set; a _<STAMP> suffix avoids clobbering anything.
#
# Usage:
#   bash server_8b_nonthink_box1.sh --dry-run
#   tmux new -s nonthink 'bash .../server_8b_nonthink_box1.sh'
#   STAMP=<box2_sweep_stamp> bash ...   # to match box2's stamp exactly (e.g. 20260712_0849)
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

SWEEP_LOG="logs_debug_memory/_sweep_nonthink_box1_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# Default stamp: matches the box2 sweep's stamp so these join the same set. Override with STAMP=.
STAMP="${STAMP:-20260712_0849}"
echo "[nonthink] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN})  stamp=${STAMP}"

# ---- env: identical to the box2 memory sweep, but ENABLE_THINKING=false ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false
export PROMPT_STYLE=revise_react
export ENABLE_THINKING=false
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000

PORTS=(8001 8002)   # both now serve Qwen3-8B on box1 (GPUs 0-3 and 4-7)

# ---- preflight: both 8B servers up ----
if [ "$DRY_RUN" = 0 ]; then
  for P in "${PORTS[@]}"; do
    curl -sf "http://localhost:${P}/v1/models" >/dev/null || { echo "ERROR: no server on :${P}"; exit 1; }
    curl -s "http://localhost:${P}/v1/models" | grep -q "Qwen3-8B" || { echo "ERROR: :${P} not serving Qwen3-8B"; exit 1; }
  done
  echo "[preflight] :8001 + :8002 serving Qwen3-8B OK"
fi

# hist first, then run seed. ORDER: hist3 (run1,2,3) then hist5 (run1,2,3).
JOBS=(
  "3|rb-async_nonthink_hist3_run1"
  "3|rb-async_nonthink_hist3_run2"
  "3|rb-async_nonthink_hist3_run3"
  "5|rb-async_nonthink_hist5_run1"
  "5|rb-async_nonthink_hist5_run2"
  "5|rb-async_nonthink_hist5_run3"
)

run_exp() {
  local port="$1" hist="$2" base="$3"
  local exp="${base}_${STAMP}"
  local memdir="Alfworld/memory/reasoningbank_${exp}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  [:${port}] hist=${hist} ENABLE_THINKING=false  -> ${exp}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  hist=${hist}"
  rm -rf "$memdir"   # fresh memory to match --overwrite
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" \
  python -u run_unified_dev_async.py --env alfworld --memory_type reasoningbank \
      --model          openai/Qwen/Qwen3-8B \
      --curation_model openai/Qwen/Qwen3-8B \
      --curation_base_url "http://localhost:${port}/v1" \
      --batch_size 10 --retrieve_num 5 --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN: 6 nonthink runs, 2-at-a-time (one per server 8001/8002), hist3 then hist5 ====="
  i=0; n=${#JOBS[@]}
  while [ $i -lt $n ]; do
    IFS='|' read -r h b <<< "${JOBS[$i]}"; run_exp 8001 "$h" "$b"; i=$((i+1))
    [ $i -lt $n ] && { IFS='|' read -r h b <<< "${JOBS[$i]}"; run_exp 8002 "$h" "$b"; i=$((i+1)); }
  done
  echo "===== DRY RUN complete — nothing executed ====="
  exit 0
fi

# Run 2 at a time — one run on :8001, one on :8002 (each a full DP=4 8B server).
echo "[$(date +%H:%M:%S)] launching 6 nonthink runs across :8001 + :8002, hist3->hist5"
i=0; n=${#JOBS[@]}
while [ $i -lt $n ]; do
  IFS='|' read -r h1 b1 <<< "${JOBS[$i]}"; run_exp 8001 "$h1" "$b1" & p1=$!; i=$((i+1))
  p2=""
  if [ $i -lt $n ]; then IFS='|' read -r h2 b2 <<< "${JOBS[$i]}"; run_exp 8002 "$h2" "$b2" & p2=$!; i=$((i+1)); fi
  wait "$p1"; [ -n "$p2" ] && wait "$p2"
  echo "[$(date +%H:%M:%S)] pair done ($i/$n launched)"
done
echo "[$(date +%H:%M:%S)] ALL 6 NONTHINK RUNS COMPLETE."
