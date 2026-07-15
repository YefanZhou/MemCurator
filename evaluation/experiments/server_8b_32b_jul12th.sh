#!/bin/bash
# ==============================================================================
# NO-MEMORY baseline sweep, ENABLE_THINKING=false, across TWO models on one node:
#   :8001 -> Qwen3-8B  DP=4 (GPUs 0-3)   -> 8B lane
#   :8002 -> Qwen3-32B DP=4 (GPUs 4-7)   -> 32B lane
# Model is fixed per server, so this is two parallel lanes (8B on 8001, 32B on 8002),
# NOT a floating queue. Each lane runs its 24 jobs serially; the two lanes run concurrently.
#
# Matrix per model:  variant{PURE,RR} x temp{1.0,0.6} x hist{3,5} x seed{1,2,3} = 24
#   -> 48 runs total (24 per lane).
#
# NOTE (intentional, per user): PURE = run_unified_hyper_async_step0bug_fix.py mandates
# <think></think>. Under ENABLE_THINKING=false this COLLIDES (Qwen3 no-think prefill) and
# PURE runs will emit degenerate output. Kept on purpose to measure the collision; the RR
# (revise_react) variant is the no-think-safe reference.
#
# Every exp_name gets a _<STAMP> suffix so re-launches never overwrite prior results.
#
# Usage:
#   bash server_8b_32b_jul12th.sh --dry-run          # print all 48, run nothing
#   tmux new -s sweep32 'bash .../server_8b_32b_jul12th.sh'   # real run, detached
#   STAMP=jul12b bash .../server_8b_32b_jul12th.sh    # custom stamp / resume label
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug

SWEEP_LOG="logs_debug/_sweep32_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[sweep32] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN})  stamp=${STAMP}"

# ---- shared env (no-memory baseline). Per-run: OPENAI_API_BASE / EXECUTOR_TEMPERATURE /
#      HISTORY_LENGTH / --model / runner. ENABLE_THINKING=false for ALL. ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export ENABLE_THINKING=false
export PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15

PURE=run_unified_hyper_async_step0bug_fix.py           # <think> mandate  (collides under no-think)
RR=run_unified_hyper_async_revise_react_step0bug_fix.py # no mandate      (no-think safe)

# ---- preflight: both servers up + serving the EXPECTED model ----
if [ "$DRY_RUN" = 0 ]; then
  for p in 8001 8002; do
    curl -sf "http://localhost:${p}/v1/models" >/dev/null || {
      echo "ERROR: no vLLM server on :${p}"; exit 1; }
  done
  curl -s http://localhost:8001/v1/models | grep -q "Qwen3-8B"  || { echo "WARN: :8001 not serving Qwen3-8B?"; }
  curl -s http://localhost:8002/v1/models | grep -q "Qwen3-32B" || { echo "WARN: :8002 not serving Qwen3-32B?"; }
  echo "[preflight] 8001 + 8002 up"
fi

# run_exp: runner port model temp hist variant_tag  (exp_name built from these + STAMP)
run_exp() {
  local runner="$1" port="$2" model="$3" temp="$4" hist="$5" vtag="$6"
  local mtag; case "$model" in *32B) mtag=32b;; *) mtag=8b;; esac
  # seed index is encoded by the caller loop via $vtag already containing runN
  local exp="baseline_${vtag}_${mtag}_temp${temp}_hist${hist}_${STAMP}"

  if [ "$DRY_RUN" = 1 ]; then
    echo "  [:${port} ${mtag}] ${runner}  temp=${temp} hist=${hist}  -> ${exp}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  ${runner}"
  EXECUTOR_TEMPERATURE="$temp" HISTORY_LENGTH="$hist" \
  OPENAI_API_BASE="http://localhost:${port}/v1" \
  python -u "$runner" --env alfworld --memory_type none \
      --model "$model" --exp_name "$exp" --concurrency 64 \
      > "logs_debug/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)"
}

# One lane = one model on one port. 24 jobs: variant x temp x hist x seed.
run_lane() {
  local port="$1" model="$2"
  for variant in PURE RR; do
    local runner; [ "$variant" = PURE ] && runner="$PURE" || runner="$RR"
    local vlow; vlow=$(printf '%s' "$variant" | tr '[:upper:]' '[:lower:]')
    for temp in 1.0 0.6; do
      for hist in 3 5; do
        for seed in 1 2 3; do
          run_exp "$runner" "$port" "$model" "$temp" "$hist" "${vlow}_run${seed}"
        done
      done
    done
  done
  echo "[$(date +%H:%M:%S)] lane :${port} (${model}) COMPLETE"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN: 8B lane (:8001) ====="; run_lane 8001 openai/Qwen/Qwen3-8B
  echo "===== DRY RUN: 32B lane (:8002) ====="; run_lane 8002 openai/Qwen/Qwen3-32B
  echo "===== DRY RUN complete — nothing executed ====="
  exit 0
fi

echo "[$(date +%H:%M:%S)] launching 2 lanes: 8B->:8001, 32B->:8002 (24 runs each, 48 total)"
run_lane 8001 openai/Qwen/Qwen3-8B  & L1=$!
run_lane 8002 openai/Qwen/Qwen3-32B & L2=$!
wait "$L1" "$L2"
echo "[$(date +%H:%M:%S)] ALL 48 RUNS COMPLETE."
