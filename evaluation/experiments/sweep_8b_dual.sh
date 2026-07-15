#!/bin/bash
set -e
# Dual-server sweep: two Qwen3-8B DP=4 servers (ports 8001 and 8002) are kept busy by
# running TWO full 140-game runs at once — one pinned to each port. Each run still covers
# all 140 games on its own server (no per-run splitting, no runner code change); we just
# parallelize across runs so all 8 GPUs are used.
#
# Prereq: both servers up on this box:
#   :8001 -> Qwen3-8B DP=4 (GPUs 0-3),  :8002 -> Qwen3-8B DP=4 (GPUs 4-7)
#
# Run from anywhere; cd into agent_eval so relative paths resolve.
cd "$(dirname "$0")/../agent_eval"

COMMON="EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY TMPDIR=$HOME/tmp"

# run_job PORT SCRIPT TEMP EXP_NAME
run_job () {
    local port="$1" script="$2" temp="$3" name="$4"
    env $COMMON EXECUTOR_TEMPERATURE=$temp OPENAI_API_BASE=http://localhost:${port}/v1 \
        python -u "$script" --env alfworld --memory_type none \
            --model openai/Qwen/Qwen3-8B --exp_name "$name" --concurrency 64 \
            > "logs_debug/${name}.log" 2>&1
}

PURE=run_unified_hyper_async_step0bug_fix.py
RR=run_unified_hyper_async_revise_react_step0bug_fix.py
TAG=jul10_8b_dual

# 9 jobs total: (pure temp1.0)x3, (reviseReact temp1.0)x3, (reviseReact temp0.6)x3.
# Dispatch two at a time — one on :8001, one on :8002 — then wait for the pair.
JOBS=()
for i in 1 2 3; do JOBS+=("$PURE|1.0|baseline_pure_async_8b_${TAG}_temp1.0_0.95_20_4096_hist3_run${i}"); done
for i in 1 2 3; do JOBS+=("$PURE|0.6|baseline_pure_async_8b_${TAG}_temp0.6_0.95_20_4096_hist3_run${i}"); done
for i in 1 2 3; do JOBS+=("$RR|1.0|baseline_reviseReact_8b_${TAG}_temp1.0_0.95_20_4096_hist3_run${i}"); done
for i in 1 2 3; do JOBS+=("$RR|0.6|baseline_reviseReact_8b_${TAG}_temp0.6_0.95_20_4096_hist3_run${i}"); done

n=${#JOBS[@]}
idx=0
while [ $idx -lt $n ]; do
    # job on :8001
    IFS='|' read -r s t nm <<< "${JOBS[$idx]}"
    echo "[dispatch] :8001  $nm"
    run_job 8001 "$s" "$t" "$nm" &
    pid1=$!
    idx=$((idx+1))

    # job on :8002 (if any left)
    pid2=""
    if [ $idx -lt $n ]; then
        IFS='|' read -r s t nm <<< "${JOBS[$idx]}"
        echo "[dispatch] :8002  $nm"
        run_job 8002 "$s" "$t" "$nm" &
        pid2=$!
        idx=$((idx+1))
    fi

    # wait for this pair before starting the next (keeps at most 2 runs live)
    wait $pid1
    [ -n "$pid2" ] && wait $pid2
    echo "[pair done] $idx/$n jobs launched so far"
done
echo "ALL DONE: $n runs across ports 8001 + 8002"
