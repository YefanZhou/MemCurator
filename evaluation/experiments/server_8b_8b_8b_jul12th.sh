#!/bin/bash
# ==============================================================================
# Two-server (8001=GPU0-3, 8002=GPU4-7) parallel scheduler for the jul-12 sweep.
# 14 runs, all reasoningbank, Qwen3-8B, bs=10, revise_react, temp 1.0.
#
# LANES: two serial "lanes", one pinned to each server, run in parallel (2 runs
# in flight at all times). Balanced 7/7, each lane = 1 slow batch run + 6 async.
# The two slow batch runs (round2/round3, ~3h each) go FIRST so they overlap.
#
# GROUPS (unique exp_name per run so nothing overwrites; memory wiped per run):
#   A  run_unified_dev.py        think  hist5  -> round2, round3        (2, BATCH)
#   B  run_unified_dev_async.py  think  hist5  -> run1/2/3              (3, async)
#   C  run_unified_dev_async.py  think  hist3  -> run1/2/3              (3, async)
#   D  run_unified_dev_async.py  nothink hist5 -> run1/2/3              (3, async)
#   E  run_unified_dev_async.py  nothink hist3 -> run1/2/3              (3, async)
#
# NOTE: PROMPT_STYLE=revise_react is held constant across think/no-think by
# request (prompt text identical; only ENABLE_THINKING differs). For the think
# groups (A,B,C) this is the intentional "mismatched-but-constant" pairing.
#
# Usage:
#   bash server_8b_8b_8b_jul12th.sh --dry-run   # print the 14 commands, run nothing
#   tmux new -s sweep 'bash .../server_8b_8b_8b_jul12th.sh'   # real run, detached
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

# Top-level scheduler log: everything this script prints (START/DONE lines, preflight,
# lane progress) is tee'd here. Per-run stdout/stderr goes to logs_debug_memory/<exp>.log.
SWEEP_LOG="logs_debug_memory/_sweep_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN})"

# ---- shared env (per-run: OPENAI_API_BASE / HISTORY_LENGTH / ENABLE_THINKING) ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false
export PROMPT_STYLE=revise_react
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000

# Timestamp appended to every exp_name so re-launching this sweep never overwrites a prior
# one's results folders. Set ONCE here, so all 14 runs of THIS launch share the same stamp
# (they group together; distinct configs still differ by their base name). Override by
# exporting STAMP before running (e.g. STAMP=jul12b bash ...).
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[sweep] run stamp: ${STAMP}  (appended to every --exp_name)"

# ---- preflight: both servers must be up (skipped in dry-run) ----
if [ "$DRY_RUN" = 0 ]; then
  for p in 8001 8002; do
    if ! curl -sf "http://localhost:${p}/v1/models" >/dev/null; then
      echo "ERROR: no vLLM server responding on :${p} — start it before running this script."
      exit 1
    fi
  done
  echo "[preflight] servers on 8001 + 8002 OK"
fi

# ---- one run ----
# args: runner  port  thinking(true|false)  hist  exp_name
run_exp() {
  local runner="$1" port="$2" thinking="$3" hist="$4" exp="$5"
  exp="${exp}_${STAMP}"   # timestamp so re-launches don't overwrite prior results
  local memdir="Alfworld/memory/reasoningbank_${exp}"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (server :${port}) ----"
    echo "  rm -rf ${memdir}"
    echo "  OPENAI_API_BASE=http://localhost:${port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} \\"
    echo "  PROMPT_STYLE=${PROMPT_STYLE} EXEC(temp=${EXECUTOR_TEMPERATURE},top_p=${EXECUTOR_TOP_P},top_k=${EXECUTOR_TOP_K},max=${EXECUTOR_MAX_TOKENS}) \\"
    echo "  CUR(temp=${CURATION_TEMPERATURE},max=${CURATION_MAX_TOKENS},think=${CURATION_ENABLE_THINKING}) SAVE_RAW=${SAVE_RAW} \\"
    echo "  python -u ${runner} --env alfworld --memory_type reasoningbank \\"
    echo "      --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \\"
    echo "      --curation_base_url http://localhost:${port}/v1 \\"
    echo "      --batch_size 10 --retrieve_num 5 --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > logs_debug_memory/${exp}.log 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  runner=${runner}  think=${thinking}  hist=${hist}"
  # Fresh memory to match --overwrite (results are cleared; memory must be too,
  # else a re-run builds fresh results on stale accumulated memory).
  [ -n "$exp" ] && rm -rf "$memdir"
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  python -u "$runner" --env alfworld --memory_type reasoningbank \
      --model          openai/Qwen/Qwen3-8B \
      --curation_model openai/Qwen/Qwen3-8B \
      --curation_base_url "http://localhost:${port}/v1" \
      --batch_size 10 --retrieve_num 5 --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> logs_debug_memory/${exp}.log"
}

# ---- the 2 slow BATCH runs: both on 8001, run first (round2 then round3) ----
BATCH_JOBS_8001=(
  "run_unified_dev.py|true|5|rb-batch_think_hist5_round2"
  "run_unified_dev.py|true|5|rb-batch_think_hist5_round3"
)

# ---- the 12 async runs: a SHARED queue both servers pull from ----
# 8002 drains this from the start; 8001 joins after its 2 batch runs finish. Self-balancing:
# whichever server is free grabs the next async job (no fixed per-server assignment / timing guess).
ASYNC_JOBS=(
  "run_unified_dev_async.py|true|5|rb-async_think_hist5_run1"
  "run_unified_dev_async.py|true|5|rb-async_think_hist5_run2"
  "run_unified_dev_async.py|true|5|rb-async_think_hist5_run3"
  "run_unified_dev_async.py|true|3|rb-async_think_hist3_run1"
  "run_unified_dev_async.py|true|3|rb-async_think_hist3_run2"
  "run_unified_dev_async.py|true|3|rb-async_think_hist3_run3"
  "run_unified_dev_async.py|false|5|rb-async_nonthink_hist5_run1"
  "run_unified_dev_async.py|false|5|rb-async_nonthink_hist5_run2"
  "run_unified_dev_async.py|false|5|rb-async_nonthink_hist5_run3"
  "run_unified_dev_async.py|false|3|rb-async_nonthink_hist3_run1"
  "run_unified_dev_async.py|false|3|rb-async_nonthink_hist3_run2"
  "run_unified_dev_async.py|false|3|rb-async_nonthink_hist3_run3"
)

QUEUE="logs_debug_memory/_async_queue.txt"
QLOCK="logs_debug_memory/_async_queue.lock"

# Atomically pop the first job line from the shared queue (flock-guarded). Empty => queue drained.
pop_job() {
  local line=""
  exec 9>"$QLOCK"
  flock 9
  if [ -s "$QUEUE" ]; then
    line=$(head -n1 "$QUEUE")
    tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"
  fi
  flock -u 9
  printf '%s' "$line"
}

# A worker pinned to one port: keep popping async jobs until the queue is empty.
async_worker() {
  local port="$1" job
  while :; do
    job=$(pop_job)
    [ -z "$job" ] && break
    IFS='|' read -r runner thinking hist exp <<< "$job"
    run_exp "$runner" "$port" "$thinking" "$hist" "$exp"
  done
  echo "[$(date +%H:%M:%S)] async_worker :${port} — queue drained, exiting"
}

# 8001's full lane: 2 batch runs first, THEN join the shared async queue.
lane_8001() {
  local job
  IFS='|' read -r r t h e <<< "${BATCH_JOBS_8001[0]}"; run_exp "$r" 8001 "$t" "$h" "$e"
  IFS='|' read -r r t h e <<< "${BATCH_JOBS_8001[1]}"; run_exp "$r" 8001 "$t" "$h" "$e"
  async_worker 8001
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN ====================="
  echo "8001: 2 batch runs FIRST (serial), then joins the shared async queue:"
  for j in "${BATCH_JOBS_8001[@]}"; do IFS='|' read -r r t h e <<< "$j"; DRY_RUN=1 run_exp "$r" 8001 "$t" "$h" "$e"; done
  echo
  echo "SHARED ASYNC QUEUE (12 jobs) — pulled by 8002 immediately + 8001 after its batch runs:"
  echo "  (port assigned dynamically by whichever worker pops the job)"
  for j in "${ASYNC_JOBS[@]}"; do IFS='|' read -r r t h e <<< "$j"; DRY_RUN=1 run_exp "$r" 8002 "$t" "$h" "$e"; done
  echo "===================== DRY RUN complete — nothing executed ============="
  exit 0
fi

# Materialize the shared async queue.
printf '%s\n' "${ASYNC_JOBS[@]}" > "$QUEUE"
: > "$QLOCK"

echo "[$(date +%H:%M:%S)] launching: 8001=batch×2 then async-worker | 8002=async-worker (shared 12-job queue)"
lane_8001 & L1=$!
async_worker 8002 & L2=$!
wait "$L1" "$L2"
echo "[$(date +%H:%M:%S)] ALL 14 RUNS COMPLETE."
