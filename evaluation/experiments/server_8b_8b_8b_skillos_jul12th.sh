#!/bin/bash
# ==============================================================================
# Two-server (8001=GPU0-3, 8002=GPU4-7) parallel scheduler for the jul-12 sweep.
# SKILLOS, Qwen3-8B executor + Qwen3-8B (untrained) curator, bs=10, revise_react,
# executor temp 1.0. Curator = thinking, temp 1.0, 4096 tokens.
#
# MATRIX: retrieve_num rn ∈ {3, 5} crossed with EVERY base config, so each
# (think × hist) cell is measured at both retrieve depths. All runs use the fast
# async runner (run_unified_dev_async.py).
#   base configs = 12 (2×2 think×hist, 3 runs each)  ×  RN_LIST {3,5}  ->  24 runs.
# Change RN_LIST (near the job matrix) to add/remove retrieve depths.
#
# This is the SkillOS twin of server_8b_8b_8b_jul12th.sh (the reasoningbank sweep):
# same harness, grid, and scheduling — the differences are --memory_type skillos,
# the sk-* exp-name prefix, the curator hypers, and the rn∈{3,5} cross, so nothing
# overwrites the reasoningbank results.
# Memory (skills.json) now lives INSIDE each result folder, so --overwrite wipes it
# along with the results (no separate rm needed).
#
# SCHEDULING: both servers (8001 + 8002) drain a single shared, flock-guarded job
# queue in parallel — whichever server is free grabs the next run (self-balancing).
#
# GROUPS (unique exp_name per run so nothing overwrites; _rn<N> suffix per depth):
#   B  think   hist5   -> run1/2/3   (3 × |RN| = 6)
#   C  think   hist3   -> run1/2/3   (3 × |RN| = 6)
#   D  nothink hist5   -> run1/2/3   (3 × |RN| = 6)
#   E  nothink hist3   -> run1/2/3   (3 × |RN| = 6)
#
# NOTE: PROMPT_STYLE=revise_react is held constant across think/no-think by
# request (prompt text identical; only ENABLE_THINKING differs). For the think
# groups (B,C) this is the intentional "mismatched-but-constant" pairing.
#
# Usage:
#   bash server_8b_8b_8b_skillos_jul12th.sh --dry-run    # print all commands, run nothing
#   RN_LIST override: edit the RN_LIST=(...) line to change the retrieve-num sweep
#   tmux new -s sweep 'bash .../server_8b_8b_8b_skillos_jul12th.sh'   # real run, detached
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

# Top-level scheduler log: everything this script prints (START/DONE lines, preflight,
# lane progress) is tee'd here. Per-run stdout/stderr goes to logs_debug_memory/<exp>.log.
SWEEP_LOG="logs_debug_memory/_sweep_skillos_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN})"

# ---- shared env (per-run: OPENAI_API_BASE / HISTORY_LENGTH / ENABLE_THINKING) ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
# Curator = thinking, temp 1.0, 4096 tokens (the SkillOS-intended curator setup).
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
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
# args: runner  port  thinking(true|false)  hist  rn(retrieve_num)  exp_name
run_exp() {
  local runner="$1" port="$2" thinking="$3" hist="$4" rn="$5" exp="$6"
  exp="${exp}_${STAMP}"   # timestamp so re-launches don't overwrite prior results

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (server :${port}) ----"
    echo "  OPENAI_API_BASE=http://localhost:${port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} \\"
    echo "  PROMPT_STYLE=${PROMPT_STYLE} EXEC(temp=${EXECUTOR_TEMPERATURE},top_p=${EXECUTOR_TOP_P},top_k=${EXECUTOR_TOP_K},max=${EXECUTOR_MAX_TOKENS}) \\"
    echo "  CUR(temp=${CURATION_TEMPERATURE},max=${CURATION_MAX_TOKENS},think=${CURATION_ENABLE_THINKING}) SAVE_RAW=${SAVE_RAW} \\"
    echo "  python -u ${runner} --env alfworld --memory_type skillos \\"
    echo "      --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \\"
    echo "      --curation_base_url http://localhost:${port}/v1 \\"
    echo "      --batch_size 10 --retrieve_num ${rn} --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > logs_debug_memory/${exp}.log 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  runner=${runner}  think=${thinking}  hist=${hist}  rn=${rn}"
  # --overwrite clears the result folder (which now also holds skills.json), so the
  # memory store is wiped along with the results — no separate rm needed.
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  python -u "$runner" --env alfworld --memory_type skillos \
      --model          openai/Qwen/Qwen3-8B \
      --curation_model openai/Qwen/Qwen3-8B \
      --curation_base_url "http://localhost:${port}/v1" \
      --batch_size 10 --retrieve_num "$rn" --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> logs_debug_memory/${exp}.log"
}

# ==============================================================================
# Job matrix: retrieve_num rn ∈ {3, 5} crossed with EVERY base config.
# Each base config below is expanded once per rn (exp name gets a _rn<N> suffix),
# so nothing collides and every (think × hist × runner) cell is measured at both
# retrieve depths. RN_LIST is the single knob controlling the rn sweep.
# Job spec fields (pipe-delimited): runner | thinking | hist | rn | exp_name
# ==============================================================================
RN_LIST=(3 5)

# ---- base configs (rn-agnostic; expanded over RN_LIST below) ----
# The 2×2 (think × hist) grid, 3 runs each (12 configs), all on the fast async runner.
ASYNC_BASE=(
  "run_unified_dev_async.py|true|5|sk-async_think_hist5_run1"
  "run_unified_dev_async.py|true|5|sk-async_think_hist5_run2"
  "run_unified_dev_async.py|true|5|sk-async_think_hist5_run3"
  "run_unified_dev_async.py|true|3|sk-async_think_hist3_run1"
  "run_unified_dev_async.py|true|3|sk-async_think_hist3_run2"
  "run_unified_dev_async.py|true|3|sk-async_think_hist3_run3"
  "run_unified_dev_async.py|false|5|sk-async_nonthink_hist5_run1"
  "run_unified_dev_async.py|false|5|sk-async_nonthink_hist5_run2"
  "run_unified_dev_async.py|false|5|sk-async_nonthink_hist5_run3"
  "run_unified_dev_async.py|false|3|sk-async_nonthink_hist3_run1"
  "run_unified_dev_async.py|false|3|sk-async_nonthink_hist3_run2"
  "run_unified_dev_async.py|false|3|sk-async_nonthink_hist3_run3"
)

# ---- expand base configs over RN_LIST into full job specs (adds rn field + _rn<N> suffix) ----
ASYNC_JOBS=()
for base in "${ASYNC_BASE[@]}"; do
  IFS='|' read -r r t h e <<< "$base"
  for rn in "${RN_LIST[@]}"; do ASYNC_JOBS+=("${r}|${t}|${h}|${rn}|${e}_rn${rn}"); done
done

QUEUE="logs_debug_memory/_async_queue_skillos.txt"
QLOCK="logs_debug_memory/_async_queue_skillos.lock"

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
    IFS='|' read -r runner thinking hist rn exp <<< "$job"
    run_exp "$runner" "$port" "$thinking" "$hist" "$rn" "$exp"
  done
  echo "[$(date +%H:%M:%S)] async_worker :${port} — queue drained, exiting"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN ====================="
  echo "SHARED ASYNC QUEUE (${#ASYNC_JOBS[@]} jobs) — both servers (8001 + 8002) drain it in parallel:"
  echo "  (port assigned dynamically by whichever worker pops the job; each base config × rn in {${RN_LIST[*]}})"
  for j in "${ASYNC_JOBS[@]}"; do IFS='|' read -r r t h rn e <<< "$j"; DRY_RUN=1 run_exp "$r" 8002 "$t" "$h" "$rn" "$e"; done
  echo "===================== DRY RUN complete — nothing executed ============="
  exit 0
fi

# Materialize the shared async queue.
printf '%s\n' "${ASYNC_JOBS[@]}" > "$QUEUE"
: > "$QLOCK"

echo "[$(date +%H:%M:%S)] launching: 8001 + 8002 both draining the shared ${#ASYNC_JOBS[@]}-job async queue in parallel"
async_worker 8001 & L1=$!
async_worker 8002 & L2=$!
wait "$L1" "$L2"
echo "[$(date +%H:%M:%S)] ALL ${#ASYNC_JOBS[@]} RUNS COMPLETE."
