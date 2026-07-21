#!/bin/bash
# ==============================================================================
# SkillOS retrieval-BUGFIX validation run (Option B: task<->task BM25 alignment,
# with the key FROZEN at skill creation).
#
# Config: think / hist3 / rn5, executor Qwen3-8B (:8001) + curator Qwen3-8B (:8002),
#         full 140 games, 3 seeds. SAVE_RAW=140 (save ALL raw traces for analysis).
#
# WHAT THE FIX DOES (in SkillOS/skills_memory.py + run_unified_dev_async.py):
#   - each skill records its CREATION task in skill['tasks'] (frozen, never grows on update)
#   - BM25 now indexes that task-key (task<->task, like ReasoningBank/MemCurator),
#     instead of the old title+full-body index.
#
# COMPARE the resulting SR to:
#   - pre-fix SkillOS think/hist3   ~= 40.5%
#   - no-memory baseline (think)    ~= (see analysis; nonthink hist3 no-mem ~34.5%)
#   - ReasoningBank think/hist3     ~= 56.9%
#
# Prereq: Qwen3-8B vLLM servers on :8001 (executor) and :8002 (curator).
# Usage:
#   bash server_8b_8b_jul14th_skillos_bugfix.sh --dry-run   # print the 3 commands
#   tmux new -s skfix 'bash server_8b_8b_jul14th_skillos_bugfix.sh'
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

# ---- config (think / hist3 / rn5) ----
EXEC_PORT="${EXEC_PORT:-8001}"
CUR_PORT="${CUR_PORT:-8002}"
MODEL="openai/Qwen/Qwen3-8B"
BATCH_SIZE="${BATCH_SIZE:-10}"       # concurrency == batch_size for this runner
RETRIEVE_NUM="${RETRIEVE_NUM:-5}"
HIST="${HIST:-3}"
THINKING="${THINKING:-true}"         # think executor
RUNNER=run_unified_dev_async.py

# Timestamp so re-launches never overwrite prior results.
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"

# ---- shared env ----
export OPENAI_API_KEY=EMPTY
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react
export SAVE_RAW=140 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000   # SAVE_RAW=140 => keep raw_trace for ALL games
# Silence the benign litellm<->pydantic serializer UserWarning (keeps real errors visible).
export PYTHONWARNINGS="ignore:Pydantic serializer warnings:UserWarning"

# ---- preflight: both servers up (skipped in dry-run) ----
if [ "$DRY_RUN" = 0 ]; then
  for p in "$EXEC_PORT" "$CUR_PORT"; do
    if ! curl -sf "http://localhost:${p}/v1/models" >/dev/null; then
      echo "ERROR: no vLLM server on :${p}. Start Qwen3-8B there first (or override EXEC_PORT/CUR_PORT)."
      exit 1
    fi
  done
  echo "[preflight] servers on :${EXEC_PORT} (exec) + :${CUR_PORT} (curator) OK"
fi

# ---- one run ----
run_seed() {
  local run_id="$1"
  local exp="sk-bugfix_think_hist${HIST}_rn${RETRIEVE_NUM}_run${run_id}_${STAMP}"
  local log="logs_debug_memory/${exp}.log"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp} ----"
    echo "  OPENAI_API_BASE=http://localhost:${EXEC_PORT}/v1 HISTORY_LENGTH=${HIST} ENABLE_THINKING=${THINKING} \\"
    echo "  PROMPT_STYLE=${PROMPT_STYLE} SAVE_RAW=${SAVE_RAW} EXEC(temp=${EXECUTOR_TEMPERATURE}) CUR(temp=${CURATION_TEMPERATURE},think=${CURATION_ENABLE_THINKING}) \\"
    echo "  python -u ${RUNNER} --env alfworld --memory_type skillos \\"
    echo "      --model ${MODEL} --curation_model ${MODEL} --curation_base_url http://localhost:${CUR_PORT}/v1 \\"
    echo "      --batch_size ${BATCH_SIZE} --retrieve_num ${RETRIEVE_NUM} --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > ${log} 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}"
  OPENAI_API_BASE="http://localhost:${EXEC_PORT}/v1" HISTORY_LENGTH="$HIST" ENABLE_THINKING="$THINKING" \
  python -u "$RUNNER" --env alfworld --memory_type skillos \
      --model          "$MODEL" \
      --curation_model "$MODEL" \
      --curation_base_url "http://localhost:${CUR_PORT}/v1" \
      --batch_size "$BATCH_SIZE" --retrieve_num "$RETRIEVE_NUM" --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "$log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> ${log}"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN (think/hist${HIST}/rn${RETRIEVE_NUM}, 3 seeds, 140 games, SAVE_RAW=${SAVE_RAW}) ====================="
  for r in 1 2 3; do run_seed "$r"; done
  echo "===================== DRY RUN complete — nothing executed ============="
  exit 0
fi

echo "[$(date +%H:%M:%S)] SkillOS bugfix validation: think/hist${HIST}/rn${RETRIEVE_NUM}, 3 seeds, 140 games each"
echo "  results -> Alfworld/results/${MODEL}/dev_sk-bugfix_*_${STAMP}_few_shot_False_skillos"
for r in 1 2 3; do
  run_seed "$r"
done
echo "[$(date +%H:%M:%S)] ALL 3 SEEDS COMPLETE."
echo "Per-run logs: logs_debug_memory/sk-bugfix_*_${STAMP}.log"
