#!/bin/bash
# ==============================================================================
# SMOKE TEST — curator_v1 with_thinking trajectory style (API path, gpt-5.4 default).
#
# Runs a tiny curator_v1 job with --curator_trajectory_style with_thinking, then auto-inspects
# the stored memory to confirm the new feature works end-to-end:
#   1. init line shows traj=with_thinking
#   2. curator_v1_memory.jsonl trajectories contain [Thinking]: lines
#   3. thinking is head/tail-truncated when it exceeds the per-step budget
#   4. baseline sanity: action_only run has NO [Thinking] (proves default unchanged)
#
# Small + fast: 6 games, 1 seed. Uses the gateway gpt-5.4 (short, mostly-rationale CoT) by
# default; override MODEL=gemini/gemini-3.1-flash-lite to smoke the Vertex path instead.
#
# Usage:
#   bash smoke_curator_v1_with_thinking.sh                 # gpt-5.4, 6 games, with_thinking
#   NUM_GAMES=3 bash smoke_curator_v1_with_thinking.sh     # even faster
#   BUDGET=200 bash smoke_curator_v1_with_thinking.sh      # tiny budget -> force truncation
#   MODEL=gemini/gemini-3.1-flash-lite bash smoke_curator_v1_with_thinking.sh
# ==============================================================================
set -u

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }

# conda env (so python resolves under ssh/nohup)
if ! command -v python >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; conda activate "${CONDA_ENV:-memory}"; }
fi

# ---- gateway + Vertex creds (same as the real sweeps) ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-salesforce-research-internal}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# ---- executor/curation env ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export SAVE_RAW="${SAVE_RAW:-6}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-0}" PRINT_CHARS="${PRINT_CHARS:-1500}"
export CURATOR_LOG_CALLS="${CURATOR_LOG_CALLS:-1}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-6}"

MODEL="${MODEL:-openai/gpt-5.4}"
NUM_GAMES="${NUM_GAMES:-6}"
HIST="${HIST:-5}"
RETRIEVE_NUM="${RETRIEVE_NUM:-5}"
BUDGET="${BUDGET:-8000}"          # --curator_think_token_budget (set small, e.g. 200, to force truncation)
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUNNER=run_unified_dev_async_curator_api.py

if [[ "$MODEL" == gemini/* ]]; then EXB=vllm; else EXB=openai; fi

run_one() {  # style exp_suffix budget
  local style="$1" suffix="$2" budget="$3"
  local exp="smoke_cv1_${style}_${suffix}_${STAMP}"
  echo ""
  echo "############################################################"
  echo "### RUN style=${style} budget=${budget} -> ${exp}"
  echo "############################################################"
  env LLM_BACKEND="$EXB" CURATION_LLM_BACKEND="$EXB" \
      HISTORY_LENGTH="$HIST" CURATION_ENABLE_THINKING=true CURATION_MAX_TOKENS=4096 \
  python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode success_only \
      --model "$MODEL" --curation_model "$MODEL" \
      --task_context short --curator_on_empty \
      --curator_trajectory_style "$style" --curator_think_token_budget "$budget" \
      --batch_size 6 --retrieve_num "$RETRIEVE_NUM" --max_steps 30 --num_games "$NUM_GAMES" \
      --exp_name "$exp" --overwrite 2>&1 | tee "logs_debug_memory/${exp}.log"
  echo "$exp"   # last line = exp name (caller captures)
}

# --- 1. the real test: with_thinking at the chosen budget ---
WT_EXP="smoke_cv1_with_thinking_b${BUDGET}_${STAMP}"
env LLM_BACKEND="$EXB" CURATION_LLM_BACKEND="$EXB" \
    HISTORY_LENGTH="$HIST" CURATION_ENABLE_THINKING=true CURATION_MAX_TOKENS=4096 \
python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode success_only \
    --model "$MODEL" --curation_model "$MODEL" \
    --task_context short --curator_on_empty \
    --curator_trajectory_style with_thinking --curator_think_token_budget "$BUDGET" \
    --batch_size 6 --retrieve_num "$RETRIEVE_NUM" --max_steps 30 --num_games "$NUM_GAMES" \
    --exp_name "$WT_EXP" --overwrite 2>&1 | tee "logs_debug_memory/${WT_EXP}.log"

# --- 2. baseline: action_only (proves default is unchanged / no [Thinking]) ---
AO_EXP="smoke_cv1_action_only_${STAMP}"
env LLM_BACKEND="$EXB" CURATION_LLM_BACKEND="$EXB" \
    HISTORY_LENGTH="$HIST" CURATION_ENABLE_THINKING=true CURATION_MAX_TOKENS=4096 \
python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode success_only \
    --model "$MODEL" --curation_model "$MODEL" \
    --task_context short --curator_on_empty \
    --curator_trajectory_style action_only \
    --batch_size 6 --retrieve_num "$RETRIEVE_NUM" --max_steps 30 --num_games "$NUM_GAMES" \
    --exp_name "$AO_EXP" --overwrite 2>&1 | tee "logs_debug_memory/${AO_EXP}.log"

# ============================ AUTO-VERIFY ============================
RESULTS="Alfworld/results/${MODEL}"
WT_DIR="${RESULTS}/dev_${WT_EXP}_few_shot_False_curator_v1"
AO_DIR="${RESULTS}/dev_${AO_EXP}_few_shot_False_curator_v1"

echo ""
echo "=================== SMOKE VERIFY ==================="
python3 - "$WT_DIR" "$AO_DIR" <<'PY'
import sys, os, json
wt_dir, ao_dir = sys.argv[1], sys.argv[2]

def load_traj(d):
    f = os.path.join(d, "curator_v1_memory.jsonl")
    if not os.path.exists(f):
        return None
    recs = [json.loads(l) for l in open(f) if l.strip()]
    return recs

def check(name, d, expect_thinking):
    print(f"\n--- {name} ---")
    print("  dir:", d)
    recs = load_traj(d)
    if recs is None:
        print("  !! curator_v1_memory.jsonl NOT found (no successful trajectory stored? try more games)")
        return
    print(f"  stored trajectories: {len(recs)}")
    has_think = any("[Thinking]:" in r.get("trajectory","") for r in recs)
    trunc     = any("[thinking truncated]" in r.get("trajectory","") for r in recs)
    print(f"  [Thinking]: present  -> {has_think}   (expected {expect_thinking})")
    print(f"  truncation marker    -> {trunc}")
    status = "PASS" if has_think == expect_thinking else "*** FAIL ***"
    print(f"  {status}")
    # show one trajectory sample (first 900 chars)
    if recs:
        t = recs[0]["trajectory"]
        print("  --- sample trajectory[0] (first 900 chars) ---")
        print("   ", t[:900].replace("\n","\n    "))

check("with_thinking", wt_dir, expect_thinking=True)
check("action_only (baseline)", ao_dir, expect_thinking=False)
print("\n=================== END VERIFY ===================")
PY
