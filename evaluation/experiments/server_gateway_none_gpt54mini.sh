#!/bin/bash
# ==============================================================================
# EXTERNAL API backend (Salesforce gateway) — memory=NONE baseline with gpt-5.4-mini.
# 3 runs (run1/2/3), executor gpt-5.4-mini via the gateway, temp=1.0, max_completion_tokens=4096.
#
# memory=none => NO curator (no --curation_model, no vLLM loaded). Only the EXECUTOR backend
# matters, so we set LLM_BACKEND=openai; CURATION_LLM_BACKEND is irrelevant here.
#
# Uses run_unified_dev_async_api.py (the _api copy) so the vLLM originals are untouched.
# gpt-5 family: `max_tokens`->`max_completion_tokens` and top_k/enable_thinking suppression are
# handled automatically by the _api runner when LLM_BACKEND=openai.
#
# Usage:
#   export OPENAI_API_KEY=<gateway bearer key>      # or rely on the default below
#   bash server_gateway_none_gpt54mini.sh --dry-run
#   tmux new -s gw_none 'bash .../server_gateway_none_gpt54mini.sh'
#   HISTS="3 5" bash ...        # override history lengths (default: 5)
#   RUNS="1 2 3" bash ...       # override run indices
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
SWEEP_LOG="logs_debug_memory/_sweep_gw_none_gpt54mini_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# ---- gateway config ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
# X_API_KEY optional; export before running if the gateway requires the header.

# ---- external EXECUTOR backend (no curator for memory=none) ----
export LLM_BACKEND=openai

# ---- sampling / env ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export SAVE_RAW="${SAVE_RAW:-10}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-15}" PRINT_CHARS="${PRINT_CHARS:-2000}"
# Gateway rate-limit guard (<= batch_size). Lower to 4 if the gateway 429s.
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-10}"

MODEL="${MODEL:-openai/gpt-5.4-mini}"
HISTS="${HISTS:-5 3}"        # space-separated history lengths
RUNS="${RUNS:-1 2 3}"      # space-separated run indices

MODEL_TAG=$(echo "${MODEL#openai/}" | tr '/.' '__')   # gpt-5.4-mini -> gpt-5_4-mini

run_exp() {
  local hist="$1" run="$2"
  local exp="gw-none_${MODEL_TAG}_temp${EXECUTOR_TEMPERATURE}_max${EXECUTOR_MAX_TOKENS}_hist${hist}_run${run}_${STAMP}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  exec=${MODEL} temp=${EXECUTOR_TEMPERATURE} max=${EXECUTOR_MAX_TOKENS} hist=${hist} run=${run} -> ${exp}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${exp}"
  HISTORY_LENGTH="$hist" \
  python -u run_unified_dev_async_api.py --env alfworld --memory_type none \
      --model "$MODEL" \
      --batch_size 10 --retrieve_num 5 --max_steps 30 \
      --exp_name "${exp}" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp} (exit $?)"
}

echo "[gateway-none] base=${OPENAI_API_BASE} exec=${MODEL} temp=${EXECUTOR_TEMPERATURE} "
echo "               max_completion_tokens=${EXECUTOR_MAX_TOKENS} hists='${HISTS}' runs='${RUNS}' "
echo "               max_conc=${MAX_CONCURRENCY} stamp=${STAMP}  (dry_run=${DRY_RUN})"

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN ====="
  for h in $HISTS; do for r in $RUNS; do run_exp "$h" "$r"; done; done
  echo "===== DRY RUN complete — nothing executed ====="
  exit 0
fi

# Preflight: gateway reachable + executor model OK.
echo "[preflight] test_gateway_api.py ${MODEL#openai/} ..."
python test_gateway_api.py "${MODEL#openai/}" || { echo "ERROR: gateway smoke test failed"; exit 1; }

# Sequential runs (external rate limits; memory=none so nothing to serialize for memory anyway).
for h in $HISTS; do for r in $RUNS; do run_exp "$h" "$r"; done; done
echo "[$(date +%H:%M:%S)] ALL RUNS COMPLETE."
