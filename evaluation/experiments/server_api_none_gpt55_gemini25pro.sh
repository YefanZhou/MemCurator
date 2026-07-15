#!/bin/bash
# ==============================================================================
# EXTERNAL API executors, memory=NONE baseline — GPT-5.5 (gateway) + Gemini-2.5-pro (Vertex).
#
# Two executors reached via DIFFERENT backends (no memory => no curator):
#   gpt-5.5          -> Salesforce gateway  (LLM_BACKEND=openai, --model openai/gpt-5.5)
#   gemini-2.5-pro   -> Vertex AI            (--model gemini/gemini-2.5-pro; gemini/ prefix
#                       routes to llm_vertexai in the _api runner — NO gateway env needed)
# The per-model env is set inside run_exp so the two backends don't cross-contaminate.
#
# Uses run_unified_dev_async_api.py (the _api copy) so the vLLM originals are untouched.
#
# DEBUGGABILITY: SAVE_RAW defaults high (=20) so a fat raw_trace (every step's prompt+response)
# is saved for many sample games; combined with a small NUM_GAMES for the first pass you get
# full per-step visibility to debug after the dry-run. Set NUM_GAMES=0 for the full 140.
#
# Usage:
#   bash server_api_none_gpt55_gemini25pro.sh --dry-run          # print the plan, run nothing
#   NUM_GAMES=6 bash server_api_none_gpt55_gemini25pro.sh        # tiny debug pass (6 games, full traces)
#   tmux new -s api_none 'bash .../server_api_none_gpt55_gemini25pro.sh'   # full run
#   MODELS="openai/gpt-5.5" bash ...                             # just one model
#   HISTS="3 5" RUNS="1 2 3" bash ...                            # override grid
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
SWEEP_LOG="logs_debug_memory/_sweep_api_none_gpt55_gem25_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# ---- gateway config (for the gpt-5.5 runs; gemini ignores these) ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
# X_API_KEY optional (gateway header); export before running if needed.

# ---- Vertex creds (for the gemini runs); the _api runner also defaults these internally ----
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-salesforce-research-internal}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# ---- sampling / env (shared) ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
# Debug knobs: keep a fat raw_trace for many games + print probe prompt/response every few steps.
export SAVE_RAW="${SAVE_RAW:-20}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-10}" PRINT_CHARS="${PRINT_CHARS:-3000}"
# Concurrency guard (<= batch_size). External APIs rate-limit; keep modest.
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"

# MODELS: space-separated executor models. openai/ -> gateway ; gemini/ -> Vertex.
MODELS="${MODELS:-openai/gpt-5.5 gemini/gemini-2.5-pro}"
HISTS="${HISTS:-5 3}"
RUNS="${RUNS:-1 2 3}"
# NUM_GAMES: 0 => all 140. A small value (e.g. 6) is a fast debug pass with full raw traces.
NUM_GAMES="${NUM_GAMES:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"

run_exp() {
  local model="$1" hist="$2" run="$3"
  local tag; tag=$(echo "$model" | tr '/.' '__')          # openai/gpt-5.5 -> openai_gpt-5_5
  local exp="api-none_${tag}_temp${EXECUTOR_TEMPERATURE}_max${EXECUTOR_MAX_TOKENS}_hist${hist}_run${run}_${STAMP}"

  # Per-model backend selection. gemini/ -> Vertex (unset LLM_BACKEND); else -> gateway (openai).
  local backend_env
  if [[ "$model" == gemini/* ]]; then
    backend_env="LLM_BACKEND=vllm"      # gemini path is chosen by the model prefix, not LLM_BACKEND
  else
    backend_env="LLM_BACKEND=openai"    # gateway (suppresses top_k/thinking, uses max_completion_tokens)
  fi

  if [ "$DRY_RUN" = 1 ]; then
    local route; [[ "$model" == gemini/* ]] && route="Vertex AI" || route="gateway (${OPENAI_API_BASE})"
    echo "  exec=${model} [${route}]  temp=${EXECUTOR_TEMPERATURE} max=${EXECUTOR_MAX_TOKENS} hist=${hist} run=${run} num_games=${NUM_GAMES} save_raw=${SAVE_RAW} -> ${exp}"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  (${backend_env})"
  env $backend_env HISTORY_LENGTH="$hist" \
  python -u run_unified_dev_async_api.py --env alfworld --memory_type none \
      --model "$model" \
      --batch_size "$BATCH_SIZE" --retrieve_num 5 --max_steps 30 --num_games "$NUM_GAMES" \
      --exp_name "${exp}" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp} (exit $?)  -> logs_debug_memory/${exp}.log"
}

echo "[api-none] models='${MODELS}'  hists='${HISTS}'  runs='${RUNS}'  num_games=${NUM_GAMES}  batch=${BATCH_SIZE}"
echo "           temp=${EXECUTOR_TEMPERATURE} max_tok=${EXECUTOR_MAX_TOKENS} save_raw=${SAVE_RAW} max_conc=${MAX_CONCURRENCY} stamp=${STAMP} (dry_run=${DRY_RUN})"
echo "           gpt-5.5 -> gateway ${OPENAI_API_BASE} ; gemini-2.5-pro -> Vertex project=${GOOGLE_CLOUD_PROJECT} loc=${GOOGLE_CLOUD_LOCATION}"

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN ====="
  n=0
  for m in $MODELS; do for h in $HISTS; do for r in $RUNS; do run_exp "$m" "$h" "$r"; n=$((n+1)); done; done; done
  echo "===== DRY RUN complete — ${n} runs planned, nothing executed ====="
  exit 0
fi

# Preflight the gateway model(s) with the smoke test (skip gemini — Vertex, no gateway smoke).
for m in $MODELS; do
  if [[ "$m" != gemini/* ]]; then
    echo "[preflight] test_gateway_api.py ${m#openai/} ..."
    python test_gateway_api.py "${m#openai/}" || { echo "ERROR: gateway smoke test failed for ${m}"; exit 1; }
  fi
done

# Sequential runs (external rate limits; memory=none so nothing to serialize).
for m in $MODELS; do for h in $HISTS; do for r in $RUNS; do run_exp "$m" "$h" "$r"; done; done; done
echo "[$(date +%H:%M:%S)] ALL RUNS COMPLETE."
