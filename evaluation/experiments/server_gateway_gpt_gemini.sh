#!/bin/bash
# ==============================================================================
# EXTERNAL API backend (Salesforce gateway) launch template — gpt-5.4-mini / gemini-3-lite
# as EXECUTOR and/or CURATOR, for reasoningbank and skillos, on ALFWorld.
#
# Uses the *_api.py runners (run_unified_dev_async_api.py / run_unified_dev_api.py) so nothing
# touches the vLLM originals. Gemini goes THROUGH the gateway (OpenAI-compatible), NOT Vertex —
# so it is passed WITHOUT the gemini/ prefix, as openai/gemini-3-lite.
#
# Backend switches (see the _api runners):
#   LLM_BACKEND=openai          -> executor uses the gateway (suppresses vLLM-only top_k/thinking)
#   CURATION_LLM_BACKEND=openai -> curator uses the gateway (reasoningbank markdown OR skillos
#                                  NATIVE tool-calls; no Qwen ✿-token path)
# Either can be left at vllm to mix (e.g. Qwen executor + GPT curator).
#
# Gateway auth: bearer OPENAI_API_KEY (+ optional X_API_KEY header). NO local vLLM server needed
# when both backends are openai — but ALFWorld data + textworld must be installed.
#
# Usage:
#   export OPENAI_API_KEY=...          # gateway bearer key (DO NOT commit)
#   export X_API_KEY=...               # optional gateway header key
#   bash server_gateway_gpt_gemini.sh --dry-run
#   tmux new -s gw 'bash .../server_gateway_gpt_gemini.sh'
#   MODEL=openai/gemini-3-lite bash ...     # override executor model
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"

# ---- gateway config ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
# OPENAI_API_KEY required for real runs (checked below, after the dry-run short-circuit).
# X_API_KEY is optional; export it before running if the gateway needs the header.

# ---- external backend for BOTH executor and curator ----
export LLM_BACKEND=openai
export CURATION_LLM_BACKEND=openai

# ---- sampling: NO top_k / enable_thinking (gateway rejects them; the _api runners suppress
#      them automatically when *_BACKEND=openai, but we also leave EXECUTOR_TOP_K unset here). ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-0.7}"
export EXECUTOR_TOP_P="${EXECUTOR_TOP_P:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export CURATION_TEMPERATURE="${CURATION_TEMPERATURE:-0.7}"
export CURATION_MAX_TOKENS="${CURATION_MAX_TOKENS:-1024}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export SAVE_RAW="${SAVE_RAW:-10}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-15}" PRINT_CHARS="${PRINT_CHARS:-2000}"
# Gateway rate-limit safety: cap concurrent executor calls (<= batch_size). async runner only.
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"

MODEL="${MODEL:-openai/gpt-5.4-mini}"        # executor model (openai/ prefix -> routed via base_url)
CURATION_MODEL="${CURATION_MODEL:-openai/gpt-5.4-mini}"

# Matrix: memory{reasoningbank,skillos} x history{3,5}. Small by default — expand as needed.
JOBS=()
for mem in reasoningbank skillos; do
  for hist in 3 5; do
    JOBS+=("${mem}|${hist}")
  done
done

run_exp() {
  local mem="$1" hist="$2"
  local pfx; [ "$mem" = skillos ] && pfx=skillos || pfx=rb
  local mtag; mtag=$(echo "$MODEL" | tr '/' '_')
  local exp="gw-${pfx}_${mtag}_hist${hist}_${STAMP}"
  if [ "$DRY_RUN" = 1 ]; then
    echo "  mem=${mem} hist=${hist} exec=${MODEL} curator=${CURATION_MODEL} -> ${exp}"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] START ${exp}"
  [ "$mem" = reasoningbank ] && rm -rf "Alfworld/memory/reasoningbank_${exp}"

  HISTORY_LENGTH="$hist" \
  python -u run_unified_dev_async_api.py --env alfworld --memory_type "$mem" \
      --model          "$MODEL" \
      --curation_model "$CURATION_MODEL" \
      --batch_size 10 --retrieve_num 5 --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)"
}

echo "[gateway] base=${OPENAI_API_BASE}  exec=${MODEL}  curator=${CURATION_MODEL}  "
echo "          LLM_BACKEND=${LLM_BACKEND} CURATION_LLM_BACKEND=${CURATION_LLM_BACKEND} "
echo "          x_api_key=$([ -n "${X_API_KEY:-}" ] && echo set || echo unset) max_conc=${MAX_CONCURRENCY} stamp=${STAMP}"

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN: ${#JOBS[@]} external-backend runs ====="
  for j in "${JOBS[@]}"; do IFS='|' read -r m h <<< "$j"; run_exp "$m" "$h"; done
  echo "===== DRY RUN complete — nothing executed ====="
  exit 0
fi

: "${OPENAI_API_KEY:?set OPENAI_API_KEY to the gateway bearer key}"

# Preflight: gateway reachable + executor model OK (uses the smoke test).
echo "[preflight] running test_gateway_api.py against ${MODEL#openai/} ..."
python test_gateway_api.py "${MODEL#openai/}" || { echo "ERROR: gateway smoke test failed"; exit 1; }

# Run sequentially (external rate limits); flip to & + wait for parallelism if the gateway allows.
for j in "${JOBS[@]}"; do IFS='|' read -r m h <<< "$j"; run_exp "$m" "$h"; done
echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} EXTERNAL-BACKEND RUNS COMPLETE."
