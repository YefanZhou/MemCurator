#!/bin/bash
# ==============================================================================
# EXTERNAL API MEMORY sweeps — executor=API AND curator=API, for reasoningbank + skillos.
# GPT-5.5 / GPT-5.4 via the Salesforce gateway; Gemini-3.1-flash-lite via Vertex.
#
# For each model, both the EXECUTOR and the CURATOR are that same API model:
#   openai/*  -> gateway  (LLM_BACKEND=openai, CURATION_LLM_BACKEND=openai; native tool-calls for
#                skillos, litellm text for reasoningbank; max_tokens->max_completion_tokens auto)
#   gemini/*  -> Vertex   (--model + --curation_model gemini/... ; curation_vertex_native for
#                skillos, Vertex text for reasoningbank; no gateway env needed)
# Per-model lanes run IN PARALLEL (different sources). Within a lane, method x hist x run run serially.
#
# Uses run_unified_dev_async_api.py. Memory (skills.json / reasoning_bank.jsonl) lives INSIDE each
# result folder and --overwrite wipes it, so no separate memory rm -rf is needed.
#
# HYPERS (UNIFIED across both methods, per request):
#   both reasoningbank + skillos: revise_react, thinking executor, exec temp 1.0,
#   EXECUTOR_MAX_TOKENS=4096, curator temp 1.0, CURATION_MAX_TOKENS=4096, retrieve_num 5,
#   hist {5,3}, seeds {1,2,3}.
# (cur_think differs cosmetically — true for skillos, false for rb — but is INERT for API models:
#  enable_thinking is a Qwen chat-template flag the gateway/Vertex don't use.)
#
# RESUME: SKIP_DONE=1 (default) skips any method/model/hist/run with a 140-idx folder (stamp-agnostic).
# STAMP: defaults to current timestamp; pass STAMP=... only to force a fixed label.
#
# Usage:
#   bash server_api_memory_gpt_gemini.sh --dry-run
#   NUM_GAMES=6 bash server_api_memory_gpt_gemini.sh              # fast debug pass
#   tmux new -s api_mem 'bash .../server_api_memory_gpt_gemini.sh'
#   METHODS=reasoningbank MODELS="openai/gpt-5.4" bash ...        # subset
#   MAX_CONCURRENCY=4 bash ...                                    # throttle (gemini safety)
# ==============================================================================
set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

# Ensure conda `memory` env is active so `python` resolves under nohup/ssh.
if [ "$DRY_RUN" = 0 ] && ! command -v python >/dev/null 2>&1; then
  CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  [ -f "$CONDA_SH" ] && { source "$CONDA_SH"; conda activate "${CONDA_ENV:-memory}"; }
  command -v python >/dev/null 2>&1 || { echo "ERROR: 'python' not found (activate the 'memory' env)"; exit 1; }
fi

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
SWEEP_LOG="logs_debug_memory/_sweep_api_memory_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1

# ---- gateway config (openai/* models) ----
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-6b13219154217a4349fcc03197526669}"
# ---- Vertex creds (gemini/* models); the _api runner also defaults these ----
export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-salesforce-research-internal}"
export GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION:-global}"

# ---- shared executor env (matches both reference sweeps: revise_react, temp 1.0, thinking) ----
export EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
export EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"     # thinking executor (both refs)
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export SAVE_RAW="${SAVE_RAW:-20}" PROMPT_SHOW_EVERY="${PROMPT_SHOW_EVERY:-15}" PRINT_CHARS="${PRINT_CHARS:-2000}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Pydantic serializer warnings:UserWarning}"

BATCH_SIZE="${BATCH_SIZE:-10}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-$BATCH_SIZE}"

# What to sweep. Each model gets its own parallel lane.
MODELS="${MODELS:-openai/gpt-5.5 openai/gpt-5.4 openai/gpt-5.4-mini gemini/gemini-3.1-flash-lite}"
METHODS="${METHODS:-reasoningbank skillos}"
RETRIEVE_NUM="${RETRIEVE_NUM:-5}"
NUM_GAMES="${NUM_GAMES:-0}"        # 0 => all 140
RUNS="${RUNS:-1 2 3}"
# hist grid — SAME for both methods now: {5,3}.
RB_HISTS="${RB_HISTS:-5 3}"
SK_HISTS="${SK_HISTS:-5 3}"
SKIP_DONE="${SKIP_DONE:-1}"

# Already finished? (140-idx folder for this method/model/hist/run, any stamp).
already_done() {
  local model="$1" method="$2" hist="$3" run="$4" d n
  for d in Alfworld/results/${model}/dev_api-${method%%_*}*_hist${hist}_run${run}_*_few_shot_False_${method}; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/idx_*.json 2>/dev/null | wc -l)
    [ "$n" -ge 140 ] && return 0
  done
  return 1
}

# args: model method hist run
run_exp() {
  local model="$1" method="$2" hist="$3" run="$4"
  local tag; tag=$(echo "$model" | tr '/.' '__')
  local pfx; [ "$method" = skillos ] && pfx=sk || pfx=rb
  local exp="api-${pfx}_${tag}_temp${EXECUTOR_TEMPERATURE}_hist${hist}_rn${RETRIEVE_NUM}_run${run}_${STAMP}"

  if [ "${SKIP_DONE}" = 1 ] && already_done "$model" "$method" "$hist" "$run"; then
    echo "[$(date +%H:%M:%S)] SKIP  ${method} ${model} hist=${hist} run=${run} (already 140-idx)"
    return 0
  fi

  # Backend + curator settings.
  #   openai/* -> gateway (both backends external); gemini/* -> Vertex (model prefix routes it).
  #   Curator max_tokens = 4096 for BOTH methods (unified). cur_think is inert for API models
  #   (enable_thinking is a Qwen chat-template flag; gateway suppresses it, Vertex ignores it) —
  #   kept only for log symmetry.
  local exec_backend cur_backend cur_think cur_max
  if [[ "$model" == gemini/* ]]; then
    exec_backend="vllm"; cur_backend="vllm"        # gemini chosen by model prefix, not the flags
  else
    exec_backend="openai"; cur_backend="openai"
  fi
  cur_max=4096
  if [ "$method" = skillos ]; then cur_think=true; else cur_think=false; fi

  if [ "$DRY_RUN" = 1 ]; then
    local route; [[ "$model" == gemini/* ]] && route="Vertex" || route="gateway"
    echo "  [${route}] method=${method} exec=${model} curator=${model} hist=${hist} run=${run} rn=${RETRIEVE_NUM} cur(think=${cur_think},max=${cur_max}) -> ${exp}"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  (exec=${exec_backend} cur=${cur_backend} cur_think=${cur_think})"
  env LLM_BACKEND="$exec_backend" CURATION_LLM_BACKEND="$cur_backend" \
      HISTORY_LENGTH="$hist" CURATION_ENABLE_THINKING="$cur_think" CURATION_MAX_TOKENS="$cur_max" \
  python -u run_unified_dev_async_api.py --env alfworld --memory_type "$method" \
      --model          "$model" \
      --curation_model "$model" \
      --batch_size "$BATCH_SIZE" --retrieve_num "$RETRIEVE_NUM" --max_steps 30 --num_games "$NUM_GAMES" \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp} (exit $?)  -> logs_debug_memory/${exp}.log"
}

# One serial lane per model: iterate method x (its hist grid) x run.
model_lane() {
  local m="$1" method hists h r
  for method in $METHODS; do
    [ "$method" = skillos ] && hists="$SK_HISTS" || hists="$RB_HISTS"
    for h in $hists; do for r in $RUNS; do run_exp "$m" "$method" "$h" "$r"; done; done
  done
  echo "[$(date +%H:%M:%S)] lane ${m} — done"
}

echo "[api-memory] models='${MODELS}'  methods='${METHODS}'  rb_hists='${RB_HISTS}' sk_hists='${SK_HISTS}' runs='${RUNS}'"
echo "             exec temp=${EXECUTOR_TEMPERATURE} think=${ENABLE_THINKING} style=${PROMPT_STYLE} rn=${RETRIEVE_NUM} num_games=${NUM_GAMES}"
echo "             max_conc=${MAX_CONCURRENCY} skip_done=${SKIP_DONE} stamp=${STAMP} (dry_run=${DRY_RUN})"
echo "             per-model lanes IN PARALLEL: openai/*=gateway ; gemini/*=Vertex. Curator=same model as executor."

if [ "$DRY_RUN" = 1 ]; then
  echo "===== DRY RUN (lanes shown sequentially; real run executes them in parallel) ====="
  n=0
  for m in $MODELS; do
    for method in $METHODS; do
      [ "$method" = skillos ] && hh="$SK_HISTS" || hh="$RB_HISTS"
      for h in $hh; do for r in $RUNS; do run_exp "$m" "$method" "$h" "$r"; n=$((n+1)); done; done
    done
  done
  echo "===== DRY RUN complete — ${n} run-slots (finished ones SKIP), nothing executed ====="
  exit 0
fi

# Preflight gateway models (skip gemini — Vertex).
for m in $MODELS; do
  if [[ "$m" != gemini/* ]]; then
    echo "[preflight] test_gateway_api.py ${m#openai/} ..."
    python test_gateway_api.py "${m#openai/}" || { echo "ERROR: gateway smoke test failed for ${m}"; exit 1; }
  fi
done

PIDS=()
for m in $MODELS; do model_lane "$m" & PIDS+=($!); done
wait "${PIDS[@]}"
echo "[$(date +%H:%M:%S)] ALL RUNS COMPLETE."
