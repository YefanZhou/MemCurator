#!/bin/bash
# ==============================================================================
# NODE A: self-contained CURATOR_V1 sweep, jul-16. Qwen3-8B executor + Qwen3-8B
# curator, bs=10, revise_react, executor temp 1.0. Curator = thinking, temp 1.0,
# 4096 tokens. This script BOTH launches the vLLM servers AND runs the sweep.
#
# curator_v1 = faithful-prompt MemCurator (curator_alfworld_v1.py): verbatim-derived
# system prompts, --curation_mode {success_only, success_and_fail}, reward-aware store,
# and per-call logging of the FULL prompt + retrieved provenance (store_index/BM25 score).
# THIS SWEEP: --curation_mode success_only (store & curate only successful trajectories).
#
# SERVERS (launched here): FOUR Qwen3-8B servers, DP=2 each, one per GPU pair:
#   :8001 GPUs 0,1   :8002 GPUs 2,3   :8003 GPUs 4,5   :8004 GPUs 6,7
# Each run uses ONE port for BOTH executor and curator (8b/8b co-located), so all
# four ports drain the shared job queue in parallel (self-balancing, flock-guarded).
#
# curator-specific axes:
#   --curation_mode success_only   (FIXED here — wins-only store; the faithful default)
#   --task_context {short, obs0}   (obs0 enriches ONLY the curator CURRENT-task
#                                   Question; not the BM25 key or stored records)
#   --curator_on_empty             (ON for all runs here — curator writes a briefing
#                                    even when nothing is retrieved)
#
# MATRIX: think{2} x hist{2} x seed{3} (=12) x rn{3,5} x task_context{short,obs0}
#         x curator_on_empty{ON} = 12 x 2 x 2 x 1 = 48 runs.  exp-name prefix cur_v1_
#         so results never collide with the original-curator sweep (cur_).
# PRIORITY ORDER: think + rn5 + hist5 jobs run FIRST (headline config early), then
# the rest. Full grid still runs; only the queue order changes.
#
# Memory (curator_v1_memory.jsonl + curator_calls.jsonl) lives INSIDE each result
# folder, so --overwrite wipes it along with the results.
#
# Usage:
#   bash server_8b_8b_curator_jul16th_curator_v1.sh --dry-run   # print servers + all jobs
#   bash server_8b_8b_curator_jul16th_curator_v1.sh --no-serve  # servers already up
#   NO_TEARDOWN=1 bash ...                             # leave vLLM servers running at the end
#   tmux new -s curAv1 'bash .../server_8b_8b_curator_jul16th_curator_v1.sh'
#   STAMP=jul16b bash .../server_8b_8b_curator_jul16th_curator_v1.sh
# ==============================================================================
set -u

DRY_RUN=0
NO_SERVE=0
for a in "$@"; do
  [ "$a" = "--dry-run" ] && DRY_RUN=1
  [ "$a" = "--no-serve" ] && NO_SERVE=1
done

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory logs_vllm

SWEEP_LOG="logs_debug_memory/_sweep_curator_v1_8b8b_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN} no_serve=${NO_SERVE})"

# ---- shared env (per-run: OPENAI_API_BASE / HISTORY_LENGTH / ENABLE_THINKING) ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000
export CURATOR_LOG_CALLS=1

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[sweep] run stamp: ${STAMP}"

RUNNER="run_unified_dev_async_curator.py"
PORTS=(8001 8002 8003 8004)

# ------------------------------------------------------------------ #
# vLLM server launch: 4 x Qwen3-8B, DP=2 each, one GPU pair per port. #
# ------------------------------------------------------------------ #
# port -> CUDA devices (DP=2 => 2 GPUs each; 4 ports => 8 GPUs)
declare -A PORT_GPUS=( [8001]="0,1" [8002]="2,3" [8003]="4,5" [8004]="6,7" )
VLLM_PIDS=()

launch_servers() {
  echo "[vllm] launching 4x Qwen3-8B DP=2 (ports ${PORTS[*]})"
  for port in "${PORTS[@]}"; do
    local gpus="${PORT_GPUS[$port]}"
    echo "[vllm] :${port} on GPUs ${gpus} -> logs_vllm/vllm_${port}.log"
    CUDA_VISIBLE_DEVICES="$gpus" vllm serve Qwen/Qwen3-8B \
        --port "$port" --served-model-name Qwen/Qwen3-8B --dtype bfloat16 \
        --data-parallel-size 2 --tensor-parallel-size 1 \
        --max-model-len 40960 --gpu-memory-utilization 0.90 \
        > "logs_vllm/vllm_${port}.log" 2>&1 &
    VLLM_PIDS+=($!)
  done
}

wait_servers() {
  echo "[vllm] waiting for all ${#PORTS[@]} servers to become ready (timeout 20m)..."
  local deadline=$(( $(date +%s) + 1200 ))
  for port in "${PORTS[@]}"; do
    until curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
      if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "ERROR: :${port} not ready before timeout — see logs_vllm/vllm_${port}.log"; exit 1
      fi
      sleep 10
    done
    echo "[vllm] :${port} READY"
  done
}

teardown_servers() {
  [ "${NO_TEARDOWN:-0}" = 1 ] && { echo "[vllm] NO_TEARDOWN=1 — leaving servers up"; return; }
  echo "[vllm] tearing down vLLM servers (pids: ${VLLM_PIDS[*]:-none})"
  for pid in "${VLLM_PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
}

# ---- one run ----
# args: port  thinking  hist  rn  task_context  on_empty(0|1)  exp_name
run_exp() {
  local port="$1" thinking="$2" hist="$3" rn="$4" tc="$5" oe="$6" exp="$7"
  exp="${exp}_${STAMP}"
  local oe_flag=""; [ "$oe" = 1 ] && oe_flag="--curator_on_empty"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (server :${port}) ----"
    echo "  OPENAI_API_BASE=http://localhost:${port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} \\"
    echo "  python -u ${RUNNER} --env alfworld --memory_type curator_v1 --curation_mode success_only \\"
    echo "      --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \\"
    echo "      --curation_base_url http://localhost:${port}/v1 \\"
    echo "      --task_context ${tc} ${oe_flag} \\"
    echo "      --batch_size 10 --retrieve_num ${rn} --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > logs_debug_memory/${exp}.log 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  think=${thinking} hist=${hist} rn=${rn} tc=${tc} on_empty=${oe}"
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode success_only \
      --model          openai/Qwen/Qwen3-8B \
      --curation_model openai/Qwen/Qwen3-8B \
      --curation_base_url "http://localhost:${port}/v1" \
      --task_context "$tc" $oe_flag \
      --batch_size 10 --retrieve_num "$rn" --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> logs_debug_memory/${exp}.log"
}

# ==============================================================================
# Job matrix. Base = think{2} x hist{2} x seed{3} (12); expanded over
# RN_LIST x TC_LIST x OE_LIST. Job spec: thinking|hist|rn|task_context|on_empty|exp_name
# PRIORITY: jobs are sorted so (think=true, rn=5, hist=5) come FIRST.
# ==============================================================================
RN_LIST=(3 5)
TC_LIST=(short obs0)
OE_LIST=(1)          # curator_on_empty ON only

BASE=(
  "true|5|cur_v1_think_hist5_run1"
  "true|5|cur_v1_think_hist5_run2"
  "true|5|cur_v1_think_hist5_run3"
  "true|3|cur_v1_think_hist3_run1"
  "true|3|cur_v1_think_hist3_run2"
  "true|3|cur_v1_think_hist3_run3"
)

# Priority key: 0 = headline (think & rn5 & hist5), 1 = everything else. Sort keeps
# the full set but front-loads the priority cell.
JOBS=()
for base in "${BASE[@]}"; do
  IFS='|' read -r t h e <<< "$base"
  for rn in "${RN_LIST[@]}"; do
    for tc in "${TC_LIST[@]}"; do
      for oe in "${OE_LIST[@]}"; do
        prio=1
        [ "$t" = true ] && [ "$rn" = 5 ] && [ "$h" = 5 ] && prio=0
        JOBS+=("${prio}|${t}|${h}|${rn}|${tc}|${oe}|${e}_rn${rn}_tc${tc}_oe${oe}")
      done
    done
  done
done
# stable sort by leading priority key, then strip the key (portable; no mapfile — the
# stock macOS bash 3.2 lacks it, and a silent fallback would leave JOBS unsorted).
_SORTED=()
while IFS= read -r _line; do _SORTED+=("$_line"); done < <(
  printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1n | sed 's/^[01]|//')
JOBS=("${_SORTED[@]}")

QUEUE="logs_debug_memory/_curator_v1_queue_8b8b_think.txt"
QLOCK="logs_debug_memory/_curator_v1_queue_8b8b_think.lock"

pop_job() {
  local line=""
  exec 9>"$QLOCK"; flock 9
  if [ -s "$QUEUE" ]; then line=$(head -n1 "$QUEUE"); tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"; fi
  flock -u 9
  printf '%s' "$line"
}

worker() {
  local port="$1" job
  while :; do
    job=$(pop_job); [ -z "$job" ] && break
    IFS='|' read -r thinking hist rn tc oe exp <<< "$job"
    run_exp "$port" "$thinking" "$hist" "$rn" "$tc" "$oe" "$exp"
  done
  echo "[$(date +%H:%M:%S)] worker :${port} — queue drained, exiting"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN ====================="
  echo "SERVERS: 4x Qwen3-8B DP=2 -> :8001(0,1) :8002(2,3) :8003(4,5) :8004(6,7)"
  echo "SHARED QUEUE (${#JOBS[@]} jobs) — 4 ports drain in parallel; priority (think,rn5,hist5) first:"
  i=0; for j in "${JOBS[@]}"; do IFS='|' read -r t h rn tc oe e <<< "$j"; i=$((i+1)); printf '[%02d] ' "$i"; DRY_RUN=1 run_exp "${PORTS[0]}" "$t" "$h" "$rn" "$tc" "$oe" "$e" | head -1; done
  echo "===================== DRY RUN complete — nothing executed ============="
  exit 0
fi

# ---- launch servers (unless --no-serve) + ensure teardown on exit ----
if [ "$NO_SERVE" = 0 ]; then
  launch_servers
  trap teardown_servers EXIT
fi
wait_servers

printf '%s\n' "${JOBS[@]}" > "$QUEUE"
: > "$QLOCK"

echo "[$(date +%H:%M:%S)] launching ${#PORTS[@]} workers draining the shared ${#JOBS[@]}-job queue"
PIDS=()
for port in "${PORTS[@]}"; do worker "$port" & PIDS+=($!); done
wait "${PIDS[@]}"
echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS COMPLETE."
