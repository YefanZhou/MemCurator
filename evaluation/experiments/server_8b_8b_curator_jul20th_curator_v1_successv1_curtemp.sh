#!/bin/bash
# ==============================================================================
# NODE: CURATOR_V1 + success_only_v1 sweep, jul-20. Qwen3-8B executor + Qwen3-8B
# curator, bs=10, revise_react. BOTH executor and curator are THINKING. This sweep
# adds a CURATOR-TEMPERATURE axis: CT_LIST=(1.0 0.6).
#
# Derived from server_8b_8b_curator_jul16th_curator_v1.sh, with 3 changes:
#   1. --curation_mode success_only_v1   (was success_only) — the REVISED success-only
#      briefing prompt (CURATOR_SYSTEM_SUCCESS_ONLY_V1), A/B vs success_only.
#   2. NEW axis CT_LIST=(1.0 0.6): curator sampling temperature (CURATION_TEMPERATURE),
#      set per-job. Doubles the job count.
#   3. exp-name carries ct{temp} + mode so folders never collide; queue/log names bumped.
# Everything else (4x Qwen3-8B DP=2 servers, shared flock queue, RN/TC/OE axes) is
# identical to jul-16.
#
# SERVERS: this node serves FOUR Qwen3-8B, DP=2 each, one GPU pair per port:
#   :8001 GPUs 0,1   :8002 GPUs 2,3   :8003 GPUs 4,5   :8004 GPUs 6,7
# Each run uses ONE port for BOTH executor and curator; 4 ports drain the shared queue.
#
# MATRIX (reduced): think{true} x hist{3,5} x seed{3} (=6) x rn{3} x tc{short}
#         x oe{ON} x CT{1.0,0.6} = 6 x 1 x 1 x 2 = 12 runs. exp prefix cur_v1sv1_.
#         (rn=5, tc=obs0 dropped for now — re-add via RN_LIST / TC_LIST below.)
# PRIORITY: ct=1.0 jobs first.
#
# Usage:
#   bash server_8b_8b_curator_jul20th_curator_v1_successv1_curtemp.sh --dry-run
#   bash server_8b_8b_curator_jul20th_curator_v1_successv1_curtemp.sh --no-serve   # servers already up
#   NO_TEARDOWN=1 bash ...                                    # leave vLLM up at the end
#   tmux new -s curv1sv1 'bash .../server_8b_8b_curator_jul20th_curator_v1_successv1_curtemp.sh --no-serve'
#   CT_LIST="1.0" bash ...                                    # single curator temp
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

SWEEP_LOG="logs_debug_memory/_sweep_curator_v1sv1_8b8b_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN} no_serve=${NO_SERVE})"

# ---- shared env (per-run: OPENAI_API_BASE / HISTORY_LENGTH / ENABLE_THINKING / CURATION_TEMPERATURE) ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
# CURATION_TEMPERATURE is set PER-JOB from CT_LIST (below); the other curation knobs are fixed.
export CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000
export CURATOR_LOG_CALLS=1

CURATION_MODE="${CURATION_MODE:-success_only_v1}"   # {success_only, success_only_v1, success_and_fail, success_and_fail_v1}
STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[sweep] run stamp: ${STAMP}  curation_mode: ${CURATION_MODE}"

RUNNER="run_unified_dev_async_curator.py"
# Ports to use (executor+curator co-located per port; workers drain the queue across these).
# Override to use a subset, e.g. PORTS="8001 8002".  With --no-serve, only these need to be up.
read -r -a PORTS <<< "${PORTS:-8001 8002 8003 8004}"

# ------------------------------------------------------------------ #
# vLLM server launch: 4 x Qwen3-8B, DP=2 each, one GPU pair per port. #
# ------------------------------------------------------------------ #
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
# args: port  thinking  hist  rn  task_context  on_empty(0|1)  cur_temp  exp_name
run_exp() {
  local port="$1" thinking="$2" hist="$3" rn="$4" tc="$5" oe="$6" ct="$7" exp="$8"
  exp="${exp}_${STAMP}"
  local oe_flag=""; [ "$oe" = 1 ] && oe_flag="--curator_on_empty"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (server :${port}) ----"
    echo "  OPENAI_API_BASE=http://localhost:${port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} CURATION_TEMPERATURE=${ct} \\"
    echo "  python -u ${RUNNER} --env alfworld --memory_type curator_v1 --curation_mode ${CURATION_MODE} \\"
    echo "      --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \\"
    echo "      --curation_base_url http://localhost:${port}/v1 \\"
    echo "      --task_context ${tc} ${oe_flag} \\"
    echo "      --batch_size 10 --retrieve_num ${rn} --max_steps 30 --exp_name ${exp} --overwrite"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  think=${thinking} hist=${hist} rn=${rn} tc=${tc} on_empty=${oe} cur_temp=${ct}"
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  CURATION_TEMPERATURE="$ct" \
  python -u "$RUNNER" --env alfworld --memory_type curator_v1 --curation_mode "$CURATION_MODE" \
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
# Job matrix. Base = think{true} x hist{3,5} x seed{3} (6); expanded over
# RN_LIST x TC_LIST x OE_LIST x CT_LIST.
# Job spec: thinking|hist|rn|task_context|on_empty|cur_temp|exp_name
# PRIORITY: ct=1.0 jobs FIRST.
# ==============================================================================
RN_LIST=(3)                                 # rn=3 only (rn=5 dropped for now)
TC_LIST=(short)                             # short only (obs0 dropped for now)
OE_LIST=(1)                                 # curator_on_empty ON only
read -r -a CT_LIST <<< "${CT_LIST:-1.0 0.6}"   # curator temperature axis

# hist={3,5}.
BASE=(
  "true|5|cur_v1sv1_think_hist5_run1"
  "true|5|cur_v1sv1_think_hist5_run2"
  "true|5|cur_v1sv1_think_hist5_run3"
  "true|3|cur_v1sv1_think_hist3_run1"
  "true|3|cur_v1sv1_think_hist3_run2"
  "true|3|cur_v1sv1_think_hist3_run3"
)

JOBS=()
for base in "${BASE[@]}"; do
  IFS='|' read -r t h e <<< "$base"
  for rn in "${RN_LIST[@]}"; do
    for tc in "${TC_LIST[@]}"; do
      for oe in "${OE_LIST[@]}"; do
        for ct in "${CT_LIST[@]}"; do
          prio=1
          [ "$ct" = "1.0" ] && prio=0   # run ct=1.0 jobs first
          JOBS+=("${prio}|${t}|${h}|${rn}|${tc}|${oe}|${ct}|${e}_rn${rn}_tc${tc}_oe${oe}_ct${ct}")
        done
      done
    done
  done
done
# stable sort by leading priority key, then strip it (portable; no mapfile).
_SORTED=()
while IFS= read -r _line; do _SORTED+=("$_line"); done < <(
  printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1n | sed 's/^[01]|//')
JOBS=("${_SORTED[@]}")

QUEUE="logs_debug_memory/_curator_v1sv1_queue_8b8b.txt"
QLOCK="logs_debug_memory/_curator_v1sv1_queue_8b8b.lock"

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
    IFS='|' read -r thinking hist rn tc oe ct exp <<< "$job"
    run_exp "$port" "$thinking" "$hist" "$rn" "$tc" "$oe" "$ct" "$exp"
  done
  echo "[$(date +%H:%M:%S)] worker :${port} — queue drained, exiting"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN ====================="
  echo "PORTS: ${PORTS[*]}  (Qwen3-8B DP=2 per port)"
  echo "CURATION_MODE=${CURATION_MODE}  CT_LIST=${CT_LIST[*]}"
  echo "SHARED QUEUE (${#JOBS[@]} jobs) — 4 ports drain in parallel; ct1.0 first:"
  i=0; for j in "${JOBS[@]}"; do IFS='|' read -r t h rn tc oe ct e <<< "$j"; i=$((i+1)); printf '[%02d] ' "$i"; DRY_RUN=1 run_exp "${PORTS[0]}" "$t" "$h" "$rn" "$tc" "$oe" "$ct" "$e" | head -1; done
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
