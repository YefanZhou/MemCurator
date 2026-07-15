#!/bin/bash
# ==============================================================================
# CONTINUATION of server_8b_8b_curator_jul13th.sh — jul-14 EXTEND.
#
# The jul-13 curator sweep (48 runs) was interrupted. As of jul-14 the box has:
#   FINISHED (12): ALL cur_think_hist5_* jobs (3 seeds x rn{3,5} x tc{short,obs0}).
#   PARTIAL  (4) : cur_think_hist3_run1_* (idx 70-110) — to be DELETED before rerun.
#   MISSING (32) : cur_think_hist3_run{2,3}, all cur_nonthink_* (hist5+hist3).
# => 36 runs remain (48 - 12 finished). This script runs exactly those 36.
#
# It is the jul-13 script with BASE reduced to the 9 UNFINISHED base configs
# (the 3 cur_think_hist5 configs are dropped — already done). Everything else
# (servers, env, RN/TC/OE axes, queue, run_exp) is IDENTICAL to jul-13.
#
# BEFORE RUNNING: delete the 4 stale partials so the analysis scanner doesn't count
# them as "partial" (they'll be re-created fresh with this run's STAMP anyway):
#   rm -rf .../results/openai/Qwen/Qwen3-8B/dev_cur_think_hist3_run1_*_20260713_0833_*_curator
#
# SERVERS: FOUR Qwen3-8B, DP=2 each: :8001(0,1) :8002(2,3) :8003(4,5) :8004(6,7).
# Each run uses ONE port for both executor and curator; 4 ports drain a shared
# flock-guarded queue in parallel.
#
# Usage:
#   bash server_8b_8b_curator_jul14th_extend.sh --dry-run   # print 36 jobs, run nothing
#   bash server_8b_8b_curator_jul14th_extend.sh --no-serve  # servers already up
#   NO_TEARDOWN=1 bash ...                                  # leave servers up at end
#   tmux new -s curExt 'bash .../server_8b_8b_curator_jul14th_extend.sh'
#   STAMP=20260713_0833 bash ...   # reuse the original stamp (matches finished hist5 naming)
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

SWEEP_LOG="logs_debug_memory/_sweep_curator_8b8b_extend_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN} no_serve=${NO_SERVE})"

# ---- shared env (per-run: OPENAI_API_BASE / HISTORY_LENGTH / ENABLE_THINKING) ----
# IDENTICAL to jul-13 so the extended runs are directly comparable to the finished ones.
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

# ---- one run ----  (IDENTICAL to jul-13)
# args: port  thinking  hist  rn  task_context  on_empty(0|1)  exp_name
run_exp() {
  local port="$1" thinking="$2" hist="$3" rn="$4" tc="$5" oe="$6" exp="$7"
  exp="${exp}_${STAMP}"
  local oe_flag=""; [ "$oe" = 1 ] && oe_flag="--curator_on_empty"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (server :${port}) ----"
    echo "  OPENAI_API_BASE=http://localhost:${port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} \\"
    echo "  python -u ${RUNNER} --env alfworld --memory_type curator \\"
    echo "      --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \\"
    echo "      --curation_base_url http://localhost:${port}/v1 \\"
    echo "      --task_context ${tc} ${oe_flag} \\"
    echo "      --batch_size 10 --retrieve_num ${rn} --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > logs_debug_memory/${exp}.log 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  :${port}  think=${thinking} hist=${hist} rn=${rn} tc=${tc} on_empty=${oe}"
  OPENAI_API_BASE="http://localhost:${port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  python -u "$RUNNER" --env alfworld --memory_type curator \
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
# Job matrix — ONLY the 9 UNFINISHED base configs (cur_think_hist5_* already done).
# Expanded over RN_LIST x TC_LIST x OE_LIST => 9 x 2 x 2 x 1 = 36 runs.
# PRIORITY: think+hist3 first (they were next in the interrupted queue), then nonthink.
# ==============================================================================
RN_LIST=(3 5)
TC_LIST=(short obs0)
OE_LIST=(1)          # curator_on_empty ON only

# NOTE: the 3 cur_think_hist5_run{1,2,3} base configs are DROPPED (finished jul-13).
BASE=(
  "true|3|cur_think_hist3_run1"
  "true|3|cur_think_hist3_run2"
  "true|3|cur_think_hist3_run3"
  "false|5|cur_nonthink_hist5_run1"
  "false|5|cur_nonthink_hist5_run2"
  "false|5|cur_nonthink_hist5_run3"
  "false|3|cur_nonthink_hist3_run1"
  "false|3|cur_nonthink_hist3_run2"
  "false|3|cur_nonthink_hist3_run3"
)

# Priority key: 0 = think (hist3) first, 1 = nonthink. Stable sort keeps the full set
# but front-loads the think-hist3 cell (the ones that were in flight when jul-13 died).
JOBS=()
for base in "${BASE[@]}"; do
  IFS='|' read -r t h e <<< "$base"
  for rn in "${RN_LIST[@]}"; do
    for tc in "${TC_LIST[@]}"; do
      for oe in "${OE_LIST[@]}"; do
        prio=1
        [ "$t" = true ] && prio=0
        JOBS+=("${prio}|${t}|${h}|${rn}|${tc}|${oe}|${e}_rn${rn}_tc${tc}_oe${oe}")
      done
    done
  done
done
# stable sort by leading priority key, then strip the key (portable; bash 3.2-safe).
_SORTED=()
while IFS= read -r _line; do _SORTED+=("$_line"); done < <(
  printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1n | sed 's/^[01]|//')
JOBS=("${_SORTED[@]}")

QUEUE="logs_debug_memory/_curator_queue_8b8b_extend.txt"
QLOCK="logs_debug_memory/_curator_queue_8b8b_extend.lock"

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
  echo "SHARED QUEUE (${#JOBS[@]} jobs) — 4 ports drain in parallel; think-hist3 first:"
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
