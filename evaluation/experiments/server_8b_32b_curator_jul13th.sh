#!/bin/bash
# ==============================================================================
# NODE B: self-contained CURATOR sweep, jul-13. Qwen3-32B EXECUTOR + Qwen3-8B
# CURATOR, bs=10, revise_react, executor temp 1.0. Curator = thinking, temp 1.0,
# 4096 tokens. This script BOTH launches the vLLM servers AND runs the sweep.
#
# SERVERS (launched here), all DP=2 (8 GPUs total):
#   :8001 GPUs 0,1  Qwen3-32B  -> EXECUTOR lane 1
#   :8002 GPUs 2,3  Qwen3-32B  -> EXECUTOR lane 2
#   :8003 GPUs 4,5  Qwen3-32B  -> EXECUTOR lane 3
#   :8004 GPUs 6,7  Qwen3-8B   -> SHARED CURATOR (all runs curate here)
# Three 32B executor lanes drain the shared job queue in parallel; every run's
# curator points at the single 8B server on :8004 (so the 8B server fields curation
# for all three lanes). This maximizes utilization: 3 games running + 1 curator.
#
# 32B-executor variant of server_8b_8b_curator_jul13th.sh: same grid, runner, and
# curator axes; only the executor model + the exec/curator server split differ.
#
# MATRIX: think{2} x hist{2} x seed{3} (=12) x rn{3,5} x task_context{short,obs0}
#         x curator_on_empty{ON} = 48 runs.
# PRIORITY ORDER: think + rn5 + hist5 jobs run FIRST, then the rest (full grid kept).
#
# Usage:
#   bash server_8b_32b_curator_jul13th.sh --dry-run     # print servers + all jobs
#   bash server_8b_32b_curator_jul13th.sh --no-serve    # servers already up
#   NO_TEARDOWN=1 bash ...                              # leave vLLM up at the end
#   tmux new -s curB 'bash .../server_8b_32b_curator_jul13th.sh'
#   STAMP=jul13b bash .../server_8b_32b_curator_jul13th.sh
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

SWEEP_LOG="logs_debug_memory/_sweep_curator_8b32b_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN} no_serve=${NO_SERVE})"

# ---- shared env ----
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
EXEC_MODEL="openai/Qwen/Qwen3-32B"
CUR_MODEL="openai/Qwen/Qwen3-8B"
EXEC_PORTS=(8001 8002 8003)   # three 32B executor lanes
CUR_PORT=8004                 # single 8B curator, shared by all lanes

# ------------------------------------------------------------------ #
# vLLM server launch: 3x Qwen3-32B (exec) + 1x Qwen3-8B (curator), DP=2 each.
# ------------------------------------------------------------------ #
declare -A PORT_GPUS=( [8001]="0,1" [8002]="2,3" [8003]="4,5" [8004]="6,7" )
declare -A PORT_MODEL=( [8001]="Qwen/Qwen3-32B" [8002]="Qwen/Qwen3-32B" [8003]="Qwen/Qwen3-32B" [8004]="Qwen/Qwen3-8B" )
ALL_PORTS=(8001 8002 8003 8004)
VLLM_PIDS=()

launch_servers() {
  echo "[vllm] launching 3x Qwen3-32B (exec) + 1x Qwen3-8B (curator), DP=2 each"
  for port in "${ALL_PORTS[@]}"; do
    local gpus="${PORT_GPUS[$port]}" model="${PORT_MODEL[$port]}"
    echo "[vllm] :${port} ${model} on GPUs ${gpus} -> logs_vllm/vllm_${port}.log"
    CUDA_VISIBLE_DEVICES="$gpus" vllm serve "$model" \
        --port "$port" --served-model-name "$model" --dtype bfloat16 \
        --data-parallel-size 2 --tensor-parallel-size 1 \
        --max-model-len 40960 --gpu-memory-utilization 0.90 \
        > "logs_vllm/vllm_${port}.log" 2>&1 &
    VLLM_PIDS+=($!)
  done
}

wait_servers() {
  echo "[vllm] waiting for all servers to become ready (timeout 30m; 32B loads slower)..."
  local deadline=$(( $(date +%s) + 1800 ))
  for port in "${ALL_PORTS[@]}"; do
    until curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; do
      if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "ERROR: :${port} not ready before timeout — see logs_vllm/vllm_${port}.log"; exit 1
      fi
      sleep 10
    done
    echo "[vllm] :${port} (${PORT_MODEL[$port]}) READY"
  done
}

teardown_servers() {
  [ "${NO_TEARDOWN:-0}" = 1 ] && { echo "[vllm] NO_TEARDOWN=1 — leaving servers up"; return; }
  echo "[vllm] tearing down vLLM servers (pids: ${VLLM_PIDS[*]:-none})"
  for pid in "${VLLM_PIDS[@]:-}"; do [ -n "$pid" ] && kill "$pid" 2>/dev/null; done
}

# ---- one run: executor on $exec_port (32B), curator on $CUR_PORT (8B) ----
# args: exec_port  thinking  hist  rn  task_context  on_empty(0|1)  exp_name
run_exp() {
  local exec_port="$1" thinking="$2" hist="$3" rn="$4" tc="$5" oe="$6" exp="$7"
  exp="${exp}_${STAMP}"
  local oe_flag=""; [ "$oe" = 1 ] && oe_flag="--curator_on_empty"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (exec :${exec_port} 32B / curator :${CUR_PORT} 8B) ----"
    echo "  OPENAI_API_BASE=http://localhost:${exec_port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} \\"
    echo "  python -u ${RUNNER} --env alfworld --memory_type curator \\"
    echo "      --model ${EXEC_MODEL} --curation_model ${CUR_MODEL} \\"
    echo "      --curation_base_url http://localhost:${CUR_PORT}/v1 \\"
    echo "      --task_context ${tc} ${oe_flag} \\"
    echo "      --batch_size 10 --retrieve_num ${rn} --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > logs_debug_memory/${exp}.log 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  exec:${exec_port}(32B) cur:${CUR_PORT}(8B)  think=${thinking} hist=${hist} rn=${rn} tc=${tc} on_empty=${oe}"
  OPENAI_API_BASE="http://localhost:${exec_port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  python -u "$RUNNER" --env alfworld --memory_type curator \
      --model          "$EXEC_MODEL" \
      --curation_model "$CUR_MODEL" \
      --curation_base_url "http://localhost:${CUR_PORT}/v1" \
      --task_context "$tc" $oe_flag \
      --batch_size 10 --retrieve_num "$rn" --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> logs_debug_memory/${exp}.log"
}

# ==============================================================================
# Job matrix (cur32_ prefix; priority think+rn5+hist5 first). Same grid as 8b/8b.
# ==============================================================================
RN_LIST=(3 5)
TC_LIST=(short obs0)
OE_LIST=(1)

BASE=(
  "true|5|cur32_think_hist5_run1"
  "true|5|cur32_think_hist5_run2"
  "true|5|cur32_think_hist5_run3"
  "true|3|cur32_think_hist3_run1"
  "true|3|cur32_think_hist3_run2"
  "true|3|cur32_think_hist3_run3"
  "false|5|cur32_nonthink_hist5_run1"
  "false|5|cur32_nonthink_hist5_run2"
  "false|5|cur32_nonthink_hist5_run3"
  "false|3|cur32_nonthink_hist3_run1"
  "false|3|cur32_nonthink_hist3_run2"
  "false|3|cur32_nonthink_hist3_run3"
)

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
# portable sort+strip (no mapfile; macOS bash 3.2 lacks it and would silently no-op).
_SORTED=()
while IFS= read -r _line; do _SORTED+=("$_line"); done < <(
  printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1n | sed 's/^[01]|//')
JOBS=("${_SORTED[@]}")

QUEUE="logs_debug_memory/_curator_queue_8b32b.txt"
QLOCK="logs_debug_memory/_curator_queue_8b32b.lock"

pop_job() {
  local line=""
  exec 9>"$QLOCK"; flock 9
  if [ -s "$QUEUE" ]; then line=$(head -n1 "$QUEUE"); tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"; fi
  flock -u 9
  printf '%s' "$line"
}

# Each worker is pinned to ONE 32B executor port; all share the 8B curator on :8004.
worker() {
  local exec_port="$1" job
  while :; do
    job=$(pop_job); [ -z "$job" ] && break
    IFS='|' read -r thinking hist rn tc oe exp <<< "$job"
    run_exp "$exec_port" "$thinking" "$hist" "$rn" "$tc" "$oe" "$exp"
  done
  echo "[$(date +%H:%M:%S)] worker exec:${exec_port} — queue drained, exiting"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN ====================="
  echo "SERVERS: :8001/8002/8003 Qwen3-32B DP=2 (exec lanes) + :8004 Qwen3-8B DP=2 (shared curator)"
  echo "SHARED QUEUE (${#JOBS[@]} jobs) — 3 exec lanes drain in parallel; priority (think,rn5,hist5) first:"
  i=0; for j in "${JOBS[@]}"; do IFS='|' read -r t h rn tc oe e <<< "$j"; i=$((i+1)); printf '[%02d] ' "$i"; DRY_RUN=1 run_exp "${EXEC_PORTS[0]}" "$t" "$h" "$rn" "$tc" "$oe" "$e" | head -1; done
  echo "===================== DRY RUN complete — nothing executed ============="
  exit 0
fi

if [ "$NO_SERVE" = 0 ]; then
  launch_servers
  trap teardown_servers EXIT
fi
wait_servers

printf '%s\n' "${JOBS[@]}" > "$QUEUE"
: > "$QLOCK"

echo "[$(date +%H:%M:%S)] launching ${#EXEC_PORTS[@]} exec-lane workers (curator shared on :${CUR_PORT})"
PIDS=()
for port in "${EXEC_PORTS[@]}"; do worker "$port" & PIDS+=($!); done
wait "${PIDS[@]}"
echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS COMPLETE."
