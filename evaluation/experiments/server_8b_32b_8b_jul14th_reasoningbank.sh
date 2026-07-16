#!/bin/bash
# ==============================================================================
# reasoningbank sweep, jul-14 — Qwen3-32B EXECUTOR + Qwen3-8B CURATOR.
# The 32B-executor counterpart of server_8b_8b_8b_jul12th.sh's headline configs
# (8B-executor think hist5 = 64.05, hist3 = 56.90). Same env, same runner, same
# grid — ONLY the executor model + the exec/curator server split differ.
#
# 12 runs, all reasoningbank + revise_react, async (think{2} x hist{2} x seed{3}):
#   think    hist5 run1/2/3   (headline; run first)
#   think    hist3 run1/2/3
#   nonthink hist5 run1/2/3
#   nonthink hist3 run1/2/3
#
# TOPOLOGY (mirrors server_8b_32b_curator_jul13th.sh; the node ip-10-0-140-50
# ALREADY serves exactly this, so default is --no-serve):
#   :8001 GPUs 0,1  Qwen3-32B  -> EXECUTOR lane 1
#   :8002 GPUs 2,3  Qwen3-32B  -> EXECUTOR lane 2
#   :8003 GPUs 4,5  Qwen3-32B  -> EXECUTOR lane 3
#   :8004 GPUs 6,7  Qwen3-8B   -> SHARED CURATOR (all runs curate here)
# Three 32B executor lanes drain a shared flock-guarded queue in parallel; every
# run's curator points at the single 8B server on :8004.
#
# Env is IDENTICAL to jul-12 reasoningbank (executor temp 1.0; curator temp 1.0,
# nonthinking, max 1024) so results are directly comparable — only the executor
# model changes (8B -> 32B). exp_name prefix rb32- distinguishes from the 8B runs.
#
# Usage:
#   bash server_8b_32b_8b_jul14th.sh --dry-run         # print servers + 6 jobs, run nothing
#   bash server_8b_32b_8b_jul14th.sh                   # DEFAULT: use existing servers (--no-serve)
#   bash server_8b_32b_8b_jul14th.sh --serve           # also launch the 4 vLLM servers here
#   NO_TEARDOWN=1 bash ... --serve                     # ...and leave them up afterwards
#   tmux new -s rb32 'bash .../server_8b_32b_8b_jul14th.sh'
#   STAMP=jul14b bash ...
# ==============================================================================
set -u

DRY_RUN=0
NO_SERVE=1            # DEFAULT: servers already up on the node -> do NOT launch
for a in "$@"; do
  [ "$a" = "--dry-run" ] && DRY_RUN=1
  [ "$a" = "--no-serve" ] && NO_SERVE=1
  [ "$a" = "--serve" ] && NO_SERVE=0
done

cd "$(dirname "$0")/../agent_eval" || { echo "cannot cd to agent_eval"; exit 1; }
mkdir -p logs_debug_memory logs_vllm

SWEEP_LOG="logs_debug_memory/_sweep_rb_8b32b_$(date +%Y%m%d_%H%M%S).log"
[ "$DRY_RUN" = 0 ] && exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "[sweep] scheduler log: ${SWEEP_LOG}  (dry_run=${DRY_RUN} no_serve=${NO_SERVE})"

# ---- shared env — IDENTICAL to server_8b_8b_8b_jul12th.sh (per-run: OPENAI_API_BASE /
#      HISTORY_LENGTH / ENABLE_THINKING — the last is set per-job from the matrix) ----
export ALFWORLD_DATA="$HOME/.cache/alfworld"
export OPENAI_API_KEY="EMPTY"
export TMPDIR="$HOME/tmp"
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false
export PROMPT_STYLE=revise_react
export SAVE_RAW=10 PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000

STAMP="${STAMP:-$(date +%Y%m%d_%H%M)}"
echo "[sweep] run stamp: ${STAMP}"

RUNNER="run_unified_dev_async.py"
EXEC_MODEL="openai/Qwen/Qwen3-32B"
CUR_MODEL="openai/Qwen/Qwen3-8B"
EXEC_PORTS=(8001 8002 8003)   # three 32B executor lanes
CUR_PORT=8004                 # single 8B curator, shared by all lanes

# ------------------------------------------------------------------ #
# vLLM server launch (only with --serve): 3x Qwen3-32B + 1x Qwen3-8B, DP=2 each.
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

# ---- preflight when using existing servers (--no-serve): verify the topology is live ----
preflight_servers() {
  for port in "${EXEC_PORTS[@]}"; do
    curl -s --max-time 5 "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "Qwen3-32B" \
      || { echo "ERROR: :${port} not serving Qwen3-32B (executor). Start servers or pass --serve."; exit 1; }
  done
  curl -s --max-time 5 "http://localhost:${CUR_PORT}/v1/models" 2>/dev/null | grep -q "Qwen3-8B" \
    || { echo "ERROR: :${CUR_PORT} not serving Qwen3-8B (curator). Start servers or pass --serve."; exit 1; }
  echo "[preflight] :8001/8002/8003 Qwen3-32B (exec) + :${CUR_PORT} Qwen3-8B (curator) OK"
}

# ---- one run: reasoningbank; executor on $exec_port (32B), curator on $CUR_PORT (8B) ----
# args: exec_port  thinking  hist  exp_name
run_exp() {
  local exec_port="$1" thinking="$2" hist="$3" exp="$4"
  exp="${exp}_${STAMP}"
  local memdir="Alfworld/memory/reasoningbank_${exp}"

  if [ "$DRY_RUN" = 1 ]; then
    echo "---- ${exp}  (exec :${exec_port} 32B / curator :${CUR_PORT} 8B) ----"
    echo "  rm -rf ${memdir}"
    echo "  OPENAI_API_BASE=http://localhost:${exec_port}/v1 HISTORY_LENGTH=${hist} ENABLE_THINKING=${thinking} \\"
    echo "  python -u ${RUNNER} --env alfworld --memory_type reasoningbank \\"
    echo "      --model ${EXEC_MODEL} --curation_model ${CUR_MODEL} \\"
    echo "      --curation_base_url http://localhost:${CUR_PORT}/v1 \\"
    echo "      --batch_size 10 --retrieve_num 5 --max_steps 30 --exp_name ${exp} --overwrite \\"
    echo "      > logs_debug_memory/${exp}.log 2>&1"
    return 0
  fi

  echo "[$(date +%H:%M:%S)] START ${exp}  exec:${exec_port}(32B) cur:${CUR_PORT}(8B)  think=${thinking} hist=${hist}"
  # Fresh memory to match --overwrite (results cleared; memory must be too).
  rm -rf "$memdir"
  OPENAI_API_BASE="http://localhost:${exec_port}/v1" HISTORY_LENGTH="$hist" ENABLE_THINKING="$thinking" \
  python -u "$RUNNER" --env alfworld --memory_type reasoningbank \
      --model          "$EXEC_MODEL" \
      --curation_model "$CUR_MODEL" \
      --curation_base_url "http://localhost:${CUR_PORT}/v1" \
      --batch_size 10 --retrieve_num 5 --max_steps 30 \
      --exp_name "$exp" --overwrite \
      > "logs_debug_memory/${exp}.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  ${exp}  (exit $?)  -> logs_debug_memory/${exp}.log"
}

# ==============================================================================
# Job matrix — 12 reasoningbank runs: think{2} x hist{2} x seed{3}.
# Priority: thinking first (headline), then nonthinking; hist5 before hist3 within each.
# Job spec: thinking|hist|exp_name
 # "true|5|rb32-async_think_hist5_run1"
 # "true|5|rb32-async_think_hist5_run2"
 # "true|5|rb32-async_think_hist5_run3"
 # "true|3|rb32-async_think_hist3_run1"
 # "true|3|rb32-async_think_hist3_run2"
 # "true|3|rb32-async_think_hist3_run3"
# ==============================================================================
BASE=(
  "false|5|rb32-async_nonthink_hist5_run1"
  "false|5|rb32-async_nonthink_hist5_run2"
  "false|5|rb32-async_nonthink_hist5_run3"
  "false|3|rb32-async_nonthink_hist3_run1"
  "false|3|rb32-async_nonthink_hist3_run2"
  "false|3|rb32-async_nonthink_hist3_run3"
)

# prio: think+hist5 (0, headline) -> think+hist3 (1) -> nonthink+hist5 (2) -> nonthink+hist3 (3).
# Stable sort keeps run1/2/3 order within each cell.
JOBS=()
for base in "${BASE[@]}"; do
  IFS='|' read -r t h e <<< "$base"
  if [ "$t" = true ]; then prio=0; else prio=2; fi
  [ "$h" = 3 ] && prio=$((prio + 1))
  JOBS+=("${prio}|${t}|${h}|${e}")
done
_SORTED=()
while IFS= read -r _line; do _SORTED+=("$_line"); done < <(
  printf '%s\n' "${JOBS[@]}" | sort -s -t'|' -k1,1n | sed 's/^[0-9]|//')
JOBS=("${_SORTED[@]}")

QUEUE="logs_debug_memory/_rb_queue_8b32b.txt"
QLOCK="logs_debug_memory/_rb_queue_8b32b.lock"

pop_job() {
  local line=""
  exec 9>"$QLOCK"; flock 9
  if [ -s "$QUEUE" ]; then line=$(head -n1 "$QUEUE"); tail -n +2 "$QUEUE" > "${QUEUE}.tmp" && mv "${QUEUE}.tmp" "$QUEUE"; fi
  flock -u 9
  printf '%s' "$line"
}

# Each worker pinned to ONE 32B executor port; all share the 8B curator on :8004.
worker() {
  local exec_port="$1" job
  while :; do
    job=$(pop_job); [ -z "$job" ] && break
    IFS='|' read -r thinking hist exp <<< "$job"
    run_exp "$exec_port" "$thinking" "$hist" "$exp"
  done
  echo "[$(date +%H:%M:%S)] worker exec:${exec_port} — queue drained, exiting"
}

if [ "$DRY_RUN" = 1 ]; then
  echo "===================== DRY RUN ====================="
  echo "SERVERS (--serve to launch; else use existing): :8001/8002/8003 Qwen3-32B DP=2 (exec) + :8004 Qwen3-8B DP=2 (curator)"
  echo "SHARED QUEUE (${#JOBS[@]} jobs) — 3 exec lanes drain in parallel; think first, hist5 before hist3:"
  i=0; for j in "${JOBS[@]}"; do IFS='|' read -r t h e <<< "$j"; i=$((i+1)); printf '[%02d] ' "$i"; DRY_RUN=1 run_exp "${EXEC_PORTS[0]}" "$t" "$h" "$e" | head -1; done
  echo "===================== DRY RUN complete — nothing executed ============="
  exit 0
fi

# ---- servers: launch (--serve) or verify existing (default --no-serve) ----
if [ "$NO_SERVE" = 0 ]; then
  launch_servers
  trap teardown_servers EXIT
  wait_servers
else
  preflight_servers
fi

printf '%s\n' "${JOBS[@]}" > "$QUEUE"
: > "$QLOCK"

echo "[$(date +%H:%M:%S)] launching ${#EXEC_PORTS[@]} exec-lane workers (curator shared on :${CUR_PORT})"
PIDS=()
for port in "${EXEC_PORTS[@]}"; do worker "$port" & PIDS+=($!); done
wait "${PIDS[@]}"
echo "[$(date +%H:%M:%S)] ALL ${#JOBS[@]} RUNS COMPLETE."
