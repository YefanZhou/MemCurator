#!/bin/bash
# curator_sweep_monitor.sh — one-shot health snapshot of a curator sweep on THIS box.
# Run it on the box (box1 or box2); it auto-detects which curator sweep is running here
# and reports: vLLM server readiness, sweep/runner liveness, run progress (DONE count,
# newest result activity + staleness), and recent errors. Exit code 0 = healthy,
# 1 = warning (stall / server down), 2 = sweep not running.
#
# Designed to be polled in a loop (from a laptop):
#   watch:  ssh -J w2 <ip> 'bash .../curator_sweep_monitor.sh'
# or via the repo's /loop skill. Pass STALL_MIN to change the stall threshold (default 25).
set -u

AE=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval
LOGDIR="$AE/logs_debug_memory"
STALL_MIN="${STALL_MIN:-25}"          # minutes without newer result file => STALL warning
HOST=$(hostname)
NOW=$(date +%s)
echo "===================================================================="
echo "curator monitor @ ${HOST}  $(date '+%F %T')"
echo "===================================================================="

# --- which curator sweep script is running here? ---
SWEEP_PROC=$(pgrep -af 'server_8b_(8b|32b)_curator_jul13th.sh' | grep -v grep | head -1)
if [ -z "$SWEEP_PROC" ]; then
  echo "STATUS: NO curator sweep script running on ${HOST}."
  echo "  (runner procs still alive: $(pgrep -fc run_unified_dev_async_curator 2>/dev/null || echo 0))"
  exit 2
fi
SCRIPT=$(echo "$SWEEP_PROC" | grep -oE 'server_8b_(8b|32b)_curator_jul13th.sh')
case "$SCRIPT" in
  *8b_8b*)  KIND="8b/8b (4x Qwen3-8B)"; PORTS="8001 8002 8003 8004"; EXP_GLOB="cur_*";   SWEEP_LOG_GLOB="_sweep_curator_8b8b_*.log";;
  *8b_32b*) KIND="8b/32b (3x32B exec + 1x8B cur)"; PORTS="8001 8002 8003 8004"; EXP_GLOB="cur32_*"; SWEEP_LOG_GLOB="_sweep_curator_8b32b_*.log";;
esac
echo "SWEEP: ${SCRIPT}  [${KIND}]"

RC=0

# --- vLLM servers ---
echo "-- vLLM servers --"
for p in $PORTS; do
  if curl -sf "http://localhost:${p}/v1/models" >/dev/null 2>&1; then
    model=$(curl -s "http://localhost:${p}/v1/models" 2>/dev/null | grep -oE 'Qwen3-[0-9]+B' | head -1)
    echo "  :${p} UP (${model:-?})"
  else
    # not-ready could be 'still loading' — check if the vllm proc/log is alive
    if pgrep -f "vllm serve.*port ${p}" >/dev/null 2>&1 || [ -f "$AE/logs_vllm/vllm_${p}.log" ]; then
      echo "  :${p} DOWN/LOADING (proc present; see logs_vllm/vllm_${p}.log)"
    else
      echo "  :${p} DOWN (no proc) !!"; RC=1
    fi
  fi
done

# --- sweep + runner liveness ---
echo "-- processes --"
echo "  sweep workers : $(pgrep -fc 'server_8b_(8b|32b)_curator_jul13th.sh' 2>/dev/null || echo 0)"
echo "  runner procs  : $(pgrep -fc run_unified_dev_async_curator 2>/dev/null || echo 0)"
echo "  vllm serve    : $(pgrep -fc 'vllm serve' 2>/dev/null || echo 0)"

# --- GPU snapshot ---
echo "-- GPUs (util% / memMiB) --"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
  | awk -F', ' '{printf "  gpu%s %s %s\n",$1,$2,$3}' | paste -sd'  ' - 2>/dev/null || echo "  (nvidia-smi unavailable)"

# --- progress: completed result folders + newest activity ---
echo "-- progress --"
# each run == one result folder dev_<exp>_..._curator with idx_*.json; a run is "done"
# when its run.log contains 'ALL ... RUNS COMPLETE' is sweep-level; per-run we count folders
RES_ROOT="$AE/Alfworld/results"
n_folders=$(find "$RES_ROOT" -maxdepth 3 -type d -name "dev_${EXP_GLOB}_curator" 2>/dev/null | wc -l | tr -d ' ')
echo "  result folders for this sweep : ${n_folders}"
# DONE lines in the sweep scheduler log
SWEEP_LOG=$(ls -t $LOGDIR/$SWEEP_LOG_GLOB 2>/dev/null | head -1)
if [ -n "$SWEEP_LOG" ]; then
  done_n=$(grep -c 'DONE ' "$SWEEP_LOG" 2>/dev/null || echo 0)
  start_n=$(grep -c 'START ' "$SWEEP_LOG" 2>/dev/null || echo 0)
  echo "  scheduler log : ${SWEEP_LOG##*/}  (STARTed=${start_n}  DONE=${done_n})"
  if grep -q 'ALL .* RUNS COMPLETE' "$SWEEP_LOG" 2>/dev/null; then
    echo "  >>> SWEEP COMPLETE <<<"
  fi
fi

# --- staleness: newest idx_*.json mtime across this sweep's result folders ---
newest=$(find "$RES_ROOT" -maxdepth 4 -path "*dev_${EXP_GLOB}_curator/idx_*.json" -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
if [ -n "$newest" ]; then
  age_min=$(( (NOW - newest) / 60 ))
  echo "  newest result written ${age_min} min ago"
  if [ "$age_min" -gt "$STALL_MIN" ] && ! grep -q 'ALL .* RUNS COMPLETE' "${SWEEP_LOG:-/dev/null}" 2>/dev/null; then
    echo "  !! STALL WARNING: no new result in >${STALL_MIN} min (and sweep not marked complete)"; RC=1
  fi
else
  echo "  (no idx_*.json yet — early startup or first run in progress)"
fi

# --- recent errors in the newest per-run log ---
echo "-- recent errors (newest run log) --"
NEWEST_RUNLOG=$(ls -t $LOGDIR/${EXP_GLOB}.log 2>/dev/null | head -1)
if [ -n "$NEWEST_RUNLOG" ]; then
  echo "  ${NEWEST_RUNLOG##*/}:"
  errs=$(grep -iE 'Traceback|Error|Exception|CUDA|refused|OOM|out of memory' "$NEWEST_RUNLOG" 2>/dev/null | grep -viE 'unraisablehook|Exception ignored' | tail -4)
  if [ -n "$errs" ]; then echo "$errs" | sed 's/^/    /'; RC=1; else echo "    (no errors)"; fi
else
  echo "  (no per-run logs yet)"
fi

echo "-- verdict --"
case $RC in
  0) echo "  HEALTHY";;
  1) echo "  WARNING (see !! lines above)";;
esac
exit $RC
