#!/bin/bash
# Launch ONE vLLM server serving Qwen/Qwen3-8B with data-parallel-size=8.
# Single port 8001, using all 8 GPUs (dp=8, tp=1 => one replica per GPU behind one endpoint).
#
# Ctrl-C (or any exit) cleanly tears down the server AND its dp worker children,
# so no orphaned processes hold GPU memory.
#
# Usage:
#   bash launch_vllm_1port_dp8.sh
#   tail -f logs/vllm_8001.log      # follow the server log
#   (Ctrl-C to stop everything) dfdfdfsdfd df

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
MAX_MODEL_LEN=40960
GPU_MEM_UTIL=0.95
PORT=8001

LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"

# Timestamped log file so each launch keeps its own log (no overwrite across restarts).
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/vllm_${PORT}_${STAMP}.log"

# Track the process-group id of the launched server.
PGIDS=()

cleanup() {
  echo ""
  echo "Shutting down the vLLM server and its worker children..."
  for pgid in "${PGIDS[@]}"; do
    # Negative pid => signal the whole process group (parent + dp workers).
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done
  # Give them a moment to exit gracefully, then hard-kill any stragglers.
  sleep 5
  for pgid in "${PGIDS[@]}"; do
    kill -KILL -- "-$pgid" 2>/dev/null || true
  done
  echo "Done."
}
trap cleanup INT TERM EXIT

echo "Launching vLLM on port $PORT with all 8 GPUs (dp=8, tp=1)"
# setsid puts the server in its own process group so we can signal the
# entire tree (dp workers included) via the group id.
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" setsid vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --dtype bfloat16 \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  > "$LOG_FILE" 2>&1 &
# With setsid, the child's PID is the PGID of its new process group.
PGIDS+=("$!")

echo "Server launching in background on port $PORT (dp=8). Log: $LOG_FILE"
echo "Press Ctrl-C to stop the server and its workers."
wait
