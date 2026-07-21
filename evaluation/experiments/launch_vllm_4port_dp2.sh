#!/bin/bash
# Launch 4 vLLM servers, each serving Qwen/Qwen3-8B with data-parallel-size=2.
# Ports 8001-8004, using 8 GPUs total (2 GPUs per server).
#
# Ctrl-C (or any exit) cleanly tears down every server AND its dp worker
# children, so no orphaned processes hold GPU memory.
#
# Usage:
#   bash launch_vllm_4port_dp2.sh
#   tail -f logs/vllm_800*.log     # follow all server logs
#   (Ctrl-C to stop everything)

set -euo pipefail

MODEL="Qwen/Qwen3-8B"
MAX_MODEL_LEN=40960
GPU_MEM_UTIL=0.95

LOG_DIR="$(dirname "$0")/logs"
mkdir -p "$LOG_DIR"

# Map each port to its pair of GPUs (dp=2 => 2 GPUs each).
declare -A PORT_GPUS=(
  [8001]="0,1"
  [8002]="2,3"
  [8003]="4,5"
  [8004]="6,7"
)

# Track the process-group id of each launched server.
PGIDS=()

cleanup() {
  echo ""
  echo "Shutting down all vLLM servers and their worker children..."
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

for PORT in 8001 8002 8003 8004; do
  GPUS="${PORT_GPUS[$PORT]}"
  echo "Launching vLLM on port $PORT with GPUs $GPUS (dp=2)"
  # setsid puts each server in its own process group so we can signal the
  # entire tree (dp workers included) via the group id.
  CUDA_VISIBLE_DEVICES="$GPUS" setsid vllm serve "$MODEL" \
    --port "$PORT" \
    --served-model-name "$MODEL" \
    --dtype bfloat16 \
    --data-parallel-size 2 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    > "$LOG_DIR/vllm_${PORT}.log" 2>&1 &
  # With setsid, the child's PID is the PGID of its new process group.
  PGIDS+=("$!")
done

echo "All 4 servers launching in background. Logs in $LOG_DIR/"
echo "Press Ctrl-C to stop all servers and their workers."
wait
