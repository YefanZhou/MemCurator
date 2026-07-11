#!/bin/bash
#SBATCH --job-name=baseline_async
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=sfr-rl
#SBATCH --gres=gpu:8
#SBATCH --partition=ml.p5en.48xlarge
#SBATCH --chdir=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
#SBATCH --output=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/slurm_outputs/%j/log.out
#SBATCH --error=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/slurm_outputs/%j/err_log.out
#SBATCH --time=4-00:00:00

set -Eeuo pipefail

# =============================================================================
# Baseline pure-async + reviseReact eval @ EXECUTOR_TEMPERATURE=1.0
# Serves Qwen3-8B (3 runs each), tears down, then Qwen3-32B (3 runs each).
# Reference structure: /Users/yefan.zhou/Research/start_code/slurm.sh
# =============================================================================

# ---- Fixed configuration --------------------------------------------------
BASE_PATH=/fsx/home/yefan.zhou
REPO_DIR=${BASE_PATH}/mem-evolve/SkillCurator-main
EVAL_DIR=${REPO_DIR}/evaluation/agent_eval
PORT=8001

# Sampling hyperparameters (temperature bumped 0.6 -> 1.0)
EXECUTOR_TEMPERATURE=1.0
EXECUTOR_TOP_P=0.95
EXECUTOR_TOP_K=20
EXECUTOR_MAX_TOKENS=4096
CONCURRENCY=64

# Naming tag reflecting the sampling params (temp1.0_topp0.95_topk20_max4096)
TAG=jul10_temp1.0_0.95_20_4096

mkdir -p ${REPO_DIR}/slurm_outputs/${SLURM_JOB_ID}
mkdir -p ${EVAL_DIR}/logs_debug

echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Starting at: $(date)"
echo "Sampling: temp=${EXECUTOR_TEMPERATURE}, top_p=${EXECUTOR_TOP_P}, top_k=${EXECUTOR_TOP_K}, max_tokens=${EXECUTOR_MAX_TOKENS}"
echo "Naming tag: ${TAG}"

# ---- Conda environment ----------------------------------------------------
echo "Activating conda environment: memory"
source ${BASE_PATH}/miniconda3/etc/profile.d/conda.sh
conda activate memory
echo "✓ Environment activated: $(conda info --envs | grep '*')"

# ---- vLLM server lifecycle ------------------------------------------------
SERVER_PID=""

cleanup_server() {
    if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "  Stopping vLLM server (PID ${SERVER_PID}, graceful)..."
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        sleep 5
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "  Force killing vLLM server (PID ${SERVER_PID})..."
            kill -KILL "${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
    # Safety net: kill any stray vllm processes
    pkill -f "vllm serve" 2>/dev/null || true
    SERVER_PID=""
    # Give the GPUs a moment to free memory before the next serve
    sleep 10
    echo "  ✓ vLLM server stopped"
}

cleanup_all() {
    echo ""
    echo "[CLEANUP] Stopping vLLM server..."
    cleanup_server
    echo "[CLEANUP] Complete"
    echo "Job finished at: $(date)"
}
trap cleanup_all EXIT INT TERM

wait_for_server() {
    local port=$1
    local timeout=600  # 10 minutes (32B can be slow to load)
    local elapsed=0
    echo "  Waiting for server on port ${port}..."
    while [ ${elapsed} -lt ${timeout} ]; do
        if curl -s -f "http://localhost:${port}/health" >/dev/null 2>&1 || \
           curl -s -f "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "  ✓ Server on port ${port} is ready! (${elapsed}s)"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "    Still waiting for port ${port}... (${elapsed}s elapsed)"
        fi
    done
    echo "  ✗ Timeout waiting for server on port ${port}"
    return 1
}

serve_model() {
    local model_name=$1
    echo ""
    echo "Launching vLLM server for ${model_name} on port ${PORT}..."
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve "${model_name}" \
        --port ${PORT} \
        --served-model-name "${model_name}" \
        --dtype bfloat16 \
        --data-parallel-size 8 \
        --tensor-parallel-size 1 \
        --max-model-len 40960 \
        --gpu-memory-utilization 0.90 \
        2>&1 | tee "${EVAL_DIR}/vllm_logs/vllm_${PORT}_$(basename ${model_name})_${SLURM_JOB_ID}.log" &
    SERVER_PID=$!
    echo "  Started vLLM (PID ${SERVER_PID})"
    wait_for_server ${PORT}
}

# ---- Eval run helpers -----------------------------------------------------
# $1 = python entrypoint, $2 = model, $3 = exp_name (no run suffix)
run_eval() {
    local script=$1
    local model=$2
    local exp_name=$3

    cd "${EVAL_DIR}"
    echo ""
    echo ">>> ${script} | ${model} | ${exp_name}"
    EXECUTOR_TEMPERATURE=${EXECUTOR_TEMPERATURE} EXECUTOR_TOP_P=${EXECUTOR_TOP_P} \
    EXECUTOR_TOP_K=${EXECUTOR_TOP_K} EXECUTOR_MAX_TOKENS=${EXECUTOR_MAX_TOKENS} \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:${PORT}/v1 TMPDIR=$HOME/tmp \
    python -u "${script}" --env alfworld --memory_type none \
        --model "openai/${model}" --exp_name "${exp_name}" \
        --concurrency ${CONCURRENCY} \
        2>&1 | tee "${EVAL_DIR}/logs_debug/${exp_name}.log"
}

mkdir -p "${EVAL_DIR}/vllm_logs"

# =============================================================================
# STAGE 1: Qwen/Qwen3-8B
# =============================================================================
echo ""
echo "=================================================="
echo "STAGE 1: Qwen/Qwen3-8B"
echo "=================================================="
serve_model "Qwen/Qwen3-8B"

for i in 1 2 3; do
    run_eval run_unified_hyper_async_step0bug_fix.py \
        "Qwen/Qwen3-8B" \
        "baseline_pure_async_${TAG}_run${i}"

    run_eval run_unified_hyper_async_revise_react_step0bug_fix.py \
        "Qwen/Qwen3-8B" \
        "baseline_reviseReact_${TAG}_run${i}"
done

# Tear down the 8B server before serving 32B
echo ""
echo "STAGE 1 complete — cleaning up Qwen3-8B vLLM before 32B..."
cleanup_server

# =============================================================================
# STAGE 2: Qwen/Qwen3-32B
# =============================================================================
echo ""
echo "=================================================="
echo "STAGE 2: Qwen/Qwen3-32B"
echo "=================================================="
serve_model "Qwen/Qwen3-32B"

for i in 1 2 3; do
    run_eval run_unified_hyper_async_step0bug_fix.py \
        "Qwen/Qwen3-32B" \
        "baseline_pure_async_32b_${TAG}_run${i}"

    run_eval run_unified_hyper_async_revise_react_step0bug_fix.py \
        "Qwen/Qwen3-32B" \
        "baseline_reviseReact_32b_${TAG}_run${i}"
done

echo ""
echo "=================================================="
echo "ALL RUNS COMPLETE"
echo "=================================================="
echo "Logs in: ${EVAL_DIR}/logs_debug/*${TAG}*.log"
# Final cleanup handled by trap
