#!/bin/bash
# Reasoning No-Memory Baseline Experiments
# Benchmarks: AIME24, AIME25, AMC23, GPQA Diamond
# Models: Qwen3-8B (port 8001), Qwen3-32B (port 8002)
# Each benchmark: 3 independent runs

export OPENAI_API_KEY="EMPTY"

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

PYTHON=/home/ouyangsiru/memory_env/bin/python
BASE_DIR=/home/ouyangsiru/MemP/ProcedureMem

cd "$BASE_DIR"

run_experiment() {
    local env="$1"
    local model_id="$2"
    local base_url="$3"
    local exp_name="$4"

    echo "======================================================"
    echo "START: env=$env  model=$model_id  exp=$exp_name"
    echo "Time: $(date)"
    echo "======================================================"

    OPENAI_API_BASE="$base_url" "$PYTHON" run_skillos_online.py \
        --env "$env" \
        --model "$model_id" \
        --batch_size 10 \
        --exp_name "$exp_name"

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

mkdir -p logs

# ---- Qwen3-8B ----
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        run_experiment "$env" "openai/Qwen/Qwen3-8B" "http://10.202.0.8:8001/v1" "baseline-8b-run${run}"
    done
done

# ---- Qwen3-32B ----
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        run_experiment "$env" "openai/Qwen/Qwen3-32B" "http://10.202.0.8:8002/v1" "baseline-32b-run${run}"
    done
done

echo "All reasoning experiments done. $(date)"
