#!/bin/bash
set -o pipefail
# ALFWorld experiments with WebShop-trained Qwen3-8B curator (cross-task transfer)
# Curator  : loaded locally via vLLM from converted checkpoint
# Executors: Qwen3-8B (port 8001) | Qwen3-32B (port 8002) | Gemini-2.5-Pro
# Memory   : skillos
# Games    : 140 (full dev split), 3 runs each

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
TRAINED_CURATOR="/home/siruo_google_com/SkillCurator/converted_models/qwen3-8b-webshop-skillos-step50"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local exp_name="$1"
    local executor_model="$2"
    local executor_url="$3"
    local run_id="$4"

    local full_exp="${exp_name}-run${run_id}"

    echo "======================================================"
    echo "START: $full_exp  executor=$executor_model"
    echo "Time: $(date)"
    echo "======================================================"

    local cmd_args=(
        --env            alfworld
        --model          "$executor_model"
        --memory_type    skillos
        --batch_size     10
        --max_steps      30
        --exp_name       "$full_exp"
        --overwrite
        --curation_model "$TRAINED_CURATOR"
        --retrieve_num   3
    )

    if [ -n "$executor_url" ]; then
        OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py "${cmd_args[@]}"
    else
        "$PYTHON" run_unified.py "${cmd_args[@]}"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $full_exp  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# 3 runs × 3 executors = 9 experiments
for run in 1 2 3; do
    run_experiment "skillos-wscur-8b" \
        "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "$run" \
        2>&1 | tee "logs/alfworld_wscur_8b_run${run}.log"

    run_experiment "skillos-wscur-32b" \
        "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "$run" \
        2>&1 | tee "logs/alfworld_wscur_32b_run${run}.log"

    run_experiment "skillos-wscur-gem" \
        "gemini/gemini-2.5-pro" "" "$run" \
        2>&1 | tee "logs/alfworld_wscur_gem_run${run}.log"
done

echo "All ALFWorld webshop-curator experiments done. $(date)"
