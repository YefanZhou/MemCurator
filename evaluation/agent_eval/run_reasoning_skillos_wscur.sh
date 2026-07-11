#!/bin/bash
set -o pipefail
# SkillOS Reasoning Experiments — WebShop-trained curator (cross-task transfer)
# Benchmarks  : AIME24, AIME25, GPQA Diamond
# Memory type : skillos
# Executors   : Qwen3-8B (port 8001) | Qwen3-32B (port 8002) | Gemini-2.5-Pro
# Curator     : qwen3-8b-webshop-skillos-step50 (local vLLM)
# Runs        : 3 independent runs per executor

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
CURATOR_CKPT="/home/siruo_google_com/SkillCurator/converted_models/qwen3-8b-webshop-skillos-step50"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local executor_model="$2"
    local executor_url="$3"
    local exp_name="$4"
    local batch_size="$5"

    echo "======================================================"
    echo "START: env=$env  executor=$executor_model  exp=$exp_name  bs=$batch_size"
    echo "Time: $(date)"
    echo "======================================================"

    local cmd_args=(
        --env            "$env"
        --model          "$executor_model"
        --memory_type    skillos
        --curation_model "$CURATOR_CKPT"
        --batch_size     "$batch_size"
        --retrieve_num   3
        --exp_name       "$exp_name"
        --overwrite
    )

    if [ -n "$executor_url" ]; then
        OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py "${cmd_args[@]}"
    else
        "$PYTHON" run_unified.py "${cmd_args[@]}"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# Qwen3-8B executor
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" \
            "skillos-wscur-8b-run${run}" "$bs" \
            2>&1 | tee "logs/reasoning_wscur_8b_${env}_run${run}.log"
    done
done

# Qwen3-32B executor
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" \
            "skillos-wscur-32b-run${run}" "$bs" \
            2>&1 | tee "logs/reasoning_wscur_32b_${env}_run${run}.log"
    done
done

# Gemini-2.5-Pro executor
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "gemini/gemini-2.5-pro" "" \
            "skillos-wscur-gem-run${run}" "$bs" \
            2>&1 | tee "logs/reasoning_wscur_gem_${env}_run${run}.log"
    done
done

echo "All skillos-wscur reasoning experiments done. $(date)"
