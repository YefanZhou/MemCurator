#!/bin/bash
set -o pipefail
# Reasoning (math) trained Qwen3-8B curator -> ALFWorld + WebShop cross-task transfer
# Curator  : loaded locally via vLLM from converted math-trained checkpoint
# Executors: Qwen3-8B (port 8001) | Qwen3-32B (port 8002) | Gemini-2.5-Pro
# Memory   : skillos
# ALFWorld : 140 games, retrieve_num=3, max_steps=30
# WebShop  : 500 goals, retrieve_num=5, max_steps=30
# Runs     : 3 per executor per env

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
MATH_CURATOR="/home/siruo_google_com/SkillCurator/converted_models/qwen3-8b-math-skillos-step50"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local env="$1"
    local exp_name="$2"
    local executor_model="$3"
    local executor_url="$4"
    local run_id="$5"
    local retrieve="$6"

    local full_exp="${exp_name}-run${run_id}"

    echo "======================================================"
    echo "START: env=$env  $full_exp  executor=$executor_model"
    echo "Time: $(date)"
    echo "======================================================"

    local cmd_args=(
        --env            "$env"
        --model          "$executor_model"
        --memory_type    skillos
        --batch_size     10
        --max_steps      30
        --exp_name       "$full_exp"
        --overwrite
        --curation_model "$MATH_CURATOR"
        --retrieve_num   "$retrieve"
    )

    if [ -n "$executor_url" ]; then
        OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py "${cmd_args[@]}"
    else
        "$PYTHON" run_unified.py "${cmd_args[@]}"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$full_exp  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# ALFWorld (140 games, retrieve_num=3) — fastest first
# ============================================================
for run in 1 2 3; do
    run_experiment "alfworld" "skillos-mathcur-8b" \
        "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "$run" 3 \
        2>&1 | tee "logs/alfworld_mathcur_8b_run${run}.log"

    run_experiment "alfworld" "skillos-mathcur-32b" \
        "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "$run" 3 \
        2>&1 | tee "logs/alfworld_mathcur_32b_run${run}.log"

    run_experiment "alfworld" "skillos-mathcur-gem" \
        "gemini/gemini-2.5-pro" "" "$run" 3 \
        2>&1 | tee "logs/alfworld_mathcur_gem_run${run}.log"
done

# ============================================================
# WebShop (500 goals, retrieve_num=5)
# ============================================================
for run in 1 2 3; do
    run_experiment "webshop" "skillos-mathcur-8b" \
        "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "$run" 5 \
        2>&1 | tee "logs/webshop_mathcur_8b_run${run}.log"

    run_experiment "webshop" "skillos-mathcur-32b" \
        "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "$run" 5 \
        2>&1 | tee "logs/webshop_mathcur_32b_run${run}.log"

    run_experiment "webshop" "skillos-mathcur-gem" \
        "gemini/gemini-2.5-pro" "" "$run" 5 \
        2>&1 | tee "logs/webshop_mathcur_gem_run${run}.log"
done

echo "All math-curator alfworld+webshop experiments done. $(date)"
