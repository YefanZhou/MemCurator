#!/bin/bash
set -o pipefail
# ALFWorld experiments — Gemini-2.5-Pro as skill curator
# Executors : Qwen3-8B (port 8001) | Qwen3-32B (port 8002) | Gemini-2.5-Pro
# Curator   : gemini/gemini-2.5-pro via Vertex AI
# Memory    : skillos
# Games     : 140 (full dev split)

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
CURATOR="gemini/gemini-2.5-pro"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local exp_name="$1"
    local executor_model="$2"
    local executor_url="$3"   # empty string for Gemini executor

    echo "======================================================"
    echo "START: exp=$exp_name  executor=$executor_model  curator=$CURATOR"
    echo "Time: $(date)"
    echo "======================================================"

    if [ -n "$executor_url" ]; then
        OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py \
            --env            alfworld \
            --model          "$executor_model" \
            --memory_type    skillos \
            --curation_model "$CURATOR" \
            --batch_size     10 \
            --max_steps      30 \
            --retrieve_num   3 \
            --exp_name       "$exp_name"
    else
        "$PYTHON" run_unified.py \
            --env            alfworld \
            --model          "$executor_model" \
            --memory_type    skillos \
            --curation_model "$CURATOR" \
            --batch_size     10 \
            --max_steps      30 \
            --retrieve_num   3 \
            --exp_name       "$exp_name"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# Qwen3-8B executor, Gemini curator
# ============================================================
run_experiment "skillos-gemcur-8b" \
    "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" \
    2>&1 | tee "logs/alfworld_gemcur_8b.log"

# ============================================================
# Qwen3-32B executor, Gemini curator
# ============================================================
run_experiment "skillos-gemcur-32b" \
    "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" \
    2>&1 | tee "logs/alfworld_gemcur_32b.log"

# ============================================================
# Gemini-2.5-Pro executor, Gemini curator
# ============================================================
run_experiment "skillos-gemcur-gem" \
    "gemini/gemini-2.5-pro" "" \
    2>&1 | tee "logs/alfworld_gemcur_gem.log"

echo "All alfworld gemcur experiments done. $(date)"
