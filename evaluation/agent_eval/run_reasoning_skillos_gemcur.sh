#!/bin/bash
set -o pipefail
# SkillOS Reasoning Experiments — Gemini 2.5 Pro as skill curator
# Benchmarks  : AIME24, AIME25, GPQA Diamond
# Memory type : skillos
# Curator     : gemini-2.5-pro via Vertex AI
# Executors   : Qwen3-8B (port 8001) | Qwen3-32B (port 8002) | Gemini-2.5-Pro (Vertex AI)
# Runs        : 3 independent runs per executor

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
CURATOR="gemini/gemini-2.5-pro"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local executor_model="$2"
    local executor_url="$3"     # empty string for Gemini executor
    local exp_name="$4"
    local batch_size="$5"

    echo "======================================================"
    echo "START: env=$env  executor=$executor_model  curator=$CURATOR  exp=$exp_name  bs=$batch_size"
    echo "Time: $(date)"
    echo "======================================================"

    if [ -n "$executor_url" ]; then
        OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py \
            --env               "$env" \
            --model             "$executor_model" \
            --memory_type       "skillos" \
            --curation_model    "$CURATOR" \
            --batch_size        "$batch_size" \
            --retrieve_num      3 \
            --exp_name          "$exp_name"
    else
        "$PYTHON" run_unified.py \
            --env               "$env" \
            --model             "$executor_model" \
            --memory_type       "skillos" \
            --curation_model    "$CURATOR" \
            --batch_size        "$batch_size" \
            --retrieve_num      3 \
            --exp_name          "$exp_name"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# SkillOS — Qwen3-8B executor, Gemini-2.5-Pro curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" \
            "skillos-gemcur-8b-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_gemcur_8b_${env}_run${run}.log"
    done
done

# ============================================================
# SkillOS — Qwen3-32B executor, Gemini-2.5-Pro curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" \
            "skillos-gemcur-32b-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_gemcur_32b_${env}_run${run}.log"
    done
done

# ============================================================
# SkillOS — Gemini-2.5-Pro executor, Gemini-2.5-Pro curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "gemini/gemini-2.5-pro" "" \
            "skillos-gemcur-gem-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_gemcur_gem_${env}_run${run}.log"
    done
done

echo "All skillos-gemcur reasoning experiments done. $(date)"
