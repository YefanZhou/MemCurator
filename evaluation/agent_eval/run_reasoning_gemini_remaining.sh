#!/bin/bash
set -o pipefail
# Missing Gemini-2.5-Pro executor experiments
# Covers:
#   (i)   No memory baseline — runs 2 & 3
#   (ii)  SkillOS, Qwen3-8B curator — runs 1, 2, 3
#   (iii) ReasoningBank, Qwen3-8B curator — runs 1, 2, 3
# Benchmarks: AIME24, AIME25, GPQA Diamond
# Executor  : gemini/gemini-2.5-pro (Vertex AI)
# Curator   : Qwen3-8B (http://10.148.0.45:8001/v1) for memory conditions

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
CURATOR_MODEL="openai/Qwen/Qwen3-8B"
CURATOR_URL="http://10.148.0.45:8001/v1"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local memory_type="$2"
    local exp_name="$3"
    local batch_size="$4"

    echo "======================================================"
    echo "START: env=$env  model=gemini-2.5-pro  memory=$memory_type  exp=$exp_name  bs=$batch_size"
    echo "Time: $(date)"
    echo "======================================================"

    if [ "$memory_type" = "none" ]; then
        "$PYTHON" run_unified.py \
            --env          "$env" \
            --model        "gemini/gemini-2.5-pro" \
            --memory_type  "none" \
            --batch_size   "$batch_size" \
            --exp_name     "$exp_name"
    else
        OPENAI_API_BASE="$CURATOR_URL" "$PYTHON" run_unified.py \
            --env               "$env" \
            --model             "gemini/gemini-2.5-pro" \
            --memory_type       "$memory_type" \
            --curation_model    "$CURATOR_MODEL" \
            --curation_base_url "$CURATOR_URL" \
            --batch_size        "$batch_size" \
            --retrieve_num      3 \
            --exp_name          "$exp_name"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$memory_type/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# (i) Baseline — runs 2 & 3 only (run 1 already complete)
# ============================================================
for run in 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "none" "baseline-gemini25pro-run${run}" "$bs" \
            2>&1 | tee "logs/baseline_gemini25pro_${env}_run${run}.log"
    done
done

# ============================================================
# (ii) SkillOS — Gemini executor, Qwen3-8B curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "skillos" "skillos-gemini25pro-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_gemini25pro_${env}_run${run}.log"
    done
done

# ============================================================
# (iii) ReasoningBank — Gemini executor, Qwen3-8B curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "reasoningbank" "rb-gemini25pro-run${run}" "$bs" \
            2>&1 | tee "logs/rb_gemini25pro_${env}_run${run}.log"
    done
done

echo "All remaining Gemini reasoning experiments done. $(date)"
