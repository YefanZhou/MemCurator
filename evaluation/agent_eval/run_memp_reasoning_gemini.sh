#!/bin/bash
set -o pipefail
# MemP (SkillOS self-curated) on Reasoning with Gemini-2.5-Pro as both executor and curator
# Benchmarks: AIME24, AIME25, GPQA Diamond — 3 runs each

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
MODEL="gemini/gemini-2.5-pro"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local run_id="$2"
    local batch_size="$3"
    local full_exp="memp-gem-run${run_id}"

    echo "======================================================"
    echo "START: env=$env  $full_exp  executor=$MODEL  curator=$MODEL"
    echo "Time: $(date)"
    echo "======================================================"

    "$PYTHON" run_unified.py \
        --env            "$env" \
        --model          "$MODEL" \
        --memory_type    skillos \
        --curation_model "$MODEL" \
        --batch_size     "$batch_size" \
        --retrieve_num   3 \
        --exp_name       "$full_exp" \
        --overwrite

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$full_exp  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" "$run" "$bs" \
            2>&1 | tee "logs/reasoning_memp_gem_${env}_run${run}.log"
    done
done

echo "All MemP reasoning gemini experiments done. $(date)"
