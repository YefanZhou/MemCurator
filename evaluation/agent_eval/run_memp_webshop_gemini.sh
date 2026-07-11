#!/bin/bash
set -o pipefail
# MemP (SkillOS self-curated) on WebShop with Gemini-2.5-Pro as both executor and curator
# Goals: 500 test, retrieve_num=5, max_steps=30, 3 runs

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
MODEL="gemini/gemini-2.5-pro"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local run_id="$1"
    local full_exp="memp-gem-run${run_id}"

    echo "======================================================"
    echo "START: $full_exp  executor=$MODEL  curator=$MODEL"
    echo "Time: $(date)"
    echo "======================================================"

    "$PYTHON" run_unified.py \
        --env            webshop \
        --model          "$MODEL" \
        --memory_type    skillos \
        --curation_model "$MODEL" \
        --batch_size     10 \
        --max_steps      30 \
        --retrieve_num   5 \
        --exp_name       "$full_exp" \
        --overwrite

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $full_exp  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

for run in 1 2 3; do
    run_experiment "$run" 2>&1 | tee "logs/webshop_memp_gem_run${run}.log"
done

echo "All MemP webshop gemini experiments done. $(date)"
