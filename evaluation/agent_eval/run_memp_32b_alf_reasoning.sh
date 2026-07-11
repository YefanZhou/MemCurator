#!/bin/bash
set -o pipefail
# MemP (SkillOS self-curated) with Qwen3-32B as both executor and curator
# Executor/Curator: openai/Qwen/Qwen3-32B @ port 8002
# ALFWorld : 140 games, retrieve_num=3, max_steps=30
# Reasoning: AIME24 (bs=3) | AIME25 (bs=3) | GPQA (bs=10), retrieve_num=3
# Runs     : 3 per benchmark

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.148.0.45:8002/v1"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
MODEL="openai/Qwen/Qwen3-32B"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local run_id="$2"
    local batch_size="$3"
    local retrieve="$4"
    local full_exp="memp-32b-run${run_id}"

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
        --max_steps      30 \
        --retrieve_num   "$retrieve" \
        --exp_name       "$full_exp" \
        --overwrite

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$full_exp  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ALFWorld — batch_size 10, retrieve_num 3
for run in 1 2 3; do
    run_experiment "alfworld" "$run" 10 3 \
        2>&1 | tee "logs/alfworld_memp_32b_run${run}.log"
done

# Reasoning — AIME24, AIME25, GPQA
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" "$run" "$bs" 3 \
            2>&1 | tee "logs/reasoning_memp_32b_${env}_run${run}.log"
    done
done

echo "All MemP 32b alfworld+reasoning experiments done. $(date)"
