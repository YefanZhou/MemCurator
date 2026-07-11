#!/bin/bash
set -o pipefail
# SkillOS Reasoning Experiments — Fine-tuned curator
# Benchmarks  : AIME24, AIME25, GPQA Diamond
# Memory type : skillos
# Executor    : Qwen3-8B (port 8001) | Qwen3-32B (port 8002)
# Curator     : skillos-qwen3-8b-step50 (local vLLM, fine-tuned)
# Runs        : 3 independent runs per executor

export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
CURATOR_CKPT="/home/siruo_google_com/SkillCurator/converted_ckpt/skillos-qwen3-8b-step50"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local executor_model="$2"   # e.g. openai/Qwen/Qwen3-8B
    local executor_url="$3"
    local exp_name="$4"
    local batch_size="$5"

    echo "======================================================"
    echo "START: env=$env  executor=$executor_model  exp=$exp_name  bs=$batch_size"
    echo "Time: $(date)"
    echo "======================================================"

    OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py \
        --env               "$env" \
        --model             "$executor_model" \
        --memory_type       "skillos" \
        --curation_model    "$CURATOR_CKPT" \
        --batch_size        "$batch_size" \
        --retrieve_num      3 \
        --exp_name          "$exp_name"

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# SkillOS — Qwen3-8B executor
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" \
            "skillos-ft-8b-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_ft_8b_${env}_run${run}.log"
    done
done

# ============================================================
# SkillOS — Qwen3-32B executor
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" \
            "skillos-ft-32b-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_ft_32b_${env}_run${run}.log"
    done
done

echo "All skillos-ft reasoning experiments done. $(date)"
