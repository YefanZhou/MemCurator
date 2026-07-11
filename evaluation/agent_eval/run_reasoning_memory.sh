#!/bin/bash
set -o pipefail   # propagate Python exit code through | tee
# Reasoning Memory Experiments
# Benchmarks : AIME24, AIME25, GPQA Diamond
# Memory types: skillos, reasoningbank
# Executor models: Qwen3-8B (port 8001), Qwen3-32B (port 8002)
# Curator model  : Qwen3-8B (always)

export OPENAI_API_KEY="EMPTY"

PYTHON=/home/ouyangsiru/memory_env/bin/python
BASE_DIR=/home/ouyangsiru/MemP/ProcedureMem

# Both SkillOS and ReasoningBank use Qwen3-8B for memory/skill extraction via vLLM
CURATOR="Qwen/Qwen3-8B"

cd "$BASE_DIR"

run_experiment() {
    local env="$1"
    local model_id="$2"
    local base_url="$3"
    local memory_type="$4"
    local exp_name="$5"
    local batch_size="$6"

    echo "======================================================"
    echo "START: env=$env  model=$model_id  memory=$memory_type  exp=$exp_name  bs=$batch_size"
    echo "Time: $(date)"
    echo "======================================================"

    OPENAI_API_BASE="$base_url" "$PYTHON" run_unified.py \
        --env                "$env" \
        --model              "$model_id" \
        --memory_type        "$memory_type" \
        --curation_model     "openai/$CURATOR" \
        --curation_base_url  "http://10.202.0.8:8001/v1" \
        --batch_size         "$batch_size" \
        --retrieve_num       3 \
        --exp_name           "$exp_name"

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$memory_type/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

mkdir -p logs

# batch size per dataset
bs_aime24=3
bs_aime25=3
bs_gpqa=10

# ============================================================
# SkillOS — Qwen3-8B executor
# ============================================================
# for run in 1 2 3; do
#     for env in aime24 aime25 gpqa; do
#         bs_var="bs_${env}"; bs="${!bs_var}"
#         run_experiment "$env" \
#             "openai/Qwen/Qwen3-8B" "http://10.202.0.8:8001/v1" \
#             "skillos" "skillos-8b-run${run}" "$bs" \
#             2>&1 | tee "logs/skillos_8b_${env}_run${run}.log"
#     done
# done

# ============================================================
# SkillOS — Qwen3-32B executor
# ============================================================
# for run in 1 2 3; do
#     for env in aime24 aime25 gpqa; do
#         bs_var="bs_${env}"; bs="${!bs_var}"
#         run_experiment "$env" \
#             "openai/Qwen/Qwen3-32B" "http://10.202.0.8:8002/v1" \
#             "skillos" "skillos-32b-run${run}" "$bs" \
#             2>&1 | tee "logs/skillos_32b_${env}_run${run}.log"
#     done
# done

# ============================================================
# ReasoningBank — Qwen3-8B executor
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-8B" "http://10.202.0.8:8001/v1" \
            "reasoningbank" "rb-8b-run${run}" "$bs" \
            2>&1 | tee "logs/rb_8b_${env}_run${run}.log"
    done
done

# ============================================================
# ReasoningBank — Qwen3-32B executor
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "openai/Qwen/Qwen3-32B" "http://10.202.0.8:8002/v1" \
            "reasoningbank" "rb-32b-run${run}" "$bs" \
            2>&1 | tee "logs/rb_32b_${env}_run${run}.log"
    done
done

echo "All reasoning memory experiments done. $(date)"
