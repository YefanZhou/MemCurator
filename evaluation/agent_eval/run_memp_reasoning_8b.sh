#!/bin/bash
set -o pipefail
# MemP on reasoning with Qwen3-8B as executor and curator
# Parallel to run_memp_all_8b.sh (which does alfworld first then reasoning)
# Uses different exp_name suffix to avoid collisions

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.148.0.45:8001/v1"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="openai/Qwen/Qwen3-8B"
CURATOR="openai/Qwen/Qwen3-8B"
CURATOR_URL="http://10.148.0.45:8001/v1"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_reasoning() {
    local env="$1"; local run_id="$2"; local bs="$3"
    local full_exp="memp-8b-qwencur-par-run${run_id}"
    echo "=== START: $env $full_exp  $(date) ==="
    "$PYTHON" run_memp_reasoning.py \
        --env           "$env" \
        --model         "$EXECUTOR" \
        --batch_size    "$bs" \
        --retrieve_num  3 \
        --exp_name      "$full_exp" \
        --overwrite \
        --mem_model     "$CURATOR" \
        --mem_base_url  "$CURATOR_URL"
    echo "=== DONE: $env $full_exp  exit=$?  $(date) ==="
}

for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_reasoning "$env" "$run" "$bs" 2>&1 | tee "logs/reasoning_memp_8b_par_${env}_run${run}.log"
    done
done

echo "All MemP 8B reasoning (parallel) done. $(date)"
