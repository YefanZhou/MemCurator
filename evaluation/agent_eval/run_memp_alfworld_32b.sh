#!/bin/bash
set -o pipefail
# MemP on ALFWorld with Qwen3-32B executor, Qwen3-8B curator (fixed)
# Executor: openai/Qwen/Qwen3-32B @ port 8002
# Curator : openai/Qwen/Qwen3-8B  @ port 8001
# Games   : 140 eval, batch=10, max_steps=30, retrieve_num=3, 3 runs

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.148.0.45:8002/v1"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="openai/Qwen/Qwen3-32B"
CURATOR="openai/Qwen/Qwen3-8B"
CURATOR_URL="http://10.148.0.45:8001/v1"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local run_id="$1"
    local full_exp="memp-32b-qwencur-run${run_id}"

    echo "======================================================"
    echo "START: $full_exp  executor=$EXECUTOR  curator=$CURATOR"
    echo "Time: $(date)"
    echo "======================================================"

    "$PYTHON" run_memp_online.py \
        --model         "$EXECUTOR" \
        --batch_size    10 \
        --max_steps     30 \
        --retrieve_num  3 \
        --use_memory \
        --overwrite \
        --exp_name      "$full_exp" \
        --mem_model     "$CURATOR" \
        --mem_base_url  "$CURATOR_URL"

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $full_exp  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

for run in 1 2 3; do
    run_experiment "$run" 2>&1 | tee "logs/alfworld_memp_32b_qwencur_run${run}.log"
done

echo "All MemP 32b alfworld experiments done. $(date)"
