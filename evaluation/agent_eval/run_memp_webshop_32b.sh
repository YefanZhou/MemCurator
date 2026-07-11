#!/bin/bash
set -o pipefail
# MemP on WebShop — Qwen3-32B executor + Qwen3-8B curator, 3 runs

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.148.0.45:8002/v1"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="openai/Qwen/Qwen3-32B"
CURATOR="openai/Qwen/Qwen3-8B"
CURATOR_URL="http://10.148.0.45:8001/v1"

cd "$BASE_DIR"
mkdir -p logs

run_webshop() {
    local run_id="$1"
    local full_exp="memp-32b-qwencur-run${run_id}"
    echo "=== START: webshop $full_exp  $(date) ==="
    "$PYTHON" run_memp_webshop.py \
        --model         "$EXECUTOR" \
        --batch_size    10 \
        --max_steps     30 \
        --retrieve_num  5 \
        --exp_name      "$full_exp" \
        --overwrite \
        --mem_model     "$CURATOR" \
        --mem_base_url  "$CURATOR_URL"
    echo "=== DONE: webshop $full_exp  exit=$?  $(date) ==="
}

for run in 1 2 3; do
    run_webshop "$run" 2>&1 | tee "logs/webshop_memp_32b_qwencur_run${run}_v2.log"
done
echo "All done $(date)"
