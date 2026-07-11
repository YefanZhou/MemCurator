#!/bin/bash
set -o pipefail
# MemP on ALFWorld + Reasoning + WebShop with Gemini executor + Qwen3-8B curator

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="gemini/gemini-2.5-pro"
CURATOR="openai/Qwen/Qwen3-8B"
CURATOR_URL="http://10.148.0.45:8001/v1"

cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_alfworld() {
    local run_id="$1"
    local full_exp="memp-gem-qwencur-run${run_id}"
    echo "=== START: alfworld $full_exp  $(date) ==="
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
    echo "=== DONE: alfworld $full_exp  exit=$?  $(date) ==="
}

run_reasoning() {
    local env="$1"; local run_id="$2"; local bs="$3"
    local full_exp="memp-gem-qwencur-run${run_id}"
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

run_webshop() {
    local run_id="$1"
    local full_exp="memp-gem-qwencur-run${run_id}"
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

# Reasoning first (faster, no env init cost)
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_reasoning "$env" "$run" "$bs" 2>&1 | tee "logs/reasoning_memp_gem_qwencur_${env}_run${run}.log"
    done
done

# ALFWorld
for run in 1 2 3; do
    run_alfworld "$run" 2>&1 | tee "logs/alfworld_memp_gem_qwencur_run${run}.log"
done

# WebShop last (longest)
for run in 1 2 3; do
    run_webshop "$run" 2>&1 | tee "logs/webshop_memp_gem_qwencur_run${run}.log"
done

echo "All MemP gem experiments done. $(date)"
