#!/bin/bash
set -o pipefail
# Webshop completion — Qwen3-32B executor lane

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="openai/Qwen/Qwen3-32B"
EXECUTOR_URL="http://10.148.0.45:8002/v1"
QWEN_CURATOR="openai/Qwen/Qwen3-8B"
QWEN_CURATOR_URL="http://10.148.0.45:8001/v1"
GEM_CURATOR="gemini/gemini-2.5-pro"
WSCUR="/home/siruo_google_com/SkillCurator/converted_models/qwen3-8b-webshop-skillos-step50"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local exp_name="$1" mem_type="$2" curator="$3" curator_url="$4"
    echo "=== START: $exp_name  mem=$mem_type  curator=${curator:-none}  $(date) ==="
    local cmd_args=(
        --env         webshop
        --model       "$EXECUTOR"
        --memory_type "$mem_type"
        --batch_size  10
        --max_steps   30
        --exp_name    "$exp_name"
        --overwrite
    )
    if [ "$mem_type" != "none" ]; then
        cmd_args+=(--curation_model "$curator" --retrieve_num 5)
        if [ -n "$curator_url" ]; then
            cmd_args+=(--curation_base_url "$curator_url")
        fi
    fi
    OPENAI_API_BASE="$EXECUTOR_URL" "$PYTHON" run_unified.py "${cmd_args[@]}"
    echo "=== DONE: $exp_name  exit=$?  $(date) ==="
}

run_experiment "rb-qwencur-32b-run2" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_rb_qwencur_32b_run2.log"
run_experiment "rb-qwencur-32b-run3" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_rb_qwencur_32b_run3.log"

run_experiment "skillos-qwencur-32b-run2" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_skillos_qwencur_32b_run2.log"
run_experiment "skillos-qwencur-32b-run3" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_skillos_qwencur_32b_run3.log"

run_experiment "skillos-gemcur-32b-run2" "skillos" "$GEM_CURATOR" "" 2>&1 | tee "logs/webshop_skillos_gemcur_32b_run2.log"
run_experiment "skillos-gemcur-32b-run3" "skillos" "$GEM_CURATOR" "" 2>&1 | tee "logs/webshop_skillos_gemcur_32b_run3.log"

run_experiment "skillos-trainedcur-32b-run3" "skillos" "$WSCUR" "" 2>&1 | tee "logs/webshop_trainedcur_32b_run3.log"

echo "=== Lane 32B done === $(date)"
