#!/bin/bash
set -o pipefail
# Webshop completion — Gemini-2.5-Pro executor lane

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="gemini/gemini-2.5-pro"
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
    "$PYTHON" run_unified.py "${cmd_args[@]}"
    echo "=== DONE: $exp_name  exit=$?  $(date) ==="
}

# rb-qwencur — need fresh 3 runs (current is partial 430/500)
run_experiment "rb-qwencur-gem-run1" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_rb_qwencur_gem_run1.log"
run_experiment "rb-qwencur-gem-run2" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_rb_qwencur_gem_run2.log"
run_experiment "rb-qwencur-gem-run3" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_rb_qwencur_gem_run3.log"

# skillos-qwencur — need run2, run3
run_experiment "skillos-qwencur-gem-run2" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_skillos_qwencur_gem_run2.log"
run_experiment "skillos-qwencur-gem-run3" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" 2>&1 | tee "logs/webshop_skillos_qwencur_gem_run3.log"

# skillos-gemcur — need run2, run3
run_experiment "skillos-gemcur-gem-run2" "skillos" "$GEM_CURATOR" "" 2>&1 | tee "logs/webshop_skillos_gemcur_gem_run2.log"
run_experiment "skillos-gemcur-gem-run3" "skillos" "$GEM_CURATOR" "" 2>&1 | tee "logs/webshop_skillos_gemcur_gem_run3.log"

# skillos-trainedcur — need run3
run_experiment "skillos-trainedcur-gem-run3" "skillos" "$WSCUR" "" 2>&1 | tee "logs/webshop_trainedcur_gem_run3.log"

echo "=== Lane Gemini done === $(date)"
