#!/bin/bash
set -o pipefail
# WebShop experiments
# Executors : Qwen3-8B (port 8001) | Qwen3-32B (port 8002) | Gemini-2.5-Pro
# Curators  : Gemini-2.5-Pro (gemcur) | Qwen3-8B (qwencur)
# Memory    : none | skillos | reasoningbank
# Goals     : 500 (full dev split)

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

export OPENAI_API_KEY="EMPTY"
export JAVA_HOME=/usr/lib/jvm/temurin-21-jdk-amd64

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
GEM_CURATOR="gemini/gemini-2.5-pro"
QWEN_CURATOR="openai/Qwen/Qwen3-8B"
QWEN_CURATOR_URL="http://10.148.0.45:8001/v1"

cd "$BASE_DIR"
mkdir -p logs

# run_experiment <exp_name> <executor_model> <executor_url> <memory_type> <curator_model> <curator_url>
# executor_url  : empty for Gemini executor
# curator_model : empty for no-memory baseline
# curator_url   : empty for Gemini curator or no-memory baseline
run_experiment() {
    local exp_name="$1"
    local executor_model="$2"
    local executor_url="$3"
    local memory_type="$4"
    local curator_model="$5"
    local curator_url="$6"

    echo "======================================================"
    echo "START: exp=$exp_name  executor=$executor_model  memory=$memory_type  curator=${curator_model:-none}"
    echo "Time: $(date)"
    echo "======================================================"

    local cmd_args=(
        --env            webshop
        --model          "$executor_model"
        --memory_type    "$memory_type"
        --batch_size     10
        --max_steps      30
        --exp_name       "$exp_name"
        --overwrite
    )

    if [ "$memory_type" != "none" ]; then
        cmd_args+=(
            --curation_model "$curator_model"
            --retrieve_num   5
        )
        if [ -n "$curator_url" ]; then
            cmd_args+=(--curation_base_url "$curator_url")
        fi
    fi

    if [ -n "$executor_url" ]; then
        OPENAI_API_BASE="$executor_url" "$PYTHON" run_unified.py "${cmd_args[@]}"
    else
        "$PYTHON" run_unified.py "${cmd_args[@]}"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# Baseline — no memory
# ============================================================
run_experiment "baseline-8b" \
    "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "none" "" "" \
    2>&1 | tee "logs/webshop_baseline_8b.log"

run_experiment "baseline-32b" \
    "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "none" "" "" \
    2>&1 | tee "logs/webshop_baseline_32b.log"

run_experiment "baseline-gemini" \
    "gemini/gemini-2.5-pro" "" "none" "" "" \
    2>&1 | tee "logs/webshop_baseline_gemini.log"

# ============================================================
# SkillOS — Gemini curator
# ============================================================
run_experiment "skillos-gemcur-8b" \
    "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "skillos" "$GEM_CURATOR" "" \
    2>&1 | tee "logs/webshop_skillos_gemcur_8b.log"

run_experiment "skillos-gemcur-32b" \
    "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "skillos" "$GEM_CURATOR" "" \
    2>&1 | tee "logs/webshop_skillos_gemcur_32b.log"

run_experiment "skillos-gemcur-gem" \
    "gemini/gemini-2.5-pro" "" "skillos" "$GEM_CURATOR" "" \
    2>&1 | tee "logs/webshop_skillos_gemcur_gem.log"

# ============================================================
# SkillOS — Qwen3-8B curator
# ============================================================
run_experiment "skillos-qwencur-8b" \
    "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" \
    2>&1 | tee "logs/webshop_skillos_qwencur_8b.log"

run_experiment "skillos-qwencur-32b" \
    "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" \
    2>&1 | tee "logs/webshop_skillos_qwencur_32b.log"

run_experiment "skillos-qwencur-gem" \
    "gemini/gemini-2.5-pro" "" "skillos" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" \
    2>&1 | tee "logs/webshop_skillos_qwencur_gem.log"

# ============================================================
# ReasoningBank — Qwen3-8B curator
# ============================================================
run_experiment "rb-qwencur-8b" \
    "openai/Qwen/Qwen3-8B" "http://10.148.0.45:8001/v1" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" \
    2>&1 | tee "logs/webshop_rb_qwencur_8b.log"

run_experiment "rb-qwencur-32b" \
    "openai/Qwen/Qwen3-32B" "http://10.148.0.45:8002/v1" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" \
    2>&1 | tee "logs/webshop_rb_qwencur_32b.log"

run_experiment "rb-qwencur-gem" \
    "gemini/gemini-2.5-pro" "" "reasoningbank" "$QWEN_CURATOR" "$QWEN_CURATOR_URL" \
    2>&1 | tee "logs/webshop_rb_qwencur_gem.log"

echo "All webshop experiments done. $(date)"
