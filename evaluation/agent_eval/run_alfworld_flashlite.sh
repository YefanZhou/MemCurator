#!/bin/bash
# Master launcher — 6 configs × 3 runs on ALFWorld with gemini-3.1-flash-lite-preview executor
# Each config runs as its own background lane, with 3 sequential runs.
set -o pipefail

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="gemini/gemini-3.1-flash-lite-preview"
QWEN_VANILLA="openai/Qwen/Qwen3-8B"
QWEN_VANILLA_URL="http://10.148.0.45:8001/v1"
GEM_CURATOR="gemini/gemini-2.5-pro"
ALFCUR="/home/siruo_google_com/SkillCurator/converted_models/qwen3-8b-alfworld-skillos-step50"

cd "$BASE_DIR"
mkdir -p logs

# ----- helpers -----
run_unified() {
    local exp="$1" mem_type="$2" curator="$3" curator_url="$4"
    local args=(--env alfworld --model "$EXECUTOR" --memory_type "$mem_type"
                --batch_size 10 --max_steps 30 --retrieve_num 3
                --exp_name "$exp" --overwrite)
    if [ "$mem_type" != "none" ] && [ -n "$curator" ]; then
        args+=(--curation_model "$curator")
        if [ -n "$curator_url" ]; then
            args+=(--curation_base_url "$curator_url")
        fi
    fi
    "$PYTHON" run_unified.py "${args[@]}"
}
run_memp() {
    local exp="$1"
    "$PYTHON" run_memp_online.py \
        --model         "$EXECUTOR" \
        --batch_size    10 --max_steps 30 --retrieve_num 3 --use_memory --overwrite \
        --exp_name      "$exp" \
        --mem_model     "$QWEN_VANILLA" --mem_base_url "$QWEN_VANILLA_URL"
}

# ----- lane definitions -----
lane_nomem() {
    for r in 1 2 3; do
        echo "[nomem r$r] START $(date)"
        run_unified "flashlite-nomem-run${r}" "none" "" "" 2>&1 | tee "logs/flashlite_alfworld_nomem_run${r}.log"
        echo "[nomem r$r] DONE $(date)"
    done
}
lane_rb() {
    for r in 1 2 3; do
        echo "[rb r$r] START $(date)"
        run_unified "flashlite-rb-qwencur-run${r}" "reasoningbank" "$QWEN_VANILLA" "$QWEN_VANILLA_URL" \
            2>&1 | tee "logs/flashlite_alfworld_rb_qwencur_run${r}.log"
        echo "[rb r$r] DONE $(date)"
    done
}
lane_skillos_qwencur() {
    for r in 1 2 3; do
        echo "[skillos-qwencur r$r] START $(date)"
        run_unified "flashlite-skillos-qwencur-run${r}" "skillos" "$QWEN_VANILLA" "$QWEN_VANILLA_URL" \
            2>&1 | tee "logs/flashlite_alfworld_skillos_qwencur_run${r}.log"
        echo "[skillos-qwencur r$r] DONE $(date)"
    done
}
lane_memp() {
    for r in 1 2 3; do
        echo "[memp r$r] START $(date)"
        run_memp "flashlite-memp-qwencur-run${r}" 2>&1 | tee "logs/flashlite_alfworld_memp_qwencur_run${r}.log"
        echo "[memp r$r] DONE $(date)"
    done
}
lane_skillos_gemcur() {
    for r in 1 2 3; do
        echo "[skillos-gemcur r$r] START $(date)"
        run_unified "flashlite-skillos-gemcur-run${r}" "skillos" "$GEM_CURATOR" "" \
            2>&1 | tee "logs/flashlite_alfworld_skillos_gemcur_run${r}.log"
        echo "[skillos-gemcur r$r] DONE $(date)"
    done
}
lane_skillos_alfcur() {
    for r in 1 2 3; do
        echo "[skillos-alfcur r$r] START $(date)"
        run_unified "flashlite-skillos-alfcur-run${r}" "skillos" "$ALFCUR" "" \
            2>&1 | tee "logs/flashlite_alfworld_skillos_alfcur_run${r}.log"
        echo "[skillos-alfcur r$r] DONE $(date)"
    done
}

# ----- launch all lanes in parallel -----
lane_nomem            &
lane_rb               &
lane_skillos_qwencur  &
lane_memp             &
lane_skillos_gemcur   &
lane_skillos_alfcur   &
wait
echo "All flash-lite alfworld lanes done $(date)"
