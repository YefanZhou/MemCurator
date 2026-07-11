#!/bin/bash
set -o pipefail
# SkillOS on ALFWorld with Qwen3-8B executor + ablation curator (no-compression GRPO ckpt)

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.148.0.45:8001/v1"

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
EXECUTOR="openai/Qwen/Qwen3-8B"
EXECUTOR_URL="http://10.148.0.45:8001/v1"
CURATOR="/home/siruo_google_com/SkillCurator/converted_models/qwen3-8b-alfworld-ablation-nocompression-step60"

cd "$BASE_DIR"
mkdir -p logs

run_experiment() {
    local run_id="$1"
    local exp_name="skillos-alfablcur-8b-run${run_id}"
    echo "=== START: $exp_name  $(date) ==="
    OPENAI_API_BASE="$EXECUTOR_URL" "$PYTHON" run_unified.py \
        --env            alfworld \
        --model          "$EXECUTOR" \
        --memory_type    skillos \
        --batch_size     10 \
        --max_steps      30 \
        --exp_name       "$exp_name" \
        --overwrite \
        --curation_model "$CURATOR" \
        --retrieve_num   3
    echo "=== DONE: $exp_name  exit=$?  $(date) ==="
}

for run in 1 2 3; do
    run_experiment "$run" 2>&1 | tee "logs/alfworld_skillos_alfablcur_8b_run${run}.log"
done
echo "All done $(date)"
