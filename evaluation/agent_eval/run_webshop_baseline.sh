#!/bin/bash
# WebShop No-Memory Baseline Experiments
# Models: Qwen3-8B (port 8001), Qwen3-32B (port 8002), Gemini-2.5-Pro (Vertex AI)
# Each model: 3 independent runs on 500 test goals

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"
export OPENAI_API_KEY="EMPTY"

PYTHON=/home/ouyangsiru/memory_env/bin/python
BASE_DIR=/home/ouyangsiru/MemP/ProcedureMem

cd "$BASE_DIR"

run_experiment() {
    local model_id="$1"
    local base_url="$2"
    local exp_name="$3"

    echo "======================================================"
    echo "START: model=$model_id  exp=$exp_name"
    echo "Time: $(date)"
    echo "======================================================"

    OPENAI_API_BASE="$base_url" "$PYTHON" run_skillos_online.py \
        --env webshop \
        --model "$model_id" \
        --split dev \
        --batch_size 10 \
        --max_steps 30 \
        --exp_name "$exp_name"

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

mkdir -p logs

# ---- Qwen3-8B: 3 runs ----
# run_experiment "openai/Qwen/Qwen3-8B"  "http://10.202.0.8:8001/v1"  "baseline-8b-run1"
# run_experiment "openai/Qwen/Qwen3-8B"  "http://10.202.0.8:8001/v1"  "baseline-8b-run2"
# run_experiment "openai/Qwen/Qwen3-8B"  "http://10.202.0.8:8001/v1"  "baseline-8b-run3"

# ---- Qwen3-32B: 3 runs ----
# run_experiment "openai/Qwen/Qwen3-32B" "http://10.202.0.8:8002/v1"  "baseline-32b-run1"
# run_experiment "openai/Qwen/Qwen3-32B" "http://10.202.0.8:8002/v1"  "baseline-32b-run2"
# run_experiment "openai/Qwen/Qwen3-32B" "http://10.202.0.8:8002/v1"  "baseline-32b-run3"

# ---- Gemini-2.5-Pro: 3 runs (uses Vertex AI; base_url unused but required by function) ----
# run_experiment "gemini/gemini-2.5-pro" "http://localhost:8000/v1"  "baseline-gemini-run1"
# run_experiment "gemini/gemini-2.5-pro" "http://localhost:8000/v1"  "baseline-gemini-run2"
# run_experiment "gemini/gemini-2.5-pro" "http://localhost:8000/v1"  "baseline-gemini-run3"

echo "All experiments done. $(date)"
