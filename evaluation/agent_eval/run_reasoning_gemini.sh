#!/bin/bash
set -euo pipefail
# Reasoning Experiments — Gemini 2.5 Pro executor (Vertex AI)
# Benchmarks  : AIME24, AIME25, GPQA Diamond
# Memory types: none | skillos | reasoningbank
# Executor    : gemini-2.5-pro via Vertex AI (no litellm)
# Curator     : Qwen3-8B via HTTP (port 8001)
# Runs        : 3 independent runs per condition

export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.202.0.8:8001/v1"   # curator only

PYTHON=/home/siruo_google_com/miniconda3/envs/memory/bin/python
BASE_DIR=/home/siruo_google_com/MemP/ProcedureMem
CURATOR_MODEL="openai/Qwen/Qwen3-8B"
CURATOR_URL="http://10.202.0.8:8001/v1"

# ============================================================
# Sanity checks
# ============================================================
echo "=== Sanity checks ==="

# 1. Python binary
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python not found: $PYTHON"
    exit 1
fi
echo "[OK] Python: $PYTHON"

# 2. run_unified.py
if [ ! -f "$BASE_DIR/run_unified.py" ]; then
    echo "[ERROR] run_unified.py not found in $BASE_DIR"
    exit 1
fi
echo "[OK] run_unified.py found"

# 3. google-genai installed (Vertex AI executor)
if ! "$PYTHON" -c "from google import genai" 2>/dev/null; then
    echo "[ERROR] google-genai not installed. Run: pip install google-genai"
    exit 1
fi
echo "[OK] google-genai available"

# 4. Curator server reachable
if ! curl -sf --max-time 5 "${CURATOR_URL}/models" -H "Authorization: Bearer EMPTY" > /dev/null 2>&1; then
    echo "[WARN] Curator server not reachable at $CURATOR_URL — memory experiments may fail"
else
    echo "[OK] Curator server reachable"
fi

# 5. Google Cloud credentials
if ! "$PYTHON" -c "
from google import genai
client = genai.Client(vertexai=True, project='zifengw-research', location='global')
print('credentials ok')
" 2>/dev/null; then
    echo "[WARN] Vertex AI credentials check failed — ensure gcloud auth is set up"
else
    echo "[OK] Vertex AI credentials"
fi

echo "=== Sanity checks passed ==="
echo ""

# ============================================================
cd "$BASE_DIR"
mkdir -p logs

bs_aime24=3
bs_aime25=3
bs_gpqa=10

run_experiment() {
    local env="$1"
    local memory_type="$2"
    local exp_name="$3"
    local batch_size="$4"

    echo "======================================================"
    echo "START: env=$env  model=gemini-2.5-pro  memory=$memory_type  exp=$exp_name  bs=$batch_size"
    echo "Time: $(date)"
    echo "======================================================"

    if [ "$memory_type" = "none" ]; then
        "$PYTHON" run_unified.py \
            --env          "$env" \
            --model        "gemini/gemini-2.5-pro" \
            --memory_type  "none" \
            --batch_size   "$batch_size" \
            --exp_name     "$exp_name"
    else
        "$PYTHON" run_unified.py \
            --env               "$env" \
            --model             "gemini/gemini-2.5-pro" \
            --memory_type       "$memory_type" \
            --curation_model    "$CURATOR_MODEL" \
            --curation_base_url "$CURATOR_URL" \
            --batch_size        "$batch_size" \
            --retrieve_num      3 \
            --exp_name          "$exp_name"
    fi

    local exit_code=$?
    echo "======================================================"
    echo "DONE: $env/$memory_type/$exp_name  exit=$exit_code  Time: $(date)"
    echo "======================================================"
    return $exit_code
}

# ============================================================
# (i) No Memory — Gemini 2.5 Pro
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "none" "baseline-gemini25pro-run${run}" "$bs" \
            2>&1 | tee "logs/baseline_gemini25pro_${env}_run${run}.log"
    done
done

# ============================================================
# (ii) SkillOS — Gemini 2.5 Pro executor, Qwen3-8B curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "skillos" "skillos-gemini25pro-run${run}" "$bs" \
            2>&1 | tee "logs/skillos_gemini25pro_${env}_run${run}.log"
    done
done

# ============================================================
# (iii) ReasoningBank — Gemini 2.5 Pro executor, Qwen3-8B curator
# ============================================================
for run in 1 2 3; do
    for env in aime24 aime25 gpqa; do
        bs_var="bs_${env}"; bs="${!bs_var}"
        run_experiment "$env" \
            "reasoningbank" "rb-gemini25pro-run${run}" "$bs" \
            2>&1 | tee "logs/rb_gemini25pro_${env}_run${run}.log"
    done
done

echo "All Gemini 2.5 Pro reasoning experiments done. $(date)"
