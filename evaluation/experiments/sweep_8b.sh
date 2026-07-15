#!/bin/bash
set -e
# Run from the agent_eval dir so relative paths (run_*.py, logs_debug/) resolve,
# regardless of where this script is invoked from.
cd "$(dirname "$0")/../agent_eval"

for i in {1..3}; do
    EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_pure_async_8b_jul10_temp1.0_0.95_20_4096_hist3_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_pure_async_8b_jul10_temp1.0_0.95_20_4096_hist3_run${i}.log

    done

for i in {1..3}; do
    EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_reviseReact_8b_jul10_temp1.0_0.95_20_4096_hist3_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_reviseReact_8b_jul10_temp1.0_0.95_20_4096_hist3_run${i}.log
    done


for i in {1..3}; do
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_reviseReact_8b_jul10_temp0.6_0.95_20_4096_hist3_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_reviseReact_8b_jul10_temp0.6_0.95_20_4096_hist3_run${i}.log
    done
