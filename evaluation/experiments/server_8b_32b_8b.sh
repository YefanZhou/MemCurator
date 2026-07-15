
SAVE_RAW=10 ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8001/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=revise_react \
PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 10 \
    --retrieve_num 5 \
    --max_steps 30 \
    --exp_name rb-revise_react_qwen3-8b_curator_1.0_nonthinking_round2 \
    --overwrite 2>&1 | tee logs_debug_memory/rb_revise_react_qwen3-8b_curator_1.0_nonthinking_round2.log



SAVE_RAW=10 ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8001/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=revise_react \
PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 10 \
    --retrieve_num 5 \
    --max_steps 30 \
    --exp_name rb-revise_react_qwen3-8b_curator_1.0_nonthinking_round3 \
    --overwrite 2>&1 | tee logs_debug_memory/rb_revise_react_qwen3-8b_curator_1.0_nonthinking_round3.log