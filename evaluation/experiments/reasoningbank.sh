


export ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY TMPDIR=$HOME/tmp
export OPENAI_API_BASE=http://localhost:8002/v1

# executor
export EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true
# curator
export CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false
# printing
export PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000 HISTORY_LENGTH=5

python run_unified_dev.py --env alfworld --memory_type reasoningbank \
    --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 10 --retrieve_num 5 --max_steps 30 \
    --exp_name "${1:-rb-qwen3-8b_curator_0.6_nonthinking_run${i}}" --overwrite