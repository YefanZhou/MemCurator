#!/bin/bash

# For gameplay LLM via litellm
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.202.0.8:8001/v1"

python run_skillos_online.py \
    --model openai/Qwen/Qwen3-8B \
    --split dev \
    --batch_size 10 \
    --max_steps 30 \
    --exp_name skillos-nomemory-8b\
    --retrieve_num 5 \
    --curation_model Qwen/Qwen3-8B
