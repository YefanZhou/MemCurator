#!/bin/bash

# Vertex AI API configuration
export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

# For gameplay LLM via litellm
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.148.0.45:8001/v1"

# python run_skillos_online.py \
#     --model openai/Qwen/Qwen3-8B \
#     --split dev \
#     --batch_size 10 \
#     --max_steps 30 \
#     --exp_name skillos-8b \
#     --retrieve_num 5 \
#     --curation_model /home/siruo_google_com/SkillCurator/converted_ckpt/skillos-qwen3-8b-step50 \
#     --use_memory \
#     --overwrite

python run_skillos_online.py \
    --model openai/Qwen/Qwen3-8B \
    --split dev \
    --batch_size 10 \
    --max_steps 30 \
    --exp_name skillos-qwen38b \
    --retrieve_num 5 \
    --curation_model Qwen/Qwen3-8B \
    --use_memory \
    --overwrite
