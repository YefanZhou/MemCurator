#!/bin/bash

# Vertex AI API configuration
export GOOGLE_CLOUD_PROJECT="zifengw-research"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="True"

# For gameplay LLM via litellm
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://10.202.0.8:8002/v1"

# For ReasoningBank's LLM (memory curation)
export API_KEY="EMPTY"

python run_reasoningbank_online.py \
    --model gemini/gemini-2.5-pro \
    --curation_model Qwen/Qwen3-8B \
    --split dev \
    --batch_size 10 \
    --max_steps 30 \
    --exp_name rb-nomemory-1 \
    --overwrite
