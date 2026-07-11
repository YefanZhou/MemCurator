# vllm serve Qwen/Qwen3-32B \
#   --dtype auto \
#   --api-key EMPTY \
#   --port 8001 \
#   --tensor-parallel-size 8

# export QWEN_URL="http://10.148.0.45:8001/v1"
export QWEN_MODEL_NAME="Qwen/Qwen3-8B"

python memory_server.py --server_url http://0.0.0.0:8001/v1 --model_name qwen3-8b > server_outputs.log 2>&1