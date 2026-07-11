vllm serve Qwen/Qwen3-32B \
  --dtype auto \
  --api-key token-abc123

python memory_server.py --server_url http://localhost:8001/v1 --model_name qwen3-32b\
  > server_outputs.log 2>&1 &