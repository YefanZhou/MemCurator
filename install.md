conda activate memory
cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main



pip install -r requirements.txt

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install -r evaluation/requirements.txt

pip install 'openai==1.78.1' \
    'opentelemetry-api==1.26.0' \
    'opentelemetry-sdk==1.26.0' \
    'opentelemetry-semantic-conventions==0.47b0' \
    'opentelemetry-proto==1.26.0' \
    'opentelemetry-exporter-otlp-proto-common==1.26.0' \
    'opentelemetry-exporter-otlp-proto-grpc==1.26.0'


pip install math_verify               # only if running reasoning envs (aime/amc/gpqa)
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld==0.4.2
pip check

export ALFWORLD_DATA=/fsx/home/yefan.zhou/.cache/alfworld
alfworld-download -f


# 2. Confirm torch + CUDA before picking the wheel
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"

python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"
2.6.0+cu124 12.4 False

