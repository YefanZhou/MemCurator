# SkillCurator — Installation

Single `memory` conda env (the setup the codebase authors ran). See
[installment_caveat.md](installment_caveat.md) for why the `pip check` warnings are benign — the
correctness gate is *does it import*, not `pip check`.

Assumed context: **Linux x86_64, glibc 2.35, CUDA 12.4, Python 3.10, torch 2.6.0+cu124.**
Adjust the repo path and the ALFWorld cache path to your machine.

```bash
conda create -n memory python=3.10 -y
conda activate memory
cd /path/to/SkillCurator-main

# 1. Training/serve stack (flash-attn is commented out in requirements.txt — a
#    build-from-source needs torch present first, so we add it as a wheel in step 3).
pip install -r requirements.txt

# 2. Confirm torch + CUDA + ABI before picking the flash-attn wheel.
#    Expected: 2.6.0+cu124 12.4 False
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"

# 3. flash-attn from the matching prebuilt wheel (cu12 / torch2.6 / cp310 / abiFALSE)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. Eval-harness deps. NOTE: this step re-downgrades openai to 1.75.0 and can bump
#    opentelemetry/protobuf — step 5 restores the pins the serving stack needs.
pip install -r evaluation/requirements.txt

# 5. Restore openai + pin opentelemetry/protobuf back down so vllm 0.8.5 (needs otel<1.27)
#    is satisfied. Keep protobuf at 4.x — vllm/ray require it; the google-api-core /
#    grpcio-status complaints about wanting >=5 are benign (that Gemini path is never imported).
pip install 'openai==1.78.1' \
    'protobuf==4.25.9' \
    'opentelemetry-api==1.26.0' \
    'opentelemetry-sdk==1.26.0' \
    'opentelemetry-semantic-conventions==0.47b0' \
    'opentelemetry-proto==1.26.0' \
    'opentelemetry-exporter-otlp-proto-common==1.26.0' \
    'opentelemetry-exporter-otlp-proto-grpc==1.26.0'


pip install "ray[default]==2.46.0" "click<8.2"
pip install "protobuf<5"

# 6. ALFWorld + reasoning extras (skip math_verify unless running aime/amc/gpqa).
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld==0.4.2
pip install math_verify

# 7. ALFWorld game data — download ONCE (~2GB); reused by every run via ALFWORLD_DATA.
export ALFWORLD_DATA=/path/to/.cache/alfworld
alfworld-download -f
```

> **Critical rule:** any time you re-run `pip install -r evaluation/requirements.txt`, redo
> step 5 — it silently re-downgrades `openai` to 1.75.0 and can bump `opentelemetry`/`protobuf`.



## Verify

All six should print OK. `pip check` metadata warnings are expected and benign — see
[installment_caveat.md](installment_caveat.md).

```bash
python -c "import vllm; print('vllm OK', vllm.__version__)"                       # 0.8.5
python -c "import chromadb; print('chromadb OK', chromadb.__version__)"           # 0.6.3
python -c "import litellm, langchain_community, openai; print('eval imports OK')"
python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)"     # 2.7.4.post1
python -c "import alfworld.agents.environment; from alfworld.agents.environment import get_environment; print('alfworld env chain OK')"
python -c "import textworld; from textworld import gym; print('textworld OK', textworld.__version__)"  # 1.7.0
```

