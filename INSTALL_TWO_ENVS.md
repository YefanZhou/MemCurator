# Two-Environment Setup: `memory-train` + `memory-eval`

An alternative to the single `memory` env. Splits the heavy training/serving stack
from the lightweight eval client so that installing eval deps (langchain / chroma,
which downgrade `openai` and `opentelemetry`) can never perturb the vLLM serving stack.

| Env | Role | Key deps |
|---|---|---|
| `memory-train` | GRPO training **+** serving Qwen3-8B/32B via vLLM | torch 2.6, vllm 0.8.5, flash-attn, verl, ray |
| `memory-eval` | Eval client: no-memory / memp / reasoningbank / skillos runs | langchain, chromadb, litellm, alfworld — **no vllm** |

The eval env talks to the serving env over HTTP (OpenAI-compatible endpoint), so it
never needs vllm/torch/flash-attn locally.

Server context (adjust paths to your machine):
- Repo: `/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main`
- Platform: Linux x86_64, glibc 2.35, CUDA 12.4, Python 3.10
- ALFWorld data (already downloaded once): `/fsx/home/yefan.zhou/.cache/alfworld`

---

## ⚠️ ALFWorld data is NOT re-downloaded

The ALFWorld game data (`json_2.1.1/`, logic files, detectors) is **filesystem data**,
independent of any conda env. You download it **once** and both envs point at it via the
`ALFWORLD_DATA` environment variable.

- **Per-env (must repeat):** `pip install alfworld==0.4.2` — the Python *package*.
- **One-time (already done):** `alfworld-download -f` — the ~2GB game *data*.

So in `memory-eval` you install the alfworld package but **skip** `alfworld-download`.
Just export `ALFWORLD_DATA=/fsx/home/yefan.zhou/.cache/alfworld` before running.

---

## Env A — `memory-train` (training + vLLM serving)

```bash
conda create -n memory-train python=3.10 -y
conda activate memory-train
cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main

# 1. Training/serve stack. flash-attn is commented out in requirements.txt
#    (source build needs torch present first); we add it from a prebuilt wheel after.
pip install -r requirements.txt

# 2. Confirm torch + CUDA + ABI (should print: 2.6.0+cu124 12.4 False)
python -c "import torch; print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"

# 3. flash-attn from the matching prebuilt wheel (cu12 / torch2.6 / cp310 / abiFALSE)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 4. Verify the serving stack imports
python -c "import vllm; print('vllm OK', vllm.__version__)"
python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)"
```

### Serve the model (leave running)

```bash
conda activate memory-train
vllm serve Qwen/Qwen3-8B --port 8001 --served-model-name Qwen/Qwen3-8B --dtype bfloat16
# 32B variant would use --port 8002 per the driver scripts
```

Endpoint: `http://localhost:8001/v1`

---

## Env B — `memory-eval` (eval client + ALFWorld)

```bash
conda create -n memory-eval python=3.10 -y
conda activate memory-eval
cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main

# 1. Eval stack (langchain / chroma / litellm / openai). Pins openai==1.75.0 and
#    otel high — harmless here because there is NO vllm in this env to conflict with.
pip install -r evaluation/requirements.txt

# 2. Deps run_unified.py imports that eval/requirements.txt omits.
#    (datasets/tqdm/pyyaml are transitive via other packages; math_verify + these are safe explicit installs.)
pip install datasets tqdm pyyaml
pip install math_verify              # only needed for reasoning envs (aime/amc/gpqa)

# 3. transformers — required only for the SkillOS in-process tokenizer path.
#    Needed if you run --memory_type skillos WITHOUT --curation_base_url.
#    Skip if you only run none / memp / reasoningbank over HTTP.
pip install 'transformers==4.51.3'

# 4. ALFWorld PACKAGE (data is reused, not re-downloaded — see note above)
pip install gymnasium==0.29.1 stable-baselines3==2.6.0
pip install alfworld==0.4.2

# 5. Sanity check imports (ignore pip-check metadata warnings — verify by import)
python -c "import litellm, langchain_community, chromadb, openai; print('eval imports OK')"
python -c "import alfworld.agents.environment; from alfworld.agents.environment import get_environment; print('alfworld env chain OK')"
python -c "import textworld; from textworld import gym; print('textworld OK', textworld.__version__)"
```

> Note: `memory-eval` deliberately has **no vllm**. Therefore any in-process curator
> (`--memory_type skillos`/`memp`/`reasoningbank` with a bare `--curation_model`) that
> would load a model via vllm will fail here. Always run the curator over HTTP with
> `--curation_base_url http://localhost:8001/v1` so it uses the served model instead.

---

## Running eval (Env B, with Env A serving)

```bash
conda activate memory-eval
export ALFWORLD_DATA=/fsx/home/yefan.zhou/.cache/alfworld   # reuse existing data
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval

# (first: patch Alfworld/base_config.yaml paths to $ALFWORLD_DATA — see below)

# no-memory
python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline

# reasoningbank (curator over HTTP, no local vllm needed)
python run_unified.py --env alfworld --memory_type reasoningbank \
    --curation_model Qwen/Qwen3-8B --curation_base_url http://localhost:8001/v1 \
    --model openai/Qwen/Qwen3-8B --exp_name rb-8b

# memp
python run_unified.py --env alfworld --memory_type memp \
    --curation_model Qwen/Qwen3-8B --curation_base_url http://localhost:8001/v1 \
    --model openai/Qwen/Qwen3-8B --exp_name memp-8b

# reasoning (memory_type ignored; needs math_verify)
python run_unified.py --env aime24 \
    --model openai/Qwen/Qwen3-8B --exp_name baseline
```

---

## Fix ALFWorld config paths (one-time, in the repo)

`evaluation/agent_eval/Alfworld/base_config.yaml` ships with the authors' hardcoded
paths (`/home/siruo_google_com/.cache/alfworld/...`). Point them at `$ALFWORLD_DATA`:

```yaml
dataset:
  data_path: '$ALFWORLD_DATA/json_2.1.1/train'
  eval_id_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_seen'
  eval_ood_data_path: '$ALFWORLD_DATA/json_2.1.1/valid_unseen'
logic:
  domain: '$ALFWORLD_DATA/logic/alfred.pddl'
  grammar: '$ALFWORLD_DATA/logic/alfred.twl2'
```

Confirm the actual layout first with `ls $ALFWORLD_DATA` and `ls $ALFWORLD_DATA/json_2.1.1`.
If `alfred.pddl`/`alfred.twl2` are not under `$ALFWORLD_DATA/logic/`, they ship inside the
alfworld package (`BUILTIN_DATA_PATH`) — point `domain`/`grammar` there instead.

---

## Why two envs (recap)

- **Hard conflict:** `chromadb 0.6.3` (eval) pulls `opentelemetry>=1.30`, while
  `vllm 0.8.5` (serve) requires `opentelemetry<1.27`. In one env this coexists only by
  luck (imports work despite pip-check warnings). Splitting removes the risk entirely:
  serve env keeps otel 1.26, eval env lets otel float.
- **openai pin:** eval wants `1.75.0`, vllm wants `>=1.78`. Separate envs = no tug-of-war.
- **Cost:** more disk + you must keep the vLLM server running for the eval env to hit.
```
