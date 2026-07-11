# SkillCurator

SkillCurator trains a **skill curator** policy via reinforcement learning that maintains a dynamic, BM25-retrievable skill memory for a frozen executor LLM. The curator observes each problem-solving attempt and updates the memory through structured `insert`/`update`/`delete` function calls; the executor uses retrieved skills to answer subsequent questions. Training is done with GRPO on top of [verl](https://github.com/volcengine/verl) and supports math, ALFWorld, and WebShop tasks.

## Algorithm

The training algorithm is documented in [algorithm.md](algorithm.md). Key ideas:

- **Two agents, one trainable.** The executor is frozen; only the curator is updated.
- **Shifted credit assignment.** The curator's action at step *t* is rewarded by the executor's accuracy at step *t+1* — the step whose memory it actually shaped.
- **Composite reward.** Accuracy + compression + function-call validity + memory-use (judged by a separate LLM) — see [algorithm.md](algorithm.md) for the formula.
- **GRPO** with K independent rollouts per problem sequence, KL penalty against a reference policy.

## Repository Structure

| Path | Purpose |
|---|---|
| [agent.py](agent.py), [main.py](main.py), [memory.py](memory.py), [functions.py](functions.py) | Core training loop and memory/agent classes |
| [skillos/](skillos/) | Curator generation managers (math, ALFWorld, WebShop) and metrics |
| [skills/](skills/) | Skill-aware agent + benchmark harness for inference-time use |
| [agent_system/](agent_system/) | Environments, multi-turn rollout loop, reward manager |
| [memory_server/](memory_server/) | vLLM-style HTTP server hosting the memory tool back-end |
| [config/](config/) | YAML configs for agent variants (e.g. [skillos-qwen3-4b_agent_0.05-0.1.yaml](config/skillos-qwen3-4b_agent_0.05-0.1.yaml)) |
| [scripts/](scripts/) | GRPO training shell scripts per model size / environment / ablation |
| [data/](data/), [data_preprocess/](data_preprocess/) | Dataset utilities, rollout analysis, plotting |
| [evaluation/agent_eval/](evaluation/agent_eval/) | Downstream evaluation comparing four memory methods (see its [README](evaluation/agent_eval/README.md)) |
| [verl/](verl/) | Vendored RL training framework |
| [convert_checkpoint.py](convert_checkpoint.py) | Convert verl checkpoints to HuggingFace format |

## Setup

```bash
pip install -r requirements.txt
```

A pre-built `flash_attn` wheel matching the verl/torch versions may be required separately.

## Training

Pick a script from [scripts/](scripts/) matching your model size and target environment:

| Script | Model | Env | Notes |
|---|---|---|---|
| `train_memory_grpo_qwen3-8b-math-compression0.05-content0.1.sh` | Qwen3-8B | math | Default math run |
| `train_memory_grpo_qwen3-8b-alfworld-no-{both,compression,content}.sh` | Qwen3-8B | ALFWorld | Reward ablations |
| `train_memory_grpo_qwen3-8b-webshop-compression0.05-content0.1.sh` | Qwen3-8B | WebShop | |
| `train_memory_grpo_qwen3-4b-compression{0.2,0.4}-content0.1.sh` | Qwen3-4B | math | Compression-weight sweep |

Each script wires up Ray, env vars, the memory server (`scripts/memory_server_config.sh`), and a verl PPO trainer call. Edit `WORKING_DIR`, `BASE_MODEL`, and data paths at the top before running.

## Evaluation

Downstream agent evaluation across no-memory / ReasoningBank / MemP / SkillOS lives in [evaluation/agent_eval/](evaluation/agent_eval/) — see [its README](evaluation/agent_eval/README.md) for the unified entry-point and per-environment examples.

## License

Apache 2.0 — see [LICENSE](LICENSE).
