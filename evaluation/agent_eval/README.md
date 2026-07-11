# agent_eval

Downstream evaluation harness comparing four memory methods across embodied, web-shopping, and reasoning benchmarks. Ported from [zjunlp/MemP](https://github.com/zjunlp/MemP).

## Methods

| `--memory_type` | Description |
|---|---|
| `none` | No memory baseline — executor answers each question independently |
| `reasoningbank` | [ReasoningBank](https://arxiv.org/abs/2402.13177) — distills past trajectories into reusable reasoning items (ALFWorld only) |
| `memp` | MemP — procedural memory of successful action sequences |
| `skillos` | SkillCurator — BM25-retrieved skill memory maintained by a trained curator |

## Environments

| `--env` | Benchmark |
|---|---|
| `alfworld` | ALFWorld embodied tasks |
| `webshop` | WebShop e-commerce navigation |
| `amc23`, `aime24`, `aime25` | Math competition problems (memory-free) |
| `gpqa` | Graduate-level science QA from [gpqa_diamond.csv](gpqa_diamond.csv) (memory-free) |

Reasoning envs ignore `--memory_type`; ReasoningBank only supports `alfworld`.

## Quick Start

The single entry point is [run_unified.py](run_unified.py):

```bash
# No memory — ALFWorld
python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline

# SkillOS (trained curator) — ALFWorld
python run_unified.py --env alfworld --memory_type skillos \
    --curation_model Qwen/Qwen3-8B \
    --model openai/Qwen/Qwen3-8B --exp_name skillos-qwen3-8b

# ReasoningBank — ALFWorld
python run_unified.py --env alfworld --memory_type reasoningbank \
    --curation_model Qwen/Qwen3-8B \
    --model openai/Qwen/Qwen3-8B --exp_name rb-qwen3-8b

# WebShop with SkillOS
python run_unified.py --env webshop --memory_type skillos \
    --curation_model Qwen/Qwen3-8B \
    --model openai/Qwen/Qwen3-8B --exp_name skillos-qwen3-8b

# Reasoning (memory_type ignored)
python run_unified.py --env aime24 \
    --model openai/Qwen/Qwen3-8B --exp_name run1
```

`--model` follows the LiteLLM convention (`openai/<name>` for an OpenAI-compatible endpoint such as a local vLLM server). `--curation_model` is the curator model that produces memory updates.

## Driver Scripts

Pre-baked shell wrappers around `run_unified.py` and the legacy per-method runners cover the common experimental matrix:

- `run_alfworld_*.sh` — ALFWorld variants (flashlite, gemini curator, ablation curators)
- `run_webshop_*.sh` — WebShop variants (8B / 32B / Gemini, complete-trained-curator combos)
- `run_memp_*.sh` — MemP across ALFWorld, WebShop, reasoning
- `run_reasoning_*.sh` — Reasoning benchmarks across baselines and SkillOS-trained curators
- `run_skillos.sh`, `run_skillos_32B.sh`, `run_reasoningbank.sh` — direct method runners

## Directory Layout

| Path | Purpose |
|---|---|
| [run_unified.py](run_unified.py) | Canonical multi-method, multi-env entry point |
| [run_skillos_online.py](run_skillos_online.py), [run_memp_online.py](run_memp_online.py), [run_reasoningbank_online.py](run_reasoningbank_online.py) | Per-method online runners |
| [run_memp_offline.py](run_memp_offline.py), [run_memp_ori.py](run_memp_ori.py), [run_memp_reasoning.py](run_memp_reasoning.py), [run_memp_webshop.py](run_memp_webshop.py) | MemP offline / variant runners |
| [memory.py](memory.py), [memory_adjust.py](memory_adjust.py), [memory_utils.py](memory_utils.py) | Shared memory data structures and helpers |
| [reasoningbank.py](reasoningbank.py), [reasoningbank_alfworld.py](reasoningbank_alfworld.py) | ReasoningBank implementation |
| [llm_api.py](llm_api.py), [eval.py](eval.py), [prompt_generator.py](prompt_generator.py) | LLM client wrappers, scoring, prompt construction |
| [SkillOS/](SkillOS/) | SkillOS curator agent (`skills_agent.py`, `skills_memory.py`, `skills_functions.py`, `generation.py`, `benchmark.py`) |
| [Alfworld/](Alfworld/) | ALFWorld prompts, examples, formatted trajectories, base config |
| [agent_system/](agent_system/) | Vendored environment wrappers, multi-turn rollout loop, reward manager |
| [webshop/](webshop/) | WebShop env adapter (without the upstream WebShop dataset; install separately) |
| [ProcedureMem/](ProcedureMem/) | Legacy MemP config |
| [gpqa_diamond.csv](gpqa_diamond.csv) | GPQA Diamond questions |

## Auxiliary Scripts

- [compute_reasoning_results.py](compute_reasoning_results.py), [compute_webshop_results.py](compute_webshop_results.py) — aggregate per-run metrics
- [plot_case_study.py](plot_case_study.py), [plot_case_study_v2.py](plot_case_study_v2.py), [plot_case_study_v3.py](plot_case_study_v3.py) — render qualitative case studies
- [plot_curator_quality.py](plot_curator_quality.py), [plot_curator_quality_3cases.py](plot_curator_quality_3cases.py), [plot_curator_quality_long.py](plot_curator_quality_long.py) — curator quality plots
- [eval.py](eval.py), [test.py](test.py) — sanity checks

## Notes

- The 18 MB [Alfworld/alfworld_format_traj.json](Alfworld/alfworld_format_traj.json) is the formatted ALFWorld trajectory dataset used by the runners — kept inline for convenience.
- The WebShop dataset itself (`webshop/webshop/`, ~229 MB) is excluded; install [WebShop](https://github.com/princeton-nlp/WebShop) separately if you need to run that environment.
- Heavy binary assets in `agent_system/environments/env_package/` (`.npy`, `.png`, etc.) were stripped — only `.py` and `.md` are kept.
