# sfr_env.sh — point the whole stack at the FAST /fsx/sfr volume (away from the buggy /fsx/home).
#
# WHY: /fsx/home has a filesystem bug (pathological latency on large recursive metadata walks —
# conda-env imports, HF hub scan, ALFWorld's 8810-dir scan, git objects). We migrated the conda env,
# model weights, ALFWorld data, and code to /fsx/sfr. This file is the SINGLE source of truth for the
# 4 knobs that make eval + training use the sfr copies, so the two paths never drift.
#
# USAGE — source it once per shell BEFORE any eval or training command:
#     source /fsx/sfr/yefan.zhou/mem-evolve/SkillCurator-main/scripts/sfr_env.sh
#   then run the eval sweep / training launcher as usual; everything resolves to sfr.
#
# It is idempotent and does NOT touch ~/.bashrc (no global change). Uses PATH-prepend for the conda
# env rather than `conda activate` because `conda activate` is flaky in non-interactive/ssh shells on
# these boxes (the home condabin shadows PATH); prepending the env's bin/ is deterministic. If you
# prefer the activate form for interactive use, `conda activate sfr-memory` also works after sourcing.

# ---- 1. conda env (sfr-memory on the sfr Miniconda) — PATH-prepend, no `conda activate` needed ----
export SFR_CONDA_ENV="/fsx/sfr/yefan.zhou/miniconda3/envs/sfr-memory"
export PATH="${SFR_CONDA_ENV}/bin:${PATH}"
# NOTE: we deliberately do NOT put ${SFR_CONDA_ENV}/lib on LD_LIBRARY_PATH. The env's python/vllm/etc.
# already resolve their own shared libs via rpath (that's why plain `conda activate` works with a clean
# LD_LIBRARY_PATH). Adding the env lib/ shadowed the SYSTEM libtinfo.so.6, spamming
# "libtinfo.so.6: no version information available (required by /bin/bash|screen)" on every command —
# harmless but noisy. Leaving LD_LIBRARY_PATH untouched removes the noise with no downside.

# ---- 2. HuggingFace model cache (Qwen3 4B/4B-Instruct/8B/32B downloaded here) ----
# vllm + verl resolve models BY NAME (e.g. Qwen/Qwen3-8B) from $HF_HOME/hub — no path args needed.
export HF_HOME="/fsx/sfr/yefan.zhou/cache/huggingface"
# HF auth: PASSTHROUGH from your shell (never hardcode the literal token in this committed file).
# vllm/verl resolve models by repo-id (e.g. Qwen/Qwen3-8B), which makes huggingface_hub hit the HF
# API to list the repo file tree even when weights are cached — an ANONYMOUS shared-box IP gets
# 429-rate-limited (noisy; the run still recovers from the local cache). An authenticated token
# raises the limit and quiets it. To use: `export HF_TOKEN=hf_...` in your shell BEFORE sourcing
# this / launching; it is inherited by the ray head+driver+workers (direct path) and by
# `ray job submit` (which snapshots the submitting shell's env). Empty = anonymous (may 429).
export HF_TOKEN="${HF_TOKEN:-}"
# huggingface_hub also reads HUGGING_FACE_HUB_TOKEN; mirror it so either name works.
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
if [ -z "${HF_TOKEN:-}" ]; then echo "[sfr_env] WARN: HF_TOKEN unset — HF calls are anonymous and may hit 429 rate-limits. export HF_TOKEN=hf_... to authenticate."; fi

# W&B auth + entity: PASSTHROUGH (never hardcode the token in this committed file). To log online:
# `export WANDB_API_KEY=wandb_...` in your shell BEFORE sourcing / launching; it is inherited by the
# ray head+driver+workers. WANDB_ENTITY is the team/user the MemCurator project lives under.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-yefan_zhou}"
if [ -z "${WANDB_API_KEY:-}" ]; then echo "[sfr_env] WARN: WANDB_API_KEY unset — online wandb logging will fail. export WANDB_API_KEY=wandb_... (or run WANDB=0 / WANDB_MODE=offline)."; fi

# ---- 3. ALFWorld game data (json_2.1.1/{train,valid_seen,valid_unseen}) ----
# config_tw.yaml / base_config.yaml paths are '$ALFWORLD_DATA/json_2.1.1/...'.
export ALFWORLD_DATA="/fsx/sfr/yefan.zhou/cache/alfworld"

# ---- 4. game_files startup-scan disk-cache (keyed on data_path → new sfr key = one fresh scan,
#         then sub-second; see docs/memcurator_startup_scan_optimization.md §8) ----
export ALFWORLD_GAMEFILES_CACHE_DIR="/fsx/sfr/yefan.zhou/mem-evolve/data/alfworld_gamefiles_cache"

# ---- misc: keep transient temp on fast sfr (interactive pip/tools); training/ray override separately ----
export TMPDIR="${TMPDIR:-/fsx/sfr/yefan.zhou/tmp}"

echo "[sfr_env] active: env=${SFR_CONDA_ENV}  HF_HOME=${HF_HOME}  ALFWORLD_DATA=${ALFWORLD_DATA}"
echo "[sfr_env] python -> $(command -v python3)"
