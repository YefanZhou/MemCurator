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
# libstdc++ etc. from the env (mirrors the launchers' LD_LIBRARY_PATH line, now env-relative).
export LD_LIBRARY_PATH="${SFR_CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"

# ---- 2. HuggingFace model cache (Qwen3 4B/4B-Instruct/8B/32B downloaded here) ----
# vllm + verl resolve models BY NAME (e.g. Qwen/Qwen3-8B) from $HF_HOME/hub — no path args needed.
export HF_HOME="/fsx/sfr/yefan.zhou/cache/huggingface"

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
