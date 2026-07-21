# MemCurator Stage-C SMOKE launcher — validate the training loop end-to-end on a TINY config.
#
# INDEPENDENT of scripts/train_memcurator_qwen3-8b-alfworld.sh (the real run): this is a throwaway
# smoke with everything turned down so it finishes in a few minutes and proves the wiring:
#   dataset pin -> reset -> retrieve from S_T -> actor briefing (two-pass thinking + loss mask)
#   -> executor episodes via env_manager -> DIRECT reward -> GRPO UID grouping -> PPO update.
# It is NOT a meaningful training run (2 steps, batch 4, n=2). If it goes green, run the real script.
#
# PREREQS you must have up first:
#   * A served FROZEN executor (nonthink Qwen3-8B), e.g.
#       CUDA_VISIBLE_DEVICES=... vllm serve Qwen/Qwen3-8B --port 8000 ...   -> EXECUTOR_API_BASE
#     (For a smoke it can share the box; it just needs to answer /v1/chat/completions.)
#   * Free GPU(s) for the verl actor (the curator being trained). NGPUS below.
#   * The Stage-B smoke dataset (already built):
#       /fsx/home/yefan.zhou/mem-evolve/data/datasets/smoke_dataset/dataset.jsonl
#
# Everything below is env-overridable so you can nudge it without editing the file.

export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/fsx/home/yefan.zhou/miniconda3/envs/memory/lib:$LD_LIBRARY_PATH
# Real-time output: force unbuffered stdout/stderr so progress (esp. the AlfredTWEnv game-file scan
# tqdm bar + Ray-forwarded driver logs) streams live to the console/log instead of appearing in
# big delayed chunks. PYTHONUNBUFFERED covers the driver + all Ray worker procs.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1        # dump a traceback on hard hangs/segfaults (SIGABRT), for debugging
export RAY_DEDUP_LOGS=0            # don't collapse identical Ray worker log lines (progress bars)
# NOTE: we deliberately do NOT `export TMPDIR=/fsx` in the launcher shell (SkillOS never did). The
# textworld libdownward-copy fix lives ONLY in runtime_env.yaml env_vars, so the WORKER actors (which
# do the per-step init_env) inherit TMPDIR=/fsx while `ray start` here keeps the local /tmp default.
# Exporting it here previously pushed Ray's session/dashboard dir onto /fsx → dashboard fail + job
# 504 under /fsx load (2026-07-20). The driver's AlfredTWEnv scan is os.walk+json only (no libdownward),
# so it does not need /fsx either. (The driver still gets TMPDIR=/fsx via runtime_env if it matters.)

#!/usr/bin/env bash
set -xeuo pipefail

# ---- Artifact root: EVERYTHING (ckpt, rollout dumps, logs) lives under
# mem-evolve/results/<exp>/<stamp>/, NEVER inside the SkillCurator-main repo (keeps the repo clean +
# off the ray working-dir upload). exp_name stays clean (used for W&B / hydra experiment_name); a
# per-run TIMESTAMP subdir makes each run auto-distinguishable so a re-run never clobbers the prior
# run's rollout/ckpt dumps (rollout files are step-keyed + "w"-truncated → same-dir would overwrite).
# Override RUN_STAMP to pin a run (e.g. RUN_STAMP=dumpenrich) or set RESULTS_ROOT/EXP_NAME as usual.
RESULTS_ROOT="${RESULTS_ROOT:-/fsx/home/yefan.zhou/mem-evolve/results}"
exp_name="${EXP_NAME:-memcurator-SMOKE-qwen3-8b-alfworld}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RESULTS_ROOT}/${exp_name}/${RUN_STAMP}"

# ---- Self-logging: mirror ALL output (this script + the ray-job-submit driver stdout, which
# `ray job submit` streams back to us since we don't pass --no-wait) to a timestamped file, so the
# run can be read after the fact regardless of how it was invoked. Override dir with SMOKE_LOG_DIR.
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-${RUN_DIR}/logs}"
mkdir -p "$SMOKE_LOG_DIR"
SMOKE_LOG="$SMOKE_LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
# Redirect stdout+stderr through tee, PREPENDING an HH:MM:SS timestamp to every line, so we can see
# which startup phase (config / AlfredTWEnv scan / NCCL / model load / MemCurator) takes how long.
# awk with fflush keeps it line-buffered/live. Carriage-return tqdm lines get a stamp too (fine).
exec > >(stdbuf -oL awk '{ "date +%H:%M:%S" | getline t; close("date +%H:%M:%S"); print t" | "$0; fflush() }' | tee -a "$SMOKE_LOG") 2>&1
echo "[smoke] logging to $SMOKE_LOG"

# ---- SMOKE knobs (tiny) ----
train_batch_size="${TRAIN_BATCH_SIZE:-4}"          # distinct targets/step = train_batch_size (n_slots//n)
val_batch_size="${VAL_BATCH_SIZE:-2}"
customized_grpo_rollout_n="${GRPO_N:-2}"           # 2 briefings per target (min for a GRPO group)
total_training_steps="${TOTAL_STEPS:-2}"           # just prove the loop turns
ppo_mini_batch_size="${PPO_MINI:-4}"               # <= train_batch_size; keep == train_batch_size
NGPUS="${NGPUS:-2}"                                # GPUs for the actor (2 is plenty for a smoke)
save_freq="${SAVE_FREQ:-100000}"                   # effectively never (skip ckpt churn)
test_freq="${TEST_FREQ:-100000}"                   # effectively never (skip val churn)

# ---- reward weights (v1: success + small validity term only) ----
compression_ratio_weight=0.0
function_content_reward_weight=0.0
function_call_reward_weight=0.0

# ---- MemCurator knobs ----
DATASET_PATH="${DATASET_PATH:-/fsx/home/yefan.zhou/mem-evolve/data/datasets/smoke_dataset/dataset.jsonl}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-openai/Qwen/Qwen3-8B}"
EXECUTOR_API_BASE="${EXECUTOR_API_BASE:-http://localhost:8000/v1}"
CURATOR_VARIANT="${CURATOR_VARIANT:-curator_alfworld_v1_api}"   # match the dataset build (v1/v1_api)
CURATION_MODE="${CURATION_MODE:-success_only_v1}"          # prompt-only variant of success_only (same
                                                           # store/mark semantics; richer curator system prompt).
                                                           # Store built success_only is compatible (v1 differs
                                                           # ONLY in the system prompt text, not storage/format).
RETRIEVE_NUM="${RETRIEVE_NUM:-3}"
HISTORY_LENGTH="${HISTORY_LENGTH:-3}"
CURATOR_ON_EMPTY="${CURATOR_ON_EMPTY:-false}"
# Executor = FROZEN, must match the frac0.5 harvest that built the pool:
#   nonthink, temp 1.0, max 4096 (ENABLE_THINKING=false EXECUTOR_TEMPERATURE=1.0 EXECUTOR_MAX_TOKENS=4096).
EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-4096}"
EXECUTOR_ENABLE_THINKING="${EXECUTOR_ENABLE_THINKING:-false}"
# top_p / top_k: match the frac0.5 harvest exactly (EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20). Without
# these the executor request omits them → vLLM defaults (top_p=1.0, top_k=-1), diverging from the
# sampling that generated the pool. top_k rides in extra_body (vLLM honors it).
EXECUTOR_TOP_P="${EXECUTOR_TOP_P:-0.95}"
EXECUTOR_TOP_K="${EXECUTOR_TOP_K:-20}"
# PROMPT_STYLE: MUST be revise_react to match the harvest (memcurator.alfworld_executor reads this
# env var when building the executor prompt). 'think' (the eval module default) collides with the
# nonthink executor. Exported below AND set in runtime_env.yaml env_vars so the WORKER actors inherit
# it (the prompt is built inside the worker's generation.py, not the driver).
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"

project_name='MemCurator'
# exp_name defined above (RUN_DIR anchor). Keep here for readability.

export BASE_MODEL='Qwen/Qwen3-8B'
export EXPERIMENT_NAME="${exp_name}"

# Rollout dumps (training + validation .jsonl) go under the run dir, NOT the repo.
export ROLLOUT_DATA_DIR="${RUN_DIR}/rollout"
# math parquet is loaded but UNUSED for alfworld (tasks come from the dataset/env); any valid path works.
TRAIN_DATA_DIR='/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/math/'
TEST_DATA_DIR='/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/math/'

# Algorithm
adv_estimator=grpo
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
use_kl_in_reward=False
kl_coef=0.0

# Sizes / lengths (same as real run — these are model limits, not smoke knobs)
max_prompt_length=32768
max_response_length=8192            # CURATOR (actor) response budget — thinking curator needs room
max_start_length=16384
max_obs_length=500
enable_thinking=true
max_turns=5

RAY_ADDRESS=${RAY_ADDRESS:-""}
WORKING_DIR="/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1
MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}
CKPTS_DIR=${CKPTS_DIR:-"${RUN_DIR}/ckpt"}

temperature=1.0                     # CURATOR actor sampling temp
top_p=1.0
top_k=-1
offload=true

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

MASTER_ADDR="localhost"
if [ -z "$RAY_ADDRESS" ]; then
    RAY_ADDRESS="http://${MASTER_ADDR}:8265"
fi

# Ray SESSION/dashboard dir MUST be on local disk, NOT /fsx. ROOT CAUSE (found 2026-07-20): the
# user's ~/.bashrc exports TMPDIR=/fsx/home/yefan.zhou/tmp for every login shell, so `ray start`
# inherits it and puts its session dir on the degraded /fsx/home volume → worker-actor registration
# times out (init_workers TimeoutError: Failed to get register_center_actor) OR the dashboard 504s.
# Removing the launcher's own export was NOT enough — the export lives in .bashrc. --temp-dir is the
# only way to override it here without editing the user's shell profile. Local /tmp = SkillOS default.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${USER:-yz}}"
mkdir -p "$RAY_TEMP_DIR"

echo "Starting Ray head (SMOKE, single node, ${NGPUS} GPUs); ray temp-dir=${RAY_TEMP_DIR}"
ray stop || true
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir="${RAY_TEMP_DIR}"
sleep 8

ray job submit --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -u -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_DATA_DIR}/train_grouped.parquet" \
    data.val_files="${TEST_DATA_DIR}/test_paired.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.max_start_length=${max_start_length} \
    data.max_obs_length=${max_obs_length} \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    trainer.logger=['console'] \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}/verl_rollout/training" \
    trainer.val_only=false \
    trainer.val_before_train=false \
    trainer.validation_data_dir="${ROLLOUT_DATA_DIR}/verl_rollout/validation" \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable \
    reward_model.compression_ratio_weight=${compression_ratio_weight} \
    reward_model.function_content_reward_weight=${function_content_reward_weight} \
    +reward_model.function_call_reward_weight=${function_call_reward_weight} \
    algorithm.norm_adv_by_std_in_grpo=false \
    enable_thinking=${enable_thinking} \
    max_turns=${max_turns} \
    use_memory_mode=true \
    do_search=true \
    customized_grpo_rollout_n=${customized_grpo_rollout_n} \
    +curator_mode=memcurator \
    +memcurator.dataset_path="${DATASET_PATH}" \
    +memcurator.curator_variant="${CURATOR_VARIANT}" \
    +memcurator.curation_mode="${CURATION_MODE}" \
    +memcurator.retrieve_num=${RETRIEVE_NUM} \
    +memcurator.executor_model="${EXECUTOR_MODEL}" \
    +memcurator.executor_api_base="${EXECUTOR_API_BASE}" \
    +memcurator.executor_temperature=${EXECUTOR_TEMPERATURE} \
    +memcurator.executor_top_p=${EXECUTOR_TOP_P} \
    +memcurator.executor_top_k=${EXECUTOR_TOP_K} \
    +memcurator.executor_max_tokens=${EXECUTOR_MAX_TOKENS} \
    +memcurator.executor_enable_thinking=${EXECUTOR_ENABLE_THINKING} \
    +memcurator.history_length=${HISTORY_LENGTH} \
    +memcurator.curator_on_empty=${CURATOR_ON_EMPTY} \
    +alfworld.num_tasks=1 \
    +alfworld.val_tasks=1 \
    +alfworld.max_steps=30 \
    +alfworld.same_task_type_per_chain=false \
    +trainer.total_training_steps=${total_training_steps} \
    env.env_name="alfworld/AlfredTWEnv" \
    env.seed=42 \
    env.rollout.n=${customized_grpo_rollout_n}

echo "SMOKE job submitted. Watch the console output above; on success you'll see ${total_training_steps} steps complete."
ray stop || true
