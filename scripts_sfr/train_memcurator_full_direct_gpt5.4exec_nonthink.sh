# MemCurator Stage-C FULL TRAINING RUN — dashboard-free _direct path, full-scale knobs.
# Derived from train_memcurator_smoke_direct_val_add.sh (same wiring: sfr env, HF-token +
# wandb-online passthrough via sfr_env.sh, PAIRED evolving val, no Ray dashboard). Only the
# SCALE defaults differ (batch 32, rollout 8, 100 steps, TEST_FREQ 5, full paired dev manifest,
# checkpointing on). Override any knob via env vars at launch.
#
# ===================================================================================================
# ABLATION #1 x #4: NON-THINKING CURATOR + GPT-5.4 EXECUTOR.
#   This is a copy of train_memcurator_full_direct_gpt5.4exec.sh with ONE ablation change:
#   the curator actor generates briefings WITHOUT <think> (enable_thinking=false, L168 below).
#   Everything else (GPT-5.4 gateway executor, the max_tokens-omit fix, dataset, evolving val)
#   is byte-identical to the thinking GPT-5.4 launcher.
#
#   Why it's safe (verified 2026-07-22): _generate_briefings runs its two-pass
#   think-force + _SENTENCE_TO_MASK loss-mask path ONLY when enable_thinking=True. With false it
#   skips that path — the whole response IS the briefing, no think-segment to mask. RL loss stays
#   well-formed. VERIFY with a 2-step smoke: actor/pg_loss non-zero, briefings non-empty + contain
#   no <think>.
#
#   Conceptual: Qwen3 (the curator) leans on <think>; a non-thinking curator likely produces WEAKER
#   briefings. This is an ABLATION ("does the curator need to think?"), NOT expected to beat the
#   thinking GPT-5.4 run. EVAL PARITY: the trained ckpt must be eval'd with the curator served
#   non-thinking too — its own baseline; do NOT compare its val curve to the thinking runs' numbers.
# ===================================================================================================
#
# Same dashboard-free SFR training smoke, PLUS the faithful evolving held-out validation
# (_run_eval_loop_alfworld): a val manifest of held-out valid_seen games in the eval runner's exact
# order, run with online-growing per-lane memory (cold -> retrieve -> brief -> execute -> writeback).
#
# The 5 val additions over the base smoke:
#   1. +memcurator.val_manifest_path=<smoke manifest>  -> dispatches to the evolving-val loop
#   2. +memcurator.val_n_lanes=2                        -> 2 independent-store lanes (avg for variance)
#   3. data.val_batch_size = n_lanes * group_size       -> the wide batch the val env_manager needs
#   4. rollout.val_kwargs.{temperature,top_p,top_k}=0.6/0.95/20 -> curator SAMPLES (not greedy;
#      Qwen3 thinking degrades under greedy) at the test-time operating point. The SPMD rollout reads
#      the val temp from val_kwargs (config), so it MUST be set here.
#   5. TEST_FREQ=1                                      -> run validation each step to exercise it
#
# Uses the TINY smoke manifest (val_manifest_smoke.jsonl: 2 seeds x 2 batches x 5 games) so a
# validation is ~2-3 min, not ~20. Swap VAL_MANIFEST_PATH + VAL_BATCH_SIZE for the full run.
#
# Usage (box1/box2, executor served on :8001, sfr env active via .bashrc):
#   export OPENAI_API_KEY=<gateway key>   # or EXECUTOR_API_KEY
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPUS=8 \    # GPT executor is REMOTE -> no local server -> all 8 GPUs train
#   bash scripts_sfr/train_memcurator_full_direct_gpt5.4exec_nonthink.sh
# (old Qwen layout for reference: CUDA_VISIBLE_DEVICES=4,5,6,7 EXECUTOR_API_BASE=http://localhost:8001/v1 NGPUS=4)
#     bash scripts_sfr/train_memcurator_smoke_direct_val_add.sh

#!/usr/bin/env bash
set -xeuo pipefail

source "$(dirname "$0")/../scripts/sfr_env.sh"

export HYDRA_FULL_ERROR=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTHONPATH="/fsx/sfr/yefan.zhou/mem-evolve/SkillCurator-main:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=0
export PROMPT_STYLE="${PROMPT_STYLE:-revise_react}"
export VLLM_ATTENTION_BACKEND=XFORMERS
# Suppress harmless pydantic-serializer UserWarnings (litellm Message/Choices extra fields,
# surfaced when wandb logs executor responses; ~298/run of noise, zero behavior impact).
# Module-scoped so real warnings stay visible; exported so ray workers inherit it.
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:pydantic.main}"

# ---- Artifact root ----
RESULTS_ROOT="${RESULTS_ROOT:-/fsx/home/yefan.zhou/mem-evolve/results}"
exp_name="${EXP_NAME:-memcurator-FULL-gpt5.4exec-alfworld-nonthink-iter1}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RESULTS_ROOT}/${exp_name}/${RUN_STAMP}"

SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-${RUN_DIR}/logs}"
mkdir -p "$SMOKE_LOG_DIR"
SMOKE_LOG="$SMOKE_LOG_DIR/smoke_$(date +%Y%m%d_%H%M%S).log"
exec > >(stdbuf -oL awk '{ "date +%H:%M:%S" | getline t; close("date +%H:%M:%S"); print t" | "$0; fflush() }' | tee -a "$SMOKE_LOG") 2>&1
echo "[smoke-valadd] logging to $SMOKE_LOG"

# ---- SMOKE knobs ----
train_batch_size="${TRAIN_BATCH_SIZE:-32}"
customized_grpo_rollout_n="${GRPO_N:-8}"
total_training_steps="${TOTAL_STEPS:-100}"
ppo_mini_batch_size="${PPO_MINI:-32}"
NGPUS="${NGPUS:-8}"   # GPT executor is REMOTE (gateway) -> no local vLLM server -> curator/trainer uses ALL 8 GPUs
save_freq="${SAVE_FREQ:-5}"
test_freq="${TEST_FREQ:-5}"                         # run evolving val every 5 steps

# ---- W&B logging (Task 5) ----
# WANDB=1 (default) -> logger backends "console,wandb"; WANDB=0 -> console only.
# The val/* + train/* alfworld metrics reach wandb automatically via the single logger.log in
# ray_trainer_alfworld.py — no code change needed.
#
# IMPORTANT — this is a *_direct launcher: it uses `ray start` + direct `python -m ...`, so it does
# NOT read verl/trainer/runtime_env.yaml. Env vars must be EXPORTED here (the ray head + driver +
# workers inherit them) — a var placed only in runtime_env.yaml is DEAD on this path. So WANDB_MODE
# and WANDB_DIR are exported below (runtime_env.yaml still carries them for the `ray job submit`
# launchers — harmless duplication).
#  - OFFLINE by default: sync later with `wandb sync ${WANDB_DIR}/wandb/offline-run-*`. Set
#    WANDB_MODE=online (+ export WANDB_API_KEY) once connectivity is confirmed.
#  - WANDB_DIR on fast /fsx/sfr, NOT /fsx/home (buggy FS).
WANDB="${WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-/fsx/sfr/yefan.zhou/mem-evolve/wandb}"
mkdir -p "$WANDB_DIR"
if [ "$WANDB" = "1" ]; then TRAINER_LOGGER_ARG="['console','wandb']"; else TRAINER_LOGGER_ARG="['console']"; fi

# ---- reward weights ----
compression_ratio_weight=0.0
function_content_reward_weight=0.0
function_call_reward_weight=0.0

# ---- MemCurator knobs ----
DATASET_PATH="${DATASET_PATH:-/fsx/sfr/yefan.zhou/mem-evolve/data/datasets/full_dataset_gpt5.4_frac0.2_iter1/dataset.jsonl}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-openai/gpt-5.4}"
EXECUTOR_API_BASE="${EXECUTOR_API_BASE:-https://gateway.salesforceresearch.ai/openai/process/v1/}"
# gateway key: EXPORT EXECUTOR_API_KEY (or OPENAI_API_KEY) at launch; never commit it.
EXECUTOR_API_KEY="${EXECUTOR_API_KEY:-${OPENAI_API_KEY:-}}"
CURATOR_VARIANT="${CURATOR_VARIANT:-curator_alfworld_v1_api}"
CURATION_MODE="${CURATION_MODE:-success_only_v1}"
RETRIEVE_NUM="${RETRIEVE_NUM:-3}"
HISTORY_LENGTH="${HISTORY_LENGTH:-3}"
CURATOR_ON_EMPTY="${CURATOR_ON_EMPTY:-false}"
EXECUTOR_TEMPERATURE="${EXECUTOR_TEMPERATURE:-1.0}"
EXECUTOR_MAX_TOKENS="${EXECUTOR_MAX_TOKENS:-}"   # EMPTY for GPT reasoning models (they reject max_tokens; would need max_completion_tokens)
# Cap concurrent executor gateway requests (default 64). The 256-slot fan-out would hammer
# the GPT gateway -> 429s -> silent Output Error->reward-0. 64 is a safe start; raise if the
# gateway is stable (watch the log for "Output Error"), lower if you see throttling.
EXECUTOR_CONCURRENCY="${EXECUTOR_CONCURRENCY:-64}"
EXECUTOR_ENABLE_THINKING="${EXECUTOR_ENABLE_THINKING:-}"   # EMPTY for GPT (no chat_template_kwargs)
EXECUTOR_TOP_P="${EXECUTOR_TOP_P:-}"                     # EMPTY for GPT (gateway default)
EXECUTOR_TOP_K="${EXECUTOR_TOP_K:-}"                     # EMPTY for GPT (no top_k)

# ---- EVOLVING-VAL knobs (#5) ----
VAL_MANIFEST_PATH="${VAL_MANIFEST_PATH:-/fsx/sfr/yefan.zhou/mem-evolve/data/val_manifest_dev_paired.jsonl}"
VAL_N_LANES="${VAL_N_LANES:-2}"                     # ADD #2: 2 independent-store lanes
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-10}"              # must match the manifest's group_size (dev=10)
# ADD #3: val_batch_size = n_lanes * group_size (the wide batch); env_manager sizes its val workers to this.
val_batch_size="${VAL_BATCH_SIZE:-$((VAL_N_LANES * VAL_GROUP_SIZE))}"
# ADD #4: curator sampling at val = eval operating point (thinking-safe; NOT greedy).
VAL_CURATOR_TEMPERATURE="${VAL_CURATOR_TEMPERATURE:-0.6}"
VAL_CURATOR_TOP_P="${VAL_CURATOR_TOP_P:-0.95}"
VAL_CURATOR_TOP_K="${VAL_CURATOR_TOP_K:-20}"
VAL_WRITEBACK_SUCCESS_ONLY="${VAL_WRITEBACK_SUCCESS_ONLY:-true}"
# val executor: THIS RUN uses the SAME executor as train (served Qwen3-8B). Leave the VAL_EXECUTOR_*
# vars unset -> the val_executor_* hydra args are OMITTED entirely below -> config defaults (None) ->
# _call_executor_batch falls back to the training executor. To point val at an API model instead,
# set VAL_EXECUTOR_MODEL / VAL_EXECUTOR_API_BASE / VAL_EXECUTOR_API_KEY and the args get added.
# Train-executor OPTIONAL args: pass Qwen-only sampling params ONLY when NON-empty. For GPT these are
# EMPTY -> omitted -> the gateway gets a clean OpenAI request (no top_k/top_p/chat_template_kwargs).
EXEC_EXTRA_ARGS=()
[ -n "$EXECUTOR_API_KEY" ]         && EXEC_EXTRA_ARGS+=("+memcurator.executor_api_key=${EXECUTOR_API_KEY}")
[ -n "$EXECUTOR_TOP_P" ]           && EXEC_EXTRA_ARGS+=("+memcurator.executor_top_p=${EXECUTOR_TOP_P}")
[ -n "$EXECUTOR_MAX_TOKENS" ]      && EXEC_EXTRA_ARGS+=("+memcurator.executor_max_tokens=${EXECUTOR_MAX_TOKENS}")
[ -n "$EXECUTOR_TOP_K" ]           && EXEC_EXTRA_ARGS+=("+memcurator.executor_top_k=${EXECUTOR_TOP_K}")
[ -n "$EXECUTOR_ENABLE_THINKING" ] && EXEC_EXTRA_ARGS+=("+memcurator.executor_enable_thinking=${EXECUTOR_ENABLE_THINKING}")

# VAL executor defaults to the SAME GPT-5.4 gateway (val uses GPT-5.4 too). Override VAL_EXECUTOR_* to change.
VAL_EXECUTOR_MODEL="${VAL_EXECUTOR_MODEL:-${EXECUTOR_MODEL}}"
VAL_EXECUTOR_API_BASE="${VAL_EXECUTOR_API_BASE:-${EXECUTOR_API_BASE}}"
VAL_EXECUTOR_API_KEY="${VAL_EXECUTOR_API_KEY:-${EXECUTOR_API_KEY}}"
VAL_EXEC_ARGS=()
[ -n "$VAL_EXECUTOR_MODEL" ]    && VAL_EXEC_ARGS+=("+memcurator.val_executor_model=${VAL_EXECUTOR_MODEL}")
[ -n "$VAL_EXECUTOR_API_BASE" ] && VAL_EXEC_ARGS+=("+memcurator.val_executor_api_base=${VAL_EXECUTOR_API_BASE}")
[ -n "$VAL_EXECUTOR_API_KEY" ]  && VAL_EXEC_ARGS+=("+memcurator.val_executor_api_key=${VAL_EXECUTOR_API_KEY}")

project_name='MemCurator'
export BASE_MODEL='Qwen/Qwen3-8B'
export EXPERIMENT_NAME="${exp_name}"

export ROLLOUT_DATA_DIR="${RUN_DIR}/rollout"
TRAIN_DATA_DIR='/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/math/'
TEST_DATA_DIR='/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/math/'

# Algorithm
adv_estimator=grpo
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
use_kl_in_reward=False
kl_coef=0.0

# Sizes / lengths
max_prompt_length=32768
max_response_length=8192
max_start_length=16384
max_obs_length=500
# ABLATION #1: non-thinking CURATOR (train+val). Flips the actor to generate briefings
# WITHOUT <think>; _generate_briefings skips the two-pass think-force/loss-mask path.
# Executor thinking is separate (EXECUTOR_ENABLE_THINKING, already empty for GPT).
enable_thinking=false
max_turns=5

WORKING_DIR="/fsx/sfr/yefan.zhou/mem-evolve/SkillCurator-main"
NNODES=1
MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}
CKPTS_DIR=${CKPTS_DIR:-"${RUN_DIR}/ckpt"}

temperature=1.0                                     # TRAIN curator temp (exploration)
top_p=1.0
top_k=-1
offload=true

# ---- Ray cluster: NO dashboard, session on local /tmp. ----
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${USER:-yz}}"
mkdir -p "$RAY_TEMP_DIR"

echo "Starting Ray head (SMOKE-VALADD, no dashboard, ${NGPUS} GPUs); ray temp-dir=${RAY_TEMP_DIR}"
echo "[smoke-valadd] evolving val: manifest=${VAL_MANIFEST_PATH} n_lanes=${VAL_N_LANES} "\
"val_batch_size=${val_batch_size} (=${VAL_N_LANES}x${VAL_GROUP_SIZE}) curator_temp=${VAL_CURATOR_TEMPERATURE}"
ray stop || true
ray start --head --include-dashboard=false --num-gpus="${NGPUS}" --temp-dir="${RAY_TEMP_DIR}"
sleep 8

export RAY_ADDRESS="auto"

cd "${WORKING_DIR}"
python3 -u -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_CURATOR_TEMPERATURE} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_CURATOR_TOP_P} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_CURATOR_TOP_K} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    trainer.logger="${TRAINER_LOGGER_ARG}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}/verl_rollout/training" \
    trainer.val_only=false \
    trainer.val_before_train=true \
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
    +memcurator.executor_concurrency=${EXECUTOR_CONCURRENCY} \
    +memcurator.history_length=${HISTORY_LENGTH} \
    +memcurator.curator_on_empty=${CURATOR_ON_EMPTY} \
    +memcurator.val_manifest_path="${VAL_MANIFEST_PATH}" \
    +memcurator.val_n_lanes=${VAL_N_LANES} \
    +memcurator.val_curator_temperature=${VAL_CURATOR_TEMPERATURE} \
    +memcurator.val_curator_top_p=${VAL_CURATOR_TOP_P} \
    +memcurator.val_curator_top_k=${VAL_CURATOR_TOP_K} \
    +memcurator.val_writeback_success_only=${VAL_WRITEBACK_SUCCESS_ONLY} \
    ${EXEC_EXTRA_ARGS[@]+"${EXEC_EXTRA_ARGS[@]}"} \
    ${VAL_EXEC_ARGS[@]+"${VAL_EXEC_ARGS[@]}"} \
    +alfworld.num_tasks=1 \
    +alfworld.val_tasks=1 \
    +alfworld.max_steps=30 \
    +alfworld.same_task_type_per_chain=false \
    +trainer.total_training_steps=${total_training_steps} \
    env.env_name="alfworld/AlfredTWEnv" \
    env.seed=42 \
    env.rollout.n=${customized_grpo_rollout_n}

echo "FULL run done; ${total_training_steps} steps + evolving paired val every ${test_freq} steps."
ray stop || true
