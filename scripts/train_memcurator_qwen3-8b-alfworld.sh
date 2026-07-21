# MemCurator (read-time briefing curator) GRPO training on ALFWorld.
#
# Trains the curator that eval runs as `run_unified_dev_async_curator.py --memory_type curator`
# (curator_alfworld.CuratorAlfworld). Adapted from train_memory_grpo_qwen3-8b-alfworld-no-both.sh
# (the SkillOS write-time curator trainer). Differences from that script:
#   * +curator_mode=memcurator            -> routes ray_trainer_alfworld to MemCuratorGenerationManager
#   * +memcurator.* block                 -> frozen pool, executor endpoint, retrieve_num, history
#   * reward = success + f*validity only  -> compression_ratio_weight=0, function_content_weight=0,
#                                            analyze_function_url unset (no Qwen judge)
#   * norm_adv_by_std_in_grpo=false       -> degenerate (all-equal) groups give zero advantage
#                                            harmlessly (matches SkillOS/sea-mem precedent; no DAPO drop)
#   * NO chains: alfworld.num_tasks=1      -> single task per step (DIRECT credit, no shift)
#
# Prereqs (serve these first, on separate replicas):
#   * EXECUTOR (frozen, temp 0.7): vllm serve Qwen/Qwen3-8B --port 8000   -> EXECUTOR_API_BASE
#   * CURATOR = the actor policy being trained (verl loads MODEL_PATH; no serving needed).
#   * POOL_PATH: a harvested curator_memory.jsonl (frozen global pool). Build one by running the
#     eval runner --memory_type curator (or none/cold) to collect successful trajectories, then
#     optionally `python -m memcurator.build_curator_stores` (v2 per-task stores). For v1 the raw
#     harvested curator_memory.jsonl IS the global pool.

# Vertex AI (unused for pure-vLLM executor; kept for parity with base script env)
export HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH=/fsx/home/yefan.zhou/miniconda3/envs/memory/lib:$LD_LIBRARY_PATH

#!/usr/bin/env bash
set -xeuo pipefail

# ---- MemCurator reward weights (v1: success + small validity term only) ----
compression_ratio_weight=0.0
function_content_reward_weight=0.0
function_call_reward_weight=0.1     # weights the briefing-validity term (1.0 if non-empty briefing)

# ---- MemCurator knobs ----
# DATASET_PATH = Stage B output (dataset.jsonl): per-target rows + frozen S_T stores. This is the
# controllable dataset-driven path (pins each slot to its target game). Build it with:
#   python -m memcurator.sample_and_select --stratify --frac 0.1 --k 8 ... --out_dir <A>
#   python -m memcurator.build_dataset --stage_a_dir <A> --out_dir <B> --pool_size 10 ...
DATASET_PATH="${DATASET_PATH:-/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/memcurator/dataset_v1/dataset.jsonl}"
EXECUTOR_MODEL="${EXECUTOR_MODEL:-openai/Qwen/Qwen3-8B}"
EXECUTOR_API_BASE="${EXECUTOR_API_BASE:-http://localhost:8000/v1}"
RETRIEVE_NUM="${RETRIEVE_NUM:-3}"
HISTORY_LENGTH="${HISTORY_LENGTH:-3}"
CURATOR_ON_EMPTY="${CURATOR_ON_EMPTY:-false}"

project_name='MemCurator'
exp_name="memcurator-qwen3-8b-alfworld-v1-fcw${function_call_reward_weight}"

export WANDB_PROJECT="${project_name}"
export WANDB_NAME="${exp_name}"
export BASE_MODEL='Qwen/Qwen3-8B'
export EXPERIMENT_NAME="${exp_name}"

# ---- Artifact root: EVERYTHING (ckpt, rollout dumps, logs) lives under
# mem-evolve/results/<exp>/<stamp>/, NEVER inside the SkillCurator-main repo. Per-run TIMESTAMP
# subdir makes each run auto-distinguishable so a re-run never clobbers prior rollout/ckpt dumps.
# Override RUN_STAMP to pin a run; note that RESUMING a run needs the SAME RUN_STAMP (ckpt lives
# under it) — pass RUN_STAMP=<the-original-stamp> to resume, else it starts fresh in a new dir.
RESULTS_ROOT="${RESULTS_ROOT:-/fsx/home/yefan.zhou/mem-evolve/results}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RESULTS_ROOT}/${exp_name}/${RUN_STAMP}"
export ROLLOUT_DATA_DIR="${RUN_DIR}/rollout"
TRAIN_DATA_DIR='/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/math/'   # unused for alfworld (tasks from env.reset)
TEST_DATA_DIR='/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/data/math/'

# Algorithm
adv_estimator=grpo
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
use_kl_in_reward=False
kl_coef=0.0

# Sizes / lengths
max_prompt_length=16384
max_response_length=4096
max_start_length=16384
max_obs_length=500
enable_thinking=true                # curator thinking on (matches jul-12 curator setup)
max_turns=5
customized_grpo_rollout_n=8         # GRPO group size = n briefings for the SAME task
train_batch_size=32
val_batch_size=32

RAY_ADDRESS=${RAY_ADDRESS:-""}
WORKING_DIR="/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-${PET_NNODES:-1}}
MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}
CKPTS_DIR=${CKPTS_DIR:-"${RUN_DIR}/ckpt"}

temperature=1.0                     # CURATOR actor sampling temp (executor temp set via memcurator.executor_temperature)
top_p=1.0
top_k=-1
offload=true

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${PET_MASTER_ADDR:-"localhost"}
if [ "$MASTER_ADDR" != "localhost" ]; then
    RESOLVED_IP=$(getent hosts "$MASTER_ADDR" | awk '{ print $1 }' | head -1)
    [ -n "$RESOLVED_IP" ] && MASTER_ADDR="$RESOLVED_IP"
fi
if [ -z "$RAY_ADDRESS" ]; then
    RAY_ADDRESS="http://${MASTER_ADDR}:8265"
fi

# Ray SESSION dir on LOCAL disk, NOT /fsx. ROOT CAUSE (2026-07-20): ~/.bashrc exports
# TMPDIR=/fsx/home/…/tmp for every login shell → `ray start` inherits it → session dir on the
# degraded /fsx/home → worker-actor registration times out / dashboard 504s. --temp-dir overrides
# the inherited TMPDIR without editing the shell profile. Per-node local dir; SkillOS default = /tmp.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_${USER:-yz}}"
mkdir -p "$RAY_TEMP_DIR"

if [ "$NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head on master node (rank $NODE_RANK); ray temp-dir=${RAY_TEMP_DIR}"
    ray stop || true
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir="${RAY_TEMP_DIR}"
    sleep 10

    ray job submit --runtime-env="${RUNTIME_ENV}" \
        --working-dir "${WORKING_DIR}" \
        -- python3 -m verl.trainer.main_ppo \
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
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
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
        trainer.logger=['console','wandb'] \
        trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}/training" \
        trainer.val_only=false \
        trainer.val_before_train=false \
        trainer.validation_data_dir="${ROLLOUT_DATA_DIR}/validation" \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq=10 \
        trainer.test_freq=5 \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.total_epochs=5 \
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
        +memcurator.curator_variant="${CURATOR_VARIANT:-curator_alfworld}" \
        +memcurator.curation_mode="${CURATION_MODE:-success_only_v1}" \
        +memcurator.retrieve_num=${RETRIEVE_NUM} \
        +memcurator.executor_model="${EXECUTOR_MODEL}" \
        +memcurator.executor_api_base="${EXECUTOR_API_BASE}" \
        +memcurator.executor_temperature=0.7 \
        +memcurator.history_length=${HISTORY_LENGTH} \
        +memcurator.curator_on_empty=${CURATOR_ON_EMPTY} \
        +alfworld.num_tasks=1 \
        +alfworld.val_tasks=1 \
        +alfworld.max_steps=30 \
        +alfworld.same_task_type_per_chain=false \
        +trainer.total_training_steps=60 \
        env.env_name="alfworld/AlfredTWEnv" \
        env.seed=42 \
        env.rollout.n=${customized_grpo_rollout_n}

    echo "Job submitted. Keeping Ray head alive..."
    while true; do echo "$(date): Master alive"; sleep 60; done
else
    echo "Starting Ray worker on node rank $NODE_RANK; ray temp-dir=${RAY_TEMP_DIR}"
    ray stop || true
    ray start --address="${MASTER_ADDR}:6379" --temp-dir="${RAY_TEMP_DIR}"
    while true; do
        echo "$(date): Worker alive"; sleep 60
        ray status >/dev/null 2>&1 || ray start --address="${MASTER_ADDR}:6379" --temp-dir="${RAY_TEMP_DIR}"
    done
fi
