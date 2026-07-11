



export OPENAI_API_KEY="<your gateway key>"
export OPENAI_API_BASE="https://gateway.salesforceresearch.ai/openai/process/v1/"

python run_unified.py \
  --env            alfworld \
  --memory_type    skillos \
  --model          openai/gpt-5-mini-2025-08-07 \
  --curation_model openai/gpt-5-mini-2025-08-07 \
  --curation_base_url "https://gateway.salesforceresearch.ai/openai/process/v1/" \
  --max_steps      30 \
  --exp_name       gateway_gpt5mini_run1




pip install -r evaluation/requirements.txt
pip install 'openai==1.78.1'          # eval reqs downgrade to 1.75 — restore for vllm
pip install math_verify               # only if running reasoning envs (aime/amc/gpqa)
pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld==0.4.2
pip check
export ALFWORLD_DATA=/fsx/home/yefan.zhou/.cache/alfworld
alfworld-download -f



# thinking
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --port 8002 --served-model-name Qwen/Qwen3-8B --dtype bfloat16

# nonthinking
vllm serve Qwen/Qwen3-8B --port 8001 --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --chat-template /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/qwen3_nothink.jinja


python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline




conda activate memory
export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
cd $HOME/mem-evolve/SkillCurator-main/evaluation/agent_eval


# python run_unified.py --env alfworld --memory_type none \
#     --model openai/Qwen/Qwen3-8B --exp_name baseline_max2 --num_games 2

export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_try1_hist5



export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8002/v1"
python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_max2_try2_thinking \
    --num_games 2 --batch_size 1 2>&1 | tee logs/run_baseline_max2_thinking.log





export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_try2_hist5 2>&1 | tee logs/baseline_try2_hist5.log



export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8002/v1"

python run_unified_nonthink.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_try3_hist5_nonthink --num_games 2 --batch_size 1  2>&1 | tee logs/baseline_nonthink_max2.log





conda activate memory
export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"


conda activate memory
export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"




python run_memp_ori.py \
    --model openai/Qwen/Qwen3-8B \
    --split dev \
    --batch_size 10 \
    --max_steps 30 \
    --exp_name memp_ori_nomem 2>&1 | tee logs/memp_ori_nomem_full.log




conda activate memory
export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8002/v1"


python run_memp_ori.py \
    --model openai/Qwen/Qwen3-8B \
    --split dev \
    --batch_size 10 \
    --max_steps 30 \
    --exp_name memp_ori_nomem_thinking 2>&1 | tee logs/memp_ori_nomem_full_thinking.log







vllm serve Qwen/Qwen3-8B --port 8001 --served-model-name Qwen/Qwen3-8B --dtype bfloat16



export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"

python run_unified_nonthink.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_smokerun_try3_hist5_nonthink --num_games 2 --batch_size 1  2>&1 | tee logs/baseline_smokerun_max2_hist5_nonthink.log







export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
cd $HOME/mem-evolve/SkillCurator-main/evaluation/agent_eval

# smoke test first (2 games)
python run_unified_nonthink_revise_react.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_react_nothink_smoke_revise_react \
    --num_games 2 --batch_size 1 --overwrite 2>&1 | tee logs/baseline_react_nothink_smoke_revise_react.log





export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
cd $HOME/mem-evolve/SkillCurator-main/evaluation/agent_eval

python run_unified_nonthink_revise_react.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_nothink_smoke_revise_react_full \
    --batch_size 10 2>&1 | tee logs/baseline_nothink_smoke_revise_react_full.log


for i in 2 3; do
    export ALFWORLD_DATA=$HOME/.cache/alfworld
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_API_BASE="http://localhost:8002/v1"
    python run_unified_nonthink_revise_react.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_nothink_smoke_revise_react_full_run${i} \
        --batch_size 20 2>&1 | tee logs/baseline_nothink_smoke_revise_react_full_run${i}.log

    done



for i in 2 3; do
    export ALFWORLD_DATA=$HOME/.cache/alfworld
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_API_BASE="http://localhost:8002/v1"
    python run_unified_nonthink_reason_tag.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_reasontag_nothink_run${i} \
        --batch_size 20 2>&1 | tee logs/baseline_reasontag_nothink_run${i}.log
done


CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-8B \
    --port 8001 \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --data-parallel-size 4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-8B \
    --port 8001 \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
     --max-model-len 40960 \
     --gpu-memory-utilization 0.90


## another server

conda activate memory

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-8B \
#     --port 8001 \
#     --served-model-name Qwen/Qwen3-8B \
#     --dtype bfloat16 \
#     --data-parallel-size 8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-8B \
  --port 8001 \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90



export ALFWORLD_DATA=$HOME/.cache/alfworld
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://localhost:8001/v1"
cd $HOME/mem-evolve/SkillCurator-main/evaluation/agent_eval

# smoke test first (2 games)
python run_unified.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_thinking_8b_temp0.7_4096_run1_smoke \
    --batch_size 1 --num_games 1 2>&1 | tee logs/baseline_thinking_8b_temp0.7_4096_run1_smoke.log
    



ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8001/v1" EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
python run_unified_hyper.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_hyper_temp0.6_top_p_0.95_top_k_20_max_4096_run1 --batch_size 30 \
    2>&1 | tee logs/baseline_hyper_temp0.6_top_p_0.95_top_k_20_max_4096_run1.log



ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8001/v1" EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
python run_unified_hyper_reason_tag.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_hyper_reason_tag_temp0.6_top_p_0.95_top_k_20_max_4096_run1 --batch_size 30 \
    2>&1 | tee logs/baseline_hyper_reason_tag_temp0.6_top_p_0.95_top_k_20_max_4096_run1.log



ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 \
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=false \
python run_unified_hyper_concurrent.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_conc_run1 --batch_size 64 \
    2>&1 | tee logs/baseline_conc_run1.log





cd $HOME/mem-evolve/SkillCurator-main/evaluation/agent_eval



ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
python -u run_unified_hyper_async.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_think_async_try2_run1 --concurrency 64 \
    2>&1 | tee logs/baseline_think_async_try2_run1.log


for i in {1..3}; do
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
    python -u run_unified_hyper_async_reason_tag.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_think_reason_tag_try2_run1_run${i} --concurrency 64 \
        2>&1 | tee logs/baseline_think_reason_tag_try2_run1_run${i}.log

    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
    python -u run_unified_hyper_async_revise_react.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_think_revise_react_try2_run1_run${i} --concurrency 64 \
        2>&1 | tee logs/baseline_think_revise_react_try2_run1_run${i}.log
done


#.log
for i in {2..3}; do
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
    python -u run_unified_hyper_async.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_think_pure_async_try2_run${i} --concurrency 64 \
        2>&1 | tee logs/baseline_think_pure_async_try2_run${i}.log
done


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-32B \
  --port 8001 \
  --data-parallel-size 8 \
  --tensor-parallel-size 1 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90


for i in {1..3}; do
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
    python -u run_unified_hyper_async.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_think_pure_async_32b_run${i} --concurrency 64 \
        2>&1 | tee logs/baseline_think_pure_async_32b_run${i}.log
done




for i in {1..3}; do
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
    python -u run_unified_hyper_async_reason_tag.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_think_reason_tag_32b_run${i} --concurrency 64 \
        2>&1 | tee logs/baseline_think_reason_tag_32b_run${i}.log

    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
    python -u run_unified_hyper_async_revise_react.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_think_revise_react_32b_run${i} --concurrency 64 \
        2>&1 | tee logs/baseline_think_revise_react_32b_run${i}.log
done





### debug 

ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
EXECUTOR_MAX_TOKENS=64 ENABLE_THINKING=false \
python -u run_unified_step0_fixtest.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-32B --exp_name step0_fix_probe \
    --batch_size 1 --num_games 1 --max_steps 3 --overwrite 2>&1 | grep "PROBE"



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-8B \
    --port 8001 \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-8B \
    --port 8001 \
    --served-model-name Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.90 \
    2>&1 | tee /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval/vllm_logs/vllm_8001_dp8.log






# Jul 10th debug w/ claude chat


ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_nonthink_revise_react_step0bug_fix.py \
    --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B \
    --exp_name baseline_reviseReact_hist5_max2 \
    --num_games 2 --batch_size 1 --overwrite \
    2>&1 | tee logs_debug/baseline_reviseReact_max2_jul10.log



ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_nonthink_step0bug_fix.py \
    --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B \
    --exp_name baseline_thinkTag_hist5_max2 \
    --num_games 2 --batch_size 1 --overwrite \
    2>&1 | tee logs_debug/baseline_thinkTag_max2_jul10.log






EXECUTOR_TEMPERATURE=0.7 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
PROMPT_SHOW_EVERY=15 ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_nonthink_revise_react_step0bug_fix.py \
    --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B \
    --exp_name baseline_reviseReact_full_jul10_temp0.7_0.95_20_4096 \
    --batch_size 28 --overwrite \
    2>&1 | tee logs_debug/baseline_reviseReact_full_jul10_temp0.7_0.95_20_4096.log




EXECUTOR_TEMPERATURE=0.7 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
PROMPT_SHOW_EVERY=15 ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_nonthink_step0bug_fix.py \
    --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B \
    --exp_name baseline_thinkTag_full_jul10_temp0.7_0.95_20_4096 \
    --batch_size 28 --overwrite \
    2>&1 | tee logs_debug/baseline_thinkTag_full_jul10_temp0.7_0.95_20_4096.log




EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_hyper_async_step0bug_fix.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_pure_async_jul10_temp0.6_0.95_20_4096_run1 \
    --concurrency 64 \
    2>&1 | tee logs_debug/baseline_pure_async_jul10_temp0.6_0.95_20_4096_run1.log


EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
    --model openai/Qwen/Qwen3-8B --exp_name baseline_reviseReact_jul10_temp0.6_0.95_20_4096_run1 \
    --concurrency 64 \
    2>&1 | tee logs_debug/baseline_reviseReact_jul10_temp0.6_0.95_20_4096_run1.log


for i in {2..3}; do
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_pure_async_jul10_temp0.6_0.95_20_4096_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_pure_async_jul10_temp0.6_0.95_20_4096_run${i}.log

    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-8B --exp_name baseline_reviseReact_jul10_temp0.6_0.95_20_4096_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_reviseReact_jul10_temp0.6_0.95_20_4096_run${i}.log
    done


for i in {1..3}; do
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_pure_async_32b_jul10_temp0.6_0.95_20_4096_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_pure_async_32b_jul10_temp0.6_0.95_20_4096_run${i}.log

    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_reviseReact_32b_jul10_temp0.6_0.95_20_4096_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_reviseReact_32b_jul10_temp0.6_0.95_20_4096_run${i}.log
    done

# for i in {1..3}; do
#     ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
#     EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
#     python -u run_unified_hyper_async_reason_tag.py --env alfworld --memory_type none \
#         --model openai/Qwen/Qwen3-8B --exp_name baseline_think_reason_tag_try2_run1_run${i} --concurrency 64 \
#         2>&1 | tee logs/baseline_think_reason_tag_try2_run1_run${i}.log

#     ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
#     EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true PRINT_CHARS=200 \
#     python -u run_unified_hyper_async_revise_react.py --env alfworld --memory_type none \
#         --model openai/Qwen/Qwen3-8B --exp_name baseline_think_revise_react_try2_run1_run${i} --concurrency 64 \
#         2>&1 | tee logs/baseline_think_revise_react_try2_run1_run${i}.log
# done


