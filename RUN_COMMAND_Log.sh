



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
    --gpu-memory-utilization 0.90 \
    2>&1 | tee /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval/vllm_logs/vllm_8001_dp8_recheck.log



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



cd evaluation
bash sweep_32b.sh

```
for i in {1..3}; do
    EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_pure_async_32b_jul10_temp1.0_0.95_20_4096_hist3_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_pure_async_32b_jul10_temp1.0_0.95_20_4096_hist3_run${i}.log

    done

for i in {1..3}; do
    EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_reviseReact_32b_jul10_temp1.0_0.95_20_4096_hist3_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_reviseReact_32b_jul10_temp1.0_0.95_20_4096_hist3_run${i}.log
    done


for i in {1..3}; do
    EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 HISTORY_LENGTH=3 \
    ENABLE_THINKING=true PRINT_CHARS=2000 PROMPT_SHOW_EVERY=15 \
    ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY=EMPTY \
    OPENAI_API_BASE=http://localhost:8001/v1 TMPDIR=$HOME/tmp \
    python -u run_unified_hyper_async_revise_react_step0bug_fix.py --env alfworld --memory_type none \
        --model openai/Qwen/Qwen3-32B --exp_name baseline_reviseReact_32b_jul10_temp0.6_0.95_20_4096_hist3_run${i} \
        --concurrency 64 \
        2>&1 | tee logs_debug/baseline_reviseReact_32b_jul10_temp0.6_0.95_20_4096_hist3_run${i}.log
    done


# Launch both vllm servers in detached 'screen' sessions, inside the 'memory' conda environment,
# so large models can maximize CUDA memory and are easy to kill/control.
# To view:    screen -r vllm1    or    screen -r vllm2
# To kill:    screen -S vllm1 -X quit    or    screen -S vllm2 -X quit

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-8B --port 8001 --served-model-name Qwen/Qwen3-8B --dtype bfloat16 --data-parallel-size 4 --tensor-parallel-size 1 --max-model-len 40960 --gpu-memory-utilization 0.95 2>&1 | tee /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval/vllm_logs/vllm_8001_dp8_recheck_1.log

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-8B --port 8002 --served-model-name Qwen/Qwen3-8B --dtype bfloat16 --data-parallel-size 4 --tensor-parallel-size 1 --max-model-len 40960 --gpu-memory-utilization 0.95 2>&1 | tee /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval/vllm_logs/vllm_8001_dp8_recheck_2.log



# default 0.7  max tokens = 1024


ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8002/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=think \
PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 10 \
    --retrieve_num 5 \
    --max_steps 30 \
    --exp_name rb-qwen3-8b_curator_0.6_nonthinking \
    --overwrite
    


ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8002/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=revise_react \
PROMPT_SHOW_EVERY=1 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 1 \
    --num_games 3 \
    --retrieve_num 5 \
    --max_steps 5 \
    --exp_name rb-qwen3-8b_curator_0.6_nonthinking_smoke \
    --overwrite 2>&1 | tee logs_debug_memory/rb_smoke_qwen3-8b_curator_0.6_nonthinking.log



ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8002/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=revise_react \
PROMPT_SHOW_EVERY=1 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 1 \
    --num_games 3 \
    --retrieve_num 5 \
    --max_steps 5 \
    --exp_name rb-qwen3-8b_curator_0.6_nonthinking_smoke \
    --overwrite 2>&1 | tee logs_debug_memory/rb_smoke_qwen3-8b_curator_0.6_nonthinking.log


SAVE_RAW=1 ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8002/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=0.6 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=revise_react \
PROMPT_SHOW_EVERY=1 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 1 \
    --num_games 3 \
    --retrieve_num 5 \
    --max_steps 5 \
    --exp_name rb-qwen3-8b_curator_0.6_nonthinking_smoke_round2 \
    --overwrite 2>&1 | tee logs_debug_memory/rb_smoke_qwen3-8b_curator_0.6_nonthinking_round2.log


GOOGLE_CLOUD_PROJECT="salesforce-research-internal" GOOGLE_CLOUD_LOCATION="global" GOOGLE_GENAI_USE_VERTEXAI="True" python evaluation/agent_eval/test_gemini_api.py



SAVE_RAW=10 ALFWORLD_DATA=$HOME/.cache/alfworld OPENAI_API_KEY="EMPTY" OPENAI_API_BASE="http://localhost:8002/v1" HISTORY_LENGTH=5 \
EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096 ENABLE_THINKING=true \
CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=1024 CURATION_ENABLE_THINKING=false \
PROMPT_STYLE=revise_react \
PROMPT_SHOW_EVERY=15 PRINT_CHARS=2000 \
    TMPDIR=$HOME/tmp \
    python -u run_unified_dev.py --env alfworld \
    --memory_type reasoningbank \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 10 \
    --retrieve_num 5 \
    --max_steps 30 \
    --exp_name rb-revise_react_qwen3-8b_curator_1.0_nonthinking_round1 \
    --overwrite 2>&1 | tee logs_debug_memory/rb_revise_react_qwen3-8b_curator_1.0_nonthinking_round1.log



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve Qwen/Qwen3-8B --port 8001 --served-model-name Qwen/Qwen3-8B --dtype bfloat16 --data-parallel-size 8 --tensor-parallel-size 1 --max-model-len 40960 --gpu-memory-utilization 0.95 2>&1 | tee /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/evaluation/agent_eval/vllm_logs/vllm_8001_dp8_server_1.log



CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen3-8B --port 8001 --served-model-name Qwen/Qwen3-8B --dtype bfloat16 --data-parallel-size 4 --tensor-parallel-size 1 --max-model-len 40960 --gpu-memory-utilization 0.95

CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve Qwen/Qwen3-32B --port 8002 --served-model-name Qwen/Qwen3-32B --dtype bfloat16 --data-parallel-size 4 --tensor-parallel-size 1 --max-model-len 40960 --gpu-memory-utilization 0.95



cd evaluation/agent_eval

export OPENAI_API_KEY=EMPTY
export ALFWORLD_DATA=$HOME/.cache/alfworld
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=1.0 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react SAVE_RAW=10

OPENAI_API_BASE=http://localhost:8001/v1 HISTORY_LENGTH=3 ENABLE_THINKING=false \
python -u run_unified_dev_async.py --env alfworld --memory_type skillos \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 3 --retrieve_num 5 --max_steps 30 \
    --num_games 10 --exp_name skillos-smoke --overwrite






## smoke test:


cd evaluation/agent_eval

export OPENAI_API_KEY=EMPTY
export ALFWORLD_DATA=$HOME/.cache/alfworld
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react SAVE_RAW=10

OPENAI_API_BASE=http://localhost:8002/v1 HISTORY_LENGTH=3 ENABLE_THINKING=false \
python -u run_unified_dev_async_curator.py --env alfworld --memory_type curator \
    --model          openai/Qwen/Qwen3-8B \
    --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 3 --retrieve_num 5 --max_steps 30 \
    --num_games 10 --exp_name curator_smoke_baseline --overwrite




cd ~/mem-evolve/SkillCurator-main/evaluation/agent_eval
export OPENAI_API_KEY=EMPTY ALFWORLD_DATA=$HOME/.cache/alfworld
export EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 EXECUTOR_MAX_TOKENS=4096
export CURATION_TEMPERATURE=0.6 CURATION_TOP_P=0.95 CURATION_TOP_K=20 CURATION_MAX_TOKENS=4096 CURATION_ENABLE_THINKING=true
export PROMPT_STYLE=revise_react SAVE_RAW=10

OPENAI_API_BASE=http://localhost:8002/v1 HISTORY_LENGTH=3 ENABLE_THINKING=false \
python -u run_unified_dev_async_curator.py --env alfworld --memory_type curator_v1 \
    --curation_mode success_only \
    --model openai/Qwen/Qwen3-8B --curation_model openai/Qwen/Qwen3-8B \
    --curation_base_url http://localhost:8001/v1 \
    --batch_size 3 --retrieve_num 5 --max_steps 30 \
    --num_games 10 --exp_name curator_v1_smoke_afterrefactor --overwrite





JOB_PARALLEL=2 bash run_curator_v1_modes_sequential.sh

JOB_PARALLEL=3 BASE=server_api_curator_v1_gpt_jul16th_extra_success_v1.sh bash run_curator_v1_modes_sequential.sh





source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory

REPO=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
OUT_DIR=/fsx/home/yefan.zhou/mem-evolve/data       # <-- set this (outside SkillCurator-main)

cd "$REPO/evaluation/agent_eval"                 # needed: Alfworld/base_config.yaml resolves here
PYTHONPATH="$REPO" \
OPENAI_API_BASE=http://localhost:8001/v1 OPENAI_API_KEY=EMPTY \
ENABLE_THINKING=false PROMPT_STYLE=revise_react \
EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 HISTORY_LENGTH=3 EXECUTOR_MAX_TOKENS=4096 \
python -m memcurator.sample_and_select \
    --num_games 4 --k 8 --max_steps 30 --concurrency 64 \
    --curator_variant curator_alfworld_v1 --curation_mode success_and_fail \
    --keep all \
    --out_dir "$OUT_DIR"




source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory

REPO=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
OUT_DIR=/fsx/home/yefan.zhou/mem-evolve/data/alfworld_hist3_frac0.2       # <-- set this (outside SkillCurator-main)

cd "$REPO/evaluation/agent_eval"                 # needed: Alfworld/base_config.yaml resolves here
PYTHONPATH="$REPO" \
OPENAI_API_BASE=http://localhost:8001/v1 OPENAI_API_KEY=EMPTY \
ENABLE_THINKING=false PROMPT_STYLE=revise_react \
EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 HISTORY_LENGTH=3 EXECUTOR_MAX_TOKENS=4096 \
python -m memcurator.sample_and_select \
    --stratify --frac 0.2 --k 8 --max_steps 30 --concurrency 128 \
    --curator_variant curator_alfworld_v1 --curation_mode success_and_fail \
    --keep all \
    --out_dir "$OUT_DIR"




# ============================================================================
# Jul-18: FROM-SCRATCH Stage-A harvest, frac=0.5 k=4 (the frac=0.2 attempt crashed + was renamed).
# The [Errno 28] No space left crash is FIXED.
#   Root cause (verified by leak probe, NOT an env-close bug): textworld/fast_downward's
#   load_lib() copies a ~35MB libdownward.so into a fresh tempfile.TemporaryDirectory() on
#   EVERY episode's init_env(). Each copy is auto-removed the instant load_lib() returns (no
#   leak; tw.close() would free nothing extra), BUT at --concurrency 128 up to ~128 copies
#   (~4.5GB) coexist momentarily. On the 31GB ROOT /tmp that peak overflowed. Every eval
#   command in this log already sets TMPDIR=$HOME/tmp (on the 12T /fsx) for this reason; the
#   sampler forgot to.  FIX (now baked into memcurator/sample_and_select._route_tmpdir):
#   temp copies are routed to $TMPDIR (else <out_dir>/_tmp) and tempfile.tempdir is set so it
#   actually takes effect. TMPDIR=$HOME/tmp below is the belt-and-suspenders (both on /fsx).
#   Verified on box: tempfile.gettempdir() -> /fsx (not /tmp).  --concurrency 64 here for extra
#   headroom (128 also works now that temp is on /fsx; the prior crash was disk, not the server).
#
# NOTE: server must be the NONTHINK executor (ENABLE_THINKING=false + revise_react, temp 1.0,
#   hist3) — launch it yourself first, e.g. launch_vllm_1port_dp8.sh on port 8001.
# ============================================================================
source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory

REPO=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
OUT_DIR=/fsx/home/yefan.zhou/mem-evolve/data/alfworld_hist3_frac0.5   # fresh dir (frac0.2 attempt renamed to *_broken)

cd "$REPO/evaluation/agent_eval"                 # needed: Alfworld/base_config.yaml resolves here
mkdir -p "$OUT_DIR"
PYTHONPATH="$REPO" \
OPENAI_API_BASE=http://localhost:8001/v1 OPENAI_API_KEY=EMPTY TMPDIR=$HOME/tmp \
ENABLE_THINKING=false PROMPT_STYLE=revise_react \
EXECUTOR_TEMPERATURE=1.0 EXECUTOR_TOP_P=0.95 EXECUTOR_TOP_K=20 HISTORY_LENGTH=3 EXECUTOR_MAX_TOKENS=4096 \
python -u -m memcurator.sample_and_select \
    --stratify --frac 0.5 --k 4 --max_steps 30 --concurrency 64 \
    --curator_variant curator_alfworld_v1 --curation_mode success_and_fail \
    --keep all \
    --out_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/run_tee.log"
# NOTE: log is DOUBLY captured (redundant on purpose): the script self-writes "$OUT_DIR/run.log",
#       AND the `| tee` writes "$OUT_DIR/run_tee.log" (different name so they don't fight).
#       Follow live with: tail -f "$OUT_DIR/run_tee.log"
#       (The prior partial run left rollouts_raw.jsonl; this RESUMES from it — crash-safe, no dupes.)
# frac 0.5, k=4: p-hat is now DIAGNOSTIC-only, NOT a target gate. Rationale: a bad/misleading
# briefing can drop even a bare-p-hat=1.0 task below 1.0, so the 8-briefing GRPO group still has
# within-group variance -> nonzero advantage -> useful "do no harm" gradient. So we don't need
# K=8 precision; K=4 gives a readable histogram while frac 0.5 buys ~2.5x more distinct source
# tasks in the pool (better BM25 retrieval realism). ~1776 targets x 4 = ~7100 rollouts. 




# source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory
# REPO=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
# A=/fsx/home/yefan.zhou/mem-evolve/data/alfworld_hist3_frac0.5
# B=/fsx/home/yefan.zhou/mem-evolve/data/datasets/fulltargets_sample_dataset_frac0.5_
# cd "$REPO"
# PYTHONPATH="$REPO" python -m memcurator.build_dataset \
#     --stage_a_dir "$A" \
#     --targets_path "$A/targets_full.jsonl" \
#     --out_dir "$B" \
#     --pool_size 10 --successes_per_task 1 \
#     --pool_category_mode mixed --self_exclude_level task_id \
#     --pool_status success_only \
#     --curation_mode success_only \
#     --seed 42



source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory
REPO=/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
A=/fsx/home/yefan.zhou/mem-evolve/data/alfworld_hist3_frac0.5
B=/fsx/home/yefan.zhou/mem-evolve/data/datasets/smoke_dataset
cd "$REPO"
PYTHONPATH="$REPO" python -m memcurator.build_dataset \
    --stage_a_dir "$A" \
    --targets_path "$A/targets_full.jsonl" \
    --out_dir "$B" \
    --pool_size 10 --successes_per_task 1 \
    --pool_category_mode mixed --self_exclude_level task_id \
    --pool_status success_only --curation_mode success_only \
    --sample_with_replacement \
    --trajectory_style both --num_targets 256 --seed 42




ssh box2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory
cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main

CUDA_VISIBLE_DEVICES=4,5,6,7 \
EXECUTOR_API_BASE=http://localhost:8001/v1 \
NGPUS=4 \
bash scripts/train_memcurator_smoke.sh



ssh box2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate memory
cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main

# clear leftover training procs on GPUs 4-7 (DON'T touch executor on 0-3)
ray stop --force 2>/dev/null; pkill -9 -f main_ppo 2>/dev/null; pkill -9 -f "ray::" 2>/dev/null; sleep 3

# launch smoke WITH validation (TEST_FREQ=1 exercises _validate + batch_memories fix)
CUDA_VISIBLE_DEVICES=4,5,6,7 EXECUTOR_API_BASE=http://localhost:8001/v1 NGPUS=4 \
TEST_FREQ=1 \
bash scripts/train_memcurator_smoke.sh




cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
ray stop --force 2>/dev/null; pkill -9 -f main_ppo 2>/dev/null; pkill -9 -f "ray::" 2>/dev/null; sleep 3

CUDA_VISIBLE_DEVICES=4,5,6,7 EXECUTOR_API_BASE=http://localhost:8001/v1 NGPUS=4 TEST_FREQ=1 \
bash scripts/train_memcurator_smoke.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 EXECUTOR_API_BASE=http://localhost:8001/v1 NGPUS=4 TEST_FREQ=1 \
bash scripts/train_memcurator_smoke.sh


# Diagnosis doc §5 → APPLIED — marked A/B/C done, documented the final signatures (6-tuple / 4-tuple / kw-only threading), and recorded the executor-turns extension beyond the original plan (the actual 0/8 diagnostic the plan lacked).

# analysis/inspect_generation_dump.py — a standing inspector (not throwaway) that reads the enriched generation/ dump and reports:

# row-count sanity (16 would flag the append bug regression; expect 8),
# curator sanity (curator_raw has <think>, briefing is stripped clean, empty-briefing count),
# the 0/8 diagnostic: for each failing slot, prints the executor's full prompt tail + raw response for the first few steps — so we can finally see why it fails (bad format/parse vs genuine wandering).
# Usage once the dump lands:


# python analysis/inspect_generation_dump.py results/memcurator-SMOKE-qwen3-8b-alfworld/<stamp>



nohup rsync -a --info=progress2 \
  /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main/ \
  /fsx/sfr/yefan.zhou/mem-evolve/SkillCurator-main/ \
  > /fsx/sfr/yefan.zhou/rsync_code_secondtime.log 2>&1 &