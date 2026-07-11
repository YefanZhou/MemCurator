import os
from datetime import datetime
import json
import vllm
import yaml
import argparse
import numpy as np
import requests
import openai
import random
import time
import copy
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from conversation_creator import ConversationCreator
from agent import MemoryAgent
from memory import Memory
from functions import get_memory_tool_schemas

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0.0) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
    except Exception:
        pass
    except TimeoutException:
        ret_score = timeout_score

    return ret_score

SYSTEM_PROMPT = """
Your are a helpful assistant that can label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The input format is:
<Question>: {question}
<Gold answer>: {gold_answer}
<Generated answer>: {generated_answer}

You will need to label the generated answer as either CORRECT or WRONG based on the following criteria:

CORRECT: Label as CORRECT if and only if **the generated answer contains the same factual information as the gold answer** (Paraphrasing, rewording, or different phrasing is acceptable)

WRONG: Label as WRONG if:
- The generated answer provides different factual information (different dates, locations, names, numbers, etc.)
- The generated answer contradicts or conflicts with the gold answer
- The generated answer is completely unrelated to the question
- The generated answer is too vague or incomplete to verify correctness
- The generated answer says "I don't know" or equivalent when a gold answer exists

Response format: Provide a brief one-sentence explanation of your reasoning, then end with exactly <label>CORRECT</label> or <label>WRONG</label>.
CRITICAL: Use only ONE label in your response - never include both CORRECT and WRONG as this will break the evaluation.
"""

ACCURACY_PROMPT = """
Give me the label <label>CORRECT</label> or <label>WRONG</label> for the following question, gold answer, and generated answer.
<Question>: {question}
<Gold answer>: {gold_answer}
<Generated answer>: {generated_answer}
"""

QUERY_PROMPT = """Now use your structured memory to assist in solving the following math problem. 

**Instruction**: 
- **In your response, you MUST first explicitly state which memory you used or why you didn't.**
- Follow the exact format provided in the example below.

---
**EXAMPLE FORMAT**:
Memory Usage: [Your memory usage analysis here...].
Reasoning: [Your step-by-step memory-grounded reasoning process here...]
Final Answer: [Your final answer here...]
---

**QUESTION**:
{question}
"""

# QUERY_PROMPT = """
# Your core memory contains meta-reasoning protocols. Your semantic memory contains general/principled knowledge including fundamental mathematical truths, definitions. Your episodic memory includes concrete strategies including specific useful insights, reusable skill/concepts derived from past experiences. Use your memory to assist in solving the user's math problems.

# Question:
# {question}
# """

def batch_process_questions_with_qwen32b(questions, batch_size=32, system_prompt=None, model="Qwen/Qwen2.5-7B-Instruct", no_thinking=False, generative_reward=False):
    """
    Process a list of questions using Qwen32B model in batches

    Args:
        questions: List of questions to process
        batch_size: Number of questions to process in each batch
        model: Qwen model to use

    Returns:
        List of responses corresponding to each question
    """

    # Import and setup Qwen client
    from openai import OpenAI
    from transformers import AutoTokenizer
    import time

    # Setup Qwen client
    client = OpenAI(
        base_url="http://10.202.0.8:8001/v1",
        api_key="EMPTY"
    )

    # Initialize tokenizer for prompt conversion
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    print(f"Starting batch processing of {len(questions)} questions with Qwen2.5-7B, batch size {batch_size}")

    all_responses = []

    # Process questions in batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(questions) + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_questions)} questions)")

        # Convert all questions in batch to prompts
        batch_prompts = []
        for question in batch_questions:
            if system_prompt is not None:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]
            else:
                messages = [
                    # {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]

            # Convert to prompt using tokenizer
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # if no_thinking:
            #     prompt += "<think></think>\n\n"

            batch_prompts.append(prompt)

        # Process the entire batch at once using completions API
        response = client.completions.create(
            model=model,
            prompt=batch_prompts,
            max_tokens=1024,
            temperature=0.0,
            stream=False
        )

        # Extract responses
        batch_responses = [choice.text for choice in response.choices]
        all_responses.extend(batch_responses)
        print(f"Completed batch {batch_num}/{total_batches}")
        # Delay between batches to avoid overloading the server
        if i + batch_size < len(questions):
            time.sleep(0.5)

    print(f"Batch processing complete. Generated {len(all_responses)} responses.")
    return all_responses


def load_agent_config(config_path):
    """Load agent configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate required fields
    required_fields = ['agent_name', 'model_name']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in config file: {config_path}")

    return config

def get_results_filename(agentic_search=False):
    """Get the appropriate results filename based on search method."""
    return "agentic_results.json" if agentic_search else "results.json"

def process_chunk_with_gpt4_mini(chunk_data):
    """Process a single chunk using GPT-4.1-mini via OpenAI API"""
    chunk, memory, agent_config, memory_agent_template = chunk_data

    # Build messages similar to process_text_with_qwen_pipeline but for chat.completions
    messages = []

    # Add memory system prompt if available
    if memory is not None:
        query = chunk[:100] + "..." if len(chunk) > 100 else chunk
        max_num_of_recent_chunks = getattr(MemoryAgent, 'MAX_MEMORY_ITEMS', 10)
        messages = memory.render_system_prompt(status='memorie', query=query, max_num_of_recent_chunks=max_num_of_recent_chunks)

    # Add user message with the chunk content
    messages.append({"role": "user", "content": chunk})

    # Get memory tools directly
    tools = get_memory_tool_schemas(memory)

    # Initialize Azure OpenAI client
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2025-01-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    # Generate response using GPT-4.1-mini with tools
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=agent_config.get('max_new_tokens', 2048),
        temperature=0.6
    )

    message = response.choices[0].message
    final_response = message.content or ""

    # Parse function calls directly from the structured response
    function_calls = []
    if message.tool_calls:
        for tool_call in message.tool_calls:
            if tool_call.type == "function":
                function_call = {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }

                # Execute the function call immediately
                tool_result = memory_agent_template._run_tool_from_function_call(
                    function_call,
                    memory
                )

                function_calls.append({
                    'function_call': function_call,
                    'tool_result': tool_result,
                    'timestamp': time.time()
                })

    return final_response, function_calls

def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Memory Agent Evaluation")
    parser.add_argument("--agent_config", type=str, required=True, help="Path to agent configuration YAML file")
    parser.add_argument("--dataset", type=str, default='AIME24') # Restricted choices
    parser.add_argument("--load_db_from", type=str, default=None) # Memory databse
    parser.add_argument("--chunk_size", type=int, default=4096, help="Chunk size for MemAgent_Bench dataset")  # add parameter chunk_size
    parser.add_argument("--save_process", action="store_true", help="Enable process tracking for Qwen models (saves detailed logs)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for batch processing")
    parser.add_argument("--agentic_search", action="store_true", help="Use agentic memory search instead of simple batch processing")
    parser.add_argument("--rollout_label", type=str, default=None, help="Label to append to output directory path, e.g., rollout_1")
    parser.add_argument("--force_reanswer_questions", action="store_true", help="Force reanswering all questions even if results file already exists")
    return parser.parse_args()


def run_streaming_sequential(args, agent_config, batch_queries_and_answers, trial):
    """
    针对单序列 (Batch Size = 1) 的流式记忆增强问答
    """
    # 1. 初始化
    # 根据配置加载提示词
    with open('config/prompts_wrt_datasource.yaml', 'r') as f:
        prompts_config = yaml.safe_load(f)

    # 1. 基础配置初始化 (由于是流式，我们只取 batch 中的第一个序列)
    memory = Memory(including_core=agent_config.get('including_core', True))
    agent = MemoryAgent(agent_config=agent_config)
    
    # 获取所有问题 (假设 batch_size=1)
    queries = batch_queries_and_answers
    streaming_history = []
    
    base_url = agent_config['external_model_url'].rstrip('/')
    endpoint = f"{base_url}/batch_process"

    print(f"🚀 starting streaming, 共 {len(queries)} 个问题...")

    for i, item in enumerate(queries):
        # 根据数据集解析问题、答案和数据源
        q_text = item[0]
        gold_ans = str(item[1])

        # --- 步骤 A: 准备当前记忆快照并调用 batch_process 接口 ---
        memory_dict = {
            'episodic': memory.episodic,
            'semantic': memory.semantic
        }
        if memory.including_core and memory.core is not None:
            memory_dict['core'] = memory.core

        # query_prompt = prompts_config['math']['query_prompt']

        # 构造符合接口要求的 Payload (Batch size 1)
        payload = {
            "memories": [memory_dict],
            "questions": [[QUERY_PROMPT.format(question=q_text) + "\n\n" + "Please reason step by step, and put your final answer within \\boxed{}"]]
            # "questions": [["Now solve the following problem:\n\n" + q_text + "\n\nPlease reason step by step, and put your final answer within \\boxed{}"]]
        }

        copied_memory = copy.deepcopy(memory_dict)

        print("Question Generation Payload:", payload)  # Debug print

        print(f"[Step {i+1}] 正在请求服务器获取答案...")
        try:
            resp = requests.post(endpoint, json=payload)
            prediction = resp.json().get('result', [[""]])[0][0]
        except Exception as e:
            print(f"❌ 服务器请求失败: {e}")
            prediction = "Error: No response"

        print(f"模型回答: {prediction}")
        print(f"标准答案: {gold_ans}")

        # --- 步骤 B: 使用 Pipeline 构建记忆更新指令 (关键点) ---
        # 1. 构造原始文本经验（基于刚才生成的 prediction）
        experience_chunk = f"### Question\n{q_text}\n\n### Attempted Solution:\n{prediction}"
        max_new_tokens = agent_config.get('max_new_tokens', 2048)
        
        # 2. 包装成 unified_prompt
        raw_update_content = prompts_config['unified_prompt'].format(
            context=experience_chunk,
            max_new_tokens=int(max_new_tokens * 0.8)
        )

        print("----------")
        print(raw_update_content)

        # 3. 使用你指定的 pipeline 处理成 Qwen 格式 (包含 Tool 调用逻辑)
        processed_update_prompt = MemoryAgent.process_text_with_qwen_pipeline(
            text=raw_update_content,
            tokenizer=agent.tokenizer,
            functions=[tool["function"] for tool in get_memory_tool_schemas(memory)],
            status='memorie',
            enable_thinking=agent_config['enable_thinking'],
            return_text=True,
            memory=memory
        )

        # --- 步骤 C: 在本地执行记忆更新 ---
        # 调用本地模型解析 processed_update_prompt 并执行存储操作
        update_sampling_params = vllm.SamplingParams(
            temperature=0.7, 
            max_tokens=4096,
        )

        print(processed_update_prompt)

        update_outputs = agent.model.generate([processed_update_prompt], update_sampling_params)
        update_response = update_outputs[0].outputs[0].text.strip()

        # 解析模型输出中的 Tool Call 并修改内存中的 memory 对象
        assistant_messages = agent._parse_response(update_response)

        print(f"记忆更新函数调用: {[msg.get('function_call') for msg in assistant_messages if msg.get('function_call')]}")

        for msg in assistant_messages:
            if msg.get("function_call"):
                agent._run_tool_from_function_call(msg["function_call"], memory)

        # --- 步骤 D: 结果记录与评估 ---
        # 这里进行简单的字符串匹配评估（建议根据你的 dataset 需求微调评估逻辑）
        if args.dataset == "AIME24" or args.dataset == "AMC23":
            is_correct = (compute_score(prediction, gold_ans) > 0)
        elif args.dataset == "GPQA":
            prompt_for_evaluation = ACCURACY_PROMPT.format(
                question=q_text,
                gold_answer=gold_ans,
                generated_answer=prediction
            )
            eval_response = batch_process_questions_with_qwen32b(
                questions=[prompt_for_evaluation],
                batch_size=1,
                system_prompt=SYSTEM_PROMPT,
                model="Qwen/Qwen2.5-7B-Instruct",
                no_thinking=True
            )[0]
            is_correct = "<label>CORRECT</label>" in eval_response  

        print(f"第 {i+1} 题评估结果: {is_correct}")
        
        streaming_history.append({
            "step": i + 1,
            "question": q_text,
            "prediction": prediction,
            "gold": gold_ans,
            "is_correct": is_correct,
            "memory_contents": copied_memory,
        })
        print(f"✅ 第 {i+1} 题完成. Correct: {is_correct}. 当前记忆条数: {len(memory.episodic) + len(memory.semantic)}")

    # 最终统计
    final_acc = sum([1 for x in streaming_history if x['is_correct']]) / len(streaming_history)
    print(f"\n🎯 最终平均准确率: {final_acc:.2%}")

    # 保存结果
    out_dir = f"./evaluation/streaming_{agent_config['agent_name']}_{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/streaming_summary_{trial}.json", "w") as f:
        json.dump({"accuracy": final_acc, "history": streaming_history}, f, indent=2)

    return final_acc


def run_nomem_baseline(args, batch_queries_and_answers, trial):

    from vllm import LLM, SamplingParams

    worker = LLM(
        model="Qwen/Qwen2.5-7B-Instruct", 
        max_model_len=16384,
    )

    use_chat = True

    if use_chat:
        prompt_template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
    else:
        prompt_template = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    
    prompts = [prompt_template.format(input=example[0]) for example in batch_queries_and_answers]

    sampling_params = SamplingParams(
        temperature=0.7, 
        max_tokens=16384,
        n=1,
        stop=["</s>", "<|im_end|>", "<|endoftext|>", "assistant", "user", "_end", "_start"],
        stop_token_ids=[151645, 151643],
    )

    responses = worker.generate(prompts, sampling_params, use_tqdm=True)

    history = []

    for i, item in enumerate(responses):
        # 根据数据集解析问题、答案和数据源
        gold_ans = str(batch_queries_and_answers[i][1])
        prediction = item.outputs[0].text.strip()
        prompt = item.prompt

        is_correct = (compute_score(prediction, gold_ans) > 0)

        history.append({
            "step": i + 1,
            "question": prompt,
            "prediction": prediction,
            "gold": gold_ans,
            "is_correct": is_correct,
        })

    # 最终统计
    final_acc = sum([1 for x in history if x['is_correct']]) / len(history)
    print(f"\n🎯 最终平均准确率: {final_acc:.2%}")

    # 保存结果
    out_dir = f"./evaluation/nomem_baseline_AIME2024"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/nomem_baseline_summary_{trial}.json", "w") as f:
        json.dump({"accuracy": final_acc, "history": history}, f, indent=2)



def main():

    args = parse_args()

    # Load agent configuration
    agent_config = load_agent_config(args.agent_config)

    # Print loaded configuration
    print(f"Loaded agent configuration:")
    print(f"  Agent name: {agent_config['agent_name']}")
    print(f"  Model name: {agent_config['model_name']}")
    if 'enable_thinking' in agent_config:
        print(f"  Enable thinking: {agent_config['enable_thinking']}")
    print(f"  Save process (Qwen only): {args.save_process}")

    from datasets import load_dataset

    queries_and_answers = []

    if args.dataset == 'AIME24':
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        # ds = ds.shuffle(seed=42)  # Shuffle the dataset to ensure randomness

        for item in ds:
            question = item['Problem']
            answer = item['Answer']
            queries_and_answers.append([question, answer])

    elif args.dataset == "AMC23":
        ds = load_dataset("zwhe99/amc23", split="test")

        for item in ds:
            question = item['question']
            answer = item['answer']
            queries_and_answers.append([question, answer])

    elif args.dataset == "GPQA":
        with open("./data/GPQA.json", "r") as f:
            data = json.load(f)
            # random sample 50 instances for quick test with fix seeds
            random.seed(42)
            sampled_data = random.sample(data, min(50, len(data)))
            for item in sampled_data:
                question = item['query']
                answer = item['answer']
                queries_and_answers.append([question, answer])

    for trial in range(16):
        run_streaming_sequential(args, agent_config, queries_and_answers, trial)
        # run_nomem_baseline(args, queries_and_answers, trial)


if __name__ == '__main__':
    main()
