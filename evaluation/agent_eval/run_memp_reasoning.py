"""
MemP on reasoning benchmarks (AIME24, AIME25, GPQA).

Mirrors run_memp_online.py but for single-turn reasoning:
  - Memory is the same `Memory` class from memory.py
  - Curator LLM is controlled via --mem_model / --mem_base_url
  - Retrieval is top-1 workflow; injected as "Past Relevant Guidelines"
  - Memory is updated after each batch (except when cold_start mode is disabled)
"""
import os
import re
import json
import argparse
import math
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_API_BASE", "http://localhost:8000/v1")

from litellm import completion


REASONING_TEMPLATE = "{question}\n\nPlease reason step by step and put final answer within \\boxed{{}}."

REASONING_TEMPLATE_WITH_GUIDELINES = """\
You are an expert problem solver.

## Past Relevant Guidelines

{retrieved_skills}

## Current Problem

{question}

Please reason step by step using the guidelines above if applicable, and put your final answer within \\boxed{{}}.
"""

GPQA_TEMPLATE = """\
Answer the following multiple-choice question. Respond with a single letter (A, B, C, or D) inside \\boxed{{}}.

Question: {question}

(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}
"""

GPQA_TEMPLATE_WITH_GUIDELINES = """\
## Past Relevant Guidelines

{retrieved_skills}

## Current Problem

Answer the following multiple-choice question. Respond with a single letter (A, B, C, or D) inside \\boxed{{}}.

Question: {question}

(A) {choice_a}
(B) {choice_b}
(C) {choice_c}
(D) {choice_d}
"""


def llm(prompt, model):
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    response = completion(
        model=model,
        messages=messages,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ['OPENAI_API_BASE'],
        num_retries=10,
        temperature=0.7,
    )
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    return "Output Error"


def load_dataset_problems(env_name):
    problems = []
    if env_name == 'gpqa':
        import random, pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), 'gpqa_diamond.csv')
        data = pd.read_csv(csv_path).to_dict('records')
        for row in data:
            choices = [row['Correct Answer'], row['Incorrect Answer 1'],
                       row['Incorrect Answer 2'], row['Incorrect Answer 3']]
            rng = random.Random(hash(row['Question']))
            rng.shuffle(choices)
            labels = ['A', 'B', 'C', 'D']
            correct_label = labels[choices.index(row['Correct Answer'])]
            problems.append({
                'question': row['Question'],
                'choices':  {'A': choices[0], 'B': choices[1], 'C': choices[2], 'D': choices[3]},
                'answer':   correct_label,
                'type':     'gpqa',
            })
    else:
        from datasets import load_dataset
        HF = {'aime24': ('math-ai/aime24', None),
              'aime25': ('math-ai/aime25', None)}
        hf_name, subset = HF[env_name]
        ds = load_dataset(hf_name, subset, trust_remote_code=True)
        split = 'test' if 'test' in ds else list(ds.keys())[0]
        col_a = {'aime24': 'solution', 'aime25': 'answer'}[env_name]
        for row in ds[split]:
            problems.append({
                'question': row['problem'],
                'answer':   str(row[col_a]),
                'type':     'aime',
            })
    return problems


def build_prompt(prob, guidelines_text=None):
    if prob['type'] == 'gpqa':
        if guidelines_text:
            return GPQA_TEMPLATE_WITH_GUIDELINES.format(
                retrieved_skills=guidelines_text,
                question=prob['question'],
                choice_a=prob['choices']['A'], choice_b=prob['choices']['B'],
                choice_c=prob['choices']['C'], choice_d=prob['choices']['D'],
            )
        return GPQA_TEMPLATE.format(
            question=prob['question'],
            choice_a=prob['choices']['A'], choice_b=prob['choices']['B'],
            choice_c=prob['choices']['C'], choice_d=prob['choices']['D'],
        )
    if guidelines_text:
        return REASONING_TEMPLATE_WITH_GUIDELINES.format(
            retrieved_skills=guidelines_text,
            question=prob['question'],
        )
    return REASONING_TEMPLATE.format(question=prob['question'])


def score_reasoning(pred_text, gold, env_name):
    if env_name == 'gpqa':
        matches = re.findall(r'\\boxed\{([^{}]*)\}', pred_text)
        if not matches:
            matches = re.findall(r'\\boxed\{\\[a-z]+\{([^{}]*)\}\}', pred_text)
        if matches:
            raw = matches[-1].strip()
            letter = re.search(r'\b([A-D])\b', raw, re.IGNORECASE)
            pred_letter = letter.group(1).upper() if letter else raw.upper()
        else:
            pred_letter = ''
            for pat in [
                r'[Ff]inal [Aa]nswer[^\n]*\b([A-D])\b',
                r'[Aa]nswer is[^\n]*\b([A-D])\b',
                r'\(([A-D])\) is correct',
                r'corresponds to \(([A-D])\)',
                r'answer: \*\*([A-D])\*\*',
            ]:
                m = re.search(pat, pred_text, re.IGNORECASE)
                if m:
                    pred_letter = m.group(1).upper()
                    break
        return 1.0 if pred_letter == gold.strip().upper() else 0.0
    from math_verify import parse, verify
    try:
        return 1.0 if verify(parse(gold), parse(pred_text)) else 0.0
    except Exception:
        return 0.0


def trajectory_text(question, response):
    return f"[Question]: {question}\n\n[Answer]: {response}"


def main(args):
    problems = load_dataset_problems(args.env)
    print(f'[INFO] Loaded {len(problems)} problems for {args.env}')

    output_path = f'Reasoning/results/{args.env}/{args.model}/{args.exp_name}_memp'
    if args.overwrite and os.path.exists(output_path):
        for f in os.listdir(output_path):
            os.remove(os.path.join(output_path, f))
    os.makedirs(output_path, exist_ok=True)

    # Memory setup (after setting env vars)
    if args.mem_model:
        os.environ["MODEL_NAME"] = args.mem_model
    if args.mem_base_url:
        os.environ["API_BASE_URL"] = args.mem_base_url
    from memory import Memory
    Memory_config = yaml.safe_load(open('ProcedureMem/config.yaml'))
    Memory_config["memory_dir"] = f"Reasoning/memory/{args.env}/memp_{args.exp_name}"
    Memory_config["retrieve_num"] = args.retrieve_num
    Pro_Mem = Memory(**Memory_config)

    finished = 0
    all_reward = 0.0
    for f in os.listdir(output_path):
        if f.endswith('.json'):
            finished += 1
            all_reward += json.load(open(f'{output_path}/{f}'))['reward']

    for i in tqdm(range(0, len(problems), args.batch_size)):
        batch = problems[i:i + args.batch_size]
        if i + len(batch) <= finished:
            continue

        # Retrieve top-1 guidelines per problem
        guidelines_list = []
        workflow_list = []
        memory_list   = []
        if len(Pro_Mem.documents) > 0:
            for prob in batch:
                hits = Pro_Mem.retrieve(prob['question'])
                if hits:
                    top = hits[0][0] if isinstance(hits[0], tuple) else hits[0]
                    q = top.metadata.get('query')
                    w = top.metadata.get('workflow')
                    memory_list.append(q)
                    workflow_list.append(w)
                    guidelines_list.append(json.dumps(
                        [{"task_name": q, "guidelines": w}], indent=4, ensure_ascii=False))
                else:
                    memory_list.append(None)
                    workflow_list.append(None)
                    guidelines_list.append(None)
        else:
            memory_list   = [None] * len(batch)
            workflow_list = [None] * len(batch)
            guidelines_list = [None] * len(batch)

        # Build prompts
        prompts = [build_prompt(prob, g) for prob, g in zip(batch, guidelines_list)]

        # Parallel inference
        responses = [''] * len(batch)
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(llm, prompts[j], args.model): j for j in range(len(batch))}
            for future in as_completed(futures):
                j = futures[future]
                try:
                    responses[j] = future.result()
                    print(f'\033[92mProblem {i+j}: {responses[j][:120]}\033[0m')
                except Exception as e:
                    print(f'Error problem {i+j}: {e}')

        # Score + save
        batch_results = []
        for j, (prob, resp) in enumerate(zip(batch, responses)):
            if i + j < finished:
                continue
            reward = score_reasoning(resp, prob['answer'], args.env)
            result = {
                'messages': [
                    {"role": "user",      "content": prompts[j]},
                    {"role": "assistant", "content": resp},
                ],
                'answer': prob['answer'],
                'reward': reward,
            }
            with open(f'{output_path}/idx_{i+j}.json', 'w') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            all_reward += reward
            batch_results.append(result)

        done = min(i + len(batch), len(problems))
        tqdm.write(f'Avg accuracy: {all_reward / done * 100:.2f}%  [{done}/{len(problems)}]')

        # Update memory after the batch
        if Pro_Mem.is_cold_start == False:
            query_list = [prob['question'] for prob in batch]
            trajectory_list = [trajectory_text(prompts[j], responses[j]) for j in range(len(batch))]
            reward_list = [score_reasoning(responses[j], batch[j]['answer'], args.env) for j in range(len(batch))]
            Pro_Mem.update(query_list, trajectory_list, reward_list, workflow_list, memory_list)

    print(f'\nFinal accuracy: {all_reward / len(problems) * 100:.2f}%  ({int(all_reward)}/{len(problems)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=['aime24', 'aime25', 'gpqa'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--retrieve_num', type=int, default=3)
    parser.add_argument('--exp_name', type=str, default='memp')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--mem_model', type=str, default=None)
    parser.add_argument('--mem_base_url', type=str, default=None)
    args = parser.parse_args()
    main(args)
