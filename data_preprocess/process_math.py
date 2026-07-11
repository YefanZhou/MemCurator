from collections import defaultdict
import os
import datasets
import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets
import random
import json
from openai import OpenAI
from transformers import AutoTokenizer
import time
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0):
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output])
        return ret_score
    except TimeoutException:
        return "timeout"
    except Exception as e:
        print(f"Error during score computation: {e}")
        return str(e)



# Setup Qwen client
client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
)

# This function processes the 'golden_answers' field from the nq dataset.
def process_golden_answers(golden_answers, to_string=True):
    """
    Processes 'golden_answers' field and returns a STRING (comma-separated) or empty string.
    Handles: list, tuple, numpy array (1D or scalar), string, number, None, etc.
    """
    items = []

    # Case 1: numpy array
    if isinstance(golden_answers, np.ndarray):
        items = [str(item) for item in golden_answers.flatten() if item is not None and pd.notna(item)]
    # Case 2: list or tuple
    elif isinstance(golden_answers, (list, tuple)):
        items = [str(item) for item in golden_answers if item is not None and pd.notna(item)]
    # Case 3: string
    elif isinstance(golden_answers, str):
        cleaned = golden_answers.strip()
        if cleaned:
            items = [cleaned]
    # Case 4: scalar number (including np.number)
    elif isinstance(golden_answers, (int, float, np.generic)):
        if not pd.isna(golden_answers):
            items = [str(golden_answers).strip()]
    # Case 5: None or empty
    elif golden_answers is None or (isinstance(golden_answers, str) and not golden_answers.strip()):
        items = []
    # Fallback: try string conversion
    else:
        s = str(golden_answers).strip()
        if s and s != "nan":
            items = [s]

    if to_string:
        return "; ".join(items) if items else ""
    else:
        return items

def query_qwen32b_math(questions, system_prompt=None):
    """
    Queries the Qwen3-32B model to solve a math problem.
    """

    # Initialize tokenizer for prompt conversion
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    # Convert all questions in batch to prompts
    batch_prompts = []
    for question in questions:
        im_message = question + "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        if system_prompt is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": im_message}
            ]
        else:
            messages = [
                {"role": "user", "content": im_message}
            ]

        # Convert to prompt using tokenizer
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        batch_prompts.append(prompt)

    # Process the entire batch at once using completions API
    response = client.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt=batch_prompts,
        max_tokens=16384,
        temperature=0.6
    )

    # Extract responses
    batch_responses = [choice.text for choice in response.choices]
    batch_responses = [(x.split("</think>")[1] if "</think>" in x else x) for x in batch_responses]

    return batch_responses


# This function processes the mathhard dataset to a standard format.
def process_math_dataset(dataset):
    """
    Processes the DeepMath-103K dataset to a unified schema.
    """

    instances = list(dataset)
    shuffled_instances = instances[:]
    random.shuffle(shuffled_instances)

    total_count = len(shuffled_instances)
    current_index = 0
    group_id_counter = 0

    print("Processing MathHard dataset in group of instances...")

    # add a progress bar to the following loop
    while current_index < total_count:

        group_size = random.randint(5,10)

        current_group = shuffled_instances[current_index: current_index + group_size]

        if not current_group:
            break

        questions = [instance['question'] for instance in current_group]
        responses = query_qwen32b_math(questions, system_prompt="You are a helpful assistant.")

        group_question = questions[-1]
        group_answer = current_group[-1]['final_answer']

        experiences = []
        for i in range(len(responses)):
            experiences.append({
                'question': questions[i],
                'response': responses[i],
                'final_answer': current_group[i]['final_answer'],
                "difficulty": current_group[i]['difficulty'],
                "topic": current_group[i]['topic'],
            })

        group_record = {
            'id': f'mathhard_group_{group_id_counter}',
            'QA_pairs': experiences,
        }

        with open("./data/math/mathhard_processed_7B.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(group_record, ensure_ascii=False) + "\n")

        group_id_counter += 1
        current_index += group_size

    return group_id_counter


# def regroup_jsonl_finetopic(input_file, output_file):
#     # 1. 加载所有 QA pairs 并按 topic 分类
#     topic_map = defaultdict(list)
    
#     print(f"Reading from {input_file}...")
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             record = json.loads(line.strip())
#             # record['QA_pairs'] 里面存的是该 batch 的所有经验
#             for qa in record.get('QA_pairs', []):
#                 topic_str = qa.get('topic', '')
#                 # 提取最后一个元素作为 key (例如: Limits)
#                 if topic_str:
#                     main_topic = [p.strip() for p in topic_str.split('->') if p.strip()][-1]
#                 else:
#                     main_topic = "Unknown"
                
#                 topic_map[main_topic].append(qa)

#     print(f"Found {len(topic_map)} unique topics.")

#     # 2. 重新打包并写入新文件
#     group_id_counter = 0
#     if os.path.exists(output_file):
#         os.remove(output_file)

#     with open(output_file, 'a', encoding='utf-8') as f_out:
#         for topic_name, all_qas in topic_map.items():
#             # 可选：打乱同一 topic 下的题目顺序
#             random.shuffle(all_qas)
            
#             total_qas = len(all_qas)
#             idx = 0
#             while idx < total_qas:
#                 # 保持你原来的逻辑：每组 5-10 条数据
#                 group_size = random.randint(5, 10)
#                 batch_qas = all_qas[idx : idx + group_size]
                
#                 group_record = {
#                     'id': f'regrouped_{topic_name}_{group_id_counter}',
#                     'main_topic': topic_name,
#                     'QA_pairs': batch_qas
#                 }
                
#                 f_out.write(json.dumps(group_record, ensure_ascii=False) + "\n")
                
#                 group_id_counter += 1
#                 idx += group_size

#     print(f"✅ Regrouping complete! Saved to {output_file}")
#     print(f"Total new groups: {group_id_counter}")

def regroup_filter_jsonl_finetopic(input_file, output_file):
    # 1. 加载所有 QA pairs 并按 topic 分类
    topic_map = defaultdict(list)
    cnt = 0
    
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line.strip())
            # record['QA_pairs'] 里面存的是该 batch 的所有经验
            for qa in record.get('QA_pairs', []):
                topic_str = qa.get('topic', '')
                # 提取最后一个元素作为 key (例如: Limits)
                if topic_str:
                    main_topic = [p.strip() for p in topic_str.split('->') if p.strip()][-1]
                else:
                    main_topic = "Unknown"
    
                accuracy = compute_score(qa['response'], qa['final_answer'])
                
                # if accuracy:
                #     keep = random.random() < 0.4
                # else:
                #     keep = True   # 错误全保留

                # if keep:
                #     topic_map[main_topic].append(qa)
                if isinstance(accuracy, str):
                    print("here!!!!", accuracy)
                    cnt += 1
                else:
                    if accuracy > 0:
                        keep = random.random() < 0.4
                    else:
                        keep = True   # 错误全保留
                    if keep:
                        topic_map[main_topic].append(qa)

    print(f"Found {len(topic_map)} unique topics.")
    print(f"Total unscored questions (None accuracy): {cnt}")

    # 2. 重新打包并写入新文件
    group_id_counter = 0
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for topic_name, all_qas in topic_map.items():
            # 可选：打乱同一 topic 下的题目顺序
            random.shuffle(all_qas)
            
            total_qas = len(all_qas)
            idx = 0
            while idx < total_qas:
                # 保持你原来的逻辑：每组 5-10 条数据
                group_size = random.randint(5, 10)
                batch_qas = all_qas[idx : idx + group_size]
                
                group_record = {
                    'id': f'regrouped_{topic_name}_{group_id_counter}',
                    'main_topic': topic_name,
                    'QA_pairs': batch_qas
                }
                
                f_out.write(json.dumps(group_record, ensure_ascii=False) + "\n")
                
                group_id_counter += 1
                idx += group_size

    print(f"✅ Regrouping complete! Saved to {output_file}")
    print(f"Total new groups: {group_id_counter}")

def main(output_dir='./data/math'):
    """
    Loads and processes MathHard train datasets.
    
    Args:
        output_dir (str): The directory to save the final combined dataset.
    """

    print("\n--- 1. Loading and processing MathHard train dataset ---")
    try:
        math_dataset = datasets.load_dataset('zwhe99/DeepMath-103K', split='train')
        num_groups = process_math_dataset_finetopic(math_dataset)
        print(f"✅ Processed {num_groups} records from MathHard.")
    except Exception as e:
        print(f"❌ Failed to process MathHard dataset: {e}")
        return
    
    print(f"✅ Successfully shuffled and re-indexed MathHard dataset. Total records: {num_groups}")
    print("Example of a combined record:")
    with open("./data/math/mathhard_processed_7B_finetopic.jsonl", "r", encoding="utf-8") as f:
        example_record = json.loads(f.readline().strip())
        print(json.dumps(example_record, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    # main()
    input_path = "./data/math/mathhard_processed_7B.jsonl"
    # with open(input_path, "r", encoding="utf-8") as f:
    #     lines = f.readlines()

    # total = 0
    # cnt = 0
    # for line in lines:
    #     record = json.loads(line.strip())
    #     for qa in record.get('QA_pairs', []):
    #         total += 1
    #         answer = qa.get('final_answer', '')
    #         score = compute_score(qa['response'], answer)
    #         if score:
    #             cnt += 1

    # print(f"Total questions: {total}")
    # print(f"Correctly answered: {cnt}")

    output_path = "./data/math/mathhard_topic_grouped_acc_filtered_abn_7B.jsonl"
    regroup_filter_jsonl_finetopic(input_path, output_path)

# python get_train_data.py