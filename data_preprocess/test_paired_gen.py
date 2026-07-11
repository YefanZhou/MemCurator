import json
import random
import pandas as pd
from datasets import load_dataset

# Set random seed for reproducibility
random.seed(42)

# Load datasets from HuggingFace
print("Loading AIME 2024 dataset...")
aime24_dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
print(f"AIME 2024 problems: {len(aime24_dataset)}")

print("Loading AMC 2023 dataset...")
amc23_dataset = load_dataset("math-ai/amc23", split="test")
print(f"AMC 2023 problems: {len(amc23_dataset)}")

print("Loading AIME 2025 dataset...")
aime25_dataset = load_dataset("math-ai/aime25", split="test")
print(f"AIME 2025 problems: {len(aime25_dataset)}")

# Prepare test instances
test_res = []
instance_id = 0

# Process AIME 2024 - all 30 problems as one test instance, repeat 16 times
aime24_questions_and_answers = [{
    "question": item['Problem'],
    "answer": str(item['Answer']),
    "original_pred": "",  # leave blank
} for item in aime24_dataset]

for repeat_idx in range(16):
    test_res.append({
        "instance_id": int(instance_id),
        "prompt": "I will provide you with sequential past experiences of attempts to solve a math problem. Please analyze each experience (e.g., whether the solution is correct, etc.) and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.",
        "questions_and_answers": json.dumps(aime24_questions_and_answers),
        "data_source": "aime24",
        "metadata": json.dumps({
            "topic": "AIME 2024",
            "data_source": "aime24",
            "difficulty": [""] * len(aime24_questions_and_answers)  # leave difficulty blank
        }),
        "num_questions": int(len(aime24_questions_and_answers))
    })
    instance_id += 1

print(f"Added {16} AIME 2024 test instances (each with {len(aime24_questions_and_answers)} problems)")

# Process AMC 2023 - all 40 problems as one test instance, repeat 16 times
amc23_questions_and_answers = [{
    "question": item['question'],
    "answer": str(item['answer']),
    "original_pred": "",  # leave blank
} for item in amc23_dataset]

for repeat_idx in range(16):
    test_res.append({
        "instance_id": int(instance_id),
        "prompt": "I will provide you with sequential past experiences of attempts to solve a math problem. Please analyze each experience (e.g., whether the solution is correct, etc.) and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.",
        "questions_and_answers": json.dumps(amc23_questions_and_answers),
        "data_source": "amc23",
        "metadata": json.dumps({
            "topic": "AMC 2023",
            "data_source": "amc23",
            "difficulty": [""] * len(amc23_questions_and_answers)  # leave difficulty blank
        }),
        "num_questions": int(len(amc23_questions_and_answers))
    })
    instance_id += 1

print(f"Added {16} AMC 2023 test instances (each with {len(amc23_questions_and_answers)} problems)")

# Process AIME 2025 - all problems as one test instance, repeat 16 times
aime25_questions_and_answers = [{
    "question": item['problem'],
    "answer": str(item['answer']),
    "original_pred": "",  # leave blank
} for item in aime25_dataset]

for repeat_idx in range(16):
    test_res.append({
        "instance_id": int(instance_id),
        "prompt": "I will provide you with sequential past experiences of attempts to solve a math problem. Please analyze each experience (e.g., whether the solution is correct, etc.) and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.",
        "questions_and_answers": json.dumps(aime25_questions_and_answers),
        "data_source": "aime25",
        "metadata": json.dumps({
            "topic": "AIME 2025",
            "data_source": "aime25",
            "difficulty": [""] * len(aime25_questions_and_answers)  # leave difficulty blank
        }),
        "num_questions": int(len(aime25_questions_and_answers))
    })
    instance_id += 1

print(f"Added {16} AIME 2025 test instances (each with {len(aime25_questions_and_answers)} problems)")

# # Load GPQA dataset from local JSON file
# print("\nLoading GPQA dataset...")
# with open("/home/ouyangsiru/SkillCurator/data/GPQA.json", "r") as f:
#     gpqa_data = json.load(f)
# print(f"GPQA total problems: {len(gpqa_data)}")

# # Randomly sample 50 problems from GPQA
# gpqa_sampled = random.sample(gpqa_data, 50)
# print(f"GPQA sampled problems: {len(gpqa_sampled)}")

# # Process GPQA - 50 sampled problems as one test instance, repeat 16 times
# gpqa_questions_and_answers = [{
#     "question": item['query'],  # use 'query' which includes choices
#     "answer": item['answer'],
#     "original_pred": "",  # leave blank
# } for item in gpqa_sampled]

# for repeat_idx in range(16):
#     test_res.append({
#         "instance_id": int(instance_id),
#         "prompt": "I will provide you with sequential past experiences of attempts to solve a math problem. Please analyze each experience (e.g., whether the solution is correct, etc.) and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed.",
#         "questions_and_answers": json.dumps(gpqa_questions_and_answers),
#         "data_source": "gpqa",
#         "metadata": json.dumps({
#             "topic": "GPQA",
#             "data_source": "gpqa",
#             "difficulty": [""] * len(gpqa_questions_and_answers)  # leave difficulty blank
#         }),
#         "num_questions": int(len(gpqa_questions_and_answers))
#     })
#     instance_id += 1

# print(f"Added {16} GPQA test instances (each with {len(gpqa_questions_and_answers)} problems)")

# Create DataFrame and save to parquet
print(f"\nTotal test instances: {len(test_res)}")
test_df = pd.DataFrame(test_res)
output_path = "/home/ouyangsiru/SkillCurator/data/math/test_paired.parquet"
test_df.to_parquet(output_path, index=False)
print(f"Saved to {output_path}")

# Verify the output
print("\nVerification:")
loaded_df = pd.read_parquet(output_path)
print(f"Loaded {len(loaded_df)} instances")
print(f"Columns: {list(loaded_df.columns)}")
print(f"\nData source distribution:")
print(loaded_df['data_source'].value_counts())
