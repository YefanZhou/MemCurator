import json

with open("./data/math/mathhard_topic_grouped_acc_filtered_abn_7B.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
print(f"Total problems: {len(data)}")

training_set = data[:5000]
validation_set = data[5000:5500]


# restruct the training set and convert to parquet files
train_res = []
for idx, item in enumerate(validation_set):
    instance_id = idx
    # prompt = "I will provide you with sequential past experiences of attempts to solve a math problem. Please analyze each experience (e.g., whether the solution is correct, topic, etc.) and decide what memory operations to perform to store this information effectively. Use memory_insert, memory_update, or memory_delete operations as needed."
    chunks = []

    if len(item['QA_pairs']) == 1:
        continue

    for exp in item['QA_pairs'][:-1]:
        chunk_item = ""
        chunk_item += f"[Question] {exp['question']}\n"
        chunk_item += f"[Attempted Solution] {exp['response']}\n"
        chunk_item += f"[Gold Answer] {exp['final_answer']}\n"
        chunk_item += f"[Difficulty] {exp['difficulty']}"
        chunks.append(chunk_item)

    questions_and_answers = [{
        "question": qa['question'],
        "answer": qa['final_answer'],
        'original_pred': qa['response'],
    } for qa in item["QA_pairs"]]


    data_source = "math"
    metadata = {
        "topic": item['main_topic'],
        "data_source": data_source,
        "difficulty": [qa['difficulty'] for qa in item["QA_pairs"]]
    }

    num_chunks = len(chunks)
    num_questions = 1

    train_res.append({
        "instance_id": int(instance_id),
        # "prompt": str(prompt),
        # "chunks": json.dumps(chunks),
        "questions_and_answers": json.dumps(questions_and_answers),
        "data_source": str(data_source),
        "metadata": json.dumps(metadata),
        # "num_chunks": int(num_chunks),
        "num_questions": int(num_questions)
    })

print("Total training instances after filtering:", end=" ")
print(len(train_res))
# store train_res into parquet file
import pandas as pd
train_df = pd.DataFrame(train_res)
train_df.to_parquet("./data/math/test_topic_acc_filtered_abn.parquet", index=False)   