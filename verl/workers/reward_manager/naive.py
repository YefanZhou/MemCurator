# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import torch
import numpy as np
import re
import ray
from tqdm import tqdm
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from openai import OpenAI
# from verl.utils.reward_score.math_verify import compute_score
from verl.utils.reward_score.math_ray import compute_score, compute_scores_chunk

# Add import for memory agent bench evaluation
from skillos.llm_agent.metrics import evaluate_wrt_source, _extract_answer_from_response


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

USE_MEMORY_SYSTEM_PROMPT = """
You are a rigorous judge tasked with determining whether a language model's generated reasoning process effectively and accurately utilizes provided memory information. 

### Data Provided:
(1) <Memory>: The context, facts, or previous solutions stored for reference.
(2) <Generated reasoning process>: The chain-of-thought (CoT) or steps produced by the model to solve the current task.

### Scoring Criteria:

#### YES:
- The reasoning process explicitly mentions specific values, logic, or facts from the <Memory>.
- The reasoning process implicitly adapts its strategy based on the <Memory> (e.g., following a specific shortcut or avoiding a mistake mentioned in memory).
- The model correctly uses the <Memory> as a foundation and grounds the current solution on <Memory>.

#### NO:
- The reasoning process is generic and would be exactly the same even if the <Memory> were empty.
- The model ignores the <Memory> and re-derives everything from scratch using only its internal parametric knowledge.
- The model explicitly states that it is not using the <Memory>.
- The model mentions the word 'memory' but does not actually apply the *content* of the memory to the logic.
- The reasoning process contradicts the <Memory> without a valid logical reason.
- The model fabricates using memory items that are not present in the <Memory>.

### Execution Instructions:
1. First, skimming through the content provided in <Memory>.
2. Second, identify specific segments in the <Generated reasoning process> that align with or diverge from the <Memory>.
3. Finally, provide your judgement as either <label>YES</label> or <label>NO</label>.

### Input Format:
<Memory>: <memory_content>
<Generated reasoning process>: <generated_reasoning_process>

### Output Format:
[Brief Analysis]: <your step-by-step reasoning>
[Judge result]: <label>YES</label> or <label>NO</label>
"""

USE_MEMORY_PROMPT = """
Give me the label <label>YES</label> or <label>NO</label> for the following generated reasoning process.
<Memory>: {memory}
<Generated reasoning process>: {generated_reasoning_process}
"""


def batch_process_questions_with_qwen32b(questions, batch_size=32, system_prompt=None, model="Qwen/Qwen3-32B", no_thinking=False, generative_reward=False):
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
        base_url=os.getenv("QWEN_URL"),
        api_key="EMPTY"
    )

    # Initialize tokenizer for prompt conversion
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=True)

    print(f"Starting batch processing of {len(questions)} questions with Qwen3-32B, batch size {batch_size}")

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
                    {"role": "user", "content": question}
                ]

            # Convert to prompt using tokenizer
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Truncate prompt if it exceeds the model's context window, reserving space for the response.
            # Using 40960 as the context window and 1024 for max_tokens to be generated.
            max_prompt_tokens = 40960 - 1024
            tokenized_prompt = tokenizer(prompt)["input_ids"]

            if len(tokenized_prompt) > max_prompt_tokens:
                print(f"Warning: Prompt for question is too long and will be truncated. Original token length: {len(tokenized_prompt)}")
                truncated_ids = tokenized_prompt[:max_prompt_tokens]
                prompt = tokenizer.decode(truncated_ids, skip_special_tokens=True)

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


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", return_separate_scores=False, compression_ratio_weight=1.0,
                 function_content_reward_weight=1.0, function_call_reward_weight=0.5, generative_reward=False, threshold=None) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.return_separate_scores = return_separate_scores
        self.compression_ratio_weight = compression_ratio_weight
        self.function_content_reward_weight = function_content_reward_weight
        self.function_call_reward_weight = function_call_reward_weight
        self.delta_accuracy_weight = 0.5
        self.generative_reward = generative_reward
        self.threshold = threshold

        self.client = OpenAI(
            base_url=os.getenv("QWEN_URL"),
            api_key="EMPTY"
        )

    def _compute_score_for_data_source(self, data_source, predicted_answer, gold_answer, question=None):
        """Compute evaluation score based on data source using the same logic as long_context_eval.py."""

        # Handle thinking tags in predicted answer

        if "<think>" in predicted_answer and "</think>" in predicted_answer:
            predicted_answer = predicted_answer.split("</think>")[1].strip()
        if "<think>" in predicted_answer:
            predicted_answer = "Empty"

        if data_source == 'booksum':
            keywords = gold_answer.split(",")
            keywords = [x.strip() for x in keywords]
            hit = 0
            for keyword in keywords:
                if keyword.lower() in predicted_answer.lower():
                    hit += 1
            return hit / len(keywords)

        elif data_source in ['math', 'aime24', 'amc23']:
            # Use math_verify's compute_score function
            # ref = compute_score.remote(predicted_answer, gold_answer)
            score = compute_score(predicted_answer, gold_answer)

            return score

        elif data_source == 'pubmed-rct' or 'ttl_train' in data_source or 'icl' in data_source:
            # PUBMED dataset evaluation: MUST be ONLY a single digit
            extracted_answer = _extract_answer_from_response(predicted_answer)

            # Remove quotes and strip whitespace
            extracted_answer = extracted_answer.strip('"\'').strip()

            # STRICT pattern: must be EXACTLY a single digit with nothing else
            single_digit_pattern = r'^\d+$'

            if not re.match(single_digit_pattern, extracted_answer):
                return 0.0

            gold_num = str(gold_answer).strip('"\'').strip()

            return 1.0 if extracted_answer == gold_num else 0.0

        elif data_source == 'lme_train':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, gold_answer, predicted_answer)

            response = self.client.chat.completions.create(
                model='Qwen/Qwen3-32B',
                messages=[{"role": "user", "content": prompt}],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                }
            )

            if "yes" in response.choices[0].message.content.strip().lower() and "no" not in response.choices[0].message.content.strip().lower():
                return 1.0
            else:
                return 0.0

        elif data_source == 'perltqa':

            if ";" in gold_answer:
                gold_answer = gold_answer.split(";")
                total_hit = 0
                for answer in gold_answer:
                    if answer.lower().strip() in predicted_answer:
                        total_hit += 1
                return total_hit / len(gold_answer)

            else:
                return 1.0 if gold_answer.lower() in predicted_answer.lower() else 0.0

        elif data_source in ['squad', 'hotpotqa']:
            # Default: containment score for QA datasets
            if isinstance(gold_answer, list):
                answer_text = str(gold_answer[0]) if gold_answer else ""
            else:
                answer_text = gold_answer.get('text', gold_answer) if isinstance(gold_answer, dict) else str(gold_answer)

            return 1.0 if answer_text.lower() in predicted_answer.lower() else 0.0

        else:
            # Memory agent bench evaluation for other datasets
            return evaluate_wrt_source({'output': predicted_answer}, gold_answer, data_source)

    def __call__(self, data: DataProto, data_sources: list, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        is_alfworld = any('alfworld' in str(ds).lower() for ds in data_sources)

        if is_alfworld:
            # AlfWorld: rewards/successes come directly from the environment (no gold answers).
            # per_chunk_rewards/successes are flat [num_chunks] lists pre-built in assembly order,
            # matching all_function_call_rewards layout exactly.
            indices_in_batch = data.meta_info['indices_in_batch']
            total_chunk_length = data.meta_info['total_chunk_length']
            total_memory_length = data.meta_info['total_memory_length']
            all_function_call_rewards = data.meta_info['all_function_call_rewards']
            all_function_call_content_rewards = data.meta_info.get('all_function_call_content_rewards', None)

            all_reward_scores = list(data.meta_info['per_chunk_rewards'])
            copy_accuracy_scores = list(data.meta_info['per_chunk_successes'])
            copy_memory_indicator_scores = [0] * len(all_reward_scores)  # N/A for AlfWorld

            compression_ratio_reward_scores = [
                1 - ml / cl if cl > 0 else 0
                for ml, cl in zip(total_memory_length, total_chunk_length)
            ]
            all_compression_ratio_reward_scores  = [compression_ratio_reward_scores[i] for i in indices_in_batch]
            copy_compression_ratio_reward_scores = all_compression_ratio_reward_scores
            copy_all_function_call_rewards = all_function_call_rewards
            copy_all_function_call_content_rewards = all_function_call_content_rewards

            assert len(copy_accuracy_scores) == len(copy_compression_ratio_reward_scores) \
                == len(copy_all_function_call_rewards) == len(copy_memory_indicator_scores)

            if not self.return_separate_scores:
                if self.threshold is not None:
                    all_reward_scores = [0 if r < self.threshold else 1 for r in all_reward_scores]
                all_reward_scores = [r + self.compression_ratio_weight * c
                                     for r, c in zip(all_reward_scores, all_compression_ratio_reward_scores)]
                if all_function_call_content_rewards is not None:
                    all_reward_scores = [
                        r + f * self.function_call_reward_weight + c * self.function_content_reward_weight
                        for r, f, c in zip(all_reward_scores, all_function_call_rewards, all_function_call_content_rewards)
                    ]
                else:
                    all_reward_scores = [r + f * self.function_call_reward_weight
                                         for r, f in zip(all_reward_scores, all_function_call_rewards)]

        else:
            # Math / QA path: compute accuracy by comparing predictions with ground-truth answers.
            predicted_answers_list = data.meta_info['predicted_answers_list']
            ground_truth_answers_list = data.meta_info['ground_truth_answers_list']
            total_chunk_length = data.meta_info['total_chunk_length']
            total_memory_length = data.meta_info['total_memory_length']
            questions_list = data.meta_info['questions_list']
            memories_list = data.meta_info['memories_list']
            all_function_call_rewards = data.meta_info['all_function_call_rewards']
            all_function_call_content_rewards = data.meta_info['all_function_call_content_rewards'] if "all_function_call_content_rewards" in data.meta_info else None

            compression_ratio_reward_scores = [1 - memory_length / chunk_length if chunk_length > 0 else 0 for memory_length, chunk_length in zip(total_memory_length, total_chunk_length)]
            all_compression_ratio_reward_scores = []
            for i in data.meta_info['indices_in_batch']:
                all_compression_ratio_reward_scores.append(compression_ratio_reward_scores[i])


            # First, collect all prompts that need to be processed by qwen32b (only when generative_reward is True)
            all_qwen_prompts = []
            qwen_prompt_mapping = []  # Track which batch item and question index each prompt belongs to

            if self.generative_reward:
                for i in range(len(ground_truth_answers_list)):
                    if data_sources[i] in ['squad', 'hotpotqa']:
                        # Collect prompts for this batch item
                        for j, (question, predicted_answer, ground_truth_answer) in enumerate(zip(questions_list[i], predicted_answers_list[i], ground_truth_answers_list[i])):
                            prompt = ACCURACY_PROMPT.format(question=question, gold_answer=ground_truth_answer, generated_answer=predicted_answer)
                            all_qwen_prompts.append(prompt)
                            qwen_prompt_mapping.append((i, j))  # Store batch item index and question index

            # Process all qwen prompts in a single batch call if there are any
            if all_qwen_prompts:
                print(f"Processing {len(all_qwen_prompts)} prompts with qwen32b in a single batch call")
                all_qwen_responses = batch_process_questions_with_qwen32b(all_qwen_prompts, batch_size=1024,no_thinking=True, generative_reward=self.generative_reward)
            else:
                all_qwen_responses = []

            # Judge memory usage by calling qwen32b for each response
            all_qwen_memory_prompts = []
            qwen_prompt_memory_mapping = []  # Track which batch item and question index each prompt belongs to

            for i in range(len(memories_list)):
                if data_sources[i] in ['math', 'aime24', 'amc23', 'aime25']:
                    # Collect prompts for this batch item
                    for j, (memory, predicted_answer) in enumerate(zip(memories_list[i], predicted_answers_list[i])):

                        if hasattr(memory, 'core'):
                            memory_content = f"\n[Core Memory]: \n{memory.core}\n"
                        else:
                            memory_content = ""

                        # judge if memory has attribute episodic
                        if hasattr(memory, 'episodic'):
                            episodic_texts = []
                            for entry in memory.episodic:
                                if isinstance(entry, dict):
                                    episodic_texts.extend(map(str, entry.values()))
                            episodic_str = "\n".join(episodic_texts)

                            memory_content += f"[Episodic Memory]:\n{episodic_str}\n"

                        if hasattr(memory, 'semantic'):
                            semantic_texts = []
                            for entry in memory.semantic:
                                if isinstance(entry, dict):
                                    semantic_texts.extend(map(str, entry.values()))
                            semantic_str = "\n".join(semantic_texts)

                            memory_content += f"[Semantic Memory]:\n{semantic_str}\n"

                        if hasattr(memory, 'skills'):
                            skills_texts = []
                            for entry in memory.skills:
                                if isinstance(entry, dict):
                                    skills_texts.extend(map(str, entry.values()))
                            skills_str = "\n".join(skills_texts)

                            memory_content += f"[Skills Memory]:\n{skills_str}\n"


                        prompt = USE_MEMORY_PROMPT.format(memory=memory_content, generated_reasoning_process=predicted_answer)
                        all_qwen_memory_prompts.append(prompt)
                        qwen_prompt_memory_mapping.append((i, j))  # Store batch item index and question index

            if all_qwen_memory_prompts:
                print(f"Processing {len(all_qwen_memory_prompts)} memory prompts with qwen32b batches")
                all_qwen_memory_responses = batch_process_questions_with_qwen32b(all_qwen_memory_prompts, batch_size=1024, system_prompt=USE_MEMORY_SYSTEM_PROMPT, no_thinking=True, generative_reward=self.generative_reward)
            else:
                all_qwen_memory_responses = []

            # Create a mapping from batch item index to qwen memory responses
            qwen_memory_responses_by_batch_item = defaultdict(list)
            for response, (batch_idx, question_idx) in zip(all_qwen_memory_responses, qwen_prompt_memory_mapping):
                qwen_memory_responses_by_batch_item[batch_idx].append(response)


            all_reward_scores = []
            all_memory_scores = []
            all_answer_list = []
            all_pred_list = []

            # Now process each batch item and calculate reward scores
            for i in range(len(ground_truth_answers_list)):
                data_source = data_sources[i]

                # Handle special case for generative reward on squad/hotpotqa
                if data_source in ['squad', 'hotpotqa', 'gpqa'] and self.generative_reward:

                    # Create a mapping from batch item index to qwen responses
                    qwen_responses_by_batch_item = defaultdict(list)
                    for response, (batch_idx, question_idx) in zip(all_qwen_responses, qwen_prompt_mapping):
                        qwen_responses_by_batch_item[batch_idx].append(response)

                    # Use the pre-computed qwen32b responses
                    batch_responses = qwen_responses_by_batch_item[i]
                    all_scores = []
                    for response in batch_responses:
                        if "<label>CORRECT</label>" in response and "<label>WRONG</label>" in response:
                            score = 0  # Default to wrong if both tags present
                        elif "<label>CORRECT</label>" in response:
                            score = 1
                        elif "<label>WRONG</label>" in response:
                            score = 0
                        else:
                            score = 0  # Default to wrong if we can't parse
                        all_scores.append(score)
                    all_reward_scores.append(np.mean(all_scores))

                else:
                    # Use the comprehensive scoring method for all other cases

                    batch_responses = qwen_memory_responses_by_batch_item[i]
                    memory_scores = []
                    for response in batch_responses:
                        if "<label>YES</label>" in response and "<label>NO</label>" in response:
                            mem_score = 0  # Default to NO if both tags present
                        elif "<label>YES</label>" in response:
                            mem_score = 1
                        elif "<label>NO</label>" in response:
                            mem_score = 0
                        else:
                            mem_score = 0  # Default to NO if we can't parse
                        memory_scores.append(mem_score)


                    for inbatch_idx, (pred, answer, question) in enumerate(zip(predicted_answers_list[i], ground_truth_answers_list[i], questions_list[i])):
                        # Use the new comprehensive scoring method
                        # score = self._compute_score_for_data_source(data_source, pred, answer, question)
                        memory_indicator = memory_scores[inbatch_idx]

                        all_answer_list.append(answer)
                        all_pred_list.append(pred)
                        all_memory_scores.append(memory_indicator)



            futures = [compute_score.remote(s, g) for s, g in zip(all_pred_list, all_answer_list)]
            all_acc_scores = ray.get(futures)

            for i in range(len(all_acc_scores)):
                all_reward_scores.append(all_acc_scores[i])

            # Save original reward scores:
            copy_accuracy_scores = all_acc_scores
            copy_compression_ratio_reward_scores = all_compression_ratio_reward_scores
            copy_all_function_call_rewards = all_function_call_rewards
            copy_all_function_call_content_rewards = all_function_call_content_rewards
            copy_memory_indicator_scores = all_memory_scores

            assert len(copy_accuracy_scores) == len(copy_compression_ratio_reward_scores) == len(copy_all_function_call_rewards) == len(copy_memory_indicator_scores)


            if not self.return_separate_scores:
                # meaning we are doing training
                if self.threshold is not None:
                    all_reward_scores = [0 if r < self.threshold else 1 for r in all_reward_scores]

                all_reward_scores = [r + self.compression_ratio_weight * c for r, c in zip(all_reward_scores, all_compression_ratio_reward_scores)]

                if all_function_call_content_rewards is not None:
                    all_reward_scores = [r + (f * self.function_call_reward_weight) + (c * self.function_content_reward_weight) for r, f, c in zip(all_reward_scores, all_function_call_rewards, all_function_call_content_rewards)]
                else:
                    all_reward_scores = [r + (f * self.function_call_reward_weight) for r, f in zip(all_reward_scores, all_function_call_rewards)]


        for i in range(len(all_reward_scores)):

            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            # valid_response_ids = response_ids[:valid_response_length]
            reward_tensor[i, valid_response_length - 1] = all_reward_scores[i]


        if not self.return_separate_scores:
            # Training
            if return_dict:
                # return {"reward_tensor": reward_tensor, "reward_extra_info": {"acc_reward_scores": acc_reward_scores, "compression_ratio_reward_scores": np.mean(compression_ratio_reward_scores)}}
                return {"reward_tensor": reward_tensor, 
                        "reward_extra_info": {
                            "acc_scores": copy_accuracy_scores,
                            "compression_ratio_reward_scores": copy_compression_ratio_reward_scores,
                            "all_function_call_rewards": copy_all_function_call_rewards,
                            "all_function_call_content_rewards": copy_all_function_call_content_rewards,
                            "memory_indicator_scores": copy_memory_indicator_scores,
                            "total_reward_scores": all_reward_scores,
                        }}
            else:
                return reward_tensor
        else:
            if return_dict:
                if all_function_call_content_rewards is not None:
                    return {"reward_tensor": reward_tensor, "reward_extra_info": {"compression_ratio_reward_scores": np.mean(compression_ratio_reward_scores), "all_function_call_rewards": all_function_call_rewards, "all_function_call_content_rewards": all_function_call_content_rewards}}
                else:
                    return {"reward_tensor": reward_tensor, "reward_extra_info": {"compression_ratio_reward_scores": np.mean(compression_ratio_reward_scores), "all_function_call_rewards": all_function_call_rewards}}
            else:
                if all_function_call_content_rewards is not None:
                    return reward_tensor, copy_accuracy_scores, copy_compression_ratio_reward_scores, copy_all_function_call_rewards, copy_all_function_call_content_rewards, copy_memory_indicator_scores
                else:
                    return reward_tensor, copy_accuracy_scores, copy_compression_ratio_reward_scores, copy_all_function_call_rewards, copy_memory_indicator_scores