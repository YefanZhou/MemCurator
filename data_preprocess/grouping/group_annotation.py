import os
import time

os.environ["GOOGLE_CLOUD_PROJECT"] = "zifengw-research"
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

from datasets import load_dataset
ds = load_dataset("zwhe99/DeepMath-103K", split="train")

ds = list(ds)

from google import genai
from google.genai import types
from google.genai.errors import ClientError
from tqdm import tqdm
import json

schema = {
    "type": "OBJECT",
    "properties": {
        "Topic": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "Skills or Capabilities": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "Math Concepts or Theorems": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "Heuristic Strategy": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
        "Common Pitfalls": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["Topic", "Skills or Capabilities", "Math Concepts or Theorems", "Heuristic Strategy", "Common Pitfalls"]
}

client = genai.Client()

instruction = """You are an expert in data annotation and mathematical reasoning. 
Given a mathematical question, generate one or more phrases (less than 5 words) that thoroughly and precisely describe the characteristics of the math problem in the following dimensions:
1. Topic
2. Skills or Capabilities
3. Math Concepts or Theorems
4. Heuristic Strategy
5. Common Pitfalls

### Requirements
- The annotations should be phrases only, avoid lengthy sentences
- Do NOT include any context or specifics from the question or solution
- Put your response in JSON format.
- Use as less phrases as possible for each dimension
- Use standardized/acknowledged phrases/terminologies only since phrases generated will be used for large-scale data processing
"""

# Auto-detect resume point from existing output file
output_file = "./group_annotation_DeepMath.jsonl"
start_idx = 0
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        start_idx = sum(1 for _ in f)
    print(f"Resuming from index {start_idx}")

# Retry settings
MAX_RETRIES = 10
BASE_DELAY = 60  # seconds

for i, item in enumerate(tqdm(ds[start_idx:], initial=start_idx, total=len(ds))):
    prompt = item['question']
    
    # Retry loop with exponential backoff
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instruction,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.HIGH
                    ),
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            
            with open(output_file, "a") as f:
                written_dict = {
                    "question": item['question'],
                    "final_answer": item['final_answer'],
                    "difficulty": item['difficulty'],
                    "topic": item['topic'],
                    "annotations": json.loads(response.text)
                }
                f.write(json.dumps(written_dict) + "\n")
            break  # Success, exit retry loop
            
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                delay = BASE_DELAY * (2 ** attempt)  # Exponential backoff
                print(f"\nRate limited at index {start_idx + i}. Waiting {delay}s before retry {attempt + 1}/{MAX_RETRIES}...")
                time.sleep(delay)
                if attempt == MAX_RETRIES - 1:
                    print(f"Max retries reached. Stopping at index {start_idx + i}")
                    raise
            else:
                print(f"\nError at index {start_idx + i}: {e}")
                raise
        except Exception as e:
            print(f"\nUnexpected error at index {start_idx + i}: {e}")
            raise