from __future__ import annotations
from typing import List, Dict, Tuple
import json
import os
import openai
import uuid
import math
import re
import numpy as np
from collections import Counter, defaultdict
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import tiktoken
import yaml

def evaluate_yaml_format(content_str):
    clean_str = content_str.strip()
    yaml_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*', re.DOTALL)
    match = yaml_pattern.search(clean_str)
    
    if not match:
        return 0.0
    
    yaml_block = match.group(1)
    
    try:
        parsed_data = yaml.safe_load(yaml_block)
        
        if not isinstance(parsed_data, dict):
            return 0.0
        
        required_keys = {'name', 'description'}
        actual_keys = set(parsed_data.keys())
        
        if actual_keys == required_keys:
            return 1.0
        else:
            return 0.0
            
    except yaml.YAMLError:
        return 0.0
    except Exception:
        return 0.0

def count_tokens(text, model="gpt-4o-mini"):
    """Count tokens using tiktoken"""
    import traceback
    
    encoding = tiktoken.encoding_for_model(model)
    
    # Convert input to string if it's not already a string
    if not isinstance(text, str):
        print(f"!!!! WARNING: Non-string input to count_tokens: {repr(text)} (type: {type(text)})")
        print("!!!! STACK TRACE:")
        traceback.print_stack()
        print("!!!! END STACK TRACE")
        
        # Handle lists by joining them
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
            print(f"!!!! FIXED: Converted list to string for tokenization: {repr(text)}")
        else:
            text = str(text)
            print(f"!!!! FIXED: Converted to string for tokenization: {repr(text)}")
    
    try:
        length = len(encoding.encode(text))
    except Exception as e:
        print(f"!!!! ERROR when processing text: {text}")
        print(f"!!!! ERROR type: {type(e).__name__}: {e}")
        print("!!!! STACK TRACE:")
        traceback.print_stack()
        print("!!!! END STACK TRACE")
        return 0
    return len(encoding.encode(text))


class SkillMemory:

    # Maximum number of items to show for semantic and episodic memories
    MAX_MEMORY_ITEMS = 20
    MEMORY_CONSOLIDATE_STEP = 5 # The number of memories to consolidate at a time
    MODEL = "gpt-4.1-mini"  # Same model as agent.py
    TOPK = 20

    def __init__(self) -> None:
        self.instructions = None
        self.skills: List[Dict[str, str]] = [] # Each dict: {'title': str, 'content': str}
        # Embeddings stored as matrices for batch operations
        self.skills_embedding_matrix: np.ndarray = np.empty((0, 1536))  # text-embedding-3-small has 1536 dimensions
        # Memory ID mappings to track which row corresponds to which memory
        self.skills_embedding_titles: List[str] = []

    def total_length(self):
        total_length = 0
        
        # Handle skills memories
        for skill in self.skills:
            title = skill.get('title', '')
            content = skill.get('content', '')
            
            if not isinstance(title, str):
                print(f"!!!! MEMORY ERROR: Non-string title found for skill title {skill.get('title')}!")
            if not isinstance(content, str):
                print(f"!!!! MEMORY ERROR: Non-string content found for skill title {skill.get('title')}!")

            total_length += count_tokens(title)
            total_length += count_tokens(content)
        
        return total_length

    def _skill_exists_by_title(self, title: str) -> bool:
        """Check if a skill with the given title already exists."""
        for skill in self.skills:
            if skill.get('title') == title:
                return True
        return False

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI's embedding model."""
        try:
            load_dotenv()
            client = openai.OpenAI()
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            # print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return np.zeros(1536)  # text-embedding-3-small has 1536 dimensions

    # --------------------------------------------------
    # Rendering helpers
    # --------------------------------------------------
    def _block(self, title: str = '', lines: List[Dict[str, str]] = []) -> str:
        if not lines:
            if title:
                return f"#### {title}\n\nEmpty."
            else:
                return f"Empty."
        
        # Convert each skill dict to a string representation
        formatted_lines = []
        for skill in lines:
            formatted_lines.append(
                f"**<skill_name>{skill.get('title')}</skill_name>**\n{skill.get('content')}")
        
        body = "\n".join(formatted_lines)
        if title:
            return f"#### {title}\n\n{body}"
        else:
            return body

    def render_system_prompt(self, status: str = "chat", query: str = None, max_num_of_recent_chunks: int = None) -> List[Dict[str, str]]:
        """Return the system prompt expected by the model.
        
        Args:
            status: The mode of operation, can be:
                - "memorie": For memorizing and storing information
        """
        
        max_num_of_recent_chunks = max_num_of_recent_chunks if max_num_of_recent_chunks is not None else self.MAX_MEMORY_ITEMS
        
        if status == "memorie":
            if max_num_of_recent_chunks > 0:
                # If max_num_of_recent_chunks is larger than actual memory count, use all memories
                if max_num_of_recent_chunks >= len(self.skills):
                    skills_items = self.skills
                else:
                    skills_items = self.skills[-max_num_of_recent_chunks:]
            else:
                skills_items = []

            # System prompt for memorizing mode - focus on understanding and storing information

            system_prompt = (f'''# Role
You are an expert with a sophisticated skills curator. Our overall goal is to accomplish agent tasks. Your primary task is to convert past experiences of agent task execution into reusable, general skills, so that they can benefit and inspire future tasks.

# Input Data
1. **Task Description**: The task to be accomplished.
2. **Past Skills**: A list of previously stored relevant skills, each with a skill name (identifier) and content.
3. **Agent Trajectory**: The step-by-step execution trace. Given the task, the agent interacts with the environment by selecting and calling specific past skills. This trajectory captures the sequence of skill invocations and the resulting transitions used to pursue the goal.
4. **Result**: Whether the agent successfully completed the task or not.
                             
# Critical Constraints:
- **Skill Format**: Extract and store important information as skills using following Markdown format **strictly**.
- **No Specifics**: Avoid problem-specific details. Remove specific numbers/names. Replace with variables/concepts.
- **No Hallucination**: Do not invent facts.
- Each skill must be **Atomic, modular, and reusable**.

# Skill Markdown Format and Content Instructions:
- **YAML Frontmatter (MANDATORY)**: Each skill MUST start with a YAML frontmatter block delimited by `---`. The YAML block MUST contain exactly two keys: `name` and `description`.
    - **Example Structure**:
    ---
    name: <Human-readable skill name>
    description: <One-sentence what/when/why/how summary, concise and actionable, this will be used for future references>
    ---
- **Markdown Body**: Immediately after the second `---`, provide instructions using Markdown headings.
    - Suggested sections: `# Workflow`, `# When NOT to use`, `# Prerequisite Constraints`. These headings are just examples, you can come up with more ideas; use and craft what's appropriate for clarity.
    - Ensure the content is atomic, general, and devoid of specific instance IDs.

# Action Guidelines
1. Analyze the agent trajectory and its result. Identify what went well and what didn't.
2. If the trajectory is correct, extract reusable knowledge or skills. If the trajectory is incorrect, identify the failure point and extract skills that can help fix the issue.
3. Compare the extracted skills with past skills. Determine whether to **insert a new skill**, **update an existing skill**, or **delete an existing skill** using the following tools.
''')

            return [
                {"role": "system", "content": system_prompt},
            ]
        
        # Other statuses are not supported by SkillMemory
        # A generic prompt for other cases
        return [{"role": "system", "content": "You are a helpful assistant."}]

    # --------------------------------------------------
    # Memory operations – called by functions.py
    # --------------------------------------------------
    def new_memory_insert(self, title: str, content: str) -> str:
        """Insert a new skill. Raises ValueError if title already exists."""
        if self._skill_exists_by_title(title):
            raise ValueError(f"Skill with title '{title}' already exists.")

        # if evaluate_yaml_format(content) == 0.0:
        #     raise ValueError(f"Content does not meet YAML format requirements.")

        new_skill = {'title': title, 'content': content}
        self.skills.append(new_skill)

        # Generate and store embedding for the new skill
        embedding_text = f"Title: {title}\nContent: {content}"
        embedding = self._get_embedding(embedding_text)

        # Append embedding to matrix
        if self.skills_embedding_matrix.size == 0:
            self.skills_embedding_matrix = embedding.reshape(1, -1)
        else:
            self.skills_embedding_matrix = np.vstack([self.skills_embedding_matrix, embedding.reshape(1, -1)])
        self.skills_embedding_titles.append(title)

        return title

    def memory_update(self, title: str, new_name: str = None, new_content: str = None) -> Dict[str, str]:
        """Update a skill's title and/or content by its title."""
        if not new_name and not new_content:
            raise ValueError("Either new_name or new_content must be provided for update.")
        # if new_content and evaluate_yaml_format(new_content) == 0.0:
        #     raise ValueError(f"New content does not meet YAML format requirements.")

        skill_to_update = None
        for skill in self.skills:
            if skill['title'] == title:
                skill_to_update = skill
                break

        if not skill_to_update:
            raise ValueError(f"Skill with title '{title}' not found.")

        if new_name:
            # Check for title uniqueness if it's being changed
            if new_name != title and self._skill_exists_by_title(new_name):
                raise ValueError(f"Skill with title '{new_name}' already exists.")
            skill_to_update['title'] = new_name

        if new_content:
            skill_to_update['content'] = new_content

        # Update embedding
        embedding_text = f"Title: {skill_to_update['title']}\nContent: {skill_to_update['content']}"
        embedding = self._get_embedding(embedding_text)

        try:
            idx = self.skills_embedding_titles.index(title)
            self.skills_embedding_matrix[idx] = embedding
            if new_name:
                self.skills_embedding_titles[idx] = new_name
        except ValueError:
            # This should not happen if skill_id is valid
            print(f"Warning: Skill title '{title}' not found in embedding matrix for update.")

        return skill_to_update

    def memory_delete(self, title: str):
        """Delete a skill by its title."""
        skill_found = False
        for i, skill in enumerate(self.skills):
            if skill['title'] == title:
                self.skills.pop(i)
                skill_found = True
                break

        if not skill_found:
            raise ValueError(f"Skill with title '{title}' not found.")

        # Delete corresponding embedding
        try:
            idx = self.skills_embedding_titles.index(title)
            self.skills_embedding_matrix = np.delete(self.skills_embedding_matrix, idx, axis=0)
            self.skills_embedding_titles.pop(idx)
        except ValueError:
            # This should not happen if skill was found in self.skills
            print(f"Warning: Skill title '{title}' not found in embedding matrix for deletion.")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on whitespace and punctuation."""
        # Convert to lowercase and split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def memory_search(self, query: str, top_k: int = None, min_score: float = 0.0, search_method: str = "bm25") -> List[Tuple[Dict[str, str], float]]:
        """Search for skills using BM25 or text embedding similarity.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return (None for all)
            min_score: Minimum score threshold (BM25 score or cosine similarity)
            search_method: Search method to use ('bm25' or 'text-embedding')
            
        Returns:
            List of tuples containing (skill_dict, score) sorted by score descending
        """
        if not self.skills or not query.strip():
            return []
        
        if search_method == "bm25":
            return self._search_bm25(query, top_k, min_score)
        elif search_method == "text-embedding":
            return self._search_embedding(query, top_k, min_score)
        else:
            raise ValueError(f"Unknown search method: {search_method}. Use 'bm25' or 'text-embedding'.")

    def _search_bm25(self, query: str, top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, str], float]]:
        """Search skills using BM25 ranking algorithm."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Tokenize all documents (skills)
        tokenized_corpus = [self._tokenize(f"{skill['title']} {skill['content']}") for skill in self.skills]
        
        if not tokenized_corpus:
            return []
        
        # Create BM25 object
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Get scores for the query
        doc_scores = bm25.get_scores(query_tokens)
        
        # Create results with scores
        results = []
        for i, skill in enumerate(self.skills):
            score = doc_scores[i]
            if score >= min_score:
                results.append((skill, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]
        
        return results

    def _search_embedding(self, query: str, top_k: int = None, min_score: float = 0.0) -> List[Tuple[Dict[str, str], float]]:
        """Search skills using text embedding cosine similarity."""
        if not self.skills or self.skills_embedding_matrix.shape[0] == 0:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        if np.allclose(query_embedding, 0):  # Check if embedding generation failed
            return []
        
        # Batch calculate cosine similarity for all embeddings at once
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), 
            self.skills_embedding_matrix
        )[0]  # Extract the first (and only) row
        
        results = []
        
        # Create a mapping from skill_id to skill for fast lookup
        title_to_skill = {skill['title']: skill for skill in self.skills}
        
        # Combine similarities with memory content
        for title, similarity in zip(self.skills_embedding_titles, similarities):
            if similarity >= min_score and title in title_to_skill:
                results.append((title_to_skill[title], float(similarity)))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit if specified
        if top_k is not None:
            results = results[:top_k]
        return results
