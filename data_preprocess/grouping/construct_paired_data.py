"""
Data construction program:
    - Reads annotations from group_annotation_DeepMath.jsonl
    - Uses PairSamplerV3 to create high-similarity groups
    - Each group is anchored by one source task, with similar targets added
    - Outputs grouped QA items (each group = N QA_pairs)
"""

import json
import os
import random
from tqdm import tqdm

from grouping_instance_v3 import PairSamplerV3, TaskItem


# ── Paths ──
ANNOTATION_FILE = "./group_annotation_DeepMath.jsonl"
OUTPUT_FILE = "../data/math/mathhard_grouped_7B.jsonl"

# ── Sampler hyper-parameters (all in one place) ──
SAMPLER_KWARGS = dict(
    seed=None,
    # encoder
    embedding_model="all-MiniLM-L6-v2",
    tau=0.60,
    # ── dependency gates (moderate for group formation) ──
    min_shared_concepts=0,           # ≥1 shared math concept
    min_shared_skills=0,             # no hard minimum on shared skills
    require_shared_strategy_or_pitfall=False,  # not required for groups
    require_progression=False,       # not required for groups
    # similarity filters (tightened for higher-quality groups)
    min_overall_similarity=0.4,
    max_topic_soft_jaccard=0.75,
    max_overall_similarity=0.95,
    # difficulty curriculum (allow all directions)
    p_easy_to_hard=0.5,
    p_same=0.3,
    p_hard_to_easy=0.2,
    easy_to_hard_gap=(0.0, 5.0),
    same_gap_abs=1.0,
    hard_to_easy_gap=(-5.0, 0.0),
    # allow negative deltas for groups
    min_difficulty_delta=-5.0,
    # scoring weights (concepts > skills ≈ strategies >> pitfalls)
    w_concepts=4.0,
    w_strategies=3.0,
    w_pitfalls=1.0,
    w_skills=3.0,
    w_topic=3.0,
    w_diff_bonus=1.0,
    # candidate search (larger budget for groups)
    max_inv_candidates=5000,
    fallback_pool=500,
    # negatives (off for now)
    include_hard_negative=False,
)

# ── Grouping parameters ──
GROUP_SIZE_RANGE = (5, 12)  # inclusive
MAX_GROUP_ATTEMPTS = 100


def load_annotations(path: str) -> list:
    """
    Load annotations from group_annotation_DeepMath.jsonl.
    Returns: list of annotation dicts.
    """
    annotations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            annotations.append(json.loads(line.strip()))
    print(f"Loaded {len(annotations)} annotations from {path}")
    return annotations


def build_tasks(annotations: list) -> list:
    """
    Build TaskItem list from annotations, filtering out easy problems.
    """
    tasks = []
    skipped = 0
    for i, ann in enumerate(annotations):
        # Filter out easy problems (difficulty < 5.0)
        if ann.get("difficulty", 0) < 5.0:
            skipped += 1
            continue

        tasks.append(TaskItem(
            task_id=str(i),
            topic=ann["annotations"]["Topic"],
            skills=ann["annotations"]["Skills or Capabilities"],
            concepts=ann["annotations"]["Math Concepts or Theorems"],
            strategies=ann["annotations"]["Heuristic Strategy"],
            pitfalls=ann["annotations"]["Common Pitfalls"],
            difficulty=ann["difficulty"],
            payload=ann,  # Store full annotation for later lookup
        ))

    print(f"Built {len(tasks)} TaskItems ({skipped} skipped)")
    return tasks


def main():
    # 1. Load data
    annotations = load_annotations(ANNOTATION_FILE)

    # 2. Build TaskItems
    tasks = build_tasks(annotations)

    if not tasks:
        print("No matching tasks found. Exiting.")
        return

    # 3. Initialize PairSampler with all key params exposed here
    sampler = PairSamplerV3(tasks, **SAMPLER_KWARGS)

    # 4. Sample groups — no reuse: each task appears in at most one group
    task_order = list(tasks)
    random.shuffle(task_order)

    # Remove existing output file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    used_ids: set = set()           # globally consumed task_ids (no reuse)
    group_id = 0
    skipped_used = 0               # source already consumed
    skipped_small_group = 0        # could not fill minimum group size

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        for src_task in tqdm(task_order, desc="Sampling groups"):
            # Skip sources already consumed by a previous group
            if src_task.task_id in used_ids:
                skipped_used += 1
                continue

            target_group_size = random.randint(*GROUP_SIZE_RANGE)
            group_tasks = [src_task]
            group_ids = {src_task.task_id}
            member_meta = []

            attempts = 0
            while len(group_tasks) < target_group_size and attempts < MAX_GROUP_ATTEMPTS:
                pair = sampler.sample_pair(
                    exclude_ids=used_ids | group_ids,
                    source=src_task,
                )
                if pair is None:
                    break

                if pair.target.task_id in group_ids:
                    attempts += 1
                    continue

                group_tasks.append(pair.target)
                group_ids.add(pair.target.task_id)
                member_meta.append({
                    "target_task_id": pair.target.task_id,
                    "mode": pair.meta["mode"],
                    "is_fallback": pair.meta["is_fallback"],
                    "difficulty_delta": pair.meta["difficulty_delta"],
                    "matched_concepts": pair.meta["matches"]["concepts"]["matched_pairs"],
                    "overall_similarity": pair.meta["overall_similarity"],
                })
                attempts += 1

            if len(group_tasks) < GROUP_SIZE_RANGE[0]:
                skipped_small_group += 1
                continue

            # Mark all tasks in this group as globally consumed
            for t in group_tasks:
                used_ids.add(t.task_id)

            # Sort group by ascending difficulty
            group_tasks.sort(key=lambda t: t.difficulty)

            qa_pairs = []
            for task in group_tasks:
                ann = task.payload
                qa_pairs.append({
                    "question": ann["question"],
                    "final_answer": ann.get("final_answer", ""),
                    "difficulty": ann.get("difficulty", 0),
                    "topic": ann["annotations"]["Topic"],
                })

            group_record = {
                "id": f"mathhard_group_{group_id}",
                "QA_pairs": qa_pairs,
                "pair_meta": {
                    "mode": "group",
                    "anchor_task_id": src_task.task_id,
                    "target_size": target_group_size,
                    "actual_size": len(group_tasks),
                    "members": member_meta,
                },
            }

            f_out.write(json.dumps(group_record, ensure_ascii=False) + "\n")
            group_id += 1

    # ── Summary ──
    total = len(task_order)
    print(f"\n{'='*50}")
    print(f"✅ Done! Wrote {group_id} groups to {OUTPUT_FILE}")
    print(f"   Tasks consumed: {len(used_ids)} / {total}")
    print(f"   Skipped (already used):      {skipped_used}")
    print(f"   Skipped (group < min size):  {skipped_small_group}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
