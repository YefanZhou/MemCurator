"""
Compute averaged accuracy across 3 trials for reasoning experiments.
Reports per-dataset and averaged results for skillos and reasoningbank,
with Qwen3-8B and Qwen3-32B executors.
"""
import os
import json
from collections import defaultdict

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "Reasoning", "results")

ENVS = ["aime24", "aime25", "amc23", "gpqa"]

SETTINGS = [
    # (label,                    path_prefix,                         exp_prefix)
    ("Baseline-8B",              "openai/Qwen/Qwen3-8B",             "baseline-8b"),
    ("Baseline-32B",             "openai/Qwen/Qwen3-32B",            "baseline-32b"),
    ("Baseline-Gemini",          "gemini/gemini-2.5-pro",            "baseline-gemini25pro"),
    ("SkillOS-8B",               "openai/Qwen/Qwen3-8B",             "skillos-8b"),
    ("SkillOS-32B",              "openai/Qwen/Qwen3-32B",            "skillos-32b"),
    ("SkillOS-GemCur-8B",        "openai/Qwen/Qwen3-8B",             "skillos-gemcur-8b"),
    ("SkillOS-GemCur-32B",       "openai/Qwen/Qwen3-32B",            "skillos-gemcur-32b"),
    ("SkillOS-GemCur-Gemini",    "gemini/gemini-2.5-pro",            "skillos-gemcur-gem"),
    ("SkillOS-QwenCur-Gemini",   "gemini/gemini-2.5-pro",            "skillos-gemini25pro"),
    ("RB-QwenCur-Gemini",        "gemini/gemini-2.5-pro",            "rb-gemini25pro"),
    ("SkillOS-FT-8B",            "openai/Qwen/Qwen3-8B",             "skillos-ft-8b"),
    ("SkillOS-FT-32B",           "openai/Qwen/Qwen3-32B",            "skillos-ft-32b"),
    ("ReasoningBank-8B",         "openai/Qwen/Qwen3-8B",             "rb-8b"),
    ("ReasoningBank-32B",        "openai/Qwen/Qwen3-32B",            "rb-32b"),
]

RUNS = [1, 2, 3]


def acc_for_dir(path):
    """Return accuracy (0-1) for a result directory, or None if missing/empty."""
    if not os.path.isdir(path):
        return None
    rewards = []
    for fname in os.listdir(path):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(path, fname)) as f:
            data = json.load(f)
        rewards.append(float(data.get("reward", 0)))
    if not rewards:
        return None
    return sum(rewards) / len(rewards)


def find_exp_dir(env, path_prefix, exp_prefix, run):
    """
    Find the experiment directory.
    Handles both naming conventions:
      - {exp_prefix}-run{run}
      - {exp_prefix}-run{run}_{memory_type}
    """
    base = os.path.join(RESULTS_DIR, env, path_prefix)
    if not os.path.isdir(base):
        return None
    for name in os.listdir(base):
        if name.startswith(f"{exp_prefix}-run{run}"):
            return os.path.join(base, name)
    return None


def main():
    col_w = 22
    env_w = 10

    header = f"{'Setting':<{col_w}}" + "".join(f"{e.upper():>{env_w}}" for e in ENVS)
    print(header)
    print("-" * len(header))

    for label, path_prefix, exp_prefix in SETTINGS:
        row = f"{label:<{col_w}}"
        for env in ENVS:
            run_accs = []
            for run in RUNS:
                d = find_exp_dir(env, path_prefix, exp_prefix, run)
                acc = acc_for_dir(d) if d else None
                if acc is not None:
                    run_accs.append(acc)

            if run_accs:
                avg = sum(run_accs) / len(run_accs)
                runs_str = "/".join(f"{a*100:.1f}" for a in run_accs)
                cell = f"{avg*100:.1f}({runs_str})"
            else:
                cell = "N/A"
            row += f"{cell:>{env_w}}"
        print(row)

    print()
    print("Format: avg%(run1/run2/run3)")


if __name__ == "__main__":
    main()
