"""
Compute average score and success rate for WebShop baseline runs.
Usage: python compute_webshop_results.py
"""
import os
import json
import glob
import numpy as np

RESULTS_BASE = "Webshop/results"

EXPERIMENTS = {
    "Qwen3-8B": {
        "model": "openai/Qwen/Qwen3-8B",
        "runs": ["baseline-8b-run1", "baseline-8b-run2", "baseline-8b-run3"],
    },
    "Qwen3-32B": {
        "model": "openai/Qwen/Qwen3-32B",
        "runs": ["baseline-32b-run1", "baseline-32b-run2", "baseline-32b-run3"],
    },
}


def load_run(model_name, exp_name, split="dev", few_shot=False, use_memory=False):
    path = os.path.join(
        RESULTS_BASE, model_name,
        f"{split}_{exp_name}_few_shot_{few_shot}_skillos_{use_memory}"
    )
    files = glob.glob(os.path.join(path, "*.json"))
    if not files:
        return None, path
    rewards = []
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        rewards.append(data["reward"])
    return rewards, path


def main():
    print("=" * 70)
    print("WebShop Baseline Results (No Memory)")
    print("=" * 70)

    all_results = {}

    for model_label, cfg in EXPERIMENTS.items():
        print(f"\n--- {model_label} ---")
        run_scores, run_srs = [], []

        for run_name in cfg["runs"]:
            rewards, path = load_run(cfg["model"], run_name)
            if rewards is None:
                print(f"  {run_name}: NOT FOUND at {path}")
                continue
            avg_score = np.mean(rewards)
            success_rate = np.mean([1.0 if r >= 1.0 else 0.0 for r in rewards])
            run_scores.append(avg_score)
            run_srs.append(success_rate)
            print(f"  {run_name}: n={len(rewards)}  avg_score={avg_score:.4f}  success_rate={success_rate:.4f}")

        if run_scores:
            print(f"  → Mean ± Std  avg_score={np.mean(run_scores):.4f} ± {np.std(run_scores):.4f}  "
                  f"success_rate={np.mean(run_srs):.4f} ± {np.std(run_srs):.4f}")
            all_results[model_label] = {
                "avg_score_mean": np.mean(run_scores),
                "avg_score_std": np.std(run_scores),
                "success_rate_mean": np.mean(run_srs),
                "success_rate_std": np.std(run_srs),
                "num_runs": len(run_scores),
            }

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<15} {'Avg Score':>20} {'Success Rate':>20}")
    print("-" * 60)
    for label, r in all_results.items():
        print(f"{label:<15} {r['avg_score_mean']:.4f} ± {r['avg_score_std']:.4f}   "
              f"{r['success_rate_mean']:.4f} ± {r['success_rate_std']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
