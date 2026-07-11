"""
eval.py — Evaluate an ALFWorld result directory.

Usage:
    python eval.py <results_dir>
"""

import os
import sys
import json
from collections import defaultdict

TASK_TYPES = ['put', 'clean', 'heat', 'cool', 'examine', 'puttwo']

NAME_PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo',
}


def get_task_type(name: str) -> str:
    for prefix, task in NAME_PREFIXES.items():
        if name.startswith(prefix):
            return task
    return 'unknown'


def count_steps(messages: list) -> int:
    return sum(1 for m in messages if m['role'] == 'assistant')


def main():
    if len(sys.argv) != 2:
        print("Usage: python eval.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not files:
        print(f"No JSON files found in {results_dir}")
        sys.exit(1)

    task_successes = defaultdict(list)
    task_steps = defaultdict(list)
    all_successes = []
    all_steps = []

    for fname in files:
        with open(os.path.join(results_dir, fname)) as f:
            r = json.load(f)
        task_type = get_task_type(r.get('name', ''))
        reward = int(r.get('reward', 0))
        steps = count_steps(r.get('messages', []))
        task_successes[task_type].append(reward)
        task_steps[task_type].append(steps)
        all_successes.append(reward)
        all_steps.append(steps)

    print(f"\nResults: {results_dir}")
    print(f"{'='*58}")
    print(f"{'Task':<12} {'Succ/Total':>12} {'SR':>8} {'Avg Steps':>12}")
    print(f"{'-'*58}")

    for t in TASK_TYPES:
        succs = task_successes.get(t, [])
        steps = task_steps.get(t, [])
        if not succs:
            print(f"{t:<12} {'N/A':>12}")
            continue
        sr = sum(succs) / len(succs)
        avg_steps = sum(steps) / len(steps)
        print(f"{t:<12} {f'{sum(succs)}/{len(succs)}':>12} {sr:>8.3f} {avg_steps:>12.1f}")

    print(f"{'-'*58}")
    n = len(all_successes)
    overall_sr = sum(all_successes) / n
    avg_steps_all = sum(all_steps) / n
    success_steps = [s for s, r in zip(all_steps, all_successes) if r]
    avg_steps_success = sum(success_steps) / len(success_steps) if success_steps else 0
    print(f"{'overall':<12} {f'{sum(all_successes)}/{n}':>12} {overall_sr:>8.3f} {avg_steps_all:>12.1f}")
    print(f"\n  Avg steps (successful games only): {avg_steps_success:.1f}")


if __name__ == '__main__':
    main()
