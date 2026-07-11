import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/home/siruo_google_com/SkillCurator/data/math/qwen3-8b-8b-skills-alfworld-signal+content0.1+compression0.05/rollout/training"
OUTPUT_PATH = os.path.join(os.path.dirname(DATA_DIR.rstrip("/")), "memory_ops_frequency.png")

# Map actual function names in data -> display labels
OP_MAP = {
    "skill_update":    "memory_update",
    "skill_delete":    "memory_delete",
    "new_skill_insert":"new_memory_insert",
}

files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")],
    key=lambda f: int(os.path.splitext(f)[0])
)

steps = []
counts = {label: [] for label in OP_MAP.values()}
totals = []

for fname in files:
    step = int(os.path.splitext(fname)[0])
    step_counts = {label: 0 for label in OP_MAP.values()}
    total = 0

    with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                d = json.loads(line)
                output = d.get("output", "")
                func_names = set(re.findall(r"✿FUNCTION✿:\s*(\w+)", output))
                for func, label in OP_MAP.items():
                    if func in func_names:
                        step_counts[label] += 1
            except json.JSONDecodeError:
                pass

    steps.append(step)
    totals.append(total)
    for label in OP_MAP.values():
        counts[label].append(step_counts[label])

# Convert to percentages
totals = np.array(totals)
pct = {label: np.array(counts[label]) / totals * 100 for label in OP_MAP.values()}

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

colors = {
    "memory_update":     "#2196F3",
    "memory_delete":     "#F44336",
    "new_memory_insert": "#4CAF50",
}
markers = {"memory_update": "o", "memory_delete": "s", "new_memory_insert": "^"}

# Top panel: raw counts
for label in OP_MAP.values():
    ax1.plot(steps, counts[label], marker=markers[label], color=colors[label],
             linewidth=1.8, markersize=5, label=label)
ax1.set_ylabel("Count per step")
ax1.set_title("Memory Operation Frequencies per Training Step\n"
              f"({os.path.basename(os.path.dirname(DATA_DIR))})", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.4)

# Bottom panel: percentage of total records
for label in OP_MAP.values():
    ax2.plot(steps, pct[label], marker=markers[label], color=colors[label],
             linewidth=1.8, markersize=5, label=label)
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Percentage of records (%)")
ax2.legend()
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"Plot saved to: {OUTPUT_PATH}")

# Print summary table
print(f"\n{'Step':>5} {'Total':>7} {'memory_update':>15} {'memory_delete':>15} {'new_memory_insert':>18}")
print("-" * 65)
for i, s in enumerate(steps):
    row = f"{s:>5} {totals[i]:>7}"
    for label in OP_MAP.values():
        row += f"  {counts[label][i]:>5} ({pct[label][i]:5.1f}%)"
    print(row)
