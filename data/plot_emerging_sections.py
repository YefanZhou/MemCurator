import os
import json
import re
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/home/siruo_google_com/SkillCurator/data/math/qwen3-8b-8b-skills-alfworld-signal+content0.1+compression0.05/rollout/training"
OUTPUT_PATH = os.path.join(os.path.dirname(DATA_DIR.rstrip("/")), "emerging_sections_frequency.png")

# Known/baseline sections to exclude (normalized)
KNOWN_SECTIONS = {
    "workflow",
    "when not to use",
    "when not to use",
    "prerequisite constraints",
    "prerequisite constraint",
}

def normalize(header: str) -> str:
    """Strip leading #, lowercase, strip trailing punctuation/whitespace."""
    h = re.sub(r'^#+\s*', '', header).strip().rstrip(':').lower()
    return h

def canonical(norm: str) -> str:
    """Group similar section names into a single canonical label."""
    if re.search(r'additional\s+note', norm):
        return "Additional Notes"
    if re.search(r'^note', norm):
        return "Additional Notes"
    if re.search(r'additional\s+consider', norm):
        return "Additional Considerations"
    if re.search(r'additional\s+constrain', norm):
        return "Additional Constraints"
    if re.search(r'additional\s+guid', norm):
        return "Additional Guidance"
    if re.search(r'additional\s+tip', norm):
        return "Additional Tips"
    if re.search(r'additional\s+recommend', norm):
        return "Additional Recommendations"
    if re.search(r'additional\s+step', norm):
        return "Additional Steps"
    if re.search(r'additional\s+action', norm):
        return "Additional Actions"
    if re.search(r'additional\s+strat', norm):
        return "Additional Strategy"
    if re.search(r'additional\s+info', norm):
        return "Additional Information"
    if re.search(r'^additional\s+instruct', norm):
        return "Additional Instructions"
    if re.search(r'^additional', norm):
        return "Additional (other)"
    if re.search(r'optim|efficien', norm):
        return "Optimization / Efficiency"
    if re.search(r'enhance|improvement|improve', norm):
        return "Enhancement / Improvement"
    if re.search(r'adapt', norm):
        return "Adaptation Strategy"
    if re.search(r'general', norm):
        return "Generalization"
    if re.search(r'best\s+pract', norm):
        return "Best Practices"
    if re.search(r'when\s+to\s+use$', norm):
        return "When to Use"
    if re.search(r'extended\s+alert', norm):
        return "Extended Alerts"
    if re.search(r'advanced\s+guid', norm):
        return "Advanced Guidance"
    return norm.title()

def extract_content(output: str):
    """Yield skill content strings from new_skill_insert and skill_update calls."""
    for match in re.finditer(r'✿ARGS✿:\s*(\{.*?\})\s*(?=✿|$)', output, re.DOTALL):
        try:
            args = json.loads(match.group(1))
            content = args.get('content') or args.get('new_content') or ''
            if content:
                yield content
        except (json.JSONDecodeError, AttributeError):
            pass

# --- Collect per-step section counts ---
files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl")],
    key=lambda f: int(os.path.splitext(f)[0])
)

steps = []
# counts_by_step[step] = Counter of canonical section names
counts_by_step = {}

for fname in files:
    step = int(os.path.splitext(fname)[0])
    counter = Counter()

    with open(os.path.join(DATA_DIR, fname), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                for content in extract_content(d.get('output', '')):
                    headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
                    for h in headers:
                        norm = normalize(h)
                        if norm in KNOWN_SECTIONS:
                            continue
                        canon = canonical(norm)
                        counter[canon] += 1
            except json.JSONDecodeError:
                pass

    steps.append(step)
    counts_by_step[step] = counter

# Find top N canonical sections by total count across all steps
total_counter = Counter()
for c in counts_by_step.values():
    total_counter.update(c)

TOP_N = 10
top_sections = [sec for sec, _ in total_counter.most_common(TOP_N)]

print(f"Top {TOP_N} emerging sections (by total count):")
for sec, cnt in total_counter.most_common(TOP_N):
    print(f"  {cnt:6d}  {sec}")

# --- Build per-step arrays ---
data = {sec: [counts_by_step[s].get(sec, 0) for s in steps] for sec in top_sections}

# --- Plot ---
cmap = plt.get_cmap("tab10")
colors = [cmap(i) for i in range(TOP_N)]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

for i, sec in enumerate(top_sections):
    ax1.plot(steps, data[sec], marker='o', markersize=4, linewidth=1.6,
             color=colors[i], label=sec)

ax1.set_ylabel("Count per step")
ax1.set_title("Emerging Markdown Sections per Training Step\n"
              "(excluding Workflow / When NOT to use / Prerequisite Constraints)", fontsize=11)
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.35)

# Stacked area for share
vals = np.array([data[sec] for sec in top_sections], dtype=float)
ax2.stackplot(steps, vals, labels=top_sections, colors=colors, alpha=0.8)
ax2.set_xlabel("Training Step")
ax2.set_ylabel("Stacked count")
ax2.legend(fontsize=8, ncol=2, loc="upper left")
ax2.grid(True, alpha=0.35)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
print(f"\nPlot saved to: {OUTPUT_PATH}")
