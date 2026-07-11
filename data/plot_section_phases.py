import os, json, re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR = "/home/siruo_google_com/SkillCurator/data/math/qwen3-8b-8b-skills-alfworld-signal+content0.1+compression0.05/rollout/training"
OUTPUT_PATH = os.path.join(os.path.dirname(DATA_DIR.rstrip("/")), "section_phases.png")

KNOWN = {'workflow', 'when not to use', 'prerequisite constraints', 'prerequisite constraint'}

def normalize(h):
    return re.sub(r'^#+\s*', '', h).strip().rstrip(':').lower()

def classify(norm):
    # Malformed: clearly truncated standard section names
    if re.match(r'^(prerequi|prere|wh$|when no$|when not to$|p$|pre$)', norm):
        return 'malformed'
    # Task-specific: numbered steps or highly specific object/task references
    if re.match(r'^\d+\.', norm):
        return 'task_specific'
    if re.search(r'specific|potato|knife|keychain|cleaning sub|identification sub|extended alert', norm):
        return 'task_specific'
    # Robustness: error/failure/retry/success handling
    if re.search(r'error|failure|retry|fallback|success criteria|alternative method|handling missing|special case', norm):
        return 'robustness'
    # Optimization / Enhancement (early peaking)
    if re.search(r'optim|efficien|enhance|improvement|improve|best pract|generaliz|adapt', norm):
        return 'optim_enhance'
    # Catch-all extra notes
    if re.search(r'additional|note$|notes$|tip|guid|recommend|instruct|consider|info|step|action|strat', norm):
        return 'additional_notes'
    return 'other_extra'

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.jsonl')],
               key=lambda f: int(os.path.splitext(f)[0]))

steps = []
metrics = {k: [] for k in ['malformed', 'task_specific', 'robustness',
                             'optim_enhance', 'additional_notes', 'other_extra', 'total_extra']}
n_records_per_step = []

for fname in files:
    step = int(os.path.splitext(fname)[0])
    counts = Counter()
    n_records = 0

    with open(os.path.join(DATA_DIR, fname)) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                d = json.loads(line)
                output = d.get('output', '')
                for m in re.finditer(
                    r'✿FUNCTION✿:\s*(new_skill_insert|skill_update|skill_insert|update_skill)\s*\n✿ARGS✿:\s*(\{.*)',
                    output, re.DOTALL):
                    n_records += 1
                    snippet = m.group(2)[:3000]
                    headers = re.findall(r'^#{1,3}\s+(.+)$', snippet, re.MULTILINE)
                    for h in headers:
                        norm = normalize(h)
                        if norm in KNOWN: continue
                        cat = classify(norm)
                        counts[cat] += 1
            except: pass

    steps.append(step)
    n_records_per_step.append(n_records)
    total_extra = sum(counts.values())
    metrics['total_extra'].append(total_extra)
    for k in ['malformed', 'task_specific', 'robustness', 'optim_enhance', 'additional_notes', 'other_extra']:
        metrics[k].append(counts[k])

steps = np.array(steps)
n_rec = np.array(n_records_per_step, dtype=float)
n_rec[n_rec == 0] = 1  # avoid div/0

# Convert to percentage of skill records
pct = {k: np.array(metrics[k]) / n_rec * 100 for k in metrics}

# Smoothing helper (3-step rolling mean)
def smooth(arr, w=3):
    return np.convolve(arr, np.ones(w)/w, mode='same')

# --- Figure: 3 panels ---
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
phase_colors = ['#E3F2FD', '#FFF9C4', '#E8F5E9']  # light blue / yellow / green
phase_spans  = [(1, 17), (18, 34), (35, 50)]
phase_labels = ['Early (1–17)\nNoisy / format-learning',
                'Mid (18–34)\nConservative / convergent',
                'Late (35–50)\nRobustness-enriching']

def add_phases(ax):
    for (x0, x1), col, lbl in zip(phase_spans, phase_colors, phase_labels):
        ax.axvspan(x0, x1, color=col, alpha=0.45, zorder=0)

# ── Panel 1: Format discipline (malformed + task-specific) ──────────────────
ax = axes[0]
add_phases(ax)
ax.bar(steps, pct['malformed'],     color='#EF5350', label='Malformed headers')
ax.bar(steps, pct['task_specific'], color='#FF8A65', bottom=pct['malformed'],
       label='Task-specific headers')
ax.set_ylabel('% of skill records')
ax.set_title('Panel 1 — Format Discipline: Malformed & Task-specific Headers', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.4)
# Annotate phase averages
for (x0, x1), col in zip(phase_spans, ['#1565C0','#F57F17','#2E7D32']):
    mask = (steps >= x0) & (steps <= x1)
    avg = (pct['malformed'][mask] + pct['task_specific'][mask]).mean()
    ax.text((x0+x1)/2, ax.get_ylim()[1]*0.85 if ax.get_ylim()[1] > 0 else 0.1,
            f'avg {avg:.2f}%', ha='center', color=col, fontsize=9, fontweight='bold')

# ── Panel 2: Additional notes + Optim/Enhance ───────────────────────────────
ax = axes[1]
add_phases(ax)
ax.plot(steps, pct['additional_notes'], color='#1E88E5', linewidth=2,
        marker='o', markersize=4, label='Additional Notes / Guidance')
ax.plot(steps, smooth(pct['additional_notes']), color='#1E88E5',
        linewidth=2.5, linestyle='--', alpha=0.5, label='(smoothed)')
ax.plot(steps, pct['optim_enhance'], color='#FB8C00', linewidth=2,
        marker='s', markersize=4, label='Optimization / Enhancement')
ax.plot(steps, smooth(pct['optim_enhance']), color='#FB8C00',
        linewidth=2.5, linestyle='--', alpha=0.5, label='(smoothed)')
ax.set_ylabel('% of skill records')
ax.set_title('Panel 2 — Elaboration: Additional Notes & Optimization/Enhancement Sections', fontweight='bold')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.4)

# ── Panel 3: Robustness sections (late-emerging) ────────────────────────────
ax = axes[2]
add_phases(ax)
ax.bar(steps, pct['robustness'], color='#43A047', label='Robustness sections\n(Error/Failure/Retry/Success)')
ax.plot(steps, smooth(pct['robustness']), color='#1B5E20',
        linewidth=2, linestyle='--', label='(smoothed)')
ax.set_ylabel('% of skill records')
ax.set_xlabel('Training Step')
ax.set_title('Panel 3 — Robustness: Error / Failure Handling / Retry / Success Criteria Sections', fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.4)

# Phase label annotations on panel 1
for (x0, x1), lbl in zip(phase_spans, phase_labels):
    axes[0].text((x0+x1)/2, -0.02, lbl, ha='center', va='top',
                 transform=axes[0].get_xaxis_transform(), fontsize=7.5, color='#444')

plt.suptitle('Evolution of Emerging Skill Sections Across Training\n'
             '(as % of total skill records per step)', fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to: {OUTPUT_PATH}")

# ── Print numerical summary ─────────────────────────────────────────────────
print(f"\n{'Phase':<18} {'Records':>8} {'Malformed%':>11} {'TaskSpec%':>10} {'AddlNotes%':>11} {'Optim%':>7} {'Robust%':>8}")
print('-' * 78)
for (x0, x1), name in zip(phase_spans, ['Early (1-17)', 'Mid (18-34)', 'Late (35-50)']):
    mask = (steps >= x0) & (steps <= x1)
    tot = int(n_rec[mask].sum())
    print(f"{name:<18} {tot:>8,} "
          f"  {pct['malformed'][mask].mean():>8.2f}%"
          f"  {pct['task_specific'][mask].mean():>8.2f}%"
          f"  {pct['additional_notes'][mask].mean():>9.2f}%"
          f"  {pct['optim_enhance'][mask].mean():>5.2f}%"
          f"  {pct['robustness'][mask].mean():>6.2f}%")
