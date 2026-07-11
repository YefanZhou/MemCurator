"""Case study: idx 50, look_at_obj_in_light (CD with DeskLamp).
The skill teaches a non-obvious semantic interpretation of
"look at X under Y": Y is a light SOURCE, not a physical location.
Without the skill, the executor either (a) turns on the lamp without
ever holding the CD, or (b) takes the CD then goes to the wrong place."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# --- Retrieved skill content (top-1 hit, BM25 = 15.41) ---
SKILL_TEXT = (
    "**InterpretLookAtObjectUnderLightSource**\n\n"
    "# Summary\n"
    "Tasks of the form \"look at X under Y\" do NOT mean\n"
    "find X physically beneath Y. They mean: examine X\n"
    "by illuminating it with light source Y.\n\n"
    "# Workflow (must execute in order)\n"
    " A. Find the target object\n"
    " B. Take it (must be in inventory)\n"
    " C. Find the light source\n"
    " D. Use the light source while holding the object\n"
    "    (e.g., `use desklamp`)"
)

# --- Compact action lists per method ---
NOMEM_STEPS = [
    ("explore desk, sidetables", False),
    ("find desklamp at sidetable 2", False),
    ("⚠ use desklamp 1 (no CD!)", False),  # task misread
    ("‘not carrying anything’", False),
    ("wander drawer 1, 2, 3", False),
    ("(8 more steps re-visiting same locations)", False),
    ("use desklamp again (still no CD)", False),
    ("never picks up CD — FAIL", False),
]
QWENCUR_STEPS = [
    ("explore desk, drawers", False),
    ("(many wasted re-visits)", False),
    ("eventually reach shelf 3", False),
    ("take cd 1 from shelf 3", False),  # finds CD
    ("→ desk 1 (wrong destination)", False),  # mistakes desk for lamp
    ("look (no effect)", False),
    ("move cd 1 to desk 1", False),     # places CD instead of using lamp
    ("(out of step budget — FAIL)", False),
]
GEMCUR_STEPS = [
    ("scan desk, drawers, sidetables", True),    # A. find object phase
    ("explore shelves 1, 2", True),
    ("→ shelf 3: see cd 1", True),
    ("take cd 1 from shelf 3", True),            # B. take object
    ("→ desk 1 (recheck)", True),
    ("→ sidetable 1", True),                     # C. find light source
    ("→ sidetable 2 (desklamp here!)", True),
    ("use desklamp 1 (holding CD) ✓", True),     # D. illuminate
]

# --- Plot ---
fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    "Task (idx 50): look_at_obj_in_light — CD with DeskLamp\n"
    '"Look at the CD under the desklamp."',
    fontsize=14, fontweight='bold', y=0.98,
)

# --- Top-right: skill box ---
ax_skill = fig.add_axes([0.55, 0.62, 0.42, 0.30])
ax_skill.axis('off')
ax_skill.add_patch(FancyBboxPatch(
    (0.0, 0.0), 1.0, 1.0,
    boxstyle="round,pad=0.02,rounding_size=0.04",
    linewidth=1.5, edgecolor="#2c7fb8", facecolor="#deebf7",
    transform=ax_skill.transAxes,
))
ax_skill.text(
    0.5, 0.94,
    "Retrieved Skill (BM25 rank-1, score 15.41)",
    fontsize=10, fontweight='bold', ha='center', color="#2c7fb8",
    transform=ax_skill.transAxes,
)
ax_skill.text(
    0.05, 0.84, SKILL_TEXT,
    fontsize=8.5, ha='left', va='top', family='monospace',
    transform=ax_skill.transAxes,
)

# --- Top-left: query + retrieval ---
ax_q = fig.add_axes([0.03, 0.62, 0.48, 0.30])
ax_q.axis('off')
ax_q.text(
    0.0, 0.95, "Agent observes task & retrieves top-3 skills:",
    fontsize=11, fontweight='bold', va='top', transform=ax_q.transAxes,
)
ax_q.text(
    0.03, 0.85, '"look at cd under the desklamp"',
    fontsize=10, fontstyle='italic', color="#444444",
    transform=ax_q.transAxes,
)
ax_q.text(
    0.0, 0.72,
    "Top-3 retrieved (gemini-curated SkillOS):",
    fontsize=9.5, fontweight='bold', va='top', transform=ax_q.transAxes,
)
retrieval = [
    ("1.", "InterpretLookAtObjectUnderLightSource", 15.41, True),
    ("2.", "LocateAllTaskObjectsBeforeExecution", 12.69, False),
    ("3.", "SequentialSearchAndAcquire", 11.31, False),
]
y0 = 0.62
for tag, name, score, is_key in retrieval:
    color = "#2c7fb8" if is_key else "#777777"
    weight = 'bold' if is_key else 'normal'
    arrow = "  ← KEY" if is_key else ""
    ax_q.text(
        0.03, y0, f"{tag} [{score:.2f}] {name}{arrow}",
        fontsize=9, color=color, fontweight=weight,
        transform=ax_q.transAxes,
    )
    y0 -= 0.08
ax_q.text(
    0.0, 0.20,
    "“under” has two senses in natural language:\n"
    "  (a) physically beneath  vs.  (b) illuminated by\n"
    "Without the curated skill, the executor mis-grounds\n"
    "this phrasing and fails the task.",
    fontsize=9, color='#aa0000', va='top',
    transform=ax_q.transAxes,
)

# --- Bottom: 3 trajectory columns ---
def render_traj(ax, title, steps, header_color='black', won=False, total_steps=30):
    ax.axis('off')
    ax.text(0.5, 1.02, title, fontsize=11, fontweight='bold',
            ha='center', color=header_color, transform=ax.transAxes)
    n = len(steps)
    h = 1.0 / (n + 0.5)
    for i, (txt, applied) in enumerate(steps):
        y = 0.95 - (i + 0.5) * h
        face = "#a1d99b" if applied else "#f7f7f7"
        edge = "#31a354" if applied else "#cccccc"
        ax.add_patch(FancyBboxPatch(
            (0.02, y - h*0.4), 0.96, h*0.8,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            linewidth=0.8, edgecolor=edge, facecolor=face,
            transform=ax.transAxes,
        ))
        ax.text(0.05, y, "•", fontsize=8, va='center', color='#666666',
                family='monospace', transform=ax.transAxes)
        ax.text(0.10, y, txt, fontsize=8.5, va='center',
                transform=ax.transAxes)
    outcome = f"✓ SUCCESS ({total_steps} steps)" if won else f"✗ FAILED ({total_steps} steps)"
    bcolor = "#31a354" if won else "#de2d26"
    ax.text(0.5, -0.02, outcome, fontsize=10, fontweight='bold',
            ha='center', color=bcolor, transform=ax.transAxes)

ax1 = fig.add_axes([0.03, 0.07, 0.30, 0.50])
render_traj(ax1, "(i) no memory", NOMEM_STEPS, header_color="#aa0000",
            won=False, total_steps=30)
ax2 = fig.add_axes([0.36, 0.07, 0.30, 0.50])
render_traj(ax2, "(ii) skillos + Qwen3-8B vanilla curator", QWENCUR_STEPS,
            header_color="#aa0000", won=False, total_steps=30)
ax3 = fig.add_axes([0.69, 0.07, 0.30, 0.50])
render_traj(ax3, "(iii) skillos + Gemini-2.5-Pro curator",
            GEMCUR_STEPS, header_color="#006d2c", won=True, total_steps=26)

# Legend
ax_leg = fig.add_axes([0.03, 0.0, 0.96, 0.05])
ax_leg.axis('off')
patches = [
    mpatches.Patch(facecolor="#a1d99b", edgecolor="#31a354",
                   label="step driven by the retrieved skill (A→B→C→D order)"),
    mpatches.Patch(facecolor="#f7f7f7", edgecolor="#cccccc",
                   label="step taken without skill guidance"),
]
ax_leg.legend(handles=patches, loc='center', ncol=2, fontsize=9, frameon=False)

# Annotation: arrow from skill to gemcur
arr_kw = dict(arrowstyle='-|>,head_width=0.25,head_length=0.5',
              color="#2c7fb8", lw=1.5,
              connectionstyle="arc3,rad=-0.25", mutation_scale=14)
fig.add_artist(FancyArrowPatch(
    (0.76, 0.62), (0.825, 0.45),
    transform=fig.transFigure, **arr_kw,
))
fig.text(0.835, 0.55, "skill drives →", fontsize=8.5, color="#2c7fb8",
         rotation=-50, fontweight='bold')

# Save
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_idx50.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_idx50.png',
            bbox_inches='tight', dpi=200)
print("Saved: case_study_idx50.pdf / .png")
