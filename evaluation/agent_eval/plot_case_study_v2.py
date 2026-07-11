"""Case study (idx 50): two horizontal swim-lanes showing key snapshots,
mimicking the WebShop-style screenshot figure."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow
from matplotlib.patches import Rectangle

# Panel definitions: (mini-snapshot text, action text, kind)
# kind in {"start", "neutral", "wrong", "right", "good", "bad"}
PANELS_NOMEM = [
    {
        "title": "Start on homepage",
        "screen": "You are in the middle of\na room. You see a bed 1,\na desk 1, a sidetable 1,\na sidetable 2, a drawer 1\n... a desklamp ...",
        "action": "go to sidetable 2",
        "kind": "start",
    },
    {
        "title": "→ sidetable 2",
        "screen": "On the sidetable 2,\nyou see a book 2,\na desklamp 1, and a pen 1.",
        "action": "use desklamp 1   ⚠",
        "kind": "wrong",
        "note": "TURNS ON LAMP\nWITHOUT HOLDING CD",
    },
    {
        "title": "Wander drawers/shelves",
        "screen": "(open drawer 2 → empty)\n(open drawer 3 → empty)\n(go to shelf 1 → cellphone)\n(re-visit sidetable 2)\n...",
        "action": "(15+ wasted moves)",
        "kind": "neutral",
        "note": "no plan to pick up\nCD before lighting",
    },
    {
        "title": "Step budget exhausted",
        "screen": "You are facing the desk 1.\nNext to it, you see nothing.",
        "action": "go to drawer 3",
        "kind": "bad",
    },
]

PANELS_GEMCUR = [
    {
        "title": "Start on homepage",
        "screen": "You are in the middle of\na room. You see a bed 1,\na desk 1, a sidetable 1,\na sidetable 2, a shelf 1\n... a desklamp ...",
        "action": "(scan for CD)",
        "kind": "start",
    },
    {
        "title": "→ shelf 3: see CD",
        "screen": "On the shelf 3,\nyou see a cd 1.",
        "action": "take cd 1 from shelf 3   ✓",
        "kind": "right",
        "note": "B. take object\n(skill step B)",
    },
    {
        "title": "→ sidetable 2",
        "screen": "On the sidetable 2,\nyou see a book 2,\na desklamp 1, and a pen 1.",
        "action": "use desklamp 1   ✓",
        "kind": "right",
        "note": "D. illuminate while\nholding the CD",
    },
    {
        "title": "Task succeeded",
        "screen": "(reward = 1)",
        "action": "—",
        "kind": "good",
    },
]

# --- colors ---
COL = {
    "start":  ("#e0e0e0", "#999999"),
    "neutral":("#f0f0f0", "#bbbbbb"),
    "wrong":  ("#fde0dd", "#de2d26"),
    "right":  ("#c7e9c0", "#31a354"),
    "good":   ("#a1d99b", "#006d2c"),
    "bad":    ("#fcae91", "#a50f15"),
}

def render_lane(ax, lane_title, panels, total_steps, won=False, title_color="black"):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # Lane header banner
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.92), 1.0, 0.07,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        linewidth=1.0, edgecolor="#888", facecolor="#f0f0f0",
    ))
    ax.text(0.5, 0.955, lane_title, fontsize=12, fontweight='bold',
            ha='center', va='center', color=title_color)

    n = len(panels)
    margin = 0.02
    panel_w = (1.0 - 2*margin - (n-1)*0.04) / n
    gap = 0.04
    y_top = 0.82
    panel_h = 0.55  # screen + action area

    for i, p in enumerate(panels):
        x0 = margin + i*(panel_w + gap)
        # title above panel
        ax.text(x0 + panel_w/2, y_top + 0.04, p["title"],
                fontsize=9, ha='center', va='bottom', fontweight='bold', color="#333")
        face, edge = COL[p["kind"]]
        # screen box
        screen_h = 0.36
        ax.add_patch(FancyBboxPatch(
            (x0, y_top - screen_h), panel_w, screen_h,
            boxstyle="round,pad=0.005,rounding_size=0.015",
            linewidth=1.5, edgecolor=edge, facecolor=face,
        ))
        ax.text(x0 + panel_w/2, y_top - 0.02,
                p["screen"], fontsize=7.5, ha='center', va='top',
                family='monospace', color="#333")
        # action label below screen
        action_y = y_top - screen_h - 0.05
        ax.text(x0 + panel_w/2, action_y, p["action"],
                fontsize=9, ha='center', va='top',
                fontweight='bold' if p["kind"] in ("wrong","right","good","bad") else 'normal',
                color=edge)
        # note (smaller, italic)
        if "note" in p:
            ax.text(x0 + panel_w/2, action_y - 0.07, p["note"],
                    fontsize=7.5, ha='center', va='top',
                    style='italic', color=edge)

        # arrow to next panel
        if i < n - 1:
            arr_y = y_top - screen_h/2
            ax.annotate("", xy=(x0 + panel_w + gap*0.95, arr_y),
                        xytext=(x0 + panel_w + 0.005, arr_y),
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#666"))

    # outcome at bottom-right
    outcome = f"✓ SUCCESS — {total_steps} steps in total" if won else f"✗ FAILED — {total_steps} steps in total"
    bcolor = "#006d2c" if won else "#a50f15"
    ax.add_patch(FancyBboxPatch(
        (0.62, 0.005), 0.36, 0.07,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        linewidth=1.2, linestyle='--', edgecolor=bcolor, facecolor='white',
    ))
    ax.text(0.80, 0.04, outcome, fontsize=10, fontweight='bold',
            ha='center', va='center', color=bcolor)

# --- compose figure ---
fig = plt.figure(figsize=(15, 7.5))

# global title
fig.suptitle(
    'Task: "Look at the CD under the desklamp."  '
    '(ALFWorld — look_at_obj_in_light, idx 50)',
    fontsize=13, fontweight='bold', y=0.97,
)

# lane 1
ax_top = fig.add_axes([0.02, 0.50, 0.96, 0.42])
render_lane(ax_top, "(i)  Baseline — no memory",
            PANELS_NOMEM, total_steps=30, won=False, title_color="#a50f15")

# lane 2
ax_bot = fig.add_axes([0.02, 0.04, 0.96, 0.42])
render_lane(ax_bot, "(iii)  SkillOS  +  Gemini-2.5-Pro curator",
            PANELS_GEMCUR, total_steps=26, won=True, title_color="#006d2c")

# Skill annotation on the SUCCESS lane (between panels 2 and 3)
fig.text(
    0.50, 0.005,
    'Driven by retrieved skill  "InterpretLookAtObjectUnderLightSource"  '
    '(BM25 score 15.41) :  "look at X under Y" ⇒ use Y to illuminate X. '
    'Workflow: A. find  B. take  C. find light  D. use light while holding object.',
    fontsize=8.5, ha='center', va='bottom', color='#2c7fb8', style='italic'
)

plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_idx50_v2.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_idx50_v2.png',
            bbox_inches='tight', dpi=200)
print("Saved case_study_idx50_v2.{pdf,png}")
