"""Case study: skill curator QUALITY matters.
Same task, same executor (gemini-3.1-flash-lite-preview), same memory mechanism (SkillOS).
Only the CURATOR differs: vanilla Qwen3-8B vs Gemini-2.5-Pro.
The vanilla curator emits a SEMANTICALLY WRONG skill that misleads the executor;
the Gemini curator emits a correct disambiguation skill that drives success."""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = '/home/siruo_google_com/MemP/ProcedureMem/case_study_renders'

# Trajectory mini-snapshot panels
NOMEM_QWENCUR_PANELS = [
    (f'{R}/00_initial_view.png', 'Start in room', '(scan environment)', 'start'),
    (f'{R}/FAIL_03_wander_drawer.png', 'Wandering drawers, shelves', '(many wasted moves)', 'neutral'),
    (f'{R}/FAIL_04_wander_bed.png', 'Eventually finds CD on shelf 3',
     'take cd 1', 'right'),
    (f'{R}/FAIL_01_at_desklamp_no_cd.png', 'Goes to "desk 1" (NOT desklamp)',
     'move cd 1 to desk 1   ⚠', 'wrong'),
]

GEMCUR_PANELS = [
    (f'{R}/00_initial_view.png', 'Start in room', '(scan for CD)', 'start'),
    (f'{R}/01_see_cd_on_desk.png', '→ shelf 3 :  see CD',
     'take cd 1   ✓  (skill step B)', 'right'),
    (f'{R}/03_at_desklamp_off_holding_cd.png', '→ sidetable 2  (holding CD)',
     'use desklamp 1   ✓  (skill step D)', 'right'),
    (f'{R}/04_lamp_ON_examining_cd_SUCCESS.png', 'Lamp ON, CD illuminated',
     'task succeeded (reward = 1)', 'good'),
]

EDGE = {'start':'#888888', 'neutral':'#bbbbbb', 'wrong':'#de2d26',
        'right':'#31a354', 'good':'#006d2c', 'bad':'#a50f15'}

# ---- Skill text content ----
QWENCUR_SKILL = (
    "**\"Check Under Light Source\"**\n"
    "(curated by vanilla Qwen3-8B)\n\n"
    "# Workflow\n"
    "1. Ensure light source is active and\n"
    "   positioned to illuminate the area\n"
    "   BENEATH it.\n"
    "2. Crouch/position self to view the\n"
    "   area directly under the light.\n"
    "3. Use 'look' command to examine the\n"
    "   underside/adjacent space.\n\n"
    "# Prerequisite Constraints\n"
    "• Must be at the same location as\n"
    "  the light source.\n"
    "• The area under the light must be\n"
    "  physically reachable."
)

GEMCUR_SKILL = (
    "**\"InterpretLookAtObjectUnderLightSource\"**\n"
    "(curated by Gemini-2.5-Pro)\n\n"
    "# Summary\n"
    "Tasks of the form \"look at X under Y\"\n"
    "do NOT mean find X physically beneath Y.\n"
    "They mean: examine X by ILLUMINATING\n"
    "it with light source Y.\n\n"
    "# Workflow (in order)\n"
    "A. Find the target object\n"
    "B. Take it (must be in inventory)\n"
    "C. Find the light source\n"
    "D. Use the light source while holding\n"
    "   the object  (e.g., `use desklamp`)"
)


def render_panels(fig, x0, y0, width, height, panels,
                  inventory_panels=None):
    """Image panels with arrows in between."""
    inventory_panels = inventory_panels or set()
    n = len(panels)
    panel_w = (width - 0.025*(n-1)) / n
    panel_h = height
    for i, (img_path, ptitle, paction, kind) in enumerate(panels):
        px = x0 + i*(panel_w + 0.025)
        # title
        ax_t = fig.add_axes([px, y0+panel_h-0.025, panel_w, 0.025])
        ax_t.axis('off')
        ax_t.text(0.5, 0.0, ptitle, fontsize=8.5, ha='center', va='bottom',
                  fontweight='bold', color='#333', transform=ax_t.transAxes)
        # image
        ax_im = fig.add_axes([px, y0+0.05, panel_w, panel_h-0.075])
        ax_im.axis('off')
        ax_im.imshow(mpimg.imread(img_path))
        ax_im.set_xticks([]); ax_im.set_yticks([])
        ax_im.add_patch(plt.Rectangle((0, 0), 511, 383, fill=False,
                                      edgecolor=EDGE[kind], linewidth=4,
                                      transform=ax_im.transData))
        if i in inventory_panels:
            ax_im.text(0.02, 0.97, '◉ Inventory: CD',
                       transform=ax_im.transAxes, fontsize=8, color='white',
                       fontweight='bold', va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="#31a354", edgecolor="none"))
        # action label
        ax_a = fig.add_axes([px, y0, panel_w, 0.04])
        ax_a.axis('off')
        ax_a.text(0.5, 0.5, paction, fontsize=8.5, ha='center', va='center',
                  fontweight='bold' if kind in ('wrong','right','good','bad') else 'normal',
                  color=EDGE[kind], transform=ax_a.transAxes)
        # arrow
        if i < n-1:
            ax_arr = fig.add_axes([px+panel_w-0.005, y0+0.05+(panel_h-0.075)/2-0.012,
                                    0.025, 0.025])
            ax_arr.axis('off')
            ax_arr.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', lw=1.4, color='#555'))


# ===== Build figure =====
fig = plt.figure(figsize=(16, 10.5))

fig.suptitle(
    'How curator quality changes outcomes — same task, same executor, only the curator differs\n'
    'Task: "Look at the CD under the desklamp."  '
    '(ALFWorld embodied — look_at_obj_in_light, idx 50,  executor: gemini-3.1-flash-lite)',
    fontsize=13, fontweight='bold', y=0.985,
)

# ---- Top row: skill content side-by-side ----
def draw_skill_box(x, y, w, h, text, edge_color, header):
    ax_b = fig.add_axes([x, y+h-0.04, w, 0.035])
    ax_b.axis('off')
    ax_b.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.02",
        linewidth=1.0, edgecolor=edge_color, facecolor="#f0f0f0",
        transform=ax_b.transAxes,
    ))
    ax_b.text(0.5, 0.5, header, fontsize=10.5, fontweight='bold',
              ha='center', va='center', color=edge_color,
              transform=ax_b.transAxes)
    ax = fig.add_axes([x, y, w, h-0.05])
    ax.axis('off')
    ax.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5, edgecolor=edge_color, facecolor="#fafafa",
        transform=ax.transAxes,
    ))
    ax.text(0.05, 0.95, text, fontsize=8.5, family='monospace',
            ha='left', va='top', transform=ax.transAxes)

draw_skill_box(0.03, 0.62, 0.46, 0.27, QWENCUR_SKILL,
               edge_color='#de2d26',
               header='✗  Vanilla Qwen3-8B curator  →  WRONG skill (mis-grounds "under")')
draw_skill_box(0.51, 0.62, 0.46, 0.27, GEMCUR_SKILL,
               edge_color='#31a354',
               header='✓  Gemini-2.5-Pro curator  →  CORRECT disambiguation skill')

# Annotation between skill boxes
fig.text(0.5, 0.585,
         'Same query "look at cd under the desklamp" → top-1 skill retrieved by BM25',
         fontsize=9, ha='center', color='#444', style='italic')

# ---- Bottom: trajectory comparison (rendered scenes) ----
fig.text(0.03, 0.535, '(i)  vanilla SkillOS + Qwen3-8B curator',
         fontsize=11, fontweight='bold', color='#a50f15')
render_panels(fig, x0=0.03, y0=0.32, width=0.94, height=0.20,
              panels=NOMEM_QWENCUR_PANELS)
fig.text(0.97, 0.32, '✗  FAILED  —  30 steps',
         fontsize=11, fontweight='bold', color='#a50f15', ha='right')

fig.text(0.03, 0.275, '(iii)  SkillOS  +  Gemini-2.5-Pro curator',
         fontsize=11, fontweight='bold', color='#006d2c')
render_panels(fig, x0=0.03, y0=0.06, width=0.94, height=0.20,
              panels=GEMCUR_PANELS, inventory_panels={2, 3})
fig.text(0.97, 0.06, '✓  SUCCESS  —  26 steps',
         fontsize=11, fontweight='bold', color='#006d2c', ha='right')

# Footer
fig.text(0.5, 0.025,
         'Across 3 runs:  vanilla qwencur  =  0/3 success     ·     gemini-curated  =  3/3 success',
         fontsize=10, ha='center', color='#2c7fb8', style='italic', fontweight='bold')

plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_curator_quality.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_curator_quality.png',
            bbox_inches='tight', dpi=200)
print('Saved case_study_curator_quality.{pdf,png}')
