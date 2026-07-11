"""Case study with real ALFWorld embodied (AI2-THOR) screenshots."""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

R = '/home/siruo_google_com/MemP/ProcedureMem/case_study_renders'

# Each panel: (image_path, title_above, action_below, kind)
NOMEM = [
    (f'{R}/00_initial_view.png',
     'Start in room',
     'go to sidetable 2',
     'start'),
    (f'{R}/FAIL_01_at_desklamp_no_cd.png',
     '→ sidetable 2  (no CD held)',
     'use desklamp 1   ⚠',
     'wrong'),
    (f'{R}/FAIL_03_wander_drawer.png',
     'Wander drawers / shelves',
     '(15+ wasted moves)',
     'neutral'),
    (f'{R}/FAIL_04_wander_bed.png',
     'Step budget exhausted',
     'go to drawer 3',
     'bad'),
]

GEMCUR = [
    (f'{R}/00_initial_view.png',
     'Start in room',
     '(scan for CD)',
     'start'),
    (f'{R}/01_see_cd_on_desk.png',
     '→ shelf 3 :  see CD',
     'take cd 1   ✓  (skill step B)',
     'right'),
    (f'{R}/03_at_desklamp_off_holding_cd.png',
     '→ sidetable 2  (holding CD)',
     'use desklamp 1   ✓  (skill step D)',
     'right'),
    (f'{R}/04_lamp_ON_examining_cd_SUCCESS.png',
     'Lamp ON, CD illuminated',
     'task succeeded  (reward = 1)',
     'good'),
]

EDGE = {
    'start':   '#888888',
    'neutral': '#bbbbbb',
    'wrong':   '#de2d26',
    'right':   '#31a354',
    'good':    '#006d2c',
    'bad':     '#a50f15',
}

def render_lane(fig, x0, y0, width, height, lane_title, panels,
                total_steps, won=False, header_color='black',
                inventory_panels=None):
    """Render one lane with image panels + arrows + outcome."""
    inventory_panels = inventory_panels or set()
    # banner
    ax_banner = fig.add_axes([x0, y0+height-0.06, width, 0.05])
    ax_banner.axis('off')
    ax_banner.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.02",
        linewidth=1.0, edgecolor="#888", facecolor="#f0f0f0",
        transform=ax_banner.transAxes,
    ))
    ax_banner.text(0.5, 0.5, lane_title, fontsize=12, fontweight='bold',
                   ha='center', va='center', color=header_color,
                   transform=ax_banner.transAxes)

    # panels
    n = len(panels)
    panel_w = (width - 0.04*(n-1)) / n
    panel_h = height - 0.07
    for i, (img_path, ptitle, paction, kind) in enumerate(panels):
        px = x0 + i*(panel_w + 0.04)
        # title above
        ax_t = fig.add_axes([px, y0+panel_h-0.03, panel_w, 0.03])
        ax_t.axis('off')
        ax_t.text(0.5, 0.0, ptitle, fontsize=9, ha='center', va='bottom',
                  fontweight='bold', color="#333", transform=ax_t.transAxes)
        # image with colored border
        ax_im = fig.add_axes([px, y0+0.06, panel_w, panel_h-0.07])
        ax_im.axis('off')
        img = mpimg.imread(img_path)
        ax_im.imshow(img)
        # colored frame
        for sp in ax_im.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(3)
            sp.set_edgecolor(EDGE[kind])
        ax_im.set_xticks([]); ax_im.set_yticks([])
        # show frame around imshow with thicker border
        ax_im.add_patch(plt.Rectangle((0, 0), img.shape[1]-1, img.shape[0]-1,
                                      fill=False, edgecolor=EDGE[kind], linewidth=4,
                                      transform=ax_im.transData))
        # inventory tag
        if i in inventory_panels:
            ax_im.text(0.02, 0.97, '◉  Inventory: CD',
                       transform=ax_im.transAxes,
                       fontsize=8.5, color="white", fontweight='bold',
                       va='top', ha='left',
                       bbox=dict(boxstyle="round,pad=0.3",
                                 facecolor="#31a354", edgecolor="none"))
        # action below
        ax_a = fig.add_axes([px, y0+0.005, panel_w, 0.05])
        ax_a.axis('off')
        weight = 'bold' if kind in ('wrong','right','good','bad') else 'normal'
        ax_a.text(0.5, 0.5, paction, fontsize=9.5, ha='center', va='center',
                  fontweight=weight, color=EDGE[kind], transform=ax_a.transAxes)

        # arrow to next panel
        if i < n - 1:
            ax_arr = fig.add_axes([px+panel_w, y0+0.06+(panel_h-0.07)/2 - 0.015,
                                    0.04, 0.03])
            ax_arr.axis('off')
            ax_arr.annotate('', xy=(0.95, 0.5), xytext=(0.05, 0.5),
                            xycoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', lw=1.6, color="#666"))

    # outcome banner
    outcome = (f"✓ SUCCESS — {total_steps} steps in total" if won
               else f"✗ FAILED — {total_steps} steps in total")
    bcolor = "#006d2c" if won else "#a50f15"
    ax_o = fig.add_axes([x0+width-0.30, y0+0.005-0.045, 0.30, 0.05])
    ax_o.axis('off')
    ax_o.add_patch(FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.02",
        linewidth=1.5, linestyle='--', edgecolor=bcolor, facecolor='white',
        transform=ax_o.transAxes,
    ))
    ax_o.text(0.5, 0.5, outcome, fontsize=11, fontweight='bold',
              ha='center', va='center', color=bcolor, transform=ax_o.transAxes)


# ----- compose -----
fig = plt.figure(figsize=(16, 9))
fig.suptitle(
    'Task: "Look at the CD under the desklamp."   '
    '(ALFWorld embodied — look_at_obj_in_light, idx 50)',
    fontsize=14, fontweight='bold', y=0.99,
)

# top lane (failure)
render_lane(fig, x0=0.02, y0=0.55, width=0.96, height=0.40,
            lane_title='(i)  Baseline — no memory',
            panels=NOMEM,
            total_steps=30, won=False, header_color="#a50f15")

# bottom lane (success)
render_lane(fig, x0=0.02, y0=0.10, width=0.96, height=0.40,
            lane_title='(iii)  SkillOS  +  Gemini-2.5-Pro curator',
            panels=GEMCUR,
            total_steps=26, won=True, header_color="#006d2c",
            inventory_panels={2, 3})  # holding CD in panels 2,3

# bottom caption: which skill drove success
fig.text(
    0.5, 0.045,
    'Driven by retrieved skill  "InterpretLookAtObjectUnderLightSource"  '
    '(BM25 score 15.41) :  "look at X under Y" ⇒ use Y to illuminate X.   '
    'Workflow:  A. find object   B. take it   C. find light   D. use light while holding object.',
    fontsize=9, ha='center', va='center', color='#2c7fb8', style='italic',
)

plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_idx50_v3.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_idx50_v3.png',
            bbox_inches='tight', dpi=200)
print('Saved case_study_idx50_v3.{pdf,png}')
