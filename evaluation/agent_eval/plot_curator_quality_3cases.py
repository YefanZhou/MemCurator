"""Three case studies showing different reasons why a better curator beats vanilla.
Each case: same task, same executor (gemini-3.1-flash-lite-preview), same memory mechanism (SkillOS).
Only the curator differs — and the kind of advantage differs across cases."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ----- Case definitions -----
CASES = [
    {
        "title": 'Case A : Semantic disambiguation',
        "task":  '"Look at the CD under the desklamp." (look_at_obj_in_light, idx 50)',
        "why":   'The vanilla skill mis-grounds "under" as a *physical* location.\n'
                 'The better curator writes a disambiguation skill that interprets it as *illuminate by*.',
        "vanilla_title": 'vanilla Qwen3-8B curator',
        "vanilla_text":
            "---\n"
            "name: Check Under Light Source\n"
            "description: Inspect the area directly beneath an\n"
            "  active light source.\n"
            "---\n"
            "# Workflow\n"
            "1. Ensure light source is active and positioned\n"
            "   to illuminate the area BENEATH it.\n"
            "2. Crouch / position self to view the area\n"
            "   directly UNDER the light.\n"
            "3. Use 'look' command to examine the\n"
            "   underside / adjacent space.\n",
        "vanilla_flaw": "✗ Reads \"under\" as physical → tells the agent to look BENEATH the lamp",
        "better_title": 'Gemini-2.5-Pro curator',
        "better_text":
            "---\n"
            "name: InterpretLookAtObjectUnderLightSource\n"
            "description: Clarifies that \"look at object X under\n"
            "  light source Y\" means examine X by illuminating it\n"
            "  with Y, not finding X physically beneath Y.\n"
            "---\n"
            "# Summary\n"
            "Tasks of the form \"look at X under Y\" do NOT mean\n"
            "find X physically beneath Y. They mean: examine X\n"
            "by ILLUMINATING it with light source Y.\n\n"
            "# Workflow (in order)\n"
            "A. Find the target object   B. Take it (inventory)\n"
            "C. Find the light source    D. Use the light\n"
            "   source while holding the object\n",
        "better_strength": "✓ Disambiguates the natural-language reading and prescribes the correct order",
        "outcome": "vanilla: 0 / 3 success    ·    Gemini-curated: 3 / 3 success",
    },
    {
        "title": 'Case B : Generalized workflow vs over-specialized atom',
        "task":  '"Put a cool bread on the countertop." (pick_cool_then_place, idx 88)',
        "why":   'Vanilla writes one narrow per-object search skill that searches the wrong places.\n'
                 'The better curator writes a *general* find-modify-place workflow with correct search ordering.',
        "vanilla_title": 'vanilla Qwen3-8B',
        "vanilla_text":
            "---\n"
            "name: Search for Bread in Storage Locations\n"
            "description: Check fridge, cabinets, drawers,\n"
            "  countertops, and sink for bread to locate it.\n"
            "---\n"
            "# Workflow\n"
            "1. Navigate to fridge and open it.\n"
            "2. Check all cabinets (1-19) sequentially.\n"
            "3. Examine countertops (1-4) for visible bread.\n"
            "4. Inspect sink area, drawers (1-4)…\n",
        "vanilla_flaw": "✗ Just a search skill — never mentions cool/place;\n   tries cabinets/fridge BEFORE open surfaces",
        "better_title": 'Gemini-2.5-Pro',
        "better_text":
            "---\n"
            "name: Find, Modify State, and Place Object\n"
            "description: A general workflow to find an object,\n"
            "  change its state (clean, heat, cool), and then\n"
            "  place it at a target location.\n"
            "---\n"
            "# Workflow\n"
            "1. Find and Pick Up the Object\n"
            "   • SCAN OPEN SURFACES FIRST (CounterTop,\n"
            "     DiningTable, Shelf) → then closed containers.\n"
            "2. Modify State\n"
            "   • To Cool: hold object near Fridge,\n"
            "     `cool <obj> with fridge`. (Don't put it in.)\n"
            "3. Navigate to destination.\n"
            "4. Place (open if container).\n",
        "better_strength": "✓ Scan-open-surfaces-first finds the bread on countertop 2 in 2 steps",
        "outcome": "vanilla: 0 / 3 success (30-step wandering)    ·    Gemini-curated: 3 / 3 (8 steps)",
    },
    {
        "title": 'Case C : Compiled in-domain recipe (alfworld-trained curator)',
        "task":  '"Put a hot mug in cabinet." (pick_heat_then_place, idx 26)',
        "why":   'Vanilla provides *atomic* skills with no composition.\n'
                 'The alfworld-trained (GRPO) curator embeds the state-requirement directly into the place-skill,\n'
                 'so the executor naturally chains heat → place.',
        "vanilla_title": 'vanilla Qwen3-8B',
        "vanilla_text":
            "---\n"
            "name: place_object_in_cabinet\n"
            "description: Place an object into a cabinet after\n"
            "  confirming it is not already present and meets\n"
            "  basic placement criteria.\n"
            "---\n"
            "# Workflow\n"
            "1. Approach the cabinet.\n"
            "2. Confirm the object is not already in the cabinet.\n"
            "3. Use 'put X in Y' or 'move X to Y'.\n"
            "4. Confirm object is placed.\n"
            "\n"
            "# Prerequisite Constraints\n"
            "• Must be at the cabinet's location.\n"
            "• Object must be in possession.\n",
        "vanilla_flaw": "✗ No mention of state requirement (\"hot\") — agent\n   never thinks to heat the mug first",
        "better_title": 'alfworld-trained curator',
        "better_text":
            "---\n"
            "name: place_item_into_surface\n"
            "description: Place an item into a surface such as a\n"
            "  countertop, coffeemachine, or table. Ensure the\n"
            "  item is in the required state (e.g., hot) if\n"
            "  specified by the task.\n"
            "---\n"
            "# Workflow\n"
            "1. Navigate to the surface.\n"
            "2. Ensure the agent is holding the item.\n"
            "3. Use the 'place' or 'move' action to put\n"
            "   the item on the target.\n"
            "\n"
            "# Prerequisite Constraints\n"
            "• Item must be in inventory.\n"
            "• Ensure the item is in the REQUIRED STATE\n"
            "  (e.g., HOT) if specified by the task.\n"
            "(Paired with `heat_object_with_appliance`.)\n",
        "better_strength": "✓ Place-skill carries the state-precondition; executor heats first, then places",
        "outcome": "vanilla: 0 / 3 success (cabinet wandering)    ·    alfworld-trained: 2 / 3 (8 steps)",
    },
]


def draw_skill(ax, header, body, edge):
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0,0), 1, 1,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4, edgecolor=edge, facecolor="#fafafa",
        transform=ax.transAxes))
    # Header bar
    ax.add_patch(FancyBboxPatch((0,0.92), 1, 0.08,
        boxstyle="round,pad=0,rounding_size=0.04",
        linewidth=0, facecolor=edge,
        transform=ax.transAxes))
    ax.text(0.5, 0.96, header, fontsize=8.5, fontweight='bold',
            ha='center', va='center', color='white', transform=ax.transAxes)
    ax.text(0.04, 0.89, body, fontsize=7.0, family='monospace',
            ha='left', va='top', transform=ax.transAxes)


fig = plt.figure(figsize=(15, 16))
fig.suptitle(
    'Why curator quality matters — three cases, three different mechanisms\n'
    'Same memory framework (SkillOS) and same executor (gemini-3.1-flash-lite-preview); only the curator changes.',
    fontsize=13, fontweight='bold', y=0.99,
)

n = len(CASES)
row_h = 0.85 / n
for i, c in enumerate(CASES):
    y_top = 0.93 - i * (row_h + 0.005)
    y_bot = y_top - row_h

    # case title bar
    ax_title = fig.add_axes([0.01, y_top - 0.04, 0.98, 0.035])
    ax_title.axis('off')
    ax_title.add_patch(FancyBboxPatch((0,0), 1, 1,
        boxstyle="round,pad=0.005,rounding_size=0.01",
        linewidth=1, edgecolor='#2c7fb8', facecolor='#deebf7',
        transform=ax_title.transAxes))
    ax_title.text(0.012, 0.5, c['title'], fontsize=11.5, fontweight='bold',
                  va='center', ha='left', color='#2c7fb8',
                  transform=ax_title.transAxes)
    ax_title.text(0.50, 0.5, c['task'], fontsize=10, va='center',
                  ha='left', color='#333', style='italic',
                  transform=ax_title.transAxes)

    # 'why' note
    ax_why = fig.add_axes([0.01, y_top - 0.075, 0.98, 0.030])
    ax_why.axis('off')
    ax_why.text(0.5, 0.9, c['why'], fontsize=9, ha='center', va='top',
                color='#555', transform=ax_why.transAxes)

    # left skill (vanilla)
    skill_y = y_bot + 0.03
    skill_h = (y_top - 0.085) - skill_y
    ax_v = fig.add_axes([0.02, skill_y, 0.475, skill_h])
    draw_skill(ax_v, '✗  ' + c['vanilla_title'], c['vanilla_text'], '#de2d26')
    ax_v_note = fig.add_axes([0.02, skill_y - 0.025, 0.475, 0.025])
    ax_v_note.axis('off')
    ax_v_note.text(0.02, 0.5, c['vanilla_flaw'], fontsize=8.5,
                   color='#a50f15', ha='left', va='center',
                   transform=ax_v_note.transAxes, fontweight='bold')

    # right skill (better)
    ax_b = fig.add_axes([0.51, skill_y, 0.475, skill_h])
    draw_skill(ax_b, '✓  ' + c['better_title'], c['better_text'], '#31a354')
    ax_b_note = fig.add_axes([0.51, skill_y - 0.025, 0.475, 0.025])
    ax_b_note.axis('off')
    ax_b_note.text(0.02, 0.5, c['better_strength'], fontsize=8.5,
                   color='#006d2c', ha='left', va='center',
                   transform=ax_b_note.transAxes, fontweight='bold')

    # outcome row
    ax_out = fig.add_axes([0.01, y_bot - 0.005, 0.98, 0.02])
    ax_out.axis('off')
    ax_out.text(0.5, 0.5, c['outcome'], fontsize=10, fontweight='bold',
                ha='center', va='center', color='#2c7fb8',
                transform=ax_out.transAxes)

plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_curator_3cases.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_curator_3cases.png',
            bbox_inches='tight', dpi=200)
print('Saved case_study_curator_3cases.{pdf,png}')
