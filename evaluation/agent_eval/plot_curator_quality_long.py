"""Curator-quality case studies with long, content-rich skills (reasoning domain).
Each case: same problem, same executor (Qwen3-8B), same memory framework (SkillOS).
Only the curator differs."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- Case A: AIME25 idx 9 — Sudoku-like grid counting ----
#  vanilla 0/3 ;  gemcur 2/3
CASE_A = {
    "title": 'Case A — Worked example & theory  vs.  vague workflow',
    "domain": 'AIME 2025 (math reasoning)',
    "task":  ('Find the number of ways to fill a 3×9 grid with digits 1–9 so each row is a permutation '
              'and each 3×3 block contains 1–9. Express the count as p^a · q^b · r^c · s^d (4 distinct '
              'primes) and return p·a + q·b + r·c + s·d.'),
    "why":   ('Vanilla outputs a generic "partition into disjoint sets" recipe with no numbers, no formulas, '
              'no example. The Gemini-curated skill provides a complete theoretical framework AND a fully-worked '
              'example for the exact 3×3 disjoint-block sub-problem.'),
    "vanilla_text":
        '---\n'
        'name: Count Grids with Row and Block Constraints\n'
        'description: Calculate the number of ways to fill a 3x9 grid with numbers\n'
        '  1-9 such that each row is a permutation and each 3x3 block contains all\n'
        '  numbers 1-9.\n'
        '---\n'
        '# Workflow\n'
        '1. Partition the numbers 1-9 into three disjoint sets of 3 for the first row\n'
        '   (one set per block).\n'
        '2. For each valid first row partition, determine the number of valid second\n'
        '   row partitions such that the sets are disjoint and cover all numbers.\n'
        '3. The third row is uniquely determined by the first two rows.\n'
        '4. Multiply the number of first-row partitions by the number of valid\n'
        '   second-row partitions to get the total number of grids.\n'
        '\n# Prerequisite Constraints\n'
        '• Each row must be a permutation of 1-9.\n'
        '• Each block must contain all numbers 1-9.\n',
    "vanilla_flaw": '✗  No actual numbers, no formulas, no worked example — '
                    'the executor is left to derive everything from scratch and times out.',
    "better_text":
        '---\n'
        'name: Count Mutually Constrained Set Partitions\n'
        'description: A method to count the number of ways to partition a set under\n'
        '  disjointness constraints relative to another fixed partition.\n'
        '---\n'
        '# Workflow\n'
        '1. Define Constraints:  R_Bi ⊆ U \\ R_Ai\n'
        '2. Categorize Elements (group elements by which complements they belong to).\n'
        '3. Set up linear system:\n'
        '   • Size constraint:   Σ_C n_Ci = |R_Bi|\n'
        '   • Exhaustion constraint:  Σ_i n_Ci = |C|\n'
        '4. Find non-negative integer solutions.\n'
        '5. For each solution: count arrangements with C(n,k) and (n!) factors.\n'
        '6. Sum over all solution families.\n'
        '\n# Example: Sudoku-like 3×3 Block Counting\n'
        'Problem: fixed partition R_A1={1,2,3}, R_A2={4,5,6}, R_A3={7,8,9};\n'
        'find # of new partitions R_Bi such that R_Bi ∩ R_Ai = ∅.\n'
        '\n• Categories:  C1={4,5,6}  C2={7,8,9}  C3={1,2,3}\n'
        '  (each lies in two complements)\n'
        '• Equations:  n_21+n_31=3,  n_12+n_32=3,  n_13+n_23=3,\n'
        '              n_12+n_13=3 (for C1), …\n'
        '• Integer solutions: 4 families.\n'
        '\n  Case 1:  R_B1=C2,  R_B2=C3,  R_B3=C1\n'
        '           ways = 1 × (3!)³ = 216\n'
        '  Case 2:  mixed,  ways = 81 × (3!)³  …\n'
        '\n• Sum across all 4 cases →  TOTAL = 12,096\n',
    "better_strength": '✓  Exposes the underlying combinatorial structure AND walks through the exact '
                       'sub-problem with formulas and counts. The executor can finish by lifting the result.',
    "outcome": 'vanilla: 0 / 3       Gemini-curated: 2 / 3',
}

# ---- Case B: GPQA idx 29 — Dye absorption color ----
#  vanilla 0/3 ; gemcur 2/3 ; ft 2/3
CASE_B = {
    "title": 'Case B — Topic-relevant skill with multi-model meta-strategy  vs.  unrelated retrieval',
    "domain": 'GPQA Diamond (chemistry / photophysics)',
    "task":  'A textile dye with extensive π-conjugation emits light at 2.3393 eV. What colour does the '
             'compound absorb?  Choices: (A) Violet (B) Red (C) Blue (D) Yellow.',
    "why":   ('Vanilla retrieves an unrelated organic-synthesis skill (BM25 picks weakly related keywords). '
              'The Gemini-curated skill is on-topic, computes the emitted wavelength explicitly for this '
              'energy, and presents BOTH interpretation models (Stokes shift vs. complementary colour) with '
              'guidance on which to use when.'),
    "vanilla_text":
        '---\n'
        'name: Analyze Multi-Step Organic Synthesis for Hydrogen Distinction\n'
        'description: Determine the number of chemically distinct hydrogen atoms in a\n'
        '  complex organic molecule by tracing the transformation sequence and\n'
        '  analyzing functional group changes.\n'
        '---\n'
        '# Workflow\n'
        '1. Identify the starting compound and its functional groups.\n'
        '2. Trace each reaction step and predict the structure of intermediates and\n'
        '   final products.\n'
        '3. Use symmetry and functional-group positions to determine distinct\n'
        '   hydrogen environments.\n'
        '4. Count distinct hydrogens based on their spatial and electronic\n'
        '   environments.\n'
        '\n# Prerequisite Constraints\n'
        '• Requires knowledge of common organic reactions and structural analysis.\n',
    "vanilla_flaw": '✗  Wrong topic entirely — about counting equivalent ¹H atoms, not photon absorption. '
                    'Misleads the executor into doing structural NMR-style reasoning.',
    "better_text":
        '---\n'
        'name: Determine Absorbed Color from Emitted Light\n'
        'description: Determines the color of light a substance absorbs based on the\n'
        '  energy or color of the light it emits, considering both Stokes shift and\n'
        '  complementary color models.\n'
        '---\n'
        '# Workflow\n'
        '1.  Calculate Emitted Wavelength & Color\n'
        '    Formula:  λ = hc / E      Constant:  hc ≈ 1240 eV·nm\n'
        '    Example:  E = 2.34 eV → λ ≈ 530 nm  →  GREEN light.\n'
        '\n2.  Two Possible Interpretation Models\n'
        '\n  Model A — Stokes Shift  (energy-rigorous):\n'
        '  • Absorbed Energy  >  Emitted Energy   ⇒   λ_abs  <  λ_emitted\n'
        '  • If emitted is Green (530nm), absorbed is Blue (~470nm) or Violet (~420nm).\n'
        '\n  Model B — Complementary Color  (introductory chemistry):\n'
        '  • Perceived color = emitted color;  absorbed = its complement on the\n'
        '    color wheel.\n'
        '  • If emitted is GREEN, absorbed is RED.\n'
        '\n3.  Choose the Right Model\n'
        '  • If Model A\'s answer is NOT in the choices, fall back to Model B.\n'
        '  • In this problem, Model A → "Blue" failed,  ⇒  apply Model B → RED.\n'
        '\n• Color wheel:  Red ↔ Green ·  Blue ↔ Orange ·  Yellow ↔ Violet\n',
    "better_strength": ('✓  Performs the exact wavelength calculation for the question\'s energy, '
                       'enumerates both interpretation models, AND provides a meta-strategy '
                       '("if A fails, fall back to B") that pins down "Red" as the intended answer.'),
    "outcome": 'vanilla: 0 / 3       Gemini-curated: 2 / 3       ft (in-domain trained): 2 / 3',
}

CASES = [CASE_A, CASE_B]


def draw_skill(ax, header, body, edge):
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0,0), 1, 1,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.4, edgecolor=edge, facecolor='#fafafa',
        transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0,0.945), 1, 0.055,
        boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=0, facecolor=edge,
        transform=ax.transAxes))
    ax.text(0.5, 0.972, header, fontsize=8.5, fontweight='bold',
            ha='center', va='center', color='white', transform=ax.transAxes)
    ax.text(0.025, 0.92, body, fontsize=7.5, family='monospace',
            ha='left', va='top', transform=ax.transAxes)


fig = plt.figure(figsize=(17, 14))
fig.suptitle(
    'Why curator quality matters in reasoning domains.\n'
    'Same problem, same executor (Qwen3-8B), same memory framework (SkillOS) — only the curator changes.',
    fontsize=13, fontweight='bold', y=0.995,
)

n = len(CASES)
total_h = 0.94
row_h = (total_h - 0.04) / n

for i, c in enumerate(CASES):
    y_top = 0.96 - i * (row_h + 0.025)
    y_bot = y_top - row_h

    # Case header
    ax_t = fig.add_axes([0.01, y_top - 0.030, 0.98, 0.030])
    ax_t.axis('off')
    ax_t.add_patch(FancyBboxPatch((0,0),1,1,
        boxstyle='round,pad=0.005,rounding_size=0.01',
        linewidth=1, edgecolor='#2c7fb8', facecolor='#deebf7',
        transform=ax_t.transAxes))
    ax_t.text(0.012, 0.5, c['title'], fontsize=11.5, fontweight='bold',
              va='center', ha='left', color='#2c7fb8',
              transform=ax_t.transAxes)
    ax_t.text(0.99, 0.5, c['domain'], fontsize=10, ha='right', va='center',
              color='#444', style='italic', transform=ax_t.transAxes)

    # Task statement
    ax_q = fig.add_axes([0.01, y_top - 0.075, 0.98, 0.040])
    ax_q.axis('off')
    ax_q.text(0.005, 0.95, 'Task:  ' + c['task'], fontsize=9.5, ha='left', va='top',
              color='#222', wrap=True, transform=ax_q.transAxes)

    # 'why' brief
    ax_w = fig.add_axes([0.01, y_top - 0.115, 0.98, 0.035])
    ax_w.axis('off')
    ax_w.text(0.5, 0.9, c['why'], fontsize=9, ha='center', va='top',
              color='#555', transform=ax_w.transAxes, style='italic')

    # skill boxes
    skill_y = y_bot + 0.035
    skill_h = (y_top - 0.130) - skill_y
    ax_v = fig.add_axes([0.01, skill_y, 0.485, skill_h])
    draw_skill(ax_v, '✗  Vanilla Qwen3-8B curator', c['vanilla_text'], '#de2d26')
    ax_b = fig.add_axes([0.505, skill_y, 0.485, skill_h])
    draw_skill(ax_b, '✓  Gemini-2.5-Pro curator', c['better_text'], '#31a354')

    # Strength/flaw notes
    ax_vn = fig.add_axes([0.01, skill_y - 0.025, 0.485, 0.025])
    ax_vn.axis('off')
    ax_vn.text(0.01, 0.5, c['vanilla_flaw'], fontsize=8.5, color='#a50f15',
               ha='left', va='center', fontweight='bold',
               transform=ax_vn.transAxes)
    ax_bn = fig.add_axes([0.505, skill_y - 0.025, 0.485, 0.025])
    ax_bn.axis('off')
    ax_bn.text(0.01, 0.5, c['better_strength'], fontsize=8.5, color='#006d2c',
               ha='left', va='center', fontweight='bold',
               transform=ax_bn.transAxes)

    # Outcome
    ax_o = fig.add_axes([0.01, y_bot, 0.98, 0.018])
    ax_o.axis('off')
    ax_o.text(0.5, 0.5, c['outcome'], fontsize=10.5, fontweight='bold',
              ha='center', va='center', color='#2c7fb8',
              transform=ax_o.transAxes)

plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_curator_long.pdf',
            bbox_inches='tight', dpi=200)
plt.savefig('/home/siruo_google_com/MemP/ProcedureMem/case_study_curator_long.png',
            bbox_inches='tight', dpi=200)
print('Saved case_study_curator_long.{pdf,png}')
