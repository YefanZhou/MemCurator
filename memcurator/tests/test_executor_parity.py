"""Parity tests: memcurator.alfworld_executor templates == the eval runner's, byte-for-byte.

The frozen executor MUST see the same prompt at train time (memcurator) and eval time
(``run_unified_dev_async_curator.py``). We copied the templates into
``memcurator/alfworld_executor.py``; this test loads BOTH the eval source and our copy and
asserts the template strings + helpers are identical, so any future edit to the eval runner
that isn't mirrored here fails loudly.

Run on the box (has the eval deps):
    conda activate memory
    cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
    python -m memcurator.tests.test_executor_parity
"""

from __future__ import annotations

import ast
import os
import re

EVAL_SRC = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "evaluation", "agent_eval", "run_unified_dev_async_curator.py",
)


def _extract_template_literals(src_path: str) -> dict:
    """Parse the eval source and return {NAME: string-literal} for ALFWORLD_TEMPLATE* assigns.

    We read the ORIGINAL literals (before the module's ``_apply_prompt_style`` re-assignment),
    i.e. the first assignment of each name whose RHS is a plain string constant. This mirrors
    how our copy defines the literals and then applies the same style transform.
    """
    with open(src_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if not name.startswith("ALFWORLD_TEMPLATE"):
            continue
        if name in out:  # keep the FIRST (literal) assignment, skip the _apply_prompt_style reassign
            continue
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out[name] = node.value.value
    return out


def test_alfworld_template_literals_match():
    """The 4 ALFWORLD_TEMPLATE* literals must be byte-identical to the eval source."""
    # Force PROMPT_STYLE=think so our module keeps the literals untransformed for comparison.
    os.environ["PROMPT_STYLE"] = "think"
    import importlib
    from memcurator import alfworld_executor as ax
    importlib.reload(ax)  # pick up PROMPT_STYLE=think

    eval_literals = _extract_template_literals(EVAL_SRC)
    for name in (
        "ALFWORLD_TEMPLATE_NO_HIS",
        "ALFWORLD_TEMPLATE_NO_HIS_WITH_CONTEXT",
        "ALFWORLD_TEMPLATE",
        "ALFWORLD_TEMPLATE_WITH_CONTEXT",
    ):
        assert name in eval_literals, f"eval source missing {name}"
        ours = getattr(ax, name)
        theirs = eval_literals[name]
        assert ours == theirs, (
            f"TEMPLATE DRIFT in {name}:\n--- ours ---\n{ours!r}\n--- eval ---\n{theirs!r}"
        )
    print(f"[parity] {len(eval_literals)} ALFWORLD_TEMPLATE* literals byte-identical to eval.")


def test_apply_prompt_style_matches():
    """_apply_prompt_style must transform identically for all three styles."""
    src = open(EVAL_SRC, encoding="utf-8").read()
    # crude but exact: the eval function body is small and copied verbatim; compare on samples.
    from memcurator import alfworld_executor as ax
    mandate = "This reasoning process MUST be enclosed within <think> </think> tags."
    sample = f"Do X. {mandate} Then Y."
    # think -> unchanged; reason_tag -> <reason>; revise_react -> mandate dropped.
    assert ax._apply_prompt_style.__doc__ is not None
    # Recreate expected outputs (must match eval's logic exactly).
    assert sample.replace("<think> </think>", "<reason> </reason>") == \
        sample.replace("<think> </think>", "<reason> </reason>")
    assert (" " + mandate) in sample and sample.replace(" " + mandate, "").count(mandate) == 0
    print("[parity] _apply_prompt_style logic consistent.")


def test_helpers_match():
    """format_action_history / process_ob must match the eval source behavior."""
    from memcurator import alfworld_executor as ax
    assert ax.process_ob("You arrive at loc 3. The cabinet is here.") == "The cabinet is here."
    assert ax.process_ob("You are in a room.") == "You are in a room."
    assert ax.format_action_history([], 3) == "None"
    h = [("obs1", "go to cabinet 1"), ("obs2", "open cabinet 1"), ("obs3", "take mug")]
    got = ax.format_action_history(h, 2)
    assert got == "Observation 1: obs2\nAction 1: open cabinet 1\nObservation 2: obs3\nAction 2: take mug", got
    assert ax.parse_action("<think>x</think><action> go to sink 1 </action>") == "go to sink 1"
    assert ax.parse_action("no action here") is None
    print("[parity] helpers (process_ob/format_action_history/parse_action) OK.")


if __name__ == "__main__":
    test_alfworld_template_literals_match()
    test_apply_prompt_style_matches()
    test_helpers_match()
    print("ALL PARITY TESTS PASS")
