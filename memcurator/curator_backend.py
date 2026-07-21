"""Unified curator-backend selector shared by sample_and_select.py and generation.py.

The eval side has three interchangeable curator modules under evaluation/agent_eval/:
  * curator_alfworld         (DEFAULT) — build_curator_messages(query, retrieved_text) [2-arg], no curation_mode
  * curator_alfworld_v1      — build_curator_messages(query, retrieved_text, curation_mode) [3-arg];
                               modes: success_only | success_and_fail; _format_case adds Result: line
  * curator_alfworld_v1_api  — same as v1 + more modes (success_only_v1, success_and_fail_v1) via a
                               registry; gateway/vertex API paths for curation.

To keep TRAINING byte-identical to whichever eval variant is being reproduced, BOTH the offline
sampler and the training rollout must build the curator prompt / render memories with the SAME
module + curation_mode. This shim selects the module by name and normalizes the small signature
differences (2-arg vs 3-arg build_curator_messages) so callers are variant-agnostic.

Selected via --curator_variant {curator_alfworld, curator_alfworld_v1, curator_alfworld_v1_api}
and --curation_mode (only meaningful for the v1/v1_api variants; ignored by the default).

APPEND (not insert) the eval dir to sys.path — the eval dir has a STALE agent_system/ copy that
would shadow the real repo-root one if inserted first (see generation.py note).
"""

from __future__ import annotations

import inspect
import os
import sys
from typing import Dict, List, Optional

VALID_VARIANTS = ("curator_alfworld", "curator_alfworld_v1", "curator_alfworld_v1_api")

_EVAL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "agent_eval"))
if _EVAL_DIR not in sys.path:
    sys.path.append(_EVAL_DIR)


class CuratorBackend:
    """Wraps one of the eval curator modules behind a uniform interface."""

    def __init__(self, variant: str = "curator_alfworld", curation_mode: str = "success_only"):
        if variant not in VALID_VARIANTS:
            raise ValueError(f"curator_variant must be one of {VALID_VARIANTS}, got {variant!r}")
        self.variant = variant
        self.curation_mode = curation_mode
        self._mod = __import__(variant)  # e.g. curator_alfworld_v1 (resolved from the eval dir)

        # build_curator_messages: 2-arg (default) vs 3-arg (v1/v1_api). Detect once.
        self._bcm = self._mod.build_curator_messages
        self._bcm_takes_mode = "curation_mode" in inspect.signature(self._bcm).parameters

        # Validate curation_mode against the module's declared modes when it has them.
        modes = getattr(self._mod, "CURATION_MODES", None)
        if modes is not None and curation_mode not in modes:
            raise ValueError(
                f"{variant} supports curation_mode in {modes}, got {curation_mode!r}"
            )
        if modes is None and curation_mode != "success_only":
            # the default curator_alfworld has no curation_mode concept.
            print(f"[curator_backend] note: {variant} ignores curation_mode "
                  f"(requested {curation_mode!r}); it has a single fixed prompt.")

        self.CuratorAlfworld = self._mod.CuratorAlfworld
        self._strip_think = self._mod._strip_think

    # -- prompt building (variant-agnostic) --
    def build_curator_messages(self, query: str, retrieved_text: str) -> List[Dict[str, str]]:
        if self._bcm_takes_mode:
            return self._bcm(query, retrieved_text, curation_mode=self.curation_mode)
        return self._bcm(query, retrieved_text)

    def strip_think(self, text: str) -> str:
        return self._strip_think(text)

    # -- trajectory rendering: use THIS module's _trajectory_to_text (byte-identical to that
    #    variant's eval store). All three define it identically today, but we call the selected
    #    module's copy so it never drifts. --
    def trajectory_to_text(self, messages: List[Dict]) -> str:
        C = self.CuratorAlfworld
        return C._trajectory_to_text(C.__new__(C), messages)

    # -- rebuild trajectories from raw per-step responses using the EVAL v1_api renderers
    #    (curator_alfworld_v1_api), which support both 'action_only' and 'with_thinking' styles.
    #    We ALWAYS use v1_api here regardless of self.variant: only v1_api defines the two styles,
    #    and its action_only renderer is byte-identical to v1's (verified — only a docstring differs),
    #    so trajectory_action_only reproduces the Stage-A text exactly. --
    @staticmethod
    def _messages_from_raw(raw_responses: List[Dict]) -> List[Dict]:
        """Reconstruct the eval-parity message list from Stage-A raw_responses: per step a
        user turn (observation) then an assistant turn (the FULL response, incl. <think>/<action>),
        EXACTLY as run_one_game builds `messages`. The renderers re-extract <action> and the CoT
        from these, so passing the full response (not the parsed action) is required for with_thinking."""
        msgs: List[Dict] = []
        for r in raw_responses:
            msgs.append({"role": "user", "content": r.get("observation", "")})
            msgs.append({"role": "assistant", "content": r.get("response", "")})
        return msgs

    def render_trajectories(self, raw_responses: List[Dict], think_token_budget: int = 8000) -> Dict[str, str]:
        """Return {'action_only': ..., 'with_thinking': ...} rebuilt via the v1_api eval methods.

        Uses curator_alfworld_v1_api.CuratorAlfworld._trajectory_to_text /
        _trajectory_to_text_with_thinking on an instance whose only needed attribute is
        think_token_budget (set here); no store/IO/LLM is constructed."""
        import importlib
        api_mod = importlib.import_module("curator_alfworld_v1_api")
        C = api_mod.CuratorAlfworld
        inst = C.__new__(C)                      # bypass __init__ (no store/LLM needed)
        inst.think_token_budget = think_token_budget
        messages = self._messages_from_raw(raw_responses)
        step_responses = [r.get("response", "") for r in raw_responses]
        return {
            "action_only": C._trajectory_to_text(inst, messages),
            "with_thinking": C._trajectory_to_text_with_thinking(inst, messages, step_responses),
        }

    # -- construct a store instance (for per-target S_T retrieval), forwarding curation_mode
    #    only to modules that accept it. --
    def make_store(self, storage_path: str, retrieve_num: int, curator_on_empty: bool = True):
        kwargs = dict(storage_path=storage_path, retrieve_num=retrieve_num,
                      curator_on_empty=curator_on_empty)
        sig = inspect.signature(self.CuratorAlfworld.__init__).parameters
        if "curation_mode" in sig:
            kwargs["curation_mode"] = self.curation_mode
        return self.CuratorAlfworld(**kwargs)
