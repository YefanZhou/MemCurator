"""Stage B — materialize the MemCurator GRPO training dataset (fully controllable).

Consumes Stage A outputs (``sample_and_select.py``):
  * ``targets.jsonl``       — one row per sampled target game: {game_file, task_id, query, category, p_hat, ...}
  * ``rollouts_raw.jsonl``  — PREFERRED pool source: the complete crash-safe ledger of ALL rollouts
                              (success AND fail), so Stage B owns the success/fail decision via
                              --pool_status. Falls back to ``pool_source.jsonl`` (already --keep-filtered
                              at harvest time) only if rollouts_raw.jsonl is absent.

For EACH target (default: include all; p_hat is carried through for optional train-time filtering),
build a FROZEN per-target store ``S_T`` drawn from OTHER tasks' trajectories, then emit a target-indexed
training index. The trainer (Stage C) iterates this index, pins each slot to ``game_file``, retrieves
from that target's ``S_T``, and runs GRPO (n briefings per pinned target).

Pool composition is FULLY ARGUMENT-CONTROLLED. The knobs split into THREE ORTHOGONAL GROUPS
(applied in this order); within a group the flags are mutually exclusive or layered as noted:

  (1) WHICH trajectories are eligible at all (global filters, applied first):
  --pool_status STATUS          success_and_fail (default, keep both) | success_only (wins only).
                                  Filtered BEFORE the per-task cap. Keep consistent with the curator's
                                  --curation_mode (a warning fires on mismatch).
  --successes_per_task M        cap: at most M trajectories PER SOURCE TASK (diversity vs depth).
  --pool_phat_band lo hi        keep only trajectories whose SOURCE task's p_hat is in [lo,hi].

  (2) WHICH of those go into a given target's store — SELF-EXCLUSION (never leak the answer key):
  --self_exclude_level LEVEL    task_id (default: drop only the target's own game) |
                                  task_type (drop the target's WHOLE category — a stress test).

  (3) The CATEGORY MIX of the target's store S_T — pick ONE of these two mechanisms:
  --pool_category_mode MODE     COARSE, all-or-nothing:
                                  mixed (default)  : draw from ALL 6 categories (distractor-heavy;
                                                     each category is only ~1/6 of candidates)
                                  same_type        : ONLY the target's own type (max relevance)
                                  cross_type_only  : EXCLUDE the target's type (stress test)
  --same_type_frac F            FINE, continuous OVERRIDE (if set, IGNORES --pool_category_mode):
                                  fixes the same-type FRACTION, e.g. pool_size=10 F=0.3 -> 3 same-type
                                  + 7 distractors. The precision dial for the "skillos hurts alfworld"
                                  probe. Leave UNSET to use --pool_category_mode instead.

  Target filter (separate axis): --target_phat_band lo hi keeps only targets whose p_hat is in [lo,hi]
                                  (default: keep all; p_hat is diagnostic-only, not a gate).

Output: ``<out_dir>/dataset.jsonl`` — one row per target:
  {target_key, game_file, task_id, query, category, p_hat, store_path, store_size}
plus per-target ``<out_dir>/stores/<target_key>.jsonl`` (CuratorAlfworld schema: loads unchanged).

Usage::
    python -m memcurator.build_dataset \\
        --stage_a_dir data/memcurator/pilot80 \\
        --out_dir     data/memcurator/dataset_v1 \\
        --pool_size 10 --successes_per_task 1 --pool_category_mode mixed \\
        --self_exclude_level task_id --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

from memcurator.build_curator_stores import _task_type_of  # reuse the type parser


try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001 — tqdm optional; degrade to a no-op wrapper
    def _tqdm(x, **kw):
        return x


def _load_jsonl(path: str, desc: Optional[str] = None) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Progress bar over lines — rollouts_raw.jsonl can be hundreds of MB, so the parse is the
    # slowest part of Stage B; show it. `desc=None` (e.g. small targets file) => no bar.
    with open(path, encoding="utf-8") as f:
        it = f if desc is None else _tqdm(f, desc=desc, unit=" lines")
        return [json.loads(l) for l in it if l.strip()]


def _cap_per_task(pool: List[Dict], m: int, rng: random.Random) -> List[Dict]:
    """Keep at most ``m`` trajectories per source task_id (status-agnostic).

    Works for both success-only pools and mixed (success+fail) pools from --keep all: caps by
    task_id regardless of status, so `--successes_per_task` bounds how many trajectories per
    source task feed the pool (the name is historical; it caps records, success or fail).
    """
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for rec in pool:
        by_task[rec["task_id"]].append(rec)
    out: List[Dict] = []
    for tid, recs in by_task.items():
        rng.shuffle(recs)
        out.extend(recs[:m])
    return out


def _category(rec: Dict) -> str:
    return rec.get("category") or _task_type_of(rec.get("task_id", ""))


def _gamefile(rec: Dict) -> str:
    return rec.get("game_file", rec.get("task_id", ""))


def _load_pool_records(stage_a_dir: str) -> List[Dict]:
    """Load the Stage A pool source, PREFERRING rollouts_raw.jsonl over pool_source.jsonl.

    rollouts_raw.jsonl is the complete, crash-safe ledger: it holds ALL rollouts (success AND
    fail, regardless of Stage A's --keep) and carries every field Stage B needs
    (task_id/query/category/game_file/trajectory/status/reward) plus extras it ignores. Reading it
    lets Stage B own the success/fail decision (see --pool_status) instead of inheriting whatever
    Stage A's --keep already filtered. Falls back to pool_source.jsonl for older harvests that
    predate rollouts_raw.jsonl or where it was cleaned up.
    """
    raw_path = os.path.join(stage_a_dir, "rollouts_raw.jsonl")
    if os.path.exists(raw_path):
        print(f"[build_dataset] pool source: rollouts_raw.jsonl (complete ledger, all statuses)")
        return _load_jsonl(raw_path, desc="load rollouts_raw")
    pool_path = os.path.join(stage_a_dir, "pool_source.jsonl")
    print(f"[build_dataset] pool source: pool_source.jsonl (rollouts_raw.jsonl absent — "
          f"NOTE: already filtered by Stage A --keep, so --pool_status may be a no-op)")
    return _load_jsonl(pool_path, desc="load pool_source")


def build_dataset(
    stage_a_dir: str,
    out_dir: str,
    pool_size: int = 10,
    successes_per_task: int = 1,
    pool_category_mode: str = "mixed",
    same_type_frac: Optional[float] = None,
    self_exclude_level: str = "task_id",
    pool_phat_band: Optional[tuple] = None,
    target_phat_band: Optional[tuple] = None,
    pool_status: str = "success_and_fail",
    curation_mode: Optional[str] = None,
    targets_path: Optional[str] = None,
    num_targets: Optional[int] = None,
    sample_with_replacement: bool = False,
    trajectory_style: str = "both",
    think_token_budget: int = 8000,
    seed: int = 42,
) -> List[Dict]:
    rng = random.Random(seed)
    # Targets may come from stage_a_dir/targets.jsonl (default) OR an INDEPENDENT file built by
    # memcurator.build_targets (e.g. the full train set, or a different frac) — a target only needs
    # a valid game_file; it need NOT have been rolled out. The POOL still comes from stage_a_dir.
    tgt_file = targets_path or os.path.join(stage_a_dir, "targets.jsonl")
    targets = _load_jsonl(tgt_file)
    print(f"[build_dataset] targets: {len(targets)} from {tgt_file}"
          + ("  (independent target file)" if targets_path else ""))
    pool_all = _load_pool_records(stage_a_dir)

    # p_hat map for source tasks (for --pool_phat_band). Built from targets that carry a p_hat.
    # NOTE: with an INDEPENDENT target file (e.g. from_trainset, p_hat=None, and disjoint from the
    # rolled-out pool tasks) this map won't cover the pool's source tasks — so --pool_phat_band only
    # makes sense when targets ⊇ pool sources AND carry p_hat (i.e. the classic stage_a_dir case).
    phat_by_task = {t["task_id"]: t.get("p_hat") for t in targets if t.get("p_hat") is not None}

    # optional target filter by p_hat band
    if target_phat_band is not None:
        lo, hi = target_phat_band
        targets = [t for t in targets if lo <= t.get("p_hat", -1) <= hi]

    # optional: sample N targets (seeded). Applied AFTER the p_hat band filter so --num_targets
    # bounds the post-filter set. Use for smoke runs (small N) or to cap/expand dataset size; omit
    # to keep ALL targets (default = full iteration over targets_full).
    #   WITHOUT replacement (default): distinct targets, requires N <= len(targets) (else no-op/keep all).
    #   WITH replacement (--sample_with_replacement): draws with duplicates, and N MAY EXCEED
    #     len(targets) (oversampling). Each duplicate becomes its OWN dataset row (distinct target_key)
    #     and gets its OWN independently-drawn store S_T (the shared rng advances per target), so two
    #     occurrences of the same game see DIFFERENT retrieval contexts — mimics eval's varying memory
    #     and SkillOS's sample-with-replacement game exposure.
    if num_targets is not None:
        if sample_with_replacement:
            targets = rng.choices(targets, k=num_targets)
            print(f"[build_dataset] sampled {len(targets)} targets WITH replacement (seed={seed}); "
                  f"{len(set(id(t) for t in targets))} distinct rows drawn")
        elif num_targets < len(targets):
            targets = rng.sample(targets, num_targets)
            print(f"[build_dataset] sampled {len(targets)} targets WITHOUT replacement (seed={seed})")
        else:
            print(f"[build_dataset] --num_targets {num_targets} >= available {len(targets)} and no "
                  f"replacement — keeping all {len(targets)} targets")

    # --- pool status filter (Stage B owns success/fail; applied BEFORE the per-task cap so the cap
    # bounds the FILTERED set). Consistency check vs the curator's curation_mode: a success_and_fail
    # pool only carries useful failure signal if the curator is ALSO run in success_and_fail (it
    # renders the Result: line); mismatches silently waste the failures — warn, don't error.
    n_before_status = len(pool_all)
    if pool_status == "success_only":
        pool_all = [r for r in pool_all if r.get("status", "success") == "success"]
    n_fail = sum(1 for r in pool_all if r.get("status") == "fail")
    print(f"[build_dataset] pool_status={pool_status}: {n_before_status} -> {len(pool_all)} records "
          f"({n_fail} fail kept)")
    if curation_mode is not None:
        if pool_status == "success_and_fail" and "success_and_fail" not in curation_mode:
            print(f"[build_dataset] WARNING: pool_status=success_and_fail but curation_mode="
                  f"{curation_mode!r} does NOT render failures (no Result: line) — the {n_fail} fail "
                  f"trajectories will be shown to the curator WITHOUT a Success/Failure label.")
        if pool_status == "success_only" and "success_and_fail" in curation_mode:
            print(f"[build_dataset] WARNING: curation_mode={curation_mode!r} expects failures but "
                  f"pool_status=success_only supplies none — every Result: line will say Success.")

    # cap per source task, then optional pool p_hat band
    pool = _cap_per_task(pool_all, successes_per_task, rng)
    if pool_phat_band is not None:
        if not phat_by_task:
            print("[build_dataset] WARNING: --pool_phat_band requested but no target carries p_hat "
                  "(independent target file?) — the band would drop the ENTIRE pool. Skipping it.")
        else:
            lo, hi = pool_phat_band
            covered = sum(1 for r in pool if r["task_id"] in phat_by_task)
            pool = [r for r in pool if lo <= (phat_by_task.get(r["task_id"], -1)) <= hi]
            print(f"[build_dataset] pool_phat_band[{lo},{hi}]: {covered} pool recs had a known p_hat; "
                  f"{len(pool)} kept.")

    stores_dir = os.path.join(out_dir, "stores")
    os.makedirs(stores_dir, exist_ok=True)

    # Trajectory rebuild: re-render each store record's trajectory from its raw_responses using the
    # EVAL v1_api renderers (action_only + with_thinking), rather than copying the Stage-A text.
    # `trajectory` (what Stage C's BM25/_format_case read) defaults to action_only. We also keep
    # trajectory_stage_a (original), trajectory_action_only, trajectory_w_thinking for inspection,
    # and ASSERT action_only == stage_a (report mismatches; never crash). raw_responses is NOT kept
    # in stores (redundant once rendered; still lives in rollouts_raw.jsonl) to keep Stage C fast.
    backend = None
    if trajectory_style != "stage_a_only":
        from memcurator.curator_backend import CuratorBackend
        backend = CuratorBackend(variant="curator_alfworld_v1_api", curation_mode="success_only")
    n_mismatch = 0
    n_no_raw = 0
    n_records = 0

    index: List[Dict] = []
    for i, tgt in enumerate(_tqdm(targets, desc="build stores", unit=" target")):
        tgt_type = tgt.get("category") or _task_type_of(tgt["task_id"])
        tgt_game = tgt.get("game_file", "")
        tgt_task_id = tgt["task_id"]

        s_t = _build_store_for_target(
            pool, tgt_type, tgt_task_id, tgt_game,
            pool_size, pool_category_mode, same_type_frac, self_exclude_level, rng,
        )

        target_key = f"t{i:06d}"
        store_path = os.path.join(stores_dir, f"{target_key}.jsonl")
        with open(store_path, "w", encoding="utf-8") as f:
            for rec in s_t:
                n_records += 1
                stage_a_traj = rec["trajectory"]  # original Stage-A rendered text (reference)
                out = {
                    "task_id": rec["task_id"], "query": rec["query"],
                    # trajectory = the field Stage C actually reads; default = action_only (set below).
                    "trajectory": stage_a_traj,
                    # Preserve the REAL status/reward (do NOT hardcode success) so success_and_fail
                    # pools carry the win/fail distinction the curator's _format_case Result: line needs.
                    "status": rec.get("status", "success"),
                    "reward": rec.get("reward", 1.0 if rec.get("status", "success") == "success" else 0.0),
                    "trajectory_stage_a": stage_a_traj,
                }
                if backend is not None:
                    raw = rec.get("raw_responses")
                    if raw:
                        rt = backend.render_trajectories(raw, think_token_budget=think_token_budget)
                        out["trajectory_action_only"] = rt["action_only"]
                        out["trajectory_w_thinking"] = rt["with_thinking"]
                        if rt["action_only"] != stage_a_traj:
                            n_mismatch += 1
                        # trajectory (training field) = the selected style's text.
                        if trajectory_style == "with_thinking":
                            out["trajectory"] = rt["with_thinking"]
                        else:  # action_only or both -> training uses action_only
                            out["trajectory"] = rt["action_only"]
                    else:
                        n_no_raw += 1  # no raw_responses to rebuild from -> keep stage_a text
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

        index.append({
            "target_key": target_key,
            "game_file": tgt_game,
            "task_id": tgt_task_id,
            "query": tgt["query"],
            "category": tgt_type,
            "p_hat": tgt.get("p_hat"),
            "store_path": store_path,
            "store_size": len(s_t),
        })

    index_path = os.path.join(out_dir, "dataset.jsonl")
    with open(index_path, "w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _report(index, pool, pool_category_mode, same_type_frac, self_exclude_level, pool_size)
    if backend is not None:
        print(f"[build_dataset] trajectory rebuild (v1_api, style={trajectory_style}, "
              f"think_token_budget={think_token_budget}): {n_records} store records | "
              f"action_only==stage_a mismatches: {n_mismatch} | records w/o raw_responses "
              f"(kept stage_a): {n_no_raw}")
        if n_mismatch:
            print(f"[build_dataset] WARNING: {n_mismatch}/{n_records} rebuilt action_only trajectories "
                  f"DIFFER from Stage-A text — inspect (rebuild should be byte-identical).")
        else:
            print(f"[build_dataset] OK: all rebuilt action_only trajectories match Stage-A text.")
    print(f"[build_dataset] wrote {len(index)} training rows -> {index_path}")
    return index


def _build_store_for_target(
    pool: List[Dict], tgt_type: str, tgt_task_id: str, tgt_game: str,
    pool_size: int, mode: str, same_type_frac: Optional[float],
    self_exclude_level: str, rng: random.Random,
) -> List[Dict]:
    """Slice ``pool`` into one target's frozen S_T per the ablation knobs."""
    # self-exclusion
    if self_exclude_level == "task_type":
        candidates = [r for r in pool if _category(r) != tgt_type]
    else:  # task_id (default): exclude the target's own game only
        candidates = [r for r in pool
                      if r["task_id"] != tgt_task_id and _gamefile(r) != tgt_game]

    def _sample(cands: List[Dict], k: int) -> List[Dict]:
        k = min(k, len(cands))
        return rng.sample(cands, k) if k > 0 else []

    # same_type_frac takes precedence: split S_T into same-type + distractor quotas
    if same_type_frac is not None:
        n_same = int(round(pool_size * same_type_frac))
        same = [r for r in candidates if _category(r) == tgt_type]
        diff = [r for r in candidates if _category(r) != tgt_type]
        s_t = _sample(same, n_same) + _sample(diff, pool_size - n_same)
        rng.shuffle(s_t)
        return s_t

    if mode == "same_type":
        pool_cands = [r for r in candidates if _category(r) == tgt_type]
    elif mode == "cross_type_only":
        pool_cands = [r for r in candidates if _category(r) != tgt_type]
    else:  # mixed
        pool_cands = candidates
    return _sample(pool_cands, pool_size)


def _report(index, pool, mode, same_type_frac, self_exclude_level, pool_size) -> None:
    n = len(index)
    if n == 0:
        print("[build_dataset] WARNING: 0 targets.")
        return
    from collections import Counter
    cats = Counter(r["category"] for r in index)
    sizes = Counter(r["store_size"] for r in index)
    empty = sum(1 for r in index if r["store_size"] == 0)
    print("\n============== dataset report ==============")
    print(f"targets            : {n}")
    print(f"pool source recs   : {len(pool)}")
    print(f"pool_category_mode : {mode}  same_type_frac={same_type_frac}  self_exclude={self_exclude_level}")
    print(f"target categories  : {dict(cats)}")
    print(f"store sizes (want {pool_size}): {dict(sorted(sizes.items()))}  empty_S_T={empty}")
    print("===========================================")


def _band(v: Optional[List[float]]) -> Optional[tuple]:
    return (v[0], v[1]) if v else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize MemCurator GRPO dataset from Stage A outputs.")
    ap.add_argument("--stage_a_dir", required=True,
                    help="Dir with targets.jsonl + rollouts_raw.jsonl (preferred) / pool_source.jsonl.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pool_size", type=int, default=10)
    ap.add_argument("--successes_per_task", type=int, default=1)
    ap.add_argument("--pool_category_mode", choices=["mixed", "same_type", "cross_type_only"], default="mixed")
    ap.add_argument("--same_type_frac", type=float, default=None)
    ap.add_argument("--self_exclude_level", choices=["task_id", "task_type"], default="task_id")
    ap.add_argument("--pool_phat_band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--target_phat_band", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--pool_status", choices=["success_and_fail", "success_only"], default="success_and_fail",
                    help="Which rollouts may enter stores (filtered from rollouts_raw BEFORE the per-task "
                         "cap): success_and_fail keeps both (for a success_and_fail curator), success_only "
                         "keeps only wins. Keep this consistent with --curation_mode.")
    ap.add_argument("--curation_mode", default=None,
                    help="The curator's curation_mode this dataset targets (e.g. success_only, "
                         "success_and_fail). Used ONLY for a consistency warning vs --pool_status; does "
                         "not change store contents. Leave unset to skip the check.")
    ap.add_argument("--targets_path", default=None,
                    help="Independent targets.jsonl (e.g. built by memcurator.build_targets: full "
                         "train set, a different frac, etc.). A target only needs a valid game_file; it "
                         "need NOT have been rolled out. Default: <stage_a_dir>/targets.jsonl. The POOL "
                         "still comes from --stage_a_dir. (With p_hat=None targets, --pool_phat_band is "
                         "auto-skipped and --target_phat_band would drop everything.)")
    ap.add_argument("--num_targets", type=int, default=None,
                    help="Sample N targets (seeded), applied AFTER any --target_phat_band. Use for "
                         "smoke runs / capping (or expanding, with replacement) dataset size. Default: all.")
    ap.add_argument("--sample_with_replacement", action="store_true",
                    help="Draw --num_targets WITH replacement: duplicates allowed and N MAY EXCEED the "
                         "target count (oversampling). Each drawn occurrence is its own dataset row with "
                         "an independently-sampled store S_T, mimicking eval's varying memory / SkillOS "
                         "sample-with-replacement exposure. Default: without replacement (distinct).")
    ap.add_argument("--trajectory_style", choices=["action_only", "with_thinking", "both", "stage_a_only"],
                    default="both",
                    help="How each store record's trajectory is rendered from raw_responses via the eval "
                         "v1_api renderers. 'both' (default) writes trajectory_action_only + "
                         "trajectory_w_thinking (+ trajectory_stage_a reference) and sets the training "
                         "`trajectory` field to action_only. 'action_only'/'with_thinking' set `trajectory` "
                         "to that style. 'stage_a_only' skips rebuild entirely (copies Stage-A text, old "
                         "behavior). action_only is asserted == Stage-A text.")
    ap.add_argument("--think_token_budget", type=int, default=8000,
                    help="Per-trajectory thinking budget (~tokens) for with_thinking rendering; split "
                         "evenly across steps and head/tail-truncated. Default 8000.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_dataset(
        stage_a_dir=args.stage_a_dir, out_dir=args.out_dir,
        pool_size=args.pool_size, successes_per_task=args.successes_per_task,
        pool_category_mode=args.pool_category_mode, same_type_frac=args.same_type_frac,
        self_exclude_level=args.self_exclude_level,
        pool_phat_band=_band(args.pool_phat_band), target_phat_band=_band(args.target_phat_band),
        pool_status=args.pool_status, curation_mode=args.curation_mode,
        targets_path=args.targets_path, num_targets=args.num_targets,
        sample_with_replacement=args.sample_with_replacement,
        trajectory_style=args.trajectory_style, think_token_budget=args.think_token_budget,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
