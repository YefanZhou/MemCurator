"""Build OFFLINE frozen per-target-task store snapshots (S_T) for MemCurator training.

Harvest, don't re-roll: the eval runner already produces successful ALFWorld
trajectories with byte-parity to eval, appended to a ``CuratorAlfworld`` store
(``curator_memory.jsonl``, one JSON record per line:
``{task_id, query, trajectory, status}``). This script slices that harvested pool
into a per-target-task frozen store ``S_T`` for each training target ``T``, so that
during GRPO the ``n`` briefings for ``T`` all read the SAME frozen ``S_T`` and the
advantage isolates briefing quality.

Design invariants (see the plan file):
  * SELF-EXCLUSION: ``S_T`` never contains ``T``'s own gamefile (``task_id``), so the
    curator can't retrieve a near-twin trajectory of the target and copy an answer key.
  * SIZE SPREAD: target tasks are assigned store sizes across a distribution
    (tiny/small/large) so the trained curator sees the full cold->rich range the eval
    store traverses as it grows online.
  * FROZEN: each ``S_T`` is written once, read-only during training.

Output: one ``S_T`` JSONL per target under ``<out_dir>/stores/<target_key>.jsonl``
(same schema as ``CuratorAlfworld``), plus a training index ``<out_dir>/index.jsonl``
with ``{target_key, task_id, query, store_path, store_size}`` per target.

The p-hat (briefed-executor success) target-selection pass is a SEPARATE step that
runs the eval executor; it is NOT done here (this module is pure, CPU-only, and unit
testable). ``select_targets_by_phat`` here only *filters* an existing index given a
precomputed ``{target_key: p_hat}`` map.

Usage::

    python -m memcurator.build_curator_stores \\
        --pool  Alfworld/results/.../curator_memory.jsonl \\
        --out_dir datasets/memcurator/alfworld_v1 \\
        --sizes 2,8,40 --size_weights 0.15,0.50,0.35 \\
        --n_targets 512 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---- record schema (must match evaluation/agent_eval/curator_alfworld.py add()) ----
# {"task_id": <gamefile-derived id>, "query": <NL task>, "trajectory": <text>, "status": "success"}
_REQUIRED_FIELDS = ("task_id", "query", "trajectory")


def load_pool(pool_path: str) -> List[Dict]:
    """Load a harvested CuratorAlfworld JSONL pool of successful trajectories."""
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"pool not found: {pool_path}")
    records: List[Dict] = []
    with open(pool_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = [k for k in _REQUIRED_FIELDS if k not in rec]
            if missing:
                raise ValueError(f"{pool_path}:{ln} record missing {missing}: {rec}")
            # keep only successes as store content (defensive; add() only writes successes)
            if rec.get("status", "success") != "success":
                continue
            records.append(rec)
    return records


def _task_type_of(task_id: str) -> str:
    """Best-effort ALFWorld task-type key from a task_id / gamefile path.

    task_ids look like ``pick_clean_then_place_in_recep-Plate-None-DiningTable-2/trial_T...``
    (the CuratorAlfworld name is ``'/'.join(gamefile.split('/')[-3:-1])``). The type is the
    leading token before the first '-'. Falls back to the whole id if no '-' present.
    """
    head = task_id.split("/")[0] if "/" in task_id else task_id
    return head.split("-")[0] if "-" in head else head


def _dedup_key(rec: Dict) -> str:
    """Dedup identical stored trajectories (same task_id + trajectory)."""
    return f"{rec['task_id']}\x00{rec['trajectory']}"


def build_stores(
    pool: List[Dict],
    out_dir: str,
    sizes: List[int],
    size_weights: List[float],
    n_targets: Optional[int] = None,
    same_type_only: bool = False,
    seed: int = 42,
) -> List[Dict]:
    """Slice ``pool`` into per-target frozen stores S_T.

    Each unique ``task_id`` in the pool becomes a candidate target ``T``. For each target
    we sample a store size from ``sizes`` (weighted by ``size_weights``) and fill S_T with
    that many pool records drawn from OTHER task_ids (self-exclusion). When
    ``same_type_only`` is True, S_T is drawn only from the target's task type; otherwise
    from the whole pool (mixed-type, with the target's type naturally over-represented via
    BM25 at read time) — mixed is the plan's default (realistic distractors).

    Returns the training index (list of dicts); also writes stores + index to ``out_dir``.
    """
    assert len(sizes) == len(size_weights), "sizes and size_weights must align"
    assert abs(sum(size_weights) - 1.0) < 1e-6, "size_weights must sum to 1.0"
    rng = random.Random(seed)

    stores_dir = os.path.join(out_dir, "stores")
    os.makedirs(stores_dir, exist_ok=True)

    # dedup pool, group by task_id and by type
    seen = set()
    deduped: List[Dict] = []
    for rec in pool:
        k = _dedup_key(rec)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(rec)

    by_task_id: Dict[str, List[Dict]] = defaultdict(list)
    by_type: Dict[str, List[Dict]] = defaultdict(list)
    for rec in deduped:
        by_task_id[rec["task_id"]].append(rec)
        by_type[_task_type_of(rec["task_id"])].append(rec)

    target_ids = list(by_task_id.keys())
    rng.shuffle(target_ids)
    if n_targets is not None:
        target_ids = target_ids[:n_targets]

    index: List[Dict] = []
    for i, tid in enumerate(target_ids):
        # representative query for the target (all records under a task_id share the NL task)
        query = by_task_id[tid][0]["query"]
        ttype = _task_type_of(tid)

        # candidate pool for S_T = everything EXCEPT the target's own task_id (self-exclusion)
        if same_type_only:
            candidates = [r for r in by_type[ttype] if r["task_id"] != tid]
        else:
            candidates = [r for r in deduped if r["task_id"] != tid]

        size = rng.choices(sizes, weights=size_weights, k=1)[0]
        size = min(size, len(candidates))
        s_t = rng.sample(candidates, size) if size > 0 else []

        target_key = f"t{i:06d}"
        store_path = os.path.join(stores_dir, f"{target_key}.jsonl")
        with open(store_path, "w", encoding="utf-8") as f:
            for rec in s_t:
                # write the exact CuratorAlfworld schema so the store loads unchanged
                f.write(json.dumps({
                    "task_id": rec["task_id"],
                    "query": rec["query"],
                    "trajectory": rec["trajectory"],
                    "status": "success",
                }, ensure_ascii=False) + "\n")

        index.append({
            "target_key": target_key,
            "task_id": tid,
            "task_type": ttype,
            "query": query,
            "store_path": store_path,
            "store_size": len(s_t),
        })

    index_path = os.path.join(out_dir, "index.jsonl")
    with open(index_path, "w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_summary(index, sizes)
    print(f"[build_curator_stores] wrote {len(index)} targets -> {index_path}")
    return index


def select_targets_by_phat(
    index_path: str,
    phat_map: Dict[str, float],
    lo: float = 0.25,
    hi: float = 0.75,
    out_path: Optional[str] = None,
) -> List[Dict]:
    """Filter a training index to targets whose briefed-executor p_hat is in [lo, hi].

    ``phat_map`` maps ``target_key -> p_hat`` (produced by a separate executor pass, e.g.
    ``memcurator.estimate_phat``). Targets missing from the map are dropped (unmeasured).
    Writes the filtered index to ``out_path`` (default: ``<index>.selected.jsonl``).
    """
    rows = [json.loads(l) for l in open(index_path, encoding="utf-8") if l.strip()]
    kept = [r for r in rows if lo <= phat_map.get(r["target_key"], -1.0) <= hi]
    out_path = out_path or index_path.replace(".jsonl", ".selected.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            r = dict(r, p_hat=phat_map[r["target_key"]])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[select_targets_by_phat] kept {len(kept)}/{len(rows)} targets "
          f"with p_hat in [{lo}, {hi}] -> {out_path}")
    return kept


def _print_summary(index: List[Dict], sizes: List[int]) -> None:
    n = len(index)
    if n == 0:
        print("[build_curator_stores] WARNING: 0 targets built (empty pool?)")
        return
    size_hist: Dict[int, int] = defaultdict(int)
    for row in index:
        size_hist[row["store_size"]] += 1
    empty = sum(1 for r in index if r["store_size"] == 0)
    types = defaultdict(int)
    for r in index:
        types[r["task_type"]] += 1
    print(f"[build_curator_stores] targets={n}  empty_S_T={empty} "
          f"({100*empty/n:.1f}%)  task_types={dict(types)}")
    print(f"[build_curator_stores] store-size histogram: "
          f"{dict(sorted(size_hist.items()))}")


def _parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build offline frozen per-target stores for MemCurator.")
    ap.add_argument("--pool", required=True,
                    help="Harvested CuratorAlfworld JSONL of successful trajectories.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sizes", type=str, default="2,8,40",
                    help="Comma-separated store sizes to sample from.")
    ap.add_argument("--size_weights", type=str, default="0.15,0.50,0.35",
                    help="Comma-separated weights (sum to 1) for --sizes.")
    ap.add_argument("--n_targets", type=int, default=None,
                    help="Max number of target tasks (default: all unique task_ids in pool).")
    ap.add_argument("--same_type_only", action="store_true",
                    help="Fill S_T only from the target's task type (default: mixed-type).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = load_pool(args.pool)
    print(f"[build_curator_stores] loaded {len(pool)} successful trajectories from {args.pool}")
    build_stores(
        pool=pool,
        out_dir=args.out_dir,
        sizes=_parse_int_list(args.sizes),
        size_weights=_parse_float_list(args.size_weights),
        n_targets=args.n_targets,
        same_type_only=args.same_type_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
