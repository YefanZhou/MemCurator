"""Build an independent ``targets.jsonl`` for Stage B — WITHOUT needing Stage A rollouts.

WHY THIS EXISTS
---------------
A "target" is just a game the curator will be trained/evaluated on. At Stage C train time the
ONLY target field the trainer consumes is ``game_file``: it pins the env to that game
(``env_manager.set_slot_game_files``), resets, and RE-DERIVES the task string from the env
(``env_manager.tasks``). The ``query`` / ``category`` / ``p_hat`` fields are used only by Stage B
(``build_dataset.py``) for store self-exclusion, category mixing, and reporting. So a target does
NOT need to have been rolled out — any valid train ``game_file`` works.

This module produces a ``targets.jsonl`` with the schema Stage B expects
(``{game_file, task_id, query, category}`` + optional ``p_hat`` passthrough) from either source:

  * ``--from_rollouts <rollouts_raw.jsonl>`` — dedupe the games that WERE rolled out (carries their
    measured p_hat / n_rollouts / n_success), i.e. reconstruct targets.jsonl from the raw ledger.
  * ``--from_trainset`` — enumerate games straight from the ALFWorld train split (NO rollouts, NO
    LLM). Optionally stratified by category and/or limited to a fraction. p_hat is absent (None).

``task_id`` is derived from ``game_file`` with EXACTLY the harvest's rule
(``"/".join(gamefile.split("/")[-3:-1]).replace("/", "_")``), so self-exclusion in Stage B matches
byte-for-byte whether a pool record came from a rollout or a from_trainset target.

``query`` (the natural-language task) requires a one-time ``env.reset()`` to read
``"Your task is to: ..."``. Since train-time re-derives it from the env anyway, ``query`` is
OPTIONAL here: by default from_trainset leaves it "" (fast, no env build); pass ``--fill_query`` to
populate it via a reset (slower — builds the env and resets each game once, no LLM).

Usage (box, conda ``memory``, cwd = evaluation/agent_eval so Alfworld/base_config.yaml resolves)::

    # (a) reconstruct targets from a raw ledger (carries p_hat):
    PYTHONPATH=$REPO python -m memcurator.build_targets \\
        --from_rollouts /fsx/.../alfworld_hist3_frac0.5/rollouts_raw.jsonl \\
        --out /fsx/.../targets_from_frac0.5.jsonl

    # (b) FULL train set (all 3553 games, no rollouts, query left blank):
    PYTHONPATH=$REPO python -m memcurator.build_targets \\
        --from_trainset --out /fsx/.../targets_fulltrain.jsonl

    # (c) stratified 20%/category from the train set, WITH query populated (one reset per game):
    PYTHONPATH=$REPO python -m memcurator.build_targets \\
        --from_trainset --stratify --frac 0.2 --fill_query \\
        --out /fsx/.../targets_frac0.2_strat.jsonl

Feed the resulting file to Stage B via ``build_dataset.py`` — see ``--targets_path`` note in that
module (Stage B reads ``<stage_a_dir>/targets.jsonl`` by default; point it at this file's dir, or
copy/symlink it in, or pass an explicit path if that flag is available).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional


# The 6 ALFWorld task-type categories (identical list to sample_and_select / env_manager).
ALFWORLD_TASK_TYPE_NAMES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


def _category_of(game_file: str) -> Optional[str]:
    """The task-type category a game_file belongs to (first matching name), or None."""
    for t in ALFWORLD_TASK_TYPE_NAMES:
        if t in game_file:
            return t
    return None


def _task_id_of(game_file: str) -> str:
    """Derive task_id from a game_file path EXACTLY as sample_and_select does at harvest time:
    the last-two path segments before the filename (``<type-...>/<trial_...>``) joined by ``_``.
    Keeping this identical is what makes Stage B self-exclusion consistent across sources.
    """
    return "/".join(game_file.split("/")[-3:-1]).replace("/", "_")


def _load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------- #
# Source (a): reconstruct targets from a rollouts_raw.jsonl ledger              #
# ---------------------------------------------------------------------------- #
def targets_from_rollouts(rollouts_path: str) -> List[Dict]:
    """One target row per DISTINCT game_file in a raw ledger, carrying measured p_hat.

    Mirrors sample_and_select's end-of-run aggregation: group rollouts by game_file, p_hat =
    wins / n_rollouts. query/category/task_id are taken from the rollout records (already present).
    """
    raw = _load_jsonl(rollouts_path)
    by_game: Dict[str, List[Dict]] = defaultdict(list)
    for r in raw:
        by_game[r["game_file"]].append(r)

    rows: List[Dict] = []
    for g, rolls in by_game.items():
        wins = sum(1 for e in rolls if e.get("won"))
        r0 = rolls[0]
        rows.append({
            "game_file": g,
            "task_id": r0.get("task_id") or _task_id_of(g),
            "query": r0.get("query", ""),
            "category": r0.get("category") or _category_of(g),
            "p_hat": wins / len(rolls),
            "n_rollouts": len(rolls),
            "n_success": wins,
        })
    return rows


# ---------------------------------------------------------------------------- #
# Source (b): enumerate targets straight from the ALFWorld train split          #
# ---------------------------------------------------------------------------- #
def _select_games(all_games: List[str], stratify: bool, frac: float,
                  num_games: Optional[int], seed: int) -> List[str]:
    """Pick games from the full pool.

    stratify=True : take ``frac`` of EACH of the 6 categories (category-balanced). num_games ignored.
    stratify=False: if frac<1 take that fraction of the flat shuffled pool; else if num_games set
                    take that many; else take ALL.
    Seeded via random.Random(seed) → reproducible.
    """
    rng = random.Random(seed)
    if stratify:
        by_cat: Dict[str, List[str]] = defaultdict(list)
        for g in all_games:
            c = _category_of(g)
            if c is not None:
                by_cat[c].append(g)
        picked: List[str] = []
        for cat in ALFWORLD_TASK_TYPE_NAMES:
            games_c = by_cat.get(cat, [])
            rng.shuffle(games_c)
            n_c = max(1, int(round(len(games_c) * frac))) if games_c else 0
            picked.extend(games_c[:n_c])
            print(f"  [stratify] {cat}: {len(games_c)} total -> {n_c} sampled ({frac:.0%})")
        rng.shuffle(picked)
        return picked

    g = list(all_games)
    rng.shuffle(g)
    if frac < 1.0:
        return g[:max(1, int(round(len(g) * frac)))]
    if num_games is not None:
        return g[:num_games]
    return g  # full set (note: unshuffled order lost, but set identity preserved)


def _fill_queries(games: List[str], config_path: str = "Alfworld/base_config.yaml") -> Dict[str, str]:
    """Populate query per game via a single env.reset() each (NO LLM). Reads the task string from
    ``"Your task is to: ..."``. Serialized under a lock (tatsu PDDL parser is not thread-safe).
    Returns {game_file: query}. On any per-game failure, that game maps to "" (train-time re-derives
    query from the env anyway, so a blank is harmless)."""
    import copy
    import threading
    import yaml
    from alfworld.agents.environment import get_environment

    lock = threading.Lock()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    template_env = get_environment(config["env"]["type"])(config, train_eval="train")

    out: Dict[str, str] = {}
    for i, g in enumerate(games):
        try:
            with lock:
                pinned = copy.deepcopy(template_env)
                pinned.game_files = [g]
                tw = pinned.init_env(batch_size=1)
                ob_raw, _info = tw.reset()
                raw = ob_raw[0]
            q = raw.split("\nYour task is to: ")[-1].strip() if "Your task is to:" in raw else ""
            out[g] = q
        except Exception as e:  # noqa: BLE001
            print(f"  [fill_query] WARN {g}: {e}")
            out[g] = ""
        if (i + 1) % 200 == 0:
            print(f"  [fill_query] {i + 1}/{len(games)} reset")
    return out


def targets_from_trainset(stratify: bool, frac: float, num_games: Optional[int],
                          fill_query: bool, seed: int,
                          config_path: str = "Alfworld/base_config.yaml") -> List[Dict]:
    """One target row per selected train game, with NO rollouts. p_hat is None (not measured)."""
    import yaml
    from alfworld.agents.environment import get_environment

    with open(config_path) as f:
        config = yaml.safe_load(f)
    template_env = get_environment(config["env"]["type"])(config, train_eval="train")
    all_games = list(template_env.game_files)
    games = _select_games(all_games, stratify, frac, num_games, seed)
    print(f"[build_targets] train games total={len(all_games)} -> selected {len(games)}")

    queries: Dict[str, str] = _fill_queries(games, config_path) if fill_query else {}

    rows: List[Dict] = []
    for g in games:
        rows.append({
            "game_file": g,
            "task_id": _task_id_of(g),
            "query": queries.get(g, ""),
            "category": _category_of(g),
            "p_hat": None,          # not measured (no rollouts)
        })
    return rows


# ---------------------------------------------------------------------------- #
def _write(rows: List[Dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _report(rows: List[Dict], out_path: str) -> None:
    cats = Counter(r["category"] for r in rows)
    n_query = sum(1 for r in rows if r.get("query"))
    have_phat = [r["p_hat"] for r in rows if r.get("p_hat") is not None]
    print("\n============== targets report ==============")
    print(f"targets written  : {len(rows)} -> {out_path}")
    print(f"categories       : {dict(cats)}")
    print(f"rows with query  : {n_query}/{len(rows)}")
    if have_phat:
        print(f"rows with p_hat  : {len(have_phat)}  mean={sum(have_phat)/len(have_phat):.3f}")
    else:
        print("rows with p_hat  : 0 (from_trainset — not measured)")
    print("============================================")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build an independent targets.jsonl for Stage B (no rollouts required).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_rollouts", metavar="ROLLOUTS_RAW_JSONL",
                     help="Reconstruct targets (with measured p_hat) from a rollouts_raw.jsonl ledger.")
    src.add_argument("--from_trainset", action="store_true",
                     help="Enumerate targets straight from the ALFWorld train split (no rollouts).")
    ap.add_argument("--out", required=True, help="Output targets.jsonl path.")
    # from_trainset selection knobs:
    ap.add_argument("--stratify", action="store_true",
                    help="[from_trainset] Sample --frac of EACH of the 6 categories (balanced).")
    ap.add_argument("--frac", type=float, default=1.0,
                    help="[from_trainset] Fraction of games (per category if --stratify, else flat). "
                         "Default 1.0 = full set.")
    ap.add_argument("--num_games", type=int, default=None,
                    help="[from_trainset, non-stratified, frac>=1] Cap to this many games (else all).")
    ap.add_argument("--fill_query", action="store_true",
                    help="[from_trainset] Populate the natural-language query via one env.reset() per "
                         "game (no LLM; slower). Otherwise query is left \"\" (train-time re-derives it).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config_path", default="Alfworld/base_config.yaml",
                    help="ALFWorld base config (resolves when cwd = evaluation/agent_eval).")
    args = ap.parse_args()

    if args.from_rollouts:
        rows = targets_from_rollouts(args.from_rollouts)
    else:
        rows = targets_from_trainset(
            stratify=args.stratify, frac=args.frac, num_games=args.num_games,
            fill_query=args.fill_query, seed=args.seed, config_path=args.config_path,
        )
    _write(rows, args.out)
    _report(rows, args.out)


if __name__ == "__main__":
    main()
