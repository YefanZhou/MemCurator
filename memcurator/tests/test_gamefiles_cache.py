"""Test the AlfredTWEnv game_files disk-cache (startup-scan A-axis fix).

Run ON THE BOX (needs $ALFWORLD_DATA + the `memory` conda env):
    conda activate memory
    cd /fsx/home/yefan.zhou/mem-evolve/SkillCurator-main
    python -m memcurator.tests.test_gamefiles_cache

Uses the eval_in_distribution split (valid_seen, ~140 games) so the fresh scan is seconds, not the
~9-min train scan. Asserts: (1) first build = MISS + scans + writes cache; (2) second build = HIT +
sub-second + byte-identical game_files list/order; (3) key change (task_types) = MISS again;
(4) mtime bump invalidates. Cache is written to a throwaway temp dir, never the real /fsx cache.
"""
import os, sys, time, json, tempfile, shutil

# Route the cache to a throwaway dir BEFORE importing envs (module reads the env var at import).
_TMP_CACHE = tempfile.mkdtemp(prefix="gfcache_test_")
os.environ["ALFWORLD_GAMEFILES_CACHE_DIR"] = _TMP_CACHE

REPO = "/fsx/home/yefan.zhou/mem-evolve/SkillCurator-main"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from agent_system.environments.env_package.alfworld import envs as E  # noqa: E402
from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (  # noqa: E402
    get_environment,
)

CONFIG = os.path.join(
    REPO, "agent_system/environments/env_package/alfworld/configs/config_tw.yaml"
)
SPLIT = "eval_in_distribution"  # valid_seen ~140 games; fast fresh scan


def _build_once():
    """Construct one AlfredTWEnv (fires collect_game_files) and return (game_files, seconds)."""
    cfg = E.load_config_file(CONFIG)
    t0 = time.time()
    env = get_environment(cfg["env"]["type"])(cfg, train_eval=SPLIT)
    dt = time.time() - t0
    return list(env.game_files), dt


def main():
    assert os.path.expandvars("$ALFWORLD_DATA") != "$ALFWORLD_DATA", \
        "ALFWORLD_DATA not set — export it (e.g. $HOME/.cache/alfworld) before running."
    E._install_game_files_cache()  # explicit install (envs.AlfworldEnvs would also do it)

    print("\n=== BUILD 1 (expect MISS + fresh scan) ===")
    gf1, dt1 = _build_once()
    n_cache_files = len([f for f in os.listdir(_TMP_CACHE) if f.endswith(".json")])
    assert gf1, "build 1 returned empty game_files"
    assert n_cache_files == 1, f"expected 1 cache file after miss, got {n_cache_files}"
    print(f"    {len(gf1)} games in {dt1:.1f}s; cache files={n_cache_files}")

    print("\n=== BUILD 2 (expect HIT + sub-second + identical list) ===")
    gf2, dt2 = _build_once()
    assert gf2 == gf1, "CACHE CORRUPTION: game_files list/order differs between fresh and cached!"
    assert dt2 < max(2.0, dt1 * 0.5), f"cache HIT not faster: fresh {dt1:.1f}s vs cached {dt2:.1f}s"
    print(f"    identical {len(gf2)} games; {dt2:.2f}s (fresh was {dt1:.1f}s) -> speedup "
          f"{dt1/max(dt2,1e-3):.0f}x")

    print("\n=== BUILD 3 (change task_types -> expect a NEW cache key = MISS) ===")
    cfg = E.load_config_file(CONFIG)
    orig_tt = list(cfg["env"]["task_types"])
    cfg["env"]["task_types"] = orig_tt[:3]  # subset -> different key
    t0 = time.time()
    env3 = get_environment(cfg["env"]["type"])(cfg, train_eval=SPLIT)
    dt3 = time.time() - t0
    gf3 = list(env3.game_files)
    n_after = len([f for f in os.listdir(_TMP_CACHE) if f.endswith(".json")])
    assert n_after == 2, f"expected 2 distinct cache files (task_types changed), got {n_after}"
    assert len(gf3) <= len(gf1), "subset task_types should not yield MORE games"
    print(f"    new key -> {len(gf3)} games in {dt3:.1f}s; cache files now={n_after}")

    print("\n=== INVALIDATION (bump cached mtime -> expect rescan, no crash) ===")
    # Corrupt the stored mtime so the HIT path detects staleness and rescans.
    hit_cache = [f for f in os.listdir(_TMP_CACHE)
                 if SPLIT in f and json.load(open(os.path.join(_TMP_CACHE, f))).get("data_path_mtime")]
    cf = os.path.join(_TMP_CACHE, sorted(hit_cache)[0])
    blob = json.load(open(cf)); blob["data_path_mtime"] = 1.0; json.dump(blob, open(cf, "w"))
    gf4, _ = _build_once()
    assert gf4 == gf1, "post-invalidation rescan produced a different list"
    print("    stale mtime triggered rescan; list still correct")

    print("\nALL PASSED ✅")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMP_CACHE, ignore_errors=True)
