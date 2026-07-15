"""
test_results_dir_scan.py — regression tests for how the runners SCAN a results folder.

BUG THIS GUARDS AGAINST (jul-12): the resume/scoring loops did
    for f in os.listdir(output_path):
        if f.endswith('.json'):
            all_reward += json.load(f)['reward']
which also matched the run_config.json sidecar we now write into the results dir → KeyError
'reward' → every run crashed ~instantly with exit 0. Fix: scan only idx_*.json result files.

These tests statically assert that EVERY `os.listdir(output_path)` loop that reads
`['reward']` or counts finished games is guarded by `startswith('idx_')`, AND run a live
scan over a temp dir containing result files + sidecars to prove non-idx files are ignored.

Run:  python3 tests/test_results_dir_scan.py
"""
import ast
import json
import os
import re
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# All 4 runners write run_config.json into output_path, so ALL of them must guard their
# results-dir scans with startswith('idx_'). (The bug originally hit the two dev runners;
# the two hyper runners had the same latent crash in their final-accuracy loop.)
RUNNERS = [
    "run_unified_dev.py",
    "run_unified_dev_async.py",
    "run_unified_hyper_async_step0bug_fix.py",
    "run_unified_hyper_async_revise_react_step0bug_fix.py",
]


# ---------------------------------------------------------------- #
# 1. STATIC CHECK: no bare `endswith('.json')` used to read rewards / count games.
#    Any listdir-of-output_path loop that touches ['reward'] or finished counts MUST
#    filter with startswith('idx_'). We enforce a simpler, robust invariant: the file
#    must not contain `endswith('.json')` UNLESS it is paired with `startswith('idx_')`
#    on the same logical line/expression.
# ---------------------------------------------------------------- #
def test_no_unfiltered_json_scan():
    pat_bad = re.compile(r"endswith\(\s*['\"]\.json['\"]\s*\)")
    pat_guard = re.compile(r"startswith\(\s*['\"]idx_['\"]\s*\)")
    # A `.json` scan is DANGEROUS only if its body CONSUMES the file as a result record —
    # reads ['reward'], json.load()s it, or counts it as a finished game. A pure delete loop
    # (os.remove, for --overwrite) is SAFE unguarded: it *should* nuke sidecars too.
    pat_consume = re.compile(r"\['reward'\]|json\.load|finished_games\s*\+=|total\s*\+=|"
                             r"all_reward\s*\+=|rew\s*\+=|finished\s*=\s*sum")
    for fn in RUNNERS:
        src = open(os.path.join(HERE, fn)).read().splitlines()
        for i, line in enumerate(src):
            if not pat_bad.search(line):
                continue
            # window: the guard may be on this line or the line above (multi-line `sum(... if)`);
            # the consuming op is on this line or the next few (loop body under `if ...json...:`).
            guard_win = "\n".join(src[max(0, i - 1): i + 1])
            body_win = "\n".join(src[i: i + 4])
            if pat_guard.search(guard_win):
                continue  # properly filtered → fine
            # unguarded: only a bug if the body actually consumes the file as a result record
            assert not pat_consume.search(body_win), (
                f"{fn}:{i+1}: `endswith('.json')` scan CONSUMES the file (reward/json.load/count) "
                f"but is NOT guarded by startswith('idx_') -> run_config.json crashes it.\n    {line.strip()}"
            )
    print(f"PASS[static]: all result-consuming .json scans guarded by startswith('idx_') "
          f"across {len(RUNNERS)} runners (delete loops correctly exempt)")


# ---------------------------------------------------------------- #
# 2. LIVE CHECK: reproduce the exact scan logic over a realistic results dir containing
#    idx_*.json result files + run_config.json + run.log, and prove:
#      (a) the OLD unfiltered scan crashes on run_config.json (documents the bug), and
#      (b) the NEW filtered scan counts only the idx files and sums rewards correctly.
# ---------------------------------------------------------------- #
def _make_results_dir():
    d = tempfile.mkdtemp()
    # 3 result files
    for k, rew in enumerate([1.0, 0.0, 1.0]):
        json.dump({"messages": [], "reward": rew, "name": f"g{k}"},
                  open(os.path.join(d, f"idx_{k}.json"), "w"))
    # sidecars the runners now write (NO 'reward' key)
    json.dump({"runner": "x", "args": {}, "resolved_hyperparams": {}},
              open(os.path.join(d, "run_config.json"), "w"))
    open(os.path.join(d, "run.log"), "w").write("some log line\n")
    return d


def _old_scan(d):  # the buggy version
    tot = 0; rew = 0.0
    for f in os.listdir(d):
        if f.endswith(".json"):
            tot += 1
            rew += json.load(open(os.path.join(d, f)))["reward"]
    return tot, rew


def _new_scan(d):  # the fixed version
    tot = 0; rew = 0.0
    for f in os.listdir(d):
        if f.startswith("idx_") and f.endswith(".json"):
            tot += 1
            rew += json.load(open(os.path.join(d, f)))["reward"]
    return tot, rew


def test_live_scan_ignores_sidecars():
    d = _make_results_dir()

    # (a) the OLD scan must blow up on run_config.json — proving the bug is real
    crashed = False
    try:
        _old_scan(d)
    except KeyError as e:
        crashed = "reward" in str(e)
    assert crashed, "expected the OLD unfiltered scan to KeyError on run_config.json"

    # (b) the NEW scan ignores sidecars: exactly 3 games, reward sum 2.0
    tot, rew = _new_scan(d)
    assert tot == 3, f"expected 3 idx files counted, got {tot}"
    assert rew == 2.0, f"expected reward sum 2.0, got {rew}"
    print("PASS[live]: filtered scan counts 3 idx files (reward=2.0), ignores run_config.json/run.log; "
          "old scan confirmed to crash on the sidecar")


if __name__ == "__main__":
    test_no_unfiltered_json_scan()
    test_live_scan_ignores_sidecars()
    print("\nALL RESULTS-DIR-SCAN TESTS PASSED")
