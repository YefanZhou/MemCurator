# Async Runner Coverage — Verification (exactly 140 games, once each)

**Question:** Does `run_unified_hyper_async_step0bug_fix.py` (and its `_reason_tag` /
`_revise_react` siblings) evaluate all 140 ALFWorld games exactly once, or is coverage
affected by `--concurrency`?

**Answer:** It evaluates **exactly 140 games, each exactly once, independent of
`--concurrency`.** Verified by both code inspection and an end-to-end run. This is a key
difference from the batch runners, which can over-count when `batch_size` does not divide 140.

---

## Why coverage is concurrency-independent (code)

The rolling-pool loop iterates over **explicit game indices**, submitting each once:

```python
# run_unified_hyper_async_step0bug_fix.py (main)
game_files = sorted(TEMPLATE_ENV.game_files)   # deterministic order; 140 files
n = len(game_files)                            # 140 (unless --num_games caps it)

todo = [(gi, game_files[gi]) for gi in range(n) if gi not in done_idx]   # each index ONCE

with ThreadPoolExecutor(max_workers=concurrency) as pool:
    futures = {pool.submit(run_one_game, gf, gi, model, max_steps): gi
               for gi, gf in todo}          # one future per game
    for fut in as_completed(futures):
        res = fut.result()
        json.dump(..., f'idx_{res["game_idx"]}.json')   # written by game_idx
```

- `todo` has one entry per index `0..n-1` — no wraparound, no padding.
- `concurrency` only bounds **how many run simultaneously** (`max_workers`), never the count.
  A finished game frees a slot that pulls the next queued game; total processed is always `len(todo)`.
- Output file is `idx_{game_idx}.json`, so re-running/resume overwrites by index and cannot
  create duplicates.

Per-game pinning was separately verified: `game_idx i` deterministically loads
`sorted(game_files)[i]` (checked under concurrency for indices 0, 1, 5, 50, 139 — all matched).

---

## Empirical verification (actual run)

Ran the real `main()` of `run_unified_hyper_async_step0bug_fix.py` over all 140 games with a
mock LLM (so it runs fast; coverage logic is identical to a real run), `--concurrency 32`:

```
Alfworld/results/mock/dev_coverage_test_few_shot_False_none/
TOTAL FILES: 140 | DISTINCT: 140 | min: 0 | max: 139
MISSING 0..139: []      # none missing
EXTRA (>139):   []      # none extra / no duplicates
```

Contiguous `idx_0.json .. idx_139.json`, 140 distinct, zero gaps, zero overshoot — at
concurrency 32. Coverage does not depend on the concurrency value.

---

## Contrast: the BATCH runners CAN over-count

The step-synchronous batch runners (`run_unified.py`, `run_unified_hyper*.py`,
`run_unified_nonthink*.py`) iterate `range(math.ceil(num_games / batch_size))` and each
`env.reset()` serves a full batch. When `batch_size` does NOT divide 140, the last batch
wraps around and re-runs games:

| batch_size | batches (ceil) | slots | result |
|---|---|---|---|
| 30 | 5 | 150 | **150 files, 10 games duplicated** (e.g. idx_6 == idx_143) |
| 8  | 18 | 144 | 144 files, 4 duplicated |
| **20** | 7 | 140 | exactly 140 ✅ |
| **28** | 5 | 140 | exactly 140 ✅ |
| **35** | 4 | 140 | exactly 140 ✅ |
| **10** | 14 | 140 | exactly 140 ✅ |

**Guidance:**
- **Async runners** (`*_hyper_async*`): any `--concurrency` is safe → always 140 once.
- **Batch runners**: use a `--batch_size` that **divides 140** (20 / 28 / 35 / 10). If you
  already have a 150-file dir, score only the 140 distinct games (dedup by `name`, or drop
  `idx_140..149`).

---

## Practical implication for scoring

- Async result dirs: 140 files, safe to average directly.
- Batch result dirs at non-dividing batch sizes: contain duplicates; de-duplicate by game
  `name` (or by index ≤ 139) before computing accuracy, or the number is skewed by the
  double-counted games.

## Files this applies to
- `run_unified_hyper_async_step0bug_fix.py`
- `run_unified_hyper_async_reason_tag_step0bug_fix.py`
- `run_unified_hyper_async_revise_react_step0bug_fix.py`
- (and their pre-step0-fix originals `run_unified_hyper_async*.py` — same rolling-pool logic)
